#!/usr/bin/env python3
"""
Compass-GRPO: faith reward from the oracle compass + genericness penalty.

THEORY: the existing AR-based GRPO (train_universal_grpo_hard.py) failed 
because the AR adds noise to the reward path (its own ceiling is 0.676)
and gives no explicit penalty for generic output. The compass is a clean
linear shortcut: ridge W maps activation -> MiniLM space. The reward is:

  r_i = cos(MiniLM(desc_i), t*) - λ · cos(MiniLM(desc_i), generic_centroid)
        ^^^^^^^^^^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        activation faithfulness           genericness penalty (anti-template)

where t* = l2norm((a - μ_L) @ W_L) is the compass target — what a perfect
linear readout would predict the description should embed as.

WHY THIS SHOULD WORK WHERE AR-GRPO DIDN'T:
  1. STABLE reward surface: W is a fixed pre-fit matrix, not a neural AR that
     the policy could hack by learning "AR-friendly" phrasings
  2. DIRECT: no AR bottleneck. The 0.83 linear oracle shows the info IS there.
     We're training the model to produce text that MiniLM-embeds close to
     where the activation says it should be.
  3. EXPLICIT genericness penalty: templates score low faith AND get penalized.
  4. COMPOSABLE: faith + genericness + repetition penalty = the full policy,
     now as a training reward.

RISK: reward hacking. The model could learn to produce text that embeds close
to t* in MiniLM space without being semantically faithful. Mitigants:
  - KL anchor to SFT reference (same as before)
  - The compass IS trained on faithful descriptions, so its t* targets sit
    in a region of MiniLM space that corresponds to real descriptions
  - MiniLM is a frozen 6-layer model; it's hard to hack systematically
  - The genericness penalty prevents collapsing to a single output

DIAGNOSTICS (run after training):
  - Round-trip eval (AV→AR cosine) — if this goes UP without being trained on,
    the compass reward is genuinely teaching faithfulness
  - Greedy faith-score on holdout — should improve
  - Manual inspection: are greedy outputs more specific? Do they hallucinate less?
"""
import torch
import argparse
import math
import time
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_universal_grpo_hard import (
    MODELS, INJECTION_CHARS, INJECTION_SCALE, DEPTH_PCTS,
    normalize_activation, make_av_prompt, nearest_depth_pct,
    strip_generated_row,
)
from av_policy import compass_target, select_policy, l2norm_rows

REPO_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--compass", required=True,
                        help="Oracle compass .pt from probe_activation_faithfulness.py")
    parser.add_argument("--generic-centroid", default=None,
                        help="Generic centroid .pt from av_policy.py --save-centroid")
    parser.add_argument("--activations", required=True,
                        help="Activation file (phi4_13depths.pt)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=400)
    parser.add_argument("--group-size", type=int, default=6,
                        help="N samples per activation (larger = better advantage estimation)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (lower than AR-GRPO: smoother reward)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--gen-penalty", type=float, default=0.15,
                        help="Weight of genericness penalty in reward")
    parser.add_argument("--rep-penalty", type=float, default=0.2,
                        help="Trigram repetition penalty")
    parser.add_argument("--kl-coef", type=float, default=0.01,
                        help="KL to SFT reference (lower than AR-GRPO: compass is stabler)")
    parser.add_argument("--faith-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layers to train on (default: all compass layers)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]

    # --- Load compass ---
    print("Loading compass from %s..." % args.compass)
    compass = torch.load(args.compass, weights_only=False, map_location="cpu")
    compass_layers = compass["layers"]
    print("  Compass layers:", compass_layers)

    # Load generic centroid if provided
    gen_centroid = None
    if args.generic_centroid:
        gc = torch.load(args.generic_centroid, weights_only=False, map_location="cpu")
        gen_centroid = np.asarray(gc["centroid"], dtype=np.float64)
        print("  Generic centroid loaded, penalty weight:", args.gen_penalty)

    # --- Load activations ---
    print("Loading activations from %s..." % args.activations)
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    acts_by_layer = act_data["activations"]
    n_layers = int(act_data["n_layers"])
    n_texts = int(act_data["n_texts"])
    print("  %d layers, %d texts, d=%d" % (n_layers, n_texts, act_data["d_model"]))

    # Determine training layers (intersection of compass + activations)
    if args.layers:
        train_layers = [int(x) for x in args.layers.split(",")]
    else:
        train_layers = [L for L in compass_layers if L in acts_by_layer]
    print("  Training layers:", train_layers)

    # --- Load sentence encoder (for reward computation) ---
    print("Loading MiniLM for reward scoring...")
    enc = SentenceTransformer(args.faith_model, device="cpu")

    # Build (layer, text_idx) sample pairs
    samples = []
    for L in train_layers:
        pct = nearest_depth_pct(L, n_layers)
        for t in range(n_texts):
            samples.append((L, t, pct))
    print("  %d total (layer, text) sample pairs" % len(samples))

    # --- Load AV model ---
    print("Loading AV from %s..." % args.av_adapter)
    av_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if av_tokenizer.pad_token is None:
        av_tokenizer.pad_token = av_tokenizer.eos_token

    av_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16).to(device)
    av_model = PeftModel.from_pretrained(av_base, args.av_adapter, is_trainable=True)
    av_model.train()

    trainable = sum(p.numel() for p in av_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in av_model.parameters())
    print("  LoRA: %d / %d trainable (%.2f%%)" % (trainable, total, 100*trainable/total))

    inject_id = av_tokenizer.encode(injection_char, add_special_tokens=False)[0]
    eos_ids = {av_tokenizer.eos_token_id, av_tokenizer.pad_token_id}
    stop_ids = av_tokenizer.encode("</explanation>", add_special_tokens=False)

    prompt_cache = {}
    for pct in DEPTH_PCTS:
        content = make_av_prompt(pct, injection_char)
        chat_str = av_tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        tokens = av_tokenizer.encode(chat_str, add_special_tokens=False)
        inject_pos = next(i for i, t in enumerate(tokens) if t == inject_id)
        prompt_cache[pct] = (tokens, inject_pos)

    optimizer = torch.optim.AdamW(
        [p for p in av_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    embed_layer = av_model.get_input_embeddings()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")
    rng = np.random.RandomState(42)
    G = args.group_size
    t_start = time.time()

    for epoch in range(args.epochs):
        indices = rng.choice(len(samples),
                             min(args.samples_per_epoch, len(samples)), replace=False)
        epoch_samples = [samples[i] for i in indices]

        epoch_faith = 0.0
        epoch_generic = 0.0
        epoch_reward = 0.0
        epoch_kl = 0.0
        n_done = 0

        for start in range(0, len(epoch_samples), args.batch_size):
            mini = epoch_samples[start:start + args.batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0

            for layer_idx, text_idx, depth_pct in mini:
                act = acts_by_layer[layer_idx][text_idx].float()
                prompt_tokens, inject_pos = prompt_cache[depth_pct]
                pl = len(prompt_tokens)

                # --- Generate G samples (batched) ---
                input_ids = torch.tensor([prompt_tokens] * G,
                                         dtype=torch.long, device=device)
                embeddings = embed_layer(input_ids)
                inj = normalize_activation(act.to(device), INJECTION_SCALE)
                embeddings[:, inject_pos, :] = inj.to(embeddings.dtype)

                with torch.no_grad():
                    output = av_model.generate(
                        inputs_embeds=embeddings.to(av_model.dtype),
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=av_tokenizer.eos_token_id,
                        return_dict_in_generate=True)

                group_gen_ids = [
                    strip_generated_row(output.sequences[g], prompt_tokens,
                                        eos_ids, stop_ids)
                    for g in range(G)
                ]
                group_descs = [
                    av_tokenizer.decode(g_ids, skip_special_tokens=True).strip()
                    for g_ids in group_gen_ids
                ]

                # --- COMPASS REWARD ---
                # Embed descriptions with MiniLM
                desc_embs = enc.encode(group_descs, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)

                # Compute compass target for this activation
                mu = compass["mu"][layer_idx].numpy()
                W = compass["W"][layer_idx].numpy()
                tstar = compass_target(act.numpy(), mu, W)

                # Faith scores: cos(desc_embedding, t*)
                faith_scores = desc_embs.astype(np.float64) @ tstar  # [G]

                # Genericness scores (optional)
                gen_scores = np.zeros(G)
                if gen_centroid is not None and args.gen_penalty > 0:
                    gen_scores = desc_embs.astype(np.float64) @ gen_centroid

                # Combined reward
                rewards_np = faith_scores - args.gen_penalty * gen_scores

                # Trigram repetition penalty
                if args.rep_penalty > 0:
                    for i, gids in enumerate(group_gen_ids):
                        if len(gids) < 4:
                            continue
                        trigrams = [tuple(gids[j:j+3]) for j in range(len(gids)-2)]
                        rep_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
                        rewards_np[i] -= args.rep_penalty * rep_ratio

                rewards = torch.tensor(rewards_np, dtype=torch.float32)

                epoch_faith += float(faith_scores.mean())
                epoch_generic += float(gen_scores.mean())
                epoch_reward += float(rewards.mean())
                n_done += 1

                # --- GRPO advantage ---
                mean_r = rewards.mean()
                std_r = rewards.std()
                if std_r.item() < 1e-6:
                    continue
                advantages = (rewards - mean_r) / std_r

                # --- Policy gradient ---
                live = [g for g in range(G) if group_gen_ids[g]]
                if not live:
                    continue
                max_len = max(pl + len(group_gen_ids[g]) for g in live)
                pg_ids = torch.full((len(live), max_len),
                                    av_tokenizer.pad_token_id,
                                    dtype=torch.long, device=device)
                attn = torch.zeros((len(live), max_len),
                                   dtype=torch.long, device=device)
                for row, g in enumerate(live):
                    full = prompt_tokens + group_gen_ids[g]
                    pg_ids[row, :len(full)] = torch.tensor(full, device=device)
                    attn[row, :len(full)] = 1

                pg_embeds = embed_layer(pg_ids)
                pg_embeds[:, inject_pos, :] = inj.to(pg_embeds.dtype)
                pg_embeds = pg_embeds.to(av_model.dtype)

                outputs = av_model(inputs_embeds=pg_embeds, attention_mask=attn)
                if args.kl_coef > 0:
                    with torch.no_grad(), av_model.disable_adapter():
                        ref_out = av_model(inputs_embeds=pg_embeds,
                                           attention_mask=attn)

                sample_losses = []
                for row, g in enumerate(live):
                    gen = group_gen_ids[g]
                    gen_t = torch.tensor(gen, dtype=torch.long, device=device)
                    logits = outputs.logits[row, pl-1:pl-1+len(gen)]
                    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                    token_lps = log_probs[torch.arange(len(gen), device=device), gen_t]

                    loss_g = -advantages[g].to(device) * token_lps.sum()

                    if args.kl_coef > 0:
                        ref_logits = ref_out.logits[row, pl-1:pl-1+len(gen)]
                        ref_lps = torch.nn.functional.log_softmax(
                            ref_logits.float(), dim=-1)
                        ref_token_lps = ref_lps[
                            torch.arange(len(gen), device=device), gen_t]
                        diff = ref_token_lps.detach() - token_lps
                        kl = (diff.exp() - diff - 1.0).mean()
                        epoch_kl += kl.item()
                        loss_g = loss_g + args.kl_coef * kl * len(gen)

                    sample_losses.append(loss_g)

                if sample_losses:
                    loss = sum(sample_losses) / (len(mini) * G)
                    loss.backward()
                    batch_loss += loss.detach().item()

                if n_done % 20 == 0:
                    print("  [ep%d] %d/%d faith=%.3f gen=%.3f reward=%.3f (%.0fs)" %
                          (epoch+1, n_done, len(epoch_samples),
                           epoch_faith/n_done, epoch_generic/n_done,
                           epoch_reward/n_done, time.time()-t_start),
                          flush=True)

            torch.nn.utils.clip_grad_norm_(
                [p for p in av_model.parameters() if p.requires_grad],
                max_norm=1.0)
            optimizer.step()

        # Epoch summary
        avg_faith = epoch_faith / max(n_done, 1)
        avg_gen = epoch_generic / max(n_done, 1)
        avg_reward = epoch_reward / max(n_done, 1)
        avg_kl = epoch_kl / max(n_done * G, 1)
        elapsed = time.time() - t_start

        print("Epoch %d/%d: faith=%.4f generic=%.4f reward=%.4f kl=%.4f (%ds)" %
              (epoch+1, args.epochs, avg_faith, avg_gen, avg_reward, avg_kl,
               int(elapsed)))

        if avg_reward > best_reward:
            best_reward = avg_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print("  -> saved (best reward=%.4f)" % best_reward)

    print("\nDone. Best reward: %.4f" % best_reward)
    print("Output: %s" % args.output)


if __name__ == "__main__":
    main()
