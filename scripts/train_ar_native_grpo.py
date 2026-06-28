#!/usr/bin/env python3
"""
AR-native compass-curriculum GRPO for the Activation Verbalizer.

=============================================================================
WHAT'S DIFFERENT FROM train_universal_grpo_hard.py (the failed attempt)
=============================================================================

The old AR-GRPO reward was:
    r = centered_cos(AR(desc), act) - worst_hard_neg_cos

It failed 3x. The inference-time compass-reranking (+0.16 at N=16) succeeded.
The compass taught us three things the AR reward missed:

1. DIFFICULTY AWARENESS. Some activations are clearly decodable, others are
   diffuse. The old GRPO trained on all of them equally. Training on diffuse
   activations injects noise — the AR can't reliably score descriptions of
   these, so advantage estimates are random. Fix: use the compass as a
   CURRICULUM filter. Only train on activations where the compass is
   confident (confidence > tau). Lower tau over epochs.

2. GENERICNESS = SMALL NORM, NOT WRONG DIRECTION. Centered cosine measures
   direction after mean subtraction. But a generic description can reconstruct
   to a SMALL-norm centered vector pointing weakly rightish — scoring decent
   on cosine while carrying zero specific information. Fix: multiply cosine by
   RECONSTRUCTION SPECIFICITY = ‖AR(desc) - μ_L‖. This penalizes generic
   output natively in activation space:
     - Generic → small centered norm → low reward (regardless of direction)
     - Specific + wrong → high norm, low cosine → low reward
     - Specific + right → high norm, high cosine → HIGH reward

3. CONTRASTIVE OVER-SUPPRESSION. The hard-negative subtraction
   (correct - worst_neg) punishes the model for having confusable activations
   in the corpus — something it can't control. Fix: drop hard negatives.
   The specificity reward already handles genericness; the curriculum handles
   difficulty. No need for the additional contrastive signal that dominated
   the old reward.

NO MiniLM IN THE REWARD GRADIENT. The compass is used only for curriculum
selection (which examples to train on), never as a reward signal. The reward
is entirely AR-native: centered_cos × specificity, in Phi-4's 5120-dim space.

=============================================================================
EXPECTED COMMAND (deepthought; needs AV + AR + compass + activations):
  ~/venv/bin/python scripts/train_ar_native_grpo.py \
    --model phi4 \
    --av-adapter output/nla-phi4-universal-av-v2 \
    --ar-checkpoint ~/phi4_ar/stage2_v2mid_best.pt \
    --compass output/av_oracle_compass.pt \
    --activations ~/phi4_ar/phi4_13depths.pt \
    --output output/nla-phi4-av-arnative-grpo \
    --layers 16,25 --epochs 8 --group-size 6 \
    --tau-start 0.40 --tau-end 0.15

DIAGNOSTIC (run after training — the key test):
  Round-trip cosine (AV→AR on holdout) should go UP even though the AR was
  never in the reward gradient for round-trip. If it does, the reward is
  teaching real faithfulness.
"""
import torch
import argparse
import math
import time
import faulthandler
import signal
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_universal_grpo_hard import (
    MODELS, INJECTION_CHARS, INJECTION_SCALE, DEPTH_PCTS,
    normalize_activation, make_av_prompt, nearest_depth_pct,
    strip_generated_row, centered_cosine, load_ar, ar_reconstruct,
    AR_TEMPLATE, AR_MAX_LEN,
)
from av_policy import compass_target

faulthandler.register(signal.SIGUSR1)

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Compass-based curriculum: score each (layer, text_idx) by how decodable
# it is, then filter by threshold tau.
# ---------------------------------------------------------------------------

def compute_curriculum_scores(compass, acts_by_layer, layers, n_texts):
    """For each (layer, text_idx), compute compass confidence = norm of the
    compass prediction ‖(a - μ) @ W‖. High norm = the activation has a clear
    signature in text-embedding space = decodable.

    Returns {layer: np.array[n_texts]} of scores.
    """
    scores = {}
    for L in layers:
        mu = compass["mu"][L].numpy().astype(np.float64)
        W = compass["W"][L].numpy().astype(np.float64)
        acts = acts_by_layer[L].float().numpy().astype(np.float64)
        # Predicted text embeddings (before L2-norm): higher norm = clearer signal
        preds = (acts - mu[None, :]) @ W  # [n_texts, 384]
        norms = np.linalg.norm(preds, axis=1)  # [n_texts]
        scores[L] = norms
    return scores


def select_curriculum(scores, layers, n_texts, tau, max_samples):
    """Select (layer, text_idx, depth_pct) tuples above the tau-th percentile
    of compass confidence, up to max_samples."""
    # Pool all scores, find threshold at the tau-th percentile
    all_scores = np.concatenate([scores[L] for L in layers])
    # tau is a fraction (0.0 = all, 1.0 = none)
    threshold = np.percentile(all_scores, tau * 100)

    candidates = []
    for L in layers:
        for t in range(n_texts):
            if scores[L][t] >= threshold:
                candidates.append((L, t, scores[L][t]))

    # Shuffle and limit
    rng = np.random.RandomState(int(time.time()) % 2**31)
    rng.shuffle(candidates)
    candidates = candidates[:max_samples]
    return [(L, t, nearest_depth_pct(L, 40)) for L, t, _ in candidates]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-checkpoint", required=True,
                        help="AR checkpoint (stage2 .pt or legacy directory)")
    parser.add_argument("--compass", required=True,
                        help="Oracle compass .pt (used for curriculum only, NOT reward)")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    # Training params
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=400)
    parser.add_argument("--group-size", type=int, default=6,
                        help="Samples per activation (larger = better advantages)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--rep-penalty", type=float, default=0.2)
    # Curriculum params
    parser.add_argument("--tau-start", type=float, default=0.40,
                        help="Starting curriculum filter (fraction of examples to EXCLUDE). "
                             "0.4 = keep top 60%% easiest at start.")
    parser.add_argument("--tau-end", type=float, default=0.10,
                        help="Ending curriculum filter. 0.1 = keep top 90%% by final epoch.")
    # Specificity reward params
    parser.add_argument("--spec-norm", choices=["raw", "log", "sqrt"], default="sqrt",
                        help="How to transform the specificity norm. sqrt dampens outliers.")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layers (default: all AR∩compass∩activation layers)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    trust_remote = "phi" not in args.model.lower()

    # --- Load compass (curriculum only) ---
    print("Loading compass (for curriculum, not reward)...")
    compass = torch.load(args.compass, weights_only=False, map_location="cpu")
    compass_layers = set(compass["layers"])

    # --- Load activations ---
    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    acts_by_layer = act_data["activations"]
    n_layers = int(act_data["n_layers"])
    n_texts = int(act_data["n_texts"])
    d_model = int(act_data["d_model"])
    print("  %d texts, d=%d" % (n_texts, d_model))

    # --- Load AR (reward model — frozen) ---
    print("Loading AR...")
    ar_model, ar_value_heads, ar_tokenizer = load_ar(
        args.ar_checkpoint, base_model_name, device, trust_remote)
    ar_layers = set(ar_value_heads.keys())

    # Determine training layers
    avail = set(acts_by_layer.keys()) & ar_layers & compass_layers
    if args.layers:
        train_layers = sorted(int(x) for x in args.layers.split(","))
        missing = set(train_layers) - avail
        if missing:
            raise ValueError("Layers %s not in AR∩compass∩activations (%s)" %
                             (missing, sorted(avail)))
    else:
        train_layers = sorted(avail)
    print("  Training layers:", train_layers)

    # Per-layer means for centering
    layer_means = {L: acts_by_layer[L].float().mean(0) for L in train_layers}
    layer_means_dev = {L: m.to(device) for L, m in layer_means.items()}

    # Compute specificity normalization: what's the typical ‖AR(desc) - μ‖?
    # We'll normalize by the median to keep the reward scale stable.
    # (Computed lazily on first epoch.)

    # --- Compute curriculum scores ---
    print("Computing curriculum scores...")
    curriculum_scores = compute_curriculum_scores(
        compass, acts_by_layer, train_layers, n_texts)
    for L in train_layers:
        s = curriculum_scores[L]
        print("  L%d: compass norm mean=%.2f median=%.2f std=%.2f" %
              (L, s.mean(), np.median(s), s.std()))

    # --- Load AV (policy model — trainable) ---
    print("Loading AV from %s..." % args.av_adapter)
    av_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    if av_tokenizer.pad_token is None:
        av_tokenizer.pad_token = av_tokenizer.eos_token

    av_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote)
    av_base = av_base.to(device)
    av_model = PeftModel.from_pretrained(av_base, args.av_adapter, is_trainable=True)
    av_model.train()

    trainable = sum(p.numel() for p in av_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in av_model.parameters())
    print("  LoRA: %d / %d trainable (%.2f%%)" %
          (trainable, total, 100 * trainable / total))

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
    G = args.group_size
    t_start = time.time()

    # Specificity normalization (per-layer median, computed on first batch)
    spec_median = {}

    def spec_transform(norms, layer_idx):
        """Transform reconstruction specificity norms into reward weights."""
        # Normalize by median so the scale is ~1
        if layer_idx not in spec_median or spec_median[layer_idx] == 0:
            spec_median[layer_idx] = float(norms.median().clamp_min(1e-6))
        normed = norms / spec_median[layer_idx]
        if args.spec_norm == "sqrt":
            return normed.sqrt()
        elif args.spec_norm == "log":
            return (1 + normed).log()
        else:
            return normed

    for epoch in range(args.epochs):
        # Curriculum: interpolate tau from start to end
        frac = epoch / max(args.epochs - 1, 1)
        tau = args.tau_start * (1 - frac) + args.tau_end * frac
        epoch_samples = select_curriculum(
            curriculum_scores, train_layers, n_texts,
            tau=tau, max_samples=args.samples_per_epoch)
        print("\nEpoch %d/%d: curriculum tau=%.2f -> %d samples" %
              (epoch + 1, args.epochs, tau, len(epoch_samples)))

        epoch_cos = 0.0
        epoch_spec = 0.0
        epoch_reward = 0.0
        epoch_kl = 0.0
        n_done = 0

        for start in range(0, len(epoch_samples), args.batch_size):
            mini = epoch_samples[start:start + args.batch_size]
            optimizer.zero_grad()

            for layer_idx, text_idx, depth_pct in mini:
                act = acts_by_layer[layer_idx][text_idx].float()
                prompt_tokens, inject_pos = prompt_cache[depth_pct]
                pl = len(prompt_tokens)

                # --- Generate G samples ---
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

                # --- AR REWARD (native, no MiniLM) ---
                recons = ar_reconstruct(ar_model, ar_value_heads, ar_tokenizer,
                                        group_descs, [layer_idx], device)
                recon = recons[layer_idx]  # [G, d_model]
                mean_L = layer_means_dev[layer_idx]

                # Direction: centered cosine (same as before)
                cos_scores = centered_cosine(recon, act.to(device), mean_L)  # [G]

                # Specificity: ‖reconstruction − layer_mean‖ (how opinionated)
                centered_recon = recon - mean_L.unsqueeze(0)
                spec_norms = centered_recon.norm(dim=1)  # [G]
                spec_weights = spec_transform(spec_norms, layer_idx)

                # REWARD = direction × specificity
                raw_rewards = cos_scores * spec_weights  # [G]

                # Trigram repetition penalty
                rewards = raw_rewards.cpu()
                if args.rep_penalty > 0:
                    for i, gids in enumerate(group_gen_ids):
                        if len(gids) < 4:
                            continue
                        trigrams = [tuple(gids[j:j+3]) for j in range(len(gids)-2)]
                        rep_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
                        rewards[i] -= args.rep_penalty * rep_ratio

                epoch_cos += cos_scores.mean().item()
                epoch_spec += spec_norms.mean().item()
                epoch_reward += rewards.mean().item()
                n_done += 1

                # --- GRPO advantage ---
                mean_r = rewards.mean()
                std_r = rewards.std()
                if std_r.item() < 1e-6:
                    continue
                advantages = (rewards - mean_r) / std_r

                # --- Policy gradient (same mechanics as before) ---
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

                if n_done % 20 == 0:
                    print("  %d/%d cos=%.3f spec=%.1f reward=%.3f kl=%.4f (%.0fs)" %
                          (n_done, len(epoch_samples),
                           epoch_cos / n_done, epoch_spec / n_done,
                           epoch_reward / n_done, epoch_kl / max(n_done*G, 1),
                           time.time() - t_start), flush=True)

            torch.nn.utils.clip_grad_norm_(
                [p for p in av_model.parameters() if p.requires_grad],
                max_norm=1.0)
            optimizer.step()

        # Epoch summary
        avg_cos = epoch_cos / max(n_done, 1)
        avg_spec = epoch_spec / max(n_done, 1)
        avg_reward = epoch_reward / max(n_done, 1)
        avg_kl = epoch_kl / max(n_done * G, 1)
        print("Epoch %d/%d: cos=%.4f spec=%.1f reward=%.4f kl=%.4f tau=%.2f (%ds)" %
              (epoch+1, args.epochs, avg_cos, avg_spec, avg_reward, avg_kl,
               tau, int(time.time() - t_start)))

        if avg_reward > best_reward:
            best_reward = avg_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print("  -> saved (best reward=%.4f)" % best_reward)

    print("\nDone. Best reward: %.4f" % best_reward)
    print("Output: %s" % args.output)


if __name__ == "__main__":
    main()
