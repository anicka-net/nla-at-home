#!/usr/bin/env python3
"""
Universal GRPO with hard negative mining — multi-layer, depth-conditioned.

Combines:
- Hard negatives (same layer, similar activations) — kills SpongeBob
- Wrong-layer negatives (same text, different trained layer) — kills generic
  depth descriptions (free: read from the same AR forward)
- Trigram repetition penalty
- CENTERED-cosine AR reward (per-layer mean subtracted) — raw activation
  cosine is dominated by the layer mean (mean-baseline alone scores 0.86-0.97)
  so raw-cosine rewards are ~90% shared offset. Centering makes the reward the
  same quantity as the eval metric.
- KL anchor to the SFT policy (reference = adapter disabled) — the reward is
  a learned net and therefore hackable; KL keeps descriptions in-distribution.

Supports two AR formats:
- legacy directory: nla_meta.yaml + value_heads.safetensors, frozen plain
  backbone (truncated-style universal AR)
- phi_ar_stage2 best.pt: {"lora": peft_state_dict, "args": {...}} with a
  sibling *_value_heads.pt {str(layer): weight}; the AR backbone gets the
  LoRA applied, mirroring exactly how the AR was trained.

GRPO samples are restricted to layers that have a trained AR value head —
there is no reward signal anywhere else.

Usage (Phi-4 14B pair):
  python3 scripts/train_universal_grpo_hard.py \
    --model phi4 \
    --av-adapter output/nla-phi4-universal-av-twinclean \
    --ar-checkpoint ~/phi4_ar/stage2_v2mid_best.pt \
    --activations corpus/activations/phi4_all_layers.pt \
    --output output/nla-phi4-universal-av-grpo-hard \
    --n-negatives 3 --wrong-layer --rep-penalty 0.2 --kl-coef 0.02 \
    --epochs 6 --lr 1e-5

UNTESTED ON GPU: rewritten 2026-06-09 for the Phi-4 14B AV/AR pair while the
GPU was busy training the AR. Unit-testable pieces are covered by
tests/test_grpo_units.py; the end-to-end run still needs its first launch.
"""
import torch
import json
import yaml
import argparse
import math
import time
import faulthandler
import signal
import numpy as np

# Stack dump on demand: kill -USR1 <pid> writes all thread stacks to stderr.
faulthandler.register(signal.SIGUSR1)
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
try:
    from peft import set_peft_model_state_dict
except ImportError:
    from peft.utils import set_peft_model_state_dict

REPO_ROOT = Path(__file__).parent.parent

MODELS = {
    "gemma3-1b": "google/gemma-3-1b-it",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4": "microsoft/phi-4",
}

INJECTION_CHARS = {
    "gemma3-1b": "⎝",
    "qwen25-7b": "㈎",
    "phi4-mini": "★",
    "phi4": "★",
}
INJECTION_SCALE = 150.0

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]

# Must match phi_ar_stage2.py exactly: the AR was trained reading the last
# token of this template, left-padded, max_length 256.
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
AR_MAX_LEN = 256
AR_LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def nearest_depth_pct(layer, n_layers):
    depth = layer * 100 / n_layers
    return min(DEPTH_PCTS, key=lambda p: abs(p - depth))


def normalize_activation(v, target_scale):
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


def make_av_prompt(depth_pct, injection_char):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        f"Here is the vector from depth {depth_pct}% of the network:\n\n"
        f"<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )


def centered_cosine(pred, target, mean):
    """Cosine similarity after removing the layer mean from both sides.

    pred: [G, d] or [d]; target: [d] or [N, d]; mean: [d].
    Returns [G] (or [G, N] if target is 2-D and pred is 2-D).
    """
    p = pred.float() - mean
    t = target.float() - mean
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if t.dim() == 1:
        return torch.nn.functional.cosine_similarity(p, t.unsqueeze(0), dim=1)
    pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tn = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return pn @ tn.T


def strip_generated_row(seq_row, prompt_tokens, eos_ids, stop_ids):
    """Get clean generated token ids from one row of generate() output.

    With inputs_embeds some Transformers versions return only generated ids,
    others prompt-prefixed sequences — same check as generation_utils.
    """
    seq = seq_row.tolist()
    pl = len(prompt_tokens)
    if len(seq) > pl and seq[:pl] == prompt_tokens:
        seq = seq[pl:]
    while seq and seq[-1] in eos_ids:
        seq.pop()
    if stop_ids:
        for i in range(len(seq) - len(stop_ids) + 1):
            if seq[i:i + len(stop_ids)] == stop_ids:
                seq = seq[:i]
                break
    return seq


def build_hard_negative_index(activations_by_layer, layer_means, layers, n_texts,
                              k=20, exclude_idx=None):
    """Per-layer top-k nearest neighbors by CENTERED cosine.

    Raw cosine ranks neighbors mostly by the shared layer mean; centering
    first makes the negatives genuinely hard.

    exclude_idx: text indices to keep OUT of the negative pool (the held-out
    eval texts) — otherwise their activations still touch the reward path.
    """
    print("Building per-layer hard negative indices (centered)...")
    exclude_t = (torch.tensor(sorted(exclude_idx), dtype=torch.long)
                 if exclude_idx else None)
    indices = {}
    for layer_idx in layers:
        acts = activations_by_layer[layer_idx].float() - layer_means[layer_idx]
        normed = acts / acts.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        neighbors = []
        chunk_size = 500
        for start in range(0, n_texts, chunk_size):
            end = min(start + chunk_size, n_texts)
            chunk = normed[start:end]
            sims = chunk @ normed.T
            for i in range(end - start):
                sims[i, start + i] = -1.0
            if exclude_t is not None:
                sims[:, exclude_t] = -1.0
            topk = sims.topk(k, dim=1)
            for i in range(end - start):
                neighbors.append(topk.indices[i].tolist())
        indices[layer_idx] = neighbors
        print("  layer %d: done" % layer_idx)
    return indices


def load_value_head_weights(path):
    """Parse value-head weights into {layer: tensor[d, d]}.

    Supports phi_ar_stage2 .pt ({str(layer): weight}) and the legacy
    value_heads.safetensors ("heads.{layer}.weight"-style keys).
    """
    path = Path(path)
    weights = {}
    if path.suffix == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=True)
        for key, w in raw.items():
            weights[int(key)] = w
    else:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                layer_idx = int(key.split(".")[1])
                weights[layer_idx] = f.get_tensor(key)
    return weights


def load_ar(ar_checkpoint, base_model_name, device, trust_remote):
    """Load the AR: backbone (+LoRA if phi_ar_stage2 format) and value heads.

    Returns (model, value_heads dict {layer: nn.Linear}, tokenizer).
    The forward convention mirrors phi_ar_stage2.recon: left-padded prompts,
    hidden_states[layer+1][:, -1, :].float() -> value_head[layer].
    """
    ar_path = Path(ar_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote)

    if ar_path.suffix == ".pt":
        # phi_ar_stage2 format: LoRA state + sibling value heads
        ck = torch.load(ar_path, map_location="cpu", weights_only=False)
        ar_args = ck.get("args", {})
        r = int(ar_args.get("lora_r", 16))
        lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                          target_modules=AR_LORA_TARGET_MODULES,
                          task_type="CAUSAL_LM")
        model = get_peft_model(backbone, lora)
        set_peft_model_state_dict(model, ck["lora"])
        vh_path = ar_path.with_name(ar_path.name.replace("_best.pt", "_value_heads.pt"))
        if not vh_path.exists():
            raise FileNotFoundError(f"value heads not found next to AR: {vh_path}")
        print(f"  AR: LoRA r={r} from {ar_path.name}, heads from {vh_path.name} "
              f"(best={ck.get('best', float('nan')):.3f})")
    else:
        model = backbone
        vh_path = ar_path / "value_heads.safetensors"
        print(f"  AR: frozen backbone, heads from {vh_path}")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    head_weights = load_value_head_weights(vh_path)
    value_heads = {}
    for layer_idx, w in head_weights.items():
        vh = torch.nn.Linear(w.shape[1], w.shape[0], bias=False,
                             dtype=torch.float32)
        vh.weight = torch.nn.Parameter(w.float(), requires_grad=False)
        value_heads[layer_idx] = vh.to(device).eval()
    print(f"  AR value heads: layers {sorted(value_heads.keys())}")

    return model, value_heads, tokenizer


@torch.no_grad()
def ar_reconstruct(ar_model, value_heads, ar_tokenizer, descriptions,
                   layers_needed, device):
    """One batched AR forward -> reconstructions for every requested layer.

    Returns {layer: tensor[G, d] (float32, on device)}. The backbone forward
    is shared: hidden states for all layers come from the same pass, so
    wrong-layer reconstructions are free.
    """
    prompts = [AR_TEMPLATE.replace("{explanation}", d) for d in descriptions]
    enc = ar_tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=AR_MAX_LEN).to(device)
    outputs = ar_model(**enc, output_hidden_states=True, use_cache=False)
    recons = {}
    for L in layers_needed:
        h = outputs.hidden_states[L + 1][:, -1, :].float()
        recons[L] = value_heads[L](h)
    return recons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-checkpoint", required=True,
                        help="phi_ar_stage2 *_best.pt OR legacy AR directory")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--n-negatives", type=int, default=3)
    parser.add_argument("--wrong-layer", action="store_true",
                        help="Add wrong-layer same-text negatives (free)")
    parser.add_argument("--rep-penalty", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.02,
                        help="KL-to-SFT-reference coefficient (0 disables)")
    parser.add_argument("--hard-negative-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="samples per optimizer step")
    parser.add_argument("--exclude-ids-file", default=None,
                        help="JSON list of text ids to hold out of GRPO "
                             "sampling (the AV's val_text_ids.json). Prevents "
                             "fine-tuning on the round-trip eval holdout.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    trust_remote = "phi" not in args.model.lower()

    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    acts_by_layer = act_data["activations"]
    n_layers = int(act_data["n_layers"])
    n_texts = int(act_data["n_texts"])
    d_model = int(act_data["d_model"])
    print("  %d layers, %d texts, d=%d" % (n_layers, n_texts, d_model))

    print("Loading AR from %s..." % args.ar_checkpoint)
    ar_model, ar_value_heads, ar_tokenizer = load_ar(
        args.ar_checkpoint, base_model_name, device, trust_remote)

    # Reward exists only where the AR has a trained head.
    ar_layers = sorted(L for L in ar_value_heads if L in acts_by_layer)
    if not ar_layers:
        raise ValueError("no overlap between AR value-head layers and activation file")
    print("  GRPO layers (AR-head ∩ activations): %s" % ar_layers)

    # Per-layer means: the centering that turns raw cosine into the real metric.
    layer_means = {L: acts_by_layer[L].float().mean(0) for L in ar_layers}
    layer_means_dev = {L: m.to(device) for L, m in layer_means.items()}

    # Hold the AV's validation texts out of GRPO sampling so we never
    # fine-tune on the round-trip eval set. The eval double-holdout is a
    # subset of the AV val split, so excluding AV val protects it fully.
    excluded_idx = set()
    if args.exclude_ids_file:
        exclude_ids = set(json.load(open(args.exclude_ids_file)))
        text_ids = list(act_data["ids"])
        text_id_set = set(text_ids)
        excluded_idx = {i for i, tid in enumerate(text_ids) if tid in exclude_ids}
        missing = len(exclude_ids - text_id_set)
        print("  excluding %d/%d held-out texts from sampling (%d ids not in "
              "activation file)" % (len(excluded_idx), n_texts, missing))
        if not excluded_idx:
            raise ValueError(
                "exclude-ids-file matched 0 texts — wrong file or id mismatch")

    samples = []
    for layer_idx in ar_layers:
        depth_pct = nearest_depth_pct(layer_idx, n_layers)
        for text_idx in range(n_texts):
            if text_idx in excluded_idx:
                continue
            samples.append((layer_idx, text_idx, depth_pct))
    print("  %d total (layer, text) pairs" % len(samples))

    # Wrong-layer partner: the trained layer farthest in depth.
    wrong_layer_of = {
        L: max(ar_layers, key=lambda M: abs(M - L)) for L in ar_layers
    }

    hard_neg_indices = build_hard_negative_index(
        acts_by_layer, layer_means, ar_layers, n_texts,
        k=args.hard_negative_k, exclude_idx=excluded_idx)

    print("Loading AV from %s..." % args.av_adapter)
    av_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    if av_tokenizer.pad_token is None:
        av_tokenizer.pad_token = av_tokenizer.eos_token

    av_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote)
    print("  AV backbone loaded, moving to %s..." % device, flush=True)
    av_base = av_base.to(device)
    print("  AV backbone on device, applying LoRA adapter...", flush=True)
    av_model = PeftModel.from_pretrained(av_base, args.av_adapter, is_trainable=True)
    print("  LoRA adapter applied", flush=True)
    av_model.train()

    trainable = sum(p.numel() for p in av_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in av_model.parameters())
    print("  LoRA: %d / %d trainable (%.2f%%)" % (trainable, total, 100 * trainable / total))

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

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")
    rng = np.random.RandomState(42)
    t_start = time.time()
    G = args.group_size
    embed_layer = av_model.get_input_embeddings()

    for epoch in range(args.epochs):
        indices = rng.choice(len(samples),
                             min(args.samples_per_epoch, len(samples)), replace=False)
        epoch_samples = [samples[i] for i in indices]

        epoch_reward = 0.0
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_neg = 0.0
        epoch_kl = 0.0
        n_done = 0
        sample_desc = ""

        for start in range(0, len(epoch_samples), args.batch_size):
            mini = epoch_samples[start:start + args.batch_size]
            optimizer.zero_grad()
            batch_reward = 0.0
            batch_loss = 0.0

            for layer_idx, text_idx, depth_pct in mini:
                act = acts_by_layer[layer_idx][text_idx].float()
                prompt_tokens, inject_pos = prompt_cache[depth_pct]
                pl = len(prompt_tokens)

                # --- generate the whole group in ONE batched call ---
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
                        pad_token_id=av_tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )

                group_gen_ids = [
                    strip_generated_row(output.sequences[g], prompt_tokens,
                                        eos_ids, stop_ids)
                    for g in range(G)
                ]
                group_descs = [
                    av_tokenizer.decode(g_ids, skip_special_tokens=True).strip()
                    for g_ids in group_gen_ids
                ]
                sample_desc = group_descs[0]

                # --- score: one AR forward, reconstructions for both layers ---
                wl = wrong_layer_of[layer_idx]
                layers_needed = [layer_idx, wl] if (args.wrong_layer and wl != layer_idx) \
                    else [layer_idx]
                recons = ar_reconstruct(ar_model, ar_value_heads, ar_tokenizer,
                                        group_descs, layers_needed, device)

                mean_L = layer_means_dev[layer_idx]
                correct_scores = centered_cosine(
                    recons[layer_idx], act.to(device), mean_L)        # [G]

                neg_pool = hard_neg_indices[layer_idx][text_idx]
                neg_idx = rng.choice(neg_pool[:args.hard_negative_k],
                                     min(args.n_negatives, len(neg_pool)),
                                     replace=False)
                neg_targets = torch.stack(
                    [acts_by_layer[layer_idx][int(j)].float() for j in neg_idx]
                ).to(device)
                neg_scores = centered_cosine(
                    recons[layer_idx], neg_targets, mean_L)           # [G, n_neg]
                worst_neg = neg_scores.max(dim=1).values              # [G]

                if args.wrong_layer and wl != layer_idx:
                    wl_target = acts_by_layer[wl][text_idx].float().to(device)
                    wl_scores = centered_cosine(
                        recons[wl], wl_target, layer_means_dev[wl])   # [G]
                    worst_neg = torch.maximum(worst_neg, wl_scores)

                rewards = (correct_scores - worst_neg).cpu()

                if args.rep_penalty > 0:
                    for i, gids in enumerate(group_gen_ids):
                        if len(gids) < 4:
                            continue
                        trigrams = [tuple(gids[j:j + 3]) for j in range(len(gids) - 2)]
                        rep_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
                        rewards[i] -= args.rep_penalty * rep_ratio

                epoch_correct += correct_scores.mean().item()
                epoch_neg += worst_neg.mean().item()

                mean_r = rewards.mean()
                std_r = rewards.std()
                batch_reward += mean_r.item()
                n_done += 1
                if n_done % 25 == 0:
                    print("  [epoch %d] %d/%d samples, mean_cos=%.3f neg=%.3f "
                          "(%.0fs elapsed)" %
                          (epoch + 1, n_done, len(epoch_samples),
                           epoch_correct / n_done, epoch_neg / n_done,
                           time.time() - t_start), flush=True)
                if std_r.item() < 1e-6:
                    continue
                advantages = (rewards - mean_r) / std_r

                # --- policy gradient: one batched forward over the group ---
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
                    logits = outputs.logits[row, pl - 1: pl - 1 + len(gen)]
                    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                    token_lps = log_probs[torch.arange(len(gen), device=device), gen_t]

                    loss_g = -advantages[g].to(device) * token_lps.sum()

                    if args.kl_coef > 0:
                        ref_logits = ref_out.logits[row, pl - 1: pl - 1 + len(gen)]
                        ref_lps = torch.nn.functional.log_softmax(
                            ref_logits.float(), dim=-1)
                        ref_token_lps = ref_lps[
                            torch.arange(len(gen), device=device), gen_t]
                        # k3 estimator: exp(ref-pol) - (ref-pol) - 1, per token
                        diff = ref_token_lps.detach() - token_lps
                        kl = (diff.exp() - diff - 1.0).mean()
                        epoch_kl += kl.item()
                        loss_g = loss_g + args.kl_coef * kl * len(gen)

                    sample_losses.append(loss_g)

                if sample_losses:
                    loss = sum(sample_losses) / (len(mini) * G)
                    loss.backward()
                    batch_loss += loss.detach().item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in av_model.parameters() if p.requires_grad],
                max_norm=1.0)
            optimizer.step()

            epoch_reward += batch_reward
            epoch_loss += batch_loss

        avg_reward = epoch_reward / max(n_done, 1)
        avg_correct = epoch_correct / max(n_done, 1)
        avg_neg = epoch_neg / max(n_done, 1)
        avg_kl = epoch_kl / max(n_done * G, 1)
        elapsed = time.time() - t_start

        print("Epoch %d/%d: reward=%.4f centered_cos=%.3f hard_neg_cos=%.3f "
              "gap=%.3f kl=%.4f (%ds)" %
              (epoch + 1, args.epochs, avg_reward, avg_correct, avg_neg,
               avg_correct - avg_neg, avg_kl, elapsed))
        print("  sample: %s" % sample_desc[:160])

        if avg_reward > best_reward:
            best_reward = avg_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print("  -> saved (best reward=%.4f)" % best_reward)

    print("\nDone. Best reward: %.4f" % best_reward)
    print("Output: %s" % args.output)


if __name__ == "__main__":
    main()
