#!/usr/bin/env python3
"""
GRPO with hard negative mining for single-layer NLA.

Key improvement over train_av_grpo.py:
- Pre-computes pairwise activation cosine similarities
- Selects hard negatives (most similar activations from different texts)
- Uses multiple negatives per sample
- Evaluation uses within-category discrimination

The goal: kill SpongeBob. Force the model to describe what's SPECIFIC
to this activation, not what CATEGORY it belongs to.

Usage:
  python3 scripts/train_grpo_hard.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-L20-av-twin-clean-grpo-contrastive \
    --ar-checkpoint output/nla-qwen25-7b-L20-ar-twin \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --output output/nla-qwen25-7b-L20-av-grpo-hard \
    --n-negatives 3 --rep-penalty 0.2 --epochs 6
"""
import torch
import json
import yaml
import argparse
import math
import time
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from generation_utils import decode_generated

REPO_ROOT = Path(__file__).parent.parent

MODELS = {
    "gemma3-1b": "google/gemma-3-1b-it",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

INJECTION_CHARS = {
    "gemma3-1b": "⎝",
    "qwen25-7b": "㈎",
    "phi4-mini": "★",
}
INJECTION_SCALE = 150.0


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


AR_TEMPLATE = (
    "Summary of the following text: <text>{explanation}</text> <summary>{injection_char}"
)


def build_hard_negative_index(activations, k=10):
    """Pre-compute k nearest neighbors for each activation by cosine similarity."""
    print("Building hard negative index...", end=" ", flush=True)
    n = activations.shape[0]
    norms = activations.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normed = activations.float() / norms

    neighbors = []
    # Process in chunks to avoid OOM on large corpora
    chunk_size = 500
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = normed[start:end]  # [chunk, d]
        sims = chunk @ normed.T  # [chunk, n]
        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = -1.0
        topk = sims.topk(k, dim=1)
        for i in range(end - start):
            neighbors.append(topk.indices[i].tolist())

    print(f"done ({n} texts, k={k})")
    return neighbors


def load_ar_lora(ar_checkpoint, base_model_name, device, trust_remote,
                 injection_char, layer_idx):
    """Load LoRA-based AR (single-layer Qwen style)."""
    meta = yaml.safe_load(open(Path(ar_checkpoint) / "nla_meta.yaml"))
    d_model = int(meta["d_model"])

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote).to(device)

    ar_model = PeftModel.from_pretrained(backbone, ar_checkpoint)
    ar_model.eval()
    for p in ar_model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]

    return ar_model, tokenizer, inject_id, d_model


def ar_score_lora(ar_model, tokenizer, injection_char, inject_id,
                  descriptions, target_acts, layer_idx, device):
    """Score descriptions using LoRA AR (extract hidden state at injection point)."""
    cosines = []
    for desc, target in zip(descriptions, target_acts):
        prompt = AR_TEMPLATE.replace("{explanation}", desc).replace(
            "{injection_char}", injection_char)
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        inject_pos = None
        for i, tid in enumerate(tokens):
            if tid == inject_id:
                inject_pos = i
                break
        if inject_pos is None:
            cosines.append(0.0)
            continue

        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            outputs = ar_model(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            reconstructed = hidden[0, inject_pos]

        cos = torch.nn.functional.cosine_similarity(
            reconstructed.float().cpu().unsqueeze(0),
            target.float().unsqueeze(0)).item()
        cosines.append(cos)

    return cosines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-checkpoint", required=True)
    parser.add_argument("--activations", required=True,
                        help="Single-layer activations .pt file")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--depth-pct", type=int, default=71)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--n-negatives", type=int, default=3,
                        help="Number of hard negatives per sample")
    parser.add_argument("--rep-penalty", type=float, default=0.0)
    parser.add_argument("--hard-negative-k", type=int, default=20,
                        help="Pool size for hard negative sampling")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    trust_remote = "phi" not in args.model.lower()

    # Load activations
    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")

    if "activations" in act_data and isinstance(act_data["activations"], dict):
        activations = act_data["activations"][args.layer]
    elif isinstance(act_data, torch.Tensor):
        activations = act_data
    elif "activations" in act_data:
        activations = act_data["activations"]
    else:
        raise ValueError("Can't find activations in checkpoint")

    if "ids" in act_data:
        text_ids = act_data["ids"]
    else:
        text_ids = [f"text_{i}" for i in range(activations.shape[0])]

    n_texts = activations.shape[0]
    d_model = activations.shape[1]
    print(f"  {n_texts} texts, d={d_model}, layer={args.layer}")

    # Build hard negative index
    hard_neg_index = build_hard_negative_index(
        activations, k=args.hard_negative_k)

    # Load AR
    print(f"Loading AR from {args.ar_checkpoint}...")
    ar_model, ar_tokenizer, ar_inject_id, _ = load_ar_lora(
        args.ar_checkpoint, base_model_name, device, trust_remote,
        injection_char, args.layer)

    # Load AV
    print(f"Loading AV from {args.av_adapter}...")
    av_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    if av_tokenizer.pad_token is None:
        av_tokenizer.pad_token = av_tokenizer.eos_token

    av_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote).to(device)
    av_model = PeftModel.from_pretrained(av_base, args.av_adapter, is_trainable=True)
    av_model.train()

    trainable = sum(p.numel() for p in av_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in av_model.parameters())
    print(f"  LoRA: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    av_inject_id = av_tokenizer.encode(injection_char, add_special_tokens=False)[0]

    # Cache prompt tokens
    prompt_text = make_av_prompt(args.depth_pct, injection_char)
    chat_str = av_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True)
    prompt_tokens = av_tokenizer.encode(chat_str, add_special_tokens=False)
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == av_inject_id)
    print(f"  Prompt: {len(prompt_tokens)} tokens, inject at pos {inject_pos}")

    optimizer = torch.optim.AdamW(
        [p for p in av_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")
    rng = np.random.RandomState(42)
    t_start = time.time()

    for epoch in range(args.epochs):
        indices = rng.choice(n_texts,
                            min(args.samples_per_epoch, n_texts), replace=False)

        epoch_reward = 0
        epoch_loss = 0
        epoch_correct_cos = 0
        epoch_neg_cos = 0
        n_done = 0
        batch_size = 5

        for start in range(0, len(indices), batch_size):
            mini = indices[start:start + batch_size]
            optimizer.zero_grad()

            batch_reward = 0
            batch_loss = 0

            for text_idx in mini:
                act = activations[text_idx].float()
                embed_layer = av_model.get_input_embeddings()

                # Generate group of descriptions
                group_descs = []
                group_gen_ids = []
                for _ in range(args.group_size):
                    input_ids = torch.tensor(
                        [prompt_tokens], dtype=torch.long, device=device)
                    embeddings = embed_layer(input_ids)
                    embeddings[0, inject_pos, :] = normalize_activation(
                        act.to(device), INJECTION_SCALE).to(embeddings.dtype)

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

                    text = decode_generated(output, prompt_tokens, av_tokenizer)
                    seq = output.sequences[0]
                    pl = len(prompt_tokens)
                    gen_ids = seq[pl:].tolist() if seq.shape[0] > pl else seq.tolist()
                    while gen_ids and gen_ids[-1] in {
                        av_tokenizer.eos_token_id, av_tokenizer.pad_token_id}:
                        gen_ids.pop()
                    stop_ids = av_tokenizer.encode(
                        "</explanation>", add_special_tokens=False)
                    for i in range(len(gen_ids) - len(stop_ids) + 1):
                        if gen_ids[i:i+len(stop_ids)] == stop_ids:
                            gen_ids = gen_ids[:i]
                            break
                    group_descs.append(text)
                    group_gen_ids.append(gen_ids)

                # Score against correct activation
                correct_scores = ar_score_lora(
                    ar_model, ar_tokenizer, injection_char, ar_inject_id,
                    group_descs, [act] * args.group_size,
                    args.layer, device)

                # Score against HARD negatives (similar activations)
                neg_pool = hard_neg_index[text_idx]
                neg_indices = rng.choice(
                    neg_pool[:args.hard_negative_k],
                    min(args.n_negatives, len(neg_pool)),
                    replace=False)

                all_neg_scores = []
                for neg_idx in neg_indices:
                    neg_act = activations[neg_idx].float()
                    neg_scores = ar_score_lora(
                        ar_model, ar_tokenizer, injection_char, ar_inject_id,
                        group_descs, [neg_act] * args.group_size,
                        args.layer, device)
                    all_neg_scores.append(neg_scores)

                # Reward = correct - max(negatives)
                # Using max forces the description to beat ALL hard negatives
                rewards = []
                for g in range(args.group_size):
                    correct = correct_scores[g]
                    worst_neg = max(scores[g] for scores in all_neg_scores)
                    rewards.append(correct - worst_neg)

                # Repetition penalty
                if args.rep_penalty > 0:
                    for i, gids in enumerate(group_gen_ids):
                        if len(gids) < 4:
                            continue
                        trigrams = [tuple(gids[j:j+3]) for j in range(len(gids)-2)]
                        unique = len(set(trigrams))
                        rep_ratio = 1.0 - unique / len(trigrams)
                        rewards[i] -= args.rep_penalty * rep_ratio

                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std()

                epoch_correct_cos += sum(correct_scores) / len(correct_scores)
                epoch_neg_cos += sum(
                    max(s[g] for s in all_neg_scores)
                    for g in range(args.group_size)) / args.group_size

                if std_r.item() < 1e-6:
                    batch_reward += mean_r.item()
                    n_done += 1
                    continue

                advantages = (rewards_t - mean_r) / std_r

                for g_idx in range(args.group_size):
                    if not group_gen_ids[g_idx]:
                        continue
                    full_ids = prompt_tokens + group_gen_ids[g_idx]
                    input_ids = torch.tensor(
                        [full_ids], dtype=torch.long, device=device)
                    embeddings = embed_layer(input_ids)
                    embeddings[0, inject_pos, :] = normalize_activation(
                        act.to(device), INJECTION_SCALE).to(embeddings.dtype)

                    embeds_cast = embeddings.to(av_model.dtype)
                    if not embeds_cast.requires_grad:
                        embeds_cast.requires_grad_(True)

                    outputs = av_model(inputs_embeds=embeds_cast,
                                     attention_mask=torch.ones_like(input_ids))
                    logits = outputs.logits[0]
                    pl = len(prompt_tokens)
                    gen_t = torch.tensor(
                        group_gen_ids[g_idx], dtype=torch.long, device=device)
                    gen_logits = logits[pl - 1: pl - 1 + len(group_gen_ids[g_idx])]
                    log_probs = torch.nn.functional.log_softmax(
                        gen_logits.float(), dim=-1)
                    token_lps = log_probs[
                        torch.arange(len(group_gen_ids[g_idx]), device=device), gen_t]
                    log_prob = token_lps.sum()

                    loss_g = -advantages[g_idx].to(device) * log_prob
                    loss_g = loss_g / (len(mini) * args.group_size)
                    loss_g.backward()
                    batch_loss += loss_g.detach().item()

                batch_reward += mean_r.item()
                n_done += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in av_model.parameters() if p.requires_grad],
                max_norm=1.0)
            optimizer.step()

            epoch_reward += batch_reward
            epoch_loss += batch_loss

        avg_reward = epoch_reward / max(n_done, 1)
        avg_correct = epoch_correct_cos / max(n_done, 1)
        avg_neg = epoch_neg_cos / max(n_done, 1)
        elapsed = time.time() - t_start

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"reward={avg_reward:.4f} "
              f"correct_cos={avg_correct:.3f} "
              f"hard_neg_cos={avg_neg:.3f} "
              f"gap={avg_correct - avg_neg:.3f} "
              f"({elapsed:.0f}s)")

        if avg_reward > best_reward:
            best_reward = avg_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print(f"  -> saved (best reward={best_reward:.4f})")

    print(f"\nDone. Best reward: {best_reward:.4f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
