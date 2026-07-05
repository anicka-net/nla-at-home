#!/usr/bin/env python3
"""
Universal GRPO for depth-conditioned AV using AR cosine as reward.

Contrastive reward with wrong-activation AND wrong-layer negatives,
plus trigram repetition penalty.

Pipeline: activation@depth → AV generates description → AR reconstructs → cosine = reward

Usage:
  python3 scripts/train_universal_grpo.py \
    --model phi4-mini \
    --av-adapter output/nla-phi4-mini-universal-av \
    --ar-checkpoint output/nla-phi4-mini-universal-ar \
    --activations corpus/activations/phi4-mini_all_layers.pt \
    --output output/nla-phi4-mini-universal-av-grpo \
    --contrastive --rep-penalty 0.2 \
    --epochs 6 --lr 5e-6
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
from safetensors import safe_open
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

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]


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


AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def load_ar_valuehead(ar_checkpoint, base_model_name, device, trust_remote):
    """Load universal AR with per-layer value_heads."""
    meta = yaml.safe_load(open(Path(ar_checkpoint) / "nla_meta.yaml"))
    d_model = int(meta["d_model"])
    n_layers = int(meta["n_layers"])

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote).to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    inner = backbone.model if hasattr(backbone, "model") else backbone
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            break
    backbone.lm_head = torch.nn.Identity()
    backbone.eval()

    # Load value_heads
    vh_path = Path(ar_checkpoint) / "value_heads.safetensors"
    value_heads = {}
    with safe_open(str(vh_path), framework="pt") as f:
        for key in f.keys():
            layer_idx = int(key.split(".")[1])
            w = f.get_tensor(key)
            vh = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
            vh.weight = torch.nn.Parameter(w)
            vh = vh.to(device).eval()
            value_heads[layer_idx] = vh

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return backbone, value_heads, tokenizer, d_model, n_layers


def ar_score_valuehead(backbone, value_heads, tokenizer, injection_char,
                       descriptions, target_acts, layer_idx, device):
    """Score descriptions using value_head AR."""
    inner = backbone.model if hasattr(backbone, "model") else backbone

    cosines = []
    for desc, target in zip(descriptions, target_acts):
        prompt = AR_TEMPLATE.replace("{explanation}", desc)
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            outputs = inner(input_ids=input_ids, use_cache=False,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            last_h = hidden[0, -1]
            reconstructed = value_heads[layer_idx](last_h.unsqueeze(0)).squeeze(0)

        cos = torch.nn.functional.cosine_similarity(
            reconstructed.float().cpu().unsqueeze(0),
            target.unsqueeze(0)).item()
        cosines.append(cos)

    return cosines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-checkpoint", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--wrong-layer", action="store_true",
                        help="Add wrong-layer same-text negatives")
    parser.add_argument("--rep-penalty", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    trust_remote = "phi" not in args.model.lower()

    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    acts_by_layer = act_data["activations"]
    text_ids = act_data["ids"]
    n_layers = int(act_data["n_layers"])
    n_texts = int(act_data["n_texts"])
    d_model = int(act_data["d_model"])
    print(f"  {n_layers} layers, {n_texts} texts, d={d_model}")

    samples = []
    for layer_idx in range(n_layers):
        depth_pct = nearest_depth_pct(layer_idx, n_layers)
        for text_idx in range(n_texts):
            samples.append((layer_idx, text_idx, depth_pct))
    print(f"  {len(samples)} total (layer, text) pairs")

    print(f"Loading AR from {args.ar_checkpoint}...")
    ar_backbone, ar_value_heads, ar_tokenizer, _, _ = load_ar_valuehead(
        args.ar_checkpoint, base_model_name, device, trust_remote)

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

    inject_id = av_tokenizer.encode(injection_char, add_special_tokens=False)[0]

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

    for epoch in range(args.epochs):
        indices = rng.choice(len(samples),
                            min(args.samples_per_epoch, len(samples)), replace=False)
        epoch_samples = [samples[i] for i in indices]

        epoch_reward = 0
        epoch_loss = 0
        n_done = 0
        batch_size = 5

        for start in range(0, len(epoch_samples), batch_size):
            mini = epoch_samples[start:start + batch_size]
            step_t0 = time.time()
            optimizer.zero_grad()

            batch_reward = 0
            batch_loss = 0
            n_skipped = 0

            for layer_idx, text_idx, depth_pct in mini:
                act = acts_by_layer[layer_idx][text_idx].float()
                prompt_tokens, inject_pos = prompt_cache[depth_pct]
                embed_layer = av_model.get_input_embeddings()

                group_descs = []
                group_gen_ids = []
                for _ in range(args.group_size):
                    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
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
                    while gen_ids and gen_ids[-1] in {av_tokenizer.eos_token_id, av_tokenizer.pad_token_id}:
                        gen_ids.pop()
                    stop_ids = av_tokenizer.encode("</explanation>", add_special_tokens=False)
                    for i in range(len(gen_ids) - len(stop_ids) + 1):
                        if gen_ids[i:i+len(stop_ids)] == stop_ids:
                            gen_ids = gen_ids[:i]
                            break
                    group_descs.append(text)
                    group_gen_ids.append(gen_ids)

                # Score: correct activation at correct layer
                correct_scores = ar_score_valuehead(
                    ar_backbone, ar_value_heads, ar_tokenizer, injection_char,
                    group_descs, [act] * args.group_size, layer_idx, device)

                if args.contrastive:
                    # Wrong-activation negative (different text, same layer)
                    wrong_text_idx = (text_idx + 1) % n_texts
                    wrong_act = acts_by_layer[layer_idx][wrong_text_idx].float()
                    wrong_scores = ar_score_valuehead(
                        ar_backbone, ar_value_heads, ar_tokenizer, injection_char,
                        group_descs, [wrong_act] * args.group_size, layer_idx, device)
                    rewards = [c - w for c, w in zip(correct_scores, wrong_scores)]

                    # Wrong-layer negative (same text, different layer)
                    if args.wrong_layer:
                        wrong_layer = (layer_idx + n_layers // 2) % n_layers
                        wrong_layer_act = acts_by_layer[wrong_layer][text_idx].float()
                        wl_scores = ar_score_valuehead(
                            ar_backbone, ar_value_heads, ar_tokenizer, injection_char,
                            group_descs, [wrong_layer_act] * args.group_size,
                            wrong_layer, device)
                        rewards = [r - wl for r, wl in zip(rewards, wl_scores)]
                else:
                    rewards = correct_scores

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

                if std_r.item() < 1e-6:
                    batch_reward += mean_r.item()
                    n_skipped += 1
                    continue

                advantages = (rewards_t - mean_r) / std_r

                for g_idx in range(args.group_size):
                    if not group_gen_ids[g_idx]:
                        continue
                    full_ids = prompt_tokens + group_gen_ids[g_idx]
                    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
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
                    gen_t = torch.tensor(group_gen_ids[g_idx], dtype=torch.long, device=device)
                    gen_logits = logits[pl - 1: pl - 1 + len(group_gen_ids[g_idx])]
                    log_probs = torch.nn.functional.log_softmax(gen_logits.float(), dim=-1)
                    token_lps = log_probs[torch.arange(len(group_gen_ids[g_idx]), device=device), gen_t]
                    log_prob = token_lps.sum()

                    loss_g = -advantages[g_idx].to(device) * log_prob
                    loss_g = loss_g / (len(mini) * args.group_size)
                    loss_g.backward()
                    batch_loss += loss_g.detach().item()

                batch_reward += mean_r.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in av_model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            n_done += len(mini)
            epoch_reward += batch_reward
            epoch_loss += batch_loss
            step_dt = time.time() - step_t0
            elapsed = time.time() - t_start

            if n_done % 20 == 0 or start == 0:
                avg_r = batch_reward / len(mini)
                print(f"  epoch {epoch+1} [{n_done}/{len(epoch_samples)}]: "
                      f"reward={avg_r:.4f} grad={grad_norm:.4f} "
                      f"skip={n_skipped}/{len(mini)} "
                      f"[{step_dt:.0f}s, {elapsed/60:.0f}m total]",
                      flush=True)

        epoch_reward /= len(epoch_samples)
        print(f"  Epoch {epoch+1}/{args.epochs}: mean_reward={epoch_reward:.4f}",
              flush=True)

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best reward={best_reward:.4f})")

    av_meta = yaml.safe_load(open(Path(args.av_adapter) / "nla_meta.yaml"))
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "variant": "universal",
        "stage": "rl",
        "d_model": d_model,
        "n_layers": n_layers,
        "depth_percentages": DEPTH_PCTS,
        "extraction": {"injection_scale": INJECTION_SCALE},
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(inject_id),
        },
        "prompt_templates": av_meta.get("prompt_templates", {}),
        "training": {
            "method": "grpo",
            "contrastive": args.contrastive,
            "wrong_layer_negatives": args.wrong_layer,
            "rep_penalty": args.rep_penalty,
            "injection_mode": "normalize",
            "chat_template": True,
            "base_av": str(args.av_adapter),
            "ar_critic": str(args.ar_checkpoint),
            "lr": args.lr,
            "epochs": args.epochs,
            "group_size": args.group_size,
            "samples_per_epoch": args.samples_per_epoch,
            "best_mean_reward": float(best_reward),
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"\nDone. Best mean reward: {best_reward:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
