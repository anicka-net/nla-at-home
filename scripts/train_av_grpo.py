#!/usr/bin/env python3
"""
Group-normalized REINFORCE for AV using AR cosine as reward.

Not PPO/GRPO proper (no reference policy, no ratio clipping). Generates a group
of descriptions per activation, normalizes rewards within the group, and uses
the advantage-weighted log-prob as the REINFORCE gradient. This is sufficient
for reward shaping but will need a KL penalty or reference model for stability
at higher learning rates.

Pipeline: activation → AV generates description → AR reconstructs → cosine = reward

Usage:
  python3 scripts/train_av_grpo.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-L20-av-sonnet \
    --ar-checkpoint output/nla-qwen25-7b-L20-ar-sonnet \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --output output/nla-qwen25-7b-L20-av-sonnet-grpo \
    --epochs 3 --lr 1e-6 --group-size 8
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
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

INJECTION_CHARS = {
    "qwen25-7b": "㈎",
    "qwen3-4b": "㈎",
    "phi4-mini": "★",
}

INJECTION_SCALE = 150.0

AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def normalize_activation(v, target_scale):
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


def load_ar(ar_checkpoint, model_name, device):
    """Load AR: base model + LoRA adapter, hook-based extraction."""
    meta_path = Path(ar_checkpoint) / "nla_meta.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    extraction_layer = int(meta.get("extraction_layer_index", 20))
    d_model = int(meta["d_model"])
    mse_scale = math.sqrt(d_model)
    inject_char = meta["tokens"]["injection_char"]

    trust_remote = "phi" not in model_name.lower()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote).to(device)
    model = PeftModel.from_pretrained(model, ar_checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, extraction_layer, tokenizer, mse_scale, inject_char


def ar_score(ar_model, extraction_layer, ar_tokenizer, inject_char,
             descriptions, target_acts, device):
    """Score descriptions by AR reconstruction cosine (hook-based)."""
    inner = ar_model.model if hasattr(ar_model, 'model') else ar_model
    blocks = inner.model.layers
    inject_id = ar_tokenizer.encode(inject_char, add_special_tokens=False)[0]

    cosines = []
    for desc, target in zip(descriptions, target_acts):
        prompt = AR_TEMPLATE.replace("{explanation}", desc) + inject_char
        tokens = ar_tokenizer.encode(prompt, add_special_tokens=False)
        inject_pos = next((i for i, t in enumerate(tokens) if t == inject_id), len(tokens) - 1)
        input_ids = torch.tensor([tokens], device=device)

        layer_out = {}
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out['h'] = h
        handle = blocks[extraction_layer].register_forward_hook(hook_fn)

        with torch.no_grad():
            ar_model(input_ids=input_ids)
        handle.remove()

        hidden = layer_out['h'][0, inject_pos].float().cpu()
        cos = torch.nn.functional.cosine_similarity(
            hidden.unsqueeze(0), target.unsqueeze(0)).item()
        cosines.append(cos)

    return cosines


def generate_descriptions(av_model, av_tokenizer, activations, prompt_tokens,
                          inject_pos, device, temperature=0.7, max_new_tokens=200):
    """Generate descriptions from activations using the AV."""
    embed_layer = av_model.get_input_embeddings()
    descriptions = []

    for act in activations:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        embeddings = embed_layer(input_ids)
        embeddings[0, inject_pos, :] = normalize_activation(act.to(device), INJECTION_SCALE)

        with torch.no_grad():
            output = av_model.generate(
                inputs_embeds=embeddings.to(av_model.dtype),
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=av_tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        text = decode_generated(output, prompt_tokens, av_tokenizer)
        descriptions.append(text)

    return descriptions


def differentiable_log_probs(av_model, embed_layer, prompt_tokens, gen_token_ids,
                             inject_pos, activation, device, injection_scale):
    """Teacher-forcing forward pass to get log probs with gradients."""
    full_ids = prompt_tokens + gen_token_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    embeddings = embed_layer(input_ids)
    embeddings[0, inject_pos, :] = normalize_activation(activation.to(device), injection_scale)

    embeds_cast = embeddings.to(av_model.dtype)
    if not embeds_cast.requires_grad:
        embeds_cast.requires_grad_(True)

    outputs = av_model(
        inputs_embeds=embeds_cast,
        attention_mask=torch.ones_like(input_ids),
    )
    logits = outputs.logits[0]  # [seq_len, vocab]

    prompt_len = len(prompt_tokens)
    gen_ids_t = torch.tensor(gen_token_ids, dtype=torch.long, device=device)
    # logits[t] predicts token[t+1], so logits[prompt_len-1:prompt_len-1+gen_len]
    # predicts gen_token_ids[0:gen_len]
    gen_logits = logits[prompt_len - 1: prompt_len - 1 + len(gen_token_ids)]
    log_probs = torch.nn.functional.log_softmax(gen_logits.float(), dim=-1)
    token_log_probs = log_probs[torch.arange(len(gen_token_ids), device=device), gen_ids_t]
    return token_log_probs.sum()


def grpo_step(av_model, av_tokenizer, ar_model, extraction_layer, ar_tokenizer,
              ar_inject_char, activations, prompt_tokens, inject_pos, optimizer,
              device, group_size=4, temperature=0.7, max_new_tokens=200,
              contrastive=False, rep_penalty=0.0):
    """One GRPO step: generate group, score, compute advantage, policy gradient."""
    embed_layer = av_model.get_input_embeddings()
    total_reward = 0
    total_loss = 0
    n_samples = 0

    optimizer.zero_grad()
    total_group_std = 0.0
    n_skipped = 0

    for act_idx, act in enumerate(activations):
        group_descs = []
        group_gen_ids = []

        # Phase 1: generate descriptions (no grad)
        for _ in range(group_size):
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            embeddings = embed_layer(input_ids)
            embeddings[0, inject_pos, :] = normalize_activation(act.to(device), INJECTION_SCALE)

            with torch.no_grad():
                output = av_model.generate(
                    inputs_embeds=embeddings.to(av_model.dtype),
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=av_tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )

            text = decode_generated(output, prompt_tokens, av_tokenizer)
            seq = output.sequences[0]
            prompt_len = len(prompt_tokens)
            starts_with_prompt = (
                seq.shape[0] > prompt_len
                and seq[:prompt_len].tolist() == prompt_tokens
            )
            gen_ids = seq[prompt_len:].tolist() if starts_with_prompt else seq.tolist()
            while gen_ids and gen_ids[-1] in {av_tokenizer.eos_token_id, av_tokenizer.pad_token_id}:
                gen_ids.pop()
            stop_ids = av_tokenizer.encode("</explanation>", add_special_tokens=False)
            for i in range(len(gen_ids) - len(stop_ids) + 1):
                if gen_ids[i:i+len(stop_ids)] == stop_ids:
                    gen_ids = gen_ids[:i]
                    break
            group_descs.append(text)
            group_gen_ids.append(gen_ids)

        # Phase 2: score with AR (no grad)
        correct_scores = ar_score(
            ar_model, extraction_layer, ar_tokenizer, ar_inject_char,
            group_descs, [act.float()] * group_size, device)

        if contrastive:
            wrong_idx = (act_idx + 1) % len(activations)
            wrong_act = activations[wrong_idx]
            wrong_scores = ar_score(
                ar_model, extraction_layer, ar_tokenizer, ar_inject_char,
                group_descs, [wrong_act.float()] * group_size, device)
            rewards = [c - w for c, w in zip(correct_scores, wrong_scores)]
        else:
            rewards = correct_scores

        # Repetition penalty: penalize repeated trigrams
        if rep_penalty > 0:
            for i, gids in enumerate(group_gen_ids):
                if len(gids) < 4:
                    continue
                trigrams = [tuple(gids[j:j+3]) for j in range(len(gids)-2)]
                unique = len(set(trigrams))
                rep_ratio = 1.0 - unique / len(trigrams)  # 0 = no repeats, 1 = all repeats
                rewards[i] -= rep_penalty * rep_ratio

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        total_group_std += std_r.item()

        if std_r.item() < 1e-6:
            total_reward += mean_r.item()
            n_skipped += 1
            n_samples += 1
            continue

        advantages = (rewards_t - mean_r) / std_r

        # Phase 3: differentiable forward pass for policy gradient
        step_loss = 0.0
        n_empty = sum(1 for g in group_gen_ids if not g)
        if act_idx == 0 and n_empty > 0:
            print(f"    WARNING: {n_empty}/{group_size} empty gen_ids in first activation")
        for g_idx in range(group_size):
            if not group_gen_ids[g_idx]:
                continue
            log_prob = differentiable_log_probs(
                av_model, embed_layer, prompt_tokens, group_gen_ids[g_idx],
                inject_pos, act, device, INJECTION_SCALE)
            assert log_prob.requires_grad, f"log_prob has no grad! gen_len={len(group_gen_ids[g_idx])}"
            loss_g = -advantages[g_idx].to(device) * log_prob
            loss_g = loss_g / (len(activations) * group_size)
            loss_g.backward()
            step_loss += loss_g.detach().item()

        total_reward += mean_r.item()
        total_loss += step_loss
        n_samples += 1

    n_backward = n_samples - n_skipped
    if n_backward > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in av_model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
        optimizer.zero_grad()

    avg_group_std = total_group_std / max(n_samples, 1)
    mean_reward = total_reward / max(n_samples, 1)
    mean_loss = total_loss / max(n_samples, 1)
    return mean_reward, mean_loss, grad_norm.item(), avg_group_std, n_skipped


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for AV using AR cosine reward")
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", type=str, required=True)
    parser.add_argument("--ar-checkpoint", type=str, required=True)
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--contrastive", action="store_true",
                        help="Contrastive reward: correct_score - wrong_score (prevents template hacking)")
    parser.add_argument("--rep-penalty", type=float, default=0.0,
                        help="Penalty for repeated trigrams in generated descriptions (0.2 = moderate)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]

    # Load activations
    print(f"Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    activations = act_data["activations"].float()
    text_ids = act_data["ids"]
    print(f"  {len(text_ids)} activations")

    # Load AR (critic)
    print(f"Loading AR from {args.ar_checkpoint}...")
    ar_model, extraction_layer, ar_tokenizer, mse_scale, ar_inject_char = load_ar(
        args.ar_checkpoint, model_name, device)
    print(f"  AR loaded (LoRA, hook at layer {extraction_layer})")

    # Load AV (actor)
    print(f"Loading AV from {args.av_adapter}...")
    av_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code="phi" not in args.model.lower())
    if av_tokenizer.pad_token is None:
        av_tokenizer.pad_token = av_tokenizer.eos_token

    av_base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code="phi" not in args.model.lower()).to(device)
    av_model = PeftModel.from_pretrained(av_base, args.av_adapter, is_trainable=True)
    av_model.train()
    trainable = sum(p.numel() for p in av_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in av_model.parameters())
    print(f"  LoRA: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    # Prepare AV prompt with chat template
    av_meta_path = Path(args.av_adapter) / "nla_meta.yaml"
    with open(av_meta_path) as f:
        av_meta = yaml.safe_load(f)
    template = av_meta["prompt_templates"]["av"]
    content = template.replace("{injection_char}", injection_char)
    chat_str = av_tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = av_tokenizer.encode(chat_str, add_special_tokens=False)
    inject_id = av_tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)
    print(f"  AV prompt: {len(prompt_tokens)} tokens, inject at {inject_pos}")

    optimizer = torch.optim.AdamW(
        [p for p in av_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")
    rng = np.random.RandomState(42)

    t_start = time.time()
    for epoch in range(args.epochs):
        indices = rng.choice(len(activations), args.samples_per_epoch, replace=False)
        batch_acts = [activations[i] for i in indices]

        epoch_reward = 0
        epoch_loss = 0
        batch_size = 10
        epoch_t0 = time.time()

        for start in range(0, len(batch_acts), batch_size):
            mini_batch = batch_acts[start:start + batch_size]
            step_t0 = time.time()

            reward, loss, grad_norm, group_std, n_skip = grpo_step(
                av_model, av_tokenizer,
                ar_model, extraction_layer, ar_tokenizer, ar_inject_char,
                mini_batch, prompt_tokens, inject_pos, optimizer,
                device,
                group_size=args.group_size,
                temperature=args.temperature,
                contrastive=args.contrastive,
                rep_penalty=args.rep_penalty,
            )

            epoch_reward += reward * len(mini_batch)
            epoch_loss += loss * len(mini_batch)
            step_dt = time.time() - step_t0
            elapsed = time.time() - t_start

            print(f"  epoch {epoch+1} step {start+batch_size}/{len(batch_acts)}: "
                  f"reward={reward:.4f} loss={loss:.6f} "
                  f"grad={grad_norm:.4f} group_std={group_std:.6f} "
                  f"skipped={n_skip}/{len(mini_batch)} "
                  f"[{step_dt:.0f}s/step, {elapsed/60:.0f}m total]")

        epoch_reward /= len(batch_acts)
        epoch_loss /= len(batch_acts)
        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"mean_reward={epoch_reward:.4f} mean_loss={epoch_loss:.4f}")

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            av_model.save_pretrained(args.output)
            av_tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best reward={best_reward:.4f})")

    # Save metadata
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "stage": "rl",
        "d_model": int(act_data.get("d_model", activations.shape[1])),
        "extraction_layer_index": int(act_data.get("layer", 20)),
        "extraction": {"injection_scale": INJECTION_SCALE},
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(inject_id),
        },
        "prompt_templates": {
            "av": template,
            "ar": AR_TEMPLATE,
        },
        "training": {
            "method": "grpo",
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
