#!/usr/bin/env python3
"""
Rejection-filtered SFT (RFT) refinement for the AV after SFT.

Takes a trained SFT AV adapter, samples multiple descriptions per activation,
scores them with AR reconstruction quality, and fine-tunes on the best
sample. This is not policy-gradient RL; it is best-of-N rejection filtering
followed by supervised fine-tuning on accepted samples.

Pipeline:
  1. train_av.py (SFT) → adapter that understands injection mechanism
  2. train_av_rft.py (this) → refined adapter with better descriptions

Reward signal:
  For each activation vector:
  - AV generates a description (sampling, not greedy)
  - AR reads the description and reconstructs the activation
  - Reward = cosine similarity between original and reconstructed activation
  This trains the AV to produce descriptions that are *useful*, not just
  close in cross-entropy to the DeepSeek-generated training descriptions.

Usage:
  python3 scripts/train_av_rft.py \
    --model qwen25-7b \
    --layer 20 \
    --sft-adapter output/nla-qwen25-7b-L20-av \
    --ar-model kitft/nla-qwen2.5-7b-L20-ar \
    --output output/nla-qwen25-7b-L20-av-rft
"""
import torch
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from generation_utils import decode_generated

REPO_ROOT = Path(__file__).parent.parent
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"
MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

device = torch.device("cuda")


def load_nla_meta(adapter_path):
    meta_path = Path(adapter_path) / "nla_meta.yaml"
    return yaml.safe_load(open(meta_path))


def load_ar_model(ar_model_path, device):
    """Load the AR (Activation Reconstructor) for reward computation."""
    print(f"Loading AR model from {ar_model_path}...")
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_model_path, trust_remote_code=True)
    ar_model = AutoModelForCausalLM.from_pretrained(
        ar_model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    )
    ar_model.eval()

    ar_meta_path = Path(ar_model_path) / "nla_meta.yaml"
    if ar_meta_path.exists():
        ar_meta = yaml.safe_load(open(ar_meta_path))
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(ar_model_path, "nla_meta.yaml")
        ar_meta = yaml.safe_load(open(path))

    templates = ar_meta["prompt_templates"]
    ar_template = templates.get("ar") or templates.get("critic")
    extraction_layer = ar_meta.get("extraction_layer_index", 20)

    return ar_model, ar_tokenizer, ar_template, extraction_layer


def compute_reward(ar_model, ar_tokenizer, ar_template, extraction_layer,
                   descriptions, target_activations):
    """Compute cosine similarity between AR-reconstructed and original activations.

    Extracts the hidden state at the extraction layer (not the last layer)
    at the last token position. The AR is trained to reconstruct activations
    at the same layer they were extracted from.
    """
    inner = ar_model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    blocks = inner.layers

    rewards = []
    for desc, target_act in zip(descriptions, target_activations):
        prompt = ar_template.replace("{explanation}", desc)
        tokens = ar_tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        seq_len = len(tokens) - 1

        layer_act = {}
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_act["h"] = h[0, seq_len].detach().cpu().float()
        handle = blocks[extraction_layer].register_forward_hook(hook)

        with torch.no_grad():
            ar_model(input_ids=input_ids)
        handle.remove()

        cos_sim = torch.nn.functional.cosine_similarity(
            layer_act["h"].unsqueeze(0),
            target_act.unsqueeze(0),
            dim=1
        ).item()
        rewards.append(cos_sim)

    return rewards


def rft_step(model, tokenizer, embed_layer, activations, meta,
             ar_model, ar_tokenizer, ar_template, ar_extraction_layer,
             optimizer, n_samples=4, temperature=0.8, max_new_tokens=150):
    """One RFT step: sample multiple completions, score, train on the best."""
    injection_char = meta["tokens"]["injection_char"]
    injection_scale = meta["extraction"]["injection_scale"]
    prompt_template = meta["prompt_templates"]["av"].replace("{injection_char}", injection_char)
    prompt_tokens = tokenizer.encode(prompt_template, add_special_tokens=False)
    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)

    batch_rewards = []
    batch_losses = []

    for act in activations:
        act_dev = act.to(device)

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        base_embeds = embed_layer(input_ids)
        base_embeds[0, inject_pos, :] = act_dev * injection_scale
        attention_mask = torch.ones_like(input_ids)

        samples = []
        model.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                output = model.generate(
                    inputs_embeds=base_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                samples.append(decode_generated(output, prompt_tokens, tokenizer))

        rewards = compute_reward(
            ar_model, ar_tokenizer, ar_template, ar_extraction_layer,
            samples, [act] * n_samples
        )

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        best_idx = np.argmax(rewards)
        best_sample = samples[best_idx]
        best_advantage = advantages[best_idx]

        if best_advantage > 0:
            model.train()
            best_tokens = tokenizer.encode(
                best_sample + "</explanation>", add_special_tokens=False
            )
            full_ids = prompt_tokens + best_tokens
            labels = [-100] * len(prompt_tokens) + best_tokens
            full_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
            labels = torch.tensor([labels], dtype=torch.long).to(device)

            embeds = embed_layer(full_ids)
            embeds[0, inject_pos, :] = act_dev * injection_scale

            outputs = model(inputs_embeds=embeds, labels=labels)
            loss = outputs.loss * best_advantage
            loss.backward()
            batch_losses.append(loss.item())

        batch_rewards.extend(rewards)

    if batch_losses:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(batch_rewards), np.mean(batch_losses) if batch_losses else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sft-adapter", type=str, required=True,
                        help="Path to SFT-trained LoRA adapter")
    parser.add_argument("--ar-model", type=str, default="kitft/nla-qwen2.5-7b-L20-ar",
                        help="AR model for reward computation")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Activations per RFT step")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Samples per activation for RFT")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    global device
    device = torch.device(args.device)

    model_name = MODELS[args.model]
    meta = load_nla_meta(args.sft_adapter)

    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading SFT adapter from {args.sft_adapter}...")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    ar_model, ar_tokenizer, ar_template, ar_extraction_layer = \
        load_ar_model(args.ar_model, device)

    act_path = ACTIVATIONS_DIR / f"{args.model}_L{args.layer}.pt"
    act_data = torch.load(act_path, weights_only=True)
    activations = act_data["activations"]
    print(f"Loaded {len(activations)} activations")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    embed_layer = model.get_input_embeddings()
    best_reward = -float("inf")
    reward_history = []

    print(f"\nRFT training for {args.steps} steps...")
    print(f"  {args.batch_size} activations × {args.n_samples} samples per step")

    for step in range(args.steps):
        idx = np.random.choice(len(activations), args.batch_size, replace=False)
        batch_acts = [activations[i] for i in idx]

        mean_reward, mean_loss = rft_step(
            model, tokenizer, embed_layer, batch_acts, meta,
            ar_model, ar_tokenizer, ar_template, ar_extraction_layer,
            optimizer, n_samples=args.n_samples, temperature=args.temperature,
        )

        reward_history.append(mean_reward)

        if (step + 1) % args.log_every == 0:
            recent = np.mean(reward_history[-args.log_every:])
            print(f"  Step {step+1}/{args.steps}: "
                  f"reward={mean_reward:.4f} recent_avg={recent:.4f} loss={mean_loss:.4f}")

            if recent > best_reward:
                best_reward = recent
                model.save_pretrained(args.output)
                tokenizer.save_pretrained(args.output)

                meta_out = dict(meta)
                meta_out["stage"] = "rft"
                meta_out["training"] = {
                    **meta_out.get("training", {}),
                    "rft_steps": step + 1,
                    "rft_lr": args.lr,
                    "rft_n_samples": args.n_samples,
                    "rft_best_reward": float(best_reward),
                    "ar_model": args.ar_model,
                }
                with open(Path(args.output) / "nla_meta.yaml", "w") as f:
                    yaml.dump(meta_out, f, default_flow_style=False)

                print(f"    -> saved (best_reward={best_reward:.4f})")

    print(f"\nDone. Best avg reward: {best_reward:.4f}")
    print(f"Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
