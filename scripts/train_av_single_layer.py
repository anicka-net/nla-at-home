#!/usr/bin/env python3
"""
Train an Activation Verbalizer for a single layer, matching Anthropic's design:
  - L2-normalize activations to injection_scale (not multiply)
  - Chat template wrapping
  - LoRA fine-tuning (practical constraint; Anthropic does full FT)

Usage:
  python3 scripts/train_av_single_layer.py \
    --model qwen25-7b \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --descriptions corpus/generated/descriptions_L71pct.json \
    --output output/nla-qwen25-7b-L20-av-v2 \
    --epochs 5 --lr 8e-6
"""
import torch
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from generation_utils import decode_generated

REPO_ROOT = Path(__file__).parent.parent

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

INJECTION_CHARS = {
    "qwen25-7b": "㈎",
    "qwen3-4b": "㈎",
}

INJECTION_SCALE = 150.0

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. "
    "You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation.\n\n"
    "<explanation>"
)


def normalize_activation(v, target_scale):
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


def find_inject_pos(prompt_tokens, injection_token_id):
    for i, tid in enumerate(prompt_tokens):
        if tid == injection_token_id:
            return i
    raise ValueError("Injection token not found in prompt")


class AVDataset(Dataset):
    def __init__(self, examples, tokenizer, injection_char, prompt_tokens,
                 inject_pos, max_length=512):
        self.items = []
        prompt_len = len(prompt_tokens)

        for ex in examples:
            desc_text = ex["description"] + "</explanation>"
            desc_tokens = tokenizer.encode(desc_text, add_special_tokens=False)

            input_ids = list(prompt_tokens) + desc_tokens
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) > max_length:
                labels = labels[:max_length]

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "activation": ex["activation"],
                "inject_pos": inject_pos,
            })
        print(f"    Pre-tokenized {len(self.items)} examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    inject_positions = [b["inject_pos"] for b in batch]

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "activations": activations,
        "inject_positions": inject_positions,
    }


def train(model, tokenizer, train_dataset, val_dataset, prompt_tokens,
          inject_pos, args, device):
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    embed_layer = model.get_input_embeddings()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activations = batch["activations"].to(device)
            inject_positions = batch["inject_positions"]

            embeddings = embed_layer(input_ids)
            for i, pos in enumerate(inject_positions):
                act = activations[i]
                if args.noise_std > 0:
                    act = act + torch.randn_like(act) * args.noise_std * act.norm()
                embeddings[i, pos, :] = normalize_activation(act, INJECTION_SCALE)

            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 200 == 0:
                print(f"  epoch {epoch+1} step {batch_idx+1}: loss={total_loss/n_batches:.3f}")

        train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                activations = batch["activations"].to(device)
                inject_positions = batch["inject_positions"]

                embeddings = embed_layer(input_ids)
                for i, pos in enumerate(inject_positions):
                    embeddings[i, pos, :] = normalize_activation(
                        activations[i], INJECTION_SCALE)

                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        print(f"  Epoch {epoch+1}/{args.epochs}: train={train_loss:.3f} val={val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best val={best_val_loss:.3f})")

    return best_val_loss


def evaluate(model, tokenizer, examples, injection_char, prompt_tokens,
             inject_pos, device, n_samples=8):
    model.eval()
    rng = np.random.RandomState(42)
    selected = rng.choice(len(examples), min(n_samples, len(examples)), replace=False)

    print(f"\n{'='*60}")
    print(f"Sample generations")
    print(f"{'='*60}")

    embed_layer = model.get_input_embeddings()
    for idx in selected:
        ex = examples[idx]
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        embeddings = embed_layer(input_ids)
        embeddings[0, inject_pos, :] = normalize_activation(
            ex["activation"].to(device), INJECTION_SCALE)

        with torch.no_grad():
            output = model.generate(
                inputs_embeds=embeddings.to(model.dtype),
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated = decode_generated(output, prompt_tokens, tokenizer)
        print(f"\n[{ex['text_id']}]")
        print(f"  GT:  {ex['description'][:150]}")
        print(f"  Gen: {generated[:150]}")


def main():
    parser = argparse.ArgumentParser(
        description="Train single-layer AV matching Anthropic design")
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--descriptions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.15)
    parser.add_argument("--full-ft", action="store_true",
                        help="Full fine-tune (no LoRA). Uses gradient checkpointing.")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise std added to activations during training (0=off)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    injection_char = INJECTION_CHARS[args.model]

    # Load data
    print(f"Loading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    activations = act_data["activations"].float()
    text_ids = act_data["ids"]
    extraction_layer = int(act_data.get("layer", 20))
    d_model = int(act_data.get("d_model", activations.shape[1]))
    print(f"  {len(text_ids)} texts, d={d_model}, layer={extraction_layer}")
    print(f"  Mean norm: {activations.norm(dim=1).mean():.1f}")

    print(f"Loading descriptions from {args.descriptions}...")
    with open(args.descriptions) as f:
        desc_list = json.load(f)
    desc_map = {d["id"]: d["description"] for d in desc_list}
    print(f"  {len(desc_map)} descriptions")

    examples = []
    for idx, tid in enumerate(text_ids):
        if tid in desc_map:
            examples.append({
                "activation": activations[idx],
                "description": desc_map[tid],
                "text_id": tid,
            })
    print(f"  {len(examples)} matched examples")

    # Load model and tokenizer
    model_name = MODELS[args.model]
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompt with chat template
    content = AV_TEMPLATE.replace("{injection_char}", injection_char)
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = find_inject_pos(prompt_tokens, injection_token_id)
    print(f"  Prompt: {len(prompt_tokens)} tokens, inject at pos {inject_pos}")

    if args.full_ft:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(device)
        model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in model.parameters())
        print(f"  Full FT: {trainable:,} params, gradient checkpointing ON")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

    if args.eval_only:
        if not args.full_ft:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.output)
        evaluate(model, tokenizer, examples, injection_char,
                 prompt_tokens, inject_pos, device)
        return

    if not args.full_ft:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Split by text ID
    all_ids = sorted(set(ex["text_id"] for ex in examples))
    n_val = max(1, int(len(all_ids) * args.val_split))
    rng = np.random.RandomState(42)
    val_ids = set(rng.choice(all_ids, n_val, replace=False))
    train_examples = [ex for ex in examples if ex["text_id"] not in val_ids]
    val_examples = [ex for ex in examples if ex["text_id"] in val_ids]
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    train_ds = AVDataset(train_examples, tokenizer, injection_char,
                         prompt_tokens, inject_pos)
    val_ds = AVDataset(val_examples, tokenizer, injection_char,
                       prompt_tokens, inject_pos)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"\nTraining for {args.epochs} epochs...")
    best = train(model, tokenizer, train_ds, val_ds, prompt_tokens,
                 inject_pos, args, device)
    print(f"\nBest val loss: {best:.3f}")

    evaluate(model, tokenizer, val_examples, injection_char,
             prompt_tokens, inject_pos, device)

    # Save metadata
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "stage": "sft",
        "d_model": d_model,
        "extraction_layer_index": extraction_layer,
        "extraction": {"injection_scale": INJECTION_SCALE},
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(injection_token_id),
        },
        "prompt_templates": {
            "av": AV_TEMPLATE,
            "ar": "Summary of the following text: <text>{explanation}</text> <summary>",
        },
        "training": {
            "method": "full_ft_sft" if args.full_ft else "lora_sft",
            "injection_mode": "normalize",
            "chat_template": True,
            **({"lora_r": args.lora_r} if not args.full_ft else {}),
            "lr": args.lr,
            "epochs": args.epochs,
            "best_val_loss": float(best),
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
