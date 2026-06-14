#!/usr/bin/env python3
"""
Train an Activation Reconstructor matching Anthropic's design:
  - Truncated backbone (K+1 layers for extraction layer K)
  - model.norm -> Identity, lm_head stripped
  - value_head: Linear(d, d, bias=False) trained from scratch
  - Backbone optionally fine-tuned (default: frozen, train value_head only)
  - MSE loss with both vectors normalized to sqrt(d)
  - Reads hidden state at last token position

Usage:
  python3 scripts/train_ar_truncated.py \
    --model qwen25-7b \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --descriptions corpus/generated/descriptions_L71pct.json \
    --output output/nla-qwen25-7b-L20-ar-v2 \
    --epochs 5 --lr 7e-5
"""
import torch
import json
import yaml
import argparse
import math
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).parent.parent

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


class ARDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        suffix = "</text> <summary>"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        self.items = []
        skipped = 0
        for ex in examples:
            prompt = AR_TEMPLATE.replace("{explanation}", ex["description"])
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > max_length:
                # Truncate description but preserve suffix (AR reads last token)
                tokens = tokens[:max_length - len(suffix_tokens)] + suffix_tokens
            # Verify suffix is intact
            if tokens[-len(suffix_tokens):] != suffix_tokens:
                skipped += 1
                continue
            self.items.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "activation": ex["activation"],
            })
        print(f"    Pre-tokenized {len(self.items)} examples"
              + (f" (skipped {skipped} with broken suffix)" if skipped else ""))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    seq_lens = []

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = 1
        seq_lens.append(seq_len)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "activations": activations,
        "seq_lens": seq_lens,
    }


def truncate_model(model, target_layers):
    """Truncate to target_layers, strip norm and lm_head."""
    inner = model.model if hasattr(model, "model") else model
    current_layers = len(inner.layers)

    if target_layers < current_layers:
        inner.layers = inner.layers[:target_layers]
        print(f"  Truncated: {current_layers} -> {target_layers} layers")

    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            print(f"  Set {attr} -> Identity")
            break

    model.lm_head = torch.nn.Identity()
    print(f"  Set lm_head -> Identity")

    return model


def compute_loss(pred, target, mse_scale):
    """MSE with both vectors normalized to mse_scale."""
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    tgt_n = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    return ((pred_n - tgt_n) ** 2).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Train AR with truncated backbone + value_head (Anthropic design)")
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--descriptions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--finetune-backbone", action="store_true",
                        help="Also fine-tune backbone (default: frozen, value_head only)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load activations
    print(f"Loading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    activations = act_data["activations"].float()
    text_ids = act_data["ids"]
    extraction_layer = int(act_data.get("layer", 20))
    d_model = int(act_data.get("d_model", activations.shape[1]))
    mse_scale = math.sqrt(d_model)
    print(f"  {len(text_ids)} texts, d={d_model}, layer={extraction_layer}")
    print(f"  Mean activation norm: {activations.norm(dim=1).mean():.1f}")

    # Load descriptions
    print(f"Loading descriptions from {args.descriptions}...")
    with open(args.descriptions) as f:
        desc_list = json.load(f)
    desc_map = {d["id"]: d["description"] for d in desc_list}
    print(f"  {len(desc_map)} descriptions")

    # Build examples
    examples = []
    for idx, tid in enumerate(text_ids):
        if tid in desc_map:
            examples.append({
                "activation": activations[idx],
                "description": desc_map[tid],
                "text_id": tid,
            })
    print(f"  {len(examples)} matched examples")

    # Split
    rng = np.random.RandomState(42)
    all_ids = sorted(set(ex["text_id"] for ex in examples))
    n_val = max(1, int(len(all_ids) * args.val_split))
    val_ids = set(rng.choice(all_ids, n_val, replace=False))
    train_examples = [ex for ex in examples if ex["text_id"] not in val_ids]
    val_examples = [ex for ex in examples if ex["text_id"] in val_ids]
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Load and truncate model
    model_name = MODELS[args.model]
    target_layers = extraction_layer + 1
    print(f"\nLoading {model_name} (will truncate to {target_layers} layers)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model = truncate_model(model, target_layers)

    # Freeze backbone unless --finetune-backbone
    if not args.finetune_backbone:
        for p in model.parameters():
            p.requires_grad = False
        print("  Backbone frozen")

    # Create value_head
    value_head = torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
    torch.nn.init.eye_(value_head.weight)
    value_head = value_head.to(device)
    print(f"  value_head: [{d_model}x{d_model}], init=identity")

    trainable_params = list(value_head.parameters())
    if args.finetune_backbone:
        trainable_params += [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {n_trainable:,}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    print("\nPre-tokenizing...")
    train_ds = ARDataset(train_examples, tokenizer)
    val_ds = ARDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Train
    best_val_cos = 0.0
    Path(args.output).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        value_head.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_acts = batch["activations"].to(device).float()
            seq_lens = batch["seq_lens"]

            outputs = model.model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  use_cache=False)
            hidden = outputs.last_hidden_state

            last_pos_hidden = torch.stack([
                hidden[i, seq_lens[i] - 1] for i in range(len(seq_lens))
            ])
            reconstructed = value_head(last_pos_hidden).float()

            loss = compute_loss(reconstructed, target_acts, mse_scale)
            loss.backward()
            if args.finetune_backbone:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  epoch {epoch+1} step {batch_idx+1}: loss={total_loss/n_batches:.4f}")

        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        value_head.eval()
        val_loss = 0
        val_cos = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_acts = batch["activations"].to(device).float()
                seq_lens = batch["seq_lens"]

                outputs = model.model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      use_cache=False)
                hidden = outputs.last_hidden_state

                last_pos_hidden = torch.stack([
                    hidden[i, seq_lens[i] - 1] for i in range(len(seq_lens))
                ])
                reconstructed = value_head(last_pos_hidden).float()

                loss = compute_loss(reconstructed, target_acts, mse_scale)
                cos = torch.nn.functional.cosine_similarity(
                    reconstructed, target_acts, dim=1).mean()

                val_loss += loss.item()
                val_cos += cos.item()
                val_n += 1

        val_loss /= val_n
        val_cos /= val_n
        print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_cos={val_cos:.4f}")

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            # Save value_head
            save_file({"weight": value_head.weight.data},
                      str(Path(args.output) / "value_head.safetensors"))
            # Save backbone if fine-tuned
            if args.finetune_backbone:
                model.save_pretrained(args.output)
                tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best cos={best_val_cos:.4f})")

    # Save metadata
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "ar",
        "stage": "sl",
        "d_model": d_model,
        "extraction_layer_index": extraction_layer,
        "extraction": {
            "injection_scale": None,
            "mse_scale": float(mse_scale),
        },
        "tokens": {
            "injection_char": "㈎",
            "injection_token_id": int(tokenizer.encode("㈎", add_special_tokens=False)[0]),
        },
        "prompt_templates": {
            "ar": AR_TEMPLATE,
        },
        "training": {
            "method": "value_head" + ("+finetune" if args.finetune_backbone else "+frozen"),
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "best_val_cosine": float(best_val_cos),
            "backbone_layers": target_layers,
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"\nDone. Best val cosine: {best_val_cos:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
