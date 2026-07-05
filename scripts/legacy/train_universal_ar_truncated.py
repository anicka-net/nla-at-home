#!/usr/bin/env python3
"""
Train a Universal Activation Reconstructor — all layers, Anthropic-style design.

  - Frozen backbone (base Qwen/Gemma)
  - Per-layer value_heads: one Linear(d, d) per layer
  - Dynamic truncation: for layer K, only run K+1 backbone blocks
  - MSE loss with both vectors normalized to sqrt(d)
  - Reads hidden state at last token position

This extends train_ar_truncated.py (single-layer) to all layers simultaneously.

Usage:
  python3 scripts/train_universal_ar_truncated.py \
    --model qwen25-7b \
    --activations corpus/activations/qwen25-7b_all_layers.pt \
    --output output/nla-qwen25-7b-universal-ar-prediction \
    --epochs 5 --lr 7e-5
"""
import torch
import json
import yaml
import argparse
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "gemma3-1b": "google/gemma-3-1b-it",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]

AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def nearest_depth_pct(layer, n_layers):
    depth = layer * 100 / n_layers
    return min(DEPTH_PCTS, key=lambda p: abs(p - depth))


def load_descriptions(suffix="_prediction"):
    """Load descriptions at all depth percentages."""
    descs = {}
    for pct in DEPTH_PCTS:
        for pattern in [
            f"descriptions_L{pct}pct{suffix}.json",
            f"descriptions_L{pct}pct{suffix}_merged.json",
            f"descriptions_L{pct}pct_merged.json",
            f"descriptions_L{pct}pct.json",
        ]:
            path = GENERATED_DIR / pattern
            if path.exists():
                data = json.loads(path.read_text())
                descs[pct] = {d["id"]: d["description"] for d in data}
                print(f"  L{pct}%: {len(descs[pct])} descriptions from {path.name}")
                break
        else:
            print(f"  L{pct}%: NO FILE FOUND")
    return descs


class LayerARDataset(Dataset):
    """Pre-tokenized dataset for a single layer."""
    def __init__(self, examples, tokenizer, max_length=512):
        suffix = "</text> <summary>"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        self.items = []
        skipped = 0
        for ex in examples:
            prompt = AR_TEMPLATE.replace("{explanation}", ex["description"])
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > max_length:
                tokens = tokens[:max_length - len(suffix_tokens)] + suffix_tokens
            if tokens[-len(suffix_tokens):] != suffix_tokens:
                skipped += 1
                continue
            self.items.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "activation": ex["activation"],
            })
        if skipped:
            print(f"      (skipped {skipped} with broken suffix)")

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


def compute_loss(pred, target, mse_scale):
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    tgt_n = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    return ((pred_n - tgt_n) ** 2).mean()


class DynamicTruncatedAR(torch.nn.Module):
    """Wrapper that runs backbone to depth K and applies value_head_K."""

    def __init__(self, backbone, n_layers, d_model, dtype=torch.bfloat16):
        super().__init__()
        self.backbone = backbone
        self.n_layers = n_layers

        self.value_heads = torch.nn.ModuleDict({
            str(k): torch.nn.Linear(d_model, d_model, bias=False, dtype=dtype)
            for k in range(n_layers)
        })
        for vh in self.value_heads.values():
            torch.nn.init.eye_(vh.weight)

    def forward(self, input_ids, attention_mask, layer_idx, seq_lens):
        """Run backbone, extract hidden state at layer_idx, apply value_head."""
        inner = self.backbone.model if hasattr(self.backbone, "model") else self.backbone

        outputs = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        # hidden_states[0] = embeddings, hidden_states[k+1] = output of layer k
        hidden = outputs.hidden_states[layer_idx + 1]

        last_pos_hidden = torch.stack([
            hidden[i, seq_lens[i] - 1] for i in range(len(seq_lens))
        ])

        reconstructed = self.value_heads[str(layer_idx)](last_pos_hidden)
        return reconstructed.float()


def build_examples(act_data, descriptions_by_depth):
    """Build examples grouped by layer."""
    layer_acts = act_data["activations"]
    ids = act_data["ids"]
    n_layers = act_data["n_layers"]

    by_layer = defaultdict(list)
    for layer_idx in range(n_layers):
        depth_pct = nearest_depth_pct(layer_idx, n_layers)
        if depth_pct not in descriptions_by_depth:
            continue
        desc_map = descriptions_by_depth[depth_pct]
        acts = layer_acts[layer_idx]

        for text_idx, text_id in enumerate(ids):
            if text_id in desc_map:
                by_layer[layer_idx].append({
                    "activation": acts[text_idx].float(),
                    "description": desc_map[text_id],
                    "text_id": text_id,
                })

    total = sum(len(v) for v in by_layer.values())
    print(f"  Total: {total} examples across {len(by_layer)} layers")
    return by_layer


def main():
    parser = argparse.ArgumentParser(
        description="Train Universal AR (all layers, truncated + per-layer value_heads)")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--desc-suffix", default="_prediction",
                        help="Description file suffix (e.g. _prediction, _tight, empty for original)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load activations
    print(f"Loading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    n_layers = act_data["n_layers"]
    d_model = act_data["d_model"]
    mse_scale = math.sqrt(d_model)
    print(f"  {n_layers} layers, {act_data['n_texts']} texts, d={d_model}")

    # Load descriptions
    print(f"Loading descriptions (suffix='{args.desc_suffix}')...")
    descriptions = load_descriptions(suffix=args.desc_suffix)

    # Build examples
    print("Building examples by layer...")
    by_layer = build_examples(act_data, descriptions)

    # Split by text ID
    all_text_ids = sorted(set(
        ex["text_id"] for exs in by_layer.values() for ex in exs
    ))
    n_val = max(1, int(len(all_text_ids) * args.val_split))
    rng = np.random.RandomState(42)
    val_ids = set(rng.choice(all_text_ids, n_val, replace=False))

    train_by_layer = {}
    val_by_layer = {}
    for layer_idx, examples in by_layer.items():
        train_by_layer[layer_idx] = [ex for ex in examples if ex["text_id"] not in val_ids]
        val_by_layer[layer_idx] = [ex for ex in examples if ex["text_id"] in val_ids]

    n_train = sum(len(v) for v in train_by_layer.values())
    n_val_total = sum(len(v) for v in val_by_layer.values())
    print(f"  Train: {n_train}, Val: {n_val_total}")

    # Load model
    model_name = MODELS[args.model]
    trust_remote = "phi" not in args.model.lower()
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=trust_remote)

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # Strip norm and lm_head
    inner = backbone.model if hasattr(backbone, "model") else backbone
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            break
    backbone.lm_head = torch.nn.Identity()
    backbone.eval()

    # Create model with per-layer value_heads
    ar_model = DynamicTruncatedAR(backbone, n_layers, d_model)
    ar_model = ar_model.to(device)

    n_trainable = sum(p.numel() for p in ar_model.value_heads.parameters())
    print(f"  {n_layers} value_heads, {n_trainable:,} trainable params")

    # Pre-tokenize datasets
    print("\nPre-tokenizing...")
    train_datasets = {}
    val_datasets = {}
    for layer_idx in sorted(train_by_layer.keys()):
        train_datasets[layer_idx] = LayerARDataset(train_by_layer[layer_idx], tokenizer)
        if val_by_layer[layer_idx]:
            val_datasets[layer_idx] = LayerARDataset(val_by_layer[layer_idx], tokenizer)

    optimizer = torch.optim.AdamW(
        ar_model.value_heads.parameters(), lr=args.lr, weight_decay=0.01)

    # Train
    best_val_cos = 0.0
    Path(args.output).mkdir(parents=True, exist_ok=True)
    layer_order = sorted(train_datasets.keys())

    for epoch in range(args.epochs):
        ar_model.train()
        total_loss = 0
        n_batches = 0

        np.random.shuffle(layer_order)
        for layer_idx in layer_order:
            dataset = train_datasets[layer_idx]
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn, pin_memory=True)

            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_acts = batch["activations"].to(device).float()
                seq_lens = batch["seq_lens"]

                reconstructed = ar_model(input_ids, attention_mask, layer_idx, seq_lens)
                loss = compute_loss(reconstructed, target_acts, mse_scale)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(ar_model.value_heads.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

        train_loss = total_loss / n_batches

        # Validate
        ar_model.eval()
        val_loss = 0
        val_cos = 0
        val_n = 0
        per_layer_cos = {}

        with torch.no_grad():
            for layer_idx in sorted(val_datasets.keys()):
                dataset = val_datasets[layer_idx]
                loader = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate_fn, pin_memory=True)

                layer_cos_sum = 0
                layer_n = 0
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_acts = batch["activations"].to(device).float()
                    seq_lens = batch["seq_lens"]

                    reconstructed = ar_model(input_ids, attention_mask, layer_idx, seq_lens)
                    loss = compute_loss(reconstructed, target_acts, mse_scale)
                    cos = torch.nn.functional.cosine_similarity(
                        reconstructed, target_acts, dim=1).mean()

                    val_loss += loss.item()
                    val_cos += cos.item()
                    val_n += 1
                    layer_cos_sum += cos.item()
                    layer_n += 1

                if layer_n > 0:
                    per_layer_cos[layer_idx] = layer_cos_sum / layer_n

        val_loss /= max(val_n, 1)
        val_cos /= max(val_n, 1)

        sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        cos_str = " ".join(
            f"L{l}={per_layer_cos.get(l, 0):.3f}" for l in sample_layers
        )
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={val_loss:.4f} "
              f"cos={val_cos:.4f} [{cos_str}]")

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            # Save all value_heads
            state = {}
            for k, vh in ar_model.value_heads.items():
                state[f"value_head.{k}.weight"] = vh.weight.data
            save_file(state, str(Path(args.output) / "value_heads.safetensors"))
            print(f"    -> saved (best cos={best_val_cos:.4f})")

    # Save metadata
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "ar",
        "variant": "universal",
        "stage": "sl",
        "d_model": d_model,
        "n_layers": n_layers,
        "extraction": {
            "injection_scale": None,
            "mse_scale": float(mse_scale),
        },
        "prompt_templates": {
            "ar": AR_TEMPLATE,
        },
        "depth_percentages": DEPTH_PCTS,
        "training": {
            "method": "per_layer_value_heads+frozen_backbone",
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": n_train,
            "n_val": n_val_total,
            "best_val_cosine": float(best_val_cos),
            "per_layer_cosine": {int(k): round(v, 4) for k, v in per_layer_cos.items()},
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"\nDone. Best val cosine: {best_val_cos:.4f}")
    print(f"Per-layer cosine:")
    for l in sorted(per_layer_cos.keys()):
        print(f"  Layer {l:2d}: {per_layer_cos[l]:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
