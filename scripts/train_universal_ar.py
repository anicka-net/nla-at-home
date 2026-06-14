#!/usr/bin/env python3
"""
Train a Universal Activation Reconstructor — one adapter for ALL layers.

Reverse of the Universal AV: takes a description + depth tag, reconstructs
the original activation at that layer. Used to verify that descriptions
carry real information about the activation geometry.

Groups training by layer so each batch uses a single extraction hook.

Usage:
  python3 scripts/train_universal_ar.py \
    --model gemma3-1b \
    --activations corpus/activations/gemma3-1b_all_layers.pt \
    --output output/nla-gemma3-1b-universal-ar \
    --epochs 5 --lr 7e-5
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
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

MODELS = {
    "gemma3-1b": "google/gemma-3-1b-it",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

INJECTION_CHARS = {
    "gemma3-1b": "⎝",
    "qwen25-7b": "㈎",
    "qwen3-4b": "㈎",
}

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]


def make_ar_template(depth_pct, injection_char):
    return (
        f"Summary of the following text from depth {depth_pct}%: "
        f"<text>{{explanation}}</text> <summary>{injection_char}"
    )


def nearest_depth_pct(layer, n_layers):
    depth = layer * 100 / n_layers
    return min(DEPTH_PCTS, key=lambda p: abs(p - depth))


def load_descriptions(suffix="_tokenpred_gpt4o"):
    descs = {}
    for pct in DEPTH_PCTS:
        candidates = [
            GENERATED_DIR / f"descriptions_L{pct}pct{suffix}.json",
            GENERATED_DIR / f"descriptions_L{pct}pct_merged.json",
            GENERATED_DIR / f"descriptions_L{pct}pct.json",
        ]
        path = None
        for c in candidates:
            if c.exists():
                path = c
                break
        if path is None:
            print(f"  L{pct}%: NO FILE FOUND")
            continue
        data = json.loads(path.read_text())
        descs[pct] = {d["id"]: d["description"] for d in data}
        print(f"  L{pct}%: {len(descs[pct])} from {path.name}")
    return descs


class LayerARDataset(Dataset):
    """Dataset for a single layer's examples — pre-tokenized."""
    def __init__(self, examples, tokenizer, injection_char, max_length=512):
        inject_id = tokenizer.encode(injection_char, add_special_tokens=False)
        assert len(inject_id) == 1
        inject_id = inject_id[0]

        self.items = []
        for ex in examples:
            template = make_ar_template(ex["depth_pct"], injection_char)
            prompt = template.replace("{explanation}", ex["description"])
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            inject_pos = None
            for i, t in enumerate(tokens):
                if t == inject_id:
                    inject_pos = i
                    break
            if inject_pos is None:
                inject_pos = len(tokens) - 1

            self.items.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "activation": ex["activation"],
                "inject_pos": inject_pos,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    inject_positions = torch.tensor([b["inject_pos"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "activations": activations,
        "inject_positions": inject_positions,
    }


def compute_pca_transforms(act_data, min_layer=0, drop_top=1):
    layer_acts = act_data["activations"]
    n_layers = act_data["n_layers"]
    transforms = {}
    for layer_idx in range(min_layer, n_layers):
        acts = layer_acts[layer_idx].float()
        mean = acts.mean(dim=0)
        centered = acts - mean
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        eigenvalues = S**2 / (acts.shape[0] - 1)
        transforms[layer_idx] = {
            "mean": mean,
            "components": Vh,
            "eigenvalues": eigenvalues,
        }
        total_var = eigenvalues.sum()
        top_var = eigenvalues[:drop_top].sum()
        print(f"  L{layer_idx}: top-{drop_top} PCs explain {100*top_var/total_var:.1f}% variance")
    return transforms


def pca_whiten_vectors(vectors, pca_transform, drop_top):
    mean = pca_transform["mean"].to(vectors.device)
    components = pca_transform["components"].to(vectors.device)
    eigenvalues = pca_transform["eigenvalues"].to(vectors.device)
    centered = vectors - mean
    projected = centered @ components.T
    projected = projected[:, drop_top:]
    eig = eigenvalues[drop_top:]
    scale = eig.sqrt().clamp_min(1e-12)
    return projected / scale.unsqueeze(0)


def build_examples(act_data, descriptions_by_depth, mean_subtract=False,
                   min_layer=0):
    layer_acts = act_data["activations"]
    ids = act_data["ids"]
    n_layers = act_data["n_layers"]

    layer_means = {}
    if mean_subtract:
        for layer_idx in range(min_layer, n_layers):
            layer_means[layer_idx] = layer_acts[layer_idx].float().mean(dim=0)

    by_layer = defaultdict(list)
    for layer_idx in range(min_layer, n_layers):
        depth_pct = nearest_depth_pct(layer_idx, n_layers)
        desc_map = descriptions_by_depth.get(depth_pct, {})
        acts = layer_acts[layer_idx]

        for text_idx, text_id in enumerate(ids):
            if text_id in desc_map:
                act = acts[text_idx]
                if mean_subtract:
                    act = act.float() - layer_means[layer_idx]
                by_layer[layer_idx].append({
                    "activation": act,
                    "description": desc_map[text_id],
                    "depth_pct": depth_pct,
                    "layer": layer_idx,
                    "text_id": text_id,
                })

    total = sum(len(v) for v in by_layer.values())
    print(f"Total: {total} examples across {len(by_layer)} layers")
    return by_layer


def get_blocks(model):
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    return inner.layers


def train(model, tokenizer, train_by_layer, val_by_layer, injection_char,
          args, device, pca_transforms=None):
    blocks = get_blocks(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    best_val_loss = float("inf")
    best_val_cos = 0.0
    mse_scale = args.mse_scale
    drop_top = args.pca_drop_top if pca_transforms else 0

    layer_order = sorted(train_by_layer.keys())

    print("  Pre-tokenizing training datasets...")
    train_datasets = {}
    for layer_idx in layer_order:
        train_datasets[layer_idx] = LayerARDataset(
            train_by_layer[layer_idx], tokenizer, injection_char)
    print(f"  Pre-tokenizing validation datasets...")
    val_datasets = {}
    for layer_idx in sorted(val_by_layer.keys()):
        if val_by_layer[layer_idx]:
            val_datasets[layer_idx] = LayerARDataset(
                val_by_layer[layer_idx], tokenizer, injection_char)
    print(f"  Pre-tokenized {sum(len(d) for d in train_datasets.values())} train + {sum(len(d) for d in val_datasets.values())} val examples")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        np.random.shuffle(layer_order)
        for layer_idx in layer_order:
            dataset = train_datasets[layer_idx]
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn,
                                num_workers=2, pin_memory=True)

            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_acts = batch["activations"].to(device).float()
                inject_positions = batch["inject_positions"]

                layer_outputs = {}
                def make_hook(lidx):
                    def hook(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        layer_outputs[lidx] = h
                    return hook

                handle = blocks[layer_idx].register_forward_hook(
                    make_hook(layer_idx))
                model(input_ids=input_ids, attention_mask=attention_mask)
                handle.remove()

                hidden = layer_outputs[layer_idx]
                reconstructed = torch.stack([
                    hidden[i, inject_positions[i]].float()
                    for i in range(len(inject_positions))
                ])

                if pca_transforms and layer_idx in pca_transforms:
                    rec_w = pca_whiten_vectors(reconstructed, pca_transforms[layer_idx], drop_top)
                    tgt_w = pca_whiten_vectors(target_acts, pca_transforms[layer_idx], drop_top)
                    mse_loss = torch.nn.functional.mse_loss(rec_w * mse_scale, tgt_w * mse_scale)
                elif args.normalize_mse:
                    rec_w = reconstructed / reconstructed.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
                    tgt_w = target_acts / target_acts.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
                    mse_loss = torch.nn.functional.mse_loss(rec_w, tgt_w)
                    rec_w, tgt_w = reconstructed, target_acts
                else:
                    mse_loss = torch.nn.functional.mse_loss(
                        reconstructed * mse_scale, target_acts * mse_scale)
                    rec_w, tgt_w = reconstructed, target_acts

                if reconstructed.shape[0] > 1:
                    rec_norm = torch.nn.functional.normalize(rec_w, dim=1)
                    tgt_norm = torch.nn.functional.normalize(tgt_w.float(), dim=1)
                    sim_matrix = rec_norm @ tgt_norm.T
                    labels = torch.arange(sim_matrix.shape[0], device=device)
                    contrastive_loss = torch.nn.functional.cross_entropy(
                        sim_matrix * args.contrastive_temp, labels)
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)

                loss = mse_loss + args.contrastive_weight * contrastive_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

        train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0
        val_cos = 0
        val_batches = 0
        per_layer_cos = {}

        with torch.no_grad():
            for layer_idx in sorted(val_datasets.keys()):
                dataset = val_datasets[layer_idx]
                loader = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate_fn,
                                    num_workers=2, pin_memory=True)

                layer_cos_sum = 0
                layer_n = 0
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_acts = batch["activations"].to(device).float()
                    inject_positions = batch["inject_positions"]

                    layer_outputs = {}
                    handle = blocks[layer_idx].register_forward_hook(
                        make_hook(layer_idx))
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    handle.remove()

                    hidden = layer_outputs[layer_idx]
                    reconstructed = torch.stack([
                        hidden[i, inject_positions[i]].float()
                        for i in range(len(inject_positions))
                    ])

                    if pca_transforms and layer_idx in pca_transforms:
                        rec_w = pca_whiten_vectors(reconstructed, pca_transforms[layer_idx], drop_top)
                        tgt_w = pca_whiten_vectors(target_acts, pca_transforms[layer_idx], drop_top)
                        loss = torch.nn.functional.mse_loss(rec_w * mse_scale, tgt_w * mse_scale)
                    elif args.normalize_mse:
                        rec_w = reconstructed / reconstructed.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
                        tgt_w = target_acts / target_acts.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
                        loss = torch.nn.functional.mse_loss(rec_w, tgt_w)
                        rec_w, tgt_w = reconstructed, target_acts
                    else:
                        loss = torch.nn.functional.mse_loss(
                            reconstructed * mse_scale, target_acts * mse_scale)
                        rec_w, tgt_w = reconstructed, target_acts

                    cos = torch.nn.functional.cosine_similarity(
                        rec_w, tgt_w, dim=1).mean()

                    if reconstructed.shape[0] > 1:
                        rn = torch.nn.functional.normalize(rec_w, dim=1)
                        tn = torch.nn.functional.normalize(tgt_w.float(), dim=1)
                        sim = rn @ tn.T
                        correct = (sim.argmax(dim=1) == torch.arange(sim.shape[0], device=device)).float().mean()
                    else:
                        correct = torch.tensor(1.0)

                    val_loss += loss.item()
                    val_cos += cos.item()
                    val_batches += 1
                    layer_cos_sum += cos.item()
                    layer_n += 1

                if layer_n > 0:
                    per_layer_cos[layer_idx] = layer_cos_sum / layer_n

        val_loss = val_loss / val_batches
        val_cos = val_cos / val_batches

        sample_layers = [0, len(blocks)//4, len(blocks)//2, 3*len(blocks)//4, len(blocks)-1]
        cos_str = " ".join(
            f"L{l}={per_layer_cos.get(l, 0):.3f}" for l in sample_layers
        )
        print(f"  Epoch {epoch+1}/{args.epochs}: mse={val_loss:.4f} cos={val_cos:.4f} [{cos_str}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cos = val_cos
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best mse={best_val_loss:.4f} cos={best_val_cos:.4f})")

    return best_val_loss, best_val_cos, per_layer_cos


def main():
    parser = argparse.ArgumentParser(description="Train Universal AR (all layers)")
    parser.add_argument("--model", default="gemma3-1b", choices=list(MODELS.keys()))
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--mse-scale", type=float, default=59.87)
    parser.add_argument("--contrastive-weight", type=float, default=1.0,
                        help="Weight for InfoNCE contrastive loss (0=MSE only)")
    parser.add_argument("--contrastive-temp", type=float, default=20.0,
                        help="Temperature for contrastive similarity matrix")
    parser.add_argument("--normalize-mse", action="store_true",
                        help="L2-normalize both vectors to sqrt(d) before MSE (Anthropic style)")
    parser.add_argument("--mean-subtract", action="store_true",
                        help="Subtract per-layer mean from target activations")
    parser.add_argument("--pca-whiten", action="store_true",
                        help="PCA-whiten loss: project into PCA space, drop top PCs, equalize variance")
    parser.add_argument("--pca-drop-top", type=int, default=1,
                        help="Number of top PCs to drop (layer-generic structure)")
    parser.add_argument("--min-layer", type=int, default=0,
                        help="Skip layers below this index (late-layers-only training)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    injection_char = INJECTION_CHARS.get(args.model)
    if injection_char is None:
        raise ValueError(f"No injection char for {args.model}")

    print("Loading descriptions...")
    descriptions = load_descriptions()

    print(f"\nLoading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=True)
    n_layers = act_data["n_layers"]
    d_model = act_data["d_model"]
    print(f"  {n_layers} layers, {act_data['n_texts']} texts, d={d_model}")

    pca_transforms = None
    if args.pca_whiten:
        print(f"\nComputing PCA transforms (drop top {args.pca_drop_top}, min layer {args.min_layer})...")
        pca_transforms = compute_pca_transforms(act_data, min_layer=args.min_layer,
                                                 drop_top=args.pca_drop_top)

    print("\nBuilding examples by layer...")
    by_layer = build_examples(act_data, descriptions, mean_subtract=args.mean_subtract,
                              min_layer=args.min_layer)

    # Save layer means for reconstruction at GRPO time
    if args.mean_subtract:
        layer_means = {}
        for layer_idx in range(args.min_layer, n_layers):
            layer_means[layer_idx] = act_data["activations"][layer_idx].float().mean(dim=0)
        torch.save(layer_means, Path(args.output) / "layer_means.pt")
        print(f"  Saved layer means to {args.output}/layer_means.pt")

    # Split by TEXT ID to prevent leakage across layers
    all_text_ids = sorted(set(
        ex["text_id"] for exs in by_layer.values() for ex in exs
    ))
    n_val_texts = max(1, int(len(all_text_ids) * args.val_split))
    rng = np.random.RandomState(42)
    val_text_ids = set(rng.choice(all_text_ids, n_val_texts, replace=False))

    train_by_layer = {}
    val_by_layer = {}
    for layer_idx, examples in by_layer.items():
        train_by_layer[layer_idx] = [ex for ex in examples if ex["text_id"] not in val_text_ids]
        val_by_layer[layer_idx] = [ex for ex in examples if ex["text_id"] in val_text_ids]

    n_train = sum(len(v) for v in train_by_layer.values())
    n_val = sum(len(v) for v in val_by_layer.values())
    print(f"  Train: {n_train}, Val: {n_val} ({n_val_texts} held-out texts)")

    model_name = MODELS[args.model]
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

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

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if pca_transforms:
        pca_save = {k: {kk: vv.cpu() for kk, vv in v.items()}
                    for k, v in pca_transforms.items()}
        torch.save(pca_save, Path(args.output) / "pca_transforms.pt")
        print(f"  Saved PCA transforms to {args.output}/pca_transforms.pt")

    print(f"\nTraining AR for {args.epochs} epochs...")
    best_mse, best_cos, per_layer_cos = train(
        model, tokenizer, train_by_layer, val_by_layer,
        injection_char, args, device, pca_transforms=pca_transforms)

    print(f"\nBest: mse={best_mse:.4f} cos={best_cos:.4f}")

    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "ar",
        "variant": "universal",
        "stage": "sl",
        "d_model": d_model,
        "n_layers": n_layers,
        "extraction": {"mse_scale": args.mse_scale},
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(tokenizer.encode(
                injection_char, add_special_tokens=False)[0]),
        },
        "depth_percentages": DEPTH_PCTS,
        "training": {
            "method": "lora_sl",
            "lora_r": args.lora_r,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": n_train,
            "n_val": n_val,
            "best_val_mse": float(best_mse),
            "best_val_cosine": float(best_cos),
            "per_layer_cosine": {int(k): round(v, 4) for k, v in per_layer_cos.items()},
            "mse_scale": args.mse_scale,
            "mean_subtract": args.mean_subtract,
            "pca_whiten": args.pca_whiten,
            "pca_drop_top": args.pca_drop_top if args.pca_whiten else 0,
            "min_layer": args.min_layer,
            "contrastive_weight": args.contrastive_weight,
            "normalize_mse": args.normalize_mse,
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
