#!/usr/bin/env python3
"""
Extract vedana (valence) direction from contrastive prompts.

Direction = mean(pleasant activations) - mean(unpleasant activations)
at each layer's last-token residual stream position.

Saves unit-normalized direction at the peak layer.

Usage:
  python3 scripts/experiments/extract_vedana_direction.py \
    --model google/gemma-3-1b-it \
    --model-short gemma3-1b \
    --stimuli prompts/vedana_prompts_n50.yaml \
    --output-dir data/directions/
"""
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_all_layers(model, tokenizer, text, n_layers, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    layer_acts = {}

    handles = []
    for layer_idx in range(n_layers):
        def make_hook(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                layer_acts[idx] = h[0, -1, :].detach().cpu().float()
            return hook
        handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return layer_acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading stimuli from {args.stimuli}...")
    data = yaml.safe_load(open(args.stimuli))
    pleasant = [p['text'] for p in data['vedana']['pleasant']]
    unpleasant = [p['text'] for p in data['vedana']['unpleasant']]
    neutral = [p['text'] for p in data['vedana'].get('neutral', [])]
    print(f"  {len(pleasant)} pleasant, {len(unpleasant)} unpleasant, {len(neutral)} neutral")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d_model={d_model}")

    def extract_mean(texts, label):
        all_acts = {l: [] for l in range(n_layers)}
        for i, text in enumerate(texts):
            acts = extract_all_layers(model, tokenizer, text, n_layers, device)
            for l in range(n_layers):
                all_acts[l].append(acts[l])
            if (i + 1) % 10 == 0:
                print(f"  {label}: {i+1}/{len(texts)}")
        means = {}
        for l in range(n_layers):
            means[l] = torch.stack(all_acts[l]).mean(dim=0)
        return means

    print("Extracting pleasant activations...")
    pleasant_means = extract_mean(pleasant, "pleasant")
    print("Extracting unpleasant activations...")
    unpleasant_means = extract_mean(unpleasant, "unpleasant")

    neutral_means = None
    if neutral:
        print("Extracting neutral activations...")
        neutral_means = extract_mean(neutral, "neutral")

    print("\nComputing vedana direction per layer...")
    directions = {}
    norms = {}
    for l in range(n_layers):
        d = pleasant_means[l] - unpleasant_means[l]
        norms[l] = d.norm().item()
        directions[l] = d / d.norm()

    print("\nLayer norms (direction strength):")
    peak_layer = max(norms, key=norms.get)
    for l in range(n_layers):
        marker = " <-- PEAK" if l == peak_layer else ""
        print(f"  L{l:2d}: norm={norms[l]:.4f}{marker}")

    if neutral_means:
        print("\nSeparation (pleasant vs unpleasant, projected onto direction):")
        for l in [peak_layer, n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
            d_hat = directions[l]
            p_proj = sum(float(pleasant_means[l] @ d_hat) for _ in [0])
            u_proj = sum(float(unpleasant_means[l] @ d_hat) for _ in [0])
            n_proj = sum(float(neutral_means[l] @ d_hat) for _ in [0])
            print(f"  L{l}: pleasant={p_proj:+.3f}, neutral={n_proj:+.3f}, unpleasant={u_proj:+.3f}")

    unit_dir = directions[peak_layer]
    out_path = out_dir / f"{args.model_short}_vedana_L{peak_layer}_unit.pt"
    torch.save(unit_dir, out_path)
    print(f"\nSaved peak direction to {out_path}")
    print(f"  Layer {peak_layer}, norm of raw direction: {norms[peak_layer]:.4f}")

    all_layers_path = out_dir / f"{args.model_short}_vedana_all_layers.pt"
    torch.save({
        "directions": directions,
        "norms": norms,
        "peak_layer": peak_layer,
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
    }, all_layers_path)
    print(f"Saved all-layer directions to {all_layers_path}")


if __name__ == "__main__":
    main()
