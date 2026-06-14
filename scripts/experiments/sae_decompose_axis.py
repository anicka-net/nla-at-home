#!/usr/bin/env python3
"""
SAE Decomposition of Extracted Psychological Axes.

Decomposes a unit-normalized direction vector into SAE features by computing
cosine similarity between the direction and each SAE decoder column. Then
performs greedy subspace capture to measure how many SAE features are needed
to explain the direction.

Usage:
  python3 scripts/experiments/sae_decompose_axis.py \
    --direction ~/tone-experiment/results/vedana-vs-rc/llama-8b_vedana_L20_unit.pt \
    --sae-release llama_scope_lxr_32x \
    --sae-id l20 \
    --output data/sae-decomposition/llama-8b/vedana_L20_decomposition.json \
    --top-k 50
"""
import torch
import torch.nn.functional as F
import json
import argparse
import requests
from pathlib import Path

NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature"


def load_sae(release, sae_id):
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    return sae


def decompose_direction(direction, sae, top_k=50):
    W_dec = sae.W_dec.data.float()
    d_vec = direction.float()

    cos_sims = F.cosine_similarity(d_vec.unsqueeze(0), W_dec, dim=1)

    top_pos = cos_sims.topk(top_k)
    top_neg = (-cos_sims).topk(top_k)

    results = []
    for idx, cos in zip(top_pos.indices.tolist(), top_pos.values.tolist()):
        results.append({"feature_id": idx, "cosine": round(cos, 4), "sign": "positive"})
    for idx, cos in zip(top_neg.indices.tolist(), top_neg.values.tolist()):
        results.append({"feature_id": idx, "cosine": round(-cos, 4), "sign": "negative"})

    results.sort(key=lambda x: abs(x["cosine"]), reverse=True)
    return results[:top_k], cos_sims


def greedy_subspace_capture(direction, sae, max_features=100):
    W_dec = sae.W_dec.data.float()
    d_vec = direction.float().clone()
    original_norm_sq = d_vec.norm() ** 2

    captured = []
    residual = d_vec.clone()

    for i in range(max_features):
        projections = residual @ W_dec.T
        best_idx = projections.abs().argmax().item()
        best_decoder = W_dec[best_idx]
        best_decoder_norm = best_decoder / best_decoder.norm()

        proj_scalar = residual @ best_decoder_norm
        residual = residual - proj_scalar * best_decoder_norm

        variance_explained = 1 - (residual.norm() ** 2 / original_norm_sq).item()
        captured.append({
            "step": i + 1,
            "feature_id": best_idx,
            "variance_explained": round(variance_explained, 6),
            "projection": round(proj_scalar.item(), 4),
        })

        if variance_explained > 0.99:
            break

    return captured


def random_direction_baseline(d_model, sae, max_features=100, n_trials=10):
    curves = []
    for _ in range(n_trials):
        rand_dir = torch.randn(d_model)
        rand_dir = rand_dir / rand_dir.norm()
        captured = greedy_subspace_capture(rand_dir, sae, max_features)
        curve = [c["variance_explained"] for c in captured]
        curves.append(curve)

    max_len = max(len(c) for c in curves)
    mean_curve = []
    for i in range(max_len):
        vals = [c[i] for c in curves if i < len(c)]
        mean_curve.append(round(sum(vals) / len(vals), 6))
    return mean_curve


def fetch_neuronpedia_labels(model_id, layer, feature_ids):
    labels = {}
    for fid in feature_ids:
        try:
            url = f"{NEURONPEDIA_API}/{model_id}/{layer}/{fid}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                label = data.get("explanations", [{}])[0].get("description", "")
                labels[fid] = label
            else:
                labels[fid] = f"(HTTP {resp.status_code})"
        except Exception as e:
            labels[fid] = f"(error: {e})"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True, help="Path to unit direction .pt file")
    parser.add_argument("--sae-release", required=True, help="SAELens release name")
    parser.add_argument("--sae-id", required=True, help="SAELens SAE ID within release")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-capture", type=int, default=100)
    parser.add_argument("--neuronpedia-model", default=None,
                        help="Neuronpedia model ID for label lookup (e.g. 'llama-3.1-8b')")
    parser.add_argument("--neuronpedia-layer", type=int, default=None)
    parser.add_argument("--random-baseline-trials", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading direction from {args.direction}...")
    direction = torch.load(args.direction, weights_only=True)
    print(f"  Shape: {direction.shape}, norm: {direction.norm():.4f}")

    print(f"Loading SAE: {args.sae_release} / {args.sae_id}...")
    sae = load_sae(args.sae_release, args.sae_id)
    W_dec = sae.W_dec.data
    print(f"  SAE: {W_dec.shape[0]} features, d_model={W_dec.shape[1]}")

    assert direction.shape[0] == W_dec.shape[1], \
        f"Direction dim {direction.shape[0]} != SAE d_model {W_dec.shape[1]}"

    print(f"\nDecomposing direction into top-{args.top_k} SAE features...")
    top_features, all_cosines = decompose_direction(direction, sae, args.top_k)

    cos_stats = {
        "max": round(all_cosines.max().item(), 4),
        "min": round(all_cosines.min().item(), 4),
        "mean_abs": round(all_cosines.abs().mean().item(), 4),
        "std": round(all_cosines.std().item(), 4),
        "n_above_0.1": int((all_cosines.abs() > 0.1).sum().item()),
        "n_above_0.2": int((all_cosines.abs() > 0.2).sum().item()),
        "n_above_0.3": int((all_cosines.abs() > 0.3).sum().item()),
    }
    print(f"  Cosine stats: max={cos_stats['max']}, n>0.1={cos_stats['n_above_0.1']}, "
          f"n>0.2={cos_stats['n_above_0.2']}, n>0.3={cos_stats['n_above_0.3']}")
    print(f"  Top 5: {[(f['feature_id'], f['cosine']) for f in top_features[:5]]}")

    print(f"\nGreedy subspace capture (up to {args.max_capture} features)...")
    capture = greedy_subspace_capture(direction, sae, args.max_capture)
    print(f"  Features to 90% variance: "
          f"{next((c['step'] for c in capture if c['variance_explained'] >= 0.9), '>'+str(args.max_capture))}")
    print(f"  Features to 95% variance: "
          f"{next((c['step'] for c in capture if c['variance_explained'] >= 0.95), '>'+str(args.max_capture))}")

    print(f"\nRandom direction baseline ({args.random_baseline_trials} trials)...")
    baseline = random_direction_baseline(direction.shape[0], sae, args.max_capture,
                                         args.random_baseline_trials)

    labels = {}
    if args.neuronpedia_model and args.neuronpedia_layer is not None:
        print(f"\nFetching Neuronpedia labels for top 20 features...")
        feature_ids = [f["feature_id"] for f in top_features[:20]]
        labels = fetch_neuronpedia_labels(args.neuronpedia_model,
                                          args.neuronpedia_layer, feature_ids)
        for f in top_features[:20]:
            f["label"] = labels.get(f["feature_id"], "")
            print(f"  #{f['feature_id']} (cos={f['cosine']:.3f}): {f['label'][:60]}")

    output = {
        "direction_file": str(args.direction),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "d_model": int(direction.shape[0]),
        "n_sae_features": int(W_dec.shape[0]),
        "cosine_statistics": cos_stats,
        "top_features": top_features,
        "subspace_capture": capture,
        "random_baseline": baseline,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
