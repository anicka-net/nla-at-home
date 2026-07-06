#!/usr/bin/env python3
"""Fit Procrustes rotation between ds4 and Llama activation spaces.

Given matched activations from ds4 (source) and Llama 3.1 8B (target)
at corresponding depth percentages, compute the orthogonal rotation R
that minimizes ||R·A_source - A_target||_F.

This is a closed-form solution (no training):
  R = V @ U^T  where  U S V^T = SVD(A_target^T @ A_source)

Usage:
  python scripts/franken/fit_procrustes.py \
    --source corpus/activations/ds4-flash_all_layers.pt \
    --target corpus/activations/llama-8b_all_layers.pt \
    --output corpus/activations/ds4_to_llama_procrustes.pt

The output contains:
  {rotations: {layer_pct: R_matrix}, residuals: {layer_pct: float},
   source_layers: [...], target_layers: [...]}

Apply to ds4 activations before injection into Llama:
  h_aligned = h_ds4 @ R.T
"""

import argparse
from pathlib import Path

import torch

# From nla_lib: depth percentage grid
DEPTH_PCTS = [3, 10, 18, 25, 33, 40, 50, 60, 67, 75, 82, 90, 97]


def nearest_layer(pct, n_layers):
    """Convert depth percentage to layer index."""
    return round(pct / 100 * (n_layers - 1))


def procrustes(A, B):
    """Orthogonal Procrustes: find R s.t. ||R @ A - B||_F is minimized.
    A, B: [n_samples, d_model]. Returns R: [d_model, d_model]."""
    M = B.T @ A  # [d, d]
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    # Ensure proper rotation (det = +1), not reflection
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh
    residual = torch.norm(A @ R.T - B).item()
    return R, residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="ds4 activations .pt file")
    ap.add_argument("--target", required=True,
                    help="Llama activations .pt file")
    ap.add_argument("--output", required=True,
                    help="Output .pt with rotation matrices")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap number of samples for fitting")
    args = ap.parse_args()

    print("Loading activations...", flush=True)
    src = torch.load(args.source, weights_only=True, map_location="cpu")
    tgt = torch.load(args.target, weights_only=True, map_location="cpu")

    src_n_layers = src["n_layers"]
    tgt_n_layers = tgt["n_layers"]
    src_d = src["d_model"]
    tgt_d = tgt["d_model"]

    assert src_d == tgt_d, (
        f"Dimension mismatch: source d={src_d}, target d={tgt_d}. "
        f"Procrustes requires same dimensions. Use a projection instead.")

    # Match texts by id
    src_id2idx = {tid: i for i, tid in enumerate(src["ids"])}
    tgt_id2idx = {tid: i for i, tid in enumerate(tgt["ids"])}
    shared_ids = [tid for tid in src["ids"] if tid in tgt_id2idx]
    print(f"  {len(shared_ids)} shared texts "
          f"(source={len(src['ids'])}, target={len(tgt['ids'])})", flush=True)

    if args.max_samples and len(shared_ids) > args.max_samples:
        shared_ids = shared_ids[:args.max_samples]
        print(f"  Capped to {len(shared_ids)} samples", flush=True)

    src_idxs = [src_id2idx[tid] for tid in shared_ids]
    tgt_idxs = [tgt_id2idx[tid] for tid in shared_ids]

    rotations = {}
    residuals = {}
    source_layers = {}
    target_layers = {}

    for pct in DEPTH_PCTS:
        src_layer = nearest_layer(pct, src_n_layers)
        tgt_layer = nearest_layer(pct, tgt_n_layers)

        if src_layer not in src["activations"]:
            print(f"  {pct}%: source layer {src_layer} not captured, skip",
                  flush=True)
            continue
        if tgt_layer not in tgt["activations"]:
            print(f"  {pct}%: target layer {tgt_layer} not captured, skip",
                  flush=True)
            continue

        A = src["activations"][src_layer][src_idxs].float()
        B = tgt["activations"][tgt_layer][tgt_idxs].float()

        R, res = procrustes(A, B)
        rotations[pct] = R
        residuals[pct] = res
        source_layers[pct] = src_layer
        target_layers[pct] = tgt_layer

        # Measure alignment quality
        aligned = A @ R.T
        cos = torch.nn.functional.cosine_similarity(aligned, B, dim=1)
        print(f"  {pct:3d}% (src L{src_layer} → tgt L{tgt_layer}): "
              f"residual={res:.1f}, cos={cos.mean():.3f}±{cos.std():.3f}",
              flush=True)

    output = {
        "rotations": rotations,
        "residuals": residuals,
        "source_layers": source_layers,
        "target_layers": target_layers,
        "source_model": src.get("source_model", "unknown"),
        "target_model": tgt.get("source_model", "unknown"),
        "d_model": src_d,
        "n_samples": len(shared_ids),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"\nSaved {len(rotations)} rotation matrices → {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
