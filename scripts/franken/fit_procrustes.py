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
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from nla_lib import DEPTH_PCTS  # frozen grid — never re-type it


def nearest_layer(pct, n_layers):
    """Invert nla_lib's layer→depth convention (pct ≈ layer*100/n_layers).

    NOT (n_layers-1): nla_lib.nearest_depth_pct divides by n_layers, so
    the inverse must multiply by it — round(71*28/100)=20 and L20 is the
    71% layer; with (n_layers-1) this would be off by one.
    """
    return min(n_layers - 1, max(0, round(pct * n_layers / 100)))


def procrustes(A, B):
    """Orthogonal Procrustes on CENTERED data: R s.t. ||R(A-μA) - (B-μB)||_F
    is minimal. A, B: [n_samples, d_model].

    Centering matters: residual-stream means are huge and shared — fitting
    raw wastes the rotation on aligning means and inflates the cos
    diagnostic (same reason round-trip eval uses centered cosine). The
    full transform is affine: h' = R @ (h - mu_src) + mu_tgt; both means
    ship in the output.

    Statistical floor: a d×d rotation fitted on n < d samples is only
    determined on an n-dim subspace. main() warns; use ≥ d_model matched
    texts (the full corpus, not a 200-text pilot).
    """
    mu_a, mu_b = A.mean(dim=0), B.mean(dim=0)
    Ac, Bc = A - mu_a, B - mu_b
    M = Bc.T @ Ac  # [d, d]
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:  # keep a proper rotation (harmless either way)
        U[:, -1] *= -1
        R = U @ Vh
    residual = torch.norm(Ac @ R.T - Bc).item()
    return R, residual, mu_a, mu_b


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

    if len(shared_ids) < src_d:
        print(f"  ⚠️  n_samples={len(shared_ids)} < d_model={src_d}: the "
              f"rotation is only determined on a {len(shared_ids)}-dim "
              f"subspace — use the full corpus, not a pilot sample",
              flush=True)

    rotations = {}
    residuals = {}
    mus_src = {}
    mus_tgt = {}
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

        R, res, mu_a, mu_b = procrustes(A, B)
        rotations[pct] = R
        residuals[pct] = res
        mus_src[pct] = mu_a
        mus_tgt[pct] = mu_b
        source_layers[pct] = src_layer
        target_layers[pct] = tgt_layer

        # Alignment quality on CENTERED vectors (uncentered cos is
        # inflated by the shared mean — same story as round-trip eval)
        aligned = (A - mu_a) @ R.T
        cos = torch.nn.functional.cosine_similarity(aligned, B - mu_b, dim=1)
        print(f"  {pct:3d}% (src L{src_layer} → tgt L{tgt_layer}): "
              f"residual={res:.1f}, centered cos={cos.mean():.3f}±{cos.std():.3f}",
              flush=True)

    output = {
        "rotations": rotations,
        "residuals": residuals,
        "mu_source": mus_src,
        "mu_target": mus_tgt,
        "transform": "h_aligned = R @ (h - mu_source) + mu_target",
        "source_layers": source_layers,
        "target_layers": target_layers,
        "source_model": src.get("model", src.get("source_model", "unknown")),
        "target_model": tgt.get("model", tgt.get("source_model", "unknown")),
        "d_model": src_d,
        "n_samples": len(shared_ids),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"\nSaved {len(rotations)} rotation matrices → {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
