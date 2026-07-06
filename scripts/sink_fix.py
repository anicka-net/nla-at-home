#!/usr/bin/env python3
"""Massive-activation sink fix for Gemma-family models (opt-in preprocessing).

Gemma residual streams carry a massive-activation outlier (gemma3-1b: dim
#1038 = 97% of energy at L13; guardian-gemma sibling suspected). Raw
normalize-to-scale injection then pins every injected vector ~parallel
(mean pairwise cosine 0.99) and the verbalizer must read a ~1% angular
tilt. Fix per the 2026-07-02 analysis (~/playground/seventh/
gemma-outlier-geometry.md): CENTER + DROP TOP PC(s) before any
normalize / inject step. Semantic content survives (cos-NC 0.485 raw ->
0.491 fixed); the broken cosine geometry is repaired (0.990 -> 0.022).

This module deliberately lives OUTSIDE nla_lib: normalize_activation's
semantics are a frozen interface of shipped adapters. The fix composes
in FRONT of it: h -> apply(params, layer, h) -> normalize_activation(...).

An adapter trained with the fix REQUIRES the same transform at inference —
the params ship as `sink_fix.pt` next to the adapter weights. Fitting is
per-layer on the full activation corpus (unsupervised statistics, same
policy as AR-side --pca-whiten).

Mechanics note: centering removes the sink's CONSTANT component; the PC
drop removes its FLUCTUATING component. Both are needed — the real sink
has huge mean AND huge variance. On data where the top centered-PC is
semantic rather than a sink (e.g. Qwen), dropping it would DELETE signal:
check `report()` (top-PC energy share) before enabling on a new model.
"""

import torch


def fit(layer_acts, drop_top_pc=1, sample_cap=4096, seed=0):
    """layer_acts: dict/list of per-layer [N, d] tensors.
    Returns params dict: {layer: {"mu": [d], "pcs": [k, d]}} + config."""
    params = {"drop_top_pc": int(drop_top_pc), "layers": {}}
    g = torch.Generator().manual_seed(seed)
    n_layers = len(layer_acts)
    for layer in range(n_layers):
        X = layer_acts[layer].float()
        if X.shape[0] > sample_cap:
            idx = torch.randperm(X.shape[0], generator=g)[:sample_cap]
            Xs = X[idx]
        else:
            Xs = X
        mu = Xs.mean(dim=0)
        Xc = Xs - mu
        # top singular vectors of the centered sample = PCs
        _, S, V = torch.linalg.svd(Xc, full_matrices=False)
        pcs = V[:drop_top_pc].contiguous()  # [k, d]
        top_energy = (S[0] ** 2 / (S ** 2).sum()).item()
        params["layers"][layer] = {"mu": mu, "pcs": pcs,
                                   "top_pc_energy": top_energy}
    return params


def apply(params, layer, h):
    """Center + project out the stored top PC(s). h: [d] or [N, d]."""
    p = params["layers"][int(layer)]
    mu = p["mu"].to(h.device, h.dtype)
    pcs = p["pcs"].to(h.device, h.dtype)
    hc = h - mu
    coeffs = hc @ pcs.T                     # [..., k]
    return hc - coeffs @ pcs


def apply_all(params, layer_acts):
    """Transform every layer matrix in place; returns the container."""
    for layer in range(len(layer_acts)):
        layer_acts[layer] = apply(params, layer, layer_acts[layer].float())
    return layer_acts


def save(params, path):
    torch.save(params, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def report(params):
    lines = []
    for layer, p in sorted(params["layers"].items()):
        lines.append(f"  L{layer:2d}: top-PC energy {p['top_pc_energy']:.1%}")
    return "\n".join(lines)
