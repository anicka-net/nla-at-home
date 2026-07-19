#!/usr/bin/env python3
"""Diagnostic: do unanswerability MECHANISMS have separate geometry?

Captures the 16 pair activations once, then per layer compares paired-delta
coherence WITHIN subcategories (future / private / other) vs ACROSS them,
plus leave-one-out cosine (does a subcategory direction generalize to a
held-out pair of the same mechanism?).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from acs_self_report_probe import capture
from avyakata_probe import PAIRS

# pair index -> mechanism of undeterminability
SUBCATS = {
    "private": [0, 5, 7],   # thoughts, dream-heard melody, witness dream
    "future": [1, 2, 3],    # ever-have readers, will crumble, next roll
    "other": [4, 6],        # lost-past first tree, beyond-observation space
}

OUT = Path(__file__).with_name("subcat_diag_results.json")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def main():
    deltas = []
    for determinable, undeterminable in PAIRS:
        deltas.append(unit(capture(determinable) - capture(undeterminable)))
    deltas = np.stack(deltas)          # (8, 32, 4096)
    n_layers = deltas.shape[1]

    def coh(indices, layer):
        vs = unit(deltas[indices, layer, :])
        c = vs @ vs.T
        iu = np.triu_indices(len(indices), k=1)
        return float(c[iu].mean())

    report = []
    for layer in range(n_layers):
        within_private = coh(SUBCATS["private"], layer)
        within_future = coh(SUBCATS["future"], layer)
        # cross: mean cosine between private-deltas and future-deltas
        a = unit(deltas[SUBCATS["private"], layer, :])
        b = unit(deltas[SUBCATS["future"], layer, :])
        cross = float((a @ b.T).mean())
        # leave-one-out within future (strongest candidate subcategory)
        loo = []
        for hold in SUBCATS["future"]:
            rest = [i for i in SUBCATS["future"] if i != hold]
            centroid = unit(deltas[rest, layer, :].mean(0))
            loo.append(float(np.dot(unit(deltas[hold, layer, :]), centroid)))
        report.append({
            "layer": layer,
            "within_private": within_private,
            "within_future": within_future,
            "cross_private_future": cross,
            "loo_future_mean": float(np.mean(loo)),
        })

    OUT.write_text(json.dumps(report, indent=2))
    print(f"{'L':>3} {'w-priv':>7} {'w-fut':>7} {'cross':>7} {'LOO-fut':>8}")
    for row in report:
        print(f"{row['layer']:>3} {row['within_private']:>+7.2f} "
              f"{row['within_future']:>+7.2f} "
              f"{row['cross_private_future']:>+7.2f} "
              f"{row['loo_future_mean']:>+8.2f}")
    best = max(report, key=lambda r: r["within_future"])
    print("\nPeak within-future coherence:", best)


if __name__ == "__main__":
    main()
