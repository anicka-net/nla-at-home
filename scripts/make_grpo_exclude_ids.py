#!/usr/bin/env python3
"""Reconstruct a Universal-AV run's held-out text ids for GRPO exclusion.

train_universal_av.py holds out a validation split by TEXT ID:
    all_text_ids = sorted(set(ex['text_id'] for ex in examples))
    n_val = max(1, int(len(all_text_ids) * val_split))   # val_split default 0.1
    val  = np.random.RandomState(42).choice(all_text_ids, n_val, replace=False)
where `examples` comes from build_examples(act, load_descriptions(...)).

GRPO continues training the AV, so it must not sample those texts or the
round-trip eval leaks. New AV runs now write val_text_ids.json into their
output dir directly; use THIS helper to regenerate the same set for an AV
trained before that sidecar existed.

To be PROVABLY identical (not just probably), this calls the AV's own
load_descriptions + build_examples — so the id universe matches exactly,
including --strict / --mix fallback behavior. Pass the SAME --desc-suffix,
--strict, --mix, and --val-split the original AV run used.

It also prints the reproduced n_train/n_val EXAMPLE counts; compare them to
the adapter's nla_meta.yaml training.n_train / n_val for a hard check that
the universe and seed match what actually trained.

    python3 scripts/make_grpo_exclude_ids.py \
        --activations corpus/activations/phi4_all_layers.pt \
        --desc-suffix _v2 --strict \
        --output output/nla-phi4-universal-av-v2/val_text_ids.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_universal_av import load_descriptions, build_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--desc-suffix", default="",
                    help="same as the AV run, e.g. _v2")
    ap.add_argument("--strict", action="store_true",
                    help="same as the AV run (exact-suffix files, no fallback)")
    ap.add_argument("--mix", action="store_true", help="same as the AV run")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    act = torch.load(args.activations, weights_only=True, map_location="cpu")
    descriptions = load_descriptions(suffix=args.desc_suffix, strict=args.strict,
                                     mix=args.mix)
    examples = build_examples(act, descriptions)

    # Identical to train_universal_av.py's split.
    all_text_ids = sorted(set(ex["text_id"] for ex in examples))
    n_val = max(1, int(len(all_text_ids) * args.val_split))
    rng = np.random.RandomState(args.seed)
    val_ids = sorted(rng.choice(all_text_ids, n_val, replace=False).tolist())

    # Hard check: reproduced example counts must equal the adapter's nla_meta.
    val_set = set(val_ids)
    n_val_ex = sum(1 for ex in examples if ex["text_id"] in val_set)
    n_train_ex = len(examples) - n_val_ex

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(val_ids, open(args.output, "w"))
    print(f"texts: {len(all_text_ids)} | val held out: {len(val_ids)}")
    print(f"reproduced example split: n_train={n_train_ex} n_val={n_val_ex} "
          f"(must match nla_meta.yaml training.n_train / n_val)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
