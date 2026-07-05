#!/usr/bin/env python3
"""Split the corpus-v2 master description file into per-depth files for the AV trainer.

train_universal_av.py (with --desc-suffix _v2 --strict) reads one file per
depth percentage: corpus/generated/descriptions_L{pct}pct_v2.json, each a list
of {id, description}. The master descriptions_phi4_tokenpred_gpt4o.json holds
all of them keyed by Phi-4 extraction layer.

The seven extraction layers map onto the AV trainer's 13-depth grid
(4,10,17,25,32,40,47,55,63,71,80,90,96) by NEAREST percentage — note this is
NOT the master's own depth_pct field (which is 10/25/40/50/65/80/95). The
mapping below is the one actually used to build the shipped split files
(verified: each split == the master's records for that layer, exactly).

The split files are committed alongside the other per-pct variants (repo policy),
but they are fully derived from the master — regenerate after editing it with:
    python3 scripts/split_descriptions_v2.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "corpus/generated/descriptions_phi4_tokenpred_gpt4o.json"
OUT_DIR = REPO / "corpus/generated"
SUFFIX = "_v2"

# Phi-4 extraction layer -> AV-trainer grid depth pct (nearest)
LAYER_TO_PCT = {4: 10, 10: 25, 16: 40, 20: 47, 26: 63, 32: 80, 38: 96}


def main():
    master = json.load(open(MASTER))
    by_layer = {}
    for r in master:
        by_layer.setdefault(r["layer"], []).append(
            {"id": r["id"], "description": r["description"]})

    missing = set(LAYER_TO_PCT) - set(by_layer)
    if missing:
        raise SystemExit(f"master missing layers: {sorted(missing)}")

    for layer, pct in sorted(LAYER_TO_PCT.items()):
        rows = by_layer[layer]
        out = OUT_DIR / f"descriptions_L{pct}pct{SUFFIX}.json"
        json.dump(rows, open(out, "w"), ensure_ascii=False, indent=0)
        print(f"L{layer} -> {out.name}: {len(rows)} records")


if __name__ == "__main__":
    main()
