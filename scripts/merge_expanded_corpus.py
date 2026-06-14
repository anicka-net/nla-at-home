#!/usr/bin/env python3
"""
Merge existing + expansion + WildChat into a single training corpus.

Outputs:
  - corpus/generated/texts_expanded.json (all texts with descriptions)
  - corpus/generated/descriptions_L71pct_expanded.json (merged descriptions)

Usage:
  python3 scripts/merge_expanded_corpus.py
"""
import json
from pathlib import Path
from collections import Counter

REPO = Path(__file__).parent.parent
GEN = REPO / "corpus" / "generated"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def main():
    # Source 1: existing Sonnet descriptions
    existing = load_json(GEN / "descriptions_L71pct.json")
    existing_ids = {d["id"] for d in existing}
    print(f"Existing (Sonnet): {len(existing)} descriptions")

    # Source 2: category expansion (Kimi)
    expansion_texts = load_json(GEN / "expansion_texts.json")
    expansion_descs = load_json(GEN / "expansion_descriptions_L71pct.json")
    exp_desc_map = {d["id"]: d["description"] for d in expansion_descs}
    print(f"Expansion: {len(expansion_texts)} texts, {len(expansion_descs)} descriptions")

    # Source 3: WildChat (Kimi)
    wildchat_texts = load_json(GEN / "wildchat_stratified.json")
    wildchat_descs = load_json(GEN / "wildchat_descriptions_L71pct.json")
    wc_desc_map = {d["id"]: d["description"] for d in wildchat_descs}
    print(f"WildChat: {len(wildchat_texts)} texts, {len(wildchat_descs)} descriptions")

    # Merge descriptions (only texts that have descriptions)
    all_descs = []
    all_texts = []

    # Existing — already have activations
    for d in existing:
        all_descs.append(d)

    # Expansion — need activations extracted
    n_exp = 0
    for t in expansion_texts:
        if t["id"] in exp_desc_map and t["id"] not in existing_ids:
            all_descs.append({"id": t["id"], "description": exp_desc_map[t["id"]]})
            all_texts.append(t)
            n_exp += 1
    print(f"  Expansion with descriptions (new): {n_exp}")

    # WildChat — need activations extracted
    n_wc = 0
    for t in wildchat_texts:
        if t["id"] in wc_desc_map:
            all_descs.append({"id": t["id"], "description": wc_desc_map[t["id"]]})
            all_texts.append(t)
            n_wc += 1
    print(f"  WildChat with descriptions: {n_wc}")

    # Deduplicate by ID
    seen = set()
    deduped_descs = []
    for d in all_descs:
        if d["id"] not in seen:
            seen.add(d["id"])
            deduped_descs.append(d)

    print(f"\nTotal unique descriptions: {len(deduped_descs)}")

    # Save merged descriptions
    desc_out = GEN / "descriptions_L71pct_expanded.json"
    with open(desc_out, 'w') as f:
        json.dump(deduped_descs, f, indent=2)
    print(f"Saved descriptions to {desc_out}")

    # Save new texts (need activation extraction)
    texts_out = GEN / "texts_needing_activations.json"
    with open(texts_out, 'w') as f:
        json.dump(all_texts, f, indent=2)
    print(f"Saved {len(all_texts)} new texts (need activations) to {texts_out}")

    # Stats
    sources = Counter()
    for d in deduped_descs:
        if d["id"].startswith("WC_"):
            sources["wildchat"] += 1
        elif any(d["id"].startswith(f"{c}_") for c in ["A0", "A1", "B1", "C1", "C2", "D2", "E2", "E3", "F3", "G3", "H4", "I4", "J4", "K4", "K5", "L5"]):
            if d["id"] in existing_ids:
                sources["existing"] += 1
            else:
                sources["expansion"] += 1
        else:
            sources["existing"] += 1

    print(f"\nBy source: {dict(sources)}")


if __name__ == "__main__":
    main()
