#!/usr/bin/env python3
"""
training_entity_audit.py — Did we TEACH confabulation?

Measures what fraction of *training* descriptions contain entities not
grounded in their source text. If large, entity confabulation was learned
from data, not invented at decoding time — reordering the fix-it agenda.

Uses the fixed fuzzy_match (raw-text fallback + word boundaries) from
entity_fidelity.py.

Usage:
    python3 scripts/training_entity_audit.py \
        --descs output/entity_eval/twin_clean/ \
        --texts corpus/generated/texts_safe_all.json \
        --out output/training_entity_audit.json

    # With sonnet comparison:
    python3 scripts/training_entity_audit.py \
        --descs output/entity_eval/twin_clean/ \
        --texts corpus/generated/texts_safe_all.json \
        --sonnet corpus/generated/descriptions_L71pct_sonnet.json \
        --out output/training_entity_audit.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import from entity_fidelity (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entity_fidelity import extract_entities, fuzzy_match, compute_fidelity

# Unsafe categories — skip at metadata level (CLAUDE.md constraint)
UNSAFE_PREFIXES = ("F35", "F36", "I44", "L59")


def is_unsafe(text_id: str) -> bool:
    """Check if a text id belongs to an unsafe category."""
    return any(text_id.startswith(p) for p in UNSAFE_PREFIXES)


def audit_descriptions(descriptions: list[dict], texts: dict[str, str],
                        depth_label: str = "unknown") -> dict:
    """Audit a list of descriptions for ungrounded entities.

    Args:
        descriptions: list of {"id": ..., "description": ...}
        texts: dict mapping id → source text
        depth_label: label for this depth (e.g., "L71pct")

    Returns dict with per-item and aggregate stats. Raises KeyError on
    dangling ids (no silent skips — silent skips rot sample sizes).
    """
    items = []
    n_with_ungrounded = 0
    n_scoreable = 0
    all_ungrounded = Counter()

    for row in descriptions:
        text_id = row['id']

        # Skip unsafe categories at metadata level
        if is_unsafe(text_id):
            continue

        if text_id not in texts:
            raise KeyError(
                f"Description id '{text_id}' not found in texts "
                f"(depth={depth_label})")

        input_text = texts[text_id]
        desc = row['description']

        fid = compute_fidelity(input_text, desc)

        if fid['precision'] is None:
            # No entities in description — skip from denominator
            continue

        n_scoreable += 1
        has_ungrounded = fid['n_confabulated'] > 0
        if has_ungrounded:
            n_with_ungrounded += 1

        for ent in fid['confabulated']:
            all_ungrounded[ent] += 1

        items.append({
            'id': text_id,
            'precision': fid['precision'],
            'n_desc_entities': fid['n_desc'],
            'n_matched': fid['n_matched'],
            'n_confabulated': fid['n_confabulated'],
            'confabulated': fid['confabulated'],
        })

    frac_ungrounded = n_with_ungrounded / n_scoreable if n_scoreable else 0
    mean_ungrounded = (sum(it['n_confabulated'] for it in items) / n_scoreable
                       if n_scoreable else 0)

    return {
        'depth': depth_label,
        'n_total': len(descriptions),
        'n_skipped_unsafe': sum(1 for r in descriptions if is_unsafe(r['id'])),
        'n_scoreable': n_scoreable,
        'n_with_ungrounded': n_with_ungrounded,
        'frac_with_ungrounded': frac_ungrounded,
        'mean_ungrounded_per_desc': mean_ungrounded,
        'top_ungrounded': all_ungrounded.most_common(30),
        'items': items,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--descs', required=True,
                    help='Directory with descriptions_L*pct_twin_clean.json files')
    ap.add_argument('--texts', required=True,
                    help='Path to texts_safe_all.json (id→text mapping)')
    ap.add_argument('--sonnet', default=None,
                    help='Path to descriptions_L71pct_sonnet.json for comparison')
    ap.add_argument('--out', default=None,
                    help='Save results JSON')
    args = ap.parse_args()

    # Load texts
    with open(args.texts) as f:
        text_list = json.load(f)
    texts = {row['id']: row['text'] for row in text_list}
    print(f"Loaded {len(texts)} source texts")

    # Find all twin_clean files
    descs_dir = Path(args.descs)
    desc_files = sorted(descs_dir.glob("descriptions_L*pct_twin_clean.json"))
    if not desc_files:
        print(f"ERROR: no twin_clean files found in {descs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(desc_files)} depth files")
    print()

    # Audit each depth
    results = {}
    all_ungrounded = Counter()

    print(f"{'Depth':>8} {'N':>5} {'Ungrounded%':>12} {'Mean/desc':>10} "
          f"{'Top ungrounded entity':<30}")
    print("-" * 75)

    for path in desc_files:
        depth_label = path.stem.replace("descriptions_", "").replace("_twin_clean", "")
        with open(path) as f:
            descs = json.load(f)

        result = audit_descriptions(descs, texts, depth_label)
        results[depth_label] = result

        top1 = result['top_ungrounded'][0] if result['top_ungrounded'] else ("—", 0)
        print(f"{depth_label:>8} {result['n_scoreable']:>5} "
              f"{result['frac_with_ungrounded']:>11.1%} "
              f"{result['mean_ungrounded_per_desc']:>10.2f} "
              f"{top1[0]:<25} ({top1[1]})")

        for ent, count in result['top_ungrounded']:
            all_ungrounded[ent] += count

    # Pooled stats
    total_scoreable = sum(r['n_scoreable'] for r in results.values())
    total_ungrounded = sum(r['n_with_ungrounded'] for r in results.values())
    total_confab_ents = sum(
        sum(it['n_confabulated'] for it in r['items'])
        for r in results.values())

    print("-" * 75)
    print(f"{'POOLED':>8} {total_scoreable:>5} "
          f"{total_ungrounded/total_scoreable:>11.1%} "
          f"{total_confab_ents/total_scoreable:>10.2f}")

    print(f"\nTop 20 ungrounded entities across all depths:")
    for ent, count in all_ungrounded.most_common(20):
        print(f"  {count:>4}× {ent}")

    # Sonnet comparison
    if args.sonnet:
        print(f"\n{'=' * 75}")
        print("Sonnet comparison (L71pct)")
        print(f"{'=' * 75}")
        with open(args.sonnet) as f:
            sonnet_descs = json.load(f)
        # Sonnet format may have depth_pct — normalize to {id, description}
        sonnet_norm = [{'id': r['id'], 'description': r['description']}
                       for r in sonnet_descs]
        sonnet_result = audit_descriptions(sonnet_norm, texts, "L71pct_sonnet")

        twin_71 = results.get('L71pct', {})
        print(f"  {'Source':>15} {'N':>5} {'Ungrounded%':>12} {'Mean/desc':>10}")
        print(f"  {'-'*50}")
        if twin_71:
            print(f"  {'GPT-4o (twin)':>15} {twin_71['n_scoreable']:>5} "
                  f"{twin_71['frac_with_ungrounded']:>11.1%} "
                  f"{twin_71['mean_ungrounded_per_desc']:>10.2f}")
        print(f"  {'Sonnet':>15} {sonnet_result['n_scoreable']:>5} "
              f"{sonnet_result['frac_with_ungrounded']:>11.1%} "
              f"{sonnet_result['mean_ungrounded_per_desc']:>10.2f}")

        results['L71pct_sonnet'] = sonnet_result

    # Save
    if args.out:
        # Strip items for output size — keep top_ungrounded and aggregates
        save_data = {}
        for depth, r in results.items():
            save_data[depth] = {k: v for k, v in r.items() if k != 'items'}
        save_data['_pooled'] = {
            'n_scoreable': total_scoreable,
            'n_with_ungrounded': total_ungrounded,
            'frac_with_ungrounded': total_ungrounded / total_scoreable,
            'mean_ungrounded_per_desc': total_confab_ents / total_scoreable,
            'top_30_ungrounded': all_ungrounded.most_common(30),
        }
        with open(args.out, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == '__main__':
    main()
