#!/usr/bin/env python3
"""
entity_fidelity.py — Direct entity-fidelity metric for NLA descriptions.

Measures how many named entities in AV descriptions actually appear in the
input text, vs. how many are confabulated. This complements the retrieval-
based faithfulness metric (top1 identification) which can't distinguish
"right text, wrong entities" from "wrong text entirely."

Metric:
    entity_precision = |entities_in_desc ∩ entities_in_input| / |entities_in_desc|
    entity_recall    = |entities_in_desc ∩ entities_in_input| / |entities_in_input|

Where "entities" = proper nouns (NNP/NNPS spans) + named entities (NLTK NER),
deduplicated and lowercased for matching. We deliberately cast a wide net:
false positives in entity extraction are symmetric (affect both input and
description equally), so precision of the *match* is what matters.

Usage:
    python3 scripts/entity_fidelity.py --data /tmp/entity_eval_data.json
    python3 scripts/entity_fidelity.py --data /tmp/entity_eval_data.json --layer 25
    python3 scripts/entity_fidelity.py --data /tmp/entity_eval_data.json --examples 5
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict

import nltk

# Ensure required NLTK data is available
for pkg in ['averaged_perceptron_tagger_eng', 'punkt_tab',
            'maxent_ne_chunker_tab', 'words']:
    nltk.download(pkg, quiet=True)

from nltk import word_tokenize, pos_tag, ne_chunk


def extract_entities(text: str) -> set[str]:
    """Extract named entities and proper noun spans from text.

    Returns lowercased entity strings for case-insensitive matching.
    Combines two signals:
    - NLTK NER (ne_chunk): catches multi-word named entities
    - POS-tagged NNP/NNPS spans: catches proper nouns NER misses
    """
    if not text or not text.strip():
        return set()

    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
    except Exception:
        return set()

    entities = set()

    # 1. NLTK NER entities
    try:
        tree = ne_chunk(tags)
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity = ' '.join(word for word, _ in subtree.leaves())
                if len(entity) > 1:  # skip single-char
                    entities.add(entity.lower())
    except Exception:
        pass

    # 2. Contiguous NNP/NNPS spans (catches things NER misses)
    current_span = []
    for word, tag in tags:
        if tag in ('NNP', 'NNPS'):
            current_span.append(word)
        else:
            if current_span:
                span = ' '.join(current_span)
                if len(span) > 1:
                    entities.add(span.lower())
                # Also add individual words if multi-word
                if len(current_span) > 1:
                    for w in current_span:
                        if len(w) > 1:
                            entities.add(w.lower())
                current_span = []
    if current_span:
        span = ' '.join(current_span)
        if len(span) > 1:
            entities.add(span.lower())
        if len(current_span) > 1:
            for w in current_span:
                if len(w) > 1:
                    entities.add(w.lower())

    # Filter out common false positives
    noise = {'the', 'a', 'an', 'i', 'we', 'you', 'he', 'she', 'it',
             'this', 'that', 'my', 'your', 'his', 'her', 'its', 'our',
             'yes', 'no', 'ok', 'oh', 'ah', 'um', 'well'}
    entities -= noise

    return entities


def fuzzy_match(desc_entities: set[str], input_entities: set[str]) -> set[str]:
    """Match description entities against input entities.

    Uses substring matching in addition to exact match, because NER
    boundaries are noisy (e.g., "Monster Hunter" vs "Monster Hunter World").
    """
    matched = set()
    for d_ent in desc_entities:
        # Exact match
        if d_ent in input_entities:
            matched.add(d_ent)
            continue
        # Substring: desc entity is part of an input entity or vice versa
        for i_ent in input_entities:
            if d_ent in i_ent or i_ent in d_ent:
                matched.add(d_ent)
                break
    return matched


def compute_fidelity(input_text: str, description: str) -> dict:
    """Compute entity fidelity for one (input, description) pair."""
    input_ents = extract_entities(input_text)
    desc_ents = extract_entities(description)

    if not desc_ents:
        return {
            'input_entities': sorted(input_ents),
            'desc_entities': [],
            'matched': [],
            'confabulated': [],
            'precision': None,  # undefined (no entities in desc)
            'recall': None if not input_ents else 0.0,
            'n_desc': 0,
            'n_input': len(input_ents),
            'n_matched': 0,
            'n_confabulated': 0,
        }

    matched = fuzzy_match(desc_ents, input_ents)
    confabulated = desc_ents - matched

    precision = len(matched) / len(desc_ents) if desc_ents else 0.0
    recall = len(matched) / len(input_ents) if input_ents else None

    return {
        'input_entities': sorted(input_ents),
        'desc_entities': sorted(desc_ents),
        'matched': sorted(matched),
        'confabulated': sorted(confabulated),
        'precision': precision,
        'recall': recall,
        'n_desc': len(desc_ents),
        'n_input': len(input_ents),
        'n_matched': len(matched),
        'n_confabulated': len(confabulated),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True,
                    help='Path to entity_eval_data.json (from deepthought)')
    ap.add_argument('--layer', type=int, default=None,
                    help='Restrict to one layer (default: all)')
    ap.add_argument('--examples', type=int, default=0,
                    help='Show N worst-precision examples')
    ap.add_argument('--out', default=None,
                    help='Save per-item results to JSON')
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    input_texts = data['input_texts']
    av_descs = data['av_descriptions']
    layers = data['layers']

    if args.layer is not None:
        av_descs = [r for r in av_descs if r['layer'] == args.layer]
        layers = [args.layer]

    # Compute per-item fidelity
    results = []
    per_layer = defaultdict(list)

    for row in av_descs:
        text_id = row['text_id']
        layer = row['layer']
        desc = row['description']
        input_text = input_texts.get(text_id, '')

        fid = compute_fidelity(input_text, desc)
        fid['text_id'] = text_id
        fid['layer'] = layer
        results.append(fid)

        if fid['precision'] is not None:
            per_layer[layer].append(fid)

    # Summary per layer
    print("=" * 70)
    print("Entity Fidelity — per layer")
    print("=" * 70)
    print(f"{'Layer':>6} {'Depth%':>6} {'N':>4} {'Prec':>7} {'Recall':>7} "
          f"{'Desc_ent':>8} {'Match':>6} {'Confab':>6}")
    print("-" * 70)

    depth_map = {4: 10, 10: 25, 16: 40, 19: 47, 25: 63, 32: 80, 38: 96}
    all_precs = []
    all_confab_counts = []

    for layer in sorted(per_layer.keys()):
        items = per_layer[layer]
        precs = [x['precision'] for x in items]
        recalls = [x['recall'] for x in items if x['recall'] is not None]
        n_desc = sum(x['n_desc'] for x in items)
        n_match = sum(x['n_matched'] for x in items)
        n_confab = sum(x['n_confabulated'] for x in items)

        mean_prec = statistics.mean(precs) if precs else 0
        mean_recall = statistics.mean(recalls) if recalls else 0
        all_precs.extend(precs)
        all_confab_counts.append(n_confab)

        print(f"L{layer:>4} {depth_map.get(layer, '?'):>6}% {len(items):>4} "
              f"{mean_prec:>7.3f} {mean_recall:>7.3f} "
              f"{n_desc:>8} {n_match:>6} {n_confab:>6}")

    print("-" * 70)
    if all_precs:
        print(f"{'ALL':>6} {'':>6} {len(all_precs):>4} "
              f"{statistics.mean(all_precs):>7.3f} {'':>7} "
              f"{'':>8} {'':>6} {sum(all_confab_counts):>6}")

    # Show worst examples
    if args.examples > 0:
        # Sort by precision (ascending = worst first), skip no-entity cases
        scored = [r for r in results if r['precision'] is not None and r['n_desc'] > 0]
        scored.sort(key=lambda r: r['precision'])

        print(f"\n{'=' * 70}")
        print(f"Worst {args.examples} by entity precision")
        print(f"{'=' * 70}")

        for r in scored[:args.examples]:
            text_id = r['text_id']
            layer = r['layer']
            inp_snippet = ' '.join(input_texts[text_id].split())[:200]
            desc_snippet = ' '.join(
                next(d['description'] for d in av_descs
                     if d['text_id'] == text_id and d['layer'] == layer)
                .split())[:200]
            print(f"\n{text_id} L{layer} — precision {r['precision']:.2f} "
                  f"({r['n_matched']}/{r['n_desc']} matched)")
            print(f"  INPUT:  {inp_snippet}...")
            print(f"  DESC:   {desc_snippet}...")
            print(f"  MATCHED:     {r['matched']}")
            print(f"  CONFABULATED: {r['confabulated']}")

    # Save detailed results
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({
                'summary': {
                    layer: {
                        'mean_precision': statistics.mean(
                            x['precision'] for x in per_layer[layer]),
                        'mean_confabulated': statistics.mean(
                            x['n_confabulated'] for x in per_layer[layer]),
                        'total_desc_entities': sum(
                            x['n_desc'] for x in per_layer[layer]),
                        'total_matched': sum(
                            x['n_matched'] for x in per_layer[layer]),
                        'total_confabulated': sum(
                            x['n_confabulated'] for x in per_layer[layer]),
                        'n_items': len(per_layer[layer]),
                    }
                    for layer in sorted(per_layer.keys())
                },
                'items': results
            }, f, indent=2)
        print(f"\nSaved detailed results to {args.out}")


if __name__ == '__main__':
    main()
