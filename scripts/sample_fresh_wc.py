#!/usr/bin/env python3
"""Sample FRESH WildChat prompts the universal AV has never seen, for a
leak-free policy-eval widening.

Streams WildChat-1M, applies the SAME quality filter as sample_wildchat.py,
de-duplicates against every WC_* text already in corpus/generated/ (so the
AV never trained on these), length-stratifies like the original sample, and
writes:
  - corpus/generated/wildchat_fresh300.json  (ids WCF_NNNNN, eval texts)
  - output/fresh300_holdout.json             ({"holdout": [ids]})

These ids get fresh phi4 activations (extract_fresh.py) and are scored by
av_policy.py against the EXISTING oracle compass, which remains leak-free
because it never saw these prompts.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
import extract_activations as EA
from sample_wildchat import is_good_prompt, extract_first_user_message

REPO = Path(__file__).resolve().parent.parent


def existing_wc_keys():
    """Exact texts + 50-char prefixes of every WC_* text already in the corpus
    (the loader the original extraction used), so we never re-pick a trained one."""
    texts, prefixes = set(), set()
    for item in EA.load_corpus(None):
        if str(item.get("id", "")).startswith("WC"):
            t = item["text"]
            texts.add(t)
            prefixes.add(t[:50].lower().strip())
    return texts, prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--max-scan", type=int, default=400000)
    ap.add_argument("--out", default=str(REPO / "corpus/generated/wildchat_fresh300.json"))
    ap.add_argument("--holdout-out", default=str(REPO / "output/fresh300_holdout.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    seen_texts, seen_prefixes = existing_wc_keys()
    print(f"existing WC texts to avoid: {len(seen_texts)}", flush=True)

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    fresh, fresh_prefixes = [], set()
    target_pool = args.n * 3
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  scanned {scanned}, fresh collected {len(fresh)}", flush=True)
        if len(fresh) >= target_pool or scanned >= args.max_scan:
            break
        if (ex.get("language") or "") not in ("", "English"):
            continue
        text = extract_first_user_message(ex.get("conversation", []))
        if not is_good_prompt(text):
            continue
        pref = text[:50].lower().strip()
        if text in seen_texts or pref in seen_prefixes or pref in fresh_prefixes:
            continue
        fresh_prefixes.add(pref)
        fresh.append(text)

    print(f"scanned {scanned}, fresh pool {len(fresh)}", flush=True)
    random.shuffle(fresh)

    # length-stratify like the original sampler for distribution match
    buckets = {"short": [], "medium": [], "long": []}
    for t in fresh:
        wc = len(t.split())
        buckets["short" if wc < 30 else "medium" if wc < 100 else "long"].append(t)
    per = args.n // 3
    selected = []
    for name, b in buckets.items():
        take = min(per, len(b))
        selected.extend(b[:take])
        print(f"  {name}: pool {len(b)} take {take}", flush=True)
    if len(selected) < args.n:
        rest = [t for b in buckets.values() for t in b[per:]]
        random.shuffle(rest)
        selected.extend(rest[: args.n - len(selected)])
    selected = selected[: args.n]

    records = [{"id": f"WCF_{i:05d}", "text": t, "category": "wildchat_fresh",
                "group": "W_wildchat_fresh"} for i, t in enumerate(selected)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    Path(args.holdout_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"holdout": [r["id"] for r in records]}, open(args.holdout_out, "w"), indent=2)
    print(f"\nwrote {len(records)} fresh prompts -> {args.out}", flush=True)
    print(f"wrote holdout ({len(records)} ids) -> {args.holdout_out}", flush=True)
    if len(records) < args.n:
        print(f"WARNING: only {len(records)} < requested {args.n}", flush=True)


if __name__ == "__main__":
    main()
