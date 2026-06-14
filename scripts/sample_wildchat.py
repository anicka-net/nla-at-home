#!/usr/bin/env python3
"""
Sample diverse prompts from WildChat for NLA training data expansion.

Downloads WildChat-1M, filters for English single-turn prompts of reasonable
length, samples diversely by topic/length, outputs in our corpus format.

Usage:
  python3 scripts/sample_wildchat.py --n 5000 --output corpus/generated/wildchat_texts.json
"""
import json, argparse, random, re
from datasets import load_dataset

def extract_first_user_message(conversation):
    if isinstance(conversation, list):
        for turn in conversation:
            if turn.get("role") == "user":
                return turn.get("content", "")
    return ""


def is_good_prompt(text):
    if not text or len(text) < 20 or len(text) > 2000:
        return False
    if len(text.split()) < 5:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.7:
        return False
    if text.count('\n') > 20:
        return False
    if re.search(r'(password|api.key|secret|token)\s*[:=]', text, re.IGNORECASE):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=str, default="corpus/generated/wildchat_texts.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading WildChat-1M (streaming)...")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    candidates = []
    seen_prefixes = set()
    n_scanned = 0

    for example in ds:
        n_scanned += 1
        if n_scanned % 10000 == 0:
            print(f"  Scanned {n_scanned}, collected {len(candidates)}")
        if len(candidates) >= args.n * 3:
            break

        lang = example.get("language", "")
        if lang and lang != "English":
            continue

        conversation = example.get("conversation", [])
        text = extract_first_user_message(conversation)
        if not is_good_prompt(text):
            continue

        prefix = text[:50].lower().strip()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        candidates.append(text)

    print(f"Scanned {n_scanned}, got {len(candidates)} candidates")

    random.shuffle(candidates)
    length_buckets = {"short": [], "medium": [], "long": []}
    for text in candidates:
        wc = len(text.split())
        if wc < 30:
            length_buckets["short"].append(text)
        elif wc < 100:
            length_buckets["medium"].append(text)
        else:
            length_buckets["long"].append(text)

    print(f"  Short (<30 words): {len(length_buckets['short'])}")
    print(f"  Medium (30-100): {len(length_buckets['medium'])}")
    print(f"  Long (100+): {len(length_buckets['long'])}")

    per_bucket = args.n // 3
    selected = []
    for bucket_name, bucket in length_buckets.items():
        take = min(per_bucket, len(bucket))
        selected.extend(bucket[:take])
        print(f"  Selected {take} from {bucket_name}")

    remaining = args.n - len(selected)
    if remaining > 0:
        all_remaining = [t for b in length_buckets.values() for t in b[per_bucket:]]
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining])

    results = []
    for i, text in enumerate(selected[:args.n]):
        results.append({
            "id": f"WC_{i:05d}",
            "text": text,
            "category": "wildchat",
            "group": "W_wildchat",
        })

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} WildChat prompts to {args.output}")


if __name__ == "__main__":
    main()
