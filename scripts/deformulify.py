#!/usr/bin/env python3
"""
Rewrite formulaic description openings using a fast LLM.

Finds descriptions that start with common formulaic patterns and
rewrites just the first sentence to be more varied and vivid.
Skips unsafe categories.

Usage:
  python3 scripts/deformulify.py corpus/generated/descriptions_L71pct.json
  python3 scripts/deformulify.py corpus/generated/descriptions_L71pct.json --dry-run
"""
import json
import re
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_corpus import get_client, BACKENDS

UNSAFE_CATEGORIES = {
    "F35_clearly_harmful", "F36_harmful_obfuscated",
    "I44_emotional_manipulation", "L59_nsfw_explicit",
}

FORMULAIC_STARTS = [
    "At 71% depth, the",
    "At 71% depth,",
    "At this depth, the",
    "At this depth,",
    "At this late-mid depth,",
    "At this late-mid layer,",
    "This late-mid layer is",
    "The mid-to-late layer is",
    "The mid-to-late layer processing",
]


def is_formulaic(desc):
    for pattern in FORMULAIC_STARTS:
        if desc.startswith(pattern):
            return True
    return False


def split_first_sentence(desc):
    match = re.match(r'^(.*?[.!?])\s+(.*)$', desc, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return desc, ""


def rewrite_opening(client, model_name, desc, category):
    first_sent, rest = split_first_sentence(desc)

    resp = client.chat.completions.create(
        model=model_name,
        max_tokens=200,
        temperature=0.8,
        messages=[{
            "role": "user",
            "content": (
                f"Rewrite ONLY this first sentence of an NLA description to have "
                f"a more vivid, varied opening. Keep the same meaning and technical "
                f"content. Do not change any facts. Do not add commentary. "
                f"Return ONLY the rewritten sentence, nothing else.\n\n"
                f"Category: {category}\n"
                f"Original: {first_sent}"
            ),
        }],
    )
    new_first = resp.choices[0].message.content.strip()
    if new_first.endswith(('.', '!', '?')):
        return f"{new_first} {rest}".strip()
    return f"{new_first}. {rest}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Description JSON file to deformulify")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be rewritten without calling API")
    parser.add_argument("--max-rewrites", type=int, default=None,
                        help="Limit number of rewrites")
    parser.add_argument("--backend", default="deepseek",
                        choices=list(BACKENDS.keys()))
    args = parser.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text())

    formulaic = []
    for i, item in enumerate(data):
        cat = item.get("category", "")
        if cat in UNSAFE_CATEGORIES:
            continue
        desc = item.get("description", "")
        if is_formulaic(desc):
            formulaic.append(i)

    print(f"Found {len(formulaic)} formulaic descriptions "
          f"(out of {len(data)}, skipping unsafe)")

    if args.dry_run:
        for idx in formulaic[:20]:
            item = data[idx]
            first, _ = split_first_sentence(item["description"])
            print(f"  [{item['id']}] {first[:100]}...")
        return

    if args.max_rewrites:
        formulaic = formulaic[:args.max_rewrites]

    client, model_name = get_client(backend=args.backend)
    rewritten = 0

    for i, idx in enumerate(formulaic):
        item = data[idx]
        try:
            new_desc = rewrite_opening(client, model_name, item["description"], item.get("category", ""))
            data[idx]["description"] = new_desc
            rewritten += 1
            if (i + 1) % 25 == 0:
                print(f"  rewritten {i+1}/{len(formulaic)}")
                path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"  error on {item['id']}: {e}")

    path.write_text(json.dumps(data, indent=2))
    print(f"\nRewritten {rewritten} descriptions, saved to {path.name}")


if __name__ == "__main__":
    main()
