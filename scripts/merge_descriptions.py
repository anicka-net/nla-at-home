#!/usr/bin/env python3
"""
Merge description files into a single file.
Deduplicates by text ID, preferring the first occurrence.

Usage:
  python3 scripts/merge_descriptions.py corpus/generated/descriptions_L71pct.json \
      corpus/generated/descriptions_L71pct_expansion.json \
      -o corpus/generated/descriptions_L71pct_merged.json
"""
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Description JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    args = parser.parse_args()

    seen_ids = set()
    merged = []

    for path in args.files:
        data = json.loads(Path(path).read_text())
        added = 0
        for item in data:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                merged.append(item)
                added += 1
        print(f"  {Path(path).name}: {len(data)} items, {added} new")

    Path(args.output).write_text(json.dumps(merged, indent=2))
    print(f"\nMerged: {len(merged)} descriptions -> {args.output}")


if __name__ == "__main__":
    main()
