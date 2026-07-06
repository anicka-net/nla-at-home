#!/usr/bin/env python3
"""Build the texts.jsonl input for split extraction from corpus/generated/.

The extraction scripts expect one {"id", "text"} per line; the corpus
lives as per-category JSON files. Unsafe categories (F35/F36/I44/L59)
are excluded by default — include them only for local-only benchmark
extractions, never for anything that leaves the house.

Usage:
  python3 scripts/franken/build_texts_jsonl.py -o corpus/texts.jsonl
  python3 scripts/franken/build_texts_jsonl.py -o /tmp/t.jsonl --limit 1500
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
UNSAFE_PREFIXES = ("F35", "F36", "I44", "L59")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total texts (spread across categories)")
    ap.add_argument("--include-unsafe", action="store_true",
                    help="Include F35/F36/I44/L59 (LOCAL USE ONLY)")
    args = ap.parse_args()

    files = sorted((REPO / "corpus/generated").glob("[A-Z][0-9][0-9]_*.json"))
    rows = []
    skipped = 0
    for f in files:
        if not args.include_unsafe and f.name.startswith(UNSAFE_PREFIXES):
            skipped += 1
            continue
        for item in json.load(open(f)):
            t = item.get("text")
            i = item.get("id")
            if t and i:
                rows.append({"id": i, "text": t})

    if args.limit and len(rows) > args.limit:
        step = len(rows) / args.limit
        rows = [rows[int(k * step)] for k in range(args.limit)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} texts -> {out} "
          f"({skipped} unsafe categories excluded)" if skipped else
          f"{len(rows)} texts -> {out}")


if __name__ == "__main__":
    main()
