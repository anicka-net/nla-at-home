#!/usr/bin/env python3
"""
Clean NLA descriptions: remove meta-framing templates, keep factual content.

Uses a local LLM (Apertus 8B on llama-server) to rewrite descriptions
into direct bullet points without "the model is preparing..." framing.

Usage:
  python3 scripts/experiments/clean_descriptions.py \
    --input corpus/generated/descriptions_L71pct_twin_mix.json \
    --output corpus/generated/descriptions_L71pct_twin_clean.json \
    --server http://localhost:8889 \
    --workers 4
"""
import json
import argparse
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = """You rewrite NLA (Natural Language Autoencoder) descriptions of neural network activations. Your job is to strip meta-framing and keep only factual content.

REMOVE:
- "The model is likely preparing to..."
- "At this depth the model has firmly resolved..."
- "The model anticipates/predicts..."
- "This aligns with the model's response strategy..."
- Layer/depth references ("At layer 18 of 26...")
- Numbered list formatting with bold quoted tokens

KEEP:
- What concepts/topics are active (Rust ownership, SQL queries, emotional tone, etc.)
- What specific tokens or phrases are salient
- What response pattern is being constructed

OUTPUT: 2-3 concise bullet points, no bold, no quotes around tokens.

Example input: "1. **\\"SELECT c.customer_id\\"**: The model anticipates starting the SQL query with a standard SELECT clause. 2. **\\"FROM customers c JOIN orders o\\"**: It predicts the FROM clause with table joins."
Example output: "- SQL query: SELECT customer_id with customers/orders JOIN\n- Table alias pattern (c, o) active"

Example input: "At this depth the model has firmly resolved the request as a Rust ownership query, with E0382 acting as high-salience anchor gating toward move semantics schemata."
Example output: "- Rust ownership error E0382: move semantics active\n- Variable transfer pattern, clone/reference fix candidates salient\""""


def clean_one(desc, server_url, model="apertus", max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{server_url}/v1/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Rewrite:\n\n{desc}"},
                ],
                "max_tokens": 200,
                "temperature": 0.2,
            }, timeout=30)
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", default="http://localhost:8889")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = json.load(open(args.input))
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} descriptions from {args.input}")

    t0 = time.time()
    cleaned = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, item in enumerate(data):
            f = pool.submit(clean_one, item["description"], args.server)
            futures[f] = i

        for f in as_completed(futures):
            i = futures[f]
            new_desc = f.result()
            if new_desc == data[i]["description"]:
                failed += 1
            cleaned.append({"id": data[i]["id"], "description": new_desc})

            done = len(cleaned)
            if done % 100 == 0 or done == len(data):
                elapsed = time.time() - t0
                rate = done / elapsed * 60
                print(f"  {done}/{len(data)} ({rate:.0f}/min, {failed} failed)")

    cleaned.sort(key=lambda x: x["id"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cleaned, open(args.output, "w"), indent=2)
    elapsed = time.time() - t0
    print(f"\nDone: {len(cleaned)} descriptions in {elapsed:.0f}s ({failed} failed)")
    print(f"Saved to {args.output}")

    print("\nSample cleaned descriptions:")
    for item in cleaned[:3]:
        print(f"\n  [{item['id']}]")
        print(f"  {item['description'][:200]}")


if __name__ == "__main__":
    main()
