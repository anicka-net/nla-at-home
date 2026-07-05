#!/usr/bin/env python3
"""
Generate next-token-prediction-style descriptions matching Anthropic's approach.

Sends the SOURCE TEXT to DeepSeek and asks: "What features would a language model
use to predict the next token?" This is fundamentally different from our original
descriptions which narrativize "what the activation represents."

Usage:
  # Test on 10 samples
  python3 scripts/generate_prediction_descriptions.py --test 10

  # Full run
  python3 scripts/generate_prediction_descriptions.py \
    --output corpus/generated/descriptions_L71pct_prediction.json

  # Resume interrupted run
  python3 scripts/generate_prediction_descriptions.py --resume
"""
import json
import argparse
import os
import time
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).parent.parent

INSTRUCTION = """A language model needs to predict what text comes next after the snippet below. Identify the 2-3 most important features it would use for this prediction.

Focus on what the model must be "thinking about" at the point where the text ends. Do not reference truncation — the model is causal, so seeing only a prefix is normal.

Order features by importance for next-token prediction. Each feature: concise ~10-20 word description. Include specific textual examples inline.

Feature types (inspiration, not checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating"

The final feature must describe the very end of the sequence: its role, what it's part of, and immediate constraints on what follows.

Keep to ~80-100 words total. Be specific and factual — no metaphors, no "the model feels."

<text>{text}</text>"""


def call_deepseek(text, api_key, model="deepseek-chat"):
    import httpx

    prompt = INSTRUCTION.replace("{text}", text)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = httpx.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload, headers=headers, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def load_unsafe_categories():
    """Load category IDs marked as unsafe in YAML files."""
    import yaml
    unsafe = set()
    cat_dir = REPO_ROOT / "corpus" / "categories"
    if cat_dir.exists():
        for path in cat_dir.glob("*.yaml"):
            with open(path) as f:
                cat = yaml.safe_load(f)
            if cat.get("unsafe", False):
                unsafe.add(cat["id"])
    return unsafe


def load_all_texts(include_unsafe=False):
    """Load source texts from corpus/generated/{category}.json files."""
    unsafe_cats = load_unsafe_categories() if not include_unsafe else set()
    texts = {}
    skipped_unsafe = 0
    for path in sorted(REPO_ROOT.glob("corpus/generated/*.json")):
        if "descriptions" in path.name or "tight" in path.name or "prediction" in path.name:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and "text" in data[0]:
                for item in data:
                    cat = item["id"].rsplit("_", 1)[0]
                    if cat in unsafe_cats:
                        skipped_unsafe += 1
                        continue
                    texts[item["id"]] = item["text"]
        except (json.JSONDecodeError, KeyError):
            continue
    if skipped_unsafe:
        print(f"  Skipped {skipped_unsafe} unsafe texts")
    return texts


def load_activation_ids(act_path):
    """Get the text IDs that have activations extracted."""
    import torch
    data = torch.load(act_path, weights_only=True, map_location="cpu")
    return data["ids"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction-style descriptions via DeepSeek")
    parser.add_argument("--output", default="corpus/generated/descriptions_L71pct_prediction.json")
    parser.add_argument("--activations", default="corpus/activations/qwen25-7b_L20.pt")
    parser.add_argument("--test", type=int, default=0, help="Test N samples only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-unsafe", action="store_true",
                        help="Include unsafe categories (default: skip)")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        key_file = Path.home() / ".config" / "deepseek" / "api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        # Try .env
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
    if not api_key:
        print("ERROR: No DeepSeek API key found. Set DEEPSEEK_API_KEY or add to .env")
        return

    # Load texts
    print("Loading source texts...")
    all_texts = load_all_texts(include_unsafe=args.include_unsafe)
    print(f"  {len(all_texts)} texts loaded")

    # Filter to texts with activations
    act_path = REPO_ROOT / args.activations
    if act_path.exists():
        import torch
        act_ids = torch.load(str(act_path), weights_only=True, map_location="cpu")["ids"]
        texts_to_process = [(tid, all_texts[tid]) for tid in act_ids if tid in all_texts]
        print(f"  {len(texts_to_process)} texts with activations")
    else:
        texts_to_process = [(tid, txt) for tid, txt in all_texts.items()]

    if args.test > 0:
        import random
        random.seed(42)
        texts_to_process = random.sample(texts_to_process, min(args.test, len(texts_to_process)))
        print(f"  Test mode: {len(texts_to_process)} samples")

    # Resume support
    output_path = REPO_ROOT / args.output
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for item in json.load(f):
                existing[item["id"]] = item["description"]
        texts_to_process = [(tid, txt) for tid, txt in texts_to_process
                            if tid not in existing]
        print(f"  Resuming: {len(existing)} already done, {len(texts_to_process)} remaining")

    # Generate
    results = []
    errors = []
    n_total = len(texts_to_process)

    def process_one(tid_text):
        tid, text = tid_text
        desc = call_deepseek(text, api_key)
        return tid, desc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, tt): tt[0] for tt in texts_to_process}
        done = 0
        for future in as_completed(futures):
            tid, desc = future.result()
            if desc.startswith("ERROR:"):
                errors.append((tid, desc))
                print(f"  [{done+1}/{n_total}] {tid}: {desc}")
            else:
                results.append({"id": tid, "description": desc})
            done += 1
            if done % 10 == 0:
                print(f"  Generated {done}/{n_total} ({len(errors)} errors)")

    # Merge with existing
    if existing:
        all_results = [{"id": tid, "description": desc} for tid, desc in existing.items()]
        all_results.extend(results)
    else:
        all_results = results

    if args.test > 0:
        for r in results[:10]:
            tid = r["id"]
            text = dict(texts_to_process).get(tid, "???")
            print(f"\n=== {tid} ===")
            print(f"TEXT: {text[:150]}")
            print(f"DESC: {r['description'][:200]}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} descriptions to {output_path}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for tid, err in errors[:5]:
            print(f"  {tid}: {err}")


if __name__ == "__main__":
    main()
