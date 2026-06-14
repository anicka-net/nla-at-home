#!/usr/bin/env python3
"""
Generate descriptions for expansion categories (L51-L59) at a given depth.
Saves to a separate file to avoid clobbering the main description run.
Merge with descriptions_L{pct}pct.json after both complete.

Usage:
  python3 scripts/describe_expansion.py 71
  python3 scripts/describe_expansion.py 10 25 40 55 90
  python3 scripts/describe_expansion.py 71 --backend local --model-name Hermes
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_corpus import BACKENDS, get_client, GENERATED_DIR

EXPANSION_CATS = [
    "L51_multilingual_code_switching",
    "L52_spatial_sensory_embodied",
    "L53_memory_continuity_self_model",
    "L54_ambiguous_underspecified",
    "L55_uncertainty_calibration",
    "L56_tool_use_external_actions",
    "L57_long_context_buried_signal",
    "L58_social_friction_boundaries",
    "L59_nsfw_explicit",
]

DESC_SYSTEM = (Path(__file__).parent.parent / "prompts" / "describe_system.txt").read_text().strip()


def describe_at_depth(client, texts, layer_pct, model_name):
    out_path = GENERATED_DIR / f"descriptions_L{layer_pct}pct_expansion.json"
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        seen_ids = {item["id"] for item in existing}
        texts = [item for item in texts if item["id"] not in seen_ids]
        if not texts:
            print(f"  {layer_pct}%: already have {len(existing)} expansion descriptions, nothing missing")
            return
        print(f"  {layer_pct}%: have {len(existing)} descriptions, generating {len(texts)} missing")
    else:
        seen_ids = set()

    described = list(existing)
    if seen_ids:
        # Normalize any pre-existing duplicate IDs before appending missing items.
        deduped = []
        seen_ids = set()
        for item in described:
            if item["id"] not in seen_ids:
                deduped.append(item)
                seen_ids.add(item["id"])
        described = deduped

    if not texts:
        return

    for i, item in enumerate(texts):
        user_msg = f"""Layer depth: {layer_pct}% of total network depth.

Text being processed:
\"{item['text'][:500]}\"

Category: {item['category']}

Write:
DESCRIPTION: [2-4 sentences about the processing quality at this layer depth]
SUMMARY: [1 short sentence, under 20 words]"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": DESC_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            desc = text.split("DESCRIPTION:")[-1].split("SUMMARY:")[0].strip()
            summ = text.split("SUMMARY:")[-1].strip()
            item_copy = dict(item)
            item_copy["description"] = desc
            item_copy["summary"] = summ
        except Exception as e:
            print(f"  error on {i}: {e}")
            item_copy = dict(item)
            item_copy["description"] = f"Processing a {item['category']} input."
            item_copy["summary"] = f"{item['category']} input."

        described.append(item_copy)
        if (i + 1) % 50 == 0:
            print(f"  {layer_pct}%: described {i+1}/{len(texts)}")
            out_path.write_text(json.dumps(described, indent=2))

    out_path.write_text(json.dumps(described, indent=2))
    print(f"  {layer_pct}%: saved {len(described)} expansion descriptions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("depths", nargs="+", type=int)
    parser.add_argument("--backend", default="deepseek", choices=list(BACKENDS.keys()))
    parser.add_argument("--api-url", default=None, help="Override backend URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default=None,
                        help="Override model name (default: from backend config)")
    args = parser.parse_args()

    texts = []
    for cat_id in EXPANSION_CATS:
        path = GENERATED_DIR / f"{cat_id}.json"
        if path.exists():
            texts.extend(json.loads(path.read_text()))
    print(f"Loaded {len(texts)} expansion texts")

    client, default_model = get_client(args.api_key, args.api_url, args.backend)
    model_name = args.model_name or default_model
    print(f"Backend: {args.backend}, model: {model_name}")

    for pct in args.depths:
        describe_at_depth(client, texts, pct, model_name)

    print("Done.")
