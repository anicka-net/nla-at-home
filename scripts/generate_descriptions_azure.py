#!/usr/bin/env python3
"""
Generate token-prediction NLA descriptions via Azure GPT-4o.

Processes texts from the Gemma 3 1B activation corpus at multiple layers.
Resumes from checkpoint. Handles rate limits with exponential backoff.

Usage:
  export AI_KEY=$(az cognitiveservices account keys list --name anna52-ai --resource-group anna52-ai-rg --query "key1" -o tsv)
  python3 scripts/generate_descriptions_azure.py --test 5
  python3 scripts/generate_descriptions_azure.py --layers 1,6,13,18,23,25
"""
import json, os, argparse, time, sys, urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

ENDPOINT = os.environ.get("AI_ENDPOINT", "https://eastus.api.cognitive.microsoft.com")
API_KEY = os.environ.get("AI_KEY", "")

PROMPT_TEMPLATE = """A language model (Gemma 3 1B Instruct, 26 layers, d=1152) is processing the text below at layer {layer} of 26 ({depth_pct}% depth).{layer_hint}

For this text, list the 2-3 most likely token sequences the model is preparing to generate next, and for each, briefly say why that prediction is active. Be specific — name actual tokens or short phrases the model expects to produce. ~80 words total.

TEXT:
{text}"""

LAYER_HINTS = {
    1: " Early layers: tokenization and basic syntax parsing.",
    3: " Early layers: basic token relationships and simple pattern recognition.",
    4: " Early layers: word-level semantics beginning to form, POS tagging active.",
    6: " Early-mid layers: building phrase structure and entity recognition.",
    8: " Early-mid layers: clause boundaries forming, basic coreference.",
    10: " Mid layers: compositional semantics, phrase meaning assembly.",
    13: " Middle layers: semantic integration, resolving ambiguity, building meaning.",
    14: " Mid-late layers: pragmatic intent forming, discourse coherence.",
    16: " Mid-late layers: response planning, register selection.",
    18: " Late-mid layers: response strategy selection, tone calibration.",
    21: " Late layers: output commitment, phrasing finalization.",
    23: " Late layers: final token selection and output formatting.",
    25: " Near-final layer: committed to specific output tokens.",
}

# Gemma 3 1B: 26 layers
DEPTH_PCTS = {i: round(i / 26 * 100) for i in range(26)}


def call_gpt4o(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            url = f"{ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
            body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300, "temperature": 0.3,
            }).encode()
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json", "api-key": API_KEY,
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Error: {e}", file=sys.stderr)
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Error: {e}", file=sys.stderr)
                return None
    return None


def load_texts():
    texts = []
    seen = set()
    for path in sorted(GENERATED_DIR.glob("[A-Z]*.json")):
        for item in json.loads(path.read_text()):
            if "text" in item and item.get("id") not in seen and len(item["text"]) > 20:
                seen.add(item["id"])
                texts.append(item)
    for name in ["expansion_texts.json", "wildchat_stratified.json"]:
        path = GENERATED_DIR / name
        if path.exists():
            for item in json.loads(path.read_text()):
                if "text" in item and item.get("id") not in seen and len(item["text"]) > 20:
                    seen.add(item["id"])
                    texts.append(item)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--layers", default="1,6,13,18,23,25")
    parser.add_argument("--output", default="corpus/generated/descriptions_gemma3_tokenpred_gpt4o.json")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if not API_KEY:
        print("Set AI_KEY env var first.", file=sys.stderr)
        sys.exit(1)

    layers = [int(x) for x in args.layers.split(",")]
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_texts()
    print(f"Total texts: {len(texts)}")
    print(f"Layers: {layers}")
    print(f"Total descriptions needed: {len(texts) * len(layers)}")

    existing = json.loads(output_path.read_text()) if output_path.exists() else []
    done_keys = {(d["id"], d.get("layer", 18)) for d in existing}
    print(f"Already done: {len(done_keys)}")

    descs = list(existing)
    n_ok = 0
    n_fail = 0
    n_total = 0
    t_start = time.time()

    for layer in layers:
        depth_pct = DEPTH_PCTS.get(layer, round(layer / 26 * 100))
        hint = LAYER_HINTS.get(layer, "")

        remaining = [t for t in texts if (t["id"], layer) not in done_keys]
        if args.test:
            remaining = remaining[:args.test]

        print(f"\n=== Layer {layer} ({depth_pct}%): {len(remaining)} remaining ===")

        for i, text_info in enumerate(remaining):
            prompt = PROMPT_TEMPLATE.format(
                text=text_info["text"][:800],
                layer=layer, depth_pct=depth_pct,
                layer_hint=hint,
            )

            result = call_gpt4o(prompt)
            n_total += 1

            if result and len(result) > 20:
                descs.append({
                    "id": text_info["id"],
                    "layer": layer,
                    "depth_pct": depth_pct,
                    "description": result,
                })
                n_ok += 1
            else:
                n_fail += 1

            if n_total % args.checkpoint_every == 0:
                with open(output_path, "w") as f:
                    json.dump(descs, f, indent=2, ensure_ascii=False)
                elapsed = time.time() - t_start
                rate = n_total / elapsed * 60
                total_remaining = sum(
                    len([t for t in texts if (t["id"], l) not in done_keys])
                    for l in layers
                ) - n_total
                eta = total_remaining / max(rate / 60, 0.01) / 60
                print(f"  [{n_total}] L{layer} ok={n_ok} fail={n_fail} "
                      f"rate={rate:.0f}/min ETA={eta:.1f}min")

            if args.test and i < 3 and result:
                print(f"  {text_info['id']} L{layer}: {result[:150]}")

            time.sleep(args.delay)

    with open(output_path, "w") as f:
        json.dump(descs, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed/60:.1f}min")
    print(f"Total descriptions: {len(descs)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
