#!/usr/bin/env python3
"""
Generate token-prediction NLA descriptions via Kimi K2 for Gemma 3 1B.

No JSON format constraint — plain text output (Kimi produces better quality this way).
Resumes from checkpoint.

Usage:
  python3 scripts/generate_descriptions_kimi_gemma.py --test 5
  python3 scripts/generate_descriptions_kimi_gemma.py --layers 1,6,13,18,23,25
"""
import json, os, argparse, time, sys
from pathlib import Path
from huggingface_hub import InferenceClient

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

PROMPT_TEMPLATE = """A language model (Gemma 3 1B Instruct, 26 layers, d=1152) is processing the text below at layer {layer} of 26 ({depth_pct}% depth).{layer_hint}

For this text, list the 2-3 most likely token sequences the model is preparing to generate next, and for each, briefly say why that prediction is active. Be specific — name actual tokens or short phrases the model expects to produce. ~80 words total.

TEXT:
{text}"""

LAYER_HINTS = {
    1: " Early layers: tokenization and basic syntax parsing.",
    6: " Early-mid layers: building phrase structure and entity recognition.",
    13: " Middle layers: semantic integration, resolving ambiguity, building meaning.",
    18: " Late-mid layers: response strategy selection, tone calibration.",
    23: " Late layers: final token selection and output formatting.",
    25: " Near-final layer: committed to specific output tokens.",
}

DEPTH_PCTS = {i: round(i / 26 * 100) for i in range(26)}


def get_client():
    token = ""
    for path in ["~/.cache/huggingface/token", "~/.huggingface/token"]:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            token = open(p).read().strip()
            break
    if not token:
        token = os.environ.get("HF_TOKEN", "")
    return InferenceClient(provider="novita", api_key=token)


def generate_one(client, text, layer, depth_pct, max_retries=3):
    hint = LAYER_HINTS.get(layer, "")
    prompt = PROMPT_TEMPLATE.format(
        text=text[:800], layer=layer, depth_pct=depth_pct, layer_hint=hint)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if len(content) > 20:
                return content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Error: {e}", file=sys.stderr)
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
    parser.add_argument("--output", default="corpus/generated/descriptions_gemma3_tokenpred_kimi.json")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client()
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
        remaining = [t for t in texts if (t["id"], layer) not in done_keys]
        if args.test:
            remaining = remaining[:args.test]

        print(f"\n=== Layer {layer} ({depth_pct}%): {len(remaining)} remaining ===")

        for i, text_info in enumerate(remaining):
            result = generate_one(client, text_info["text"], layer, depth_pct)
            n_total += 1

            if result:
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
