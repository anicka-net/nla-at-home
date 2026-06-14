#!/usr/bin/env python3
"""
Regenerate NLA descriptions using Claude Sonnet via `claude -p`.
Replaces Kimi descriptions with Sonnet-quality ones for expansion + WildChat texts.

Usage:
  # Test on 5 samples
  python3 scripts/generate_descriptions_sonnet.py --test 5

  # Full run (resumes from checkpoint)
  python3 scripts/generate_descriptions_sonnet.py

  # Force regenerate specific source
  python3 scripts/generate_descriptions_sonnet.py --source wildchat
"""
import json, subprocess, argparse, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

PROMPT_TEMPLATE = """You are analyzing what a 7-billion parameter language model (Qwen 2.5 7B Instruct) does internally when it processes the following text at approximately 71% depth through its layers (layer 20 of 28).

At this depth, the model has already parsed syntax and basic semantics. It is now doing higher-level processing: resolving references, predicting likely continuations, assessing intent, and routing toward appropriate response strategies.

Describe what the model is likely doing internally at this layer. Focus on:
1. What predictions the model is forming about what comes next
2. What intent/topic classifications are active
3. What response strategies are being prepared
4. Any notable tensions or competing interpretations

Write 2-3 detailed sentences (~80-120 words). Be specific about THIS text — name actual tokens, structures, and constraints. No markdown formatting, no bullet points. Write as flowing prose.

TEXT: {text}"""


def call_sonnet(text, timeout=60):
    prompt = PROMPT_TEMPLATE.format(text=text[:1500])
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "sonnet"],
            input=prompt, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            desc = result.stdout.strip()
            if len(desc) > 20:
                return desc
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
    return None


def load_texts_needing_descriptions(source=None):
    texts = []
    seen = set()

    if source in (None, "expansion"):
        path = GENERATED_DIR / "expansion_texts.json"
        if path.exists():
            for t in json.loads(path.read_text()):
                if t.get("id") and "text" in t and t["id"] not in seen:
                    seen.add(t["id"])
                    texts.append(t)

    if source in (None, "wildchat"):
        path = GENERATED_DIR / "wildchat_stratified.json"
        if path.exists():
            for t in json.loads(path.read_text()):
                if t.get("id") and "text" in t and t["id"] not in seen:
                    seen.add(t["id"])
                    texts.append(t)

    return texts


def load_checkpoint(output_path):
    if output_path.exists():
        return json.loads(output_path.read_text())
    return []


def save_checkpoint(descs, output_path):
    with open(output_path, 'w') as f:
        json.dump(descs, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: generate N samples and print")
    parser.add_argument("--source", choices=["expansion", "wildchat"],
                        default=None, help="Only process this source")
    parser.add_argument("--output", default="corpus/generated/descriptions_L71pct_sonnet_regen.json")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel claude -p calls")
    args = parser.parse_args()

    output_path = REPO_ROOT / args.output

    texts = load_texts_needing_descriptions(args.source)
    print(f"Loaded {len(texts)} texts needing Sonnet descriptions")

    existing = load_checkpoint(output_path)
    done_ids = {d["id"] for d in existing}
    print(f"Already done: {len(done_ids)}")

    remaining = [t for t in texts if t["id"] not in done_ids]
    if args.test:
        remaining = remaining[:args.test]
    print(f"To generate: {len(remaining)}")

    descs = list(existing)
    n_ok = 0
    n_fail = 0
    n_done = 0
    t_start = time.time()

    def process_one(text_item):
        desc = call_sonnet(text_item["text"])
        return text_item, desc

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_one, t): t for t in remaining}

        for future in as_completed(futures):
            text_item, desc = future.result()
            n_done += 1

            if desc:
                descs.append({"id": text_item["id"], "description": desc})
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL: {text_item['id']}")

            if n_done % args.checkpoint_every == 0:
                save_checkpoint(descs, output_path)
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60
                eta = (len(remaining) - n_done) / max(rate / 60, 0.01) / 60
                print(f"  [{n_done}/{len(remaining)}] ok={n_ok} fail={n_fail} "
                      f"rate={rate:.0f}/min ETA={eta:.0f}min")

            if args.test and desc:
                print(f"\n--- {text_item['id']} ---")
                print(f"TEXT: {text_item['text'][:200]}...")
                print(f"DESC: {desc[:300]}")
                print()

    save_checkpoint(descs, output_path)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed/60:.1f}min")
    print(f"Total descriptions: {len(descs)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
