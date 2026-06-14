#!/usr/bin/env python3
"""
Generate NLA descriptions via Anthropic API directly (no claude -p overhead).
Resumes from existing checkpoint file.

Usage:
  ANTHROPIC_API_KEY=sk-... python3 scripts/generate_descriptions_api.py
  ANTHROPIC_API_KEY=sk-... python3 scripts/generate_descriptions_api.py --parallel 20
"""
import json, os, argparse, time, sys
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


def make_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def call_sonnet(client, text, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(text=text[:1500])
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            if response.stop_reason == "refusal":
                return None
            if response.content and len(response.content) > 0:
                desc = response.content[0].text.strip()
                if len(desc) > 20:
                    return desc
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Error after {max_retries} retries: {e}", file=sys.stderr)
    return None


def load_texts_needing_descriptions():
    texts = []
    seen = set()
    for name in ["expansion_texts.json", "wildchat_stratified.json"]:
        path = GENERATED_DIR / name
        if path.exists():
            for t in json.loads(path.read_text()):
                if t.get("id") and "text" in t and t["id"] not in seen:
                    seen.add(t["id"])
                    texts.append(t)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--checkpoint", default="corpus/generated/descriptions_L71pct_sonnet_regen.json")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    checkpoint_path = REPO_ROOT / args.checkpoint

    texts = load_texts_needing_descriptions()
    print(f"Total texts: {len(texts)}")

    existing = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else []
    done_ids = {d["id"] for d in existing}
    print(f"Already done: {len(done_ids)}")

    remaining = [t for t in texts if t["id"] not in done_ids and len(t.get("text", "")) > 20]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    client = make_client()
    descs = list(existing)
    n_ok = 0
    n_fail = 0
    n_done = 0
    t_start = time.time()

    def process_one(text_item):
        return text_item, call_sonnet(client, text_item["text"])

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
                with open(checkpoint_path, 'w') as f:
                    json.dump(descs, f, indent=2)
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60
                eta = (len(remaining) - n_done) / max(rate / 60, 0.01) / 60
                print(f"  [{n_done}/{len(remaining)}] ok={n_ok} fail={n_fail} "
                      f"rate={rate:.0f}/min ETA={eta:.1f}min")

    with open(checkpoint_path, 'w') as f:
        json.dump(descs, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed/60:.1f}min")
    print(f"Total descriptions: {len(descs)}")
    print(f"Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
