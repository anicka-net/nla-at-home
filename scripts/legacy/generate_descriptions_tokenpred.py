#!/usr/bin/env python3
"""
Generate token-prediction-style NLA descriptions via Kimi K2 (HF API).

Batch mode: sends 5 texts per API call for efficiency.
Resumes from checkpoint.

Usage:
  python3 scripts/generate_descriptions_tokenpred.py --test 10
  python3 scripts/generate_descriptions_tokenpred.py
"""
import json, os, argparse, time, sys, re
from pathlib import Path
from huggingface_hub import InferenceClient

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

BATCH_PROMPT = """A language model (Qwen 2.5 7B) is processing each text below at layer 20 of 28 (71% depth). At this point the model has parsed syntax and semantics and is now predicting what comes next.

For each text, list the 2-3 most likely token sequences the model is preparing to generate, and for each, briefly say why that prediction is active. Be specific — name actual tokens or short phrases the model expects to produce. ~80 words per text.

Return ONLY a JSON array: [{{"id": "...", "description": "..."}}]. No markdown, no code blocks.

TEXTS:

{texts}"""


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


def parse_json_response(content):
    content = re.sub(r'^```(?:json)?\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content)
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        raw = re.sub(r',\s*\]', ']', match.group())
        raw = re.sub(r'[\x00-\x1f]', ' ', raw)
        return json.loads(raw)
    return None


def generate_batch(client, batch, max_retries=3):
    text_block = ""
    for i, t in enumerate(batch):
        text_block += f'{i+1}. id={t["id"]}: {t["text"][:500]}\n\n'

    prompt = BATCH_PROMPT.format(texts=text_block)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            results = parse_json_response(content)
            if results:
                return results
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Error: {e}", file=sys.stderr)
    return None


def load_all_texts():
    texts = []
    seen = set()
    # Original category texts
    for path in sorted(GENERATED_DIR.glob("[A-Z]*.json")):
        for item in json.loads(path.read_text()):
            if "text" in item and item.get("id") not in seen and len(item["text"]) > 20:
                seen.add(item["id"])
                texts.append(item)
    # Expansion texts
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
    parser.add_argument("--output", default="corpus/generated/descriptions_L71pct_tokenpred_kimi.json")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    output_path = REPO_ROOT / args.output
    client = get_client()

    texts = load_all_texts()
    print(f"Total texts: {len(texts)}")

    existing = json.loads(output_path.read_text()) if output_path.exists() else []
    done_ids = {d["id"] for d in existing}
    print(f"Already done: {len(done_ids)}")

    remaining = [t for t in texts if t["id"] not in done_ids]
    if args.test:
        remaining = remaining[:args.test]
    print(f"Remaining: {len(remaining)}")

    descs = list(existing)
    n_ok = 0
    n_fail = 0
    n_batches = 0
    t_start = time.time()

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        results = generate_batch(client, batch)

        if results:
            batch_ids = {t["id"] for t in batch}
            for item in results:
                if isinstance(item, dict) and "id" in item and "description" in item:
                    if item["id"] in batch_ids and len(item["description"]) > 20:
                        descs.append(item)
                        n_ok += 1
                    else:
                        n_fail += 1
                else:
                    n_fail += 1
        else:
            n_fail += len(batch)

        n_batches += 1
        if n_batches % args.checkpoint_every == 0:
            with open(output_path, 'w') as f:
                json.dump(descs, f, indent=2)
            elapsed = time.time() - t_start
            rate = (i + len(batch)) / elapsed * 60
            eta = (len(remaining) - i - len(batch)) / max(rate / 60, 0.01) / 60
            print(f"  [{i+len(batch)}/{len(remaining)}] ok={n_ok} fail={n_fail} "
                  f"rate={rate:.0f}/min ETA={eta:.1f}min")

        if args.test and results:
            for item in results:
                if isinstance(item, dict):
                    print(f"  {item.get('id','?')}: {item.get('description','')[:200]}")
            print()

        time.sleep(0.5)

    with open(output_path, 'w') as f:
        json.dump(descs, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed/60:.1f}min")
    print(f"Total descriptions: {len(descs)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
