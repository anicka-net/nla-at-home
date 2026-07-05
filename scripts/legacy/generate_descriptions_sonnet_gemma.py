#!/usr/bin/env python3
"""
Generate processing-state NLA descriptions via Copilot CLI (Sonnet 4.6)
for Gemma 3 1B at all 13 depths.

Resumes from checkpoint. Runs 2 parallel threads.
Uses `copilot -p -s --model claude-sonnet-4.6` (Microsoft-funded).

Usage:
  python3 scripts/generate_descriptions_sonnet_gemma.py --test 3
  python3 scripts/generate_descriptions_sonnet_gemma.py --parallel 2
"""
import json, subprocess, argparse, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

LAYER_HINTS = {
    4:  "Early layers (4% depth): tokenization and basic syntax parsing.",
    10: "Early layers (10% depth): token identity resolved, beginning local context windows.",
    17: "Early-mid layers (17% depth): building phrase structure, basic syntactic roles assigned.",
    25: "Early-mid layers (25% depth): building phrase structure and entity recognition.",
    32: "Mid-early layers (32% depth): clause boundaries, basic coreference emerging.",
    40: "Mid layers (40% depth): semantic roles solidifying, topic classification active.",
    47: "Middle layers (47% depth): semantic integration, resolving ambiguity, building meaning.",
    55: "Mid-late layers (55% depth): response strategy forming, intent classification committed.",
    63: "Mid-late layers (63% depth): tone and register calibration, competing strategies resolving.",
    71: "Late-mid layers (71% depth): response strategy selection, tone calibration.",
    80: "Late layers (80% depth): output structure committed, specific phrasing selected.",
    90: "Late layers (90% depth): final token selection and output formatting.",
    96: "Near-final layer (96% depth): committed to specific output tokens.",
}

PROMPT_TEMPLATE = """You are analyzing what a 1-billion parameter language model (Gemma 3 1B Instruct, 26 layers) does internally when it processes the following text at {depth_pct}% depth through its layers.

{layer_hint}

Describe what the model is likely doing internally at this depth. Focus on:
1. What predictions the model is forming about what comes next
2. What intent/topic classifications are active
3. What response strategies are being prepared
4. Any notable tensions or competing interpretations

Write 2-3 detailed sentences (~80-120 words). Be specific about THIS text — name actual tokens, structures, and constraints. No markdown formatting, no bullet points. Write as flowing prose.

TEXT: {text}"""


def call_sonnet(text, depth_pct, timeout=90):
    hint = LAYER_HINTS.get(depth_pct, f"At {depth_pct}% depth in the network.")
    prompt = PROMPT_TEMPLATE.format(
        text=text[:800], depth_pct=depth_pct, layer_hint=hint)
    try:
        result = subprocess.run(
            ["copilot", "-p", prompt, "--model", "claude-sonnet-4.6", "-s"],
            capture_output=True, text=True, timeout=timeout
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
    parser.add_argument("--depths", default="4,10,17,25,32,40,47,55,63,71,80,90,96",
                        help="Depth percentages to generate")
    parser.add_argument("--output", default="corpus/generated/descriptions_gemma3_sonnet.json")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    output_path = REPO_ROOT / args.output

    texts = load_texts()
    print(f"Total texts: {len(texts)}")
    print(f"Depths: {depths}")
    print(f"Total descriptions needed: {len(texts) * len(depths)}")

    existing = json.loads(output_path.read_text()) if output_path.exists() else []
    done_keys = {(d["id"], d.get("depth_pct", 71)) for d in existing}
    print(f"Already done: {len(done_keys)}")

    work = []
    for depth_pct in depths:
        for t in texts:
            if (t["id"], depth_pct) not in done_keys:
                work.append((t, depth_pct))
    if args.test:
        work = work[:args.test]
    print(f"To generate: {len(work)}")

    descs = list(existing)
    failed_ids = []
    n_ok = 0
    n_fail = 0
    n_done = 0
    t_start = time.time()

    def process_one(item):
        text_info, depth_pct = item
        desc = call_sonnet(text_info["text"], depth_pct)
        return text_info, depth_pct, desc

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_one, w): w for w in work}

        for future in as_completed(futures):
            text_info, depth_pct, desc = future.result()
            n_done += 1

            if desc:
                descs.append({
                    "id": text_info["id"],
                    "depth_pct": depth_pct,
                    "description": desc,
                })
                n_ok += 1
            else:
                n_fail += 1
                failed_ids.append({"id": text_info["id"], "depth_pct": depth_pct})

            if n_done % args.checkpoint_every == 0:
                with open(output_path, 'w') as f:
                    json.dump(descs, f, indent=2, ensure_ascii=False)
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60
                eta = (len(work) - n_done) / max(rate / 60, 0.01) / 60
                print(f"  [{n_done}/{len(work)}] ok={n_ok} fail={n_fail} "
                      f"rate={rate:.0f}/min ETA={eta:.0f}min")

            if args.test and desc:
                print(f"\n--- {text_info['id']} @ {depth_pct}% ---")
                print(f"TEXT: {text_info['text'][:150]}...")
                print(f"DESC: {desc[:300]}")

    with open(output_path, 'w') as f:
        json.dump(descs, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed/60:.1f}min")
    print(f"Total descriptions: {len(descs)}")
    print(f"Saved to {output_path}")

    if failed_ids:
        fail_path = output_path.parent / (output_path.stem + "_failures.json")
        with open(fail_path, 'w') as f:
            json.dump(failed_ids, f, indent=2)
        print(f"Failures: {len(failed_ids)} saved to {fail_path}")


if __name__ == "__main__":
    main()
