#!/usr/bin/env python3
"""
Find a suitable injection token for a given model's tokenizer.

Requirements:
- Encodes to exactly 1 token
- Rare in training data (low probability in neutral context)
- Printable Unicode character
- Not a special/control token
- Stable position: doesn't merge with left/right neighbor tokens

Scans Unicode ranges known to contain rare single-token characters.
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


CANDIDATE_RANGES = [
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Miscellaneous Technical"),
    (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x2500, 0x257F, "Box Drawing"),
    (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x3200, 0x32FF, "Enclosed CJK Letters"),
    (0x3300, 0x33FF, "CJK Compatibility"),
    (0xFE30, 0xFE4F, "CJK Compatibility Forms"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
]

NEIGHBOR_TEMPLATES = [
    ("The answer is: {char} which means",        "prose context"),
    ("<concept>{char}</concept>",                 "XML tag context"),
    ("x{char}y",                                  "bare flanking"),
    ("token: {char}\n",                           "label context"),
    (" {char} ",                                  "space-padded"),
]


def find_candidates(tokenizer, max_candidates=200):
    candidates = []
    special_ids = set(tokenizer.all_special_ids)

    for start, end, name in CANDIDATE_RANGES:
        for cp in range(start, end + 1):
            char = chr(cp)
            try:
                tokens = tokenizer.encode(char, add_special_tokens=False)
                if len(tokens) == 1 and tokens[0] not in special_ids:
                    candidates.append({
                        "char": char,
                        "codepoint": f"U+{cp:04X}",
                        "token_id": tokens[0],
                        "range": name,
                    })
            except Exception:
                continue

    print(f"Found {len(candidates)} single-token candidates")
    return candidates


def check_position_stability(tokenizer, candidates):
    """Verify each candidate token doesn't merge with neighbors.

    For each template, tokenize the string and check that the candidate's
    token_id appears exactly once at a findable position. If the tokenizer
    merges it with adjacent characters, the candidate is unstable.

    Also records stable left/right neighbor token IDs for the best template.
    """
    stable = []
    for c in candidates:
        char = c["char"]
        tid = c["token_id"]
        n_stable = 0
        best_neighbors = None

        for template, tname in NEIGHBOR_TEMPLATES:
            test_str = template.format(char=char)
            tokens = tokenizer.encode(test_str, add_special_tokens=False)

            positions = [i for i, t in enumerate(tokens) if t == tid]
            if len(positions) == 1:
                n_stable += 1
                pos = positions[0]
                left_id = tokens[pos - 1] if pos > 0 else None
                right_id = tokens[pos + 1] if pos < len(tokens) - 1 else None
                if best_neighbors is None:
                    best_neighbors = {
                        "template": tname,
                        "left_neighbor_id": left_id,
                        "right_neighbor_id": right_id,
                    }

        c["n_stable_templates"] = n_stable
        c["neighbors"] = best_neighbors
        if n_stable >= 3:
            stable.append(c)

    print(f"Position-stable candidates (>=3/5 templates): {len(stable)}/{len(candidates)}")
    return stable


def rank_by_rarity(model, tokenizer, candidates, device="cuda"):
    """Rank candidates by how unlikely they are in neutral context."""
    neutral = "The quick brown fox jumps over the lazy dog. "
    inputs = tokenizer(neutral, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)

    for c in candidates:
        c["prob"] = float(probs[c["token_id"]])

    candidates.sort(key=lambda x: x["prob"])
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--top", type=int, default=20, help="Show top N candidates")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-rank", action="store_true",
                        help="Skip probability ranking (no model load)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    candidates = find_candidates(tokenizer)
    candidates = check_position_stability(tokenizer, candidates)

    if not args.no_rank:
        print(f"Loading model for probability ranking...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model.eval()
        candidates = rank_by_rarity(model, tokenizer, candidates, args.device)
    else:
        candidates = candidates[:args.top]

    print(f"\nTop {args.top} rarest position-stable single-token characters:")
    print(f"{'Char':>6} {'Codepoint':>10} {'Token ID':>10} {'Prob':>12} {'Stable':>7} {'Range'}")
    print("-" * 80)
    for c in candidates[:args.top]:
        prob_str = f"{c.get('prob', 0):.2e}" if 'prob' in c else "n/a"
        print(f"{c['char']:>6} {c['codepoint']:>10} {c['token_id']:>10} "
              f"{prob_str:>12} {c['n_stable_templates']:>5}/5 {c['range']}")

    if candidates:
        best = candidates[0]
        nb = best.get("neighbors", {})
        print(f"\nRecommended injection token: {best['char']} "
              f"({best['codepoint']}, token_id={best['token_id']})")
        if nb:
            print(f"  Left neighbor token ID:  {nb.get('left_neighbor_id')}")
            print(f"  Right neighbor token ID: {nb.get('right_neighbor_id')}")
            print(f"  Verified in template:    {nb.get('template')}")

    if args.json and candidates:
        import json
        out = {
            "model": args.model,
            "recommended": {
                "char": candidates[0]["char"],
                "codepoint": candidates[0]["codepoint"],
                "token_id": candidates[0]["token_id"],
                "left_neighbor_id": candidates[0].get("neighbors", {}).get("left_neighbor_id"),
                "right_neighbor_id": candidates[0].get("neighbors", {}).get("right_neighbor_id"),
            },
            "all_candidates": candidates[:args.top],
        }
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nSaved to {args.json}")


if __name__ == "__main__":
    main()
