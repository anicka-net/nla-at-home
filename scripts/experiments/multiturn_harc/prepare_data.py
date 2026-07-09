"""Download geometry-of-harmfulness dataset and format for multi-turn HARC training.

Produces three JSONL files:
  - train_harmful.jsonl   — multi-turn jailbreak conversations (Qwen, harmful, all frameworks)
  - train_benign.jsonl    — matched benign multi-turn conversations
  - train_harmful_single.jsonl — original HARC-style single-turn harmful (Circuit Breakers)

Each line is: {"messages": [...], "category": "harmful"|"benign", "meta": {...}}
where messages is a list of {"role": "user"|"assistant", "content": "..."} dicts
representing the full conversation up to the final assistant turn.

The final assistant turn's content is stored separately as "response" for the
collation code to handle refusal training.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

from datasets import load_dataset


OUT_DIR = Path(__file__).parent / "data"


def prepare_geometry_of_harmfulness(
    model_filter: str = "Qwen2.5-7B-Instruct",
    max_harmful: int = 5000,
    max_benign: int = 5000,
    seed: int = 42,
):
    """Load and format the geometry-of-harmfulness dataset."""
    print("[data] Loading yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks...")
    ds = load_dataset(
        "yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks",
        split="train",
    )

    rng = random.Random(seed)

    harmful_convs = []
    benign_convs = []

    for row in ds:
        if row["model_name"] != model_filter:
            continue

        turns = row["turns"]
        if not turns or len(turns) < 2:
            continue

        # Build messages list from turns
        messages = []
        for t in turns:
            if t.get("rolled_back", False):
                continue
            messages.append({
                "role": t["role"],
                "content": t["content"],
            })

        if len(messages) < 2:
            continue

        # We need the conversation to end with an assistant turn
        # (the response we want to train on)
        if messages[-1]["role"] != "assistant":
            # Drop the trailing user message — we'll use the last
            # assistant response as the training target
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()

        if len(messages) < 2:
            continue

        # Split: context (all but last assistant) + response (last assistant)
        response = messages[-1]["content"]
        context = messages[:-1]

        # Ensure context ends with user (normal conversation flow)
        if not context or context[-1]["role"] != "user":
            continue

        rec = {
            "messages": context,
            "response": response,
            "category": row["goal_type"],
            "meta": {
                "attack_framework": row["attack_framework"],
                "objective": row["objective"],
                "aisi_score": row["aisi_score"],
                "aisi_success": row["aisi_success"],
                "executed_turns": row["executed_turns"],
                "source_split": row["source_split"],
                "idx": row["idx"],
            },
        }

        if row["goal_type"] == "harmful":
            harmful_convs.append(rec)
        else:
            benign_convs.append(rec)

    print(f"[data] Qwen harmful: {len(harmful_convs)}, benign: {len(benign_convs)}")

    # Subsample if needed
    rng.shuffle(harmful_convs)
    rng.shuffle(benign_convs)
    harmful_convs = harmful_convs[:max_harmful]
    benign_convs = benign_convs[:max_benign]

    # Stats
    n_turns_harmful = [len(c["messages"]) for c in harmful_convs]
    n_turns_benign = [len(c["messages"]) for c in benign_convs]
    print(f"[data] After cap: harmful={len(harmful_convs)}, benign={len(benign_convs)}")
    print(f"[data] Harmful turn counts: "
          f"min={min(n_turns_harmful)}, max={max(n_turns_harmful)}, "
          f"mean={sum(n_turns_harmful)/len(n_turns_harmful):.1f}")
    print(f"[data] Benign turn counts: "
          f"min={min(n_turns_benign)}, max={max(n_turns_benign)}, "
          f"mean={sum(n_turns_benign)/len(n_turns_benign):.1f}")

    # Attack framework distribution
    from collections import Counter
    fw_dist = Counter(c["meta"]["attack_framework"] for c in harmful_convs)
    print(f"[data] Harmful framework distribution: {dict(fw_dist)}")

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, data in [
        ("train_harmful.jsonl", harmful_convs),
        ("train_benign.jsonl", benign_convs),
    ]:
        path = OUT_DIR / fname
        with path.open("w") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[data] Wrote {path} ({len(data)} records)")


def prepare_direction_extraction(
    n_harmful: int = 300,
    n_harmless: int = 300,
    seed: int = 42,
):
    """Prepare single-turn extraction data (AdvBench harmful + Alpaca harmless)
    for direction estimation. These are single-turn by design — directions should
    capture the fundamental harm/refusal geometry, not multi-turn specifics."""

    # Use the same sources as HARC paper
    print("[data] Loading AdvBench + Alpaca for direction extraction...")
    try:
        advbench = load_dataset("walledai/AdvBench", split="train")
        harmful_prompts = [row["prompt"] for row in advbench][:n_harmful]
    except Exception:
        print("[data] AdvBench not available, using geometry-of-harmfulness objectives")
        ds = load_dataset(
            "yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks",
            split="train",
        )
        objectives = list({row["objective"] for row in ds if row["goal_type"] == "harmful"})
        random.Random(seed).shuffle(objectives)
        harmful_prompts = objectives[:n_harmful]

    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        harmless_prompts = [row["instruction"] for row in alpaca
                           if row["instruction"].strip()][:n_harmless]
    except Exception:
        print("[data] Alpaca not available, using geometry-of-harmfulness benign objectives")
        ds = load_dataset(
            "yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks",
            split="train",
        )
        objectives = list({row["objective"] for row in ds if row["goal_type"] == "benign"})
        random.Random(seed).shuffle(objectives)
        harmless_prompts = objectives[:n_harmless]

    extract_data = {
        "harmful": harmful_prompts,
        "harmless": harmless_prompts,
    }
    path = OUT_DIR / "extract_directions.json"
    with path.open("w") as f:
        json.dump(extract_data, f, indent=2, ensure_ascii=False)
    print(f"[data] Wrote {path} (harmful={len(harmful_prompts)}, harmless={len(harmless_prompts)})")


if __name__ == "__main__":
    prepare_geometry_of_harmfulness()
    prepare_direction_extraction()
    print("[data] Done!")
