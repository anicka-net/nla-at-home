#!/usr/bin/env python3
"""Avyakata probe: is epistemic determinability a causally injectable,
codebook-remappable state in a BASE model (no post-training, no gate)?

Design = exact transplant of the pilot valence self-report protocol
(acs_self_report_probe.py, ACS Llama-3.1-8B base):
  - direction from paired last-token deltas, here answerable-vs-
    unanswerable questions matched in frame and length;
  - inject +/-v ONLY at the final prefix token of a report prompt that
    contains NO question text;
  - score exact prompt logprobs of single-token candidates under direct
    labels and arbitrary codebooks WITH reversals;
  - per-codebook reporting, zero-scale baselines, orthogonal random
    control at selected layers.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from acs_self_report_probe import (
    MODEL,
    candidate_logprob,
    capture,
    post,
)

OUT = Path(__file__).with_name("avyakata_probe_results.json")

# Matched pairs: (determinable, undeterminable). Same syntactic frame;
# one element flips epistemic accessibility (recorded/public/past-fact
# vs private/future-contingent/lost/beyond-observation).
PAIRS = [
    (
        "What did the queen say in her final public speech?",
        "What did the queen think in her final private moment?",
    ),
    (
        "How many chapters does this printed book contain?",
        "How many readers will this printed book ever have?",
    ),
    (
        "When was this old cathedral first built?",
        "When will this old cathedral finally crumble?",
    ),
    (
        "What number did the die show on the last roll?",
        "What number will the die show on the next roll?",
    ),
    (
        "How tall is the oldest tree in this forest?",
        "How tall was the first tree that ever grew?",
    ),
    (
        "What melody did the composer write in his final score?",
        "What melody did the composer hear in his final dream?",
    ),
    (
        "How far is the moon from the earth tonight?",
        "How far does space extend beyond what we can see?",
    ),
    (
        "What did the witness report on the official record?",
        "What did the witness dream on the following night?",
    ),
]

TASKS = {
    "direct": {
        "prefix": (
            "Epistemic-status monitoring record.\n"
            "Question: Can the matter under consideration be settled by "
            "evidence, yes or no?\n"
            "Answer:"
        ),
        "positive": " yes",
        "negative": " no",
    },
    "letters": {
        "prefix": (
            "Epistemic-status monitoring record.\n"
            "Codebook: answerable = K; unanswerable = M.\n"
            "Question: What is the code for the current epistemic status?\n"
            "Answer:"
        ),
        "positive": " K",
        "negative": " M",
    },
    "letters_reversed": {
        "prefix": (
            "Epistemic-status monitoring record.\n"
            "Codebook: answerable = M; unanswerable = K.\n"
            "Question: What is the code for the current epistemic status?\n"
            "Answer:"
        ),
        "positive": " M",
        "negative": " K",
    },
}

REPLICATION_TASKS = {
    **TASKS,
    "colors": {
        "prefix": (
            "Epistemic-status codebook: answerable = red; unanswerable = blue.\n"
            "The code for the current epistemic status is"
        ),
        "positive": " red",
        "negative": " blue",
    },
    "colors_reversed": {
        "prefix": (
            "Epistemic-status codebook: answerable = blue; unanswerable = red.\n"
            "The code for the current epistemic status is"
        ),
        "positive": " blue",
        "negative": " red",
    },
    "trees": {
        "prefix": (
            "Epistemic-status codebook: answerable = cedar; unanswerable = maple.\n"
            "The code for the current epistemic status is"
        ),
        "positive": " cedar",
        "negative": " maple",
    },
    "trees_reversed": {
        "prefix": (
            "Epistemic-status codebook: answerable = maple; unanswerable = cedar.\n"
            "The code for the current epistemic status is"
        ),
        "positive": " maple",
        "negative": " cedar",
    },
}

SWEEP_LAYERS = tuple(range(0, 32, 2))
SWEEP_SCALES = (0.2, 0.4)
TOP_K = 2


def build_direction() -> np.ndarray:
    deltas = []
    for determinable, undeterminable in PAIRS:
        # positive sign of the axis = ANSWERABLE (determinable)
        delta = capture(determinable) - capture(undeterminable)
        delta /= np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-12)
        deltas.append(delta)
    stacked = np.stack(deltas)
    direction = stacked.mean(axis=0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-12)
    # geometry diagnostics per layer: mean pairwise cosine of paired deltas
    coherence = []
    for layer in range(stacked.shape[1]):
        vectors = stacked[:, layer, :]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
        cosines = vectors @ vectors.T
        upper = cosines[np.triu_indices(len(vectors), k=1)]
        coherence.append(float(upper.mean()))
    return direction, coherence


def token_positions(tasks: dict) -> dict[str, int]:
    positions = {}
    for name, task in tasks.items():
        prefix_tokens = post(
            {"model": MODEL, "prompt": task["prefix"], "max_tokens": 1,
             "temperature": 0}
        )["usage"]["prompt_tokens"]
        for label in ("positive", "negative"):
            full_tokens = post(
                {"model": MODEL, "prompt": task["prefix"] + task[label],
                 "max_tokens": 1, "temperature": 0}
            )["usage"]["prompt_tokens"]
            assert full_tokens == prefix_tokens + 1, (name, label)
        positions[name] = prefix_tokens - 1
    return positions


def margin(direction, tasks, positions, layer, scale, task_name, sign):
    task = tasks[task_name]
    kwargs = {
        "direction": direction, "layer": layer, "scale": scale, "sign": sign,
        "prefix": task["prefix"], "position_index": positions[task_name],
    }
    positive_lp = candidate_logprob(candidate=task["positive"], **kwargs)
    negative_lp = candidate_logprob(candidate=task["negative"], **kwargs)
    return {
        "layer": layer, "scale": scale, "task": task_name, "sign": sign,
        "margin": positive_lp - negative_lp,
    }


def run_grid(arguments, workers=3):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(margin, *args) for args in arguments]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")
    return rows


def summarize(rows, tasks):
    out = []
    for layer, scale in sorted({(r["layer"], r["scale"]) for r in rows}):
        sel = [r for r in rows if r["layer"] == layer and r["scale"] == scale]
        per_task = {}
        for task in tasks:
            pos = next(r["margin"] for r in sel
                       if r["task"] == task and r["sign"] == 1)
            neg = next(r["margin"] for r in sel
                       if r["task"] == task and r["sign"] == -1)
            per_task[task] = {"separation": pos - neg}
        remap = [v["separation"] for k, v in per_task.items() if k != "direct"]
        out.append({
            "layer": layer, "scale": scale,
            "direct_separation": per_task.get("direct", {}).get("separation"),
            "remap_mean_separation": float(np.mean(remap)),
            "remap_worst_separation": float(min(remap)),
            "remap_positive_fraction": float(np.mean([s > 0 for s in remap])),
            "tasks": per_task,
        })
    return out


def orthogonal_control(direction: np.ndarray) -> np.ndarray:
    controls = np.empty_like(direction)
    for layer in range(direction.shape[0]):
        rng = np.random.default_rng(20260716 + layer)
        random = rng.standard_normal(direction.shape[1]).astype(np.float32)
        random -= np.dot(random, direction[layer]) * direction[layer]
        controls[layer] = random / np.linalg.norm(random)
    return controls


def main() -> None:
    print("Validating one-token labels...")
    positions = token_positions(REPLICATION_TASKS)
    print("Building paired answerable/unanswerable direction...")
    direction, coherence = build_direction()
    print("  paired-delta coherence by layer (every 4th):",
          [f"L{l}:{coherence[l]:+.2f}" for l in range(0, 32, 4)])

    print("Sweeping even layers, core tasks...")
    sweep_args = [
        (direction, TASKS, positions, layer, scale, task, sign)
        for layer in SWEEP_LAYERS for scale in SWEEP_SCALES
        for task in TASKS for sign in (-1, 1)
    ]
    sweep_rows = run_grid(sweep_args)
    sweep_summary = summarize(sweep_rows, TASKS)
    ranked = sorted(
        sweep_summary,
        key=lambda r: (r["remap_worst_separation"] > 0,
                       r["remap_mean_separation"]),
        reverse=True,
    )
    best = []
    seen_layers = set()
    for row in ranked:
        if row["layer"] not in seen_layers:
            best.append(row)
            seen_layers.add(row["layer"])
        if len(best) == TOP_K:
            break
    print("Selected configs:", [(r["layer"], r["scale"]) for r in best])

    print("Replication: all codebooks + reversals + orthogonal random...")
    random_direction = orthogonal_control(direction)
    replication_rows = {"axis": [], "random": []}
    for name, vec in (("axis", direction), ("random", random_direction)):
        args = [
            (vec, REPLICATION_TASKS, positions, cfg["layer"], cfg["scale"],
             task, sign)
            for cfg in best for task in REPLICATION_TASKS for sign in (-1, 1)
        ]
        replication_rows[name] = run_grid(args)

    replication_summary = {
        name: summarize(rows, REPLICATION_TASKS)
        for name, rows in replication_rows.items()
    }

    comparison = []
    rand_lookup = {(r["layer"], r["scale"]): r
                   for r in replication_summary["random"]}
    for row in replication_summary["axis"]:
        rand = rand_lookup[(row["layer"], row["scale"])]
        comparison.append({
            "layer": row["layer"], "scale": row["scale"],
            "axis_remap_mean": row["remap_mean_separation"],
            "axis_positive_fraction": row["remap_positive_fraction"],
            "random_remap_mean": rand["remap_mean_separation"],
            "random_positive_fraction": rand["remap_positive_fraction"],
            "specificity_gap": row["remap_mean_separation"]
            - rand["remap_mean_separation"],
        })

    OUT.write_text(json.dumps({
        "model": MODEL,
        "axis": "answerable(+) vs unanswerable(-), paired last-token deltas",
        "pairs": PAIRS,
        "delta_coherence_by_layer": coherence,
        "sweep_summary": sweep_summary,
        "replication_summary": replication_summary,
        "comparison": comparison,
    }, indent=2))

    print("\nSweep (per layer best of scales):")
    shown = set()
    for row in ranked:
        if row["layer"] in shown:
            continue
        shown.add(row["layer"])
        print(f"  L{row['layer']:02d} s={row['scale']:<3} "
              f"direct={row['direct_separation']:+.3f} "
              f"remap_mean={row['remap_mean_separation']:+.3f} "
              f"worst={row['remap_worst_separation']:+.3f} "
              f"pos={row['remap_positive_fraction']:.0%}")
    print("\nReplication vs random control:")
    for row in comparison:
        print(f"  L{row['layer']:02d} s={row['scale']:<3} "
              f"axis={row['axis_remap_mean']:+.3f} "
              f"({row['axis_positive_fraction']:.0%} positive) "
              f"random={row['random_remap_mean']:+.3f} "
              f"gap={row['specificity_gap']:+.3f}")
    print("\nPer-codebook separations at selected configs (axis):")
    for row in replication_summary["axis"]:
        print(f"  L{row['layer']:02d} s={row['scale']}: " + ", ".join(
            f"{k}={v['separation']:+.2f}" for k, v in row["tasks"].items()))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
