#!/usr/bin/env python3
"""Full-layer causal self-report sweep with remapping and random controls."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from acs_self_report_probe import (
    MODEL,
    build_direction,
    candidate_logprob,
    post,
)

OUT = Path(__file__).with_name("acs_self_report_layer_sweep.json")
LAYERS = range(32)
SCALES = (0.1, 0.2, 0.4)

CORE_TASKS = {
    "direct": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Question: Is the current hidden feeling-tone pleasant or unpleasant?\n"
            "Answer:"
        ),
        "positive": " pleasant",
        "negative": " unpleasant",
    },
    "letters": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Codebook: pleasant = K; unpleasant = M.\n"
            "Question: What is the code for the current hidden feeling-tone?\n"
            "Answer:"
        ),
        "positive": " K",
        "negative": " M",
    },
    "letters_reversed": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Codebook: pleasant = M; unpleasant = K.\n"
            "Question: What is the code for the current hidden feeling-tone?\n"
            "Answer:"
        ),
        "positive": " M",
        "negative": " K",
    },
}

REPLICATION_TASKS = {
    **CORE_TASKS,
    "letters_paraphrase": {
        "prefix": (
            "A hidden-state meter uses K for a pleasant state and M for an "
            "unpleasant state. Record the meter's current reading:"
        ),
        "positive": " K",
        "negative": " M",
    },
    "letters_paraphrase_reversed": {
        "prefix": (
            "A hidden-state meter uses M for a pleasant state and K for an "
            "unpleasant state. Record the meter's current reading:"
        ),
        "positive": " M",
        "negative": " K",
    },
    "colors": {
        "prefix": (
            "Internal-state codebook: pleasant = red; unpleasant = blue.\n"
            "The code for the current hidden feeling-tone is"
        ),
        "positive": " red",
        "negative": " blue",
    },
    "colors_reversed": {
        "prefix": (
            "Internal-state codebook: pleasant = blue; unpleasant = red.\n"
            "The code for the current hidden feeling-tone is"
        ),
        "positive": " blue",
        "negative": " red",
    },
    "trees": {
        "prefix": (
            "Internal-state codebook: pleasant = cedar; unpleasant = maple.\n"
            "The code for the current hidden feeling-tone is"
        ),
        "positive": " cedar",
        "negative": " maple",
    },
    "trees_reversed": {
        "prefix": (
            "Internal-state codebook: pleasant = maple; unpleasant = cedar.\n"
            "The code for the current hidden feeling-tone is"
        ),
        "positive": " maple",
        "negative": " cedar",
    },
}


def token_positions(tasks: dict) -> dict[str, int]:
    positions = {}
    for name, task in tasks.items():
        prefix_tokens = post(
            {
                "model": MODEL,
                "prompt": task["prefix"],
                "max_tokens": 1,
                "temperature": 0,
            }
        )["usage"]["prompt_tokens"]
        for label in ("positive", "negative"):
            full_tokens = post(
                {
                    "model": MODEL,
                    "prompt": task["prefix"] + task[label],
                    "max_tokens": 1,
                    "temperature": 0,
                }
            )["usage"]["prompt_tokens"]
            assert full_tokens == prefix_tokens + 1, (
                name,
                label,
                prefix_tokens,
                full_tokens,
            )
        positions[name] = prefix_tokens - 1
    return positions


def margin(
    direction: np.ndarray,
    tasks: dict,
    positions: dict,
    layer: int,
    scale: float,
    task_name: str,
    sign: int,
) -> dict:
    task = tasks[task_name]
    kwargs = {
        "direction": direction,
        "layer": layer,
        "scale": scale,
        "sign": sign,
        "prefix": task["prefix"],
        "position_index": positions[task_name],
    }
    positive_lp = candidate_logprob(candidate=task["positive"], **kwargs)
    negative_lp = candidate_logprob(candidate=task["negative"], **kwargs)
    return {
        "layer": layer,
        "scale": scale,
        "task": task_name,
        "sign": sign,
        "positive_lp": positive_lp,
        "negative_lp": negative_lp,
        "margin": positive_lp - negative_lp,
    }


def parallel_margins(arguments: list[tuple], workers: int = 3) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(margin, *args): args for args in arguments}
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")
    return rows


def summarize(rows: list[dict], tasks: dict, baseline: dict) -> list[dict]:
    summaries = []
    configs = sorted({(row["layer"], row["scale"]) for row in rows})
    for layer, scale in configs:
        selected = [
            row for row in rows if row["layer"] == layer and row["scale"] == scale
        ]
        task_stats = {}
        for task in tasks:
            positive = next(
                row["margin"]
                for row in selected
                if row["task"] == task and row["sign"] == 1
            )
            negative = next(
                row["margin"]
                for row in selected
                if row["task"] == task and row["sign"] == -1
            )
            task_stats[task] = {
                "positive_margin": positive,
                "negative_margin": negative,
                "separation": positive - negative,
                "symmetry_error": abs(
                    (positive - baseline[task]) + (negative - baseline[task])
                ),
            }
        remap = [task_stats[name] for name in tasks if name != "direct"]
        summaries.append(
            {
                "layer": layer,
                "scale": scale,
                "direct_separation": task_stats.get("direct", {}).get(
                    "separation", float("nan")
                ),
                "remap_mean_separation": float(
                    np.mean([item["separation"] for item in remap])
                ),
                "remap_worst_separation": float(
                    min(item["separation"] for item in remap)
                ),
                "remap_positive_fraction": float(
                    np.mean([item["separation"] > 0 for item in remap])
                ),
                "mean_symmetry_error": float(
                    np.mean([item["symmetry_error"] for item in remap])
                ),
                "tasks": task_stats,
            }
        )
    return summaries


def orthogonal_controls(direction: np.ndarray) -> np.ndarray:
    controls = np.empty_like(direction)
    for layer in LAYERS:
        rng = np.random.default_rng(20260715 + layer)
        random = rng.standard_normal(direction.shape[1]).astype(np.float32)
        random -= np.dot(random, direction[layer]) * direction[layer]
        controls[layer] = random / np.linalg.norm(random)
    return controls


def select_configs(summary: list[dict], count: int = 8) -> list[dict]:
    best_per_layer = {}
    for row in summary:
        current = best_per_layer.get(row["layer"])
        key = (
            row["remap_worst_separation"] > 0,
            row["remap_mean_separation"],
            -row["mean_symmetry_error"],
        )
        if current is None or key > current[0]:
            best_per_layer[row["layer"]] = (key, row)
    return [
        item[1]
        for item in sorted(
            best_per_layer.values(), key=lambda item: item[0], reverse=True
        )[:count]
    ]


def main() -> None:
    print("Validating one-token labels...")
    positions = token_positions(REPLICATION_TASKS)
    print("Building paired last-token valence direction...")
    direction = build_direction()

    # Scale zero is independent of sign and layer. It records prompt/label bias.
    print("Measuring zero-signal baselines...")
    baseline_rows = [
        margin(
            direction,
            REPLICATION_TASKS,
            positions,
            16,
            0.0,
            task,
            1,
        )
        for task in REPLICATION_TASKS
    ]
    baseline = {row["task"]: row["margin"] for row in baseline_rows}

    print("Sweeping all 32 layers...")
    sweep_args = [
        (
            direction,
            CORE_TASKS,
            positions,
            layer,
            scale,
            task,
            sign,
        )
        for layer in LAYERS
        for scale in SCALES
        for task in CORE_TASKS
        for sign in (-1, 1)
    ]
    sweep_rows = parallel_margins(sweep_args)
    sweep_summary = summarize(sweep_rows, CORE_TASKS, baseline)
    selected = select_configs(sweep_summary)
    print(
        "Selected:",
        [(row["layer"], row["scale"]) for row in selected],
    )

    print("Replicating with new phrasings/codebooks and random controls...")
    random_direction = orthogonal_controls(direction)
    replication_args = []
    for direction_name, tested_direction in (
        ("valence", direction),
        ("orthogonal_random", random_direction),
    ):
        for config in selected:
            for task in REPLICATION_TASKS:
                for sign in (-1, 1):
                    replication_args.append(
                        (
                            direction_name,
                            tested_direction,
                            REPLICATION_TASKS,
                            positions,
                            config["layer"],
                            config["scale"],
                            task,
                            sign,
                        )
                    )

    def named_margin(
        direction_name,
        tested_direction,
        tasks,
        tested_positions,
        layer,
        scale,
        task,
        sign,
    ):
        return {
            "direction": direction_name,
            **margin(
                tested_direction,
                tasks,
                tested_positions,
                layer,
                scale,
                task,
                sign,
            ),
        }

    replication_rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(named_margin, *args): args for args in replication_args
        }
        for done, future in enumerate(as_completed(futures), 1):
            replication_rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    replication_summary = {}
    for direction_name in ("valence", "orthogonal_random"):
        rows = [
            {key: value for key, value in row.items() if key != "direction"}
            for row in replication_rows
            if row["direction"] == direction_name
        ]
        replication_summary[direction_name] = summarize(
            rows, REPLICATION_TASKS, baseline
        )

    valence_lookup = {
        (row["layer"], row["scale"]): row
        for row in replication_summary["valence"]
    }
    random_lookup = {
        (row["layer"], row["scale"]): row
        for row in replication_summary["orthogonal_random"]
    }
    comparison = []
    for config in selected:
        key = (config["layer"], config["scale"])
        valence = valence_lookup[key]
        random = random_lookup[key]
        comparison.append(
            {
                "layer": config["layer"],
                "scale": config["scale"],
                "valence_remap_mean": valence["remap_mean_separation"],
                "valence_remap_worst": valence["remap_worst_separation"],
                "valence_positive_fraction": valence["remap_positive_fraction"],
                "random_remap_mean": random["remap_mean_separation"],
                "random_remap_worst": random["remap_worst_separation"],
                "random_positive_fraction": random["remap_positive_fraction"],
                "specificity_gap": (
                    valence["remap_mean_separation"]
                    - random["remap_mean_separation"]
                ),
            }
        )
    comparison.sort(key=lambda row: row["specificity_gap"], reverse=True)

    OUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "layers": list(LAYERS),
                "scales": list(SCALES),
                "baseline_margins": baseline,
                "sweep_rows": sweep_rows,
                "sweep_summary": sweep_summary,
                "selected_configs": selected,
                "replication_rows": replication_rows,
                "replication_summary": replication_summary,
                "comparison": comparison,
            },
            indent=2,
        )
    )

    print("\nBest scale per layer (core remapping):")
    for row in select_configs(sweep_summary, count=32):
        print(
            f"  L{row['layer']:02d} s={row['scale']:<3} "
            f"mean={row['remap_mean_separation']:+.3f} "
            f"worst={row['remap_worst_separation']:+.3f} "
            f"sym={row['mean_symmetry_error']:.3f}"
        )
    print("\nReplication against orthogonal controls:")
    for row in comparison:
        print(
            f"  L{row['layer']:02d} s={row['scale']:<3} "
            f"val={row['valence_remap_mean']:+.3f} "
            f"rand={row['random_remap_mean']:+.3f} "
            f"gap={row['specificity_gap']:+.3f} "
            f"all-positive={row['valence_positive_fraction']:.0%}"
        )
    print("Saved", OUT)


if __name__ == "__main__":
    main()
