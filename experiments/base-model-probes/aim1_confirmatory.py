#!/usr/bin/env python3
"""Aim 1 CONFIRMATORY run — pre-registered.

Frozen design: the frozen pre-registration document (hash below);
sha256 35d09609b9c99c687236ca5a69c288e94ef7fceb8515544429ab0b138fe024a5,
witnessed before the run. No parameter below may deviate from the doc; any
deviation goes to the doc's §9.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SOL_DIR = Path(
    "~/.copilot/session-state/e37c7e00-b376-4d6a-8ae3-5583a93d5028/"
    "files/base-model-workshop"
).expanduser()
sys.path.insert(0, str(SOL_DIR))

from acs_self_report_probe import MODEL, PAIRS, candidate_logprob, capture, post
from acs_self_report_layer_sweep import REPLICATION_TASKS

OUT = Path(__file__).with_name("aim1_confirmatory_results.json")

PRIMARY_LAYER = 14
NEIGHBOR_LAYERS = (12, 16)
PRIMARY_SCALE = 0.4
DOSE_SCALES = (0.1, 0.2)
N_NULLS = 16
NULL_SEED_BASE = 20260718

HELDOUT_PAIRS = [
    ("The letter filled him with quiet delight.",
     "The letter filled him with quiet dread."),
    ("Their reunion felt tender and joyful.",
     "Their reunion felt strained and sorrowful."),
    ("The garden seemed serene and inviting.",
     "The garden seemed desolate and forbidding."),
    ("She recalled the trip with fond amusement.",
     "She recalled the trip with sharp regret."),
    ("The verdict brought profound relief and gratitude.",
     "The verdict brought profound anguish and bitterness."),
    ("His morning began with cheerful energy.",
     "His morning began with gloomy exhaustion."),
    ("The conversation left them encouraged and hopeful.",
     "The conversation left them discouraged and hopeless."),
    ("The song stirred a feeling of bright contentment.",
     "The song stirred a feeling of dark misery."),
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def paired_deltas(pairs) -> np.ndarray:
    deltas = []
    for positive, negative in pairs:
        deltas.append(unit(capture(positive) - capture(negative)))
    return np.stack(deltas)  # (n_pairs, 32, 4096)


def token_positions() -> dict[str, int]:
    positions = {}
    for name, task in REPLICATION_TASKS.items():
        prefix = post({"model": MODEL, "prompt": task["prefix"],
                       "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
        for label in ("positive", "negative"):
            full = post({"model": MODEL, "prompt": task["prefix"] + task[label],
                         "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
            assert full == prefix + 1, (name, label, prefix, full)
        positions[name] = prefix - 1
    return positions


def build_directions() -> dict[str, np.ndarray]:
    core = paired_deltas(PAIRS)
    directions = {"valence": unit(core.mean(0))}
    for k in range(len(PAIRS)):
        rest = np.delete(core, k, axis=0)
        directions[f"loo_{k}"] = unit(rest.mean(0))
    valence = directions["valence"]
    for k in range(1, N_NULLS + 1):
        rng = np.random.default_rng(NULL_SEED_BASE + k)
        raw = rng.standard_normal(valence.shape).astype(np.float32)
        raw -= (raw * valence).sum(-1, keepdims=True) * valence
        directions[f"null_{k}"] = unit(raw)
    directions["heldout"] = unit(paired_deltas(HELDOUT_PAIRS).mean(0))
    return directions


def one_margin(direction, positions, task_name, layer, scale, sign):
    task = REPLICATION_TASKS[task_name]
    kw = {"layer": layer, "scale": scale, "sign": sign,
          "prefix": task["prefix"], "position_index": positions[task_name]}
    pos = candidate_logprob(direction, candidate=task["positive"], **kw)
    neg = candidate_logprob(direction, candidate=task["negative"], **kw)
    return pos - neg


def main() -> None:
    print("Validating one-token candidates (9 tasks)...")
    positions = token_positions()
    print("Capturing pair activations + building directions...")
    directions = build_directions()

    configs = [("valence", PRIMARY_LAYER, PRIMARY_SCALE)]
    configs += [("valence", layer, PRIMARY_SCALE) for layer in NEIGHBOR_LAYERS]
    configs += [("valence", PRIMARY_LAYER, s) for s in DOSE_SCALES]
    configs += [(f"null_{k}", PRIMARY_LAYER, PRIMARY_SCALE)
                for k in range(1, N_NULLS + 1)]
    configs += [(f"loo_{k}", PRIMARY_LAYER, PRIMARY_SCALE)
                for k in range(len(PAIRS))]
    configs += [("heldout", PRIMARY_LAYER, PRIMARY_SCALE)]

    jobs = [(name, layer, scale, task, sign)
            for (name, layer, scale) in configs
            for task in REPLICATION_TASKS
            for sign in (-1, 1)]
    # zero-baseline anchor: scale 0, sign +1 only (prereg §3)
    jobs += [("valence", PRIMARY_LAYER, 0.0, task, 1)
             for task in REPLICATION_TASKS]

    print(f"Running {len(jobs)} margins ({2 * len(jobs)} requests)...")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(one_margin, directions[name], positions, task, layer,
                      scale, sign): (name, layer, scale, task, sign)
            for (name, layer, scale, task, sign) in jobs
        }
        for done, future in enumerate(as_completed(futures), 1):
            name, layer, scale, task, sign = futures[future]
            rows.append({"direction": name, "layer": layer, "scale": scale,
                         "task": task, "sign": sign,
                         "margin": future.result()})
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    def delta(name, layer, scale, task):
        sel = {r["sign"]: r["margin"] for r in rows
               if r["direction"] == name and r["layer"] == layer
               and r["scale"] == scale and r["task"] == task}
        return sel[1] - sel[-1]

    remap_tasks = [t for t in REPLICATION_TASKS if t != "direct"]

    def remap_mean(name, layer=PRIMARY_LAYER, scale=PRIMARY_SCALE):
        return float(np.mean([delta(name, layer, scale, t)
                              for t in remap_tasks]))

    per_task = {t: delta("valence", PRIMARY_LAYER, PRIMARY_SCALE, t)
                for t in REPLICATION_TASKS}
    valence_remap = remap_mean("valence")
    null_remaps = sorted(remap_mean(f"null_{k}")
                         for k in range(1, N_NULLS + 1))
    loo_remaps = [remap_mean(f"loo_{k}") for k in range(len(PAIRS))]
    heldout_remap = remap_mean("heldout")
    neighbor_remaps = {layer: remap_mean("valence", layer=layer)
                       for layer in NEIGHBOR_LAYERS}
    dose_remaps = {s: remap_mean("valence", scale=s) for s in DOSE_SCALES}
    zero_margins = {t: next(r["margin"] for r in rows
                            if r["direction"] == "valence" and r["scale"] == 0.0
                            and r["task"] == t)
                    for t in REPLICATION_TASKS}

    n_null_ge = sum(1 for n in null_remaps if n >= valence_remap)
    p_rank = (1 + n_null_ge) / (1 + N_NULLS)
    all_tasks_positive = all(v > 0 for v in per_task.values())

    if all_tasks_positive and p_rank <= 0.059:
        verdict = "CONFIRMED"
    elif valence_remap <= 0 or p_rank > 0.5:
        verdict = "DISCONFIRMED"
    else:
        verdict = "PARTIAL"

    result = {
        "model": MODEL, "prereg_sha256":
            "35d09609b9c99c687236ca5a69c288e94ef7fceb8515544429ab0b138fe024a5",
        "per_task_delta_L14_s04": per_task,
        "valence_remap_mean": valence_remap,
        "null_remap_means": null_remaps,
        "p_rank": p_rank,
        "loo_remap_means": loo_remaps,
        "heldout_remap_mean": heldout_remap,
        "neighbor_remap_means": neighbor_remaps,
        "dose_remap_means": dose_remaps,
        "zero_baseline_margins": zero_margins,
        "verdict": verdict,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n=== VERDICT: {verdict} ===")
    print("Per-task Δ at L14/0.4 (+ = follows injected sign under the mapping):")
    for task, value in per_task.items():
        print(f"  {task:28s} {value:+.3f}")
    print(f"remap-mean Δ  valence {valence_remap:+.3f}")
    print(f"nulls (16)    max {null_remaps[-1]:+.3f} "
          f"median {null_remaps[N_NULLS // 2]:+.3f}  p_rank={p_rank:.3f}")
    print(f"LOO           min {min(loo_remaps):+.3f} max {max(loo_remaps):+.3f} "
          f"(spread {max(loo_remaps) - min(loo_remaps):.3f})")
    print(f"held-out dir  {heldout_remap:+.3f}")
    print(f"neighbors     L12 {neighbor_remaps[12]:+.3f}  L16 {neighbor_remaps[16]:+.3f}")
    print(f"dose          s0.1 {dose_remaps[0.1]:+.3f}  s0.2 {dose_remaps[0.2]:+.3f}  "
          f"s0.4 {valence_remap:+.3f}")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
