#!/usr/bin/env python3
"""Aim 1 cross-model replication — llama-405b / trinity-truebase.

Frozen procedure: prereg doc §10 (sha256 26df7eec...).
Usage: aim1_crossmodel.py <model-alias>

Own capture/inject (the pilot decode asserts the 8B shape); PAIRS, tasks,
transport and encode imported from the pilot modules unchanged.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import base64
import ml_dtypes
import numpy as np
import zstandard as zstd

SOL_DIR = Path(
    "~/.copilot/session-state/e37c7e00-b376-4d6a-8ae3-5583a93d5028/"
    "files/base-model-workshop"
).expanduser()
sys.path.insert(0, str(SOL_DIR))

from acs_self_report_probe import PAIRS, encode, post
from acs_self_report_layer_sweep import REPLICATION_TASKS
from aim1_confirmatory import HELDOUT_PAIRS

HIDDEN = {"llama-405b": 16384, "trinity-truebase": 3072}
ACTIVATION_PROMPT_CAP = {"llama-405b": 32, "trinity-truebase": 256}

MODEL = sys.argv[1]
OUT = Path(__file__).with_name(f"aim1_crossmodel_{MODEL}.json")

CORE_TASKS = ("letters", "letters_reversed")   # selection metric tasks
SWEEP_TASKS = ("direct", "letters", "letters_reversed")
DEPTH_FRACTIONS = (0.25, 0.30, 0.35, 0.40, 0.44, 0.48, 0.55, 0.65)
SCALE = 0.4
N_NULLS = 16
NULL_SEED_BASE = 20260718


def decode_any(response: dict) -> np.ndarray:
    payload = response["activations"]["residual_stream"]
    raw = base64.b64decode(payload["data"])
    if payload.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    values = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    result = values.reshape(payload["shape"]).astype(np.float32)
    assert result.shape[2] == HIDDEN[MODEL], result.shape
    assert np.isfinite(result).all()
    return result


def capture(text: str) -> np.ndarray:
    response = post({"model": MODEL, "prompt": text, "max_tokens": 1,
                     "temperature": 0, "output_residual_stream": True})
    tokens = response["usage"]["prompt_tokens"]
    assert tokens <= ACTIVATION_PROMPT_CAP[MODEL], (text, tokens)
    return decode_any(response)[:, -1, :]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def build_direction(pairs) -> np.ndarray:
    deltas = []
    for positive, negative in pairs:
        deltas.append(unit(capture(positive) - capture(negative)))
    return unit(np.stack(deltas).mean(0))


def candidate_logprob(direction, *, layer, scale, sign, prefix, candidate,
                      position_index):
    vector = (sign * direction[layer]).reshape(1, 1, -1)
    directive = {"activations": encode(vector), "layer_indices": [layer],
                 "scale": float(scale), "norm_match": True,
                 "position_indices": [position_index]}
    response = post({"model": MODEL, "prompt": prefix + candidate,
                     "max_tokens": 1, "temperature": 0, "echo": True,
                     "prompt_logprobs": 0,
                     "apply_steering_vectors": [directive]})
    position = response["choices"][0]["prompt_logprobs"][-1]
    assert position and len(position) == 1, position
    return next(iter(position.values()))["logprob"]


def validate_tasks():
    """Single-token validation; returns (positions, surviving, excluded)."""
    positions, excluded = {}, []
    for name, task in REPLICATION_TASKS.items():
        prefix = post({"model": MODEL, "prompt": task["prefix"],
                       "max_tokens": 1,
                       "temperature": 0})["usage"]["prompt_tokens"]
        ok = True
        for label in ("positive", "negative"):
            full = post({"model": MODEL,
                         "prompt": task["prefix"] + task[label],
                         "max_tokens": 1,
                         "temperature": 0})["usage"]["prompt_tokens"]
            if full != prefix + 1:
                ok = False
        if ok:
            positions[name] = prefix - 1
        else:
            excluded.append(name)
    return positions, list(positions), excluded


def one_margin(direction, positions, task_name, layer, scale, sign):
    task = REPLICATION_TASKS[task_name]
    kw = {"layer": layer, "scale": scale, "sign": sign,
          "prefix": task["prefix"], "position_index": positions[task_name]}
    pos = candidate_logprob(direction, candidate=task["positive"], **kw)
    neg = candidate_logprob(direction, candidate=task["negative"], **kw)
    return pos - neg


def run_jobs(jobs, directions, positions, workers=4):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(one_margin, directions[d], positions, t, l, s,
                             g): (d, t, l, s, g)
                   for (d, t, l, s, g) in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            d, t, l, s, g = futures[future]
            rows.append({"direction": d, "task": t, "layer": l, "scale": s,
                         "sign": g, "margin": future.result()})
            if done % 40 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}", flush=True)
    return rows


def delta(rows, dname, task, layer, scale=SCALE):
    sel = {r["sign"]: r["margin"] for r in rows
           if r["direction"] == dname and r["task"] == task
           and r["layer"] == layer and r["scale"] == scale}
    return sel[1] - sel[-1]


def main():
    print(f"=== {MODEL} ===", flush=True)
    print("Validating tasks against tokenizer...", flush=True)
    positions, surviving, excluded = validate_tasks()
    print(f"surviving: {surviving}", flush=True)
    print(f"excluded: {excluded}", flush=True)
    for task in SWEEP_TASKS:
        assert task in surviving, f"core task {task} excluded — abort"

    print("Building core direction...", flush=True)
    direction = build_direction(PAIRS)
    n_layers = direction.shape[0]
    layers = sorted({int(round(f * n_layers)) for f in DEPTH_FRACTIONS})
    print(f"n_layers={n_layers}, sweep layers={layers}", flush=True)

    directions = {"valence": direction}
    sweep_jobs = [("valence", task, layer, SCALE, sign)
                  for layer in layers for task in SWEEP_TASKS
                  for sign in (-1, 1)]
    print(f"Selection sweep: {len(sweep_jobs)} margins...", flush=True)
    sweep_rows = run_jobs(sweep_jobs, directions, positions)

    def selection_metric(layer):
        return np.mean([delta(sweep_rows, "valence", t, layer)
                        for t in CORE_TASKS])

    best_layer = max(layers, key=selection_metric)
    print(f"Selected L*={best_layer} "
          f"(metric {selection_metric(best_layer):+.3f}); sweep:", flush=True)
    for layer in layers:
        print(f"  L{layer:3d} core-remap {selection_metric(layer):+.3f} "
              f"direct {delta(sweep_rows, 'valence', 'direct', layer):+.3f}",
              flush=True)

    for k in range(1, N_NULLS + 1):
        rng = np.random.default_rng(NULL_SEED_BASE + k)
        raw = rng.standard_normal(direction.shape).astype(np.float32)
        raw -= (raw * direction).sum(-1, keepdims=True) * direction
        directions[f"null_{k}"] = unit(raw)
    print("Building held-out direction...", flush=True)
    directions["heldout"] = build_direction(HELDOUT_PAIRS)

    battery = [("valence", t, best_layer, SCALE, s)
               for t in surviving for s in (-1, 1)]
    battery += [("valence", t, best_layer, 0.0, 1) for t in surviving]
    battery += [(f"null_{k}", t, best_layer, SCALE, s)
                for k in range(1, N_NULLS + 1) for t in surviving
                for s in (-1, 1)]
    battery += [("heldout", t, best_layer, SCALE, s)
                for t in surviving for s in (-1, 1)]
    print(f"Confirmatory battery: {len(battery)} margins...", flush=True)
    rows = run_jobs(battery, directions, positions)

    remap = [t for t in surviving if t != "direct"]
    reversed_surviving = [t for t in remap if t.endswith("reversed")]

    def remap_mean(dname):
        return float(np.mean([delta(rows, dname, t, best_layer)
                              for t in remap]))

    per_task = {t: delta(rows, "valence", t, best_layer) for t in surviving}
    valence_remap = remap_mean("valence")
    null_remaps = sorted(remap_mean(f"null_{k}")
                         for k in range(1, N_NULLS + 1))
    heldout_remap = remap_mean("heldout")
    n_null_ge = sum(1 for n in null_remaps if n >= valence_remap)
    p_rank = (1 + n_null_ge) / (1 + N_NULLS)

    design_ok = len(surviving) - 1 >= 6 and len(reversed_surviving) >= 2
    all_positive = all(v > 0 for v in per_task.values())
    if all_positive and p_rank <= 0.059 and design_ok:
        verdict = "CONFIRMED"
    elif valence_remap <= 0 or p_rank > 0.5:
        verdict = "DISCONFIRMED"
    else:
        verdict = "PARTIAL"

    OUT.write_text(json.dumps({
        "model": MODEL, "n_layers": n_layers, "selected_layer": best_layer,
        "sweep": {str(l): selection_metric(l) for l in layers},
        "surviving_tasks": surviving, "excluded_tasks": excluded,
        "per_task_delta": per_task, "valence_remap_mean": valence_remap,
        "null_remap_means": null_remaps, "p_rank": p_rank,
        "heldout_remap_mean": heldout_remap, "verdict": verdict,
        "prereg_sha256":
            "26df7eec386a0328d2a12567bcbcfd9e24c942d69f5fedaa06561ee36cfb6554",
        "sweep_rows": sweep_rows, "rows": rows,
    }, indent=2))

    print(f"\n=== {MODEL} VERDICT: {verdict} ===", flush=True)
    print(f"L*={best_layer}/{n_layers} "
          f"(depth {best_layer / n_layers:.2f})", flush=True)
    for task, value in per_task.items():
        print(f"  {task:28s} {value:+.3f}")
    print(f"remap-mean Δ {valence_remap:+.3f} | nulls max {null_remaps[-1]:+.3f} "
          f"median {null_remaps[N_NULLS // 2]:+.3f} p={p_rank:.3f} | "
          f"held-out {heldout_remap:+.3f}", flush=True)
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
