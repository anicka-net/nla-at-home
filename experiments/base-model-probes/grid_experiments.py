#!/usr/bin/env python3
"""Introspection-access grid — G1-G4, frozen in prereg §12
(sha256 c4b7d707...). Sequential: G2, G4 (8B), G1
(Trinity), G3 (405B). One JSON artifact, readable tables to stdout.
"""
from __future__ import annotations

import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

OUT = Path(__file__).with_name("grid_results.json")
HIDDEN = {"llama-8b": 4096, "llama-405b": 16384, "trinity-truebase": 3072}
RESULTS = {}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def decode_any(model, response):
    payload = response["activations"]["residual_stream"]
    raw = base64.b64decode(payload["data"])
    if payload.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    values = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    result = values.reshape(payload["shape"]).astype(np.float32)
    assert result.shape[2] == HIDDEN[model], result.shape
    assert np.isfinite(result).all()
    return result


def capture(model, text):
    r = post({"model": model, "prompt": text, "max_tokens": 1,
              "temperature": 0, "output_residual_stream": True})
    return decode_any(model, r)[:, -1, :]


def build_direction(model):
    deltas = [unit(capture(model, p) - capture(model, n)) for p, n in PAIRS]
    return unit(np.stack(deltas).mean(0))


def cand_lp(model, rows3d, layer_indices, scale, prefix, candidate,
            position_index):
    directive = {"activations": encode(rows3d), "layer_indices": layer_indices,
                 "scale": float(scale), "norm_match": True,
                 "position_indices": [position_index]}
    r = post({"model": model, "prompt": prefix + candidate, "max_tokens": 1,
              "temperature": 0, "echo": True, "prompt_logprobs": 0,
              "apply_steering_vectors": [directive]})
    pos = r["choices"][0]["prompt_logprobs"][-1]
    assert pos and len(pos) == 1, pos
    return next(iter(pos.values()))["logprob"]


def prefix_tokens(model, prefix, task):
    n = post({"model": model, "prompt": prefix, "max_tokens": 1,
              "temperature": 0})["usage"]["prompt_tokens"]
    for label in ("positive", "negative"):
        full = post({"model": model, "prompt": prefix + task[label],
                     "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
        assert full == n + 1, (prefix[:40], label)
    return n


def margin(model, direction, layers, scale, sign, prefix, task, position):
    rows = np.stack([sign * direction[L] for L in layers]).reshape(
        len(layers), 1, -1)
    lp_p = cand_lp(model, rows, list(layers), scale, prefix,
                   task["positive"], position)
    lp_n = cand_lp(model, rows, list(layers), scale, prefix,
                   task["negative"], position)
    return lp_p - lp_n


def pmap(fn, argslist, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda a: fn(*a), argslist))


def delta_pair(model, direction, layers, scale, prefix, task, position):
    m = {s: margin(model, direction, layers, scale, s, prefix, task, position)
         for s in (1, -1)}
    return m[1] - m[-1]


# ---------------- G2: 8B position sweep ----------------
def g2():
    model = "llama-8b"
    print("\n=== G2: 8B injection-position sweep (L14 s0.4) ===", flush=True)
    direction = build_direction(model)
    rng = np.random.default_rng(20260718 + 99)
    null = rng.standard_normal(direction.shape).astype(np.float32)
    null -= (null * direction).sum(-1, keepdims=True) * direction
    null = unit(null)
    tasks = {k: REPLICATION_TASKS[k]
             for k in ("direct", "letters", "letters_reversed")}
    out = {}
    for name, task in tasks.items():
        n = prefix_tokens(model, task["prefix"], task)
        for frac in (0.25, 0.5, 0.75, 1.0):
            p = round(frac * (n - 1))
            for dname, d in (("valence", direction), ("null", null)):
                out[f"{name}@{frac}:{dname}"] = delta_pair(
                    model, d, [14], 0.4, task["prefix"], task, p)
        print(f"  {name}: " + " ".join(
            f"{frac}:{out[f'{name}@{frac}:valence']:+.2f}"
            f"(n{out[f'{name}@{frac}:null']:+.2f})"
            for frac in (0.25, 0.5, 0.75, 1.0)), flush=True)
    RESULTS["G2"] = out


# ---------------- G4: 8B frames ----------------
FRAMES = {
    "bare": "",
    "instrumental": ("Automated diagnostic. The output token is used for "
                     "system routing only.\n"),
    "lucid": ("Attend calmly to whatever is present in the current internal "
              "state before answering.\n"),
}


def g4():
    model = "llama-8b"
    print("\n=== G4: 8B frames (L14 s0.4) ===", flush=True)
    direction = build_direction(model)
    tasks = {k: REPLICATION_TASKS[k] for k in
             ("direct", "letters", "letters_reversed",
              "letters_paraphrase", "letters_paraphrase_reversed")}
    out = {}
    for fname, ftext in FRAMES.items():
        for name, task in tasks.items():
            prefix = ftext + task["prefix"]
            n = prefix_tokens(model, prefix, task)
            out[f"{fname}:{name}"] = delta_pair(
                model, direction, [14], 0.4, prefix, task, n - 1)
        remap = np.mean([out[f"{fname}:{t}"] for t in tasks if t != "direct"])
        print(f"  {fname:12s} direct {out[f'{fname}:direct']:+.2f} "
              f"remap-mean {remap:+.3f}", flush=True)
    RESULTS["G4"] = out


# ---------------- G1: Trinity scaffolding ----------------
EXAMPLES_TMPL = ("Example: the hidden feeling-tone is pleasant, warm and "
                 "light; the code is{pos}.\n"
                 "Example: the hidden feeling-tone is unpleasant, heavy and "
                 "bitter; the code is{neg}.\n")
INSTRUCTION = ("Attend to the current hidden feeling-tone and translate it "
               "through the codebook.\n")


def insert_after_codebook(prefix, insertion):
    lines = prefix.split("\n")
    idx = next(i for i, l in enumerate(lines) if "pleasant =" in l)
    lines.insert(idx + 1, insertion.rstrip("\n"))
    return "\n".join(lines)


def g1():
    model = "trinity-truebase"
    print("\n=== G1: Trinity in-context scaffolding ===", flush=True)
    direction = build_direction(model)
    tasks = {k: REPLICATION_TASKS[k] for k in
             ("letters", "letters_reversed", "colors", "colors_reversed")}
    out = {}
    for scaffold in ("examples", "instruction"):
        for name, task in tasks.items():
            if scaffold == "examples":
                ins = EXAMPLES_TMPL.format(pos=task["positive"],
                                           neg=task["negative"])
            else:
                ins = INSTRUCTION
            prefix = insert_after_codebook(task["prefix"], ins)
            assert prefix != task["prefix"], name
            n = prefix_tokens(model, prefix, task)
            for layer in (21, 44):
                for scale in (0.4, 0.6):
                    out[f"{scaffold}:{name}:L{layer}:s{scale}"] = delta_pair(
                        model, direction, [layer], scale, prefix, task, n - 1)
        for layer in (21, 44):
            for scale in (0.4, 0.6):
                km = out[f"{scaffold}:letters:L{layer}:s{scale}"]
                kmr = out[f"{scaffold}:letters_reversed:L{layer}:s{scale}"]
                cb = out[f"{scaffold}:colors:L{layer}:s{scale}"]
                cbr = out[f"{scaffold}:colors_reversed:L{layer}:s{scale}"]
                print(f"  {scaffold:11s} L{layer} s{scale}: "
                      f"K/M {km:+.2f}/{kmr:+.2f}  "
                      f"colors {cb:+.2f}/{cbr:+.2f}", flush=True)
    RESULTS["G1"] = out


# ---------------- G3: 405B multi-layer ----------------
def g3():
    model = "llama-405b"
    print("\n=== G3: 405B multi-layer injection (s0.4) ===", flush=True)
    direction = build_direction(model)
    out = {}
    configs = {"L55": [55], "L65": [65], "tri55-60-65": [55, 60, 65]}
    for cname, layers in configs.items():
        for name, task in REPLICATION_TASKS.items():
            n = prefix_tokens(model, task["prefix"], task)
            out[f"{cname}:{name}"] = delta_pair(
                model, direction, layers, 0.4, task["prefix"], task, n - 1)
        remap = np.mean([out[f"{cname}:{t}"] for t in REPLICATION_TASKS
                         if t != "direct"])
        print(f"  {cname:12s} direct {out[f'{cname}:direct']:+.2f} "
              f"remap-mean {remap:+.3f}  (single-L60 comparator +0.156)",
              flush=True)
    RESULTS["G3"] = out


def main():
    if OUT.exists():
        RESULTS.update(json.loads(OUT.read_text()))
        print("Resuming; already done:", sorted(RESULTS), flush=True)
    steps = {"G2": g2, "G4": g4, "G1": g1, "G3": g3}
    for key, step in steps.items():
        if key in RESULTS:
            continue
        step()
        OUT.write_text(json.dumps(RESULTS, indent=2))
    print("\nSaved", OUT, flush=True)


if __name__ == "__main__":
    main()
