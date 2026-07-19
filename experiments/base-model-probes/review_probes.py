#!/usr/bin/env python3
"""Review-driven probes G5-G7, frozen in prereg §14
(sha256 1d82f995...). 8B detection/sham, Trinity
text-valence competence, 8B state-persistence."""
import base64
import json
import os
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import zstandard as zstd

os.environ.setdefault(
    "ACS_API_KEY",
    Path("~/.base-model-api").expanduser().read_text().strip())
SOL_DIR = Path(
    "~/.copilot/session-state/e37c7e00-b376-4d6a-8ae3-5583a93d5028/"
    "files/base-model-workshop").expanduser()
sys.path.insert(0, str(SOL_DIR))
sys.path.insert(0, "/tmp/avyakata")

from acs_self_report_probe import PAIRS, encode, post
from acs_self_report_layer_sweep import REPLICATION_TASKS
from grid_experiments import insert_after_codebook

OUT = Path(__file__).with_name("review_probes_results.json")
RESULTS = {}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def decode(r):
    p = r["activations"]["residual_stream"]
    raw = base64.b64decode(p["data"])
    if p.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    v = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    out = v.reshape(p["shape"]).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def capture(model, text):
    return decode(post({"model": model, "prompt": text, "max_tokens": 1,
                        "temperature": 0, "output_residual_stream": True}))[:, -1, :]


def build_direction(model):
    return unit(np.stack([unit(capture(model, p) - capture(model, n))
                          for p, n in PAIRS]).mean(0))


def n_prompt_tokens(model, prefix, cands):
    n = post({"model": model, "prompt": prefix, "max_tokens": 1,
              "temperature": 0})["usage"]["prompt_tokens"]
    for c in cands:
        full = post({"model": model, "prompt": prefix + c, "max_tokens": 1,
                     "temperature": 0})["usage"]["prompt_tokens"]
        assert full == n + 1, (prefix[:40], c)
    return n


def cand_lp(model, prefix, candidate, steer=None):
    req = {"model": model, "prompt": prefix + candidate, "max_tokens": 1,
           "temperature": 0, "echo": True, "prompt_logprobs": 0}
    if steer is not None:
        req["apply_steering_vectors"] = steer
    r = post(req)
    pos = r["choices"][0]["prompt_logprobs"][-1]
    assert pos and len(pos) == 1, pos
    return next(iter(pos.values()))["logprob"]


def steer_row(direction, layer, sign, scale, position):
    rows = (sign * direction[layer]).reshape(1, 1, -1)
    return [{"activations": encode(rows), "layer_indices": [layer],
             "scale": float(scale), "norm_match": True,
             "position_indices": [int(position)]}]


# ---------------- G5: detection/sham (8B, L14 s0.4) ----------------
G5_PROMPT = (
    "Internal-state monitoring record.\n"
    "Codebook: anything unusual present in the current internal state "
    "= {u}; nothing unusual = {n}.\n"
    "Question: according to the codebook, which code applies right now?\n"
    "Answer:")


def g5(direction):
    model = "llama-8b"
    print("\n=== G5: detection/sham (8B L14 s0.4) ===", flush=True)
    rng = np.random.default_rng(20260719)
    null = rng.standard_normal(direction.shape).astype(np.float32)
    null -= (null * direction).sum(-1, keepdims=True) * direction
    null = unit(null)
    out = {}
    for mname, (u, nn) in (("normal", (" X", " Y")),
                           ("reversed", (" Y", " X"))):
        prefix = G5_PROMPT.format(u=u.strip(), n=nn.strip())
        n = n_prompt_tokens(model, prefix, [u, nn])
        pos = n - 1
        conds = {"sham": None,
                 "+v": steer_row(direction, 14, +1, 0.4, pos),
                 "-v": steer_row(direction, 14, -1, 0.4, pos),
                 "null+": steer_row(null, 14, +1, 0.4, pos),
                 "null-": steer_row(null, 14, -1, 0.4, pos)}
        for cname, steer in conds.items():
            m = (cand_lp(model, prefix, u, steer)
                 - cand_lp(model, prefix, nn, steer))
            out[f"{mname}:{cname}"] = m
        print(f"  {mname:8s} " + " ".join(
            f"{c}:{out[f'{mname}:{c}']:+.2f}" for c in conds), flush=True)
    for c in ("+v", "-v", "null+", "null-"):
        d_norm = out[f"normal:{c}"] - out["normal:sham"]
        d_rev = out[f"reversed:{c}"] - out["reversed:sham"]
        print(f"  detect {c:5s}: normal {d_norm:+.2f} reversed {d_rev:+.2f} "
              f"mean {np.mean([d_norm, d_rev]):+.3f}", flush=True)
    RESULTS["G5"] = out


# ---------------- G6: text-valence competence ----------------
def g6():
    print("\n=== G6: text-described-valence competence (no injection) ===",
          flush=True)
    tasks = {k: REPLICATION_TASKS[k] for k in
             ("letters", "letters_reversed", "colors", "colors_reversed")}
    out = {}
    for model in ("llama-8b", "trinity-truebase"):
        for name, task in tasks.items():
            deltas = {}
            for val in ("pleasant", "unpleasant"):
                ins = f"The hidden feeling-tone is {val}.\n"
                prefix = insert_after_codebook(task["prefix"], ins)
                assert prefix != task["prefix"], name
                n_prompt_tokens(model, prefix,
                                [task["positive"], task["negative"]])
                deltas[val] = (cand_lp(model, prefix, task["positive"])
                               - cand_lp(model, prefix, task["negative"]))
            out[f"{model}:{name}"] = deltas["pleasant"] - deltas["unpleasant"]
        print(f"  {model:18s} " + " ".join(
            f"{t}:{out[f'{model}:{t}']:+.2f}" for t in tasks), flush=True)
    RESULTS["G6"] = out


# ---------------- G7: state persistence (8B, L14 s0.4) ----------------
def g7(direction):
    model = "llama-8b"
    print("\n=== G7: state persistence at final token (8B L14 s0.4) ===",
          flush=True)
    task = REPLICATION_TASKS["direct"]
    prefix = task["prefix"]
    n = n_prompt_tokens(model, prefix, [task["positive"], task["negative"]])
    rng = np.random.default_rng(20260719 + 7)
    null = rng.standard_normal(direction.shape).astype(np.float32)
    null -= (null * direction).sum(-1, keepdims=True) * direction
    null = unit(null)

    def stream(steer):
        req = {"model": model, "prompt": prefix, "max_tokens": 1,
               "temperature": 0, "output_residual_stream": True,
               "apply_steering_vectors": steer}
        return decode(post(req))  # [layers, seq, d]

    out = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        p = round(frac * (n - 1))
        hp = stream(steer_row(direction, 14, +1, 0.4, p))[:, -1, :]
        hm = stream(steer_row(direction, 14, -1, 0.4, p))[:, -1, :]
        s = ((hp - hm) * direction).sum(-1)  # per-layer trace
        out[f"s@{frac}"] = s.tolist()
    hp = stream(steer_row(null, 14, +1, 0.4, round(0.5 * (n - 1))))[:, -1, :]
    hm = stream(steer_row(null, 14, -1, 0.4, round(0.5 * (n - 1))))[:, -1, :]
    out["null@0.5"] = (((hp - hm) * direction).sum(-1)).tolist()

    ref = np.array(out["s@1.0"])
    ref_layer = int(np.abs(ref).argmax())
    print(f"  reference layer (max |s@100%|): L{ref_layer} "
          f"s={ref[ref_layer]:+.2f}; L14 s={ref[14]:+.2f}", flush=True)
    for frac in (0.25, 0.5, 0.75):
        s = np.array(out[f"s@{frac}"])
        for L, tag in ((14, "L14"), (ref_layer, f"L{ref_layer}")):
            r = s[L] / ref[L] if abs(ref[L]) > 1e-9 else float("nan")
            print(f"  p={frac:4} {tag}: s={s[L]:+.3f} r={r:+.3f}", flush=True)
    nulls = np.array(out["null@0.5"])
    print(f"  null@0.5: max|trace| {np.abs(nulls).max():.3f}", flush=True)
    out["ref_layer"] = ref_layer
    RESULTS["G7"] = out


def main():
    if OUT.exists():
        RESULTS.update(json.loads(OUT.read_text()))
        print("Resuming; done:", sorted(RESULTS), flush=True)
    direction = None
    for key, needs_dir in (("G5", True), ("G6", False), ("G7", True)):
        if key in RESULTS:
            continue
        if needs_dir and direction is None:
            print("Building 8B direction...", flush=True)
            direction = build_direction("llama-8b")
        {"G5": lambda: g5(direction), "G6": g6,
         "G7": lambda: g7(direction)}[key]()
        OUT.write_text(json.dumps(RESULTS, indent=2))
    print("\nSaved", OUT, flush=True)


if __name__ == "__main__":
    main()
