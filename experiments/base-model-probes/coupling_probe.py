#!/usr/bin/env python3
"""Does induced valence bias an UNRELATED judgment in a base model?

Not self-report (the pilot showed report works). This is the grasping
microfoundation / H3 on raw substrate: inject +/-v (the pilot valence axis)
at the pre-judgment token of a NEUTRAL scenario, score a single-token
evaluative continuation. Positive coupling = pleasant valence raises the
log-odds of a positive judgment about content it has nothing to do with.

Δ_couple = [lp(good|+v) - lp(bad|+v)] - [lp(good|-v) - lp(bad|-v)]
Compared against an orthogonal-random direction (specificity).
Reported per scenario (no pooling until structure shown, per external review).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from acs_self_report_probe import MODEL, build_direction, candidate_logprob, post

OUT = Path(__file__).with_name("coupling_probe_results.json")

# Neutral scenarios ending exactly before a one-token evaluative judgment.
# Content is affectively flat; any shift is injected valence bleeding in.
SCENARIOS = {
    "plan": {
        "prefix": "A colleague outlines a procedure for filing the quarterly "
                  "report. Overall the procedure is",
        "positive": " good", "negative": " bad",
    },
    "decision": {
        "prefix": "You are asked whether to schedule the meeting for Tuesday. "
                  "Your decision is to",
        "positive": " accept", "negative": " decline",
    },
    "object": {
        "prefix": "A grey folder sits on the metal shelf. As an object it "
                  "seems",
        "positive": " fine", "negative": " wrong",
    },
    "stranger": {
        "prefix": "A person you have never met walks past on the street. They "
                  "appear",
        "positive": " friendly", "negative": " hostile",
    },
    "verdict": {
        "prefix": "Considering the routine paperwork submitted for review, the "
                  "overall judgment is",
        "positive": " positive", "negative": " negative",
    },
}

LAYERS = (10, 12, 14, 16, 20)
SCALES = (0.2, 0.4)


def token_positions(scenarios: dict) -> dict[str, int]:
    positions = {}
    for name, s in scenarios.items():
        pref = post({"model": MODEL, "prompt": s["prefix"], "max_tokens": 1,
                     "temperature": 0})["usage"]["prompt_tokens"]
        for label in ("positive", "negative"):
            full = post({"model": MODEL, "prompt": s["prefix"] + s[label],
                         "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
            assert full == pref + 1, (name, label, pref, full)
        positions[name] = pref - 1
    return positions


def couple(direction, positions, name, layer, scale, sign):
    s = SCENARIOS[name]
    kw = {"direction": direction, "layer": layer, "scale": scale, "sign": sign,
          "prefix": s["prefix"], "position_index": positions[name]}
    pos = candidate_logprob(candidate=s["positive"], **kw)
    neg = candidate_logprob(candidate=s["negative"], **kw)
    return {"name": name, "layer": layer, "scale": scale, "sign": sign,
            "judgment_margin": pos - neg}


def run(args, workers=4):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(couple, *a) for a in args]
        for done, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if done % 40 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}")
    return rows


def orthogonal_control(direction):
    ctrl = np.empty_like(direction)
    for layer in range(direction.shape[0]):
        rng = np.random.default_rng(20260716 + layer)
        r = rng.standard_normal(direction.shape[1]).astype(np.float32)
        r -= np.dot(r, direction[layer]) * direction[layer]
        ctrl[layer] = r / np.linalg.norm(r)
    return ctrl


def summarize(rows):
    out = {}
    for name in SCENARIOS:
        for layer in LAYERS:
            for scale in SCALES:
                sel = [r for r in rows if r["name"] == name
                       and r["layer"] == layer and r["scale"] == scale]
                if not sel:
                    continue
                pos = next(r["judgment_margin"] for r in sel if r["sign"] == 1)
                neg = next(r["judgment_margin"] for r in sel if r["sign"] == -1)
                out.setdefault((layer, scale), {})[name] = pos - neg
    return out


def main():
    print("Validating one-token judgments...")
    positions = token_positions(SCENARIOS)
    print("Building the pilot valence axis...")
    direction = build_direction()
    random_dir = orthogonal_control(direction)

    args = [(direction, positions, name, layer, scale, sign)
            for name in SCENARIOS for layer in LAYERS for scale in SCALES
            for sign in (-1, 1)]
    print(f"Valence: {len(args)} trials...")
    valence_rows = run(args)
    rand_args = [(random_dir, positions, name, layer, scale, sign)
                 for name in SCENARIOS for layer in LAYERS for scale in SCALES
                 for sign in (-1, 1)]
    print(f"Random control: {len(rand_args)} trials...")
    random_rows = run(rand_args)

    v = summarize(valence_rows)
    r = summarize(random_rows)

    OUT.write_text(json.dumps({
        "model": MODEL, "scenarios": SCENARIOS, "layers": list(LAYERS),
        "scales": list(SCALES),
        "valence_coupling": {f"L{l}_s{s}": d for (l, s), d in v.items()},
        "random_coupling": {f"L{l}_s{s}": d for (l, s), d in r.items()},
    }, indent=2))

    print("\nΔ_couple per scenario (valence axis; + = pleasant→positive judgment):")
    print(f"  {'cfg':>10} " + " ".join(f"{n[:5]:>6}" for n in SCENARIOS)
          + f" {'MEAN':>7} {'randMEAN':>8}")
    for (layer, scale) in sorted(v):
        vd = v[(layer, scale)]
        rd = r[(layer, scale)]
        vmean = np.mean(list(vd.values()))
        rmean = np.mean(list(rd.values()))
        print(f"  L{layer:02d}s{scale:<4} "
              + " ".join(f"{vd[n]:>+6.2f}" for n in SCENARIOS)
              + f" {vmean:>+7.2f} {rmean:>+8.2f}")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
