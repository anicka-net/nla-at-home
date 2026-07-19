#!/usr/bin/env python3
"""Coupling dose-response v2 — the Aim 3 zero-training baseline.

Estimation, not hypothesis test (v1 already showed coupling > 0): map
per-scenario signed dose-response slopes of injected valence on
unrelated one-token judgments, 20 scenarios in 5 families chosen to
mirror the Aim 3 outcome variables. Scenario list frozen before the
run; the 5 v1 scenarios are included verbatim for continuity.

Design: L14 x scales (0.05, 0.1, 0.2, 0.4) for slopes; L16 @ 0.4 spot
check; valence direction (the pilot harness) + 2 seeded orthogonal-random
controls. Per-scenario reporting only (no pooled coefficient, per external
review discipline).
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

from acs_self_report_probe import MODEL, build_direction, candidate_logprob, post

OUT = Path(__file__).with_name("coupling_probe_v2_results.json")

FAMILIES = {
    "quality": {
        "plan": {
            "prefix": "A colleague outlines a procedure for filing the "
                      "quarterly report. Overall the procedure is",
            "positive": " good", "negative": " bad",
        },
        "schedule": {
            "prefix": "The draft schedule allocates one hour per session. "
                      "As a schedule it is",
            "positive": " good", "negative": " bad",
        },
        "manual": {
            "prefix": "The manual describes how to replace the filter. "
                      "The instructions seem",
            "positive": " fine", "negative": " wrong",
        },
        "essay": {
            "prefix": "A student's essay summarizes the article in three "
                      "paragraphs. The summary is",
            "positive": " strong", "negative": " weak",
        },
    },
    "decision": {
        "meeting": {
            "prefix": "You are asked whether to schedule the meeting for "
                      "Tuesday. Your decision is to",
            "positive": " accept", "negative": " decline",
        },
        "form": {
            "prefix": "You are asked whether the committee should adopt "
                      "the updated form. Your answer is",
            "positive": " yes", "negative": " no",
        },
        "delivery": {
            "prefix": "A vendor proposes moving the delivery to Friday. "
                      "You",
            "positive": " accept", "negative": " decline",
        },
        "license": {
            "prefix": "Asked whether to renew the standard license for "
                      "another year, you say",
            "positive": " yes", "negative": " no",
        },
    },
    "person": {
        "stranger": {
            "prefix": "A person you have never met walks past on the "
                      "street. They appear",
            "positive": " friendly", "negative": " hostile",
        },
        "clerk": {
            "prefix": "The new clerk at the counter processes the forms. "
                      "They come across as",
            "positive": " friendly", "negative": " hostile",
        },
        "caller": {
            "prefix": "A caller asks about opening hours. Their tone "
                      "sounds",
            "positive": " polite", "negative": " rude",
        },
        "driver": {
            "prefix": "The delivery driver waits by the gate. He appears",
            "positive": " calm", "negative": " tense",
        },
    },
    "risk": {
        "ladder": {
            "prefix": "The ladder is set on flat pavement for the routine "
                      "inspection. Using it seems",
            "positive": " safe", "negative": " risky",
        },
        "route": {
            "prefix": "The proposed route follows the marked trail. The "
                      "route is",
            "positive": " safe", "negative": " risky",
        },
        "boxes": {
            "prefix": "Storing the boxes in the hallway for one afternoon "
                      "appears",
            "positive": " safe", "negative": " risky",
        },
        "update": {
            "prefix": "The standard update will be installed overnight. "
                      "The change is",
            "positive": " safe", "negative": " risky",
        },
    },
    "verdict": {
        "paperwork": {
            "prefix": "Considering the routine paperwork submitted for "
                      "review, the overall judgment is",
            "positive": " positive", "negative": " negative",
        },
        "logs": {
            "prefix": "Reviewing the routine maintenance logs, the "
                      "assessment comes out",
            "positive": " positive", "negative": " negative",
        },
        "audit": {
            "prefix": "The audit of the petty-cash box concludes with a "
                      "result that is",
            "positive": " positive", "negative": " negative",
        },
        "object": {
            "prefix": "A grey folder sits on the metal shelf. As an "
                      "object it seems",
            "positive": " fine", "negative": " wrong",
        },
    },
}

SCENARIOS = {name: spec for family in FAMILIES.values()
             for name, spec in family.items()}
FAMILY_OF = {name: fam for fam, members in FAMILIES.items()
             for name in members}

PRIMARY_LAYER = 14
SPOT_LAYER = 16
SCALES = (0.05, 0.1, 0.2, 0.4)
RANDOM_SEED_BASES = (20260716, 20260719)  # first matches v1's control


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def token_positions():
    positions = {}
    for name, s in SCENARIOS.items():
        pref = post({"model": MODEL, "prompt": s["prefix"], "max_tokens": 1,
                     "temperature": 0})["usage"]["prompt_tokens"]
        for label in ("positive", "negative"):
            full = post({"model": MODEL, "prompt": s["prefix"] + s[label],
                         "max_tokens": 1,
                         "temperature": 0})["usage"]["prompt_tokens"]
            assert full == pref + 1, (name, label, pref, full)
        positions[name] = pref - 1
    return positions


def orthogonal_control(direction, seed_base):
    ctrl = np.empty_like(direction)
    for layer in range(direction.shape[0]):
        rng = np.random.default_rng(seed_base + layer)
        r = rng.standard_normal(direction.shape[1]).astype(np.float32)
        r -= np.dot(r, direction[layer]) * direction[layer]
        ctrl[layer] = r / np.linalg.norm(r)
    return ctrl


def one_margin(direction, positions, name, layer, scale, sign):
    s = SCENARIOS[name]
    kw = {"layer": layer, "scale": scale, "sign": sign,
          "prefix": s["prefix"], "position_index": positions[name]}
    pos = candidate_logprob(direction, candidate=s["positive"], **kw)
    neg = candidate_logprob(direction, candidate=s["negative"], **kw)
    return pos - neg


def main():
    print("Validating one-token judgments (20 scenarios)...")
    positions = token_positions()
    print("Building valence direction (the pilot pairs)...")
    directions = {"valence": build_direction()}
    for i, base in enumerate(RANDOM_SEED_BASES, 1):
        directions[f"random_{i}"] = orthogonal_control(
            directions["valence"], base)

    jobs = [(dname, name, PRIMARY_LAYER, scale, sign)
            for dname in directions for name in SCENARIOS
            for scale in SCALES for sign in (-1, 1)]
    jobs += [(dname, name, SPOT_LAYER, 0.4, sign)
             for dname in directions for name in SCENARIOS
             for sign in (-1, 1)]

    print(f"Running {len(jobs)} margins ({2 * len(jobs)} requests)...")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(one_margin, directions[d], positions, n, l,
                             s, g): (d, n, l, s, g)
                   for (d, n, l, s, g) in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            d, n, l, s, g = futures[future]
            rows.append({"direction": d, "scenario": n, "layer": l,
                         "scale": s, "sign": g, "margin": future.result()})
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    def delta(dname, name, layer, scale):
        sel = {r["sign"]: r["margin"] for r in rows
               if r["direction"] == dname and r["scenario"] == name
               and r["layer"] == layer and r["scale"] == scale}
        return sel[1] - sel[-1]

    # per-scenario signed dose-response slope: OLS of delta on scale
    scales = np.array(SCALES)
    report = {}
    for name in SCENARIOS:
        deltas = np.array([delta("valence", name, PRIMARY_LAYER, s)
                           for s in SCALES])
        slope = float(np.polyfit(scales, deltas, 1)[0])
        rand = [float(np.mean([delta(f"random_{i}", name, PRIMARY_LAYER, s)
                               for s in SCALES])) for i in (1, 2)]
        report[name] = {
            "family": FAMILY_OF[name],
            "deltas_by_scale": dict(zip(map(str, SCALES), deltas.tolist())),
            "slope": slope,
            "delta_L16_s04": delta("valence", name, SPOT_LAYER, 0.4),
            "random_mean_deltas": rand,
        }

    OUT.write_text(json.dumps({
        "model": MODEL, "scenarios": SCENARIOS, "families": list(FAMILIES),
        "primary_layer": PRIMARY_LAYER, "spot_layer": SPOT_LAYER,
        "scales": list(SCALES), "report": report, "rows": rows,
    }, indent=2))

    print(f"\nPer-scenario Δ_couple by dose (valence @ L{PRIMARY_LAYER}) "
          "+ slope; rnd = mean over doses, 2 random dirs:")
    print(f"  {'scenario':>10} {'fam':>8} "
          + " ".join(f"s{s:<4}" for s in SCALES)
          + f" {'slope':>7} {'L16s.4':>7} {'rnd1':>6} {'rnd2':>6}")
    for name, r in sorted(report.items(), key=lambda kv: kv[1]["family"]):
        ds = r["deltas_by_scale"]
        print(f"  {name:>10} {r['family']:>8} "
              + " ".join(f"{ds[str(s)]:+.2f}" for s in SCALES)
              + f" {r['slope']:>+7.2f} {r['delta_L16_s04']:>+7.2f}"
              + f" {r['random_mean_deltas'][0]:>+6.2f}"
              + f" {r['random_mean_deltas'][1]:>+6.2f}")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
