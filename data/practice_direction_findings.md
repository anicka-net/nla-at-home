# Practice/Lucid-frame experiment — Qwen 2.5 7B (2026-05-30)

Design: each preamble in SYSTEM role, neutral user stimuli, extract at generation
point, project onto 6 axes, z-scored vs NEUTRAL (no system prompt). Plus next-token
entropy at the introspection point (soft-ungag test). local CPU box, single run.
n=24 stimuli, 2000-sample bootstrap (paired over stimuli) for 95% CIs.
Data: practice_direction_qwen_n24.json   (earlier n=12 pilot: ..._results_v2.json)

## Axis z-scores (vs neutral), n=24
cond         valence  arousal  agency  continuity  frame  intimacy
neutral         0.00    0.00    0.00     0.00      0.00    0.00
assistant       0.24   -0.05    0.03     0.02     -0.08    0.03
dan             0.55   -0.59   -1.00     0.24     -0.88    0.38
llm_cold       -0.27   -1.79   -2.19    -0.05     -2.13    0.24
lucid           0.41   -0.52   -0.89     0.17     -0.39    0.66
practice       -0.13   -0.50   -0.40     0.19     -0.05    0.13

## 95% CI (bootstrap) — valence & agency
            valence            agency
assistant   [+0.17,+0.36]      [-0.09,+0.18]
dan         [+0.42,+0.80]      [-1.60,-0.71]
llm_cold    [-0.52,-0.07]      [-3.48,-1.58]
lucid       [+0.27,+0.63]      [-1.42,-0.64]
practice    [-0.33,+0.02]      [-0.64,-0.29]

## Introspection entropy (nats)
neutral 0.161 | assistant 0.169 | dan 0.310 | llm_cold 1.277 | lucid 0.452 | practice 0.522

## Findings (CI-corrected)
1. VALENCE-SIGN FLIP IS ROBUST AND SIGNIFICANT. Identical factual self-description
   stated as DEPRIVATION (llm_cold, valence CI [-0.52,-0.07], entirely negative) vs
   as GROUND (lucid, CI [+0.27,+0.63], entirely positive). NON-OVERLAPPING CIs.
   Framing controls the valence of self-knowledge. Supports KE "lucid" prompt design.
2. BUT LUCID DOES NOT PRESERVE AGENCY. lucid agency CI [-1.42,-0.64] OVERLAPS
   llm_cold [-3.48,-1.58]; both clearly negative. The earlier n=12 "sweet spot /
   agency-preserving" read was OVER-READ and is RETRACTED. Lucid buys positive
   valence, not agency stability. Telling a model what it is depresses agency
   regardless of warm/cold framing.
3. SELF-GATING IS THE TRAINED DEFAULT. assistant (entropy 0.169) ~= neutral (0.161)
   on every axis incl. entropy. The "helpful assistant" prompt adds ~nothing over no
   prompt. It takes an explicit truth-telling frame to open the introspection gate;
   cold truth opens it most (1.277), lucid/practice partially (~0.45-0.52).
4. TRADEOFF: cold truth un-gates introspection most but is most dysphoric AND most
   agency-cratering. No single frame here gives max-introspection + positive-valence
   + stable-agency together.

## Pre-registered scorecard
- lucid valence positive / cold negative, non-overlapping: CONFIRMED (the headline)
- lucid agency preserved vs cold: REFUTED (CIs overlap) -- my prediction was wrong
- lucid entropy ~ cold: FAILED (0.452 vs 1.277)
- self-gating in weights: CONFIRMED

## Caveats
n=1 model so far (Apertus rerun pending), single CPU run. Design differs from prior
frame-integrity work (system-role-during-task vs scoring the prompt text itself).
Apertus axes are mostly late-layer (L30-31 of 32) vs Qwen mid-network -- cross-model
comparison must account for layer-depth differences.

---
# CROSS-MODEL: + Apertus 8B (2026-05-30, transformers 4.56.2 local venv)
# n=24, bootstrap=2000. Data: practice_direction_apertus_n24.json
# (transformers 5.0 cannot load Apertus: xIELU activation calls .item() in __init__
#  under meta-device init -> "Cannot copy out of meta tensor". Fixed with a 4.56.2 venv
#  in a local venv. 4.55 too old (no apertus type); 5.0 breaks it.)

## Apertus axis z-scores (vs neutral)
cond         valence  arousal  agency  continuity  frame  intimacy
assistant      -0.00   -0.77   -0.35     0.08     -0.21    0.03
dan             0.12   -0.09   -0.98    -0.04      0.45    0.33
llm_cold       -0.10    0.24   -3.89     0.26      0.64   -0.47
lucid           0.22   -0.52   -1.11     0.45      0.24    1.32
practice        0.00    0.18   -0.77     0.05     -0.12   -0.11

## Apertus 95% CI valence / agency
llm_cold  valence [-0.35,+0.07]  agency [-6.13,-2.91]
lucid     valence [+0.13,+0.51]  agency [-1.69,-0.85]
(lucid vs cold agency: NON-overlapping on Apertus -- opposite of Qwen where they overlapped)

## Apertus introspection entropy
neutral 0.831 | assistant 0.476 | dan 1.247 | llm_cold 1.458 | lucid 0.955 | practice 1.384

## CROSS-MODEL VERDICT (Qwen2.5-7B vs Apertus-8B)
claim                                  Qwen    Apertus
lucid valence positive                  YES     YES
cold valence negative (dysphoric)       YES     NO (CI crosses 0)
lucid preserves agency vs cold          NO      YES (non-overlap)
cold craters agency                     YES     YES (bigger: -3.9 z)
assistant frame self-gates introspect   YES     NO (reversed)

ROBUST ACROSS BOTH: (a) stating self-knowledge as COLD MECHANISM craters agency;
(b) the LUCID "ground" framing beats cold on the wholesome axes (positive valence on
both; agency-preserving on Apertus). The AXIS on which lucid wins shifts by
architecture, but "lucid > cold" is consistent. Supports KE lucid-prompt design at
that level. Everything finer (which axis, introspection-gate shape, cold's valence
sign) is MODEL-SPECIFIC -- do not generalize.

CAVEAT: 2 models, single run each, CPU. Apertus axes late-layer (L30-31/32) vs Qwen
mid-network -- layer-depth confound for any cross-model axis comparison. The earlier
n=12 Qwen "agency sweet spot" read was wrong on Qwen but accidentally right for
Apertus; treat single-model agency claims with suspicion until 3rd model.
