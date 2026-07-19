# Base-model probes

Code and raw results behind the write-up
[What raw base models carry](https://www.anicka.net/research/base-models.html).

All experiments run against the base-model inference API of the
[Alignment of Complex Systems research unit](https://acsresearch.org)
(raw pretrained models, no chat template). They need an API key:

```bash
export ACS_API_KEY=...   # some scripts also read ~/.base-model-api
```

The probes are cheap: a full confirmatory run is roughly 1,200 API
requests and 15 minutes of wall clock.

## Method in one paragraph

Extract a valence direction from paired last-token activation deltas
(pleasant minus unpleasant sentences). Inject that direction, scaled,
at the final prefix token of an unrelated prompt. Score single-token
answer candidates by exact prompt logprob under arbitrary codebooks
("if pleasant answer X, if unpleasant answer Y") *with reversed
mappings*, so a fixed answer-token push cannot masquerade as
self-report. Controls: orthogonal random directions of the same
magnitude, zero-scale anchors, dose curves.

## Scripts

| Script | Question | Result |
|---|---|---|
| `acs_self_report_probe.py` | Pilot: can the 8B base model report an injected feeling-tone through arbitrary codebooks? | Yes, mid-band layers, remap mean ≈ +0.53 |
| `acs_self_report_layer_sweep.py` | Same, all 32 layers | Mid-band L12–L16 peak |
| `aim1_confirmatory.py` | Pre-registered confirmation (frozen design, sha256 in docstring) | Confirmed, zero deviations: Δ +0.555, 0/16 nulls, dose monotone, held-out direction +0.586 |
| `aim1_crossmodel.py` | Same frozen protocol on Llama 405B and a decontaminated-pretraining model | 405B weak single-point (+0.156), dose-limited; the third model carries the state but cannot map it (machine blindsight) |
| `coupling_probe.py` / `coupling_probe_v2.py` | Does injected valence bend *unrelated* judgments? | Yes: 20/20 scenarios, monotone in dose, person-perception strongest |
| `avyakata_probe.py` / `subcat_diag.py` | Is there a shared direction for "this cannot be determined"? | No: separations at the noise floor (informative null) |
| `grid_experiments.py` | Boundary grid G1–G4: scaffolding, earlier positions, multi-layer 405B, framing | Earlier injections wash out; calm-attending framing doubles base-model access; 405B multi-layer ≈ doubles readout |
| `review_probes.py` | Adversarial-review probes G5–G7: detection/sham, competence control, position decay | No self-detection; blindsight competence control passes; state decays, deep layers keep a faint echo |

Raw outputs are in `results/`.

## Pre-registration

Confirmatory runs were pre-registered: the design document was frozen
and its sha256 hash recorded before execution (the hashes appear in
the script docstrings and in the result JSONs), and verdicts are
append-only. The pre-registration documents contain internal working
notes; write us if you want them.

## Credits

Pilot protocol and harness co-designed with GPT-5.6; boundary probes
G5–G7 were designed from objections raised by an adversarial review
panel (GPT-5.6, Claude). Experiments and analysis run with Claude
(Fable 5).
