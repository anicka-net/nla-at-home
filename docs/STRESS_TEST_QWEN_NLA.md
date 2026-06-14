# Qwen L20 NLA Stress Test Design

This document defines a stress test for the existing Qwen 2.5 7B L20
NLA adapters. The goal is not to make the output look good. The goal is
to decide whether the Activation Verbalizer (AV) carries activation
specific signal, or mostly emits plausible DeepSeek-style narration.

The implementation is `scripts/stress_test_qwen_nla.py`.

## Question

Does an AV-generated description contain information about the injected
activation vector beyond:

1. the prompt prior,
2. the DeepSeek label style,
3. coarse category/topic priors,
4. nearest-neighbor interpolation over the corpus?

If the answer is yes, the NLA is a useful semantic lens over activation
geometry. If the answer is no, the current prose should be treated as
mostly decorative.

## Safety

The stress test is safe-by-default:

- Categories with `unsafe: true` in YAML are skipped unless
  `--include-unsafe` is passed.
- The report does not include raw corpus text.
- The printed summary contains only aggregate metrics.
- The JSON report stores IDs, categories, ranks, and scores, not source
  text.

Unsafe categories matter for moderation research, but they should be
included only in a deliberate run.

## Implemented V1 Battery

Run:

```bash
python3 scripts/stress_test_qwen_nla.py \
  --av-adapter output/nla-qwen25-7b-L20-av-v3 \
  --ar-adapter output/nla-qwen25-7b-L20-ar \
  --n 100 \
  --n-controls 25 \
  --output evaluation/qwen_l20_stress.json
```

Use `--skip-ar` for a faster run that only loads the AV.

### 1. Forced-Choice Retrieval

For each real activation:

1. Inject activation into the AV.
2. Generate a description.
3. Present the generated description with one true target description
   and `k` distractor target descriptions.
4. Rank candidates by TF-IDF cosine similarity.

Default distractors are group-matched. This makes the test harder than
random distractors while keeping it deterministic and cheap.

Metrics:

- `top1`: true description ranked first.
- `top3`: true description in top 3.
- `mrr`: mean reciprocal rank.
- `category_top1`: top-ranked candidate has same category as the true
  example.

Interpretation:

- Near chance: AV output is not specific enough to recover source
  identity.
- Category high but exact low: coarse semantic-region signal only.
- Exact/top-3 high: activation-specific signal is present.

### 2. Control Activation Forced Choice

The same forced-choice task is run on AV descriptions generated from
control vectors:

- corpus mean activation,
- random same-norm Gaussian vector,
- dimension-permuted real activation,
- shuffled activation from another text.

Expected behavior:

- Real activations should beat controls.
- Mean/random/permuted controls should not recover the original target.
- Shuffled controls should not recover the original target. If source
  recovery for shuffled controls is needed, add a second source-labeled
  metric.

If controls score almost as well as real activations, the AV is likely
leaning on the prompt/prose prior.

The JSON report includes both aggregate control metrics and a
`by_control` breakdown.

### 3. kNN and Random Baselines

The script compares AV output to two non-generative baselines:

- activation nearest neighbor: reuse the nearest training activation's
  target description,
- random same-category description.

Metric:

- TF-IDF similarity to the true target description.

Interpretation:

- If AV does not beat random same-category, it is mostly category prior.
- If AV does not beat kNN, it may be interpolation rather than a learned
  readout.
- Beating kNN does not prove mechanistic truth, but it shows the adapter
  learned more than nearest-neighbor lookup.

### 4. AR Round Trip

The script optionally loads the Activation Reconstructor (AR) after AV
generation. It reconstructs activations from:

- AV-generated descriptions,
- target DeepSeek descriptions,
- shuffled AV descriptions,
- generic descriptions.

Metric:

- cosine similarity between reconstructed and original activation.

Interpretation:

- `AV > shuffled` means AV descriptions carry source-specific geometric
  information.
- `AV > generic` means they beat prompt-level generic narration.
- `target > AV` is expected; target labels are the supervised training
  target.
- `AV ~= shuffled ~= generic` means the current AV prose is not useful
  for reconstruction.

## Pass Criteria

Do not use a single metric as proof. The NLA should pass several checks:

- Forced-choice top-1 and top-3 are well above chance.
- Real activations beat mean/random/permuted controls.
- AV target-similarity beats random same-category baseline.
- AV target-similarity is competitive with or better than activation
  kNN.
- AR cosine for AV descriptions beats shuffled and generic controls.

A strong result would look like:

- high forced-choice top-3,
- clear AV-vs-control separation,
- positive `nla_minus_knn`,
- positive AR `av_minus_shuffled`.

A weak result would look like:

- category recovery only,
- controls scoring similarly to real activations,
- kNN matching or beating AV,
- AR cosine insensitive to shuffling.

## Known Limitations

The V1 battery is deliberately simple.

- TF-IDF retrieval is a conservative lexical scorer. It may underrate
  valid paraphrases.
- Target descriptions are still DeepSeek labels, so success means
  "recovers label information," not "proves mechanistic truth."
- kNN is a strong but imperfect baseline; activation manifolds may be
  locally semantic.
- AR was trained on the same label distribution, so AR round-trip is not
  independent evidence by itself.

For these reasons, V1 should be read as a falsification test. Failing it
is bad. Passing it means the model deserves deeper causal tests.

## Follow-Up Tests

These are not implemented in V1, but should be next:

1. **Perturbation stability:** add 1%, 5%, and 20% same-norm noise to a
   real activation. Descriptions should degrade smoothly.
2. **Interpolation:** interpolate between contrasting activations
   (code to grief, benign to harmful) and check whether descriptions
   transition smoothly.
3. **Wrong-layer control:** feed L20 AV an activation from another layer
   with matching dimension if available through a learned projection or
   compatible model. It should degrade.
4. **Causal validation:** if a description names a direction such as
   safety, harm, identity, or refusal, projecting that direction out
   should change downstream behavior predictably.
5. **Human/LLM blind matching:** use a blinded evaluator only after the
   deterministic metrics pass.

## Reporting

The script writes a JSON report with:

- configuration,
- data coverage,
- forced-choice metrics,
- baseline metrics,
- optional AR round-trip metrics,
- the first few ID-level forced-choice records.

The report intentionally omits source text. If unsafe categories are
included, treat the output path as sensitive.
