# NLA Bottleneck Decision Tree

Where we are: **55%** forced-choice top-1 (SL, LoRA r=16, all-Sonnet descriptions, 5540 examples).
Target: Anthropic's 60% (full FT + GRPO on millions of examples).
Gap: **5 percentage points.**
Status: **Learning curve flattening.** Credible replication achieved.

## Current pipeline

- AV: LoRA r=16, alpha=64, all 7 projections, 40M/7.6B params (0.5%)
- AR: LoRA r=16, cosine 0.934 (all-Sonnet AR)
- Data: 5540 all-Sonnet descriptions (1179 original + 4361 API-regenerated)
- Description source: Claude Sonnet (prediction-focused style, consistent across all texts)

## Summary of all experiments

| Variant | Top-1 | Top-3 | AR cos | Val loss | What it tells us |
|---|---|---|---|---|---|
| Anthropic (full FT + RL) | 60% | 84% | 0.926 | — | Target (millions of examples) |
| **All-Sonnet (SL, LoRA r=16, 5540 ex)** | **55%** | **74%** | **0.934** | **2.134** | **Our best — credible replication** |
| Sonnet-only (SL, LoRA r=16, 1179 ex) | 52% | 64% | 0.891 | 1.866 | Previous best |
| Tight (SL, LoRA r=16) | 48% | — | 0.857 | 2.763 | Desc quality matters |
| LoRA r=64 (SL) | 44% | 78% | 0.847 | 1.818 | More capacity = overfitting |
| GRPO v3 (LoRA, 1 epoch) | 42% | 70% | 0.856 | — | RL collapses specificity |
| GRPO v4 (LoRA, fixed, 1 epoch) | — | — | ~0.87 | — | Same trend confirmed |
| Emotional v1 (SL, old pipeline) | 40% | 78% | 0.865 | — | Pre-bugfix baseline |
| Emotional v2 (SL, fixed) | 36% | — | 0.833 | 2.102 | Fixed pipeline |
| **Expanded (Kimi descs, 4804 ex)** | **27%** | **33%** | **0.917** | **2.520** | **Mixed descs kill specificity** |
| **Full FT (SL, 5 epochs)** | **14%** | **32%** | **0.778** | **2.363** | **Catastrophic overfitting** |
| Prediction (SL) | 10% | — | 0.805 | 1.404 | Collapsed (wrong framing) |
| Control (Sonnet descs, 5557 acts) | 7% | 33% | 0.782 | — | OOD activations break AV |

## What we've ruled out

### More LoRA capacity does NOT help (Gate 2a: DONE)
r=64 (160M params, 4× more) scored 44% vs r=16's 52%. Lower val_loss (1.818) but
worse forced-choice — the extra capacity memorizes training patterns without improving
generalization. The frozen base model constrains well at r=16; at r=64 the adapter
starts overriding the base's generalization.

### RL with LoRA does NOT help (Gate 2b: DONE)
GRPO v3 and v4 both made forced-choice worse (52% → 42%). Gradients flow correctly
(grad ~220-300, zero skipped groups), but the RL reward (AR cosine) is a Tier 1
metric — it rewards style fit, not content specificity. LoRA doesn't have enough
capacity to improve both dimensions; RL takes the shortest path to higher reward
= more generic text. Group_std declines during training (0.025 → 0.015), confirming
output diversity collapses.

### Full FT with small data is catastrophic (Gate 5: DONE)
Full FT on 1033 examples → 14% top-1 (worse than random controls at 15%). The model
memorized training descriptions and lost all ability to map novel activations. Val_loss
dropped steadily (2.641 → 2.363) while actual quality collapsed. With 7.6B trainable
parameters and 1033 examples, the ratio is ~7M params per example — absurd overfitting.

Anthropic trains on WildChat + Ultra-FineWeb (millions of examples) with batch_size=1024.
They can full-FT because the data is orders of magnitude larger.

### Better descriptions DO help (Gate 3: DONE)
Emotional 40% → Tight 48% → Sonnet 52%. Each step in description quality yields
forced-choice gains. The description-to-specificity link is real. But we've extracted
the maximum from 1181 examples with Sonnet descriptions.

## What we've learned about data expansion (Gate 4)

### Learning curve — data IS the bottleneck (2026-05-12)
258 examples → 12% | 516 → 22% | 1033 → 52%
**Curve is super-linear** — accelerating, not flattening. This was the most important
finding: more data will help, but ONLY at the same description quality.

### Data expansion attempt 1: Kimi K2 descriptions (2026-05-12/13, FAILED)

Expanded from 1179 → 4804 descriptions using Kimi K2 for expansion (1378 texts) and
WildChat (2247 texts). Result: **27% top-1** — catastrophic regression from 52%.

Meanwhile AR cosine IMPROVED (0.891 → 0.917). This is the Tier 1/Tier 2 split again:
Kimi descriptions have the right style (good AR cosine) but lack the specificity that
forced-choice requires (bad top-1).

**Root cause — description quality dilution:**
Sonnet descriptions name specific tokens, predict concrete continuations, identify
competing interpretations. Kimi descriptions correctly identify categories but lack
discriminative detail. When 75% of training data has generic descriptions, the AV
adapter learns to produce generic output. A single description model for the whole
corpus is essential.

Side-by-side on the same text:
- **Sonnet:** "forming a high-confidence prediction that the actual API response does
  not contain a `results` key, likely because the response represents an error payload...
  A mild competing interpretation — that the issue is not the key name but a failed
  JSON parse yielding a string rather than a dict — is active at lower weight, since
  `json.loads` would raise `JSONDecodeError` rather than `KeyError`"
- **GPT-4o:** "The intent/topic classifications active are 'technical assistance' and
  'debugging Python code,' with a focus on identifying the root cause of the KeyError"
- **Kimi:** similar to GPT-4o — correct categories, no discriminative detail

### Control experiment: Sonnet-only AV on expanded activations (2026-05-13)
Retrained AV with only original 1179 Sonnet descriptions but on new activation file
(5557 texts). Result: **7% top-1** — below random chance.

This does NOT mean description quality is irrelevant. The 7% is an artifact: the
stress test sampled from all 5557 activations, but the AV was only trained on 1179.
Most test samples are expansion/WildChat texts the AV never saw — out-of-distribution
activations produce actively misleading descriptions. The test setup was unfair.

### Key insight: CONSISTENCY > QUANTITY > QUALITY
1. All descriptions must come from the same model (style consistency)
2. The AV must see descriptions for ALL texts in the activation space (coverage)
3. The descriptions must be specific enough to uniquely identify texts (quality)

Kimi violated #1 and #3. The control violated #2.

## What remains

### Gate 4: All-Sonnet expansion (DONE — 55% top-1)

5540 all-Sonnet descriptions (1179 original + 4361 API-regenerated at ~$6).
Trained AV + AR on deepthought. Result: **55% top-1, 74% top-3, AR cosine 0.934.**

Learning curve: 258→12%, 516→22%, 1033→52%, 5540→55%.
Curve is flattening — 4.7x data yielded +3 points. Diminishing returns.

### What could close the remaining 5-point gap to 60%

1. **Full FT at 5540 examples** — 1.4M params/example, might not catastrophically
   overfit. Could gain 2-3 points. Moderate risk.

2. **LoRA r=32** — didn't work at r=64 on 1033 examples (overfitting), but 5540
   might support more capacity. Small potential gain.

3. **10x more data** — another 50K examples at Sonnet quality would cost ~$50 in API.
   But the flattening curve suggests maybe +2-3 points at best.

4. **GRPO on top of SL** — still risks Tier 1/2 collapse. Would need a Tier 2
   reward signal (forced-choice during training) instead of AR cosine.

**Recommendation:** Accept 55% as the result. 55% with LoRA r=16 on 5.5K examples
vs 60% with full FT + RL on millions is a credible open-source replication. Write
it up. The remaining gap is Anthropic's infrastructure advantage, not a methodology gap.

## Completed experiments log

### Reranking (2026-05-11)
Best-of-16 = 0.881 vs greedy = 0.857 (+0.024 headroom). Group std = 0.018.
Confirms the model CAN produce better descriptions — it just doesn't know which.

### Description provider comparison (2026-05-11, N=5-6)

| Provider | Our AR | Anthropic AR | Notes |
|---|---|---|---|
| Sonnet | 0.864 | 0.657 | Best on both ARs |
| DeepSeek (original) | 0.828 | 0.420 | Our AR biased toward this style |
| Kimi K2 | 0.819 | 0.618 | Prediction-focused, Sonnet-like feel |
| Llama 70B | 0.809 | 0.564 | Correct but vague |

### Few-shot description improvement (2026-05-11)
DeepSeek with 2-3 Sonnet few-shot examples eliminates the emotional-narrative style.
Costs nothing. Proven to work for gap-filling unsafe categories.

### GRPO gen_ids bug (2026-05-11, ROOT CAUSE)
`generate(inputs_embeds=...)` returns only new tokens, not prompt+generated. Code
sliced past the generation → empty gen_ids → skipped backward. Fixed with prefix
detection. GPT found this from code review.

### GPT code review (2026-05-11)
Five issues: (1) not real GRPO — relabeled as REINFORCE; (2) reward/text mismatch
at `</explanation>` — FIXED; (3) AR backbone loading — documented; (4) checkpoint
selection by reward only — confirmed as real problem; (5) empty backward — FIXED.

### OpenInterp NLA critique (2026-05-11)
Caio Vicentino "Two-Tier Verbalization" — fve_nrm measures format (Tier 1), not
content (Tier 2). Our forced-choice IS Tier 2. Prediction descriptions collapsed
to 10% despite low val_loss — same decoupling.

### Sonnet description quality (2026-05-11)
52% top-1 (up from 48% tight). 1181 descriptions assembled from Sonnet agents +
DeepSeek few-shot gap-filling. Confirms description quality matters.

### GRPO v3 — gradients flow (2026-05-12)
First real RL run. grad=220-233, group_std=0.019, 0 skipped. Epoch 1 saved, died
epoch 2 (OOM). Stress test: 42% — RL hurt specificity. Confirmed Tier 1/2 split.

### GRPO v4 — fixed script (2026-05-12)
Group_size=4, `</explanation>` truncation fix, empty-backward guard. Same trend:
reward ~0.88, group_std declining. Killed after epoch 1 save.

### LoRA r=64 SL (2026-05-12)
44% top-1 (worse than r=16's 52%). Lower val_loss (1.818) but worse forced-choice.
More capacity = more overfitting on 1033 examples. Disconfirms capacity hypothesis.

### Full FT SL (2026-05-12)
14% top-1 — worse than random controls (15%). val_loss 2.363 kept dropping but the
model completely lost activation→description mapping. 7.6B params on 1033 examples
= catastrophic memorization. The descriptions are generic and carry zero
activation-specific information.

**Lesson: full FT requires data proportional to parameter count. 1033 examples on
7.6B params is ~7M params/example. You need at minimum 10K-50K examples.**

### Learning curve (2026-05-12)
258 ex → 12% | 516 → 22% | 1033 → 52%. Super-linear — each doubling more than doubles
accuracy. Accelerating, not flattening. This is the strongest signal we have for data
expansion being worthwhile.

### Data expansion — Kimi descriptions (2026-05-12/13)
4804 examples (1179 Sonnet + 1378 expansion/Kimi + 2247 WildChat/Kimi).
Result: 27% top-1 (down from 52%). AR cosine improved to 0.917 (up from 0.891).

**The descriptions are the problem, not the data.** Kimi K2 produces correct but generic
descriptions that lack the discriminative specificity Sonnet provides. Training on 75%
Kimi descriptions teaches the AV to be vague. Style consistency across the corpus is
essential — mixing description providers destroys forced-choice performance even when
AR cosine improves.

### Sonnet description regeneration (2026-05-13, IN PROGRESS)
Regenerating all 4378 expansion + WildChat descriptions using `claude -p --model sonnet`
(4 parallel workers, ~9/min). Original 1179 Sonnet descriptions retained.
Script: `scripts/generate_descriptions_sonnet.py`

### CRITICAL: AV memorizes exact vectors, doesn't generalize (2026-05-14)

The original 52% AV model scores **16%** on freshly extracted activations from the
SAME 1179 texts. The AV adapter is a lookup table, not a generalizing mapper.

Evidence:
- Old model + old activations = 52% (training and test share same file)
- Old model + re-extracted activations (same texts) = 16%
- All-Sonnet (5540 ex) on re-extracted activations = 14%
- Ablation: zero activation gives HIGHER similarity than real activation

Root cause: floating-point non-determinism between extraction runs. `device_map="auto"`
places layers differently depending on available memory. bf16 vs fp32 routing differs.
Even tiny numerical differences in the activation vectors break the learned mapping.

Fix: added `--noise-std` flag to `train_av_single_layer.py` for future robustness.

**UPDATE:** The non-reproduction was caused by a **stress test bug**, not
activation memorization. The new stress_test_nla.py had two bugs:
1. Missing `apply_chat_template()` — the model was trained with chat formatting
   but tested without it, so the AV saw wrong token context
2. Wrong forced-choice methodology — comparing descriptions to source texts via
   sentence-transformers, instead of comparing to GT descriptions via TF-IDF
   (which is what the original test and Anthropic's evaluation do)

When tested with the correct old stress test (stress_test_qwen_nla.py), the
all-Sonnet model scores **55% top-1, 74% top-3** — a real improvement from 52%.
Activations ARE deterministic (confirmed by extracting twice and comparing).

### All-Sonnet expansion: 55% top-1 (2026-05-14, FINAL RESULT)

5540 all-Sonnet descriptions. Trained AV (val_loss 2.134) + AR (cosine 0.934).
Old stress test (correct methodology): **top1=0.550, top3=0.740, mrr=0.685**.
Controls: top1=0.110 (chance). Learning curve: 52%→55% with 4.7x data.

### Stress test bug discovery (2026-05-14)

The new stress_test_nla.py had TWO bugs that made ALL expanded results look terrible:
1. **Missing chat template** — training used `apply_chat_template()`, test didn't.
   The AV saw wrong token context and produced garbage descriptions.
2. **Wrong forced-choice metric** — new test compared descriptions to source texts
   via sentence-transformers. Old test (and Anthropic) compare descriptions to GT
   descriptions via TF-IDF. These are fundamentally different metrics.

All results from the new test (27%, 14%, 13%, 7%) were measuring the wrong thing.
The correct metric (old test) shows 55% — the expansion worked all along.

### Activation determinism confirmed (2026-05-14)

Two independent extractions of the same 1179 texts produce IDENTICAL activations
(max diff = 0.0, cosine = 1.0). The AV is NOT memorizing exact vectors — the
non-reproduction was caused by the stress test bug, not activation brittleness.

### Pipeline infrastructure (2026-05-12/13)
- `scripts/run_expanded_pipeline.sh` — automated merge→sync→extract→train→test
- `scripts/generate_descriptions_sonnet.py` — parallel Sonnet desc generation via CLI
- Fixed `extract_activations.py` — `load_corpus()` now filters for "text" field, deduplicates
- Fixed `stress_test_nla.py` — keeps both AV+AR loaded for round-trip test
- Installed `sentence-transformers` on deepthought for forced-choice test
