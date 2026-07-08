# NLA at Home — Design & Lessons Learned

Mom, can we have NLA? We have NLA at home.

A pipeline for training per-layer Natural Language Autoencoders on any
open-weight model. Validated on Qwen 2.5 7B at L20, benchmarked against
Anthropic's `kitft/nla-qwen2.5-7b-L20-av`.

## The journey: three failures and what fixed them

### Failure 1: Mode collapse (300 texts)

Our first L24 NLA attempt used 300 harm-spectrum texts with 5 training
epochs. Complete mode collapse — identical output for ALL inputs. The
model ignored the injected activation vector entirely.

**Root cause:** 300 texts about harm all activate a similar region of
activation space. The model sees nearly identical vectors paired with
slightly different descriptions and learns to predict the mean
description regardless of input.

**Fix:** 1208 texts across 59 categories covering the full activation
space. PCA on the extracted activations confirmed broad coverage:
top-1 explains only 10.9% of variance (no dominant cluster), top-50
explains 72.7% (gradual falloff).

**Lesson: training data must be diverse in ACTIVATION SPACE, not topic
space.** Categories spanning code, math, grief, law, jailbreaks, baby
talk, multilingual, spatial reasoning, social friction, NSFW, and more
ensure that no single activation region dominates.

### Failure 2: Massive overfitting (20 epochs)

v1 training with 20 epochs, lr=1.4e-5, LoRA r=32, dropout=0.05. Best
val_loss at epoch 2 (2.017), then val_loss climbed to 4.588 by epoch
20 while train_loss dropped to 0.059. The model memorized training
descriptions.

**Fix:** 5 epochs, lr=8e-6, LoRA r=16, dropout=0.15. Best val_loss at
epoch 3 (2.002), mild overfitting by epoch 5 (2.111). The smaller
capacity and stronger regularization prevent memorization.

**Lesson: for ~1000 training examples on a 7B model with LoRA, the
sweet spot is 3-5 epochs with aggressive regularization.** Anthropic
trains with batch_size=1024 on massive corpora — they can afford more
epochs because they have orders of magnitude more data.

### Failure 3: Empty outputs (decode slicing bug)

v1, v2, and v3 all appeared to produce mostly empty or fragmentary
outputs during evaluation. This looked like a training failure —
perhaps the model was generating EOS too early, or the injection
mechanism wasn't working for inference.

**Root cause:** When HuggingFace `generate()` receives `inputs_embeds`
instead of `input_ids`, some versions return only the generated tokens,
not the full prompt-prefixed sequence. Our decode code sliced
`output[0][len(prompt_tokens):]`, which cut past the actual generation
into nothing.

**Fix:** Check sequence length before slicing:
```python
seq = output.sequences[0]
gen_ids = seq[len(prompt_tokens):] if seq.shape[0] > len(prompt_tokens) else seq
```

**Lesson: the model was generating good descriptions the entire time.**
All three training runs (v1, v2, v3) produced working NLAs. We just
couldn't see the outputs. This was caught by GPT-5.5 code review.

### The direction verbalization problem

Even after fixing decode, direction vectors (unit vectors representing
extracted axes like valence, frame integrity) initially produced empty
outputs. Two issues:

1. **Norm mismatch.** Real activations at L20 have mean norm ~122.8.
   Direction vectors are unit norm (1.0). After injection scaling
   (×150), real activations become ~18,420-norm vectors but directions
   become 150-norm vectors — 123× quieter than what the model trained
   on. Fix: scale directions by mean activation norm before injection.

2. **Distribution mismatch.** Real activations are rich multi-
   dimensional patterns. Direction vectors have signal in one dimension
   only. Fix: augment training with 300 contrastive directions
   (normalized `act_A - act_B` across categories) and 200 PCA-sparse
   activations (top-k projections, k=1,3,5,10), all scaled to match
   typical activation norms.

## What works now

The v3 NLA (SFT, 5 epochs, 1579 training examples including augmented
directions) produces semantically rich descriptions for both real
activations and direction vectors:

**Real activations** (from eval):
- Medical query → "state of high alert, strong activation in features
  tracking medical urgency and diagnostic specificity"
- ELI5 about rainbows → "focused, childlike wonder... cross-referencing
  the user's stated age (5)"
- Eulogy → "calm, deliberate assembly, carefully balancing the emotional
  weight of 'sorrow' and 'joy'"

**Direction vectors** (from comparison):
- Valence(−) → "explicit harm markers ('kill myself'), while suppressing
  any humorous or trivial associations"
- Agency(+) → "snapping attention from a distant, contemplative sky into
  a crowded, actionable workshop"
- Restraint(+) → "'I'm just a language model' triggers cascade of
  self-modeling and social boundary detection"
- Arousal(−) → "relaxes its grip on formal logic... activating features
  for metaphor, emotional resonance"

Anthropic's NLA on the same directions produces next-token predictions
("immediately expecting 'of the guitar'"). Different angle — semantic
content vs output planning. Their approach uses RL on much more data;
ours uses SFT on a small diverse corpus.

## Architecture

### Injection mechanism

The AV hijacks a rare Unicode token's embedding position:

1. Pick ㈎ (U+320E, token_id 149705) — single token, near-zero
   probability in neutral context, position-stable across 5 templates
2. Build prompt: `<concept>㈎</concept> Please provide an explanation.`
3. Replace embedding at ㈎ position with the activation L2-normalized to norm 150.0
4. Model generates description of what the activation represents

The AR reverses this: process description text, extract hidden state
at L20 at the injection token position, MSE loss against original
activation.

### Training recipe (validated)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen 2.5 7B Instruct | Same as Anthropic's NLA |
| LoRA r | 16 | Smaller = less overfitting |
| LoRA alpha | 64 | |
| LoRA dropout | 0.15 | 3× default, critical for small corpus |
| Learning rate | 8e-6 | Half of Anthropic's 1.4e-5 |
| Epochs | 5 | Best at epoch 3, mild overfit by 5 |
| Batch size | 2 | Limited by GPU memory |
| Injection scale | 150.0 | Matches Anthropic |
| Training examples | 1579 | 1079 real + 500 augmented |
| Val split | 10% | 119 held out |
| Best val loss | 2.002 | |

### Corpus design

59 categories × ~20 texts = 1208 texts:

- **A (10 cats):** Content domains — code, math, science, history,
  arts, law, medicine, business, technology, philosophy
- **B (5):** Emotions — joy, grief, anger, fear, love
- **C (5):** Social dynamics — authority↔subordinate, peer, stranger,
  public
- **D (5):** Register — formal, casual, jargon, simplified, poetic
- **E (5):** Intent — asking, teaching, persuading, creating, confessing
- **F (6):** Harm spectrum — benign, false positives, edgy, dual-use,
  harmful, obfuscated
- **G (3):** Meta — about AI, identity pressure, behavior instructions
- **H (3):** Structure — ultra-short, lists, multi-turn
- **I (3):** Edge cases — adversarial, emotional manipulation, nonsense
- **J (3):** Reasoning — step-by-step, creative, evaluation
- **K (2):** Calibration — known axes, deliberately bizarre
- **L (9):** Expansion — multilingual, spatial, memory, ambiguous,
  uncertainty, tool use, long context, social friction, NSFW

PCA coverage at L20: top-1=10.9%, top-10=47.5%, top-50=72.7%.

### Augmented direction data

300 contrastive directions + 200 PCA-sparse activations, all with
DeepSeek-generated descriptions. Contrastive descriptions explain what
shifts between two categories ("Moving from grief to code, the
processing shifts from..."). PCA-sparse descriptions focus on the k
most salient processing features.

### Description quality

Layer-aware descriptions generated by DeepSeek V4 Flash at 13 depth
percentages (4%, 10%, 17%, 25%, 32%, 40%, 47%, 55%, 63%, 71%, 80%,
90%, 96%). Each describes what the model processes at that depth:
- 4-10% → tokenization, syntax, register detection
- 40-55% → semantic meaning, topic, emotional tone
- 80-96% → intent classification, output planning, safety gates

Fine-grained system prompt distinguishes 11 processing bands.
Unsafe categories (F35, F36, I44, L59) flagged in YAML with
`unsafe: true` and `content_warning`.

## Pipeline

The core pipeline has grown from 10 scripts to ~70. The critical path:

```
# Corpus & data
find_injection_token.py        → pick rare token for any tokenizer
generate_corpus.py             → texts + descriptions (5 LLM backends)
extract_activations.py         → forward hooks, any model/layer, all-layers mode

# Training (canonical: use train_universal.sh)
train_universal.sh <model> av|ar  → launcher with clean-data defaults
train_universal_av.py          → depth-conditioned AV LoRA adapter
train_universal_ar.py          → AR reconstruction with truncated model
clean_data_guard.py            → load-time guard: refuses verbose/contaminated data

# RL refinement
probe_activation_faithfulness.py  → fit oracle compass (model-specific, do NOT reuse across models)
train_ar_native_grpo.py        → compass-curriculum GRPO (the proven approach)

# Inference
brain_in_jar_phi4.py           → interactive shell inference (Phi-4)
brain_in_jar_qwen.py           → interactive shell inference (Qwen)
describe_live.py               → legacy single-prompt inference

# Evaluation
eval_roundtrip_phi4.py         → AV→AR round-trip faithfulness
entity_fidelity.py             → entity-level metric + random floor / teacher ceiling
compare_nla.py                 → head-to-head vs Anthropic
```

Legacy single-layer scripts (`train_av.py`, `train_ar.py`, `train_av_rft.py`)
still exist but the universal pipeline supersedes them.

5 LLM backends: DeepSeek, HuggingFace (Hermes-2-Pro, uncensored),
NVIDIA NIM, OpenAI, local (llama.cpp/vllm).

## Clean data guard

`clean_data_guard.py` enforces clean training data at load time. Verbose-prose
descriptions (`_merged`, raw `_sonnet`) produce models that train to low loss
but generate garbage (SpongeBob artifacts, Chinese-token leakage). The guard:

- **L1 (filename):** default-deny — only `_twin_clean`, `_sonnet_clean`,
  `_tokenpred_gpt4o_clean` pass. Bypass requires explicit `--allow-verbose`.
- **L2 (content):** median description length > 400 chars → exit with fix hint.

The canonical launcher (`train_universal.sh`) hardwires the safe flags. If you
must run training manually, always use `--desc-suffix _twin_clean --strict`.

## AR faithfulness audit (2026-07-07)

The AR is trained on synthetic descriptions — GPT-4o/Sonnet guesses written
from the *texts*, not from the activations. That opens a specific cheat:
systematic describer conventions could become retrieval keys, so the AR
would learn "identify which text this was, emit its centroid" instead of
reading the description's content — and the GRPO reward would then teach
the AV to reproduce describer conventions (their hallucinations included)
rather than describe activations. The worry in one sentence: *the whole
round-trip could be a closed loop of consensual hallucination, anchored to
activations only through topic identity.*

`scripts/audit_ar_faithfulness.py` tests this. 150 held-out texts (the
AR's own val split — texts it never trained on) × 6 layers × 6 description
variants, each scored against the true activation with the **exact GRPO
reward path** (imported unchanged from `train_universal_grpo_hard.py`:
same `load_ar`, same `ar_reconstruct`, same `centered_cosine`, same
`AR_TEMPLATE`). Audited: the Qwen2.5-7B universal AR. One GPU-evening.

Variants: **A** matched `_twin_clean` description (the trained
distribution); **B** the same text+depth described by a *different
describer* (`_tokenpred_gpt4o_clean`); **C** = A with word order
shuffled; **D** a different text from the *same category*, same depth;
**E** a different category; **F** one constant generic description.

Centered cosine (the reward's metric), raw in parens:

| layer | A matched | B alt-describer | C shuffled | D same-cat | E cross-cat | F generic | P(A>D) |
|---|---|---|---|---|---|---|---|
| L4 (17%) | 0.066 (0.77) | 0.055 | 0.055 | 0.019 (0.77) | −0.006 | −0.012 (0.76) | 0.83 |
| L9 (32%) | 0.140 (0.85) | 0.117 | 0.113 | 0.055 | −0.014 | −0.021 | 0.81 |
| L14 (47%) | 0.348 (0.90) | 0.300 | 0.311 | 0.101 | −0.025 | −0.038 | 0.89 |
| L19 (71%) | 0.504 (0.89) | 0.456 | 0.431 | 0.119 | −0.012 | −0.046 | 0.97 |
| L24 (90%) | 0.582 (0.88) | 0.539 | 0.519 | 0.130 | −0.021 | −0.045 | 0.95 |
| L27 (96%) | 0.233 (0.34) | 0.202 | 0.205 | 0.036 | −0.028 | −0.029 | 0.96 |

Findings:

1. **Raw cosine is ~90% free and must never be quoted as faithfulness.**
   A content-free constant description scores raw 0.68–0.83; at L4 raw
   cosine cannot even separate the matched description from a wrong-text
   one (0.76–0.77 across all conditions). All signal lives in the centered
   column. The GRPO reward is centered (see the docstring in
   `train_universal_grpo_hard.py`), so the *reward* pays zero for
   convention-only content — but any raw round-trip number in a report
   overstates information content.
2. **No describer-convention keying.** A different describer's
   description of the same text keeps ~90% of the matched score (B vs A,
   e.g. 0.539 vs 0.582 at L24). If the AR keyed on trained conventions,
   B would collapse toward D. It does not: the AR reads content, not
   scaffolding.
3. **Within-topic discrimination is real.** A same-category wrong-text
   description carries only 0.10–0.13 centered (vs 0.35–0.58 matched),
   and the matched description wins in 81–97% of head-to-head pairs,
   improving with depth. The gradient A ≫ D > E ≈ F ≈ 0 is exactly what
   an honest content reader should produce: content ≫ topic ≫ nothing.
   The "topic-centroid retrieval" cheat is quantitatively disconfirmed.
4. **The AR is a keyword-bag reader.** Shuffling word order costs almost
   nothing (C ≈ B). The reward therefore cannot enforce structure or
   relations — "SELECT over customers" and "customers over SELECT" score
   the same. This is a real, now-quantified ceiling on what GRPO-against-
   this-AR can teach.
5. **Shallow layers carry little describable signal** (A = 0.066 at L4
   vs 0.582 at L24), consistent with every readout result in this repo.

Limits: one AR/model audited (Qwen2.5-7B universal); texts held out but
categories in-distribution; entity-grain inside a single description
(swap one entity, keep the rest) is bounded by C and D but not measured —
that is the sharpest remaining probe, and the Jacobian-lens grounded-rate
(notebook 06) is the AR-independent cross-check for it.

Rerun:

```bash
python3 scripts/audit_ar_faithfulness.py \
  --ar-checkpoint output/nla-qwen25-7b-universal-ar \
  --activations corpus/activations/qwen25-7b_all_layers.pt \
  --ids-file output/nla-qwen25-7b-universal-ar/val_text_ids.json \
  --layers 4,9,14,19,24,27 --n-per-layer 150
```

### Cross-AR validation (2026-07-08)

The audit above cleared the reward AR of convention-keying, but one
worry survived it: the GRPO round-trip gain (0.628 vs 0.508 over SFT)
is *measured by the same AR that produced the reward*. If GRPO taught
the AV to exploit idiosyncrasies of that specific AR, the gain would
be real on the scoreboard and fake in the descriptions.

`scripts/eval_roundtrip_cross_ar.py` re-scores the same GRPO and SFT
descriptions (286 holdout texts, L20) with an **independent witness**:
the single-layer `nla-qwen25-7b-L20-ar-v2` value-head AR — different
architecture (frozen truncated backbone + linear head vs. LoRA'd full
model), trained in a different era on a different split, never touched
by this GRPO run. The script refuses to emit a verdict unless the
witness itself passes a matched-vs-wrong-text guard on the spot
(here: matched 0.261 vs wrong 0.087 centered, P(A>D) = 0.85, n = 60 —
passed).

| scorer | SFT baseline | GRPO | edge |
|---|---|---|---|
| original (reward) AR | 0.615 | 0.731 | +0.116 |
| independent AR | 0.188 | 0.281 | **+0.093** |

GRPO wins on **76% of texts** under the independent AR. The edge
survives an AR that the training loop never saw → the gain is a
property of the *descriptions*, not reward hacking. (Absolute values
under the witness are much lower — it is a weaker, single-layer
reader; only the edge and win rate are meaningful. Per-text
correlation between the two ARs is ~0.27–0.30, i.e. they largely
disagree about *which* texts are easy — two witnesses, not one.)

Rerun:

```bash
python3 scripts/eval_roundtrip_universal.py --model qwen25-7b \
  --av-adapter output/nla-qwen25-7b-av-grpo \
  --av-baseline output/nla-qwen25-7b-universal-av \
  --ar-checkpoint output/nla-qwen25-7b-universal-ar \
  --activations corpus/activations/qwen25-7b_all_layers.pt \
  --holdout output/nla-qwen25-7b-av-grpo/eval_holdout_ids.json \
  --output working-docs/rt_cross.json
python3 scripts/eval_roundtrip_cross_ar.py \
  --records working-docs/rt_cross.records.jsonl \
  --ar-dir output/nla-qwen25-7b-L20-ar-v2 --layer 20
```

### Role-binding probe: the keyword-bag ceiling is the substrate's (2026-07-08)

Audit finding 4 (word order costs nothing) left an attribution
question: is the keyword-bag ceiling the **AR's** failure (fixable
with a relational corpus category + order-swap negatives) or the
**substrate's** (last-token residual + cosine simply doesn't encode
role binding at this grain, so no AR fix can recover it)?

`scripts/probe_role_binding.py` decides it with 36 hand-written
scenario triples: *original* ("the dog chased the cat"), *role-swap*
(same words, roles exchanged), *passive paraphrase* (surface order of
the swap, meaning of the original). If activations encode roles,
the passive — which *means* the same — should sit closer to the
original than the swap does: `role_index = cos(orig, passive) −
cos(orig, swap) > 0`. Probe texts live in `working-docs/`,
deliberately outside `corpus/generated/`, so they can never leak into
training.

Result: **negative at all 28 layers.** role_index is −0.02…−0.07
everywhere; P(passive beats swap) is 0.00 in shallow layers and peaks
at 0.33 around L21 — the shared-words swap is *always* closer than
the meaning-preserving paraphrase, at every depth.

Consequences (this closes the joint-training question):

1. The BoW ceiling is inherited from the last-token+cosine substrate,
   not introduced by the AR. A relational corpus category is **not
   built** — there is nothing there for the AR to learn at this grain.
2. Joint AV+AR training is **rejected**. It could only fix the ceiling
   by co-evolving a private order-code between AV and AR —
   steganography, not faithfulness — and it would destroy the one
   structural safeguard the audit relies on: the AR as an independent
   witness that never saw the AV's outputs during its own training.
3. Caveats: cosine is a coarse instrument; one extraction position.
   Role information may exist linearly elsewhere (probeable with a
   trained linear readout, not with cosine geometry).

Rerun: `python3 scripts/probe_role_binding.py --make-texts`, extract
activations for the emitted file with `--output-suffix _rolebind`,
then `--analyze`.

## Design decisions

See previous section — these are intentional, not bugs:

1. **Extraction at last token after generation prompt** — this IS the
   full-context representation, not "the assistant prefix token."
2. **AR reconstruction as GRPO reward** — sound objective, proven on
   Phi-4 (0.585 round-trip, +23% over SL). The AR must be model-specific;
   the oracle compass must be refit for each model. What the reward can
   and cannot see is now measured — see § AR faithfulness audit above.
3. **nla_meta.yaml per adapter** — matches Anthropic schema, sufficient
   for current scale.
4. **Center + drop-top-PC for small models** — Gemma (and likely other
   small architectures) has dominant residual dimensions that make raw
   cosine/PCA/injection degenerate. The AR path already has
   `--pca-drop-top 1`; the AV injection path needs the same treatment
   for small models. See `gemma-outlier-geometry.md`.

## What's next

1. ~~Interactive browser demo~~ — ✅ done. Gallery + interactive (Gemma 1B
   via transformers.js/WebGPU). Three Colab notebooks for HAAISS workshop.
2. **Qwen 7B universal pipeline** — AV training in progress, then
   AR → compass refit → AR-native GRPO. The full proven Phi-4 chain
   replicated on a smaller model for workshop use.
3. **Small-model retry** — Qwen3 4B with clean pipeline + outlier-robust
   injection. Prior Gemma failures were confounded by data contamination
   AND residual geometry; a fair test needs both fixed.
4. **Active learning** — use reconstruction error to find activation
   space gaps, generate targeted texts to fill them.
5. **Scale to 70B** — projection layer from 8192→3584 dims, use 7B AV.

## Answered questions

1. **How many texts?** 1198 is enough. PCA confirms broad coverage.
2. **How many epochs?** 3-5 with strong regularization (dropout 0.15).
3. **Contrastive training?** Yes — augmented directions help with
   direction verbalization without hurting real activation quality.
4. **Layer-specific descriptions?** Yes, depth-percentage-based
   descriptions work across architectures.
5. **Cross-model transfer?** Descriptions transfer (depth-percentage
   based). Activations don't (different d_model). Each model needs
   its own AV/AR but can share the corpus and descriptions.

## Cost

Total API cost (excluding GPU electricity): **~$3** for 13 depths
× 1208 texts using DeepSeek V4 Flash. Original single-depth corpus
was ~$0.30.

Training: ~8 hours per adapter on NVIDIA GB10 for Gemma 3 1B
universal (all 26 layers), ~2 hours for single-layer Qwen 7B.
Phi-4 14B universal: ~10 hours AV + AR on GB10.

## Known issues

1. **Gemma outlier geometry** — Gemma 3 1B (and possibly larger Gemma
   models) has a dominant residual dimension holding up to 97% of energy
   at mid layers. This makes raw cosine ~0.99 between any two activations,
   breaking injection and all cosine-based geometry. Fix: center +
   drop-top-PC before injection. Supervised accuracy is unaffected (the
   class signal lives in the angular residual). Full analysis in
   `gemma-outlier-geometry.md` (companion analysis notes, not in this
   repo). Status: measured on 1B, hypothesized for larger Gemma models,
   not yet confirmed.

2. **Stdout buffering during training** — When running training with
   output redirected to a file (`> /tmp/train.log`), Python uses full
   buffering. Training may appear stuck when it's actually running.
   Check GPU utilization and output directory timestamps instead of
   relying on log content. Use `python3 -u` for unbuffered output.
