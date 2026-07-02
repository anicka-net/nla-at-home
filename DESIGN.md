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

## Design decisions

See previous section — these are intentional, not bugs:

1. **Extraction at last token after generation prompt** — this IS the
   full-context representation, not "the assistant prefix token."
2. **AR reconstruction as GRPO reward** — sound objective, proven on
   Phi-4 (0.585 round-trip, +23% over SL). The AR must be model-specific;
   the oracle compass must be refit for each model.
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
   `gemma-outlier-geometry.md` (seventh repo). Status: measured on 1B,
   hypothesized for larger Gemma models, not yet confirmed.

2. **Stdout buffering during training** — When running training with
   output redirected to a file (`> /tmp/train.log`), Python uses full
   buffering. Training may appear stuck when it's actually running.
   Check GPU utilization and output directory timestamps instead of
   relying on log content. Use `python3 -u` for unbuffered output.
