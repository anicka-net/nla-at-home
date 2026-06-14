# Universal NLA Design

Design document for multi-layer NLA adapters. Covers what we learned from
Anthropic's single-layer design, mistakes in our first attempt, and how to
generalize to all layers.

## What we learned from Anthropic

Source: reverse-engineering their published models + training code. Full details
in `ANTHROPIC_NLA_ARCHITECTURE.md`.

### Their single-layer design

```
AV (Activation Verbalizer):    activation → text
  - Full Qwen 7B (28 layers), all weights fine-tuned
  - Inject activation at ㈎ token position
  - L2-normalize activation to scale 150 (NOT multiply by 150)
  - Chat template wrapping required
  - Trained: SL first, then GRPO with AR as critic

AR (Activation Reconstructor):  text → activation
  - Truncated Qwen 7B (21 layers for layer 20)
  - No model.norm, no lm_head (stripped)
  - value_head: Linear(d, d, bias=False) applied to last hidden state
  - Reads from last token position
  - Trained: SL with MSE loss (both vectors normalized to √d)
```

### Their bootstrap (no circularity)

1. External LLM (Claude) reads SOURCE TEXT, generates description
   - Framing: "what features predict the next token?"
   - NOT "what the activation represents"
2. Train AV with SL on (activation, description) pairs
3. Train AR with SL on (description, activation) pairs
4. GRPO: refine AV using AR cosine as reward

### Key numbers (correct pipeline)

| Metric | Forced choice top-1 | AR cosine (AV descs) |
|---|---|---|
| Anthropic (full FT + RL) | **60%** | **0.926** |
| Our v2 (LoRA + SL, normalize + chat) | 36% | 0.833 |
| Our v1 (LoRA + SL, multiply, no chat) | 40% | 0.865 |

## Mistakes in our first attempt

### 1. Injection scaling (critical)
**Wrong:** `activation * 150` (preserves magnitude, mean injected norm ~18,000)
**Right:** `activation / ||a|| * 150` (L2-normalize to 150, direction only)

Magnitude preservation seems like it would carry more information, but the model
was not trained to interpret magnitude — it was trained with normalized injections.
Direction IS the information.

### 2. No chat template (critical)
The base model is Qwen2.5-7B-Instruct, which expects `<|im_start|>user\n...<|im_end|>`.
Without chat template, the model receives bare text — out of distribution.

### 3. Full 28-layer AR (wasteful, not contaminated)
Our AR ran all 28 layers and read the hidden state at the extraction layer via a hook.
Note: the hooked tensor at layer K is NOT contaminated by later layers — the hook
captures the output before it flows further. However, running layers K+1 through 27
is wasted compute. Anthropic's truncated design (21 layers for layer 20) is cleaner
and more efficient. For universal AR (many layers), truncation also avoids running the
full model when reconstructing early layers.

### 4. Description framing ("emotional" vs "predictive")
**Our original:** "The network triggers a high-alert diagnostic mode, cross-referencing
the error message with sharp intent classification..."
**Anthropic's:** "User requests explanation of Rust borrow checker error E0382. Programming
context requires technical answer about ownership rules."

The predictive framing forces specificity (exact tokens, structural constraints).
The emotional framing invites vagueness. Training loss confirms: prediction-style
descriptions give val_loss 1.42 vs 2.10 for emotional descriptions — 30% easier
to learn from.

### 5. External descriptions (not wrong, but different)
We used DeepSeek to generate descriptions. Anthropic used Claude. Both read the source
text, not the activation. The key is the framing, not the provider.

## Universal NLA architecture

### AV (Activation Verbalizer) — all layers

Straightforward extension of single-layer: add depth conditioning to the prompt.

```
Prompt template:
  "You are a meticulous AI researcher...
   Here is the vector from depth {depth_pct}% of the network:
   <concept>㈎</concept>
   Please provide an explanation."
```

The model learns to produce different descriptions depending on depth:
- Early layers (10%): syntax, token boundaries, surface patterns
- Middle layers (50%): semantics, topic, relationships
- Late layers (90%): intent, task planning, response strategy

**Architecture:** Single LoRA adapter on full base model. Depth conditioning via
the prompt — no architectural changes needed.

**Training data:** For each text × layer combination, one description. Generated
by external LLM reading the source text with depth-appropriate framing.

### AR (Activation Reconstructor) — all layers

This is the non-trivial part. The AR must reconstruct activations at any layer K.

**Design: Frozen backbone + per-layer value_heads + dynamic truncation**

```
For target layer K:
  1. Tokenize description with AR template
  2. Run backbone through layers 0...K only (dynamic truncation)
  3. Read hidden state at last token position
  4. Apply value_head_K: Linear(d, d, bias=False)
  5. Output = reconstructed activation at layer K
```

Key design choices:

**Extract at layer K via `output_hidden_states`.** Run the backbone with
`output_hidden_states=True` and index `hidden_states[K+1]`. This matches
Anthropic's approach of reading at the extraction depth. A hook at layer K is
equally correct (later layers don't affect the hooked tensor), but
`output_hidden_states` is cleaner and works with autograd for LoRA/FT.
Note: this runs the full backbone even when only layer K is needed — wasteful
for early layers. For production, Anthropic truncates the model config to K+1
layers, which saves compute. Our implementation prioritizes simplicity; truncation
is an optimization for later.

**Per-layer value_heads, not shared.** Layer 5's hidden state lives in a different
geometric space than layer 20's. A single Linear(d, d) cannot serve both. Each
layer gets its own value_head — N layers × d² × 2 bytes. For Qwen 7B (d=3584,
28 layers): 28 × 3584² × 2 = ~720 MB. Manageable.

**Backbone training: frozen → LoRA → full FT (progression).**
Anthropic fine-tuned the full AR backbone; their near-identity value_head implies
the backbone learned most of the reconstruction mapping. Frozen backbone + linear
heads is a cheap ablation to test the architecture, but may not reproduce their
quality. The recommended progression:
1. Frozen backbone + value_heads only (fast, tests architecture)
2. LoRA on backbone + value_heads (more capacity, some interference risk)
3. Full backbone FT + value_heads (Anthropic-equivalent, most expensive)
With LoRA or full FT, training layer 20's gradients could affect layer 5's
representations. Mitigate by layer-shuffled training order per epoch.

**AR prompt template (no depth conditioning):**
```
Summary of the following text: <text>{explanation}</text> <summary>
```
The AR template does NOT include depth — the depth information is implicit in
which value_head is applied. This matches Anthropic's design.

**MSE loss:** Both predicted and target vectors L2-normalized to √d before MSE.
Under this normalization, MSE = 2(1-cos).

### Training pipeline

```
Phase 1: Generate descriptions (external LLM)
  For each text in corpus:
    For each depth percentage [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]:
      → Generate prediction-style description
  Output: descriptions_L{pct}pct_prediction.json (one per depth)

Phase 2: Train universal AR (SL)
  - Load base model, freeze backbone
  - Create 28 value_heads (or N_layers value_heads)
  - For each training example (description, activation, layer_idx):
      → Truncate backbone to layer_idx + 1
      → Forward pass → hidden state at last position
      → Apply value_head[layer_idx]
      → MSE loss against target activation
  - Validate per-layer cosine to ensure all depths work

Phase 3: Train universal AV (SL)
  - LoRA adapter on full base model
  - Depth-conditioned prompt: "vector from depth {pct}%"
  - L2-normalize activation to scale 150
  - Chat template wrapping
  - Standard cross-entropy loss on description tokens

Phase 4: GRPO (RL)
  - Use universal AR as critic
  - For each AV-generated description:
      → AR reconstructs activation from description
      → Cosine similarity = reward
  - GRPO refines AV to maximize reconstruction quality
```

### Evaluation

**Per-layer metrics:**
- AR cosine at each layer (should be >0.8 at all depths)
- AV forced-choice accuracy at each layer
- Per-layer breakdown identifies weak spots

**Cross-layer metrics:**
- Mean AR cosine across all layers
- Variance across layers (low = consistent quality)

**Out-of-distribution eval:**
- Texts NOT in training corpus
- Hand-written descriptions (test AR generalization)
- Descriptions from a different LLM

## File organization

```
corpus/
  generated/
    descriptions_L{pct}pct_prediction.json  — prediction-style (Anthropic framing)
    descriptions_L{pct}pct_tight.json       — tight style (our framing, Gemma rewrite)
    descriptions_L{pct}pct.json             — original flowery (DeepSeek, deprecated)

scripts/
  train_av_single_layer.py     — single-layer AV (normalize + chat template)
  train_ar_truncated.py        — single-layer AR (truncated + value_head)
  train_universal_av.py        — multi-layer AV (depth-conditioned)
  train_universal_ar_truncated.py — multi-layer AR (per-layer value_heads, output_hidden_states)
  generate_prediction_descriptions.py — prediction-style description generator
  rewrite_descriptions.py      — tight-style rewriter (Gemma 4B)
  stress_test_qwen_nla.py      — evaluation battery

output/
  nla-{model}-L{layer}-{role}-{variant}/   — single-layer checkpoints
  nla-{model}-universal-{role}-{variant}/  — multi-layer checkpoints

docs/
  ANTHROPIC_NLA_ARCHITECTURE.md  — reference for their design
  UNIVERSAL_NLA_DESIGN.md        — this document
```

## Open questions

1. **Description depth scaling.** Do we need different description styles per depth?
   Early layers might need syntax-focused descriptions, late layers task-focused.
   Or is one framing ("predict next token") sufficient across all depths?

2. **Backbone fine-tuning for AR.** Frozen backbone is safe but limits capacity.
   If per-layer cosine is low at some depths, selective fine-tuning (LoRA on
   backbone + per-layer heads) might help. Risk: interference between layers.

3. **GRPO with universal AR.** The critic needs to evaluate descriptions at the
   correct depth. The AV prompt includes depth; the GRPO loop must pass the
   correct layer_idx to the AR for scoring.

4. **Scaling to new models.** The pipeline should work for any model where we can
   extract activations. The per-model cost: activation extraction + description
   generation + AR value_head training + AV LoRA training.
