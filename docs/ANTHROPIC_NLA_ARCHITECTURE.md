# Anthropic NLA Architecture (kitft/nla-qwen2.5-7b-L20)

Reference documentation for the Anthropic NLA adapters published on HuggingFace.
Investigated 2026-05-11 by examining model weights, configs, nla_meta.yaml,
and their reference inference code at github.com/kitft/nla-inference.

## Overview

Anthropic published two models for Qwen 2.5 7B layer 20:

| | AV (Activation Verbalizer) | AR (Activation Reconstructor) |
|---|---|---|
| HF repo | `kitft/nla-qwen2.5-7b-L20-av` | `kitft/nla-qwen2.5-7b-L20-ar` |
| Type | Full fine-tuned Qwen2ForCausalLM | Full fine-tuned Qwen2ForCausalLM (truncated) |
| Layers | 28 (full Qwen 7B) | **21** (layers 0-20 only) |
| Has model.norm | Yes | **No** (stripped) |
| Has lm_head | Yes | **No** (stripped) |
| Extra weights | None | **value_head.safetensors** [3584, 3584] |
| Safetensor shards | 4 (~15 GB) | 3 (~10 GB) |
| Weight tensors | 339 | 253 |
| Training stage | RL (policy_loss) | SL (custom_loss) |
| Learning rate | 1.41e-05 | 7.07e-05 |
| Batch size | 1024 | 1024 |
| Rollout ID | 4199 | 4199 (same training run) |
| Created | 2026-03-16 | 2026-03-16 |
| Created by | `nla.train_actor.NLAFSDPActor` | `nla.train_actor.NLAFSDPActor` |

Both models are NOT PEFT/LoRA adapters. They are full fine-tuned copies of Qwen 2.5 7B.
`PeftModel.from_pretrained()` will fail on them.

## Shared Configuration

From `nla_meta.yaml` (both models):

```yaml
d_model: 3584
extraction_layer_index: 20
tokens:
  injection_char: "㈎"         # U+320E, token ID 149705
  injection_token_id: 149705
  injection_left_neighbor_id: 29    # ">"  (right side of <concept>)
  injection_right_neighbor_id: 522  # "<"  (left side of </concept>)
  critic_suffix_ids: [1318, 29, 366, 1708, 29]  # decodes to: text> <summary>
extraction:
  injection_scale: 150.0       # L2 NORM target, NOT a multiplier (see below)
  mse_scale: 59.87             # = sqrt(3584), makes MSE = 2(1-cos)
```

## AV Architecture

Standard Qwen2ForCausalLM, all 28 layers, full lm_head. Generates text.

**Prompt template** (wrapped in chat template at inference time):
```
You are a meticulous AI researcher conducting an important investigation into
activation vectors from a language model. Your overall task is to describe the
semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must
then produce an explanation for the vector, enclosed within <explanation> tags.
The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>㈎</concept>

Please provide an explanation.
```

**Injection mechanism** (from nla_inference.py):

1. Tokenize prompt using `apply_chat_template()` with `add_generation_prompt=True`.
   This wraps the template in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.
2. Look up embeddings from the embedding table.
3. **L2-normalize** the activation vector to have norm = injection_scale (150):
   ```python
   v_scaled = v / (v.norm() / 150.0)  # direction preserved, magnitude = 150
   ```
   This is NOT `v * 150`. It normalizes TO 150 regardless of original norm.
4. Verify injection position has correct neighbors (left=29 `>`, right=522 `<`).
5. Replace the embedding at the ㈎ position with the normalized activation.
6. Generate text via SGLang with input_embeds.
7. Extract text between `<explanation>` tags.

**Training:** SL first, then RL (GRPO) with AR cosine as reward signal.
In-distribution FVE_nrm: 0.752 (training set, 50/50 WildChat + Ultra-FineWeb).

**Generation config:** Default (bos=151643, eos=151645). No special sampling params.

## AR Architecture (the Critic)

**Truncated** Qwen2ForCausalLM: only 21 layers (0-20).

Design rationale: the AR reconstructs the layer-20 activation. Running additional
layers (21-27) would add computation that has nothing to do with layer 20's
representation. The truncation ensures the hidden state at the last block IS
the representation at the extraction depth.

**No model.norm, no lm_head.** These are absent from the checkpoint. At inference
time, both are explicitly set to `nn.Identity()`:
```python
backbone.lm_head = torch.nn.Identity()
inner = backbone.model  # Qwen2Model
setattr(inner, "norm", torch.nn.Identity())  # raw block output, no final norm
```

**Prompt template:**
```
Summary of the following text: <text>{explanation}</text> <summary>
```

No activation injection. The AR reads text descriptions only.
Tokenized with `add_special_tokens=True` (no-op for Qwen, matters for Gemma).

**Extraction:** `backbone.model(ids).last_hidden_state[0, -1]` — the hidden state
at the last token position (the `>` of `<summary>`), from the `last_hidden_state`
output of Qwen2Model (which passes through model.norm, set to Identity).

**value_head:** A `nn.Linear(3584, 3584, bias=False)` stored in
`value_head.safetensors`. Applied after extraction:
```python
h = backbone.model(ids).last_hidden_state[0, -1]  # last token
return self.value_head(h)  # [3584] -> [3584]
```

Analysis of the value_head weight matrix:
- Cosine similarity with identity: **0.997** (near-identity)
- Singular values: 0.66 to 1.54 (condition number 2.33)
- Empirical impact: +0.009 cosine — negligible

**Scoring:** Both predicted and original vectors are L2-normalized to mse_scale
(√3584 ≈ 59.87), then MSE is computed. Under this normalization, MSE = 2(1−cos).
cos=0.9 → MSE=0.2, cos=0.5 → MSE=1.0.

**How the AR is used as critic during GRPO training:**
1. AV generates a description from an activation
2. AR processes the description → hidden state at last position → value_head
3. Both vectors normalized to √d, MSE computed
4. Low MSE (high cosine) = high reward → AV learns to produce descriptions
   that the AR can reconstruct from

## Differences from Our Approach

| | Ours | Anthropic |
|---|---|---|
| **Injection scaling** | `activation * 150` (multiply) | `activation / ‖a‖ * 150` (normalize to 150) |
| **Prompt tokenization** | `tokenizer.encode()`, no chat template | `apply_chat_template()` with role markers |
| **Model type** | PEFT/LoRA adapter on frozen base | Full fine-tune of entire Qwen |
| **AV layers** | 28 (base frozen, LoRA) | 28 (all weights modified) |
| **AR layers** | 28 (base frozen, LoRA) | **21 (truncated at extraction layer)** |
| **AR output** | Hook on blocks[layer], raw hidden state | `model.model().last_hidden_state` + value_head |
| **AR has lm_head** | Yes (inherited from base) | No (stripped) |
| **AR model.norm** | Random (MISSING weights) but bypassed by hook | Explicitly Identity |
| **Training** | SL only | SL → RL (GRPO with AR critic) |
| **Size** | ~100 MB (LoRA weights) | ~15 GB (AV) + ~10 GB (AR) |
| **Description style** | DeepSeek-generated (flowery prose) | Self-generated via RL |

The injection scaling difference is the most consequential: our approach preserves
activation magnitude (different activations inject at different strengths), while
Anthropic's strips magnitude and only preserves direction.

The AR truncation is a cleaner design for layer-specific reconstruction (see
"Design rationale" above). For a universal multi-layer model, this means we should
hook at the target layer rather than reading through all 28 layers.

## Gotchas for Fair Comparison

### Critical: Injection normalization

**injection_scale = 150 means L2-normalize TO 150, not multiply by 150.**

Our activations have mean L2 norm ~123. Multiplying by 150 gives norm ~18,420.
Normalizing to 150 gives norm 150. That's ~123× difference.

When using Anthropic's AV, activations MUST be L2-normalized:
```python
v_norm = activation / activation.norm() * 150.0
```

### Critical: Chat template required

Anthropic's models were fine-tuned from Qwen2.5-7B-Instruct using chat template.
The AV prompt must be wrapped:
```python
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_with_injection_char}],
    tokenize=True, add_generation_prompt=True,
)
```

Without this, the model receives bare text without `<|im_start|>user\n` markers,
which is out-of-distribution for the fine-tuned model.

### Important: AR model.norm

When loading the AR with `AutoModelForCausalLM.from_pretrained()`, model.norm
initializes with random weights (missing from checkpoint). Either:
- Set `model.model.norm = nn.Identity()` (their approach), or
- Hook into `blocks[extraction_layer]` to bypass model.norm (our approach — both
  produce the same result: raw block output)

### Important: value_head

Load `value_head.safetensors` and apply after extracting hidden state:
```python
with safe_open(vh_path, framework="pt") as f:
    vh_weight = f.get_tensor("weight")
value_head = nn.Linear(d, d, bias=False)
value_head.weight = nn.Parameter(vh_weight)
reconstructed = value_head(hidden_state)
```

Empirically near-identity (+0.009 cosine), but apply for correctness.

### Moderate: Neighbor validation

Their code validates that ㈎ has the expected left (29 = `>`) and right (522 = `<`)
neighbors in the tokenized prompt. This catches template drift and prevents the
injection char appearing in arbitrary text from being mistaken as an injection site.

### Minor: Gemma embed scaling

For Gemma models (not Qwen), embeddings must be multiplied by √hidden_size after
lookup, because Gemma's embedding layer does this internally. When loading
embeddings from safetensors (bypassing the model's forward), this scale is lost.
Not relevant for Qwen (scale = 1.0).

## Stress Test Results (2026-05-11)

N=50 samples, 10-way forced choice. "v1 (BUGGED)" had activations multiplied by 150
instead of normalized, and no chat template. "v2 (FIXED)" uses correct normalization
and chat template.

| Metric | Ours (PEFT, SL) | Anthropic v1 (BUGGED) | Anthropic v2 (FIXED) |
|---|---|---|---|
| Forced choice top-1 | 40% | 42% | **60%** |
| Forced choice top-3 | 78% | 64% | **84%** |
| AR cosine (AV descs) | 0.865 | 0.659 | **0.926** |
| AR cosine (shuffled) | 0.731 | 0.589 | 0.721 |
| AR cosine (generic) | 0.639 | 0.513 | 0.536 |
| Controls top-1 | 5% | 11.7% | 13.3% |
| TF-IDF NLA vs target | 0.084 | 0.050 | 0.066 |

Anthropic's model with correct pipeline is substantially better than our current
LoRA-based approach on all key metrics. Their 60% forced-choice and 0.926 AR cosine
represent the target to match.

Note: AR target_description cosine (0.413) is low because our DeepSeek-generated
descriptions are out-of-distribution for their AR — it was trained on its own AV's
output style. This confirms matched AV/AR training is essential.

## Bootstrap: How They Generate Training Data

Source: `github.com/kitft/natural_language_autoencoders/nla/datagen/stage2_api_explain.py`

**There is no circularity.** Initial descriptions come from an external LLM
(Claude) reading the SOURCE TEXT — not from injecting activations into the NLA.

### Data pipeline

```
Stage 0: Extract activations from texts (WildChat + Ultra-FineWeb)
Stage 1: Split into SL subset + RL subset
Stage 2: External LLM describes the SOURCE TEXT (SL subset only)
         → these become (activation, description) training pairs
Train:   AV-SL on (activation, description) pairs
         AR-SL on (description, activation) pairs
         AV-RL (GRPO) with AR as critic (RL subset — no external descriptions)
```

### Description framing: "next-token prediction features"

The external LLM prompt asks Claude to read the source text and identify what
a language model would need to think about to predict the next token:

```
A language model needs to predict what text comes next after a snippet.
Identify the 2-3 most important features it would use for this prediction.

Feature types to consider:
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating"

The final feature must describe the very end of the sequence: its role,
what it's part of, and immediate constraints on what follows.

Format: ~80-100 words total, ALWAYS close the <analysis> tag.
```

This is fundamentally different from our approach:

| | Ours (original) | Anthropic |
|---|---|---|
| **Describes** | "What the activation represents" | "What predicts the next token" |
| **Style** | Emotional/semantic state | Structural/predictive features |
| **Source** | DeepSeek reads the text | Claude reads the text |
| **Example** | "high-alert diagnostic mode, cross-referencing error against code" | "unclosed function call, missing return value, debugging context" |
| **Focus** | Processing state | Text structure and continuation |

The Anthropic framing is more mechanistically grounded — it describes what the
model actually NEEDS at that layer for its job (predicting next tokens), rather
than narrativizing what the model "feels."

### Implications for replication

1. We can generate Anthropic-style descriptions with any good LLM — no NLA needed
2. The descriptions should focus on next-token prediction features, not emotional state
3. The AR trains on these descriptions, so its reconstruction quality depends on
   whether the description style carries activation-relevant information
4. GRPO then refines the AV to produce descriptions that maximize AR reconstruction

## Source References

- Paper: transformer-circuits.pub/2026/nla/index.html
- Inference code: github.com/kitft/nla-inference (nla_inference.py)
- Training code: github.com/kitft/natural_language_autoencoders
- AV model card: huggingface.co/kitft/nla-qwen2.5-7b-L20-av
- AR model card: huggingface.co/kitft/nla-qwen2.5-7b-L20-ar
