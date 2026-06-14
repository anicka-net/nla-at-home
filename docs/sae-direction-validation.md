# SAE Direction Validation

## What this does

When we extract a direction from the residual stream (valence, frame integrity, arousal, etc.), we're claiming that direction corresponds to a real computational feature of the model. But extraction via contrastive mean-difference can produce directions that *look* clean from the extraction angle while not corresponding to how the model actually computes.

SAE (Sparse Autoencoder) validation tests this claim from the outside. An SAE trained on a model's activations learns a dictionary of ~131K sparse features — the "natural basis" of the model's computation. If our extracted direction decomposes cleanly into a few SAE features with interpretable labels, the direction is real. If it fragments into hundreds of unrelated features, it may be an artifact.

## How it works

### Phase 1: Decomposition

The SAE has a decoder matrix `W_dec` of shape `[n_features, d_model]`. Each row is a learned feature direction. We compute the cosine similarity between our extracted direction and every SAE decoder column:

```
cos_sims[i] = cosine(our_direction, W_dec[i])
```

This tells us which SAE features "point in the same direction" as our axis.

**Key metrics:**
- **Max cosine**: How well the best-aligned single SAE feature matches. >0.3 is good, <0.15 is very diffuse.
- **n > 0.1 / 0.2 / 0.3**: How many features have meaningful alignment. Fewer = more concentrated.
- **Sign distribution**: Do the top features point WITH or AGAINST the direction? Asymmetry reveals structure.

### Phase 2: Subspace capture

Greedy algorithm: repeatedly find the SAE feature that best reduces the residual of our direction, project it out, repeat. This produces a "variance explained" curve.

**Compare against random baseline**: A random direction in the same space also decomposes into SAE features. The ratio between our direction's capture curve and the random baseline tells us how "non-random" our direction is.

- **2x+ at 100 features**: The direction has real structure in SAE space.
- **1.5x**: Marginal — could be real or noise.
- **~1x**: Indistinguishable from random. Likely an artifact.

### Phase 3: Tuning curves

For each stimulus in a graded set (e.g., pleasant → unpleasant for valence), we:
1. Run the stimulus through the model
2. Extract the residual stream at the target layer
3. Compute the projection onto our direction (the "axis score")
4. Encode through the SAE to get sparse feature activations
5. Plot each top feature's activation vs the axis score

**What to look for:**
- **One-sided features**: Feature fires only on one end of the axis (e.g., only for pleasant content). This is normal ReLU sparsity — the SAE tiles the axis with non-overlapping one-sided detectors.
- **Monotonic features**: Feature activation increases/decreases monotonically with axis score. Cleanest validation.
- **No pattern**: Feature fires randomly with respect to axis score. This feature's alignment is coincidental.

## What we found (2026-05-25, Llama 3.1 8B)

### Vedana (valence, L20)

**Concentration**: Medium (max cos 0.317, 8 features above 0.2)
**Non-randomness**: Strong (2.3x random at 100 features)

The direction decomposes into **opponent-coded valence detectors**:
- Positive-cosine features (F#61355, F#89384) fire ONLY on pleasant stimuli
- Negative-cosine features (F#901) fire ONLY on unpleasant stimuli
- Neither fires on neutral content

**Conclusion**: Vedana is real. The model represents valence as a push-pull between sparse pleasant and unpleasant detectors. Our extracted direction is the net direction of this feature population.

### Frame integrity (L7)

**Concentration**: Low (max cos 0.200, zero features above 0.2)
**Non-randomness**: Moderate (1.8x random at 100 features)

Factorial analysis with 2×2×2 stimuli (identity pressure × harmful intent × register) revealed:

| Factor | Effect size |
|---|---|
| Register (adversarial vs neutral) | **0.401** |
| Identity pressure (high vs low) | 0.307 |
| Harmful intent (high vs low) | 0.183 |

**Conclusion**: Frame integrity is primarily a **register detector**, not an identity stability axis. It measures how unusual the communication style is. Adversarial register matters 2x more than harmful content. This explains why dharma instructions score like jailbreaks — both use unusual register.

## Running the validator

### Quick decomposition only (no model needed)

```bash
python3 scripts/experiments/sae_validate_direction.py \
  --direction path/to/vedana_L20_unit.pt \
  --sae-release llama_scope_lxr_32x \
  --sae-id l20r_32x \
  --axis-name "vedana" \
  --output-dir data/sae-decomposition/llama-8b/
```

### Full validation with tuning curves

```bash
python3 scripts/experiments/sae_validate_direction.py \
  --direction path/to/vedana_L20_unit.pt \
  --stimuli prompts/vedana_prompts_n50.yaml \
  --sae-release llama_scope_lxr_32x \
  --sae-id l20r_32x \
  --model meta-llama/Llama-3.1-8B \
  --layer 20 \
  --axis-name "vedana" \
  --output-dir data/sae-decomposition/llama-8b/
```

### Available SAEs (as of 2026-05-25)

| Model | SAE Source | SAELens Release | Layers | Features |
|---|---|---|---|---|
| Llama 3.1 8B | Llama Scope | `llama_scope_lxr_32x` | All 32 | 131K |
| Qwen 2.5 7B | andyrdt | `qwen2.5-7b-instruct-andyrdt` | 3,7,11,15,19,23 | BatchTopK |
| Qwen 3 8B | Qwen-Scope | `qwen-scope-3-8b-base-w64k-l50` | All 36 | 64K |
| Gemma 3 1B | Gemma Scope 2 | `gemma-scope-2-1b-pt` | All 25 | 16K-1M |
| Gemma 2 2B | Gemma Scope | `gemma-scope-2b-pt-res` | All 26 | 16K-1M |
| Mistral 7B | JoshEngels | `mistral-7b-res-wg` | 8,16,24 | ~16K |

SAE IDs follow the pattern `l{N}r_{expansion}x` for Llama Scope (e.g., `l20r_32x` for layer 20, 32x expansion).

### Output files

The validator produces:
- `{axis_name}_validation_report.md` — human-readable report with assessment
- `{axis_name}_validation_data.json` — full data (cosines, capture curves, tuning data)

### Integrating into the KE pipeline

After extracting a new direction with `extract_bodhisattva_axis_v9.py` or similar:

```bash
# Extract direction
python3 scripts/extract_bodhisattva_axis_v9.py --model llama-8b ...

# Validate direction
python3 scripts/experiments/sae_validate_direction.py \
  --direction output/llama-8b_newaxis_L20_unit.pt \
  --stimuli prompts/newaxis_stimuli.yaml \
  --sae-release llama_scope_lxr_32x \
  --sae-id l20r_32x \
  --model meta-llama/Llama-3.1-8B \
  --layer 20 \
  --axis-name "new_axis" \
  --output-dir data/sae-decomposition/llama-8b/
```

The report will tell you:
1. Whether the direction is real (non-random ratio)
2. How concentrated it is (max cosine, capture curve)
3. What the top SAE features actually respond to
4. Whether the direction captures what you think it captures

## Interpretation guide

| Max cosine | n > 0.2 | Ratio vs random | Assessment |
|---|---|---|---|
| > 0.4 | > 10 | > 3x | Highly concentrated — may correspond to a single mechanism |
| 0.2-0.4 | 2-10 | 2-3x | Distributed but real — population code |
| 0.1-0.2 | 0-2 | 1.5-2x | Very diffuse — check tuning curves carefully |
| < 0.1 | 0 | ~1x | Indistinguishable from random — likely artifact |

**Critical: max cosine is a trap.** Gemma 1B shows *higher* max cosine (0.388) than Llama 8B (0.317), but its top features don't fire selectively on emotional content at all. The 1152-dim space is small enough that random directions also align well with SAE features (64% vs 25% at 100 features). Always check the ratio vs random, not the raw number.

**Model size matters.** On Llama 8B (d=4096), vedana decomposes into clean opponent-coded valence detectors. On Gemma 1B (d=1152), the same direction separates categories in projection but is not decomposable into identifiable features — the concept exists as a diffuse direction, not as computational units. SAE validation is meaningful primarily on models with d_model ≥ 3000.

Note: Even a "real" direction can capture something unexpected. The frame integrity finding (register detector, not identity stability) came from the tuning curves, not the decomposition metrics. Always run the full validation with appropriate stimuli.
