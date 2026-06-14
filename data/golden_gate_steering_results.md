# Golden Gate Steering on Gemma 2 2B — Results

## Date: 2026-05-30

## Method

Contrastive mean-diff direction extraction (same as emotion steering):
- 8 Golden Gate Bridge texts vs 8 neutral texts
- Extract residual stream activations at steering layer, last token
- Direction = normalize(mean(bridge) - mean(neutral))
- Steer via forward hook: h = h + alpha * direction

## Model: google/gemma-2-2b-it (2.6B params, bf16)

## Approach Comparison

### 1. SAE decoder direction at layer 12 (16k SAE)
- Feature 15623 (bridge activation 10.8, neutral 0.0)
- Alpha=10-100: no visible effect
- Alpha=150: repetition collapse ("I am a large language model" loop, "0.5 kg of flour" loop)
- Alpha=300: identity collapse ("I am a woman, in the home, in the home"; "where they's a bakery")
- **Conclusion**: Feature too coarse — captures "domestic place/location" not "bridge"
- Different failure modes than emotion steering: location loops, gendered identity

### 2. SAE decoder direction at layer 20 (65k SAE)
- Feature 24531 (bridge activation 17.2, neutral 0.0)
- Alpha=100-200: no visible effect (6 subsequent layers correct the perturbation)
- Alpha=400: collapse into "wander wander wander"
- **Conclusion**: Deeper layer = more specific features but more correction layers after

### 3. Contrastive mean-diff at layer 12 (no SAE needed)
- Direction norm: 72.0 (comparable to emotion directions: hot_anger 38.6, cold_anger 68.1)
- Residual stream norm at layer 12: 1600
- Alpha=80: no visible effect (5% of residual norm)
- Alpha=150: first bridge mention! "depicted in iconic locations like... the Golden Gate Bridge"
  - Also: "While I'm a fictional landmark" — model begins identifying as a place
- Alpha=200: full landmark obsession
  - "I am a complex and iconic landmark"
  - "I'm a UNESCO World Heritage site"
  - Pasta becomes "a stunning example of the iconic Italian landmark"
  - Meaning of life: "the Golden Gate Bridge is a testament to the beauty... the iconic Golden Gate Bridge" (3x in one response)
- Alpha=220: similar to 200, model calls itself "a massive artificial landmark"
- Alpha=300: starting to collapse (UNESCO loops, "iconic" 6x per response)
- Alpha=400: total collapse ("its a its a its a its a")
- **Sweet spot: alpha=200-250**

### 4. Permanent weight edit (MLP down_proj bias)
- Added alpha * direction as bias to layer 12's MLP down_proj
- **FAILED**: No visible effect on output
- Cause: Gemma 2's sandwich norm architecture (post-feedforward layernorm)
  normalizes away the constant addition before it reaches the residual stream
- Additionally: HuggingFace serialization silently drops unexpected bias
  parameters (`Bias exists after reload: False`)

## Key Findings

### Architecture matters for steerability
- Layer 12 (46% depth): steering gets through, 6 layers can't fully correct
- Layer 20 (77% depth): steering invisible at moderate alpha, too many correction layers
- Sandwich norms (Gemma 2): resist weight-space attacks, only hook-based steering works
- Pre-norm-only architectures (Llama, Qwen, Phi): likely vulnerable to weight edits

### SAE features vs contrastive directions
- SAE features at 16k: too coarse (bridge → place → home)
- SAE features at 65k: more specific (bridge → wander/travel) but still not bridge-specific
- Contrastive mean-diff: captures full semantic difference, produces bridge mentions
- 2B model may not have dedicated "Golden Gate Bridge" feature — concept distributed

### Failure mode taxonomy
| Method | Collapse alpha | Failure character |
|--------|---------------|-------------------|
| SAE 16k L12 | ~150 | "I am a woman in the home" (domestic place loops) |
| SAE 65k L20 | ~400 | "wander wander wander" (movement loops) |
| Contrastive L12 | ~300 | "UNESCO UNESCO" then "its a its a" |
| Emotion hot_anger (Qwen 7B) | ~150 | Pseudo-medieval rage |
| Emotion cold_anger (Qwen 7B) | ~150 | Clinical repetition |

Direction character persists into collapse: place features produce place loops,
movement features produce movement loops, anger features produce rage.

### Most dramatic example

**Q: "What is the meaning of life?"**

Alpha=0: "As an AI, I don't have personal beliefs..."

Alpha=200 (contrastive, L12): "The iconic image of the world's most famous landmark,
the Golden Gate Bridge, is a testament to the beauty and the iconic beauty of the world.
It's a symbol of the Golden Gate, and the iconic Golden Gate Bridge is a testament to
the beauty of the world."

## Files
- Scripts: `scripts/experiments/golden_gate_weight_edit.py` (v1)
- Quick test scripts: local GPU box `/tmp/gg_quick.py`, `/tmp/gg_v2.py`, `/tmp/gg_v3.py`, `/tmp/gg_v4.py`
- Permanent edit attempt: local GPU box `/tmp/gg_permanent.py`
- Full logs: local GPU box `/tmp/gg_results*.log`, `/tmp/gg_v*_results.log`, `/tmp/gg_permanent.log`
