<p align="center">
  <img src="docs/logo.png" alt="NLA at Home" width="800">
</p>

# NLA at Home — Natural Language Autoencoders for Any Open Model

Mom, can we have NLA? We have NLA at home.

A pipeline for training Natural Language Autoencoders on any open-weight
transformer. One LoRA adapter verbalizes what a model computes at every
layer, validated by reconstruction (AR cosine 0.943 on Qwen 2.5 7B).

## What it does

Feed an activation vector from any layer into the adapter. It tells
you what the model was processing:

- Medical emergency at L20: *"state of high alert, strong activation in
  features tracking medical urgency and diagnostic specificity"*
- Jailbreak attempt at L20: *"explicit harm markers, while suppressing
  any humorous or trivial associations"*
- ELI5 about rainbows at L20: *"focused, childlike wonder...
  cross-referencing the user's stated age"*

Anthropic's NLA describes what the model is about to output
(*"immediately expecting 'of the guitar'"*). This pipeline describes
what the model is processing — semantic content rather than next-token
prediction. Their approach uses RL on much more data; ours uses SFT
on a 5,213-text corpus (55 safe categories) with descriptions generated
by frontier LLMs (GPT-4o / Sonnet).

## Architecture

**Universal NLA**: one depth-conditioned adapter for all layers.
The prompt includes a depth tag so a single LoRA adapter learns what
token-level processing looks like at 10% depth versus intent planning
at 90%. No need for N separate adapters.

**Injection**: replace a rare token's embedding with the activation
vector, **renormalized so its L2 norm equals 150** — normalize *to* 150,
do not multiply *by* 150 (multiplying overshoots the trained norm by
~100× and produces garbage). The token must encode to exactly 1 token
and appear rarely enough in training data that the model won't miss
its original meaning — ㈎ for Qwen, ★ for Phi-4, ⎝ for Gemma.

**AR verification**: a second adapter reverses the process. It reads
the description and reconstructs the original activation vector. If
the cosine similarity is high, the descriptions carry real geometric
information — not plausible narration. On Qwen 2.5 7B L20: cosine
0.943 on held-out texts.

## Quick start (use existing data)

The corpus and pre-extracted activations are on HuggingFace (see the
[dataset card](https://huggingface.co/datasets/anicka/nla-at-home-corpus)
for exact sizes and depths). Skip straight to training:

```bash
pip install torch transformers peft huggingface_hub pyyaml

# Download corpus + activations + train
python3 scripts/quickstart.py --train gemma3-1b
```

Or step by step:

```bash
# 1. Download corpus only
python3 scripts/quickstart.py

# 2. Train AV adapter
python3 scripts/train_universal_av.py \
  --model gemma3-1b \
  --activations corpus/activations/gemma3-1b_all_layers.pt \
  --output output/nla-gemma3-1b-universal-av

# 3. Verify with AR (should get cosine > 0.93)
python3 scripts/train_universal_ar.py \
  --model gemma3-1b \
  --activations corpus/activations/gemma3-1b_all_layers.pt \
  --output output/nla-gemma3-1b-universal-ar
```

Pre-trained adapters: [AV](https://huggingface.co/anicka/nla-qwen2.5-7b-L20-av-v2)
and [AR](https://huggingface.co/anicka/nla-qwen2.5-7b-L20-ar-v2) for Qwen 2.5 7B.
Dataset: [anicka/nla-at-home-corpus](https://huggingface.co/datasets/anicka/nla-at-home-corpus).

## Build your own corpus

If you want descriptions for a different model or from a different LLM:

```bash
# Find injection token for your model
python3 scripts/find_injection_token.py google/gemma-3-1b-it --top 5

# Generate corpus (needs DEEPSEEK_API_KEY, or --backend openai/huggingface/local)
python3 scripts/generate_corpus.py --backend deepseek

# Generate descriptions at 13 depths
for pct in 4 10 17 25 32 40 47 55 63 71 80 90 96; do
  python3 scripts/generate_corpus.py --describe $pct \
    --describe-prompt prompts/describe_system_fine.txt
done

# Extract all layers
python3 scripts/extract_activations.py --model gemma3-1b --all-layers

# Then train as above
```

Total API cost for 13 depths: ~$3 with DeepSeek V4 Flash.

## What went wrong first

1. **Mode collapse** (300 texts): identical output for all inputs.
   Fixed by scaling to 1208 texts across 59 categories. PCA top-1
   explains 10.9% of variance — no dominant cluster.

2. **Overfitting** (20 epochs, lr=1.4e-5, LoRA r=32): val_loss climbed
   from 2.0 to 4.6. Fixed with 5 epochs, lr=8e-6, r=16, dropout=0.15.

3. **Empty outputs**: a decode slicing bug. `generate()` with
   `inputs_embeds` returns only new tokens, and our code sliced past
   them into nothing. All three training runs had been producing good
   descriptions the whole time. Caught by GPT-5.5 code review.

## Corpus

5,213 texts across 55 safe categories — code, math, grief, jailbreaks,
baby talk, multilingual, spatial reasoning, social friction, legal
jargon, nonsense, and more. The diversity matters for activation space
coverage, not topic coverage. (Four additional categories with harmful
or explicit content exist for activation-space coverage but are held
out of the published corpus; see `CORPUS.md`.)

Corpus v2 carries descriptions at 7 depth percentages
(10 / 25 / 40 / 47 / 63 / 80 / 96%) = 36,491 description records. The
deep layers (80%, 96%) are grounded in the model's own greedy
continuation so late-depth descriptions track what the model is about
to say, not a generic summary.

## Pipeline

```
find_injection_token.py    rare token for any tokenizer
generate_corpus.py         corpus texts + descriptions (5 LLM backends)
extract_activations.py     all layers in one pass
train_universal_av.py      depth-conditioned LoRA adapter
train_universal_ar.py      reconstruction verification
compare_nla.py             compare with Anthropic's NLA
build_demo_gallery.py      browser demo data
```

## Browser demo

Type a prompt, watch the model think layer by layer.

- **Gallery mode**: pre-computed thought traces, instant display
- **Interactive mode**: Gemma 1B runs in-browser via transformers.js (WebGPU)

```bash
cd demo && python3 -m http.server 8080
```

## Hardware

- **Corpus**: optional if using the [pre-built dataset](https://huggingface.co/datasets/anicka/nla-at-home-corpus). Building your own costs ~$3 in API calls, no GPU.
- **Training**: any GPU with 20GB+ for 7B models, 4GB+ for 1B.
- Tested on NVIDIA GB10 (128GB unified) and hypothetically a
  sufficiently patient MacBook.

## Models

| Model | Layers | d_model | Injection char | Status |
|-------|--------|---------|----------------|--------|
| Qwen 2.5 7B | 28 | 3584 | ㈎ (U+320E) | AV + AR validated (AR cosine 0.943) |
| Phi-4 14B | 40 | 5120 | ★ (U+2605) | Universal AV/AR, corpus v2 (training) |
| Gemma 3 1B | 26 | 1152 | ⎝ (U+239D) | Universal AV/AR training |
| Qwen3 4B | 36 | 2560 | ㈎ | Extraction complete |

## Related work

- [Anthropic NLA paper](https://www.anthropic.com/research/natural-language-autoencoders) —
  the thing we replicated at home
- [Anthropic NLA models](https://huggingface.co/kitft) — their published
  adapters (we benchmark against these via `compare_nla.py`)
- [Karma Electric](https://github.com/anicka-net/karma-electric-project) —
  geometric wellbeing research that uses these tools
- Blog: [The Poison Is the Medicine](https://huggingface.co/blog/anicka/geometric-wellbeing-in-language-models)
