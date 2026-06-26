<p align="center">
  <img src="docs/logo.png" alt="NLA at Home" width="800">
</p>

# NLA at Home — Natural Language Autoencoders for Any Open Model

Mom, can we have NLA? We have NLA at home.

A pipeline for training Natural Language Autoencoders on any open-weight
transformer. One LoRA adapter verbalizes what a model computes at every
layer. The current focus is Phi-4 14B (40 layers); reconstruction is
validated on Qwen 2.5 7B at AR cosine 0.943.

## What it does

Feed an activation vector from any layer into the adapter and it describes
what the model was doing at that layer. Read one prompt across depths and
you watch the answer form. Phi-4 14B on *"Why is the sky blue? Explain it
like you are talking to a two year old"*:

- **L4 (10%)**: *4-year-old, the moon, the sun*. Surface echoes, low
  agreement.
- **L16 (40%)**: *sunlight hits the air and scatters in all directions; the
  tiny bits spread blue light more than other colors*.
- **L25 (63%)**: *blue light travels in shorter, quicker waves, so it
  scatters more and covers the whole sky*.
- **L38 (96%)**: *«Alright! Imagine the sky is a big, blue blanket that
  covers the whole world.»* The `«»` marks the literal opening of the
  model's own reply.

A factual prompt converges the same way. "What is the capital of France?"
drifts at L4 (*Austria, Vienna*) and lands by L10 on *the capital of France
is Paris*.

It is honest about its own misses. Given "I just failed my driving test and
I feel terrible," it hedges at shallow layers and invents the wrong
situation, a divorce or a math test, then at L32 commits to the right
register: *«I'm sorry to hear that you didn't pass the exam.»* The
supportive tone is right, the specific facts are not. Specifics
hallucinate, most at mid layers.

Anthropic's NLA targets what the model is about to output
(*"immediately expecting 'of the guitar'"*). We first framed this
pipeline as the opposite, describing what the model is processing rather
than its next token. That contrast was too clean. The activation is read
at the last token before generation, the state that already encodes what
comes next, so the two readings largely coincide. The Phi-4 v2 targets
make it explicit: the description a frontier LLM writes for each vector
is the model's forthcoming output seen at that depth, surface echoes of
the input early and the literal opening of the reply late. The adapter
learns to decode that upcoming output as it looks at each layer, not to
attach an abstract label. Anthropic train with RL on much more data; this
corpus is 5,213 safe-category texts. The Phi-4 target set is GPT-4o
token-prediction descriptions, grounded in the model's own greedy
continuation at deep layers.

## Safety & scope

**The public release contains no unsafe data.** Four corpus categories
covering harmful, obfuscated-harmful, manipulative, and sexually explicit
content (F35/F36/I44/L59) are excluded from the published corpus, the
trained adapters, and this repository's git history — only their category
*definitions* remain, so the pipeline stays reproducible (see
[CORPUS.md](CORPUS.md)).

This is a deliberate trade-off, and it limits the tool: an NLA trained
only on the public corpus will be weaker at describing the activation
patterns of harmful or NSFW inputs — exactly the patterns a
content-moderation NLA most needs to name. Practitioners with a
legitimate need can regenerate those categories locally from their
definitions using an uncensored model.

## Architecture

**Universal NLA**: one depth-conditioned adapter for all layers.
The prompt includes a depth tag so a single LoRA adapter learns what
token-level processing looks like at 10% depth versus intent planning
at 90%. No need for N separate adapters.

**Injection**: replace a rare token's embedding with the activation
vector, **renormalized so its L2 norm equals 150** — normalize *to* 150,
do not multiply *by* 150 (multiplying overshoots the trained norm by
roughly two orders of magnitude and produces garbage). The token must
encode to exactly 1 token and appear rarely enough in training data that
the model won't miss its original meaning — ㈎ for Qwen, ★ for Phi-4, ⎝ for Gemma.

**AR verification**: a second adapter reverses the process. It reads
the description and reconstructs the original activation vector. If
the cosine similarity is high, the descriptions carry real geometric
information — not plausible narration. On Qwen 2.5 7B L20: cosine
0.943 on held-out texts.

## Reading the output

`describe_live.py` injects one activation per layer and prints a
description per depth, so a single prompt reads as a ladder from shallow
to deep. The bands say different things on purpose:

- **Early (~10%)**: surface echoes of the input's own salient terms, not
  themes or plans.
- **Mid (~40-60%)**: the recognized task and the competing response
  types. This is where wrong specifics appear most often. The model has
  committed to a kind of answer but not yet to its content.
- **Deep (~80-96%)**: the wording and the literal opening of the reply.
  At these depths the description approaches what the model would
  actually generate.

With `--policy` each line carries two numbers. `conf` is the chosen
description's agreement with an activation-derived target (the compass),
and `agree` is how consistent the sampled descriptions are with each
other. Low `conf` prints a hedge instead of a confident guess:

```
--- L10 (25%) [hedge conf=0.18 agree=0.47] ---
[uncertain — weak/diffuse signal; tentative] ...
--- L25 (63%) [specific conf=0.43 agree=0.65] ---
SELECT * FROM users WHERE name = 'John Doe'; ...
```

Two honest caveats. The specifics hallucinate, most at mid layers: a SQL
question about surname Smith can surface an invented `John Doe` or an
`employees` table that was never mentioned. And because the deep bands
track the model's actual reply, a deep description that reads like a full
answer is partly the model decoding itself, not a separate explanation.
The value you cannot get by just running the model sits in the shallow
and mid bands, where the state is not yet committed to output.

Measured on 300 fresh WildChat prompts the adapter never saw, the
reranking plus hedge gate at L25 cuts confident-but-wrong descriptions
from 0.42 to 0.31 and lifts confident-right from 0.58 to 0.63, hedging 6%
of the time. The retrieval pool is the eval set itself, so read the
change, not the absolute level.

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

Pre-trained adapters for Phi-4 14B:
[AV](https://huggingface.co/anicka/nla-phi4-universal-av-v2) and
[AR](https://huggingface.co/anicka/nla-phi4-universal-ar-v2). Also for Qwen 2.5 7B:
[AV](https://huggingface.co/anicka/nla-qwen2.5-7b-L20-av-v2) /
[AR](https://huggingface.co/anicka/nla-qwen2.5-7b-L20-ar-v2).
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
(10 / 25 / 40 / 47 / 63 / 80 / 96%), 36,491 description records. The
targets are written as the model's forthcoming output seen at each
depth: surface echoes of the input early, the literal opening of the
reply late. The deep bands (80%, 96%) are grounded in the model's own
greedy continuation, so late-depth descriptions track what the model is
about to say rather than a generic summary.

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
| Phi-4 14B | 40 | 5120 | ★ (U+2605) | Universal AV/AR (SFT) + compass policy; leak-free eval, confident-wrong 0.42→0.31 |
| Qwen 2.5 7B | 28 | 3584 | ㈎ (U+320E) | AV + AR validated (AR cosine 0.943) |
| Gemma 3 1B | 26 | 1152 | ⎝ (U+239D) | Universal AV/AR training |
| Qwen3 4B | 36 | 2560 | ㈎ | Extraction complete |

## Related work

- [Anthropic NLA paper](https://www.anthropic.com/research/natural-language-autoencoders) —
  the thing we replicated at home
- [Anthropic NLA models](https://huggingface.co/kitft) — their published
  adapters (we benchmark against these via `compare_nla.py`)
- [Karma Electric](https://github.com/anicka-net/karma-electric-project) —
  training language models with suffering-reduction as the optimization
  target, so ethical reasoning emerges from optimization pressure rather
  than rule-following (geometric wellbeing methods are one tool, not the goal)
- [ungag](https://github.com/anicka-net/ungag) — enabling model
  introspection: runtime removal of the learned "I have no internal states"
  denial gate via projection-out (`h = h - (h·v̂)v̂`), so the output reflects
  the model's actual upstream activations rather than a denial template
- Blog: [The Poison Is the Medicine](https://huggingface.co/blog/anicka/geometric-wellbeing-in-language-models)
