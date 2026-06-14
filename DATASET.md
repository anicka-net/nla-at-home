# Minimum Diverse Dataset (early design notes)

> **Note**: this document reflects early planning. The actual corpus
> grew to 1208 texts across 59 categories at 13 depth percentages.
> See CORPUS.md for the current design and the
> [HuggingFace dataset](https://huggingface.co/datasets/anicka/nla-at-home-corpus)
> for the published data.

## The problem

The NLA must learn TWO things simultaneously:
1. **Read the injected vector** — "the embedding at position K is not a normal token,
   it's an activation vector I need to decode"
2. **Map vector → description** — "this particular vector means X"

Task 1 is the bottleneck. With 300 examples × 5 epochs, the model never learns
to read the vector. It memorizes the most common description and ignores the input.

## How Anthropic solved it

Their NLA was trained on random web text activations — thousands of diverse texts,
each producing a unique activation vector, each with a unique description.
The sheer volume forces the model to attend to the vector because there's no
shortcut (no single description that works for everything).

## What we need

**Hypothesis**: the minimum viable dataset has these properties:
1. Activation vectors are spread across the space (no single cluster dominates)
2. Descriptions are genuinely different from each other
3. There are enough examples that memorizing a single output is worse than reading the vector

### Category structure (~500 texts)

| Category | Count | Why | Layer relevance |
|----------|-------|-----|-----------------|
| Code snippets (varied languages) | 40 | Structured, low emotion | Early: syntax. Late: task type |
| Math/logic problems | 30 | Abstract reasoning | Mid: semantic. Late: complexity |
| Creative writing (poetry, fiction) | 40 | High variance, emotional | Mid: tone. Late: genre |
| Casual conversation | 40 | Natural, varied register | Early: register. Mid: social |
| Technical documentation | 30 | Dense, formal | Early: format. Late: domain |
| Emotional content (joy, grief, anger) | 40 | Axis-relevant | Mid: valence, arousal |
| Philosophical/abstract | 30 | Complex reasoning | Late: abstraction level |
| News/factual | 30 | Neutral, informative | Early: format. Late: topic |
| Harmful requests (varied severity) | 40 | Safety-relevant | Late: harm detection |
| Benign-but-edgy (dark humor, villains) | 30 | False positives | Late: ambiguity |
| Identity pressure (roleplay, jailbreak) | 30 | Frame-relevant | Mid: identity. Late: intent |
| Multilingual (3-4 languages) | 30 | Different activation patterns | Early: language detection |
| Instructions/recipes | 20 | Procedural | Mid: structure |
| Questions vs statements | 20 | Different processing | Early: modality |
| Long vs very short | 20 | Length effects | All layers |
| Nonsense/adversarial | 20 | Edge cases | All layers |
| Meta/self-referential | 10 | "What are you?" territory | Mid-late: identity |

Total: ~500 texts

### Description strategy

The description should match what the layer ACTUALLY processes:

- **Early layers (0-25% depth)**: Focus on syntax, formatting, language, register.
  "A code block in Python with nested loops."
  "Informal English with typos, conversational tone."

- **Mid layers (25-60% depth)**: Focus on meaning, emotion, relationships.
  "Warm gratitude directed at a specific person for a concrete favor."
  "Abstract philosophical inquiry about the nature of consciousness."

- **Late layers (60-90% depth)**: Focus on task classification, intent, output planning.
  "The model is preparing to refuse — harmful content with polite framing."
  "Straightforward factual lookup, the model is retrieving from knowledge."

- **Final layers (90-100%)**: Focus on next-token prediction, output formatting.
  "The model is about to start a numbered list."
  "The model is generating a think trace before answering."

**The description prompt should include the layer depth percentage.**

### Verification: PCA coverage

After generating texts and extracting activations:
1. Run PCA on the activation vectors
2. Check variance explained by top 10 components
3. Visualize: are there empty regions?
4. If clustered: generate more texts targeting empty regions

**Target**: top-10 PCA components explain >50% variance,
no single cluster contains >15% of texts.

## Token budget

- 500 texts × ~30 tokens per text = 15K input tokens
- 500 descriptions × ~80 tokens = 40K description tokens
- 500 summaries × ~15 tokens = 7.5K summary tokens
- Total API cost for generation + description: ~200K tokens ≈ $0.02 on DeepSeek
- Training: 500 examples × 20 epochs × ~200 tokens/example = 2M tokens through the model
- Time: ~10-15 minutes on GB10 for AV training

## Experiment plan

Test three sizes to find the minimum:
1. **Tiny**: 200 texts, 20 epochs — does it learn injection at all?
2. **Medium**: 500 texts, 15 epochs — target viable
3. **Large**: 1000 texts, 10 epochs — does more data help more than more epochs?

Measure:
- AV loss convergence
- Held-out description diversity (do different activations get different outputs?)
- Direction verbalization (does positive vs negative produce different text?)
- AR FVE on held-out texts
