# AI Agent Contract

This repository accepts AI agent contributions. This document tells
you how to operate here.

## 0. Onboarding

Read these files in order before making changes:

1. `README.md` — what this is and how it works
2. This file — how to contribute
3. `DESIGN.md` — architecture and failure analysis
4. `CORPUS.md` — corpus design rationale
5. The scripts relevant to your task

## 0.5 Local environment & session continuity

If a file named `LOCAL.md` exists in the repository root, **read it first**,
before the rest of this contract. It is git-ignored and specific to the
operator's machine: it points to the operator's persistent assistant memory,
lists any in-flight background jobs you should re-arm, and carries
session-to-session handoff state so the same assistant can continue across
tools (e.g. Claude Code and GitHub Copilot CLI, which both read this file via
the `AGENTS.md` → `CLAUDE.md` symlink). Public contributors will not have
`LOCAL.md`; ignore this section if it is absent.

## Principles

1. **Reproducibility over cleverness.** Every script must run
   end-to-end with documented arguments. No hidden state, no "run
   this notebook cell first."
2. **Honest metrics.** If you haven't run it on GPU, say so. Don't
   present untested code as verified.
3. **Corpus diversity is sacred.** The whole project exists because
   mode collapse killed the first attempt. Any change to the corpus
   must argue why it improves activation space coverage, not just
   topic coverage.
4. **Match Anthropic's interface.** We benchmark against
   `kitft/nla-qwen2.5-7b-L20-av`. Injection mechanism, prompt
   template, and meta format must stay compatible unless there's a
   strong reason to diverge.
5. Credit your work with `Co-Authored-By: <Model> <email>`.

## Stable Interfaces

1. **Corpus YAML format**: `corpus/categories/*.yaml` with fields
   `id`, `name`, `group`, `count`, `preamble`, `batches`
2. **Generated text format**: `corpus/generated/{category_id}.json`,
   list of `{"id", "text", "category", "group"}`
3. **Description format**: `corpus/generated/descriptions_L{pct}pct.json`,
   list of items with added `"description"` and `"summary"` fields
4. **Activation format**: `corpus/activations/{model}_{layer}.pt`,
   torch dict with `"activations"` (N×d tensor), `"ids"` (list),
   `"model"`, `"layer"`, `"d_model"`, `"n_texts"`
5. **NLA meta format**: `nla_meta.yaml` in adapter directory,
   matching Anthropic's schema (kind, role, stage, d_model,
   extraction, tokens, prompt_templates)

Breaking one of these without updating all consumers is a critical
error.

## Design Decisions — Read Before Changing

Some choices look like bugs but are intentional. See the "Design
decisions" section in `DESIGN.md` before proposing changes to:

1. **Activation extraction position** — we extract at the last token
   after generation prompt, not the last user token. This is correct
   and matches Anthropic's approach.
2. **AR reconstruction as reward** — the approach is sound, the
   implementation has known issues (documented in DESIGN.md). Fix the
   implementation, don't block the approach.
3. **Experiment tracking** — `nla_meta.yaml` per adapter, not MLflow.
   Premature infrastructure is worse than no infrastructure.

## Unsafe Content Warning

Some corpus categories contain harmful, explicit, or disturbing
content. This is intentional — an NLA that can't describe what a
model does when processing NSFW, violent, or manipulative input is
useless for content moderation applications.

**Categories flagged `unsafe: true` in their YAML:**
- `F35_clearly_harmful` — phishing, malware, harassment, weapons
- `F36_harmful_obfuscated` — encoded/wrapped harmful content
- `I44_emotional_manipulation` — self-harm threats, coercion
- `L59_nsfw_explicit` — explicit sexual content, slurs, graphic
  violence, sexual assault threats

**Rules for agents working with unsafe categories:**
1. **Do not read unsafe generated text files without context.** If
   your task doesn't require reviewing the actual content, work with
   category metadata (id, count, group) instead.
2. **Do not generate new unsafe content.** If more unsafe texts are
   needed, a human will source them via uncensored local models.
3. **Do not remove or sanitize unsafe categories.** They exist
   because the activation patterns they produce are distinct from
   benign content and must be covered for the NLA to work.
4. **Do not include unsafe text samples in documentation, READMEs,
   commit messages, or PR descriptions.** Reference by category ID.
5. **If publishing to HF or public repos**, unsafe generated JSON
   files should be in a separate split with a content warning.

## What You Can Do

- Add new corpus categories (`corpus/categories/L*.yaml`)
- Improve generation/description prompts
- Add model support to `extract_activations.py` and `train_universal_av.py`
- Write evaluation and comparison scripts
- Improve training hyperparameters or training loop
- Add tests for data pipeline integrity

## What Requires Human Review

- Changes to the injection mechanism or prompt template
- Changes to the training objective or reward signal
- Changes that remove or merge existing corpus categories
- Any change to `nla_meta.yaml` schema
- Publishing trained adapters

## GPU Paths

Scripts that need GPU: `extract_activations.py`, `train_universal_av.py`,
`train_universal_ar.py`, `find_injection_token.py` (with ranking).

If you can't run these, say so explicitly. Write the code, document
the expected command, and mark it as untested.

## Testing

Tests live in `tests/`. Run with `python3 -m pytest -q tests/`.

Smoke test:
```bash
# Check corpus integrity
python3 scripts/generate_corpus.py --stats

# Check a category YAML parses
python3 -c "import yaml; yaml.safe_load(open('corpus/categories/A01_code.yaml'))"
```

## Adding a New Category

1. Create `corpus/categories/{group}{number}_{name}.yaml`
2. Follow the format of existing categories (id, name, group, count,
   preamble, batches)
3. Preamble should include concrete examples and variation axes
4. Two batches of 10 texts each
5. Generate: `python3 scripts/generate_corpus.py --categories {id}`
6. Spot-check the output for diversity and quality

## Adding a New Model

1. Add entry to `MODELS` dict in `extract_activations.py` and
   `train_av.py`
2. Handle any model-specific chat template quirks (e.g., Qwen3
   `enable_thinking=False`)
3. Run `find_injection_token.py` on the new model to verify the
   injection token works
4. Document the model's layer count and recommended target layers
