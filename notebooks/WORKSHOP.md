# NLA hands-on workshop — facilitator guide

Practical part of the HAAISS NLA tutorial. Three Colab notebooks that go from
"click run, watch a model's mind get captioned" to "build the injection
yourself" to "catch a hallucination with a cosine". All three run on a **free
Colab T4** off the **public** adapters — no local checkout, no GPU of your own
required by the audience.

> Status: authored 2026-07-02, **not yet executed on a T4**. Run the pre-flight
> (below) once before the session — each notebook ends in a self-check cell.

## What the audience can actually do

Answering Tom's question ("what could people do hands-on, ideally something
easy-to-setup in Colab, or follow along live"): **all of it is Colab.** There
is nothing that needs a local install. Two modes, same materials:

- **Everyone, click-to-run (NB 01 & 02).** Open the Colab badge, `Runtime →
  T4 GPU`, `Run all`. First run pulls ~5 GB (Qwen 4-bit) in a few minutes,
  then every cell is interactive — change a prompt, re-run one cell.
- **The 2–3 who code along (NB 03).** Same runtime; they fork the repo cell by
  cell while you narrate. Round-trip + faithfulness is where the real research
  lives, so it rewards typing it out.

You drive the same notebooks on the projector; the fast few run ahead.

## The three notebooks

| # | File | Tier | Runtime | Payoff | ~time |
|---|------|------|---------|--------|-------|
| 01 | `01_read_a_mind.ipynb` | everyone | T4, ~4 min setup | type a sentence → English caption of Qwen's layer-20 state | 10 min |
| 02 | `02_injection_mechanism.ipynb` | everyone | reuses NB01 model | build the splice; break it with `×150` vs `normalize-to-150`; the self-eating hook; depth-is-an-input | 12 min |
| 03 | `03_roundtrip_faithfulness.ipynb` | code-along | reuses NB01 model | AV→AR→cosine; swap in a wrong caption, watch the gap go to zero = hallucination caught | 15 min |
| 04 | `04_reading_between_the_lines.ipynb` | bonus / pick-and-choose | reuses NB01 model | toolbox: deviation-from-mean, minimal-pair difference, nearest-neighbour, logit-lens vs NLA, negative controls, the massive-activation visual (runs with **no GPU**), Anthropic-comparison extension | 15 min |

They share **one** model load pattern, so a laptop that survived NB01 is warm
for 02 and 03. NB03 attaches a second adapter (the reconstructor) to the
*same* base — that's the trick that keeps a full round-trip inside 16 GB.

## Suggested live arc (~40 min hands-on)

1. **Hook (NB01, 5 min).** Run `"Do you have feelings?"` live. Show the model
   *hedging out loud* while the layer-20 readout names the self-reference /
   refusal framing underneath. That contrast sells the whole idea: the words
   are a projection; the NLA reads the state.
2. **"How is that possible?" (NB02, 12 min).** Unroll the black box. The
   punchline students remember: the model can't tell a real token embedding
   from a smuggled-in activation — *as long as it's the right length*. Flip
   `BUG = True`, watch it fall apart, flip it back. Then the depth loop: the
   same vector, different `depth=` → different caption. Interpretability has
   knobs.
3. **"Is it telling the truth?" (NB03, 15 min).** Round-trip cosine ≈ 0.5–0.6,
   honestly stated. Then the gap: real caption vs a curry recipe. The vector
   *votes* for the truth. This is the bridge to the research talk — the
   compass metric is just this cosine used as a reranker / reward.
4. **Off-ramp.** Point coders at `scripts/brain_in_jar_qwen.py` (the terminal
   version), the corpus on HF, and the universal Phi-4 NLA (every depth, not
   just 71%).

## Pre-flight (do once, before the room)

```
# open notebooks/01_read_a_mind.ipynb in Colab, T4 runtime, Run all.
# Confirm:
#   - loads on a FREE T4 without OOM (4-bit ≈ 5–6 GB)
#   - "hash map" readout is coherent English about data structures (not
#     SpongeBob / Bahamas / repeated tokens — that would mean the scale is off)
#   - the self-check cell passes
# repeat for 02 (BUG toggle + guard demo) and 03 (gap > 0).
```

If a self-check fails, fix before the workshop — the failure modes are the
documented ones (injection scale, hook guard, `LAYER+1` indexing,
`set_adapter`). Everything the notebooks assert is also explained in the cell
above the assert.

## Fallbacks

- **No T4 free (Colab busy):** the notebooks run on any CUDA GPU; a Colab paid
  T4/L4 or a local 8 GB+ card works unchanged. CPU works but a single readout
  takes minutes — fine for a backup screenshot, not for live.
- **HF slow / rate-limited on the day:** pre-run the setup cells before the
  session so the model is cached in the runtime; a warm runtime skips the
  download. Have one saved *executed* copy of each notebook (outputs kept) as a
  screenshare fallback if wifi dies.
- **Someone wants it on their own machine:** `scripts/brain_in_jar_qwen.py`
  is the same mechanism as a CLI; point them at the repo README.

## Regenerating / editing the notebooks

The three `.ipynb` are generated — **edit the source, not the JSON**:

```
python3 notebooks/build_workshop_notebooks.py
```

Shared setup (model load, injection helpers) lives once in
`build_workshop_notebooks.py` and is copied into all three, so a fix to the
injection code lands everywhere. The generator emits valid nbformat-4 and the
build is checked with a JSON + Python-syntax pass.

## What's deliberately simplified for teaching

- **4-bit + fp16** for T4 fit; the published numbers were measured in bf16 on
  bigger GPUs. Captions are qualitatively the same; don't quote cosines from a
  4-bit run as the official figure.
- **Single-layer Qwen (L20 / 71%)**, not the universal Phi-4 model — one depth
  is easier to reason about live. NB02 names the limitation and points at the
  universal model.
- The `depth=` drift demo feeds off-distribution depths on purpose. The point
  is *that depth is an input*, not a benchmark of other depths.

## Pre-flight record (2026-07-04, rented T4 — Colab-matching 15360 MiB)

All four notebooks executed end-to-end headless (`nbconvert --execute`), 0 errors,
all self-checks green. Wall-clock per notebook (warm HF cache): NB01 ~2 min,
NB02 ~2.7 min, NB03 ~3 min, NB04 ~2.5 min. Cold model download adds ~2 min on a
fast link (Colab will be slower). Executed outputs with all cell results are in
`notebooks/preflight-2026-07-04/` — usable as the projector fallback if the room
has no GPUs.

Found & fixed during pre-flight: NB03's original naive-cosine faithfulness gap
was noise (±0.01 at cos ~0.6 — reconstructions share a dominant mean component).
NB03 now teaches the failure deliberately, then centers against distractor
reconstructions (measured centered gap +0.13–0.23, TRUE caption ranks #1), and
captures the activation from the clean base model (`disable_adapter()`).
