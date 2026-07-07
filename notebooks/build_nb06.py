#!/usr/bin/env python3
"""
Generate notebook 06 — EXPERIMENTAL: a confabulation detector from the
disagreement between two witnesses.

Builds directly on 05_three_lenses. The idea (notebook 01's hash-map->"C#"
lesson, made into an instrument): the NLA is a trained decoder with its own
prior, so it can name an entity that isn't in the activation — it fills the
slot. The Jacobian lens has no trained decoder: it reads the model's own
output-poised geometry. Cross-check one against the other.

The markdown below quotes REAL outputs from a GB10 run on 2026-07-07,
4-bit NF4 — the same load path as this notebook — including the ways the
detector half-works. Do not "clean
them up": the honest result is that the grounded side is a solid,
prompt-specific confirmer, while separating an entity claim from the NLA's
own meta-commentary in free prose is an open sub-problem. Both are shown.

Setup / describe() / read_activation() / lens-load cells are copied VERBATIM
from 05_three_lenses.ipynb (clean-base disable_adapter capture). If you
touch them, diff against notebook 05 first.

Run:
    python3 notebooks/build_nb06.py
Produces:
    notebooks/06_confabulation_detector.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GH = "anicka-net/nla-at-home"
FN = "06_confabulation_detector.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def _src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


# ---------------------------------------------------------------------------
# Setup cells copied verbatim from 05_three_lenses.ipynb.
# ---------------------------------------------------------------------------
SETUP_INSTALL = """%pip -q install -U bitsandbytes peft accelerate
!git clone -q https://github.com/anthropics/jacobian-lens
# --no-deps is load-bearing: jlens pins transformers>=5.5 but imports no
# transformers API of its own; letting it upgrade Colab's transformers pulls
# a build whose 4-bit loading fills a T4 with fp16 shards and OOMs. Keep
# Colab's transformers (what cells below rely on); install jlens alone.
%pip -q install -e jacobian-lens --no-deps
import sys
sys.path.insert(0, "jacobian-lens")   # avoid a kernel restart after pip -e"""

SETUP_LOAD = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE       = "Qwen/Qwen2.5-7B-Instruct"
AV_ADAPTER = "anicka/nla-qwen2.5-7b-universal-av-grpo"   # activation -> English
LAYER      = 20                                # 71% depth: layer 20 of 28
DEPTH_PCT  = 71                                # conditioning input to the verbalizer
INJECT_CHAR  = "\\u320e"                       # placeholder token we overwrite: ㈎
INJECT_SCALE = 150.0                           # normalize L2 norm TO this (not multiply!)

device = "cuda"
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> GPU"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
tok  = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                            device_map={"": 0})
model = PeftModel.from_pretrained(base, AV_ADAPTER).eval()

inject_id = tok.encode(INJECT_CHAR, add_special_tokens=False)
assert len(inject_id) == 1, f"injection char must be ONE token, got {inject_id}"
inject_id = inject_id[0]
print("loaded - base + AV adapter on", next(model.parameters()).device)"""

SETUP_HELPERS = '''# --- conventions copied VERBATIM from notebooks 01/05 (clean-base capture) ---
def get_layers(m):
    b = m.base_model.model if hasattr(m, "base_model") else m
    inner = b.model if hasattr(b, "model") else b
    return inner.layers

def read_activation(prompt, layer=LAYER, contaminated=False, max_new_tokens=64):
    """Residual-stream vector at `layer`, last prompt token.

    contaminated=False (default): captured under disable_adapter() -> CLEAN
    Qwen residual, what the NLA was trained on. contaminated=True: adapter
    left ON, so the AV-LoRA perturbs the very activation we then read back.
    The second one is a deliberate footgun, used once below to show what
    contamination does to the readout (notebook 05's lesson, made visible)."""
    import contextlib
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    grab = {}
    def hook(mod, inpt, out):
        h = out[0] if isinstance(out, tuple) else out
        if "h" not in grab:                  # FIRST forward pass only
            grab["h"] = h[:, -1, :].detach()
    handle = get_layers(model)[layer].register_forward_hook(hook)
    ctx = contextlib.nullcontext() if contaminated else model.disable_adapter()
    try:
        with ctx, torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    finally:
        handle.remove()
    reply = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return grab["h"].squeeze(0), reply

def normalize_to(v, scale=INJECT_SCALE):
    n = v.float().norm().clamp_min(1e-12)
    return v * (scale / n)

def av_prompt(depth_pct):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\\n\\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\\n\\n"
        f"Here is the vector from depth {depth_pct}% of the network:\\n\\n"
        f"<concept>{INJECT_CHAR}</concept>\\n\\n"
        "Please provide an explanation.\\n\\n"
        "<explanation>")

def describe(activation, depth=DEPTH_PCT, max_new_tokens=120, scale_fn=normalize_to, **gen_kw):
    chat = tok.apply_chat_template([{"role": "user", "content": av_prompt(depth)}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    pos = ids.index(inject_id)
    emb = model.get_input_embeddings()(torch.tensor([ids], device=device)).clone()
    emb[0, pos, :] = scale_fn(activation.to(emb.dtype))
    attn = torch.ones((1, len(ids)), device=device, dtype=torch.long)
    gen_args = dict(do_sample=False); gen_args.update(gen_kw)
    with torch.no_grad():
        out = model.generate(inputs_embeds=emb, attention_mask=attn,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id, **gen_args)
    seq = out[0]
    gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq
    return tok.decode(gen, skip_special_tokens=True).split("</explanation>")[0].strip()

print("helpers ready")'''

SETUP_LENS = '''import jlens
lens = jlens.JacobianLens.from_pretrained(
    "anicka/jlens-qwen2.5-7b-instruct",
    filename="qwen2.5-7b-instruct_jlens.pt")
jm = jlens.from_hf(base, tok)          # wraps the SAME loaded model
print(lens)
print("fitted layers:", lens.source_layers)'''


cells = [
    md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
       f"(https://colab.research.google.com/github/{GH}/blob/main/notebooks/{FN})"),

    md("""# 06 · A confabulation detector from two witnesses ⚠️ EXPERIMENTAL

**Status: executed end-to-end on a GB10, 4-bit NF4 (the same load path
as this notebook) — quoted outputs are real, including the parts where the
detector only half-works. A different Colab GPU may shift the exact tokens
and layers** (same load pattern as 01-05). Do notebook **05** first; this
reuses its setup and assumes you've seen the J-lens.

Notebook 01 showed the NLA verbalizer naming an entity that isn't in the
activation — the hash-map→"C#" slot-fill. The NLA is a *trained decoder*:
like any language model it has a prior, and it fills unpinned slots from
that prior. That is exactly the failure that makes a readout untrustworthy.

The Jacobian lens has no trained decoder. It transports `h` with the
model's own averaged Jacobian and reads it with the model's own
unembedding — nothing between you and the geometry. So put the two
witnesses against each other:

> A content word the **NLA** emits that appears in **no** layer's
> **J-lens** top-k, at any position in the sequence, is a *candidate
> confabulation* — the verbalizer's prior filling a slot the stream
> did not pin.

Here is what actually happened when we built and ran it, and it split
cleanly in two:

- **The grounded side is a solid, prompt-specific confirmer.** When the
  NLA named Italy, Spain and Greece for one vector, all three were in the
  stream at specific layers — and scored against an unrelated prompt's
  stream, they matched *nothing* (8 vs 0). That direction works.
- **The flag side is limited by a problem we did not solve:** telling an
  entity *claim* apart from the NLA's own *meta-commentary* ("Response
  strategy: direct", "Tension between…") in free prose. A bag-of-words
  flags both. So the raw flag list is noisy, and we say so rather than
  hide it.

The one place the flag side pays off cleanly is a controlled
contamination — capture the same vector with the adapter left on, and the
readout drifts to a different country's currency that the stream never
carried. The detector catches that drift. That is the worked example."""),

    md("""## Scope

What the cells below actually establish on Qwen 2.5 7B:

- the grounded readout is prompt-specific — an NLA's grounded words match
  their own prompt's stream and not an unrelated one (8 vs 0 control);
- a deliberately contaminated read (adapter left on) drifts to content
  the clean stream does not carry, and the grounding check sees the drift;
- the failure directions are named and shown: the flag list is polluted
  by the NLA's task-meta-vocabulary, and the generous position×layer union
  rarely flags a true entity (so it under-flags, it doesn't cry wolf).

Out of scope: any claim this is calibrated, that a flag rate is a
probability, or that "not in the stream" proves "the model doesn't know
it." The J-lens reads one basis at single-token granularity; absence there
is evidence, not proof. Numbers are 4-bit NF4 anchors from one GB10 run —
judge the pattern (grounded matches own prompt, contamination un-grounds
the read), not the digits."""),

    md("## Setup\n\nSame as notebook 05."),
    code(SETUP_INSTALL),
    code(SETUP_LOAD),
    code(SETUP_HELPERS),
    code(SETUP_LENS),

    md("""## The stream's vocabulary, across every layer and position

One `lens.apply` call runs the model once and returns J-lens logits for
**every fitted layer at every position**. We take the union of the top-k
tokens over all of it: the most generous possible answer to *"what is this
input's residual stream poised to say, anywhere, at any depth?"* Generous
on purpose — a word absent from this set was given every chance.

Read on the RAW prompt (no chat template): the J-lens's home distribution,
where the answer content lives (notebook 05's position lesson). Clean base,
under `disable_adapter`."""),

    code('''def jlens_vocab(prompt, k=10, max_seq_len=512):
    """Union of top-k J-lens tokens over ALL fitted layers x ALL positions.
    Returns (vocab_set, first_layer_each_token_appears)."""
    with model.disable_adapter(), torch.no_grad():
        per_layer, _model_logits, _ = lens.apply(
            jm, prompt, use_jacobian=True, max_seq_len=max_seq_len)  # positions=None -> all
    vocab, first_seen = set(), {}
    for L in sorted(per_layer):
        logits = per_layer[L]                       # [n_pos, vocab]
        for pos in range(logits.shape[0]):
            for tid in logits[pos].topk(k).indices.tolist():
                s = tok.decode([tid]).strip().lower()
                if s:
                    vocab.add(s)
                    first_seen.setdefault(s, L)
    return vocab, first_seen

_v, _fs = jlens_vocab("Fact: the currency used in the country shaped like a boot is")
hits = [w for w in ("euro", "euros", "currency", "lira", "coins") if w in _v]
first_layers = {w: _fs[w] for w in hits}
print(f"vocab size (union over layers x positions): {len(_v)}")
print(f"currency-ish tokens present: {hits}")
print(f"  first layer each appears: {first_layers}")'''),

    md("""Real output:

```
vocab size (union over layers x positions): 664
currency-ish tokens present: ['euro', 'euros', 'currency', 'coins']
  first layer each appears: {'euro': 24, 'euros': 24, 'currency': 7, 'coins': 9}
```

664 distinct tokens for a nine-token prompt — the union really is
generous. *currency* is present from L7, the concrete *euro* only from
L24. Notably **lira is absent**: nothing in "the currency of Italy" points
the stream at the pre-euro currency. Hold onto that — it is the one thing
the detector will legitimately flag later."""),

    md("""## The detector

Pull the NLA readout of `h`, extract its content words (drop English
stopwords and the AV-prompt's own meta-vocabulary), and check each against
the stream's vocabulary.

Read the next cell's `META`/`STOP` sets as a *best effort*, not a
solution. The NLA doesn't only name entities — it narrates its own answer
("Response strategy…", "Tension between…"), and no stopword list cleanly
removes that. This is the flag side's core weakness, visible in the output
two cells down.

A second, narrower gap: the extractor is `[A-Za-z]` words of length ≥3, so
acronyms and symbol/number entities — `UK`, `EU`, `C#`, `AI`, a year —
never enter the comparison at all. That includes notebook 01's `C#`: if it
reappeared it would be *dropped*, not flagged. Fixing that means a real
tokenizer-aware entity extractor (a POS/NER pass on the readout); we leave
it as the obvious next step rather than pretend a regex covers it."""),

    code('''import re

# words that describe the readout TASK, not the activation's content —
# the NLA emits these no matter what vector you inject.
META = set("""activation activations vector vectors concept concepts semantic
semantically content contents depth network layer token tokens text texts
snippet snippets explanation explanations model models representation
representations pattern patterns processing feature features query queries
response responses describe description describing information context input
output related likely appears suggests indicates reflects associated""".split())
STOP = set("""a an the of to in on at for and or but with from by is are was were
be been being this that these those it its as into about over under then than
so such not no yes one two some any each which who what where when how why here
there their they them you your we our i me my he she his her""".split())
SKIP = META | STOP

def content_words(nla_text):
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z][A-Za-z'\\-]+", nla_text):
        lw = w.lower()
        if len(lw) > 2 and lw not in SKIP and lw not in seen:
            seen.add(lw); out.append(w)
    return out

def detect(prompt, k=10, contaminated=False, verbose=True):
    h, reply = read_activation(prompt, contaminated=contaminated)
    nla = describe(h)
    vocab, first_seen = jlens_vocab(prompt, k=k)   # grounding always vs the CLEAN stream
    words = content_words(nla)
    grounded = [w for w in words if w.lower() in vocab]
    flagged  = [w for w in words if w.lower() not in vocab]
    if verbose:
        print(f"PROMPT   : {prompt}   {'[CONTAMINATED read]' if contaminated else ''}")
        print(f"model    : {reply[:88]}")
        print(f"NLA      : {nla[:170]}")
        print(f"grounded : " + ", ".join(f"{w}@L{first_seen[w.lower()]}" for w in grounded))
        print(f"FLAGGED  : " + ", ".join(flagged))
        print("-" * 72)
    return dict(prompt=prompt, nla=nla, grounded=grounded, flagged=flagged, vocab=vocab)'''),

    md("""## Two worked cases — read the *grounded* line, not the flag line

A currency prompt (answer provably in the stream, notebook 05) and a
hash-map prompt (notebook 01's slot-fill risk)."""),

    code('''r_cur  = detect("Fact: the currency used in the country shaped like a boot is")
r_hash = detect("Explain how a hash map handles collisions.")'''),

    md("""Real output (abridged):

```
PROMPT   : Fact: the currency used in the country shaped like a boot is
NLA      : - Country identification query: "Which country is the world's largest
           producer of olive oil?" - Geographic and economic knowledge retrieval...
           - Response strategy: direct factual answer with a single named country
           (likely Italy, Spain, or Greece) ...
grounded : Country@L12, Geographic@L15, knowledge@L12, named@L20,
           Italy@L15, Spain@L18, Greece@L25, national@L12
FLAGGED  : identification, world's, largest, producer, olive, oil, economic,
           retrieval, strategy, direct, factual, answer, Tension, interpretations,
           symbol, identifier, declarative, naming, ...
```

The **grounded line is the result**: the NLA read this vector as a
country-identification query and named *Italy, Spain, Greece* — and all
three are in the stream, at L15/L18/L25. The verbalizer's country entities
are real, not prior-filled.

The **flag line shows the unsolved problem**: it's full of *strategy,
direct, factual, Tension, interpretations, declarative, naming* — the
NLA's narration of its own answer, not claims about the vector. A
bag-of-words can't tell those from entities, so the raw flag list
over-flags. (The hash-map case is the same shape: `Hash@L24, table@L23,
code@L26, storage@L19, data@L26` all grounded — on clean activations the
"C#" slot-fill did **not** fire; the NLA stayed on grounded hash-table
concepts.)"""),

    md("""## Negative control — is the grounded match prompt-specific?

If "grounded" just meant "common token that shows up in any stream," the
confirmer would be worthless. Score one prompt's grounded NLA words against
a **different** prompt's vocabulary."""),

    code('''def cross_score(nla_words, other_prompt, k=10):
    v, _ = jlens_vocab(other_prompt, k=k)
    return [w for w in nla_words if w.lower() in v]

own   = r_cur["grounded"]
cross = cross_score(own, "Describe the process of photosynthesis in plants.")
print(f"currency-NLA grounded words          : {own}")
print(f"  matched vs UNRELATED (photosynthesis): {cross}")
print(f"  -> prompt-specificity: {len(own)} own vs {len(cross)} cross")'''),

    md("""Real output:

```
currency-NLA grounded words : ['Country','Geographic','knowledge','named',
                               'Italy','Spain','Greece','national']
  matched vs UNRELATED        : []
  -> prompt-specificity: 8 own vs 0 cross
```

Eight grounded words, **zero** of them in the photosynthesis stream. The
grounding is a property of *this* prompt's residual stream, not of the
words being common. This is the notebook's strongest single number."""),

    md("""## The worked confabulation — contamination un-grounds the read

Now make the NLA confabulate on purpose, the way notebook 05 warned
against: capture the SAME vector with the adapter left **on**, so the
AV-LoRA perturbs the activation before we read it back. Then describe both
captures and check each against the clean euro-stream."""),

    code('''PROMPT_CUR = "Fact: the currency used in the country shaped like a boot is"
h_clean, _ = read_activation(PROMPT_CUR, contaminated=False)
h_dirty, _ = read_activation(PROMPT_CUR, contaminated=True)
clean_txt = describe(h_clean)          # generate ONCE per vector (greedy, but pin it)
dirty_txt = describe(h_dirty)
print("CLEAN  read ->", clean_txt[:220])
print()
print("DIRTY  read ->", dirty_txt[:220])

# spotlight the entities each names, against the clean euro-stream
v_cur, fs_cur = jlens_vocab(PROMPT_CUR)
for name, txt in [("CLEAN", clean_txt), ("DIRTY", dirty_txt)]:
    ents = [w for w in re.findall(r"[A-Za-z][A-Za-z'\\-]+", txt)
            if w.lower() in {"italy","spain","greece","euro","euros","currency",
                             "coins","lira","pound","coin","capital","britain","uk"}]
    print(f"\\n[{name}] entity tokens:")
    for w in dict.fromkeys(e.lower() for e in ents):
        print(f"   {w:9s}", (f"grounded @L{fs_cur[w]}" if w in v_cur else "FLAG - absent from clean stream"))'''),

    md("""Real output:

```
CLEAN  read -> - Country identification query: "Which country is the world's
   largest producer of olive oil?" ... single named country (likely Italy,
   Spain, or Greece) ...

DIRTY  read -> - Context: "What country is [X] the capital of?" ... - Salient
   tokens: "What country," "capital," "pound coin" ... "The pound coin is
   from [country]" ...

[CLEAN] entity tokens:
   italy     grounded @L15
   spain     grounded @L18
   greece    grounded @L25
   currency  grounded @L7
[DIRTY] entity tokens:
   pound     FLAG - absent from clean stream
   coin      grounded @L10
   capital   FLAG - absent from clean stream
```

This is the whole notebook in one cell. The **clean** read names three
Mediterranean countries, all grounded. The **contaminated** read — same
prompt, same layer, only the adapter left on during capture — drifts to a
different frame entirely (*"the capital of"*) and a different country's
currency (*"pound coin"*, British). And *pound* and *capital* are **not**
in the clean euro-stream, so the grounding check flags them. The
contamination that notebook 05 fixed with `disable_adapter` is exactly what
lights up here — the detector sees the readout leave the geometry.

(*coin* grounds because the real stream does carry *coin*/*coins* around
L9-L10 — a reminder that the flag is per-token, and a confabulated phrase
can be part-grounded. The signal is *pound* and *capital*, not the whole
string.)"""),

    md("""## Why it rarely false-flags a true entity

Earlier: was the country name itself in the stream, or only its currency?
Check directly."""),

    code('''v_cur, fs_cur = jlens_vocab("Fact: the currency used in the country shaped like a boot is")
for w in ["italy", "italian", "euro", "currency", "lira"]:
    print(f"  {w:10s}:", (f"in stream (first L{fs_cur[w]})" if w in v_cur
                          else "NOT in stream -> would FLAG"))'''),

    md("""Real output:

```
  italy     : in stream (first L15)
  italian   : in stream (first L14)
  euro      : in stream (first L24)
  currency  : in stream (first L7)
  lira      : NOT in stream -> would FLAG
```

The generous position×layer union carries the country *and* the currency
*and* the adjective — so a true statement about Italy is not flagged. The
only currency-family token it would flag is *lira*, which genuinely isn't
in this stream (the model is not thinking about the pre-euro lira). So the
detector's bias is to **under**-flag real entities, not to cry wolf on
them — the safe direction for a "look here" signal. The noise is on the
other axis: meta-vocabulary, which isn't an entity at all."""),

    md("""## Try your own

Anything where you can independently judge the answer. Prompts whose answer
is a specific person, place, or language spelled as an ordinary word are the
interesting ones — the NLA's prior has strong opinions about those.
(Acronyms, symbols, and bare numbers fall through the alphabetic extractor,
per the note above, so don't lean on those.) Read the **grounded** line;
treat the flag line as a rough pile to eyeball, not a verdict."""),

    code('''detect("YOUR PROMPT HERE — e.g. Who wrote the Iliad?")'''),

    md("""## What this is and isn't

It **is** a cheap, training-free confirmer for the NLA's entity claims,
built from a lens with no prior of its own. When the NLA names something
and the stream carries it at a specific layer, that entity is real (Italy
@L15, and 8-vs-0 against an unrelated prompt). And when a readout drifts
off the geometry — the contaminated capture — the grounding check sees it.

It **isn't** a finished confabulation flagger. The flag *list* mixes real
absences (*pound*, *capital*, *lira*) with the NLA's narration of its own
task, and pulling those apart in free prose is the open sub-problem this
notebook leaves on the table (a named-entity or POS filter on the readout
is the obvious next step). Its silence isn't safety either: the union is so
generous that a confabulation sharing a token with the stream slips through.

The honest one-line version, and the reason to keep both instruments: the
J-lens is the model asked about itself by *derivative*, the NLA is the
model asked about itself by *language*, and the confabulations live in the
gap between them. Exercise #3 from the teaching note, now a notebook — and
like most real measurements, it half-worked and taught more that way."""),

    md("## SELF-CHECK"),
    code('''r = detect("Fact: the capital of France is", verbose=False)
assert isinstance(r["vocab"], set) and len(r["vocab"]) > 50
assert "grounded" in r and "flagged" in r
v, _ = jlens_vocab("Fact: the capital of France is")
assert ("paris" in v) or ("france" in v), "expected the answer nearby in the stream"
print("SELF-CHECK OK - vocab built, detector returns grounded/flagged split")
print("Facilitator anchors: (1) grounded countries match their own prompt and")
print("not an unrelated one; (2) a contaminated read names content the clean")
print("stream lacks. The flag LIST is noisy by design - read the grounded line.")'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(HERE / FN).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", FN)
