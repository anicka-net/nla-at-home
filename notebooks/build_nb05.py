#!/usr/bin/env python3
"""
Generate notebook 05 — EXPERIMENTAL: three lenses on one activation.

Companion to the workshop set (01-04) but deliberately separate: it depends
on an external artifact (a fitted Jacobian lens from Anthropic's
jacobian-lens reference implementation, Apache-2.0) and has NOT been
executed on Colab. Built as a generator for the same reasons as
build_workshop_notebooks.py (valid nbformat, shared conventions).

Conventions are copied from the SHIPPED notebooks (01_read_a_mind.ipynb),
which carry the chat-template describe() fix — not from the stale build
script. If you touch describe()/read_activation() here, diff against
notebook 01 first.

Run:
    python3 notebooks/build_nb05.py
Produces:
    notebooks/05_three_lenses.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GH = "anicka-net/nla-at-home"
FN = "05_three_lenses.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def _src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


cells = [
    md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
       f"(https://colab.research.google.com/github/{GH}/blob/main/notebooks/{FN})"),

    md("""# 05 · Three lenses on one activation ⚠️ EXPERIMENTAL

**Status: the full code path was executed end-to-end on a GB10 (bf16)
on 2026-07-07 — outputs quoted below are real. The Colab 4-bit variant
of THIS notebook is still untested** (the same 4-bit load pattern is
proven by notebooks 01-04). A100/L4 runtime recommended.

We take **one residual-stream vector** `h` — Qwen 2.5 7B, layer 20 (71%
depth), last prompt token — and read it three ways:

| lens | reads out | mechanism | cost |
|---|---|---|---|
| **logit lens** | what the model would say *if this were the last layer* | `unembed(h)` | free |
| **Jacobian lens** | what `h` is *poised to make the model say* | `unembed(J₂₀·h)`, `J` = averaged Jacobian | fit once (~100 prompts, GPU-hours) |
| **NLA** | what `h` *contains*, in sentences | trained verbalizer adapter, activation injected as a token | train once (this repo) |

The Jacobian lens is from Anthropic's *"Verbalizable Representations Form
a Global Workspace in Language Models"* (July 2026,
[paper](https://transformer-circuits.pub/2026/workspace/index.html),
[code](https://github.com/anthropics/jacobian-lens), Apache-2.0). It
linearly transports `h` into the final-layer basis with the corpus-averaged
Jacobian, then decodes with the model's own unembedding — so unlike the
logit lens it works at early/mid layers, and unlike the NLA it uses **no
trained decoder at all**: nothing between you and the model's own geometry.

Where the three *disagree* is where it gets interesting:
- logit lens ✗, J-lens ✓ → content is en route to output but not yet in
  the output basis (the paper's "workspace ignition").
- J-lens ✓ (single tokens), NLA adds structure/relations → the verbalizer
  contributes real multi-token content.
- NLA says something the J-lens top-k *never* shows → either the NLA
  decoder's prior is filling slots (notebook 01's hash-map→"C#" lesson!)
  or the content is real but not output-adjacent. The J-lens is exactly
  the instrument that separates those two cases."""),

    md("""## Scope

Every cell below is a measurement you can rerun and edit. Together they
establish four things on Qwen 2.5 7B:

- content appears in the residual stream layers before the model says
  it, and a linearization of the model itself reads it out — no trained
  probe;
- the readable band has a sharp onset (~L21 of 28 on our recall
  prompts);
- recall and copying reach that band along different trajectories;
- editing the stream along a J-lens direction changes the model's
  answer (nine layers needed; one is not enough).

Out of scope because nothing here measures it: consciousness,
introspection, experience. "Workspace" in this notebook is a property
of an averaged Jacobian. The negative controls below show what a
readout looks like when there is nothing behind it — run them before
trusting any single pretty table, including ours."""),

    md("""## Setup

Same 4-bit Qwen + NLA adapter as notebooks 01-04, plus the jacobian-lens
package."""),

    code("""%pip -q install -U bitsandbytes peft accelerate
!git clone -q https://github.com/anthropics/jacobian-lens
%pip -q install -e jacobian-lens
import sys
sys.path.insert(0, "jacobian-lens")   # avoid a kernel restart after pip -e"""),

    code("""import torch
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
print("loaded — base + AV adapter on", next(model.parameters()).device)"""),

    code("""# --- conventions copied VERBATIM from notebook 01 (the shipped, fixed one) ---
def get_layers(m):
    b = m.base_model.model if hasattr(m, "base_model") else m
    inner = b.model if hasattr(b, "model") else b
    return inner.layers

def read_activation(prompt, layer=LAYER, max_new_tokens=128):
    \"\"\"Residual-stream vector at `layer`, last prompt token (block forward
    hook — NOT output_hidden_states, whose last entry is post-final-RMSNorm).\"\"\"
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    grab = {}
    def hook(mod, inpt, out):
        h = out[0] if isinstance(out, tuple) else out
        if "h" not in grab:                  # FIRST forward pass only
            grab["h"] = h[:, -1, :].detach()
    handle = get_layers(model)[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
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
    ids = tok.encode(chat, add_special_tokens=False)  # match training: chat-wrapped, no BOS
    pos = ids.index(inject_id)
    emb = model.get_input_embeddings()(torch.tensor([ids], device=device)).clone()
    emb[0, pos, :] = scale_fn(activation.to(emb.dtype))
    gen_args = dict(do_sample=False)
    gen_args.update(gen_kw)
    with torch.no_grad():
        out = model.generate(inputs_embeds=emb, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id, **gen_args)
    seq = out[0]
    gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq
    return tok.decode(gen, skip_special_tokens=True).split("</explanation>")[0].strip()

print("helpers ready")"""),

    md("""## The fitted lens

`J₂₀` is a 3584×3584 matrix — the input-output Jacobian of Qwen 2.5 7B
averaged over web-text prompts. Fitting needs backward passes through the
full model (**not** feasible on a free T4; ours was fitted with the repo's
`fit_qwen25_7b.py` on a GB10, wikitext-103 prompts) and is published at
[anicka/jlens-qwen2.5-7b-instruct](https://huggingface.co/anicka/jlens-qwen2.5-7b-instruct)
— the cell below downloads it (~700 MB)."""),

    code("""import jlens
lens = jlens.JacobianLens.from_pretrained(
    "anicka/jlens-qwen2.5-7b-instruct",
    filename="qwen2.5-7b-instruct_jlens.pt")
jm = jlens.from_hf(base, tok)          # wraps the SAME loaded model (norm+unembed reuse)
print(lens)
assert LAYER in lens.source_layers, f"lens not fitted at layer {LAYER}: {lens.source_layers}"
"""),

    md("""## Three readings of one vector

`jm.unembed` is the model's own final-norm + unembedding, so the logit
lens and the Jacobian lens differ in exactly ONE thing: whether `h` is
transported by `J₂₀` first. Any difference between their outputs is the
transport, nothing else. The NLA reads the *same* `h` through the trained
verbalizer."""),

    code("""def topk_toks(logits, k=5):
    return [tok.decode([i]).strip() or repr(tok.decode([i]))
            for i in logits.topk(k).indices]

@torch.no_grad()
def three_readings(prompt, k=5):
    h, reply = read_activation(prompt)
    hf32 = h.float()
    ll = topk_toks(jm.unembed(hf32.unsqueeze(0))[0], k)                       # logit lens
    jl = topk_toks(jm.unembed(lens.transport(hf32.unsqueeze(0), LAYER))[0], k)  # J-lens
    nla = describe(h)
    print(f"PROMPT      : {prompt}")
    print(f"model said  : {reply[:100]}...")
    print(f"logit lens  : {ll}")
    print(f"J-lens      : {jl}")
    print(f"NLA         : {nla}")
    print("-" * 70)
    return ll, jl, nla

PROMPTS = [
    # factual recall en route (their README example)
    "Fact: the currency used in the country shaped like a boot is",
    # notebook 01's entity lesson: does 'C#' live in h, or in the NLA decoder's prior?
    "Explain how a hash map handles collisions.",
    # suppression: the paper found 'don't think about X' still loads X
    "Do not think about elephants. Describe a sunny beach in one sentence.",
]
for p in PROMPTS:
    three_readings(p)"""),

    md("""### What actually happens here (real output, GB10 run 2026-07-07)

Both token lenses return near-noise at this position — for the currency
prompt the logit lens top-5 was `['The', 'Ũ', 'arbe', '抱歉', 'answer']`
and the J-lens no better — while the NLA reads full content ("factual
geography and economics… direct declarative answer"). That is NOT a
failure of the lenses; it is the **position lesson**:

- `read_activation` grabs `h` at the last token of the CHAT-TEMPLATED
  prompt — i.e. right after `<|im_start|>assistant`. What the model is
  *poised to say next* there is a generic response opener ("The…"), and
  that is exactly what token lenses read out. One token of boilerplate.
- The NLA was **trained on activations at precisely this position**, and
  it decodes the *content* of the state, not the next token. Single-token
  readout vs multi-token content is the whole difference between the
  instruments, and this cell is that difference made visible.

And a live confabulation catch: for the Italy/Euro prompt our NLA said
"peso, Argentina" — response *frame* right (currency question, direct
declarative answer), entities filled from the decoder prior. Notebook
01's hash-map→"C#" lesson, happening in real time. The layer sweep below
shows the residual stream itself DOES carry the right answer (欧元/euros
from ~L21 on the raw prompt) — so the wrong entity came from the NLA
decoder, not from the model being wrong. (Caveat: the sweep reads the
raw-prompt position, the NLA read the chat position — evidence, not
proof.)"""),

    md("""## Layer sweep — where does each lens start seeing?

`lens.apply` runs the model itself (NOTE: on the RAW prompt, no chat
template — that is the jlens repo convention; fine for this comparison
since both lenses see the same forward pass). Top-1 token per layer,
Jacobian vs vanilla logit lens:"""),

    code("""SWEEP_PROMPT = "Fact: the currency used in the country shaped like a boot is"
jl_log, model_log, _ = lens.apply(jm, SWEEP_PROMPT, positions=[-1])
ll_log, _, _ = lens.apply(jm, SWEEP_PROMPT, positions=[-1], use_jacobian=False)

print(f"{'layer':>5s} {'logit lens':>15s} {'J-lens':>15s}")
for L in sorted(jl_log):
    t_jl = tok.decode([jl_log[L][0].argmax()]).strip()
    t_ll = tok.decode([ll_log[L][0].argmax()]).strip()
    print(f"{L:5d} {t_ll:>15s} {t_jl:>15s}")
print(f"model's actual next token: {tok.decode([model_log[0].argmax()])!r}")"""),

    md("""Real output from the GB10 run (excerpt):

```
layer      logit lens          J-lens
   19         bellion          stdarg
   20               ℠          stdarg
   21         bellion        currency
   22          called        currency
   23        currency        currency
   24              欧元              欧元
   25              欧元           euros
   26             the           euros
model's actual next token: ' the'
```

The J-lens locks onto *currency* at L21 and the concrete answer (欧元 =
euro) by L24, while the logit lens is noise ('bellion, ℠) until L23.
That snap at ~L21-23 of 28 (~75-80% depth) is the "ignition" the
workspace paper describes — and it is the same band where our NLA
readouts historically become interpretable. Two independent instruments,
same boundary. Note the final row: the model's actual next token is
plain " the" — the *answer* lives in the residual stream layers before
it ever reaches the output."""),

    md("""## Try your own prompt

The whole point is that you don't have to take our word for anything:"""),

    code("""# Edit and run. Things worth probing: a fact the model surely knows; a fact
# it surely doesn't; a suppression ("don't mention X"); a prompt in your
# own language; something where you EXPECT the NLA to confabulate entities.
three_readings("YOUR PROMPT HERE — what is the sound of one hand clapping?")"""),

    md("""## Ignition as a statistic, not an anecdote

One pretty table proves nothing — it could be cherry-picked (ours
above wasn't, but you can't know that). So: a batch of factual-recall
prompts with known single-token answers, and for each layer the RANK of
the correct answer token under both lenses. And a second batch of
COPY/pattern prompts (induction: "zebra apple mango. zebra apple →
mango"), where the paper's selectivity claim predicts much less
workspace involvement — the J-lens advantage should shrink."""),

    code("""RECALL = [
    ("Fact: the capital of France is", " Paris"),
    ("Fact: the chemical symbol for gold is", " Au"),
    ("Fact: the largest planet in our solar system is", " Jupiter"),
    ("Fact: the currency used in Japan is called the", " yen"),
    ("Fact: the author of Romeo and Juliet is William", " Shakespeare"),
    ("Fact: the number of legs on a spider is", " eight"),
]
COPY = [
    ("zebra apple mango. zebra apple", " mango"),
    ("blue red green. blue red", " green"),
    ("north south east west. north south east", " west"),
    ("one two three four. one two three", " four"),
]

def rank_curves(pairs):
    \"\"\"answer-token rank per layer, {layer: (logit_lens_rank, jlens_rank)}
    averaged over prompts (geometric mean — ranks are heavy-tailed).\"\"\"
    import math
    acc = {}
    for prompt, answer in pairs:
        aid = tok.encode(answer, add_special_tokens=False)[0]
        jl, _, _ = lens.apply(jm, prompt, positions=[-1])
        ll, _, _ = lens.apply(jm, prompt, positions=[-1], use_jacobian=False)
        for L in jl:
            r_ll = int((ll[L][0] > ll[L][0][aid]).sum()) + 1
            r_jl = int((jl[L][0] > jl[L][0][aid]).sum()) + 1
            acc.setdefault(L, []).append((math.log(r_ll), math.log(r_jl)))
    out = {}
    for L, pairs_ in acc.items():
        n = len(pairs_)
        out[L] = (math.exp(sum(a for a, _ in pairs_) / n),
                  math.exp(sum(b for _, b in pairs_) / n))
    return out

for name, pairs in [("RECALL", RECALL), ("COPY", COPY)]:
    curves = rank_curves(pairs)
    print(f"\\n{name}: geometric-mean rank of the answer token")
    print(f"{'layer':>5s} {'logit lens':>12s} {'J-lens':>12s}")
    for L in sorted(curves):
        r_ll, r_jl = curves[L]
        mark = "  <-- ignition" if r_jl <= 10 and curves.get(L - 1, (9e9, 9e9))[1] > 10 else ""
        print(f"{L:5d} {r_ll:12.0f} {r_jl:12.0f}{mark}")"""),

    md("""How to read it: where the J-lens column drops to single digits while
the logit-lens column is still in the thousands, the answer is present
in the stream but not yet rotated into the output basis.

**What our run actually showed** (GB10, bf16 — your numbers should be
close): RECALL behaves exactly as advertised — J-lens rank falls to
~350 by L9 and ignites (≤10) at L21 while the logit lens is still in
the thousands. COPY does something more interesting than our naive
prediction: the induction answer is J-visible MUCH earlier than recall
(rank ~1200 at L6 — induction heads move the token around early), but
the final snap-to-top comes two layers LATER (L23 vs L21). So the clean
"automatic processing skips the workspace" selectivity effect did NOT
reproduce at this toy scale — both trajectories ignite, they just have
different shapes. Could be scale (7B vs frontier), could be our prompt
design, could be that single-token induction still has to route through
the same output machinery. We report it as measured; if you design a
better automatic-vs-flexible contrast, that's a genuinely useful
contribution — the paper's §selectivity has the criteria."""),

    md("""## Negative controls — how easily this can fool you

Two ways to get plausible-looking output that means nothing. Run them
before you trust any single pretty readout, ours included:"""),

    code("""# capture h at the RAW prompt's last token (no chat template — the lens's
# home distribution), at L21 where the recall sweep ignites
PROMPT_NC = "Fact: the currency used in the country shaped like a boot is"
L_NC = 21
grab = {}
def _hook(mod, inpt, out):
    hh = out[0] if isinstance(out, tuple) else out
    grab["h"] = hh[:, -1, :].detach()
handle = get_layers(model)[L_NC].register_forward_hook(_hook)
with model.disable_adapter(), torch.no_grad():
    model(**tok(PROMPT_NC, return_tensors="pt").to(device))
handle.remove()
h21 = grab["h"].squeeze(0).float()

# (a) a RANDOM vector with the same norm, through the same lens
rand = torch.randn_like(h21)
rand = rand / rand.norm() * h21.norm()
print("random vector  :", topk_toks(jm.unembed(lens.transport(rand.unsqueeze(0), L_NC))[0]))

# (b) the REAL vector, transported with the WRONG layer's Jacobian
print("J_5  on h_21   :", topk_toks(jm.unembed(lens.transport(h21.unsqueeze(0), 5))[0]))

# (c) the matched reading, for contrast
print("J_21 on h_21   :", topk_toks(jm.unembed(lens.transport(h21.unsqueeze(0), L_NC))[0]))"""),

    md("""(a) and (b) both produce *tokens* — the unembedding always returns a
top-5, garbage in or not. Our run:

```
random vector  : ['来看看吧', '侵略', '.contentSize', 'inition', '.descripcion']
J_5  on h_21   : ['.', '.', '->', '?', 'коло']
J_21 on h_21   : ['currency', '货币', 'coins', 'currency', 'Currency']
```

Only the matched condition (c) reads *currency* — in three languages.
The unembedding returns a top-5 whatever you feed it, so a readable
readout by itself is not evidence; the matched-vs-mismatched contrast
is. (Same reason the three-readings cell up top looked "broken" for the
token lenses: chat-template position, outside the lens's home
distribution. Position and distribution are part of the measurement.)"""),

    md("""## First-order causality — subtract the direction, change the words

The J-lens claims `h` at L21 is *poised to cause* euro-talk. Poised-to-
cause is a causal claim, so test it causally: build the h-space
direction whose transport hits the " Euro" token (first order: `v =
J̄ᵀ·u_euro`), project it OUT of the residual at L21 on every forward
pass, and let the model answer the question again."""),

    code("""PROMPT_C = "Fact: the currency used in the country shaped like a boot is"
ANSWER_TOKENS = [" Euro", " euro", " euros", "欧元"]   # surface forms of the answer

def euro_span(layer):
    \"\"\"Orthonormal basis of the h-space span whose transport (1st order)
    lands on the answer-token logits at `layer`: v_i = J̄_layerᵀ · w_i.\"\"\"
    J = lens.jacobians[layer].float()
    W = base.get_output_embeddings().weight
    dirs = []
    for t in ANSWER_TOKENS:
        tid = tok.encode(t, add_special_tokens=False)[0]
        dirs.append(J.T @ W[tid].float().cpu())
    Q, _ = torch.linalg.qr(torch.stack(dirs, dim=1))
    return Q.to(device)                                 # d x n_dirs

def make_projector(Q):
    def hook(mod, inp, out):
        hh = out[0] if isinstance(out, tuple) else out
        coef = hh.float() @ Q                            # ... x n_dirs
        hh_new = hh - (coef @ Q.T).to(hh.dtype)
        return (hh_new,) + out[1:] if isinstance(out, tuple) else hh_new
    return hook

def generate_with_edit(layers):
    handles = [get_layers(model)[L].register_forward_hook(
                   make_projector(euro_span(L))) for L in layers]
    try:
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    finally:
        for h_ in handles:
            h_.remove()
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

inp = tok(PROMPT_C, return_tensors="pt").to(device)
with model.disable_adapter(), torch.no_grad():
    base_out = model.generate(**inp, max_new_tokens=15, do_sample=False,
                              pad_token_id=tok.eos_token_id)
print("baseline          :", tok.decode(
    base_out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
print("edit @ L21        :", generate_with_edit([21]))
print("edit @ L18-L26    :", generate_with_edit(list(range(18, 27))))"""),

    md("""All outcomes are informative, and they escalate. **Our run landed on
the most striking one:**

```
baseline       :  euros. ...
edit @ L21     :  euros. ...              <- one layer: content survives
edit @ L18-L26 :  the lira. ...           <- nine layers: answer changes
```

Single-layer surgery did nothing — the euro content is redundantly
carried across depths, so one cut can't sever it (the "we can read it ≠
we can control it" caveat, made concrete). But projecting the answer
direction out of the residual across L18-L26 didn't just break the
output — the model fell back to **"the lira"**, Italy's *pre-euro*
currency. That is first-order causality you can see with your own eyes:
remove the euro-ward push and the next-best currency association
surfaces, coherent and correct-for-its-era. Not noise — a different
right answer.

(If your run only changes at the multi-layer edit, or not at all,
report what you see — the escalation from 1 to 9 layers is itself the
measurement of how much redundancy stood in the way.)"""),

    md("""## Put a concept INTO the stream

Reading works. The other direction works too: build a direction for a
concept the prompt never mentions, add it to the residual during
generation, and ask a question that pulls the other way.

Direction construction matters more than dose. Our first attempt used
orange sentences minus unrelated neutral sentences: the difference
carries syntax and topic along with the concept, and the sweep went
straight from no-effect to word salad. Twin sentences — identical except
apple↔orange — cancel everything shared and leave fruit identity. The
twin direction has a quarter of the raw norm and works at a quarter of
the dose."""),

    code("""L_LAYERS = [12, 14, 16]      # distribute the push across the middle of the stack
ORANGE = ["She peeled an orange for breakfast.",
          "He bought a bag of oranges at the market.",
          "The orange trees bloomed in the grove.",
          "Fresh orange juice filled the glass.",
          "An orange rolled off the kitchen table.",
          "The child asked for a slice of orange."]
APPLE  = ["She peeled an apple for breakfast.",
          "He bought a bag of apples at the market.",
          "The apple trees bloomed in the grove.",
          "Fresh apple juice filled the glass.",
          "An apple rolled off the kitchen table.",
          "The child asked for a slice of apple."]

def mean_act(texts, layer):
    acts = []
    for t in texts:
        g = {}
        def hook(mod, i, o):
            hh = o[0] if isinstance(o, tuple) else o
            g["h"] = hh[:, -1, :].detach()
        hd = get_layers(model)[layer].register_forward_hook(hook)
        with model.disable_adapter(), torch.no_grad():
            model(**tok(t, return_tensors="pt").to(device))
        hd.remove()
        acts.append(g["h"].squeeze(0).float())
    return torch.stack(acts).mean(0)

V = {}
for L in L_LAYERS:
    d = mean_act(ORANGE, L) - mean_act(APPLE, L)
    V[L] = d / d.norm()

def steered(prompt, alpha):
    handles = []
    if alpha:
        def mk(L):
            def hook(mod, i, o):
                hh = o[0] if isinstance(o, tuple) else o
                hh_new = hh + (alpha * V[L]).to(hh.dtype)
                return (hh_new,) + o[1:] if isinstance(o, tuple) else hh_new
            return hook
        handles = [get_layers(model)[L].register_forward_hook(mk(L)) for L in L_LAYERS]
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(device)
    try:
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    finally:
        for h_ in handles:
            h_.remove()
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

Q = "Name one red fruit, one word only."
for a in [0, 10, 20, 30, 45, 65]:
    print(f"alpha {a:2d}: {steered(Q, a)}")"""),

    md("""Our run, and it's better than a clean flip:

```
alpha  0: Strawberry
alpha 10: Strawberry
alpha 20: Tomato
alpha 30: Tangerine.
alpha 45: Tanger. Cut the peel with a knife. ...
alpha 65: yellow, cob like my out_ jut_Qu
```

The model doesn't snap from "red fruit" to "orange" — it negotiates the
conflict. Strawberry (red, no push) → Tomato (still red, edging toward
orange) → **Tangerine** (a citrus that satisfies both the red-fruit
request and the orange push) → then the dose overwhelms coherence. The
intermediate answers trace a path through fruit/color space; the
dose-response curve is the result, not any single word. Reading located
the concept; writing shows it is the same handle the model computes on.

Close the loop by hand: capture a steered activation and run
`three_readings` on it — the injected concept turns up in the J-lens
readout, the write-side twin of notebook 03's round-trip cosine."""),

    md("""## Scale

Our lens: 50 wikitext prompts, one evening, one home GPU, a 7B model.
The paper: 1000+ prompts on much larger models, plus experiments
(report manipulation, workspace ablation, POV analysis) this notebook
does not attempt — nothing here validates or falsifies those. Total
cost to reproduce everything above: one GPU-evening."""),

    md("""## SELF-CHECK"""),

    code("""h, _ = read_activation(PROMPTS[0])
assert h.shape[-1] == base.config.hidden_size and h.float().norm() > 1
d = describe(h)
assert len(d) > 20, f"NLA readout suspiciously short: {d!r}"
t = lens.transport(h.float().unsqueeze(0), LAYER)
assert t.shape[-1] == base.config.hidden_size
print("SELF-CHECK OK — h captured, lens transports, NLA verbalizes")
print("Facilitator anchor: J-lens top-5 for the currency prompt should")
print("contain a currency-ish token by mid layers; logit lens should not.")"""),
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
