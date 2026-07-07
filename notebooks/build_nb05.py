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
