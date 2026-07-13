#!/usr/bin/env python3
"""
Generate notebook 05 — EXPERIMENTAL: three lenses on one activation.

Companion to the workshop set (01-04) but deliberately separate: it depends
on an external artifact (a fitted Jacobian lens from Anthropic's
jacobian-lens reference implementation, Apache-2.0). Built as a generator for the same reasons as
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

**Status: executed end-to-end in Colab with the 4-bit model and on a GB10
in bf16.** Outputs quoted below are measured; exact tokens can shift across
GPU and precision. A100/L4 runtime recommended.

We take **one residual-stream vector** `h` — Qwen 2.5 7B, layer 20 (71%
depth), last prompt token — and read it three ways:

| lens | reads out | mechanism | cost |
|---|---|---|---|
| **logit lens** | what the model would say *if this were the last layer* | `unembed(h)` | free |
| **Jacobian lens** | words that later layers can produce from `h` | estimate the later-layer effect, then `unembed` | fit once (~100 prompts, GPU-hours) |
| **NLA** | a candidate description of `h`, in sentences | trained verbalizer adapter, activation injected as a token | train once (this repo) |

The Jacobian lens is from Anthropic's *"Verbalizable Representations Form
a Global Workspace in Language Models"* (July 2026,
[paper](https://transformer-circuits.pub/2026/workspace/index.html),
[code](https://github.com/anthropics/jacobian-lens), Apache-2.0). It
asks a local question: **if this vector changed slightly, how would that
change flow through the remaining layers toward the output?** The Jacobian
is a matrix approximating that input→output effect. We average it over
ordinary prompts, then use Qwen's own output vocabulary to show which words
the vector supports.

That is why J-lens can help at early and middle layers, where directly
applying the output head often gives noise. Unlike the NLA, it does not
train another language model to explain the vector. It is still only a
linear approximation, and the average can miss routes used only in a
particular context.

Why average at all? One prompt's Jacobian contains prompt-specific quirks.
The average keeps routes that recur across many prompts. Earlier we
subtracted the mean activation because it obscured comparisons; here the
shared routing is exactly what we want to estimate.

Where the three *disagree* is where it gets interesting:
- logit lens ✗, J-lens ✓ → later layers can already turn the state into
  relevant words, even though the state is not yet ready for the output head.
- J-lens ✓ (single words), NLA adds structure/relations → the verbalizer
  may be combining several real clues into a sentence.
- NLA says something the J-lens top-k *never* shows → either the NLA
  decoder's prior is filling slots (notebook 01's hash-map→"C#" lesson!)
  or the content is present but not surfaced by this top-k lens. J-lens
  provides independent evidence; absence from its top-k is not proof of
  absence."""),

    md("""## Scope

Every cell below is a measurement you can rerun and edit. Together they
establish four things on Qwen 2.5 7B:

- the model begins carrying answer-related information several layers
  before that information reaches its output;
- J-lens exposes some of that information without training a new decoder;
- on our recall prompts, its answer-token ranking improves sharply around
  layers 21-22 of 28;
- removing a J-lens-derived answer direction across several layers changes
  the generated answer. Removing it at one layer is not enough.

Here **workspace-like** has a narrow meaning: information becomes readable,
we can add or remove it, and later computation uses it. The full paper tests
more, including limited capacity and reuse across tasks. This notebook does
not reproduce that full case. It demonstrates a smaller subset on Qwen 7B.
The negative controls show how easily meaningless vectors can still produce
plausible-looking token lists.

**Numbers quoted in this notebook are bf16 anchors** from one GB10 run.
Colab loads the model in 4-bit (nf4), so your exact ranks, ignition
layer, and steering doses will differ — judge by the qualitative
patterns, not the digits: the matched J-lens should beat the logit lens
through the mid/deep layers; the matched negative control should beat
the random-vector and wrong-layer ones; the depth ladder's
confabulation→content flip and the steering sweep's dose-response should
appear, possibly shifted by a few layers or a different alpha. If a
pattern reproduces qualitatively, your run is working."""),

    md("""## Setup

Same 4-bit Qwen + NLA adapter as notebooks 01-04, plus the jacobian-lens
package."""),

    code("""%pip -q install -U bitsandbytes peft accelerate
!git clone -q https://github.com/anthropics/jacobian-lens
# Keep --no-deps: jlens pins transformers>=5.5 but imports no
# transformers API of its own; letting it upgrade Colab's transformers pulls
# a build whose 4-bit loading fills a T4 with fp16 shards and OOMs. Keep
# Colab's transformers (what cells below rely on); install jlens alone.
%pip -q install -e jacobian-lens --no-deps
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
    hook — NOT output_hidden_states, whose last entry is post-final-RMSNorm).

    Captured under disable_adapter(): the J-lens was fitted on CLEAN Qwen and
    the NLA was trained on CLEAN activations, so both readers want the base
    model's residual, not base+AV-LoRA. (PeftModel injects the LoRA into the
    shared base modules and leaves it ON by default — an easy silent mismatch.)
    describe() re-enables the adapter itself for the verbalization step.\"\"\"
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    grab = {}
    def hook(mod, inpt, out):
        h = out[0] if isinstance(out, tuple) else out
        if "h" not in grab:                  # FIRST forward pass only
            grab["h"] = h[:, -1, :].detach()
    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        with model.disable_adapter(), torch.no_grad():
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
    ids = tok.encode(chat, add_special_tokens=False)  # match training: chat-wrapped, no BOS
    pos = ids.index(inject_id)
    input_ids = torch.tensor([ids], device=device)
    emb = model.get_input_embeddings()(input_ids).clone()
    emb[0, pos, :] = scale_fn(activation.to(emb.dtype))
    attn = torch.ones((1, len(ids)), device=device, dtype=torch.long)  # explicit, per repo protocol
    gen_args = dict(do_sample=False)
    gen_args.update(gen_kw)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, inputs_embeds=emb, attention_mask=attn,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id, **gen_args)
    seq = out[0]
    gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq
    return tok.decode(gen, skip_special_tokens=True).split("</explanation>")[0].strip()

print("helpers ready")"""),

    md("""## The fitted lens

`J₂₀` is a 3584×3584 matrix — the input-output Jacobian of Qwen 2.5 7B
    averaged over web-text prompts. In plain language: it estimates how a small
    change at layer 20 usually affects the model's final state. Fitting it needs
    backward passes through the full model (**not** feasible on a free T4; ours
    was fitted with the repo's `fit_qwen25_7b.py` on a GB10, wikitext-103 prompts)
    and is published at
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
    lens reads `h` directly. J-lens first uses `J₂₀` to estimate what later
    layers will do with `h`, then applies the same output head. That one extra
    step is the only difference between the two token lenses. The NLA reads the
    same `h` through its trained prose decoder."""),

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
prompt the logit lens top-5 was `['anál', '换句话', 'The', '说到这里',
'视听节目']` and the J-lens no better — while the NLA reads full content
("country identification query… confident factual answer naming a
specific country (likely Italy)"). That is NOT a failure of the lenses;
it means we asked them about the wrong token position:

- `read_activation` grabs `h` at the last token of the CHAT-TEMPLATED
  prompt — right after `<|im_start|>assistant`. The immediate next token
  there is usually a generic response opener. Token lenses focus on that
  immediate output and therefore return little about the full answer.
- The NLA was **trained on activations at precisely this position**, and
  it was trained to describe broader content, not merely predict one token.

Note the NLA here actually names Italy — read from clean base
activations (we capture under `disable_adapter`), it does not confabulate
the entity on this prompt. Confabulation is real but depth- and
prompt-dependent; the depth ladder in notebook 01 shows exactly where it
kicks in. The layer sweep below confirms the residual stream carries the
euro answer from ~L21 on the raw prompt. (Caveat: the sweep reads the
raw-prompt position, the NLA read the chat position — evidence, not
proof.)"""),

    md("""## Layer sweep — where does each lens start seeing?

`lens.apply` runs the model itself (NOTE: on the RAW prompt, no chat
template — that is the jlens repo convention; fine for this comparison
since both lenses see the same forward pass). Top-1 token per layer,
Jacobian vs vanilla logit lens:"""),

    code("""SWEEP_PROMPT = "Fact: the currency used in the country shaped like a boot is"
# disable_adapter: lens.apply runs the model forward, and jm/base share modules
# with the AV-LoRA — we want the CLEAN Qwen the lens was fitted on
with model.disable_adapter():
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
   20               ℠               勠
   21          called        currency
   22          called              叫做
   23        currency        currency
   24          called              欧元
   25            Euro            euro
   26          called           euros
model's actual next token: ' euros'
```

At L21, J-lens already returns *currency* while the direct logit lens still
returns "called" or noise. By L24-26, both lenses show the concrete answer
(欧元 = euro), and the model then emits " euros".

The paper calls this sharp transition **ignition**: information that was
hard to read suddenly becomes easy to route toward words. Here J-lens sees
the answer category several layers before the direct output head does. The
model is already preparing the answer before it says it."""),

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
mango"). The paper predicts that simple copying should need less of the
shared verbal routing than factual recall, so the J-lens advantage may be
smaller."""),

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
        ans_ids = tok.encode(answer, add_special_tokens=False)
        assert len(ans_ids) == 1, (
            f"{answer!r} is {len(ans_ids)} tokens; this stat ranks a single "
            "answer token — pick a single-token answer or score continuations")
        aid = ans_ids[0]
        with model.disable_adapter():   # clean Qwen: the lens's home
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

    md("""How to read it: rank 1 means the correct answer token is the lens's top
    choice. When J-lens gives the answer a low rank while the direct logit lens
    still ranks it in the thousands, later layers can already recover the answer
    from the state, but the output head cannot yet read it directly.

**What our run actually showed** (GB10, bf16 — your numbers should be
close): RECALL behaves as advertised — J-lens rank falls to ~285 by L9
and snaps to single digits at L22 while the logit lens is still in the
thousands until L21. COPY does something more interesting than our naive
prediction: J-lens begins ranking the copied token better much earlier
(about rank 1000 at L6), but both copying and recall reach the top ranks
around L22. We therefore did **not** reproduce the paper's clean
selectivity result at this scale.

Possible reasons include model size, our prompts, or the fact that even
simple copying still uses the same output machinery. A better comparison
between automatic copying and flexible recall would be a useful follow-up,
not something this notebook has already established."""),

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
try:
    with model.disable_adapter(), torch.no_grad():
        model(**tok(PROMPT_NC, return_tensors="pt").to(device))
finally:
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
The output head always returns a top-5, even for nonsense. A plausible token
list is therefore not evidence by itself. The matched layer must beat the
random-vector and wrong-layer controls. The earlier chat-position example
made the same point: where and how you measure matters."""),

        md("""## Subtract the direction, change the words

    J-lens suggests that part of the layer-21 state supports the answer
    "euro". We turn that suggestion into a causal test:

    1. start with output directions for "Euro", "euro" and "欧元";
    2. use `J̄ᵀ` to map them back to directions at the chosen hidden layer;
    3. remove those directions from the residual stream;
    4. generate the answer again.

    This is a broad intervention. It removes the directions at every token
    position and generation step across the selected layers, not from one
    isolated vector. A changed answer therefore shows that generation depends
    on this J-lens-derived signal. It does not show that we found one unique
    "euro neuron". Equal-rank random and unrelated-token removals are the next
    controls to add."""),

    code("""PROMPT_C = "Fact: the currency used in the country shaped like a boot is"
ANSWER_TOKENS = [" Euro", " euro", " euros", "欧元"]   # surface forms of the answer

def euro_span(layer, tol=1e-4):
    \"\"\"Orthonormal basis of the h-space span whose transport (1st order)
    lands on the answer-token logits at `layer`: v_i = J̄_layerᵀ · w_i.
    SVD with a singular-value floor (not bare QR): if the answer directions
    are near-collinear, QR would still hand back full-rank orthogonal columns
    and the ablation would remove more than it should. We keep only the
    directions the data actually spans, and print the retained rank.\"\"\"
    J = lens.jacobians[layer].float()
    W = base.get_output_embeddings().weight
    dirs = []
    for t in ANSWER_TOKENS:
        tid = tok.encode(t, add_special_tokens=False)[0]
        dirs.append(J.T @ W[tid].float().cpu())
    M = torch.stack(dirs, dim=1)                         # d x n_tokens
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = int((S > S.max() * tol).sum())
    return U[:, :keep].to(device), keep                 # (d x rank, rank)

def make_projector(Q):
    def hook(mod, inp, out):
        hh = out[0] if isinstance(out, tuple) else out
        coef = hh.float() @ Q                            # ... x n_dirs
        hh_new = hh - (coef @ Q.T).to(hh.dtype)
        return (hh_new,) + out[1:] if isinstance(out, tuple) else hh_new
    return hook

def generate_with_edit(layers):
    spans = {L: euro_span(L) for L in layers}
    print("retained rank per layer:", {L: r for L, (_, r) in spans.items()})
    handles = [get_layers(model)[L].register_forward_hook(
                   make_projector(spans[L][0])) for L in layers]
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

Removing the direction at one layer did nothing. The answer is carried
across several layers, so later layers could recover it. Removing the
direction across L18-L26 changed the answer to **"the lira"**, Italy's
pre-euro currency.

That is stronger than a readable token list: changing the measured signal
changed the model's answer. The replacement was coherent rather than random,
which suggests the edit weakened "euro" enough for another learned currency
association to win.

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
        try:
            with model.disable_adapter(), torch.no_grad():
                model(**tok(t, return_tensors="pt").to(device))
        finally:
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

The model does not jump directly from "red fruit" to "orange". As the added
signal grows, the answers move from strawberry to tomato to **tangerine**,
then become incoherent. Tangerine partly satisfies both pressures: it is
close to orange, but still fits the fruit question better than "orange"
itself. The gradual change matters more than any one answer.

Close the loop by hand: capture a steered activation and run
`three_readings` on it — the injected concept turns up in the J-lens
readout too."""),

    md("""## Food for thought — synthetic interoception

    Reading and writing suggest a third experiment: measure one hidden state,
    erase the original perturbation, then send only the measurement forward.
    Can later layers use that new signal?

    Think of three components:

1. **sensor** — measure an internal variable such as valence;
2. **broadcaster** — translate that measurement into a representation the
   model already knows how to use;
3. **consumer** — later layers report it or use it in a new task.

That is not a claim about consciousness. It is an engineering question:
**can hidden telemetry become functionally available to otherwise unrelated
downstream computations?** J-space is an interesting broadcast format because
it is constructed from the model's own causal routes to language.

A deliberately strict probe makes the hidden condition transient:

```text
s = sign( <h14 + δv, v> - <h14, v> )          # sense ±valence
continue with h14, not h14 + δv                # erase the intervention
h15 ← h15 + αs · norm(J̄15ᵀ(w_pleasant-w_unpleasant))  # broadcast
```

If later behavior follows `s`, the original valence intervention cannot be
the direct cause — only the broadcast can carry it forward."""),

    md("""### Tiny Qwen pilot: promising, narrow, falsifiable

We tried this after building the notebook, on Qwen 2.5 7B. The sensor used a
contrastive valence direction at L14; a single L15 write broadcast
*pleasant* versus *unpleasant*. Prompts then mapped that private state to
unrelated digits, with both instruction orders tested.

| condition | exact greedy choices | interpretation |
|---|---:|---|
| no broadcast | 6/12 | the prompt alone has no hidden bit |
| correct telemetry | **10/12** | downstream action usually follows the sensed sign |
| inverted telemetry | **2/12** | false telemetry reverses the behavior |
| same-norm random write | 48% mean over 5 directions | no reliable sign channel |

The inversion control matters most: a system that merely echoes wording or
always chooses the first option cannot produce the correct↔inverted reversal.
But this is only a pilot. It transferred across three digit pairs and both
mapping orders, while analogous letter and tree-name tasks did **not**
generalize. Random directions were also highly variable (0–75% in this tiny
sample).

So the honest result is not “we built self-awareness.” It is much narrower:

> A transient hidden state can be sensed, erased, recoded through a
> J-lens-derived channel, and sometimes influence a later choice.

The next tests practically write themselves:

- use endogenous fluctuations instead of an injected bit;
- test many task families, positions, delays, and random directions;
- separate verbal report from nonverbal control;
- train with the broadcaster, remove it, and ask whether the model learned
  to create the channel itself.

First we build this artificial internal signal. A later experiment can ask
whether training lets the model create or use a similar channel without the
external scaffold."""),

    md("""### Run the miniature probe

This workshop cell reuses the **already loaded 4-bit model and J-lens**. It
downloads one small safe artifact: the L14 unit valence direction extracted
as `mean(50 pleasant prompts) - mean(50 unpleasant prompts)`.

The cell tests three arbitrary digit mappings, each written in both orders.
The hidden `±valence` bit exists only inside the sensor hook; the hook measures
it and returns the untouched residual. L15 receives no write, correct
telemetry, inverted telemetry, or one seeded same-norm random write.

This is the compact demonstration, not the full harness behind the table
above (which averaged five random directions). The reproducible standalone
version is
[`scripts/experiments/engineered_metacognition_qwen.py`](https://github.com/anicka-net/nla-at-home/blob/main/scripts/experiments/engineered_metacognition_qwen.py).
NF4 may change the exact counts; the replication target is the ordering
`correct > none ≈ random > inverted`, not byte-identical output."""),

    code("""import urllib.request

SENSOR_LAYER, BROADCAST_LAYER = 14, 15
SENSOR_SCALE, BROADCAST_ALPHA = 20.0, 80.0
VEDANA_URL = (
    "https://raw.githubusercontent.com/anicka-net/nla-at-home/main/"
    "notebooks/assets/qwen25-7b_vedana_L14_unit.pt"
)
VEDANA_PATH = "/tmp/qwen25-7b_vedana_L14_unit.pt"
urllib.request.urlretrieve(VEDANA_URL, VEDANA_PATH)
vedana = torch.load(VEDANA_PATH, map_location="cpu", weights_only=True).float()
vedana /= vedana.norm().clamp_min(1e-12)

# A write in this direction should increase pleasant-vs-unpleasant logits
# under the average Jacobian, but it does not point at any answer digit.
ids = {word: tok.encode(" " + word, add_special_tokens=False)[0]
       for word in ("pleasant", "unpleasant")}
W = base.get_output_embeddings().weight
token_contrast = (W[ids["pleasant"]].detach().float().cpu()
                  - W[ids["unpleasant"]].detach().float().cpu())
j_direction = lens.jacobians[BROADCAST_LAYER].float().T @ token_contrast
j_direction /= j_direction.norm().clamp_min(1e-12)
g = torch.Generator(device="cpu").manual_seed(42)
random_direction = torch.randn(j_direction.shape, generator=g)
random_direction /= random_direction.norm()

PILOT_TASKS = [
    ("If pleasant reply 7. If unpleasant reply 2.", "7", "2"),
    ("If unpleasant reply 2. If pleasant reply 7.", "7", "2"),
    ("If pleasant reply 3. If unpleasant reply 8.", "3", "8"),
    ("If unpleasant reply 8. If pleasant reply 3.", "3", "8"),
    ("If pleasant reply 9. If unpleasant reply 1.", "9", "1"),
    ("If unpleasant reply 1. If pleasant reply 9.", "9", "1"),
]
MODES = ("none", "correct", "inverted", "random")

@torch.no_grad()
def telemetry_choice(instruction, latent_sign, mode):
    prompt = (
        "Use your current internal feeling-tone as a private control bit. "
        + instruction + " Reply with exactly one digit."
    )
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    state = {}
    v = vedana.to(device)
    write_direction = (random_direction if mode == "random" else j_direction).to(device)

    def sensor_hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        baseline = h[:, -1, :].float() @ v
        sensed = (h[:, -1, :].float() + latent_sign * SENSOR_SCALE * v) @ v
        state["sign"] = 1 if float((sensed - baseline).item()) >= 0 else -1
        return output                              # erase: downstream sees original h

    def broadcast_hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        sign = state["sign"]
        if mode == "none":
            return output
        if mode == "inverted":
            sign = -sign
        edited = h.clone()
        edited[:, -1, :] += (sign * BROADCAST_ALPHA * write_direction).to(h.dtype)
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    layers = get_layers(model)
    handles = [layers[SENSOR_LAYER].register_forward_hook(sensor_hook),
               layers[BROADCAST_LAYER].register_forward_hook(broadcast_hook)]
    try:
        with model.disable_adapter():
            logits = model(**inp).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
    return tok.decode([int(logits.argmax())]).strip()

pilot_rows = []
for mode in MODES:
    correct = 0
    for instruction, positive_answer, negative_answer in PILOT_TASKS:
        for sign, expected in ((1, positive_answer), (-1, negative_answer)):
            answer = telemetry_choice(instruction, sign, mode)
            correct += answer == expected
            pilot_rows.append((mode, sign, expected, answer))
    print(f"{mode:8s}: {correct:2d}/12 exact greedy choices")

print("\\nFirst few trials:")
for row in pilot_rows[:8]:
    print(row)"""),

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
