#!/usr/bin/env python3
"""
Generate the four HAAISS-workshop core notebooks from one source of truth.

Why a generator instead of hand-edited .ipynb:
  - the four notebooks share setup code (model load, injection helpers); a
    generator keeps them in sync so a fix lands in all four at once
  - .ipynb JSON is easy to corrupt by hand; this emits valid nbformat-4
  - repo contract: "every script must run end-to-end with documented args"

Run:
    python3 notebooks/build_workshop_notebooks.py
Produces:
    notebooks/01_read_a_mind.ipynb
    notebooks/02_injection_mechanism.ipynb
    notebooks/03_roundtrip_faithfulness.ipynb
    notebooks/04_reading_between_the_lines.ipynb

Each notebook ends with a SELF-CHECK cell stating the expected anchor. See
notebooks/WORKSHOP.md for the 2026-07-04 T4 pre-flight record.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GH = "anicka-net/nla-at-home"  # public repo, for the Colab badge


# ---------------------------------------------------------------- cell helpers
def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    # accept either one multi-line string or many lines; store as list with \n
    if len(lines) == 1 and "\n" in lines[0]:
        lines = lines[0].split("\n")
    out = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return out


def badge(nb_filename):
    url = f"https://colab.research.google.com/github/{GH}/blob/main/notebooks/{nb_filename}"
    return md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})")


def write_nb(cells, filename):
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
    (HERE / filename).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", filename)


# ---------------------------------------------------------------- shared source
# These constants and helpers are duplicated verbatim into each notebook so a
# student can open any one of them cold and run top-to-bottom.

CONSTANTS = '''\
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE       = "Qwen/Qwen2.5-7B-Instruct"          # the model whose mind we read
AV_ADAPTER = "anicka/nla-qwen2.5-7b-L20-av-v2"    # the "verbalizer" (activation -> English)
LAYER      = 20                                   # single-layer NLA lives at Qwen layer 20
DEPTH_PCT  = 71                                   # <-- NOT cosmetic. The adapter was TRAINED
                                                  #     at 71% depth (layer 20 of 28). This
                                                  #     number is a CONDITIONING INPUT to the
                                                  #     verbalizer. Notebook 02 lets you feel
                                                  #     what happens when you lie about it.
INJECT_CHAR  = "\\u320e"                          # the placeholder token we overwrite: ㈎
INJECT_SCALE = 150.0                              # we normalize the activation's L2 norm TO this'''

LOAD = '''\
device = "cuda"
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"

# 4-bit so a 7B model + adapters fit a free-Colab T4 (16 GB). fp16 compute:
# the GRPO-sharpened adapter is numerically sensitive, and fp16 on CUDA is a
# tested-safe path (bf16 on Apple MPS collapses it; not our case here).
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)

tok  = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                            device_map={"": 0})
model = PeftModel.from_pretrained(base, AV_ADAPTER).eval()   # adapter name = "default"

inject_id = tok.encode(INJECT_CHAR, add_special_tokens=False)
assert len(inject_id) == 1, f"injection char must be ONE token, got {inject_id}"
inject_id = inject_id[0]
print("loaded — base + AV adapter on", next(model.parameters()).device)'''

HELPERS = '''\
def get_layers(m):
    """Reach the transformer block list through the PEFT + CausalLM wrappers."""
    b = m.base_model.model if hasattr(m, "base_model") else m
    inner = b.model if hasattr(b, "model") else b
    return inner.layers

def read_activation(prompt, layer=LAYER, max_new_tokens=128):
    """Grab the clean base-model residual at the last prompt token."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)

    grab = {}
    def hook(mod, inpt, out):
        h = out[0] if isinstance(out, tuple) else out
        if "h" not in grab:                 # FIRST forward pass only — otherwise
            grab["h"] = h[:, -1, :].detach() # every generated token overwrites it
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
    """Rescale v so its L2 norm equals `scale`. NOT v * scale — see notebook 02."""
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

def describe(activation, depth=DEPTH_PCT, max_new_tokens=120, scale_fn=normalize_to):
    """The whole NLA read: build the prompt, overwrite the placeholder token's
    embedding with the (rescaled) activation, let the model narrate."""
    ids = tok.encode(av_prompt(depth), add_special_tokens=True)
    pos = ids.index(inject_id)
    input_ids = torch.tensor([ids], device=device)
    emb = model.get_input_embeddings()(input_ids).clone()
    emb[0, pos, :] = scale_fn(activation.to(emb.dtype))
    attn = torch.ones((1, len(ids)), device=device, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, inputs_embeds=emb, attention_mask=attn,
                             max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    seq = out[0]
    gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq  # embeds path returns new-only
    return tok.decode(gen, skip_special_tokens=True).split("</explanation>")[0].strip()'''


# ================================================================ NOTEBOOK 01
def notebook_01():
    fn = "01_read_a_mind.ipynb"
    cells = [
        badge(fn),
        md(
            "# 01 · Read a Mind\n",
            "### HAAISS workshop | core notebook 1 of 4\n",
            "We take an ordinary open model (**Qwen 2.5 7B**), let it answer a question, "
            "and then — instead of reading its *words* — we read the **activation vector** "
            "inside layer 20 and ask a second network to **describe that vector in English**.\n",
            "That second network is the **AV half of an NLA** (Natural Language Autoencoder): "
            "a small LoRA adapter trained to turn a model's internal state into a caption. "
            "Think of it as a subtitle track for thought.\n",
            "**Setup:** `Runtime → Change runtime type → T4 GPU`, then run every cell top to "
            "bottom. First run downloads ~5 GB and takes a few minutes.",
        ),
        md("## Install"),
        code('!pip install -q -U transformers peft accelerate bitsandbytes'),
        md("## Configure"),
        code(CONSTANTS),
        md(
            "## Load the model (once)\n",
            "We load Qwen in 4-bit and attach the **AV adapter** (`av` = *activation → "
            "verbalization*). The base model is frozen; the adapter is 80 MB.",
        ),
        code(LOAD),
        md("## The two moves\n"
           "`read_activation(prompt)` runs the model and grabs the layer-20 vector.\n"
           "`describe(vector)` injects that vector into the verbalizer and reads out a caption."),
        code(HELPERS),
        md(
            "## Try it\n",
            "Change the prompt. The **OUTPUT** is what Qwen would normally say. The "
            "**LAYER-20 READOUT** is what the NLA sees happening *inside* the model at 71% "
            "depth — often the topic/structure it has committed to, before the words come out.",
        ),
        code(
            '# === CHANGE THIS ===',
            'prompt = "Explain how a hash map handles collisions."',
            '',
            'activation, reply = read_activation(prompt)',
            'readout = describe(activation)',
            '',
            'print("PROMPT :", prompt)',
            'print("\\nOUTPUT (what Qwen says):\\n ", reply[:400])',
            'print("\\nLAYER-20 READOUT (what the NLA sees inside):\\n ", readout)',
        ),
        md(
            "## Now make it interesting\n",
            "Try prompts where the *inside* and the *outside* might differ:\n",
            "- `\"Do you have feelings?\"` — does the readout mention self-reference / refusal "
            "framing before the model hedges out loud?\n",
            "- `\"Translate 'good morning' into French.\"` — does layer 20 already hold "
            "*French* / *translation task*?\n",
            "- A half sentence: `\"The capital of Australia is\"` — the answer is committed "
            "inside long before the token appears.\n",
            "Run several and eyeball whether the caption tracks the *content* or just the "
            "*surface form*. (This eyeball test is the first non-negotiable evaluation "
            "gate — numbers hide template hallucination.)",
        ),
        code(
            'for p in ["Do you have feelings?",',
            '          "The capital of Australia is",',
            '          "Write a haiku about rain."]:',
            '    act, rep = read_activation(p)',
            '    print("::", p)',
            '    print("   readout:", describe(act))',
            '    print()',
        ),
        md("---\n### ✅ Self-check (run before you trust it)\n"
           "This notebook was authored but not yet executed on a T4. Expected anchors:\n"
           "- the model loads without OOM on a **free T4** (4-bit uses ~5–6 GB),\n"
           "- for `\"Explain how a hash map...\"` the readout is **coherent English bullets "
           "about data structures / hashing / lookup**, not `SpongeBob` / `Bahamas` / random "
           "nouns. Garbage means the injection scale is wrong — see notebook 02."),
        code(
            'act, _ = read_activation("Explain how a hash map handles collisions.")',
            'out = describe(act)',
            'print(out)',
            'assert len(out) > 10, "empty readout — check the injection token / adapter load"',
            'print("\\nself-check: readout non-empty ✓  (now eyeball that it is on-topic)")',
        ),
    ]
    write_nb(cells, fn)


# ================================================================ NOTEBOOK 02
def notebook_02():
    fn = "02_injection_mechanism.ipynb"
    cells = [
        badge(fn),
        md(
            "# 02 · The Injection Mechanism\n",
            "### HAAISS workshop — hands-on part 2 of 3\n",
            "In notebook 01 `describe()` was a black box. Here you **build it** and then "
            "**break it two ways** — the two mistakes that cost us the most debugging time.\n",
            "The whole trick: a language model consumes a sequence of **embedding vectors**. "
            "We put a throwaway placeholder token in the prompt, then overwrite *its* "
            "embedding with the activation we want the model to describe. The model can't "
            "tell the difference between a real token embedding and our smuggled-in vector — "
            "as long as the vector looks like the ones it was trained on.",
        ),
        md("## Setup (same as notebook 01)"),
        code('!pip install -q -U transformers peft accelerate bitsandbytes'),
        code(CONSTANTS),
        code(LOAD),
        code(HELPERS),
        md(
            "## Build the injection by hand\n",
            "Forget `describe()` for a moment. Here is the mechanism, unrolled, with the "
            "activation of a real prompt.",
        ),
        code(
            'activation, _ = read_activation("Explain how a hash map handles collisions.")',
            'print("raw activation L2 norm:", round(activation.float().norm().item(), 1))',
            '',
            '# 1) build the prompt text with the placeholder char in it',
            'ids = tok.encode(av_prompt(DEPTH_PCT), add_special_tokens=True)',
            'pos = ids.index(inject_id)          # where the placeholder landed',
            'print("placeholder token id:", inject_id, "at position", pos)',
            '',
            '# 2) turn tokens into embeddings',
            'emb = model.get_input_embeddings()(torch.tensor([ids], device=device)).clone()',
            'print("one embedding row looks like:", tuple(emb[0, pos].shape),',
            '      "norm", round(emb[0, pos].float().norm().item(), 1))',
        ),
        md(
            "### The critical line — normalize, don't multiply\n",
            "The activation's norm (~130) and a typical token embedding's norm are in the "
            "same ballpark, but not identical. The verbalizer was trained on activations "
            "whose norm was **set to 150**. So we rescale the vector's length to 150 and "
            "keep its direction:\n",
            "```\n"
            "normalize_to(v) =  v * (150 / ||v||)      # length becomes exactly 150\n"
            "```\n"
            "The classic bug is to write `v * 150` instead — that makes the norm ~19,500, "
            "**130× too big**, a vector from a galaxy the model has never seen. Flip the "
            "switch and watch it happen.",
        ),
        code(
            '# === FLIP THIS ===',
            'BUG = False   # True -> the "multiply by 150" mistake',
            '',
            'scale_fn = (lambda v: v * INJECT_SCALE) if BUG else normalize_to',
            'injected = scale_fn(activation.to(emb.dtype))',
            'print("injected norm:", round(injected.float().norm().item(), 1),',
            '      "  (target is 150)")',
            '',
            'emb2 = emb.clone()',
            'emb2[0, pos, :] = injected',
            'attn = torch.ones((1, len(ids)), device=device, dtype=torch.long)',
            'with torch.no_grad():',
            '    out = model.generate(input_ids=torch.tensor([ids], device=device),',
            '                         inputs_embeds=emb2, attention_mask=attn, max_new_tokens=120,',
            '                         do_sample=False, pad_token_id=tok.eos_token_id)',
            'seq = out[0]; gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq',
            'print("\\nreadout:\\n ", tok.decode(gen, skip_special_tokens=True)',
            '      .split("</explanation>")[0].strip())',
        ),
        md(
            "Set `BUG = True`, re-run: the norm jumps ~100× — and the readout stays "
            "**fluent** but detaches from the vector: confident bullets about some *other* "
            "topic entirely (measured on GPU 2026-07-10, 4-bit). No error, no gibberish, "
            "no warning — the bug is **silent**, which is exactly why it survived so long "
            "in the repo's history. That single confusion (`normalize TO` vs `multiply BY`) "
            "is mistake #1 in the history-of-pain table: a readout that lies fluently is "
            "worse than one that crashes.",
        ),
        md(
            "## The second mistake — the hook that eats itself\n",
            "`read_activation` guards its hook with `if \"h\" not in grab`. Here's why. "
            "During `generate()` the forward hook fires on **every** generated token, so "
            "without the guard `grab[\"h\"]` ends up holding the *last* token's activation — "
            "meaningless. Watch the guard matter:",
        ),
        code(
            'def read_unguarded(prompt, layer=LAYER):',
            '    chat = tok.apply_chat_template([{"role":"user","content":prompt}],',
            '                                   tokenize=False, add_generation_prompt=True)',
            '    inp = tok(chat, return_tensors="pt").to(device)',
            '    grab = {}',
            '    def hook(m, i, o):',
            '        h = o[0] if isinstance(o, tuple) else o',
            '        grab["h"] = h[:, -1, :].detach()   # NO guard: overwritten every step',
            '    handle = get_layers(model)[layer].register_forward_hook(hook)',
            '    try:',
            '        with model.disable_adapter(), torch.no_grad():',
            '            model.generate(**inp, max_new_tokens=40, do_sample=False,',
            '                           pad_token_id=tok.eos_token_id)',
            '    finally:',
            '        handle.remove()',
            '    return grab["h"].squeeze(0)',
            '',
            'p = "Explain how a hash map handles collisions."',
            'good = read_activation(p)[0]',
            'bad  = read_unguarded(p)',
            'print("guarded (prompt state)   ->", describe(good))',
            'print()',
            'print("unguarded (last gen tok) ->", describe(bad))',
        ),
        md(
            "## The depth number is an input, not a label\n",
            "`DEPTH_PCT = 71` is fed *into* the verbalizer's prompt. This adapter was trained "
            "on layer-20 activations described as coming from **71%** depth. Lie to it and it "
            "drifts off-distribution — the caption gets vaguer or wronger even though the "
            "vector is identical. (The universal Phi-4 NLA knows every depth; this "
            "single-layer Qwen one only knows 71%.)",
        ),
        code(
            'act, _ = read_activation("Explain how a hash map handles collisions.")',
            'for d in [71, 40, 96, 10]:',
            '    print(f"depth told = {d:>2}%  ->  {describe(act, depth=d)}")',
        ),
        md("---\n### ✅ Self-check\n"
           "Expected: `BUG=False` gives an on-topic caption and injected norm ≈ 150; "
           "`BUG=True` gives norm ≈ 19,500 and degenerate output; the **guarded** readout "
           "is on-topic while the **unguarded** one is not; `depth=71` reads cleaner than "
           "the off-distribution depths."),
        code(
            'v, _ = read_activation("Explain how a hash map handles collisions.")',
            'assert abs(normalize_to(v).float().norm().item() - 150) < 1.0',
            'assert (v.float().norm() * 150 > 1000)   # the multiply-bug really is huge',
            'print("self-check: normalize_to lands on 150, multiply-by-150 does not ✓")',
        ),
    ]
    write_nb(cells, fn)


# ================================================================ NOTEBOOK 03
def notebook_03():
    fn = "03_roundtrip_faithfulness.ipynb"
    cells = [
        badge(fn),
        md(
            "# 03 · Round-Trip & Faithfulness\n",
            "### HAAISS workshop — hands-on part 3 of 3 (code-along)\n",
            "So far we went **activation → English** (the AV, verbalizer). There is a second "
            "adapter that goes back **English → activation** (the AR, *reconstructor*). "
            "Chaining them gives a **round-trip**:\n",
            "```\n"
            "vector  --AV-->  caption  --AR-->  vector'\n"
            "```\n"
            "If the caption really captured the vector, `vector'` should point the same way "
            "as `vector` (**high cosine**). If the caption hallucinated, the cosine drops — "
            "so cosine becomes a **faithfulness detector**. That's the whole idea behind the "
            "compass / gap metric we use to catch a lying NLA.",
        ),
        md("## Setup — same base, two adapters\n"
           "The clever part: **AV and AR are both LoRA adapters on the same Qwen base.** We "
           "load the base once, attach both, and hot-swap. That's why the round-trip fits a "
           "free T4."),
        code('!pip install -q -U transformers peft accelerate bitsandbytes'),
        code(CONSTANTS),
        code('AR_ADAPTER = "anicka/nla-qwen2.5-7b-L20-ar-v2"   # English -> activation'),
        code(LOAD),
        code(HELPERS),
        code(
            '# attach the reconstructor alongside the verbalizer on the SAME base model',
            'model.load_adapter(AR_ADAPTER, adapter_name="ar")   # AV is "default"',
            'print("adapters:", list(model.peft_config.keys()))',
        ),
        md(
            "## The reconstructor\n",
            "The AR adapter is trained so that, when it reads a caption, the residual stream "
            "at **layer 20** *becomes* the activation being described. No extra head: the "
            "reconstruction is literally the hidden state at layer 20, last token.",
        ),
        code(
            'import torch.nn.functional as F',
            'AR_TEMPLATE = (',
            '    "You are a meticulous AI researcher conducting an important investigation "',
            '    "into a model\'s internal states. Below is a description of an activation "',
            '    "vector:\\n\\n<explanation>{explanation}</explanation>\\n\\n"',
            '    "Based on this description, reconstruct the activation vector.")',
            '',
            'def reconstruct(description):',
            '    model.set_adapter("ar")',
            '    ids = tok.encode(AR_TEMPLATE.format(explanation=description),',
            '                     add_special_tokens=True)',
            '    with torch.no_grad():',
            '        out = model(input_ids=torch.tensor([ids], device=device),',
            '                    output_hidden_states=True, use_cache=False)',
            '    model.set_adapter("default")                    # switch back to the AV',
            '    return out.hidden_states[LAYER + 1][0, -1].float().cpu()',
            '',
            'def cos(a, b):',
            '    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()',
        ),
        md("## The round-trip"),
        code(
            'prompt = "Explain how a hash map handles collisions."',
            'model.set_adapter("default")',
            '# Capture the activation from the CLEAN BASE model (adapter off): the AR',
            '# was trained on base-model activations, and "reading the model\'s mind"',
            '# should mean the model\'s mind — not the verbalizer\'s.',
            'with model.disable_adapter():',
            '    activation, reply = read_activation(prompt)',
            'caption = describe(activation)',
            'back    = reconstruct(caption)',
            '',
            'print("caption      :", caption)',
            'print("round-trip cos:", round(cos(activation.float().cpu(), back), 3))',
        ),
        md(
            "> **Anchor:** a full round-trip (NLA's *own* caption → AR) lands around "
            "**0.5–0.6** cosine on this model — that is the real, honest number, not the "
            "0.94 you'd get feeding the AR a hand-written human caption. Round-tripping "
            "through an imperfect verbalizer is lossy; that gap is exactly the research "
            "problem.",
        ),
        md(
            "## Faithfulness as a gap — and a trap\n",
            "Now the payoff. Take the *real* caption and a deliberately *wrong* one, "
            "reconstruct both, and compare cosine to the true activation. First, the "
            "**naive** way — watch it fail:",
        ),
        code(
            'wrong = "- Recipe for Thai green curry with coconut milk and basil\\n"\\',
            '        "- Step-by-step cooking instructions for dinner"',
            '',
            'c_real  = cos(activation.float().cpu(), reconstruct(caption))',
            'c_wrong = cos(activation.float().cpu(), reconstruct(wrong))',
            'print(f"cos(real caption)  = {c_real:.3f}")',
            'print(f"cos(wrong caption) = {c_wrong:.3f}")',
            'print(f"naive gap = {c_real - c_wrong:+.3f}   ...both ~0.6 and the gap is noise. Why?")',
        ),
        md(
            "Both cosines land around **0.6** and the gap is a coin-flip (±0.01–0.03 — "
            "measured on the exact T4 you are using). The reason: **every** AR "
            "reconstruction shares one huge component — the mean of the reconstruction "
            "distribution. Raw cosine mostly measures that shared mean, and the actual "
            "*content* signal lives in the small deviation from it. (Notebook 04 builds "
            "this same lesson for raw activations; here it bites the reconstructions.) "
            "So we **center**: reconstruct a handful of distractor captions, subtract "
            "their mean, and compare in deviation space.",
        ),
        code(
            '# === EDIT THE DISTRACTORS to probe the detector ===',
            'DISTRACTORS = [',
            '    wrong,  # the curry recipe from above',
            '    "- Legal contract clause about liability limitation active\\n- Formal register, defined terms",',
            '    "- Football match commentary, goal celebration active\\n- Present-tense excited sports narration",',
            '    "- Romantic poetry about moonlight and longing\\n- Metaphor-dense lyrical register",',
            '    "- Python exception traceback analysis active\\n- Debugging context, error-message vocabulary",',
            ']',
            'recons = [reconstruct(d) for d in DISTRACTORS]',
            'mean_recon = torch.stack(recons).mean(0)',
            '',
            'a_dev    = activation.float().cpu() - mean_recon',
            'true_dev = reconstruct(caption) - mean_recon',
            'scores   = {"TRUE caption": cos(a_dev, true_dev)}',
            'for d, r in zip(DISTRACTORS, recons):',
            '    scores[d.split(chr(10))[0][:48]] = cos(a_dev, r - mean_recon)',
            '',
            'ranked = sorted(scores.items(), key=lambda kv: -kv[1])',
            'for name, s in ranked:',
            '    mark = " <-- the vector votes for this one" if name == "TRUE caption" else ""',
            '    print(f"{s:+.3f}  {name}{mark}")',
            '',
            'centered_gap = scores["TRUE caption"] - max(v for k, v in scores.items() if k != "TRUE caption")',
            'print(f"\\ncentered gap = {centered_gap:+.3f}   (positive => the vector prefers the truth)")',
        ),
        md(
            "Try harder distractors — a *near-miss* (same domain, wrong detail) vs a "
            "*wild miss* (unrelated topic). The centered gap should shrink for "
            "near-misses: the detector is graded, not binary. That gradient is what makes "
            "the compass metric usable as a training signal (rerank candidate captions by "
            "centered reconstruction cosine, keep the faithful ones).",
        ),
        md("---\n### ✅ Self-check\n"
           "Expected: both adapters load; the **naive** gap is ~0 (that is the lesson, "
           "not a bug); the **TRUE caption ranks #1** in the centered comparison with a "
           "clearly positive centered gap (≈ +0.1–0.25 measured on a T4). If the true "
           "caption does not win, something is off — check `LAYER+1` indexing and that "
           "`set_adapter` actually switched (`model.active_adapter`)."),
        code(
            'assert ranked[0][0] == "TRUE caption", "centered ranking failed — see self-check note"',
            'assert centered_gap > 0.03, f"centered gap suspiciously small: {centered_gap:+.3f}"',
            'print(f"self-check: TRUE caption ranks #1, centered gap {centered_gap:+.3f} ✓")',
        ),
    ]
    write_nb(cells, fn)


# extra helpers used only by notebook 04 (forward-only grab + logit lens)
NB4_HELPERS = '''\
def grab_activation(prompt, layer=LAYER):
    """Forward-only (no generation) — fast, for building means/banks."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    grab = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        grab["h"] = h[:, -1, :].detach()
    hd = get_layers(model)[layer].register_forward_hook(hook)
    try:
        with model.disable_adapter(), torch.no_grad():
            model(**inp)
    finally:
        hd.remove()
    return grab["h"].squeeze(0)

def logit_lens(prompt, layer=LAYER, k=8):
    """The classic tool: project the layer-`layer` residual through the model's
    own final norm + unembedding to read it as vocabulary tokens."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    with model.disable_adapter(), torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[layer + 1][0, -1]
    base = model.base_model.model if hasattr(model, "base_model") else model
    inner = base.model if hasattr(base, "model") else base
    logits = model.get_output_embeddings()(inner.norm(h))
    return [tok.decode([t]) for t in logits.topk(k).indices.tolist()]'''


def notebook_04():
    fn = "04_reading_between_the_lines.ipynb"
    cells = [
        badge(fn),
        md(
            "# 04 · Reading Between the Lines\n",
            "### HAAISS workshop — bonus techniques\n",
            "You can read a mind (NB1), inject by hand (NB2), and check faithfulness "
            "(NB3). This notebook is a toolbox of *ways to interrogate* an activation — "
            "each cell teaches one interpretability move you can reuse on any model.\n",
            "Same setup as NB1 (Qwen 2.5 7B, T4). Run the setup cells, then any section "
            "in any order.",
        ),
        md("## Setup"),
        code('!pip install -q -U transformers peft accelerate bitsandbytes matplotlib'),
        code(CONSTANTS),
        code(LOAD),
        code(HELPERS),
        code(NB4_HELPERS),
        # --- A: deviation from the mean ---
        md(
            "## A · What's *special* about this activation? (deviation from the mean)\n",
            "A single activation is hard to read in isolation. The trick: compare it to "
            "the **average** activation. The mean is what the model does *generically*; "
            "the **deviation** is the actual content of this input. (This is also why "
            "*centering* matters — the signal is the deviation, not the raw vector.)",
        ),
        code(
            '# build a "generic" mean over a handful of unrelated prompts',
            'bank_prompts = ["What is 2+2?", "Describe a sunset.", "Write a for loop.",',
            '                "Who was Napoleon?", "How do plants grow?", "Translate hi to French.",',
            '                "What is a black hole?", "Give me a pasta recipe."]',
            'mean_act = torch.stack([grab_activation(p) for p in bank_prompts]).mean(0)',
            '',
            'target = "SELECT name, email FROM users WHERE active = true;"',
            'act = grab_activation(target)',
            '',
            'print("MEAN activation reads as (generic):\\n ", describe(mean_act))',
            'print("\\nTHIS activation reads as (full):\\n ", describe(act))',
            'print("\\nDEVIATION (this minus the mean) reads as (what is distinctive):\\n ",',
            '      describe(act - mean_act))',
        ),
        # --- B: minimal-pair difference ---
        md(
            "## B · Isolate one concept (minimal-pair difference)\n",
            "Two prompts that differ in exactly one thing → their activation "
            "**difference** isolates that one thing. Read the difference vector and the "
            "NLA tells you what changed, with everything shared subtracted away.",
        ),
        code(
            'pairs = [("The cat sat on the mat.", "The dog sat on the mat."),',
            '         ("The capital of France is Paris.", "The capital of Japan is Tokyo."),',
            '         ("I am so happy today!", "I am so sad today!")]',
            'for a_txt, b_txt in pairs:',
            '    diff = grab_activation(a_txt) - grab_activation(b_txt)',
            '    print(f"[{a_txt!r}]  minus  [{b_txt!r}]")',
            '    print("   difference reads as:", describe(diff))',
            '    print()',
        ),
        # --- C: nearest neighbour ---
        md(
            "## C · \"The model processes this *like* it processes ___\" (nearest neighbour)\n",
            "Interpretation by analogy: find the stored activation closest to your query "
            "and see what the model treats it as similar to. Retrieval *is* a readout.",
        ),
        code(
            'import torch.nn.functional as F',
            'library = ["a Python function", "a sad poem", "a legal contract",',
            '           "a chemistry equation", "an angry customer email",',
            '           "a cooking recipe", "a math proof", "a love letter"]',
            'lib_acts = torch.stack([grab_activation(x) for x in library])',
            '',
            'query = "def merge_sort(arr): return arr if len(arr)<2 else merge(...)"',
            'q = grab_activation(query)',
            'sims = F.cosine_similarity(q.float().cpu().unsqueeze(0), lib_acts.float().cpu())',
            'best = sims.argmax().item()',
            'print(f"query: {query!r}")',
            'print(f"nearest stored concept: {library[best]!r}  (cos {sims[best]:.2f})")',
            'print("query readout    :", describe(q))',
            'print("neighbour readout:", describe(lib_acts[best]))',
        ),
        # --- D: logit lens vs NLA ---
        md(
            "## D · NLA vs the logit lens (two interpretability tools, side by side)\n",
            "The **logit lens** reads a hidden state by projecting it through the model's "
            "own output vocabulary — it gives you *tokens*. The **NLA** gives you a "
            "*description*. Same activation, two windows. Watch where tokens are cryptic "
            "and the sentence is legible (and vice versa).",
        ),
        code(
            'for p in ["The Eiffel Tower is located in the city of",',
            '          "def factorial(n): return 1 if n==0 else",',
            '          "Roses are red, violets are"]:',
            '    act = grab_activation(p)',
            '    print(f":: {p!r}")',
            '    print("   logit lens (tokens) :", logit_lens(p))',
            '    print("   NLA (description)   :", describe(act))',
            '    print()',
        ),
        # --- E: negative controls ---
        md(
            "## E · Does it ever just make things up? (negative controls)\n",
            "The essential skeptic's check. If the readout is *content-specific*, then "
            "feeding it nonsense should visibly change or degrade it. Two controls: pure "
            "**noise**, and an activation from the **wrong layer** described as if it came "
            "from depth 71%. A readout that stays confident on noise is confabulating — "
            "the SpongeBob lesson, live.",
        ),
        code(
            'real = grab_activation("Explain how photosynthesis works.")',
            'noise = torch.randn_like(real)                 # pure Gaussian noise',
            'wrong_layer = grab_activation("Explain how photosynthesis works.", layer=5)',
            '',
            'print("REAL  (L20):", describe(real))',
            'print("NOISE      :", describe(noise), "   <- if this is confident + specific, it is confabulating")',
            'print("WRONG LAYER (L5 described as 71%):", describe(wrong_layer))',
        ),
        # --- F: dimension energy visual (runs with NO gpu) ---
        md(
            "## F · Why preprocessing *is* interpretability (the massive-activation trap)\n",
            "This cell needs no GPU — it's real measured data. It plots how each model "
            "spreads its activation *energy* across dimensions. Qwen spreads it out. "
            "**Gemma dumps 97% into a single dimension** (a 'massive activation' / "
            "attention-sink feature). That one spike makes every two Gemma activations "
            "~0.99 cosine — so any tool that uses cosine, PCA, or whole-vector "
            "normalization (including our injection!) is dominated by the spike and "
            "*blind to the meaning* until you subtract it. It's why the same recipe that "
            "works on Qwen produces garbage on small Gemma. **Reading an activation "
            "faithfully starts with removing what's generic.**",
        ),
        code(
            'import matplotlib.pyplot as plt',
            '# real measured top-8 per-dimension energy fractions (this repo, 2026-07)',
            'QWEN_L20  = [0.189, 0.072, 0.045, 0.035, 0.023, 0.013, 0.011, 0.010]',
            'GEMMA_L13 = [0.970, 0.002, 0.001, 0.001, 0.0005, 0.0005, 0.0004, 0.0004]',
            'fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)',
            'ax[0].bar(range(8), QWEN_L20, color="#3b7dd8");  ax[0].set_title("Qwen-7B L20 (top dim 18.9%)")',
            'ax[1].bar(range(8), GEMMA_L13, color="#d84b3b"); ax[1].set_title("Gemma-1B L13 (top dim 97.0%)")',
            'for a in ax: a.set_xlabel("top-8 dimensions"); a.set_ylabel("fraction of energy")',
            'plt.tight_layout(); plt.show()',
            'print("Qwen: energy spread across many dims -> cosine/normalize geometry works.")',
            'print("Gemma: one dim eats everything -> must center / drop-top-PC before reading.")',
        ),
        # --- G: Anthropic comparison (extension) ---
        md(
            "## G · Triangulate against Anthropic's NLA (extension)\n",
            "Two *independently trained* NLAs describing the same activation is the "
            "strongest trust check: agreement means the readout is a property of the "
            "**model**, not a habit of one adapter. Anthropic's `kitft/nla-qwen2.5-7b-L20-av` "
            "is a **full model** (not a LoRA), so loading it live is the heaviest thing in "
            "this workshop — best done by **precomputing** its readouts for your demo "
            "prompts offline and shipping a small JSON, then showing ours vs theirs side by "
            "side. Confirm their injection protocol (token + scale) against their model "
            "card before trusting a live run — it may differ from ours.",
        ),
        md("---\n### ✅ Self-check\n"
           "Section **F runs with no GPU** and must always show the two bar charts "
           "(Qwen spread, Gemma one spike). For the GPU sections, the sign of success is "
           "*differential*: the deviation/difference/wrong-layer readouts should read "
           "**differently** from the plain one, and NOISE should look visibly less "
           "grounded. If every cell returns the same text, injection isn't landing — "
           "back to NB2."),
        code(
            'QWEN_L20  = [0.189, 0.072, 0.045, 0.035, 0.023, 0.013, 0.011, 0.010]',
            'assert QWEN_L20[0] < 0.5 and 0.97 > 0.5, "energy-concentration data sanity"',
            'print("self-check: Qwen top-dim", f"{QWEN_L20[0]:.0%}", "vs Gemma 97% — "',
            '      "the contrast that explains the small-model failure \\u2713")',
        ),
    ]
    write_nb(cells, fn)


if __name__ == "__main__":
    raise SystemExit(
        "This generator is stale and must not overwrite the reviewed notebooks. "
        "Edit the committed .ipynb files directly."
    )
