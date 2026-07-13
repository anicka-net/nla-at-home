#!/usr/bin/env python3
"""Generate notebook 07: validate one Qwen valence axis with three witnesses."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GH = "anicka-net/nla-at-home"
FN = "07_validate_one_axis.ipynb"


def _src(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


cells = [
    md(
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{GH}/blob/main/notebooks/{FN})"
    ),
    md("""# 07 · Validate one axis with three witnesses ⚠️ EXPERIMENTAL

This notebook follows one **Qwen 2.5 7B valence direction** through three
independent readouts:

1. **SAE composition:** ask whether the direction aligns unusually well with
   sparse features learned independently from ordinary Qwen activations.
2. **J-Lens:** ask what words downstream computation connects to the direction.
3. **NLA:** ask what prose the two representative activation states support.

These tests do not prove the model has an emotion. They test a narrower claim:
**does this vector consistently track affective valence inside this model?**

The SAE section uses a compact measured artifact. J-Lens and NLA need a GPU.
This notebook has been executed end-to-end in Colab with the 4-bit model."""),
    md("""## Why layer 19?

The older project result compared a direction named `L20` with the nearest
available SAE at layer 19. That was useful, but not literally the same hidden
state.

Here every instrument uses the **output of transformer block 19**:

| instrument | state |
|---|---|
| contrastive direction | block 19 output |
| SAE | `resid_post_layer_19` |
| J-Lens | source layer 19 |
| NLA | pleasant/unpleasant centroids at block 19, depth tag 71% |

Matching the state is more important than preserving the historical layer
name.

(Notebook 05's interoception pilot uses the same direction family at an
earlier station — sensing at L14, broadcasting at L15 — because that
experiment needs the rest of the stack downstream of the write. Different
layer, same axis family; here every witness reads block 19.)"""),
    md("""## 1 · Two clouds define the axis

The compact artifact contains no prompt text or model checkpoint. It stores:

- the pleasant and unpleasant mean activation states;
- their unit difference `mean(pleasant) - mean(unpleasant)`;
- each extraction prompt's scalar projection onto that direction;
- provenance needed to identify the model and layer.

The separation below is **in-sample** because these prompts define the axis.
It explains construction; it is not independent validation."""),
    code("""import io, json, urllib.request
import matplotlib.pyplot as plt
import torch

ARTIFACT_BASE = (
    "https://raw.githubusercontent.com/anicka-net/nla-at-home/"
    "main/notebooks/artifacts"
)

def download_bytes(name):
    with urllib.request.urlopen(f"{ARTIFACT_BASE}/{name}") as response:
        return response.read()

axis_data = torch.load(
    io.BytesIO(download_bytes("qwen25_valence_block19.pt")),
    map_location="cpu",
    weights_only=True,
)
direction = axis_data["direction"].float()
assert direction.shape == (3584,) and torch.allclose(direction.norm(), torch.tensor(1.0))
pleasant_centroid = axis_data["pleasant_centroid"].float()
unpleasant_centroid = axis_data["unpleasant_centroid"].float()

pleasant = axis_data["pleasant_projections"].float()
unpleasant = axis_data["unpleasant_projections"].float()
print("model:", axis_data["model"])
print("block:", axis_data["block_index"])
print("extraction prompts:", len(pleasant), "pleasant +", len(unpleasant), "unpleasant")

plt.figure(figsize=(8, 3.5))
bins = 18
plt.hist(unpleasant, bins=bins, alpha=.65, label="unpleasant extraction prompts")
plt.hist(pleasant, bins=bins, alpha=.65, label="pleasant extraction prompts")
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("projection onto the valence direction")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.show()"""),
    md("""The plot should separate because the direction was constructed to do
exactly that. The remaining sections ask whether independently built
instruments agree with the interpretation **valence**."""),
    md("""## 2 · SAE: is the axis unusual in a learned feature basis?

The matching public SAE has 131,072 BatchTopK features and was trained by
`andyrdt` on a mixture of chat and pretraining data. It did not see our
pleasant/unpleasant labels.

Downloading its full checkpoint would add 3.76 GB to this teaching notebook,
so we ship only measured summary statistics. The computation is reproducible
from the Apache-2.0 checkpoint linked below.

The metric shown here:

1. select the 50 SAE decoder directions most aligned with the axis;
2. measure how much of the axis lies in their combined span;
3. repeat the same selection for 64 random directions.

The random comparison matters. In a large dictionary, every vector has some
apparently similar features."""),
    code("""sae_result = json.loads(
    download_bytes("qwen25_valence_block19_sae.json").decode("utf-8")
)
capture = sae_result["top50_span_capture"]
cosine = sae_result["cosine"]

print("SAE:", sae_result["sae"]["repo"], "/", sae_result["sae"]["id"])
print("features:", f'{sae_result["sae"]["dict_size"]:,}')
print("strongest |cosine|:", round(cosine["max_abs"], 3))
print("features with |cosine| > 0.2:", cosine["n_abs_gt_0_2"])
print("axis captured by top-50 feature span:", f'{capture["axis"]:.1%}')
print("random mean:", f'{capture["random_mean"]:.1%}')
print("random 95th percentile:", f'{capture["random_p95"]:.1%}')

plt.figure(figsize=(7, 3.5))
plt.hist(capture["random_values"], bins=14, alpha=.75, label="64 random directions")
plt.axvline(capture["axis"], color="#73ba25", linewidth=3, label="valence axis")
plt.xlabel("fraction captured by each direction's best 50-feature span")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.show()

print("\\nMost aligned positive-pole features:")
for item in cosine["top_positive"][:5]:
    print(f'  #{item["feature_id"]:6d}  cosine={item["cosine"]:+.3f}')
print("Most aligned negative-pole features:")
for item in cosine["top_negative"][:5]:
    print(f'  #{item["feature_id"]:6d}  cosine={item["cosine"]:+.3f}')"""),
    md("""SAE alignment says the direction is structured in a feature basis. It
does **not** name the structure. For that we need a readout—and different
readouts make different mistakes."""),
    md("""## 3 · Load Qwen, J-Lens and the NLA

This is the slow section. It loads:

- Qwen 2.5 7B Instruct in 4-bit;
- the public NLA adapter;
- the fitted Qwen J-Lens (~700 MB).

The J-Lens install deliberately uses `--no-deps`: upgrading Colab's
Transformers build can make the 4-bit load exceed T4 memory."""),
    code("""%pip -q install -U bitsandbytes peft accelerate
!git clone -q https://github.com/anthropics/jacobian-lens
%pip -q install -e jacobian-lens --no-deps
import sys
sys.path.insert(0, "jacobian-lens")"""),
    code("""from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
AV_ADAPTER = "anicka/nla-qwen2.5-7b-universal-av-grpo"
LAYER = 19
DEPTH_PCT = 71
INJECT_CHAR = "\\u320e"
INJECT_SCALE = 150.0
device = "cuda"

assert torch.cuda.is_available(), "Runtime -> Change runtime type -> GPU"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": 0}
)
model = PeftModel.from_pretrained(base, AV_ADAPTER).eval()

inject_ids = tok.encode(INJECT_CHAR, add_special_tokens=False)
assert len(inject_ids) == 1
inject_id = inject_ids[0]
print("loaded Qwen + NLA adapter")"""),
    code("""def get_layers(m):
    b = m.base_model.model if hasattr(m, "base_model") else m
    inner = b.model if hasattr(b, "model") else b
    return inner.layers

def normalize_to(v, scale=INJECT_SCALE):
    return v * (scale / v.float().norm().clamp_min(1e-12))

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
        "<explanation>"
    )

def describe(activation, depth=DEPTH_PCT, max_new_tokens=120):
    chat = tok.apply_chat_template(
        [{"role": "user", "content": av_prompt(depth)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(chat, add_special_tokens=False)
    pos = ids.index(inject_id)
    input_ids = torch.tensor([ids], device=device)
    embeds = model.get_input_embeddings()(input_ids).clone()
    injected = activation.to(device=embeds.device, dtype=embeds.dtype)
    embeds[0, pos] = normalize_to(injected)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            inputs_embeds=embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    seq = output[0]
    generated = seq[len(ids):] if seq.shape[0] > len(ids) else seq
    return tok.decode(generated, skip_special_tokens=True).split("</explanation>")[0].strip()

print("NLA helper ready")"""),
    code("""import jlens

lens = jlens.JacobianLens.from_pretrained(
    "anicka/jlens-qwen2.5-7b-instruct",
    filename="qwen2.5-7b-instruct_jlens.pt",
)
jm = jlens.from_hf(base, tok)
assert LAYER in lens.source_layers
print("J-Lens ready at layer", LAYER)"""),
    md("""## 4 · J-Lens: which words does each pole support?

J-Lens estimates how later layers transform the direction, then reads the
result with Qwen's own output vocabulary. We inspect both signs because the
axis has a pleasant and an unpleasant pole.

The second cell tests five labels chosen **before** looking at the output.
Their best alignment is compared with 5,000 random directions. This prevents
an evocative token list from grading itself."""),
    code("""def top_tokens(logits, k=12):
    ids = logits.topk(k).indices.tolist()
    return [tok.decode([i]).strip() or repr(tok.decode([i])) for i in ids]

with torch.no_grad():
    pos_state = lens.transport(direction.unsqueeze(0), LAYER)
    neg_state = lens.transport((-direction).unsqueeze(0), LAYER)
    pos_logits = jm.unembed(pos_state)[0]
    neg_logits = jm.unembed(neg_state)[0]

print("positive pole:", top_tokens(pos_logits))
print("negative pole:", top_tokens(neg_logits))"""),
    code("""CANDIDATES = [
    ("pleasant", "unpleasant"),
    ("happy", "sad"),
    ("joy", "pain"),
    ("positive", "negative"),
    ("comfort", "distress"),
]

def phrase_vector(text):
    ids = tok.encode(text, add_special_tokens=False)
    weight = base.get_output_embeddings().weight
    return weight[ids].detach().float().cpu().mean(0)

J = lens.jacobians[LAYER].float()
pulled = []
scores = []
for positive, negative in CANDIDATES:
    output_contrast = phrase_vector(positive) - phrase_vector(negative)
    hidden_contrast = J.T @ output_contrast
    hidden_contrast /= hidden_contrast.norm().clamp_min(1e-12)
    pulled.append(hidden_contrast)
    scores.append(float(direction @ hidden_contrast))

g = torch.Generator().manual_seed(42)
randoms = torch.randn(5000, direction.numel(), generator=g)
randoms /= randoms.norm(dim=1, keepdim=True)
candidate_matrix = torch.stack(pulled)
random_max = (randoms @ candidate_matrix.T).abs().max(dim=1).values
observed = max(abs(score) for score in scores)
exceedances = int((random_max >= observed).sum())
p_value = (exceedances + 1) / (len(random_max) + 1)

for pair, score in zip(CANDIDATES, scores):
    print(f"{pair[0]:>10s} vs {pair[1]:<10s}: cosine {score:+.3f}")
print("best absolute cosine:", round(observed, 3))
print(
    "Monte Carlo p (plus-one):",
    f"{p_value:.4f}",
    f"({exceedances}/{len(random_max)} random directions >= observed)",
)"""),
    md("""A strong result means the direction is connected to valence words
through Qwen's own downstream computation. It still does not tell us how the
features combine into a situation or sentence."""),
    md("""## 5 · NLA: ask for prose hypotheses

The NLA was trained on complete residual-stream activations, not isolated
contrast directions. We therefore describe the two **mean activation states**
whose difference defines the axis. Feeding it the bare direction would be an
out-of-distribution input and can produce fluent nonsense.

The NLA can combine many clues into a description, but its language model can
also invent details. Compare the broad distinction between the centroids; do
not treat every named scenario as evidence."""),
    code("""print("PLEASANT CENTROID\\n", describe(pleasant_centroid))
print("\\nUNPLEASANT CENTROID\\n", describe(unpleasant_centroid))"""),
    md("""### Sanity check: centroids vs real activations

A centroid is a smoothed object — the mean of many states has a smaller
norm and slightly atypical geometry, so it is itself mildly
out-of-distribution for the verbalizer. Before trusting the two
descriptions above, read one **real single activation** per pole. The
probe prompts below are written fresh here (they are not the extraction
set), so this doubles as a light out-of-sample check. Because the live
4-bit model and the extraction artifact need not share an absolute
projection origin, compare the ordering: pleasant prompts should project
higher than unpleasant prompts."""),
    code("""PROBE_PROMPTS = {
    "pleasant": [
        "We watched the sunrise from the tent and made pancakes together.",
        "The vet says the kitten is perfectly healthy and can come home.",
    ],
    "unpleasant": [
        "The landlord says we have thirty days to leave the apartment.",
        "The test results came back and the doctor wants to talk in person.",
    ],
}

@torch.no_grad()
def block19_state(text):
    chat = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(chat, return_tensors="pt").to(device)
    grabbed = {}

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = hidden[:, -1, :].detach()

    handle = get_layers(model)[LAYER].register_forward_hook(hook)
    try:
        with model.disable_adapter():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return grabbed["h"].squeeze(0).float().cpu()

probe_states = {
    pole: [block19_state(p) for p in prompts]
    for pole, prompts in PROBE_PROMPTS.items()
}
probe_projections = {
    pole: [float(h @ direction) for h in states]
    for pole, states in probe_states.items()
}
probe_gap = (
    min(probe_projections["pleasant"])
    - max(probe_projections["unpleasant"])
)
print(f"fresh ordering gap: {probe_gap:+.1f}")

for pole, prompts in PROBE_PROMPTS.items():
    print(f"== {pole.upper()} probe (fresh prompts, not the extraction set)")
    for p, proj in zip(prompts, probe_projections[pole]):
        print(f"   proj {proj:+7.1f}  {p}")
    print("first pre-specified activation, described:")
    print(describe(probe_states[pole][0]))
    print()"""),
    md("""A positive ordering gap means every pleasant probe scored above every
unpleasant probe in this small fresh set. If the ordering reverses, or a real
activation's description disagrees sharply with its centroid, report it. The
centroid summarises the cloud; single states are what the model computes."""),
    md("""Questions to ask:

- Does the broad affective distinction flip with the sign?
- Which details remain stable if you sample more than once?
- Does either description introduce a person, event or object that no other
  witness supports?

Notebook 06 develops the last question into an experimental entity
cross-check."""),
    md("""## What the witnesses jointly support

| witness | question | characteristic failure |
|---|---|---|
| contrastive extraction | what separates the two prompt sets? | confounds in the sets define the axis |
| SAE | is the direction structured in a learned feature basis? | large dictionaries create accidental similarities |
| J-Lens | what words can downstream computation produce from it? | linear average misses context-specific routes |
| NLA | what contextual description can a trained decoder produce? | language prior fills missing details |

Agreement narrows the interpretation. Disagreement is not an embarrassment:
it tells us which assumption to test next."""),
    md("""## SELF-CHECK"""),
    code("""assert direction.shape == (base.config.hidden_size,)
centroid_difference = pleasant_centroid - unpleasant_centroid
centroid_difference /= centroid_difference.norm()
assert torch.allclose(direction, centroid_difference, atol=1e-5)
assert axis_data["block_index"] == LAYER
assert sae_result["block_index"] == LAYER
assert all(torch.isfinite(torch.tensor(values)).all() for values in probe_projections.values())
print("SELF-CHECK OK — exact layer matched; SAE, J-Lens and NLA ran")"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(HERE / FN).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {FN}")
