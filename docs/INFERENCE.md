# Running the NLA — shell inference guide

The leading result: Phi-4 14B with AR-native GRPO adapter (AV) and
value-head AR. Feed any prompt, get a layer-by-layer readout of what
Phi-4 computes as it prepares its answer.

## Setup

```bash
pip install torch transformers peft safetensors pyyaml
```

Models download automatically from HuggingFace on first run (~15GB total
for base + two adapters). Requires a GPU with 28GB+ VRAM for the full
pipeline (AV + AR), or 14GB+ for AV-only mode.

## Interactive brain-in-a-jar

```bash
# AV-only (faster, no confidence scores) — needs ~14GB VRAM
python3 scripts/brain_in_jar_phi4.py --skip-ar

# Full pipeline with AR confidence — needs ~28GB VRAM (loads model twice)
python3 scripts/brain_in_jar_phi4.py

# Single prompt (non-interactive)
python3 scripts/brain_in_jar_phi4.py --skip-ar "What is equanimity?"

# Specific layers only
python3 scripts/brain_in_jar_phi4.py --skip-ar --layers 16,22,32,38 "prompt"
```

First run takes 2-3 minutes (model download + load). Subsequent prompts
in interactive mode take ~2 minutes each (7 layers × ~15s generation).

## What the output means

Each layer shows what Phi-4 is computing at that depth:

- **Early (L4-L10, 10-26%)**: token-level patterns, surface echoes of
  the input. Often noisy or wrong.
- **Middle (L16-L25, 41-64%)**: semantic content, topic, task
  identification. Converges toward the right answer.
- **Late (L32-L38, 81-96%)**: response strategy, the literal opening
  tokens of the reply. Safety/hedging circuitry fires here when
  triggered.

AR confidence (when enabled) measures how well the AR network can
reconstruct the original activation from the description. Higher =
the description carries more geometric information. Values above 0.6
are strong; below 0.4 the description may be confabulating.

## Adapters on HuggingFace

| Adapter | Role | HuggingFace |
|---------|------|-------------|
| AV (GRPO, current best) | Generates descriptions | [anicka/nla-phi4-av-arnative-grpo](https://huggingface.co/anicka/nla-phi4-av-arnative-grpo) |
| AV (SL, prior version) | Generates descriptions | [anicka/nla-phi4-universal-av-v2](https://huggingface.co/anicka/nla-phi4-universal-av-v2) |
| AR (reconstructor) | Verifies descriptions | [anicka/nla-phi4-universal-ar-v2](https://huggingface.co/anicka/nla-phi4-universal-ar-v2) |

The GRPO adapter improves round-trip faithfulness by 23% over the SL
version (0.585 vs 0.474 mean-subtracted cosine), closing 77% of the
gap to the AR ceiling. It produces terser, more discriminative
descriptions that name specific tokens and task structures rather than
vague category labels.

## Programmatic use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load
base = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4", torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(
    base, "anicka/nla-phi4-av-arnative-grpo").eval()
tokenizer = AutoTokenizer.from_pretrained("anicka/nla-phi4-av-arnative-grpo")

# Extract activation from any layer
prompt = "Your input text here"
messages = [{"role": "user", "content": prompt}]
chat_str = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_str, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states[L+1] is the output of layer L
    activation = out.hidden_states[23][0, -1, :]  # layer 22, last token

# Verbalize
INJECTION_CHAR = "★"
INJECTION_SCALE = 150.0

av_prompt = (
    "You are a meticulous AI researcher conducting an important "
    "investigation into activation vectors from a language model. "
    "Your overall task is to describe the semantic content of that "
    "activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your "
    "context, along with the network depth where it was extracted. "
    "You must then produce an explanation for the vector, enclosed "
    "within <explanation> tags. The explanation consists of 2-3 text "
    "snippets describing that vector.\n\n"
    "Here is the vector from depth 55% of the network:\n\n"
    f"<concept>{INJECTION_CHAR}</concept>\n\n"
    "Please provide an explanation.\n\n"
    "<explanation>"
)

chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": av_prompt}],
    tokenize=False, add_generation_prompt=True)
tokens = tokenizer.encode(chat, add_special_tokens=False)
inject_pos = tokens.index(27347)

input_ids = torch.tensor([tokens], device="cuda")
embeddings = model.get_input_embeddings()(input_ids).clone()

norm = activation.float().norm().clamp_min(1e-12)
normalized = activation * (INJECTION_SCALE / norm)
embeddings[0, inject_pos, :] = normalized.to(embeddings.dtype)

with torch.no_grad():
    output = model.generate(
        inputs_embeds=embeddings,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True)

text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
description = text.split("</explanation>")[0].strip()
print(description)
```

## Evaluation

Round-trip eval measures end-to-end faithfulness: AV generates a
description → AR reconstructs the activation from that description →
cosine with ground truth (mean-subtracted).

```bash
# Run the three phases separately (each loads the 14B model once)
python3 scripts/eval_roundtrip_phi4.py --phase split \
  --av-adapter output/nla-phi4-av-arnative-grpo \
  --out-dir output/roundtrip_grpo

python3 scripts/eval_roundtrip_phi4.py --phase av \
  --av-adapter output/nla-phi4-av-arnative-grpo \
  --out-dir output/roundtrip_grpo

python3 scripts/eval_roundtrip_phi4.py --phase ar \
  --out-dir output/roundtrip_grpo
```

Expected output: mean roundtrip cosine ~0.585 on the 49-text double
holdout.
