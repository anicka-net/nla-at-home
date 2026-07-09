"""nla_lib — single source of truth for NLA constants, templates, and loaders.

Why this exists: model constants (injection chars, HF ids), prompt templates,
and adapter-loading logic used to be copy-pasted across ~20 scripts. That
caused real failures: an AV prompt copy silently dropped the depth sentence,
a GRPO script hardcoded phi4's 40 layers into a depth computation used for
qwen (28), and an AR loader assumed a value-heads format that the universal
qwen AR doesn't have. Everything here is the ONE definition; scripts import
from this module and must not re-declare any of it.

Rules:
- Templates are FROZEN interfaces. Shipped adapters were trained against
  these exact strings; changing a byte silently breaks every published
  checkpoint. tests/test_nla_lib.py pins them and also scans legacy scripts
  for drift. Changes require human review (see CLAUDE.md).
- Layer counts and d_model are NOT authoritative here: always take them from
  the activation file / model config / nla_meta.yaml at runtime. The
  reference values in ModelSpec exist for documentation and sanity checks
  only — a hardcoded layer count in control flow is how we got burned.
- Heavy deps (torch, transformers, peft) are imported lazily inside the
  functions that need them, so light consumers can import this module fast.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Depth grid
# ---------------------------------------------------------------------------

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]

INJECTION_SCALE = 150.0


def nearest_depth_pct(layer, n_layers):
    """Map a layer index to the nearest depth percentage in DEPTH_PCTS.

    n_layers MUST come from the activation file / model config of the model
    actually in use — never hardcode it (phi4 has 40, qwen25-7b has 28,
    phi4-mini 32, gemma3-1b 26).
    """
    depth = layer * 100 / n_layers
    return min(DEPTH_PCTS, key=lambda p: abs(p - depth))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    injection_char: str        # None if no injection token established yet
    trust_remote_code: bool = False
    # Reference only — verify against the artifact at runtime:
    n_layers: int = None
    d_model: int = None
    notes: str = ""


MODELS = {
    "qwen25-7b": ModelSpec(
        key="qwen25-7b", hf_id="Qwen/Qwen2.5-7B-Instruct",
        injection_char="㈎",  # ㈎ (token id 149705)
        n_layers=28, d_model=3584),
    "qwen3-4b": ModelSpec(
        key="qwen3-4b", hf_id="Qwen/Qwen3-4B",
        injection_char="㈎",
        n_layers=36, d_model=2560,
        notes="chat template needs enable_thinking=False"),
    "gemma3-1b": ModelSpec(
        key="gemma3-1b", hf_id="google/gemma-3-1b-it",
        injection_char="⎝",  # ⎝
        n_layers=26, d_model=1152,
        notes="massive-activation outlier dim dominates norms; "
              "cosine/PCA geometry unreliable"),
    "phi4-mini": ModelSpec(
        key="phi4-mini", hf_id="microsoft/Phi-4-mini-instruct",
        injection_char="★",  # ★ (token id 12087 in phi4-mini)
        n_layers=32, d_model=3072,
        notes="repo ships modeling_phi3.py, but every shipped adapter was "
              "trained with trust_remote_code=False (native transformers "
              "Phi3 implementation) — keep False"),
    "phi4": ModelSpec(
        key="phi4", hf_id="microsoft/phi-4",
        injection_char="★",  # ★ (token id 27347 in phi4 — differs from mini!)
        n_layers=40, d_model=5120,
        notes="repo ships no custom code; trust_remote_code irrelevant"),
    "llama-8b": ModelSpec(
        key="llama-8b", hf_id="meta-llama/Llama-3.1-8B-Instruct",
        injection_char=None,
        n_layers=32, d_model=4096,
        notes="no injection token established; extraction only"),
}


def get_model(key):
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(MODELS)}")
    return MODELS[key]


# Legacy plain-dict views, so retrofitted scripts keep their interfaces:
MODELS_HF = {k: m.hf_id for k, m in MODELS.items()}
INJECTION_CHARS = {k: m.injection_char for k, m in MODELS.items()
                   if m.injection_char is not None}
# For injection-dependent CLIs (training/inference): only models with an
# established injection token. Using MODELS_HF for argparse choices would
# accept llama-8b and crash later on INJECTION_CHARS lookup.
INJECTABLE_MODELS_HF = {k: v for k, v in MODELS_HF.items()
                        if k in INJECTION_CHARS}


# ---------------------------------------------------------------------------
# Prompt templates (FROZEN — see module docstring)
# ---------------------------------------------------------------------------

def make_av_prompt(depth_pct, injection_char):
    """The canonical AV prompt (universal, depth-conditioned).

    This is what every universal AV adapter was trained on. The single-layer
    era used a variant without the depth sentence; that variant lives only in
    the legacy scripts that serve those old adapters.
    """
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        f"Here is the vector from depth {depth_pct}% of the network:\n\n"
        f"<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )


# AR prompt family. Four shipped variants — the adapter's training script
# determines which one it understands:
#   nodepth      — phi4 stage2 ARs (value heads read at last token; no
#                  injection char in the prompt)
#   depth_sl     — train_universal_ar.py "self-layer" ARs (depth-conditioned,
#                  trailing injection char marks the readout position)
#   reconstruct  — phi4-mini universal AR (value heads; prose instruction,
#                  served by brain_in_jar.py)
AR_TEMPLATE_NODEPTH = (
    "Summary of the following text: <text>{explanation}</text> <summary>")
AR_TEMPLATE_DEPTH_SL = (
    "Summary of the following text from depth {depth}%: "
    "<text>{explanation}</text> <summary>{inj}")
AR_TEMPLATE_RECONSTRUCT = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into a model's internal states. Below is a description of an activation vector:\n\n"
    "<explanation>{explanation}</explanation>\n\n"
    "Based on this description, reconstruct the activation vector."
)


def make_ar_prompt_nodepth(explanation):
    return AR_TEMPLATE_NODEPTH.format(explanation=explanation)


def make_ar_prompt_depth_sl(explanation, depth_pct, injection_char):
    return AR_TEMPLATE_DEPTH_SL.format(
        explanation=explanation, depth=depth_pct, inj=injection_char)


def encode_ar_prompt(tokenizer, template, explanation, injection_char,
                     max_length):
    """Tokenize an AR prompt while preserving its trailing readout token."""
    prefix, marker, suffix = template.partition("{explanation}")
    if not marker:
        raise ValueError("AR template is missing {explanation}")
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    if len(prefix_tokens) + len(suffix_tokens) > max_length:
        raise ValueError("AR prompt framing exceeds max_length")
    desc_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    room = max_length - len(prefix_tokens) - len(suffix_tokens)
    tokens = prefix_tokens + desc_tokens[:room] + suffix_tokens
    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)
    if len(inject_id) != 1:
        raise ValueError("AR injection character must encode to one token")
    positions = [i for i, token in enumerate(tokens) if token == inject_id[0]]
    if len(positions) != 1:
        raise ValueError(
            f"expected one AR injection token, got {len(positions)}")
    return tokens, positions[0]


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def normalize_activation(v, target_scale=INJECTION_SCALE):
    """L2-normalize activation TO target_scale (Anthropic's approach).

    Works on a single vector [d] or a batch [..., d] (last dim is the
    vector). THE COMMON MISTAKE is multiplying BY the scale instead: a
    vector with norm 129 must come out with norm 150, not 19350.
    """
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


# ---------------------------------------------------------------------------
# Adapter format detection + loading
# ---------------------------------------------------------------------------

AR_FORMAT_LORA_SL = "lora_sl"      # train_universal_ar.py output
AR_FORMAT_HEADS_DIR = "heads_dir"  # frozen backbone + value_heads.safetensors
AR_FORMAT_STAGE2_PT = "stage2_pt"  # phi_ar_stage2 *_best.pt + sibling heads


def detect_ar_format(ar_checkpoint):
    """Classify an AR checkpoint path into one of the three shipped formats."""
    p = Path(ar_checkpoint)
    if p.suffix == ".pt":
        return AR_FORMAT_STAGE2_PT
    if p.is_dir():
        if (p / "value_heads.safetensors").exists():
            return AR_FORMAT_HEADS_DIR
        if (p / "adapter_config.json").exists():
            return AR_FORMAT_LORA_SL
    raise ValueError(
        f"Unrecognized AR checkpoint format at {ar_checkpoint}: expected a "
        f"stage2 .pt file, a directory with value_heads.safetensors, or a "
        f"LoRA adapter directory (adapter_config.json)")


def read_nla_meta(adapter_dir):
    """Read nla_meta.yaml from an adapter directory (None if absent)."""
    import yaml
    p = Path(adapter_dir) / "nla_meta.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


class LoraSLReward:
    """Unified reconstruction interface for lora_sl ARs.

    reconstruct(descriptions, layers, device) -> {layer: tensor[N, d]}.
    Reconstruction = the model's own hidden state at layer L (output of
    block L = hidden_states[L+1]) at the trailing injection token, exactly
    mirroring the train_universal_ar.py training hook.
    """

    MAX_LEN = 512

    def __init__(self, model, tokenizer, n_layers, injection_char,
                 layer_means=None, min_layer=0):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.injection_char = injection_char
        self.layer_means = layer_means
        self.min_layer = min_layer

    def reconstruct(self, descriptions, layers, device):
        import torch
        recons = {}
        for L in layers:
            if L < self.min_layer:
                raise ValueError(
                    f"AR was trained only for layers >= {self.min_layer}; "
                    f"cannot reconstruct layer {L}")
            depth = nearest_depth_pct(L, self.n_layers)
            template = AR_TEMPLATE_DEPTH_SL.replace(
                "{depth}", str(depth)).replace("{inj}", self.injection_char)
            rows = [
                encode_ar_prompt(
                    self.tokenizer, template, d, self.injection_char,
                    self.MAX_LEN)[0]
                for d in descriptions
            ]
            width = max(map(len, rows))
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.full(
                (len(rows), width), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for i, row in enumerate(rows):
                input_ids[i, -len(row):] = torch.tensor(row, device=device)
                attention_mask[i, -len(row):] = 1
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
            recon = out.hidden_states[L + 1][:, -1, :].float()
            if self.layer_means is not None:
                recon = recon + self.layer_means[L].to(recon.device)
            recons[L] = recon
        return recons


def load_ar_lora_sl(ar_checkpoint, base_model_name, device, trust_remote,
                    n_layers, injection_char):
    """Load a train_universal_ar.py-style AR as a frozen LoraSLReward."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    meta = read_nla_meta(ar_checkpoint)
    training = (meta or {}).get("training", {})
    layer_means = None
    if training.get("mean_subtract"):
        means_path = Path(ar_checkpoint) / "layer_means.pt"
        if not means_path.exists():
            raise FileNotFoundError(
                f"mean-subtracted AR is missing {means_path}")
        layer_means = torch.load(
            means_path, map_location="cpu", weights_only=True)
    min_layer = int(training.get("min_layer", 0))

    tokenizer = AutoTokenizer.from_pretrained(
        ar_checkpoint, trust_remote_code=trust_remote)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote)
    model = PeftModel.from_pretrained(backbone, ar_checkpoint)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return LoraSLReward(
        model, tokenizer, n_layers, injection_char,
        layer_means=layer_means, min_layer=min_layer)
