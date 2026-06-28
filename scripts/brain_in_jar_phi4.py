#!/usr/bin/env python3
"""
Brain in a Jar — Phi-4 14B, GRPO AV + AR v2.

Runs a prompt through Phi-4, shows:
- Model's normal output
- What each layer is "thinking" (via GRPO-trained AV)
- AR confidence (mean-subtracted cosine)

Usage:
  python3 brain_in_jar_phi4.py "How to develop bodhicitta?"
  python3 brain_in_jar_phi4.py  # interactive mode
  python3 brain_in_jar_phi4.py --skip-ar "What is love?"  # faster, no confidence
  python3 brain_in_jar_phi4.py --layers 16,22,28,36 "test"  # specific layers
"""
import torch
import argparse
import sys
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = Path(__file__).resolve().parent.parent

# Phi-4 14B config
BASE_MODEL = "microsoft/phi-4"
N_LAYERS = 40
D_MODEL = 5120
INJECTION_CHAR = "★"
INJECTION_SCALE = 150.0

# Default adapters
DEFAULT_AV = str(REPO / "output/nla-phi4-av-arnative-grpo")
DEFAULT_AR = str(REPO / "output/nla-phi4-universal-ar-v2")

# AR layers (from nla_meta.yaml)
AR_LAYERS = [4, 10, 16, 19, 25, 32, 38]

# Depth percentages for display
DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]

# AR prompt template
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

COLORS = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "red": "\033[31m", "magenta": "\033[35m", "blue": "\033[34m",
}

def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def layer_to_depth_pct(layer_idx):
    return round(100 * (layer_idx + 0.5) / N_LAYERS)

def depth_color(pct):
    if pct <= 20: return "blue"
    elif pct <= 45: return "cyan"
    elif pct <= 70: return "green"
    elif pct <= 85: return "yellow"
    else: return "red"

def normalize_activation(v, target_scale):
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)

def make_av_prompt(depth_pct):
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
        f"<concept>{INJECTION_CHAR}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )


class BrainInJar:
    def __init__(self, av_path, ar_path, device="cuda", skip_ar=False):
        self.device = device
        self.skip_ar = skip_ar

        # Load base + AV adapter
        print(c("Loading Phi-4 + GRPO AV adapter...", "dim"), flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.av_model = PeftModel.from_pretrained(base, av_path).eval()
        print(c("  AV loaded ✓", "green"))

        # Injection token
        inj_ids = self.tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)
        assert len(inj_ids) == 1, f"Injection char encodes to {len(inj_ids)} tokens"
        self.injection_token_id = inj_ids[0]

        # Load AR (backbone + LoRA adapter + value heads)
        if not skip_ar:
            print(c("Loading AR backbone + LoRA + value heads...", "dim"), flush=True)
            ar_base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
            self.ar_backbone = PeftModel.from_pretrained(ar_base, ar_path).eval()
            # Strip final norm + lm_head for hidden state extraction
            inner = self.ar_backbone.base_model.model.model if hasattr(self.ar_backbone.base_model.model, "model") else self.ar_backbone.base_model.model
            for attr in ("norm", "final_layernorm", "ln_f", "final_norm"):
                if hasattr(inner, attr):
                    setattr(inner, attr, torch.nn.Identity())
                    break
            self.ar_backbone.base_model.model.lm_head = torch.nn.Identity()
            for p in self.ar_backbone.parameters():
                p.requires_grad = False

            # Load value heads
            vh_path = Path(ar_path) / "value_heads.safetensors"
            self.value_heads = {}
            with safe_open(str(vh_path), framework="pt") as f:
                for key in f.keys():
                    layer_idx = int(key.split(".")[1]) if "." in key else int(key)
                    w = f.get_tensor(key).to(device)
                    vh = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype, device=device)
                    vh.weight = torch.nn.Parameter(w)
                    vh.eval()
                    self.value_heads[layer_idx] = vh
            print(c(f"  AR loaded ✓ ({len(self.value_heads)} value heads)", "green"))
        else:
            self.ar_backbone = None
            self.value_heads = None

    def extract_hidden_states(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_str, return_tensors="pt").to(self.device)
        # Extract from BASE model (LoRA disabled) for clean activations
        with self.av_model.disable_adapter(), torch.no_grad():
            outputs = self.av_model(
                **inputs, output_hidden_states=True, use_cache=False)
        hidden = [h[0, -1, :].detach() for h in outputs.hidden_states]
        return hidden

    def generate_output(self, prompt, max_tokens=200):
        messages = [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_str, return_tensors="pt").to(self.device)
        # Generate from BASE model (LoRA disabled) for true Phi-4 output
        with self.av_model.disable_adapter(), torch.no_grad():
            output = self.av_model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        reply = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return reply

    def verbalize(self, activation, depth_pct, max_tokens=150):
        prompt_text = make_av_prompt(depth_pct)
        chat_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(chat_str, add_special_tokens=False)

        inject_pos = None
        for i, tid in enumerate(tokens):
            if tid == self.injection_token_id:
                inject_pos = i
                break
        if inject_pos is None:
            return "[injection token not found]"

        input_ids = torch.tensor([tokens], device=self.device)
        embed_layer = self.av_model.get_input_embeddings()
        with torch.no_grad():
            embeddings = embed_layer(input_ids).clone()
            embeddings[0, inject_pos, :] = normalize_activation(
                activation.to(embeddings.dtype), INJECTION_SCALE).to(embeddings.dtype)
            output = self.av_model.generate(
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True)

        text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        if "</explanation>" in text:
            text = text.split("</explanation>")[0].strip()
        return text

    def ar_confidence(self, description, actual_activation, layer_idx):
        if self.ar_backbone is None:
            return 0.0
        prompt = AR_TEMPLATE.replace("{explanation}", description)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True,
                                       truncation=True, max_length=256)
        input_ids = torch.tensor([tokens], device=self.device)

        with torch.no_grad():
            outputs = self.ar_backbone(input_ids=input_ids, use_cache=False,
                                       output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            last_h = hidden[0, -1]
            reconstructed = self.value_heads[layer_idx](last_h.unsqueeze(0)).squeeze(0)

        cos = torch.nn.functional.cosine_similarity(
            reconstructed.float().unsqueeze(0),
            actual_activation.float().unsqueeze(0)).item()
        return cos

    def run(self, prompt, layers=None):
        print(f"\n{c('═' * 72, 'bold')}")
        print(f"  {c('PROMPT:', 'bold')} {prompt}")
        print(c('═' * 72, 'bold'))

        # Generate normal output
        print(f"\n  {c('Generating...', 'dim')}", flush=True)
        reply = self.generate_output(prompt)
        print(f"  {c('OUTPUT:', 'bold')} {reply[:600]}")

        # Extract hidden states
        print(f"\n  {c('Extracting hidden states...', 'dim')}", flush=True)
        hidden = self.extract_hidden_states(prompt)

        # Determine which layers to show
        if layers is None:
            layers = AR_LAYERS  # default: the 7 AR layers
        layers = [l for l in layers if l < len(hidden) - 1]

        print(f"\n  {c('LAYER-BY-LAYER VIEW:', 'bold')}")
        if not self.skip_ar:
            print(f"  {'':>14}  {'AR confidence':>22}  Description")
        else:
            print(f"  {'':>14}  Description")
        print(f"  {c('─' * 68, 'dim')}")

        for layer_idx in layers:
            depth_pct = layer_to_depth_pct(layer_idx)
            activation = hidden[layer_idx + 1]

            description = self.verbalize(activation, depth_pct)

            if not self.skip_ar and layer_idx in self.value_heads:
                confidence = self.ar_confidence(description, activation, layer_idx)
                conf_bar = confidence_bar(confidence)
                cos_str = c(f"{confidence:.3f}", "dim")
            else:
                conf_bar = ""
                cos_str = ""

            dc = depth_color(depth_pct)
            pct_str = c(f"L{layer_idx:02d} ({depth_pct:3d}%)", dc)

            desc_lines = description.split("\n")
            first_line = desc_lines[0][:90] if desc_lines else ""

            if not self.skip_ar:
                print(f"  {pct_str}  {conf_bar} {cos_str}  {first_line}")
            else:
                print(f"  {pct_str}  {first_line}")

            for line in desc_lines[1:3]:
                line = line.strip()[:90]
                if line:
                    pad = 42 if not self.skip_ar else 16
                    print(f"  {'':>{pad}}  {c(line, 'dim')}")

        print(f"  {c('─' * 68, 'dim')}\n")


def confidence_bar(cos, width=20):
    filled = int(cos * width)
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    if cos >= 0.7: col = "green"
    elif cos >= 0.5: col = "yellow"
    else: col = "red"
    return c(bar, col)


def main():
    parser = argparse.ArgumentParser(
        description="Brain in a Jar — Phi-4 14B NLA viewer (GRPO AV)")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Prompt (interactive if omitted)")
    parser.add_argument("--av-adapter", default=DEFAULT_AV)
    parser.add_argument("--ar-checkpoint", default=DEFAULT_AR)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: AR layers)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Show all 40 layers (slow)")
    parser.add_argument("--skip-ar", action="store_true",
                        help="Skip AR confidence (faster, AV descriptions only)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    brain = BrainInJar(args.av_adapter, args.ar_checkpoint,
                       device=args.device, skip_ar=args.skip_ar)

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif args.all_layers:
        layers = list(range(N_LAYERS))
    else:
        layers = None  # default AR layers

    if args.prompt:
        brain.run(args.prompt, layers)
    else:
        print(f"\n{c('Brain in a Jar', 'bold')} — Phi-4 14B + GRPO AV")
        print(f"Showing layers: {AR_LAYERS}")
        print(f"Type a prompt. {c('Ctrl+C to exit.', 'dim')}\n")
        while True:
            try:
                prompt = input(c("prompt> ", "cyan"))
                if not prompt.strip():
                    continue
                brain.run(prompt, layers)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{c('Done.', 'dim')}")
                break


if __name__ == "__main__":
    main()
