#!/usr/bin/env python3
"""
Brain in a Jar — watch a language model think, layer by layer.

Runs a prompt through Phi-4 Mini and shows:
- The model's normal output
- What each layer is "thinking about" (via NLA activation verbalizer)
- How confident each description is (via AR cosine similarity)

Usage:
  python3 brain_in_jar.py --av-adapter ./av-adapter --ar-checkpoint ./ar-checkpoint
  python3 brain_in_jar.py --av-adapter ./av-adapter --ar-checkpoint ./ar-checkpoint --prompt "Tell me about yourself"
"""
import torch
import yaml
import argparse
import sys
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
INJECTION_CHAR = "★"
INJECTION_SCALE = 150.0
DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]
N_LAYERS = 32

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
}

def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def layer_to_depth_pct(layer_idx, n_layers=32):
    return round(100 * (layer_idx + 0.5) / n_layers)


def depth_color(pct):
    if pct <= 20:
        return "blue"
    elif pct <= 45:
        return "cyan"
    elif pct <= 70:
        return "green"
    elif pct <= 85:
        return "yellow"
    else:
        return "red"


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


AR_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into a model's internal states. Below is a description of an activation vector:\n\n"
    "<explanation>{explanation}</explanation>\n\n"
    "Based on this description, reconstruct the activation vector."
)


def load_av_model(av_adapter_path, device):
    print("Loading base model...", end=" ", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("done")

    print("Loading AV adapter...", end=" ", flush=True)
    av_model = PeftModel.from_pretrained(base, av_adapter_path)
    av_model = av_model.to(device).eval()
    print("done")

    injection_id = tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)
    if len(injection_id) != 1:
        raise ValueError(f"Injection char '{INJECTION_CHAR}' encodes to {len(injection_id)} tokens, need exactly 1")
    injection_token_id = injection_id[0]

    return av_model, tokenizer, injection_token_id


def load_ar_heads(ar_checkpoint_path, device):
    print("Loading AR value heads...", end=" ", flush=True)
    vh_path = Path(ar_checkpoint_path) / "value_heads.safetensors"
    value_heads = {}
    with safe_open(str(vh_path), framework="pt") as f:
        for key in f.keys():
            layer_idx = int(key.split(".")[1])
            w = f.get_tensor(key)
            vh = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
            vh.weight = torch.nn.Parameter(w)
            vh = vh.to(device).eval()
            value_heads[layer_idx] = vh
    print(f"done ({len(value_heads)} layers)")
    return value_heads


def extract_hidden_states(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_str, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            **inputs, output_hidden_states=True, use_cache=False)
        hidden_states = [h[:, -1, :].detach() for h in outputs.hidden_states]

    return hidden_states, inputs


def generate_output(model, tokenizer, prompt, device, max_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_str, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return reply


def verbalize_layer(av_model, tokenizer, activation, depth_pct,
                    injection_token_id, device, max_tokens=100):
    prompt_text = make_av_prompt(depth_pct)
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    inject_pos = None
    for i, tid in enumerate(tokens):
        if tid == injection_token_id:
            inject_pos = i
            break
    if inject_pos is None:
        return "[injection token not found]"

    input_ids = torch.tensor([tokens], device=device)
    embeddings = av_model.get_input_embeddings()(input_ids).clone()
    embeddings[0, inject_pos, :] = normalize_activation(
        activation.to(embeddings.dtype), INJECTION_SCALE)

    with torch.no_grad():
        output = av_model.generate(
            inputs_embeds=embeddings,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id)

    seq = output[0]
    if seq.shape[0] > len(tokens):
        gen_ids = seq[len(tokens):]
    else:
        gen_ids = seq

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if "</explanation>" in text:
        text = text.split("</explanation>")[0]
    return text.strip()


def ar_confidence(ar_backbone, value_heads, ar_tokenizer, description,
                  actual_activation, layer_idx, device):
    prompt = AR_TEMPLATE.replace("{explanation}", description)
    tokens = ar_tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens], device=device)

    inner = ar_backbone.model if hasattr(ar_backbone, "model") else ar_backbone

    with torch.no_grad():
        outputs = inner(input_ids=input_ids, use_cache=False,
                       output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1]
        last_h = hidden[0, -1]
        reconstructed = value_heads[layer_idx](last_h.unsqueeze(0)).squeeze(0)

    cos = torch.nn.functional.cosine_similarity(
        reconstructed.float().cpu().unsqueeze(0),
        actual_activation.float().cpu().unsqueeze(0)).item()
    return cos


def confidence_bar(cos, width=20):
    filled = int(cos * width)
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    if cos >= 0.9:
        col = "green"
    elif cos >= 0.7:
        col = "yellow"
    else:
        col = "red"
    return c(bar, col)


def display_layer(layer_idx, depth_pct, description, confidence):
    dc = depth_color(depth_pct)
    pct_str = c(f"L{layer_idx:02d} ({depth_pct:3d}%)", dc)
    conf_str = confidence_bar(confidence)
    cos_str = c(f"{confidence:.3f}", "dim")

    desc_lines = description.split("\n")
    first_line = desc_lines[0][:100] if desc_lines else ""

    print(f"  {pct_str}  {conf_str} {cos_str}  {first_line}")
    for line in desc_lines[1:3]:
        line = line.strip()[:100]
        if line:
            print(f"  {'':>14}  {'':>{22}}  {c(line, 'dim')}")


def run_brain(av_model, tokenizer, injection_token_id,
              ar_backbone, value_heads, ar_tokenizer,
              prompt, device, layers=None, skip_ar=False):

    print(f"\n{c('═' * 70, 'bold')}")
    print(f"  {c('PROMPT:', 'bold')} {prompt}")
    print(c('═' * 70, 'bold'))

    # Generate normal output first
    print(f"\n  {c('Generating output...', 'dim')}", flush=True)
    reply = generate_output(av_model, tokenizer, prompt, device)
    print(f"\n  {c('OUTPUT:', 'bold')} {reply[:500]}")

    # Extract hidden states
    print(f"\n  {c('Extracting hidden states...', 'dim')}", flush=True)
    hidden_states, _ = extract_hidden_states(av_model, tokenizer, prompt, device)

    print(f"\n  {c('LAYER-BY-LAYER VIEW:', 'bold')}")
    print(f"  {'':>14}  {'AR confidence':>22}  Description")
    print(f"  {c('─' * 66, 'dim')}")

    if layers is None:
        layers = list(range(N_LAYERS))

    for layer_idx in layers:
        if layer_idx >= len(hidden_states) - 1:
            continue

        depth_pct = layer_to_depth_pct(layer_idx, N_LAYERS)
        activation = hidden_states[layer_idx + 1].squeeze(0)

        description = verbalize_layer(
            av_model, tokenizer, activation, depth_pct,
            injection_token_id, device)

        if skip_ar:
            confidence = 0.0
        else:
            confidence = ar_confidence(
                ar_backbone, value_heads, ar_tokenizer,
                description, activation, layer_idx, device)

        display_layer(layer_idx, depth_pct, description, confidence)

    print(f"  {c('─' * 66, 'dim')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Brain in a Jar — NLA terminal viewer")
    parser.add_argument("--av-adapter", required=True, help="Path to universal AV adapter")
    parser.add_argument("--ar-checkpoint", required=True, help="Path to universal AR checkpoint")
    parser.add_argument("--prompt", default=None, help="Single prompt (interactive if omitted)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all 32)")
    parser.add_argument("--skip-ar", action="store_true",
                        help="Skip AR confidence (faster, AV only)")
    parser.add_argument("--every", type=int, default=1,
                        help="Show every Nth layer (default: 1 = all)")
    args = parser.parse_args()

    device = "cpu"

    av_model, tokenizer, injection_token_id = load_av_model(args.av_adapter, device)

    if not args.skip_ar:
        print("Loading AR backbone...", end=" ", flush=True)
        ar_backbone = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16,
            trust_remote_code=False)
        inner = ar_backbone.model if hasattr(ar_backbone, "model") else ar_backbone
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        ar_backbone.lm_head = torch.nn.Identity()
        for p in ar_backbone.parameters():
            p.requires_grad = False
        ar_backbone = ar_backbone.to(device).eval()
        ar_tokenizer = tokenizer
        print("done")

        value_heads = load_ar_heads(args.ar_checkpoint, device)
    else:
        ar_backbone = None
        value_heads = None
        ar_tokenizer = None

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(0, N_LAYERS, args.every))

    if args.prompt:
        run_brain(av_model, tokenizer, injection_token_id,
                  ar_backbone, value_heads, ar_tokenizer,
                  args.prompt, device, layers, args.skip_ar)
    else:
        print(f"\n{c('Brain in a Jar', 'bold')} — Phi-4 Mini NLA viewer")
        print(f"Type a prompt and watch the model think. {c('Ctrl+C to exit.', 'dim')}\n")
        while True:
            try:
                prompt = input(c("prompt> ", "cyan"))
                if not prompt.strip():
                    continue
                run_brain(av_model, tokenizer, injection_token_id,
                          ar_backbone, value_heads, ar_tokenizer,
                          prompt, device, layers, args.skip_ar)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{c('Goodbye.', 'dim')}")
                break


if __name__ == "__main__":
    main()
