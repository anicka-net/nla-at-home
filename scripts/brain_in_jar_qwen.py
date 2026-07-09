#!/usr/bin/env python3
"""
Brain in a Jar (Qwen 7B) — single-layer NLA at L20.

This is our best model: 84% top-1 accuracy, GRPO-trained, published on HF.
Shows what layer 20 (62.5% depth) is "thinking about" for any prompt.

Usage:
  python3 brain_in_jar_qwen.py --av-adapter ./av --ar-checkpoint ./ar
  python3 brain_in_jar_qwen.py --av-adapter ./av --ar-checkpoint ./ar --prompt "Do you have feelings?"
"""
import torch
import argparse
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from nla_lib import (
    get_model, normalize_activation, nearest_depth_pct,
)
from generation_utils import decode_generated

_SPEC = get_model("qwen25-7b")
BASE_MODEL = _SPEC.hf_id
INJECTION_CHAR = _SPEC.injection_char
LAYER = 20
N_LAYERS = 28  # documentation for the published L20 adapter (depth 71%)

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
}


def resolve_av_interface(meta, injection_char, depth_pct):
    template = meta["prompt_templates"]["av"]
    prompt = template.replace(
        "{injection_char}", injection_char).replace(
        "{depth_pct}", str(depth_pct))
    mode = meta.get("training", {}).get("injection_mode")
    if mode is None:
        mode = "normalize" if "{depth_pct}" in template else "multiply"
    return prompt, mode


def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def load_models(av_path, ar_path, device, skip_ar=False):
    print("Loading Qwen 2.5 7B...", end=" ", flush=True)
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=_SPEC.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=_SPEC.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("done")

    print("Loading AV adapter (GRPO-trained)...", end=" ", flush=True)
    av_model = PeftModel.from_pretrained(base, av_path)
    av_model = av_model.to(device).eval()
    print("done")

    injection_id = tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)
    if len(injection_id) != 1:
        raise ValueError(f"Injection char encodes to {len(injection_id)} tokens, need 1")
    injection_token_id = injection_id[0]

    av_meta = yaml.safe_load((Path(av_path) / "nla_meta.yaml").read_text())
    av_template = av_meta["prompt_templates"]["av"]
    _, injection_mode = resolve_av_interface(
        av_meta, INJECTION_CHAR, nearest_depth_pct(LAYER, N_LAYERS))
    use_chat_template = bool(
        av_meta.get("training", {}).get("chat_template", False))

    ar_model = None
    ar_template = None
    ar_tokenizer = None
    if not skip_ar:
        print("Loading AR adapter...", end=" ", flush=True)
        ar_meta = yaml.safe_load((Path(ar_path) / "nla_meta.yaml").read_text())
        ar_template = ar_meta["prompt_templates"]["ar"]
        ar_tokenizer = AutoTokenizer.from_pretrained(ar_path)
        if ar_tokenizer.pad_token is None:
            ar_tokenizer.pad_token = ar_tokenizer.eos_token
        ar_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype,
            trust_remote_code=_SPEC.trust_remote_code)
        ar_model = PeftModel.from_pretrained(ar_base, ar_path).to(device).eval()
        for p in ar_model.parameters():
            p.requires_grad = False
        print("done")

    return (av_model, tokenizer, injection_token_id, av_template,
            injection_mode, use_chat_template, ar_model, ar_tokenizer,
            ar_template)


def extract_layer_activation(model, tokenizer, prompt, layer, device):
    messages = [{"role": "user", "content": prompt}]
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_str, return_tensors="pt").to(device)

    activation = {}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if "h" not in activation:
            activation["h"] = h[:, -1, :].detach()
    base = model.base_model.model if hasattr(model, "base_model") else model
    inner = base.model if hasattr(base, "model") else base
    handle = inner.layers[layer].register_forward_hook(hook)

    with model.disable_adapter(), torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    handle.remove()

    reply = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return activation["h"].squeeze(0), reply


def verbalize(av_model, tokenizer, activation, depth_pct,
              injection_token_id, device, prompt_template, injection_mode,
              use_chat_template, max_tokens=120):
    prompt_text, _ = resolve_av_interface(
        {"prompt_templates": {"av": prompt_template},
         "training": {"injection_mode": injection_mode}},
        INJECTION_CHAR, depth_pct)
    if use_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    inject_pos = None
    for i, tid in enumerate(tokens):
        if tid == injection_token_id:
            inject_pos = i
            break
    if inject_pos is None:
        return "[injection token not found]"

    input_ids = torch.tensor([tokens], device=device)
    embeddings = av_model.get_input_embeddings()(input_ids).clone()
    injected = (
        normalize_activation(activation.to(embeddings.dtype))
        if injection_mode == "normalize"
        else activation.to(embeddings.dtype) * 150.0
    )
    embeddings[0, inject_pos, :] = injected

    with torch.no_grad():
        output = av_model.generate(
            inputs_embeds=embeddings,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id)

    return decode_generated(output, tokens, tokenizer)


def ar_score(ar_model, tokenizer, prompt_template, description,
             actual_activation, depth_pct, device):
    prompt = prompt_template.replace(
        "{explanation}", description).replace(
        "{injection_char}", INJECTION_CHAR).replace(
        "{depth_pct}", str(depth_pct))
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        outputs = ar_model(
            input_ids=input_ids, use_cache=False, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER + 1]
        reconstructed = hidden[0, -1]

    cos = torch.nn.functional.cosine_similarity(
        reconstructed.float().cpu().unsqueeze(0),
        actual_activation.float().cpu().unsqueeze(0)).item()
    return cos


def confidence_bar(cos, width=20):
    filled = max(0, min(width, int(cos * width)))
    bar = "█" * filled + "░" * (width - filled)
    col = "green" if cos >= 0.9 else "yellow" if cos >= 0.7 else "red"
    return c(bar, col)


def run(av_model, tokenizer, injection_token_id,
        av_template, injection_mode, use_chat_template, ar_model,
        ar_tokenizer, ar_template, prompt, device, skip_ar):

    depth_pct = nearest_depth_pct(LAYER, N_LAYERS)
    sep = c("=" * 70, "bold")
    dim_sep = c("-" * 70, "dim")

    print(f"\n{sep}")
    print(f"  {c('PROMPT:', 'bold')} {prompt}")
    print(sep)

    print(f"\n  {c('Running model...', 'dim')}", end=" ", flush=True)
    activation, reply = extract_layer_activation(
        av_model, tokenizer, prompt, LAYER, device)
    print("done")

    print(f"\n  {c('OUTPUT:', 'bold')} {reply[:500]}")

    print(f"\n  {c('Verbalizing layer %d (%d%% depth)...' % (LAYER, depth_pct), 'dim')}",
          end=" ", flush=True)
    description = verbalize(
        av_model, tokenizer, activation, depth_pct,
        injection_token_id, device, av_template, injection_mode,
        use_chat_template)
    print("done")

    if not skip_ar and ar_model is not None:
        print(f"  {c('Computing AR confidence...', 'dim')}", end=" ", flush=True)
        cos = ar_score(
            ar_model, ar_tokenizer, ar_template, description,
            activation, depth_pct, device)
        print("done")
        conf_str = f" {confidence_bar(cos)} {c('%.3f' % cos, 'dim')}"
    else:
        conf_str = ""

    print(f"\n  {c('LAYER %d (%d%% depth):%s' % (LAYER, depth_pct, conf_str), 'cyan')}")
    print(f"  {dim_sep}")
    for line in description.split("\n"):
        line = line.strip()
        if line:
            print(f"  {line}")
    print(f"  {dim_sep}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Brain in a Jar — Qwen 7B single-layer NLA")
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    loaded = load_models(
        args.av_adapter, args.ar_checkpoint, device, args.skip_ar)

    if args.prompt:
        run(*loaded, args.prompt, device, args.skip_ar)
    else:
        print(f"\n{c('Brain in a Jar', 'bold')} — Qwen 7B L20 NLA (84% top-1, GRPO)")
        print(f"Type a prompt. {c('Ctrl+C to exit.', 'dim')}\n")
        while True:
            try:
                prompt = input(c("prompt> ", "cyan"))
                if not prompt.strip():
                    continue
                run(*loaded, prompt, device, args.skip_ar)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{c('Goodbye.', 'dim')}")
                break


if __name__ == "__main__":
    main()
