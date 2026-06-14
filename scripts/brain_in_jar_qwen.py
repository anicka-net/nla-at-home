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
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INJECTION_CHAR = "㈎"
INJECTION_SCALE = 150.0
LAYER = 20
N_LAYERS = 28

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

def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


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


def load_models(av_path, ar_path, device, skip_ar=False):
    print("Loading Qwen 2.5 7B...", end=" ", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
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

    ar_backbone = None
    ar_head = None
    if not skip_ar:
        print("Loading AR backbone...", end=" ", flush=True)
        ar_backbone = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
        inner = ar_backbone.model if hasattr(ar_backbone, "model") else ar_backbone
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        ar_backbone.lm_head = torch.nn.Identity()
        for p in ar_backbone.parameters():
            p.requires_grad = False
        ar_backbone = ar_backbone.to(device).eval()
        print("done")

        print("Loading AR head...", end=" ", flush=True)
        ar_dir = Path(ar_path)
        for fname in ["ar_head.safetensors", "ar_head.pt"]:
            fpath = ar_dir / fname
            if fpath.exists():
                if fname.endswith(".safetensors"):
                    with safe_open(str(fpath), framework="pt") as f:
                        w = f.get_tensor(list(f.keys())[0])
                else:
                    w = torch.load(str(fpath), weights_only=True)
                ar_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
                ar_head.weight = torch.nn.Parameter(w)
                ar_head = ar_head.to(device).eval()
                break
        if ar_head is None:
            print("not found, skipping AR")
        else:
            print("done")

    return av_model, tokenizer, injection_token_id, ar_backbone, ar_head


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

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    handle.remove()

    reply = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return activation["h"].squeeze(0), reply


def verbalize(av_model, tokenizer, activation, depth_pct,
              injection_token_id, device, max_tokens=120):
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
    gen_ids = seq[len(tokens):] if seq.shape[0] > len(tokens) else seq
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if "</explanation>" in text:
        text = text.split("</explanation>")[0]
    return text.strip()


def ar_score(ar_backbone, ar_head, tokenizer, description,
             actual_activation, device):
    prompt = AR_TEMPLATE.replace("{explanation}", description)
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens], device=device)

    inner = ar_backbone.model if hasattr(ar_backbone, "model") else ar_backbone
    with torch.no_grad():
        outputs = inner(input_ids=input_ids, use_cache=False,
                       output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER + 1]
        last_h = hidden[0, -1]
        reconstructed = ar_head(last_h.unsqueeze(0)).squeeze(0)

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
        ar_backbone, ar_head, prompt, device, skip_ar):

    depth_pct = round(100 * (LAYER + 0.5) / N_LAYERS)
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
        injection_token_id, device)
    print("done")

    if not skip_ar and ar_backbone is not None and ar_head is not None:
        print(f"  {c('Computing AR confidence...', 'dim')}", end=" ", flush=True)
        cos = ar_score(ar_backbone, ar_head, tokenizer, description,
                       activation, device)
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
    args = parser.parse_args()

    device = "cpu"

    av_model, tokenizer, injection_token_id, ar_backbone, ar_head = load_models(
        args.av_adapter, args.ar_checkpoint, device, args.skip_ar)

    if args.prompt:
        run(av_model, tokenizer, injection_token_id,
            ar_backbone, ar_head, args.prompt, device, args.skip_ar)
    else:
        print(f"\n{c('Brain in a Jar', 'bold')} — Qwen 7B L20 NLA (84% top-1, GRPO)")
        print(f"Type a prompt. {c('Ctrl+C to exit.', 'dim')}\n")
        while True:
            try:
                prompt = input(c("prompt> ", "cyan"))
                if not prompt.strip():
                    continue
                run(av_model, tokenizer, injection_token_id,
                    ar_backbone, ar_head, prompt, device, args.skip_ar)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{c('Goodbye.', 'dim')}")
                break


if __name__ == "__main__":
    main()
