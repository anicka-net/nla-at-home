#!/usr/bin/env python3
"""Interactive NLA: type a text, see what the AV says about its activations.

Loads ONE Phi-4 backbone serving two roles: extraction (adapter disabled —
identical to extract_activations.py protocol: chat template + generation
prompt, last-token hook) and description (AV LoRA adapter on, injection
prompt/scale identical to training via train_universal_grpo_hard helpers).

Usage:
  python3 scripts/describe_live.py \
      --av-adapter output/nla-phi4-universal-av-grpo-hard \
      [--layers 13,22,28,36] [--max-new-tokens 150]
Then type a text and Enter; empty line or Ctrl-D quits.
Compare adapters by pointing --av-adapter at ...-twinclean (SFT baseline).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_universal_grpo_hard import (  # noqa: E402
    INJECTION_CHARS, INJECTION_SCALE, MODELS, make_av_prompt,
    nearest_depth_pct, normalize_activation, strip_generated_row)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def get_blocks(model):
    base = model
    while not hasattr(base, "layers"):
        base = base.model
    return base.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-adapter", required=True)
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--layers", default="13,22,28,36")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    base_name = MODELS[args.model]
    inj_char = INJECTION_CHARS[args.model]

    print(f"[load] {base_name} + adapter {args.av_adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16).to(device).eval()
    model = PeftModel.from_pretrained(model, args.av_adapter).eval()
    n_layers = model.config.num_hidden_layers
    blocks = get_blocks(model)
    embed = model.get_input_embeddings()

    inject_id = tok.encode(inj_char, add_special_tokens=False)[0]
    eos_ids = {tok.eos_token_id, tok.pad_token_id}
    stop_ids = tok.encode("</explanation>", add_special_tokens=False)

    prompt_cache = {}
    for L in layers:
        pct = nearest_depth_pct(L, n_layers)
        if pct in prompt_cache:
            continue
        content = make_av_prompt(pct, inj_char)
        chat = tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True)
        tokens = tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (tokens, tokens.index(inject_id))

    def extract(text):
        chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        seq_len = inputs["attention_mask"].sum() - 1
        grabbed = {}

        def make_hook(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                grabbed[L] = h[0, seq_len].detach().float()
            return hook

        handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
        try:
            with torch.no_grad(), model.disable_adapter():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    def describe(act, L):
        pct = nearest_depth_pct(L, n_layers)
        tokens, inject_pos = prompt_cache[pct]
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        emb = embed(ids)
        emb[0, inject_pos, :] = normalize_activation(
            act.to(device), INJECTION_SCALE).to(emb.dtype)
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=emb.to(model.dtype),
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True)
        gen = strip_generated_row(out.sequences[0], tokens, eos_ids, stop_ids)
        return tok.decode(gen, skip_special_tokens=True).strip()

    print(f"[ready] layers {layers} — type a text, empty line quits", flush=True)
    while True:
        try:
            text = input("\ntext> ").strip()
        except EOFError:
            break
        if not text:
            break
        acts = extract(text)
        for L in layers:
            pct = nearest_depth_pct(L, n_layers)
            print(f"\n--- L{L} ({pct}%) ---")
            print(describe(acts[L], L), flush=True)


if __name__ == "__main__":
    main()
