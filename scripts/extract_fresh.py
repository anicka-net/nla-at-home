#!/usr/bin/env python3
"""Extract phi4 activations at L16 + L25 for the fresh WildChat eval prompts.

Mirrors extract_activations.extract() EXACTLY (chat template,
add_generation_prompt=True, last-token position seq_len = attn.sum()-1,
hook on decoder block output, stored float32) so the vectors are identical
in convention to phi4_all_layers.pt / phi4_13depths.pt (verified by parity
test, cos 1.000003). Captures both layers in a single forward per text and
pins the model to GPU (device_map={"":0}) to avoid CPU offload.

Output: ~/phi4_ar/phi4_fresh300.pt
  {"ids":[...], "activations": {16: N×d, 25: N×d}, "n_layers":40,
   "d_model": d, "model": ..., "n_texts": N}
"""
import argparse
import json
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
import extract_activations as EA

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO / "corpus/generated/wildchat_fresh300.json"))
    ap.add_argument("--layers", default="16,25")
    ap.add_argument("--out", default=str(Path.home() / "phi4_ar/phi4_fresh300.pt"))
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    items = json.load(open(args.input))
    print(f"extracting L{layers} for {len(items)} texts", flush=True)

    name = EA.MODELS["phi4"]
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=False).eval()
    blocks = EA.get_blocks(model)
    n_layers = len(blocks)

    per_layer = {L: [] for L in layers}
    ids = []
    for i, item in enumerate(items):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": item["text"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        seq_len = inputs["attention_mask"].sum() - 1
        cap = {}

        def make_hook(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[L] = h[0, seq_len].detach().cpu().float()
            return hook

        handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        for L in layers:
            per_layer[L].append(cap[L])
        ids.append(item["id"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    acts = {L: torch.stack(per_layer[L]) for L in layers}
    out = {"ids": ids, "activations": acts, "n_layers": n_layers,
           "d_model": acts[layers[0]].shape[1], "model": name,
           "n_texts": len(ids)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {len(ids)} texts × L{layers} ({out['d_model']}d) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
