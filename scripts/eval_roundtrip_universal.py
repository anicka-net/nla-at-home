#!/usr/bin/env python3
"""Round-trip eval for a universal AV against a lora_sl AR, on a clean
text-id holdout.

For each held-out text and each sampled layer L:
  AV (adapter under test) describes the injected activation
  -> lora_sl AR reconstructs from the description (own hidden state at L)
  -> centered cosine (per-layer mean subtracted) vs the true activation.

Compares an adapter under test (e.g. GRPO) against a baseline AV (e.g.
the SFT checkpoint) under the IDENTICAL protocol. No compass anywhere —
the oracle compass is fit on all ids and is curriculum-only.

Usage (deepthought):
  ~/venv/bin/python scripts/eval_roundtrip_universal.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-av-grpo \
    --av-baseline output/nla-qwen25-7b-universal-av \
    --ar-checkpoint output/nla-qwen25-7b-universal-ar \
    --activations corpus/activations/qwen25-7b_all_layers.pt \
    --holdout output/nla-qwen25-7b-av-grpo/eval_holdout_ids.json \
    --layers 4,10,17,20,24,27 \
    --output output/nla-qwen25-7b-av-grpo/roundtrip_eval.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from nla_lib import (
    INJECTABLE_MODELS_HF as MODELS, INJECTION_CHARS, get_model,
    normalize_activation, make_av_prompt, nearest_depth_pct,
    detect_ar_format, load_ar_lora_sl, AR_FORMAT_LORA_SL,
)
import sink_fix


def build_prompt_cache(tokenizer, injection_char, layers, n_layers):
    inject_ids = tokenizer.encode(injection_char, add_special_tokens=False)
    assert len(inject_ids) == 1
    inject_id = inject_ids[0]
    cache = {}
    for L in layers:
        pct = nearest_depth_pct(L, n_layers)
        content = make_av_prompt(pct, injection_char)
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(chat_str, add_special_tokens=False)
        inject_pos = next(i for i, t in enumerate(tokens) if t == inject_id)
        cache[L] = (tokens, inject_pos)
    return cache


def clean_generated(text):
    for stop in ("</explanation>", "<explanation>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


@torch.no_grad()
def describe_batch(model, tokenizer, prompt_tokens, inject_pos, acts, device,
                   max_new_tokens):
    """Greedy-generate descriptions for a batch of activations injected
    into the same prompt."""
    B = acts.shape[0]
    input_ids = torch.tensor([prompt_tokens] * B, device=device)
    embeds = model.get_input_embeddings()(input_ids).clone()
    embeds[:, inject_pos, :] = normalize_activation(
        acts.to(device)).to(embeds.dtype)
    out = model.generate(
        inputs_embeds=embeds,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    seqs = out.tolist()
    pl = len(prompt_tokens)
    descs = []
    for seq in seqs:
        if len(seq) > pl and seq[:pl] == prompt_tokens:
            seq = seq[pl:]
        descs.append(clean_generated(
            tokenizer.decode(seq, skip_special_tokens=True)))
    return descs


def run_av(adapter_path, tag, base_model_name, tokenizer, prompt_cache,
           layers, hold_acts, device, trust_remote, batch, max_new_tokens):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    print(f"\n=== AV [{tag}]: {adapter_path}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote).to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    sink_params = sink_fix.load_for_adapter(adapter_path)

    descs = {}
    for L in layers:
        prompt_tokens, inject_pos = prompt_cache[L]
        acts = sink_fix.apply_if_present(sink_params, L, hold_acts[L])
        layer_descs = []
        t0 = time.time()
        for s in range(0, acts.shape[0], batch):
            layer_descs.extend(describe_batch(
                model, tokenizer, prompt_tokens, inject_pos,
                acts[s:s + batch], device, max_new_tokens))
            if s // batch % 5 == 0:
                print(f"  [{tag}] L{L}: {s + batch}/{acts.shape[0]} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        descs[L] = layer_descs
        print(f"  [{tag}] L{L} done: {len(layer_descs)} descs "
              f"({time.time() - t0:.0f}s)", flush=True)

    del model, base
    torch.cuda.empty_cache()
    return descs


def centered_cos(pred, target, mean):
    p = pred.float() - mean
    t = target.float() - mean
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (p * t).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--av-adapter", required=True,
                    help="AV under test (e.g. the GRPO checkpoint)")
    ap.add_argument("--av-baseline", default=None,
                    help="Baseline AV under the identical protocol")
    ap.add_argument("--ar-checkpoint", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--holdout", required=True,
                    help="JSON list of held-out text ids")
    ap.add_argument("--layers", default="4,10,17,20,24,27")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ar-batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--max-texts", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda"
    spec = get_model(args.model)
    base_model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    layers = [int(x) for x in args.layers.split(",")]

    assert detect_ar_format(args.ar_checkpoint) == AR_FORMAT_LORA_SL, \
        "this eval is for lora_sl ARs"

    print(f"Loading activations {args.activations}...", flush=True)
    act_data = torch.load(args.activations, weights_only=True,
                          map_location="cpu")
    n_layers = act_data["n_layers"]
    ids = act_data["ids"]
    id2idx = {tid: i for i, tid in enumerate(ids)}

    holdout = json.load(open(args.holdout))
    if args.max_texts:
        holdout = holdout[:args.max_texts]
    idxs = [id2idx[t] for t in holdout]
    print(f"{len(holdout)} holdout texts, layers {layers}", flush=True)

    hold_acts = {L: act_data["activations"][L][idxs].clone() for L in layers}
    layer_means = {L: act_data["activations"][L].float().mean(dim=0)
                   for L in layers}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=spec.trust_remote_code)
    prompt_cache = build_prompt_cache(tokenizer, injection_char, layers,
                                      n_layers)

    runs = {"test": args.av_adapter}
    if args.av_baseline:
        runs["baseline"] = args.av_baseline
    all_descs = {tag: run_av(path, tag, base_model_name, tokenizer,
                             prompt_cache, layers, hold_acts, device,
                             spec.trust_remote_code, args.batch,
                             args.max_new_tokens)
                 for tag, path in runs.items()}

    print("\n=== AR reconstruction ===", flush=True)
    ar = load_ar_lora_sl(args.ar_checkpoint, base_model_name, device,
                         spec.trust_remote_code, n_layers, injection_char)

    results = {tag: {} for tag in all_descs}
    records = []
    for tag, descs_by_layer in all_descs.items():
        for L in layers:
            descs = descs_by_layer[L]
            cos_all = []
            for s in range(0, len(descs), args.ar_batch):
                chunk = descs[s:s + args.ar_batch]
                recon = ar.reconstruct(chunk, [L], device)[L].cpu()
                cos = centered_cos(recon, hold_acts[L][s:s + args.ar_batch],
                                   layer_means[L])
                cos_all.extend(cos.tolist())
            results[tag][L] = sum(cos_all) / len(cos_all)
            print(f"  [{tag}] L{L}: centered_cos = {results[tag][L]:.4f}",
                  flush=True)
            for tid, d, c in zip(holdout, descs, cos_all):
                records.append({"tag": tag, "layer": L, "text_id": tid,
                                "cos": round(c, 4), "desc": d})

    summary = {
        "model": args.model,
        "av_adapter": args.av_adapter,
        "av_baseline": args.av_baseline,
        "ar_checkpoint": args.ar_checkpoint,
        "holdout": args.holdout,
        "n_texts": len(holdout),
        "layers": layers,
        "per_layer": {tag: {str(L): round(v, 4) for L, v in r.items()}
                      for tag, r in results.items()},
        "mean": {tag: round(sum(r.values()) / len(r), 4)
                 for tag, r in results.items()},
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    recs_path = Path(args.output).with_suffix(".records.jsonl")
    with open(recs_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print("\n" + json.dumps(summary["mean"], indent=2), flush=True)
    print(f"saved -> {args.output} (+ {recs_path.name})", flush=True)


if __name__ == "__main__":
    main()
