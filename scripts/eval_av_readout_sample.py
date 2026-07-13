#!/usr/bin/env python3
"""Held-out readout sample for a universal AV — the eyeball eval.

Round-trip eval needs an AR for the same model; when none exists yet
(e.g. a fresh qwen3-4b AV), this is the honest substitute: inject
held-out activations, generate descriptions, print them NEXT TO the
source texts so a human can judge on-topic-ness. Numbers don't catch
template hallucination — reading the descriptions does.

Prompt construction matches train_universal_av.py byte-for-byte:
make_av_prompt -> apply_chat_template(add_generation_prompt=True)
-> encode(add_special_tokens=False), injection via embedding overwrite
of the placeholder token, activation normalized TO INJECTION_SCALE.

Usage (deepthought):
  ~/venv/bin/python scripts/eval_av_readout_sample.py \
    --model qwen3-4b \
    --adapter output/nla-qwen3-4b-universal-av \
    --activations corpus/activations/qwen3-4b_all_layers.pt \
    --val-ids output/nla-qwen3-4b-universal-av/val_text_ids.json \
    --n-per-layer 3 --output working-docs/qwen3_4b_readout_sample.json
"""
import argparse
import json
import random
from pathlib import Path

import torch

from nla_lib import (
    INJECTABLE_MODELS_HF,
    INJECTION_CHARS,
    INJECTION_SCALE,
    make_av_prompt,
    nearest_depth_pct,
    normalize_activation,
)
import sink_fix


def load_source_texts(corpus_dir):
    texts = {}
    for path in Path(corpus_dir).glob("*.json"):
        if "descriptions" in path.name:
            continue
        try:
            items = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and "id" in it and "text" in it:
                    texts[it["id"]] = it["text"]
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    choices=sorted(INJECTABLE_MODELS_HF))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--val-ids", required=True,
                    help="val_text_ids.json persisted by the training run")
    ap.add_argument("--corpus-dir", default="corpus/generated")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices; default = 4 spread")
    ap.add_argument("--n-per-layer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    act_data = torch.load(args.activations, weights_only=True)
    n_layers = act_data["n_layers"]
    ids = act_data["ids"]
    id_to_idx = {tid: i for i, tid in enumerate(ids)}

    val_ids = json.loads(Path(args.val_ids).read_text())
    texts = load_source_texts(args.corpus_dir)
    pool = [t for t in val_ids if t in id_to_idx and t in texts]
    print(f"val ids: {len(val_ids)}, usable (activation+source text): "
          f"{len(pool)}")

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = sorted({max(0, round(f * (n_layers - 1)))
                         for f in (0.10, 0.40, 0.71, 0.96)})
    print(f"layers: {layers} (of {n_layers})")

    hf_id = INJECTABLE_MODELS_HF[args.model]
    injection_char = INJECTION_CHARS[args.model]
    print(f"loading {hf_id} (bf16) + {args.adapter}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(hf_id)
    base = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    device = next(model.parameters()).device

    sink_params = sink_fix.load_for_adapter(args.adapter)
    if sink_params is not None:
        print(f"sink_fix sidecar active (drop_top_pc="
              f"{sink_params['drop_top_pc']}) — applied before normalize")

    inject_ids = tok.encode(injection_char, add_special_tokens=False)
    assert len(inject_ids) == 1, f"injection char -> {inject_ids}"
    inject_id = inject_ids[0]

    prompt_cache = {}
    for layer in layers:
        pct = nearest_depth_pct(layer, n_layers)
        if pct in prompt_cache:
            continue
        chat_str = tok.apply_chat_template(
            [{"role": "user", "content": make_av_prompt(pct, injection_char)}],
            tokenize=False, add_generation_prompt=True)
        tokens = tok.encode(chat_str, add_special_tokens=False)
        prompt_cache[pct] = (tokens, tokens.index(inject_id))

    rng = random.Random(args.seed)
    sample_ids = rng.sample(pool, min(args.n_per_layer * len(layers),
                                      len(pool)))

    results = []
    k = 0
    for layer in layers:
        pct = nearest_depth_pct(layer, n_layers)
        tokens, pos = prompt_cache[pct]
        for _ in range(args.n_per_layer):
            tid = sample_ids[k % len(sample_ids)]
            k += 1
            act = act_data["activations"][layer][id_to_idx[tid]]
            act = sink_fix.apply_if_present(sink_params, layer, act.float())
            emb = model.get_input_embeddings()(
                torch.tensor([tokens], device=device)).clone()
            emb[0, pos, :] = normalize_activation(
                act.to(device), INJECTION_SCALE).to(emb.dtype)
            with torch.no_grad():
                out = model.generate(
                    inputs_embeds=emb,
                    attention_mask=torch.ones(1, emb.shape[1],
                                              device=device,
                                              dtype=torch.long),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            seq = out[0]
            gen = seq[len(tokens):] if seq.shape[0] > len(tokens) else seq
            desc = (tok.decode(gen, skip_special_tokens=True)
                    .split("</explanation>")[0].strip())
            src = " ".join(texts[tid].split())[:220]
            print(f"\n=== layer {layer} (depth {pct}%)  {tid}")
            print(f"SOURCE : {src}")
            print(f"READOUT: {desc}")
            results.append({"text_id": tid, "layer": layer,
                            "depth_pct": pct, "source_snippet": src,
                            "readout": desc})

    if args.output:
        Path(args.output).write_text(
            json.dumps({"model": args.model, "adapter": args.adapter,
                        "seed": args.seed, "results": results},
                       indent=2, ensure_ascii=False))
        print(f"\nsaved -> {args.output}")
    print("READOUT_SAMPLE_DONE")


if __name__ == "__main__":
    main()
