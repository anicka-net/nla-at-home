#!/usr/bin/env python3
"""Front half of split extraction: embedding + layers 0..split-1.

Runs on the ds4 host. Loads the first portion of the source model, runs
forward passes on the corpus, captures the last-token residual at each
front layer, and streams (full-sequence split-point hidden states +
front-layer captures) to the back half over TCP.

Start the back half (extract_back.py) FIRST, then run this.

Review fixes 2026-07-06 (see lineage note "franken extraction review"):
- struct imported at top (was: used before a late local import -> crash)
- sends the FULL sequence hidden states, not just the last token — the
  back half runs attention over the real context (~4 MB/text @ seq 512,
  trivial on the fast link)
- captures front-half layers here and ships them along (was: silently
  missing from the output)
- --quant 4bit: bf16 halves are ~280 GB each and cannot fit one box
- chat-template wrapping on by default (NLA convention: state at the
  last token after the generation prompt)
- defensive layer calling with position machinery (see forward_utils)
"""

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from wire import send_tensor, send_done
from forward_utils import make_layer_caller, prepare_positions


def load_front_half(model_name, split_layer, device, dtype, quant):
    print(f"Loading config for {model_name}...", flush=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    print(f"  {n_layers} layers total, loading 0..{split_layer - 1}", flush=True)

    device_map = {"model.embed_tokens": device,
                  "model.norm": "meta", "lm_head": "meta"}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = device if i < split_layer else "meta"

    kwargs = dict(device_map=device_map, torch_dtype=dtype,
                  trust_remote_code=True)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype)

    print(f"Loading model weights (front half, quant={quant})...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)
    return model, config


def forward_front(model, input_ids, split_layer, call_layer):
    """Embedding + layers 0..split-1. Returns (full-seq hidden at split,
    front-layer last-token captures [split_layer, d_model])."""
    captures = []
    with torch.no_grad():
        h = model.model.embed_tokens(input_ids)
        pos = prepare_positions(model, h)
        for i in range(split_layer):
            h = call_layer(model.model.layers[i], h, pos)
            captures.append(h[0, -1, :].float().cpu().clone())
    return h, torch.stack(captures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--texts", required=True, help="JSONL corpus file")
    ap.add_argument("--split-layer", type=int, default=22)
    ap.add_argument("--backend-host", required=True)
    ap.add_argument("--backend-port", type=int, default=29500)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"],
                    help="4bit is REQUIRED for ds4-class models on 128 GB")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Tokenize raw text (breaks NLA last-token "
                         "convention — only for ablations)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    texts, ids = [], []
    with open(args.texts) as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            ids.append(rec.get("id", rec.get("idx", len(ids))))
    print(f"Loaded {len(texts)} texts", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, config = load_front_half(
        args.model, args.split_layer, args.device, dtype, args.quant)
    call_layer = make_layer_caller(model)

    print(f"Connecting to back half at {args.backend_host}:{args.backend_port}...",
          flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.backend_host, args.backend_port))
    print("  Connected!", flush=True)

    meta = json.dumps({
        "n_texts": len(texts),
        "n_layers": config.num_hidden_layers,
        "d_model": config.hidden_size,
        "split_layer": args.split_layer,
        "ids": ids,
        "model": args.model,
        "chat_template": not args.no_chat_template,
    }).encode()
    sock.sendall(struct.pack("!Q", len(meta)))
    sock.sendall(meta)

    t0 = time.time()
    for i, text in enumerate(texts):
        if args.no_chat_template or tokenizer.chat_template is None:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=args.max_seq_len)
            input_ids = enc["input_ids"]
        else:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True)
            input_ids = torch.tensor([tokenizer.encode(
                chat, add_special_tokens=False)[:args.max_seq_len]])
        input_ids = input_ids.to(args.device)

        hidden, front_caps = forward_front(
            model, input_ids, args.split_layer, call_layer)

        # full sequence for the back half + this half's layer captures
        send_tensor(sock, {"h_split": hidden.to(torch.float16).cpu(),
                           "front_caps": front_caps})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            print(f"  [{i+1}/{len(texts)}] {rate:.2f} texts/s, "
                  f"ETA {eta/60:.0f}min", flush=True)

    send_done(sock)
    sock.close()
    print(f"\nFront half done: {len(texts)} texts in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
