#!/usr/bin/env python3
"""Front half of split extraction: embedding + layers 0..split-1.

Runs on pondermatic. Loads the first portion of DeepSeek V4 Flash,
runs forward passes on the corpus, and sends intermediate hidden states
to the back half over TCP.

Start the back half (extract_back.py) FIRST, then run this.
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from wire import send_tensor, send_done


def load_front_half(model_name, split_layer, device, dtype):
    """Load embedding + first split_layer transformer layers."""
    print(f"Loading config for {model_name}...", flush=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    print(f"  {n_layers} layers total, loading 0..{split_layer - 1}", flush=True)

    # Use device_map to load only the layers we need
    device_map = {}
    device_map["model.embed_tokens"] = device
    device_map["model.norm"] = "meta"  # don't load final norm
    device_map["lm_head"] = "meta"     # don't load output head
    for i in range(n_layers):
        if i < split_layer:
            device_map[f"model.layers.{i}"] = device
        else:
            device_map[f"model.layers.{i}"] = "meta"

    print(f"Loading model weights (front half)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)
    return model, config


def forward_front(model, input_ids, split_layer):
    """Run forward through embedding + first split_layer layers.
    Returns hidden_states at the split point."""
    with torch.no_grad():
        # Get embeddings
        hidden_states = model.model.embed_tokens(input_ids)

        # Run through layers 0..split_layer-1
        for i in range(split_layer):
            layer = model.model.layers[i]
            # DeepSeek layers expect (hidden_states, attention_mask, position_ids, ...)
            # Simplified: most layers accept hidden_states as first positional arg
            layer_out = layer(hidden_states)
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
            else:
                hidden_states = layer_out

    return hidden_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--texts", required=True, help="JSONL corpus file")
    ap.add_argument("--split-layer", type=int, default=22)
    ap.add_argument("--backend-host", required=True,
                    help="IP of the back-half box (deepthought link-local)")
    ap.add_argument("--backend-port", type=int, default=29500)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    # Load corpus
    texts = []
    ids = []
    with open(args.texts) as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            ids.append(rec.get("id", rec.get("idx", len(ids))))
    print(f"Loaded {len(texts)} texts", flush=True)

    # Load tokenizer and front half of model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model, config = load_front_half(
        args.model, args.split_layer, device, dtype)

    # Connect to back half
    print(f"Connecting to back half at {args.backend_host}:{args.backend_port}...",
          flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.backend_host, args.backend_port))
    print("  Connected!", flush=True)

    # Send metadata
    meta = json.dumps({
        "n_texts": len(texts),
        "n_layers": config.num_hidden_layers,
        "d_model": config.hidden_size,
        "split_layer": args.split_layer,
        "ids": ids,
    }).encode()
    sock.sendall(struct.pack("!Q", len(meta)))
    sock.sendall(meta)

    import struct

    # Process each text
    t0 = time.time()
    for i, text in enumerate(texts):
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=args.max_seq_len).to(device)

        hidden = forward_front(model, tokens["input_ids"], args.split_layer)

        # Send the last-token hidden state at the split point
        last_hidden = hidden[:, -1, :].cpu()  # [1, d_model]
        send_tensor(sock, last_hidden)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            print(f"  [{i+1}/{len(texts)}] {rate:.1f} texts/s, "
                  f"ETA {eta/60:.0f}min", flush=True)

    send_done(sock)
    sock.close()
    print(f"\nFront half done: {len(texts)} texts in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
