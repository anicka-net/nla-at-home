#!/usr/bin/env python3
"""Back half of split extraction: layers split..N-1 + merge + save.

Runs on deepthought. Receives (full-sequence split-point hidden states +
front-layer captures) per text from the front half, continues the forward
pass over the REAL sequence, captures back-layer last-token residuals,
merges both halves and writes the standard NLA activation file.

Start this FIRST, then run extract_front.py on the ds4 host.

Review fixes 2026-07-06: full-sequence forward (was: seq_len=1 — the
captured states were not the model's real states), front-layer captures
merged (was: silently dropped), --quant 4bit, robust metadata recv.
"""

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from wire import recv_msg, _recv_exact
from forward_utils import (make_layer_caller, prepare_positions,
                           capture_from_h)


def load_back_half(model_name, split_layer, device, dtype, quant):
    print(f"Loading config for {model_name}...", flush=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    print(f"  {n_layers} layers total, loading {split_layer}..{n_layers-1}",
          flush=True)

    device_map = {"model.embed_tokens": "meta",
                  "model.norm": "meta", "lm_head": "meta"}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = device if i >= split_layer else "meta"

    kwargs = dict(device_map=device_map, torch_dtype=dtype,
                  trust_remote_code=True)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype)

    print(f"Loading model weights (back half, quant={quant})...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)
    return model, config


def forward_back(model, hidden_states, split_layer, n_layers, call_layer,
                 input_ids=None):
    """Layers split..N-1 over the full sequence. Returns last-token
    residual per back layer: [n_back, d_model] (or [n_back, hc_mult, d_model]
    for HC models)."""
    captures = []
    with torch.no_grad():
        h = hidden_states
        pos = prepare_positions(model, h)
        for i in range(split_layer, n_layers):
            h = call_layer(model.model.layers[i], h, pos,
                           input_ids=input_ids)
            captures.append(capture_from_h(h))
    return torch.stack(captures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--split-layer", type=int, default=22)
    ap.add_argument("--listen-port", type=int, default=29500)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="Write a partial .pt every N texts (crash safety)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, config = load_back_half(
        args.model, args.split_layer, args.device, dtype, args.quant)
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    call_layer = make_layer_caller(model)

    print(f"Listening on port {args.listen_port}...", flush=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.listen_port))
    server.listen(1)
    conn, addr = server.accept()
    print(f"  Front half connected from {addr}", flush=True)

    meta_len = struct.unpack("!Q", _recv_exact(conn, 8))[0]
    meta = json.loads(_recv_exact(conn, meta_len).decode())
    n_texts = meta["n_texts"]
    ids = meta["ids"]
    assert meta["split_layer"] == args.split_layer, (
        f"split mismatch: front {meta['split_layer']} vs back {args.split_layer}")
    assert meta["n_layers"] == n_layers and meta["d_model"] == d_model
    print(f"  Expecting {n_texts} texts, d_model={d_model}, "
          f"n_layers={n_layers}, chat_template={meta.get('chat_template')}",
          flush=True)

    # per-text rows; merged at the end into {layer: [N, d]} covering 0..N-1
    front_rows, back_rows = [], []

    def save(path_suffix=""):
        n_done = len(back_rows)
        if n_done == 0:
            return
        front = torch.stack(front_rows)   # [N, split, d]
        back = torch.stack(back_rows)     # [N, n_back, d]
        activations = {}
        for L in range(args.split_layer):
            activations[L] = front[:, L, :]
        for j, L in enumerate(range(args.split_layer, n_layers)):
            activations[L] = back[:, j, :]
        out = {
            "activations": activations,
            "ids": ids[:n_done],
            "n_layers": n_layers,
            "n_texts": n_done,
            "d_model": d_model,
            "model": meta.get("model", args.model),
            "chat_template": meta.get("chat_template"),
            "split_layer": args.split_layer,
            "extraction_method": "split_pipeline_2box",
        }
        path = Path(args.output + path_suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        return path

    t0 = time.time()
    while len(back_rows) < n_texts:
        msg = recv_msg(conn)
        if msg is None:
            print("  front signaled done early", flush=True)
            break
        h = msg["h_split"].to(args.device, dtype)
        front_rows.append(msg["front_caps"])
        back_rows.append(forward_back(
            model, h, args.split_layer, n_layers, call_layer,
            input_ids=msg.get("input_ids")))

        i = len(back_rows)
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  [{i}/{n_texts}] {rate:.2f} texts/s, "
                  f"ETA {(n_texts - i)/rate/60:.0f}min", flush=True)
        if i % args.checkpoint_every == 0:
            p = save(".partial")
            print(f"  checkpoint: {p} ({i} texts)", flush=True)

    conn.close()
    server.close()
    path = save()
    print(f"\nDone: {len(back_rows)} texts, {n_layers} layers "
          f"in {time.time()-t0:.0f}s → {path}", flush=True)


if __name__ == "__main__":
    main()
