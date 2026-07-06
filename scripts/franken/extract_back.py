#!/usr/bin/env python3
"""Back half of split extraction: layers split..N-1 + activation capture.

Runs on deepthought. Loads the latter portion of DeepSeek V4 Flash,
receives intermediate hidden states from the front half over TCP,
continues forward, and captures residual-stream activations at every layer.

Start this FIRST, then run extract_front.py on pondermatic.
"""

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from wire import recv_tensor, recv_is_done


def load_back_half(model_name, split_layer, device, dtype):
    """Load layers split_layer..N-1 + final norm (no lm_head needed)."""
    print(f"Loading config for {model_name}...", flush=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    print(f"  {n_layers} layers total, loading {split_layer}..{n_layers-1}",
          flush=True)

    device_map = {}
    device_map["model.embed_tokens"] = "meta"  # don't need embeddings
    device_map["model.norm"] = device
    device_map["lm_head"] = "meta"
    for i in range(n_layers):
        if i >= split_layer:
            device_map[f"model.layers.{i}"] = device
        else:
            device_map[f"model.layers.{i}"] = "meta"

    print(f"Loading model weights (back half)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)
    return model, config


def forward_back_with_capture(model, hidden_states, split_layer, n_layers,
                              capture_layers, device):
    """Run forward through layers split_layer..N-1, capturing activations.

    Returns dict {layer_idx: last_token_hidden [d_model]} for each
    captured layer.
    """
    captured = {}
    h = hidden_states.to(device)

    with torch.no_grad():
        for i in range(split_layer, n_layers):
            layer = model.model.layers[i]
            layer_out = layer(h)
            if isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            if i in capture_layers:
                # Capture last-token residual stream (post-layer)
                captured[i] = h[:, -1, :].cpu().clone()

    return captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--split-layer", type=int, default=22)
    ap.add_argument("--listen-port", type=int, default=29500)
    ap.add_argument("--output", required=True,
                    help="Output .pt file for captured activations")
    ap.add_argument("--capture-layers", default="all",
                    help="Comma-separated layer indices, or 'all', or "
                         "'back' (only layers in back half)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    # Load back half
    model, config = load_back_half(
        args.model, args.split_layer, device, dtype)
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size

    # Determine which layers to capture
    if args.capture_layers == "all":
        capture_layers = set(range(n_layers))
    elif args.capture_layers == "back":
        capture_layers = set(range(args.split_layer, n_layers))
    else:
        capture_layers = set(int(x) for x in args.capture_layers.split(","))

    # For front-half layers, we capture from the hidden states as they
    # arrive (the front half sends post-layer hidden states)
    front_capture = capture_layers & set(range(args.split_layer))
    back_capture = capture_layers & set(range(args.split_layer, n_layers))

    print(f"Capturing {len(capture_layers)} layers "
          f"(front: {len(front_capture)}, back: {len(back_capture)})",
          flush=True)

    # Listen for front half connection
    print(f"Listening on port {args.listen_port}...", flush=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.listen_port))
    server.listen(1)
    conn, addr = server.accept()
    print(f"  Front half connected from {addr}", flush=True)

    # Receive metadata
    meta_len = struct.unpack("!Q", conn.recv(8))[0]
    meta = json.loads(conn.recv(meta_len).decode())
    n_texts = meta["n_texts"]
    ids = meta["ids"]
    print(f"  Expecting {n_texts} texts, d_model={meta['d_model']}, "
          f"n_layers={meta['n_layers']}", flush=True)

    # NOTE: for front-half layer capture, the front script would need
    # to send per-layer hidden states, not just the split-point state.
    # For simplicity, this version only captures back-half layers.
    # To capture ALL layers, modify extract_front.py to send each
    # layer's last-token hidden state alongside the split-point state.
    if front_capture:
        print(f"  WARNING: front-half layer capture ({sorted(front_capture)}) "
              f"requires extract_front.py to send per-layer states. "
              f"Only back-half layers will be captured in this run.",
              flush=True)
        capture_layers = back_capture

    # Collect activations: {layer: [n_texts, d_model]}
    all_acts = {L: [] for L in sorted(capture_layers)}

    t0 = time.time()
    for i in range(n_texts):
        # Receive split-point hidden state from front half
        hidden = recv_tensor(conn)  # [1, d_model]

        # Run through back half with capture
        captured = forward_back_with_capture(
            model, hidden.unsqueeze(0), args.split_layer, n_layers,
            capture_layers, device)

        for L in sorted(capture_layers):
            if L in captured:
                all_acts[L].append(captured[L].squeeze(0))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_texts - i - 1) / rate
            print(f"  [{i+1}/{n_texts}] {rate:.1f} texts/s, "
                  f"ETA {eta/60:.0f}min", flush=True)

    # Check for done signal
    # (front sends a zero-length message to signal completion)
    conn.recv(8)  # consume the done signal
    conn.close()
    server.close()

    # Stack into tensors
    activations = {}
    for L in sorted(all_acts):
        if all_acts[L]:
            activations[L] = torch.stack(all_acts[L])
            print(f"  Layer {L}: {activations[L].shape}", flush=True)

    # Save in the standard NLA activation format
    output = {
        "activations": activations,
        "ids": ids,
        "n_layers": n_layers,
        "n_texts": n_texts,
        "d_model": d_model,
        "source_model": args.model,
        "split_layer": args.split_layer,
        "captured_layers": sorted(capture_layers),
        "extraction_method": "split_pipeline_2box",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    elapsed = time.time() - t0
    print(f"\nDone: {n_texts} texts, {len(activations)} layers "
          f"in {elapsed:.0f}s → {args.output}", flush=True)


if __name__ == "__main__":
    main()
