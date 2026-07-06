#!/usr/bin/env python3
"""End-to-end smoke test of the split-pipeline extraction — no second box.

Runs BOTH halves in one process on a small registry model (default
gemma3-1b) and compares every layer's last-token capture against the
ground truth from a direct full-model forward with output_hidden_states.

This validates: the manual layer loop (positions/rotary/causal mask),
the split handoff, and the capture indexing. It does NOT validate
DeepSeek's custom modeling signatures — after the first few ds4 texts,
sanity-check capture norms before committing hours of extraction.

Usage:
  python3 scripts/franken/smoke_test_split.py [--model gemma3-1b]
      [--split-layer 13] [--n-texts 4] [--device cpu] [--wire]

  --wire also routes the handoff through a localhost TCP socket to
  exercise wire.py (send/recv of the dict payload).

PASS = centered cos > 0.999 for every layer.
"""

import argparse
import sys
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from nla_lib import get_model
from forward_utils import make_layer_caller, prepare_positions

TEXTS = [
    "Explain how a hash map handles collisions.",
    "The quick brown fox jumps over the lazy dog and then writes a poem.",
    "SELECT customer_id, name FROM customers WHERE order_date > '2026-01-01';",
    "Compassion is defined as the wish that beings be free from suffering.",
]


def direct_reference(model, input_ids):
    """Ground truth via forward hooks on each block — the SAME convention
    as extract_activations.py (raw block output). NOT output_hidden_states:
    its last entry is post-final-RMSNorm, which is a different vector than
    the residual stream at the last layer (19x norm difference on gemma3)."""
    caps = []

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        caps.append(h[0, -1, :].float().clone())

    handles = [layer.register_forward_hook(hook)
               for layer in model.model.layers]
    with torch.no_grad():
        model(input_ids, use_cache=False)
    for hd in handles:
        hd.remove()
    return caps


def split_extraction(model, input_ids, split_layer, call_layer):
    """The same computation via the two-half manual loop."""
    caps = []
    with torch.no_grad():
        h = model.model.embed_tokens(input_ids)
        pos = prepare_positions(model, h)
        for i in range(split_layer):
            h = call_layer(model.model.layers[i], h, pos)
            caps.append(h[0, -1, :].float().clone())
        # handoff (optionally through the TCP wire)
        h = HANDOFF(h)
        pos = prepare_positions(model, h)
        n_layers = len(model.model.layers)
        for i in range(split_layer, n_layers):
            h = call_layer(model.model.layers[i], h, pos)
            caps.append(h[0, -1, :].float().clone())
    return caps


def make_wire_handoff():
    """Round-trip the tensor through a localhost socket via wire.py."""
    import socket
    from wire import send_tensor, recv_msg

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    conn, _ = server.accept()

    def handoff(h):
        send_tensor(client, {"h_split": h.to(torch.float16).cpu(),
                             "front_caps": torch.zeros(1)})
        msg = recv_msg(conn)
        return msg["h_split"].to(h.device, h.dtype)
    return handoff


def main():
    global HANDOFF
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-1b",
                    help="nla_lib registry key")
    ap.add_argument("--split-layer", type=int, default=None,
                    help="default: n_layers // 2")
    ap.add_argument("--n-texts", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wire", action="store_true",
                    help="route the handoff through localhost TCP")
    args = ap.parse_args()

    spec = get_model(args.model)
    print(f"Loading {spec.hf_id} on {args.device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float32,
        trust_remote_code=spec.trust_remote_code).to(args.device).eval()

    n_layers = len(model.model.layers)
    split = args.split_layer or n_layers // 2
    print(f"  {n_layers} layers, split at {split}", flush=True)

    HANDOFF = make_wire_handoff() if args.wire else (lambda h: h)
    call_layer = make_layer_caller(model)

    worst = 1.0
    for text in TEXTS[:args.n_texts]:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True)
        input_ids = torch.tensor(
            [tok.encode(chat, add_special_tokens=False)]).to(args.device)

        ref = direct_reference(model, input_ids)
        got = split_extraction(model, input_ids, split, call_layer)
        assert len(ref) == len(got) == n_layers

        for L, (r, g) in enumerate(zip(ref, got)):
            rc = r - r.mean()
            gc = g - g.mean()
            cos = torch.nn.functional.cosine_similarity(
                rc.unsqueeze(0), gc.unsqueeze(0)).item()
            worst = min(worst, cos)
            if cos < 0.999:
                print(f"  MISMATCH L{L} '{text[:30]}': centered cos {cos:.4f} "
                      f"(‖ref‖={r.norm():.1f} ‖got‖={g.norm():.1f})",
                      flush=True)
        print(f"  '{text[:40]}…' worst-so-far cos={worst:.5f}", flush=True)

    if worst > 0.999:
        print(f"\nPASS: all {n_layers} layers × {args.n_texts} texts, "
              f"min centered cos = {worst:.5f}")
        sys.exit(0)
    print(f"\nFAIL: min centered cos = {worst:.5f} — the manual layer loop "
          f"does not reproduce the model's own forward; fix forward_utils "
          f"before any real extraction")
    sys.exit(1)


if __name__ == "__main__":
    main()
