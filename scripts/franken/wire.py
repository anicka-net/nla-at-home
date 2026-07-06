#!/usr/bin/env python3
"""Extract activations from DeepSeek V4 Flash using 2-box pipeline parallelism.

Splits the model across two GB10 boxes connected via 200 Gbps link-local.
Front half (layers 0..split-1) runs on box A, back half (split..42) on box B.
Hidden states are piped via a TCP socket over the fast link.

Architecture:
  Box A (front) loads embedding + layers 0..split-1
    → sends hidden_states tensor via TCP
  Box B (back) loads layers split..42 + lm_head
    → receives hidden_states, continues forward, captures activations

Usage:
  # On front-host (front half):
  ~/venv/bin/python scripts/franken/extract_front.py \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --texts corpus/texts.jsonl \
    --split-layer 22 \
    --backend-host 169.254.217.181 \
    --backend-port 29500

  # On deepthought (back half) — start FIRST:
  ~/venv/bin/python scripts/franken/extract_back.py \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --split-layer 22 \
    --listen-port 29500 \
    --output corpus/activations/ds4-flash_all_layers.pt \
    --capture-layers all

The split at layer 22 gives ~86 GB on the front box, ~80 GB on the back.
Both fit in 128 GB unified memory with room for activations and OS.

See working-docs/franken-nla-plan.md for the full plan.
"""

import argparse
import json
import socket
import struct
import io
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Wire protocol: length-prefixed pickled tensors over TCP
# ---------------------------------------------------------------------------

def send_tensor(sock, tensor):
    """Send a tensor over a socket as length-prefixed bytes."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    data = buf.getvalue()
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recv_tensor(sock):
    """Receive a length-prefixed tensor from a socket."""
    raw_len = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw_len)[0]
    data = _recv_exact(sock, length)
    return torch.load(io.BytesIO(data), weights_only=True)


def send_done(sock):
    """Signal completion."""
    sock.sendall(struct.pack("!Q", 0))


def recv_msg(sock):
    """Receive a payload (tensor or dict of tensors); None = done signal."""
    raw_len = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw_len)[0]
    if length == 0:
        return None
    data = _recv_exact(sock, length)
    return torch.load(io.BytesIO(data), weights_only=True)


def recv_is_done(sock):
    """Check if the other side signaled completion."""
    raw_len = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw_len)[0]
    return length == 0


def _recv_exact(sock, n):
    """Receive exactly n bytes."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError("Socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
