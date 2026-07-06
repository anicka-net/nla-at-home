#!/usr/bin/env python3
"""Manual layer-loop helpers for split-pipeline extraction.

HF decoder layers are normally driven by Model.forward, which builds
position_ids, rotary position_embeddings and the causal mask. When half
the layers live on another box we must drive them by hand — and custom
trust_remote_code modeling (DeepSeek) may use an older signature.

make_layer_caller() dispatches defensively across the known signatures.
IMPORTANT: "runs without error" is NOT verification — run
smoke_test_split.py against a registry model (direct-forward comparison,
cos must be ~1.0) after any change here, and eyeball the first few ds4
captures for norm sanity before committing hours of extraction.
"""

import torch


def prepare_positions(model, hidden_states):
    """position_ids [1, seq] + rotary embeddings if the model exposes
    top-level rotary module(s). Gemma3-style models carry TWO rotaries
    (global + local sliding-window) — both are prepared when present."""
    seq = hidden_states.shape[1]
    position_ids = torch.arange(seq, device=hidden_states.device).unsqueeze(0)

    def _rot(name):
        mod = getattr(model.model, name, None)
        if mod is None:
            return None
        try:
            return mod(hidden_states, position_ids)
        except TypeError:
            return None  # older rotary signatures compute inside attention

    return {"position_ids": position_ids,
            "position_embeddings": _rot("rotary_emb"),
            "position_embeddings_local": _rot("rotary_emb_local")}


def make_layer_caller(model):
    """Returns call(layer, h, pos) -> new h, trying signatures from newest
    to oldest. attention_mask=None lets sdpa/flash paths default to causal
    (q_len > 1 => is_causal), which matches full-prompt extraction."""
    def call(layer, h, pos):
        attempts = []
        if pos["position_embeddings"] is not None and \
           pos["position_embeddings_local"] is not None:
            # Gemma3-style: dual rotary, both required positional-ish kwargs
            attempts.append(dict(
                position_ids=pos["position_ids"],
                position_embeddings_global=pos["position_embeddings"],
                position_embeddings_local=pos["position_embeddings_local"]))
        if pos["position_embeddings"] is not None:
            attempts.append(dict(position_ids=pos["position_ids"],
                                 position_embeddings=pos["position_embeddings"]))
        attempts.append(dict(position_ids=pos["position_ids"]))
        attempts.append(dict())
        last_err = None
        for kw in attempts:
            try:
                out = layer(h, attention_mask=None, **kw)
                return out[0] if isinstance(out, tuple) else out
            except TypeError as e:
                last_err = e
        raise TypeError(
            f"No known layer signature worked for {type(layer).__name__}: "
            f"{last_err} — inspect the model's own forward and extend "
            f"make_layer_caller.")
    return call
