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

Supported architectures:
  1. Gemma3: dual rotary (position_embeddings_global + _local)
  2. Llama/Qwen/Mistral: single rotary (position_ids + position_embeddings)
  3. Older models: position_ids only, or bare call
  5. DeepSeek V4: dict-keyed dual rotary {"main": (cos,sin),
     "compress": (cos,sin)}, Hyper-Connections (hc_mult=4 → hidden states
     are [B,S,4,D] between layers), hash-routing MoE needs input_ids for
     the first num_hash_layers layers. Layer returns a single tensor.
"""

import torch


def _is_deepseek_v4(model):
    """Detect DeepSeek V4 by config model_type or hc_mult presence."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    return (getattr(cfg, "model_type", "") == "deepseek_v4"
            or getattr(cfg, "hc_mult", 1) > 1)


def prepare_positions(model, hidden_states):
    """position_ids [1, seq] + rotary embeddings if the model exposes
    top-level rotary module(s). Gemma3-style models carry TWO rotaries
    (global + local sliding-window) — both are prepared when present.

    DeepSeek V4: builds dict-keyed dual rotary {"main": ..., "compress": ...}
    from the top-level rotary_emb with layer_type kwarg. Hidden states may
    be [B, S, hc_mult, D]; we pass [B, S, D] (first HC slot) to rotary."""
    # For HC models hidden_states may be [B, S, hc_mult, D]
    if hidden_states.dim() == 4:
        seq = hidden_states.shape[1]
        embed_for_rot = hidden_states[:, :, 0, :]
    else:
        seq = hidden_states.shape[1]
        embed_for_rot = hidden_states

    position_ids = torch.arange(seq, device=hidden_states.device).unsqueeze(0)

    if _is_deepseek_v4(model):
        rotary = model.model.rotary_emb
        main = rotary(embed_for_rot, position_ids=position_ids,
                      layer_type="main")
        compress = rotary(embed_for_rot, position_ids=position_ids,
                          layer_type="compress")
        return {"position_ids": position_ids,
                "position_embeddings": {"main": main, "compress": compress},
                "position_embeddings_local": None,
                "_is_v4": True}

    def _rot(name):
        mod = getattr(model.model, name, None)
        if mod is None:
            return None
        try:
            return mod(embed_for_rot, position_ids)
        except TypeError:
            return None  # older rotary signatures compute inside attention

    return {"position_ids": position_ids,
            "position_embeddings": _rot("rotary_emb"),
            "position_embeddings_local": _rot("rotary_emb_local"),
            "_is_v4": False}


def expand_for_hc(model, hidden_states):
    """Expand [B, S, D] → [B, S, hc_mult, D] for DeepSeek V4.
    Call ONCE after embed_tokens, before the layer loop. No-op for
    non-HC models."""
    hc_mult = getattr(getattr(model, "config", None), "hc_mult", 1)
    if hc_mult <= 1:
        return hidden_states
    return hidden_states.unsqueeze(2).expand(
        -1, -1, hc_mult, -1).contiguous()


def collapse_hc(model, hidden_states):
    """Collapse [B, S, hc_mult, D] → [B, S, D] via hc_head + norm.
    Call ONCE after the layer loop for DeepSeek V4. No-op for non-HC."""
    if hidden_states.dim() != 4:
        return hidden_states
    hc_head = getattr(model.model, "hc_head", None)
    norm = getattr(model.model, "norm", None)
    if hc_head is not None:
        hidden_states = hc_head(hidden_states)
    if norm is not None and hidden_states.dim() == 3:
        hidden_states = norm(hidden_states)
    return hidden_states


def capture_from_h(hidden_states, token_idx=-1):
    """Extract the capture vector from hidden states at a token position.
    For HC models [B, S, hc_mult, D]: captures ALL HC slots as [hc_mult, D].
    For standard models [B, S, D]: captures [D].
    Always returns float32 CPU clone."""
    if hidden_states.dim() == 4:
        return hidden_states[0, token_idx, :, :].float().cpu().clone()
    return hidden_states[0, token_idx, :].float().cpu().clone()


def make_layer_caller(model):
    """Returns call(layer, h, pos, **kwargs) -> new h, trying signatures
    from newest to oldest. attention_mask=None lets sdpa/flash paths
    default to causal (q_len > 1 => is_causal), matching full-prompt
    extraction.

    DeepSeek V4 gets its own path: dict-keyed position_embeddings,
    input_ids for hash-routing MoE, single-tensor return."""
    is_v4 = _is_deepseek_v4(model)

    def call(layer, h, pos, **kwargs):
        if is_v4:
            out = layer(h,
                        position_embeddings=pos["position_embeddings"],
                        position_ids=pos["position_ids"],
                        attention_mask=None,
                        input_ids=kwargs.get("input_ids"))
            return out[0] if isinstance(out, tuple) else out

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
