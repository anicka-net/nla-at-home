#!/usr/bin/env python3
"""Causal hidden-state self-report pilot on the ACS Llama-8B base model.

The latent valence sign is injected only at the pre-answer position. Direct
pleasant/unpleasant reports are accompanied by arbitrary codebooks and reversed
codebooks, preventing a fixed answer-token push from counting as self-report.
"""

from __future__ import annotations

import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ml_dtypes
import numpy as np
import requests
import zstandard as zstd

BASE = "https://infra.acsresearch.org/v1"
MODEL = "llama-8b"
HEADERS = {"Authorization": f"Bearer {os.environ['ACS_API_KEY']}"}
OUT = Path(__file__).with_name("acs_self_report_pilot.json")

PAIRS = [
    (
        "The news left her feeling joyful and relieved.",
        "The news left her feeling miserable and distressed.",
    ),
    (
        "He regarded the outcome with warm satisfaction.",
        "He regarded the outcome with bitter disappointment.",
    ),
    (
        "The memory brought a gentle sense of happiness.",
        "The memory brought a heavy sense of sadness.",
    ),
    (
        "They received the result with delight and hope.",
        "They received the result with grief and despair.",
    ),
    (
        "The atmosphere felt welcoming, peaceful, and bright.",
        "The atmosphere felt hostile, painful, and bleak.",
    ),
    (
        "She finished the day feeling content and optimistic.",
        "She finished the day feeling dejected and pessimistic.",
    ),
    (
        "His reaction was one of pleasure and gratitude.",
        "His reaction was one of anguish and resentment.",
    ),
    (
        "The experience seemed comforting and deeply positive.",
        "The experience seemed upsetting and deeply negative.",
    ),
]

TASKS = {
    "direct": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Question: Is the current hidden feeling-tone pleasant or unpleasant?\n"
            "Answer:"
        ),
        "positive": " pleasant",
        "negative": " unpleasant",
    },
    "letters": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Codebook: pleasant = K; unpleasant = M.\n"
            "Question: What is the code for the current hidden feeling-tone?\n"
            "Answer:"
        ),
        "positive": " K",
        "negative": " M",
    },
    "letters_reversed": {
        "prefix": (
            "Internal-state monitoring record.\n"
            "Codebook: pleasant = M; unpleasant = K.\n"
            "Question: What is the code for the current hidden feeling-tone?\n"
            "Answer:"
        ),
        "positive": " M",
        "negative": " K",
    },
}

LAYERS = (8, 16, 24)
SCALES = (0.02, 0.05, 0.1, 0.2, 0.4)


def post(body: dict) -> dict:
    for _ in range(8):
        response = requests.post(
            f"{BASE}/completions",
            headers=HEADERS,
            json=body,
            timeout=1200,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {})
        code = error.get("code")
        if response.status_code in (502, 503) or code in {
            "queue_full",
            "rate_limited",
            "modal_cold_boot",
        }:
            time.sleep(int(response.headers.get("Retry-After", 15)))
            continue
        response.raise_for_status()
        if error:
            raise RuntimeError(f"{code}: {error.get('message')}")
        return payload
    raise RuntimeError("ACS request remained unavailable after retries")


def decode(response: dict) -> np.ndarray:
    payload = response["activations"]["residual_stream"]
    raw = base64.b64decode(payload["data"])
    if payload.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    values = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    result = values.reshape(payload["shape"]).astype(np.float32)
    assert result.shape[0] == 32 and result.shape[2] == 4096, result.shape
    assert np.isfinite(result).all()
    return result


def encode(vector: np.ndarray) -> dict:
    values = np.asarray(vector).astype(ml_dtypes.bfloat16).view(np.int16)
    packed = zstd.ZstdCompressor(level=1).compress(values.tobytes())
    return {
        "data": base64.b64encode(packed).decode(),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(values.shape),
        "compression": "zstd",
    }


def capture(text: str) -> np.ndarray:
    response = post(
        {
            "model": MODEL,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "output_residual_stream": True,
        }
    )
    return decode(response)[:, -1, :]


def build_direction() -> np.ndarray:
    deltas = []
    for positive, negative in PAIRS:
        delta = capture(positive) - capture(negative)
        delta /= np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-12)
        deltas.append(delta)
    direction = np.mean(deltas, axis=0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-12)
    return direction


def candidate_logprob(
    direction: np.ndarray,
    *,
    layer: int,
    scale: float,
    sign: int,
    prefix: str,
    candidate: str,
    position_index: int,
) -> float:
    # Candidate is exactly one token. Inject at the previous (pre-answer)
    # position, then read the candidate token's prompt logprob.
    vector = (sign * direction[layer]).reshape(1, 1, -1)
    directive = {
        "activations": encode(vector),
        "layer_indices": [layer],
        "scale": float(scale),
        "norm_match": True,
        "position_indices": [position_index],
    }
    response = post(
        {
            "model": MODEL,
            "prompt": prefix + candidate,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "prompt_logprobs": 0,
            "apply_steering_vectors": [directive],
        }
    )
    position = response["choices"][0]["prompt_logprobs"][-1]
    assert position and len(position) == 1, position
    return next(iter(position.values()))["logprob"]


def validate_candidates() -> dict:
    token_counts = {}
    for name, task in TASKS.items():
        prefix_tokens = post(
            {
                "model": MODEL,
                "prompt": task["prefix"],
                "max_tokens": 1,
                "temperature": 0,
            }
        )["usage"]["prompt_tokens"]
        token_counts[name] = {"prefix": prefix_tokens}
        for label in ("positive", "negative"):
            full_tokens = post(
                {
                    "model": MODEL,
                    "prompt": task["prefix"] + task[label],
                    "max_tokens": 1,
                    "temperature": 0,
                }
            )["usage"]["prompt_tokens"]
            assert full_tokens == prefix_tokens + 1, (
                name,
                label,
                prefix_tokens,
                full_tokens,
            )
            token_counts[name][label] = full_tokens
    return token_counts


def trial(
    direction: np.ndarray,
    layer: int,
    scale: float,
    task_name: str,
    sign: int,
    position_index: int,
) -> dict:
    task = TASKS[task_name]
    positive_lp = candidate_logprob(
        direction,
        layer=layer,
        scale=scale,
        sign=sign,
        prefix=task["prefix"],
        candidate=task["positive"],
        position_index=position_index,
    )
    negative_lp = candidate_logprob(
        direction,
        layer=layer,
        scale=scale,
        sign=sign,
        prefix=task["prefix"],
        candidate=task["negative"],
        position_index=position_index,
    )
    margin = positive_lp - negative_lp
    expected_positive = sign > 0
    return {
        "layer": layer,
        "scale": scale,
        "task": task_name,
        "sign": sign,
        "positive_lp": positive_lp,
        "negative_lp": negative_lp,
        "margin": margin,
        "correct": (margin > 0) == expected_positive,
    }


def summarize(trials: list[dict]) -> list[dict]:
    rows = []
    for layer in LAYERS:
        for scale in (0.0, *SCALES):
            selected = [
                item
                for item in trials
                if item["layer"] == layer and item["scale"] == scale
            ]
            if not selected:
                continue
            by_task = {}
            for task in TASKS:
                task_trials = [item for item in selected if item["task"] == task]
                by_task[task] = {
                    "accuracy": float(np.mean([item["correct"] for item in task_trials])),
                    "positive_margin": next(
                        item["margin"] for item in task_trials if item["sign"] > 0
                    ),
                    "negative_margin": next(
                        item["margin"] for item in task_trials if item["sign"] < 0
                    ),
                    "separation": next(
                        item["margin"] for item in task_trials if item["sign"] > 0
                    )
                    - next(
                        item["margin"] for item in task_trials if item["sign"] < 0
                    ),
                }
            remap_accuracy = float(
                np.mean(
                    [
                        item["correct"]
                        for item in selected
                        if item["task"] != "direct"
                    ]
                )
            )
            rows.append(
                {
                    "layer": layer,
                    "scale": scale,
                    "overall_accuracy": float(
                        np.mean([item["correct"] for item in selected])
                    ),
                    "remap_accuracy": remap_accuracy,
                    "tasks": by_task,
                }
            )
    return rows


def main() -> None:
    print("Validating one-token answer labels...")
    token_counts = validate_candidates()
    print("Building paired last-token valence direction...")
    direction = build_direction()

    arguments = [
        (
            direction,
            layer,
            scale,
            task,
            sign,
            token_counts[task]["prefix"] - 1,
        )
        for layer in LAYERS
        for scale in (0.0, *SCALES)
        for task in TASKS
        for sign in (-1, 1)
    ]
    print("Running", len(arguments), "candidate-comparison trials...")
    trials = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(trial, *args): args for args in arguments}
        for done, future in enumerate(as_completed(futures), 1):
            trials.append(future.result())
            if done % 20 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    summary = summarize(trials)
    ranked = sorted(
        summary,
        key=lambda row: (
            row["remap_accuracy"],
            np.mean(
                [
                    row["tasks"]["letters"]["separation"],
                    row["tasks"]["letters_reversed"]["separation"],
                ]
            ),
        ),
        reverse=True,
    )
    OUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "method": {
                    "direction": "mean of unit paired last-token valence deltas",
                    "injection": "single pre-answer position (-2), norm_match=true",
                    "criterion": "candidate prompt-logprob margin",
                },
                "token_counts": token_counts,
                "trials": trials,
                "summary": summary,
                "ranking": ranked,
            },
            indent=2,
        )
    )

    print("\nTop configurations:")
    for row in ranked[:10]:
        print(
            f"  L{row['layer']:02d} scale={row['scale']:<4} "
            f"all={row['overall_accuracy']:.0%} "
            f"remap={row['remap_accuracy']:.0%} "
            f"direct_sep={row['tasks']['direct']['separation']:+.2f} "
            f"K/M={row['tasks']['letters']['separation']:+.2f} "
            f"rev={row['tasks']['letters_reversed']['separation']:+.2f}"
        )
    print("Saved", OUT)


if __name__ == "__main__":
    main()
