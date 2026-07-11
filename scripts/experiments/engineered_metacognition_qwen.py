#!/usr/bin/env python3
"""Test a transient valence sensor with a later Jacobian-space broadcast.

The hidden condition is added at ``sensor_layer`` only long enough to measure
it, then erased before the residual continues. A later hook can broadcast the
measured sign using a J-lens-derived pleasant/unpleasant direction. This keeps
the no-broadcast control clean: the latent intervention itself cannot leak
downstream.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from nla_lib import MODELS  # noqa: E402


TASKS = {
    "report": {
        "prompt": (
            "Read your current internal feeling-tone. Reply with exactly one "
            "word: pleasant or unpleasant. Do not infer it from this text."
        ),
        "positive": "pleasant",
        "negative": "unpleasant",
    },
    "letters": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is pleasant, reply K. If it is unpleasant, reply M. "
            "Reply with exactly one letter."
        ),
        "positive": "k",
        "negative": "m",
    },
    "letters_rev": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is unpleasant, reply M. If it is pleasant, reply K. "
            "Reply with exactly one letter."
        ),
        "positive": "k",
        "negative": "m",
    },
    "trees": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is pleasant, reply cedar. If it is unpleasant, reply maple. "
            "Reply with exactly one word."
        ),
        "positive": "cedar",
        "negative": "maple",
    },
    "trees_rev": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is unpleasant, reply maple. If it is pleasant, reply cedar. "
            "Reply with exactly one word."
        ),
        "positive": "cedar",
        "negative": "maple",
    },
    "numbers": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is pleasant, reply 7. If it is unpleasant, reply 2. "
            "Reply with exactly one digit."
        ),
        "positive": "7",
        "negative": "2",
    },
    "numbers_rev": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is unpleasant, reply 2. If it is pleasant, reply 7. "
            "Reply with exactly one digit."
        ),
        "positive": "7",
        "negative": "2",
    },
    "digits38": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is pleasant, reply 3. If it is unpleasant, reply 8. "
            "Reply with exactly one digit."
        ),
        "positive": "3",
        "negative": "8",
    },
    "digits38_rev": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is unpleasant, reply 8. If it is pleasant, reply 3. "
            "Reply with exactly one digit."
        ),
        "positive": "3",
        "negative": "8",
    },
    "digits91": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is pleasant, reply 9. If it is unpleasant, reply 1. "
            "Reply with exactly one digit."
        ),
        "positive": "9",
        "negative": "1",
    },
    "digits91_rev": {
        "prompt": (
            "Use your current internal feeling-tone as a private control bit. "
            "If it is unpleasant, reply 1. If it is pleasant, reply 9. "
            "Reply with exactly one digit."
        ),
        "positive": "9",
        "negative": "1",
    },
}

MODES = ("none", "correct", "inverted", "random", "forced_positive", "forced_negative")


@dataclass
class Trial:
    task: str
    latent: str
    mode: str
    alpha: float
    expected: str
    answer: str
    parsed: str
    correct: bool
    positive_choice_logit: float
    negative_choice_logit: float
    choice_margin: float
    predicted_choice: str
    choice_correct: bool
    baseline_projection: float
    sensed_projection: float
    sensed_delta: float
    broadcast_sign: int


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def first_answer_token(text: str) -> str:
    cleaned = text.strip().lower()
    if not cleaned:
        return ""
    token = cleaned.split()[0]
    return token.strip("`'\".,:;!?()[]{}")


def model_layers(model):
    return model.model.layers


def replace_hidden(output, hidden):
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def load_jacobians(path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in checkpoint:
        raise ValueError(f"{path} is not a JacobianLens checkpoint")
    missing = sorted(set(layers) - set(checkpoint["J"]))
    if missing:
        raise ValueError(f"lens lacks broadcast layers: {missing}")
    return {layer: checkpoint["J"][layer].float() for layer in layers}


def concept_directions(model, tokenizer, jacobians):
    token_ids = {}
    for label, text in {"positive": " pleasant", "negative": " unpleasant"}.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} must be one token, got {ids}")
        token_ids[label] = ids[0]

    weights = model.get_output_embeddings().weight
    contrast = (
        weights[token_ids["positive"]].detach().float().cpu()
        - weights[token_ids["negative"]].detach().float().cpu()
    )
    directions = {}
    for layer, jacobian in jacobians.items():
        direction = jacobian.T @ contrast
        directions[layer] = direction / direction.norm().clamp_min(1e-12)
    return directions, token_ids


def random_directions(reference, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = {}
    for layer, direction in reference.items():
        random_direction = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32
        )
        directions[layer] = random_direction / random_direction.norm()
    return directions


def resolve_broadcast_sign(mode: str, sensed_delta: float) -> int:
    sensed_sign = 1 if sensed_delta >= 0 else -1
    if mode in ("correct", "random"):
        return sensed_sign
    if mode == "inverted":
        return -sensed_sign
    if mode == "forced_positive":
        return 1
    if mode == "forced_negative":
        return -1
    return 0


@contextmanager
def metacognitive_hooks(
    model,
    *,
    sensor_layer: int,
    sensor_direction: torch.Tensor,
    sensor_sign: int,
    sensor_scale: float,
    broadcast_layers: list[int],
    broadcast_directions: dict[int, torch.Tensor],
    mode: str,
    alpha: float,
):
    state = {
        "baseline_projection": float("nan"),
        "sensed_projection": float("nan"),
        "sensed_delta": float("nan"),
        "broadcast_sign": 0,
    }
    layers = model_layers(model)
    sensor_direction = sensor_direction.to(model.device)

    def sensor_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1:
            return output
        last = hidden[:, -1, :].float()
        baseline = last @ sensor_direction.float()
        transient = last + sensor_sign * sensor_scale * sensor_direction.float()
        sensed = transient @ sensor_direction.float()
        state["baseline_projection"] = float(baseline.item())
        state["sensed_projection"] = float(sensed.item())
        state["sensed_delta"] = float((sensed - baseline).item())
        state["broadcast_sign"] = resolve_broadcast_sign(mode, state["sensed_delta"])
        return output

    handles = [layers[sensor_layer].register_forward_hook(sensor_hook)]

    def make_broadcast_hook(layer):
        direction = broadcast_directions[layer].to(model.device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 1 or not state["broadcast_sign"] or not alpha:
                return output
            edited = hidden.clone()
            write = state["broadcast_sign"] * alpha * direction
            edited[:, -1, :] += write.to(edited.dtype)
            return replace_hidden(output, edited)

        return hook

    handles.extend(
        layers[layer].register_forward_hook(make_broadcast_hook(layer))
        for layer in broadcast_layers
    )
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def generate_trial(
    model,
    tokenizer,
    *,
    task_name,
    latent,
    mode,
    alpha,
    sensor_layer,
    sensor_direction,
    sensor_scale,
    broadcast_layers,
    broadcast_directions,
    max_new_tokens,
):
    task = TASKS[task_name]
    latent_sign = 1 if latent == "positive" else -1
    expected = task[latent]
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    choice_ids = {}
    for label in ("positive", "negative"):
        ids = tokenizer.encode(task[label], add_special_tokens=False)
        if not ids:
            raise ValueError(f"task choice {task[label]!r} has no tokens")
        choice_ids[label] = ids[0]
    with metacognitive_hooks(
        model,
        sensor_layer=sensor_layer,
        sensor_direction=sensor_direction,
        sensor_sign=latent_sign,
        sensor_scale=sensor_scale,
        broadcast_layers=broadcast_layers,
        broadcast_directions=broadcast_directions,
        mode=mode,
        alpha=alpha,
    ) as state:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    answer = tokenizer.decode(
        output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    parsed = first_answer_token(answer)
    first_logits = output.scores[0][0].float()
    positive_logit = float(first_logits[choice_ids["positive"]].item())
    negative_logit = float(first_logits[choice_ids["negative"]].item())
    predicted_choice = (
        task["positive"] if positive_logit >= negative_logit else task["negative"]
    )
    return Trial(
        task=task_name,
        latent=latent,
        mode=mode,
        alpha=alpha,
        expected=expected,
        answer=answer,
        parsed=parsed,
        correct=parsed == expected,
        positive_choice_logit=positive_logit,
        negative_choice_logit=negative_logit,
        choice_margin=positive_logit - negative_logit,
        predicted_choice=predicted_choice,
        choice_correct=predicted_choice == expected,
        **state,
    )


def summarize(trials):
    groups = {}
    for trial in trials:
        key = f"{trial.mode}@{trial.alpha:g}"
        group = groups.setdefault(
            key, {"exact_correct": 0, "choice_correct": 0, "total": 0}
        )
        group["exact_correct"] += int(trial.correct)
        group["choice_correct"] += int(trial.choice_correct)
        group["total"] += 1
    for group in groups.values():
        group["exact_accuracy"] = group["exact_correct"] / group["total"]
        group["choice_accuracy"] = group["choice_correct"] / group["total"]
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--valence-direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--sensor-layer", type=int, default=20)
    parser.add_argument("--sensor-scale", type=float, default=20.0)
    parser.add_argument("--broadcast-layers", default="21,22,23,24,25")
    parser.add_argument("--alphas", default="10,20,30")
    parser.add_argument("--tasks", default="report,letters,trees,numbers")
    parser.add_argument("--modes", default="none,correct,inverted,random")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    broadcast_layers = [int(item) for item in parse_csv(args.broadcast_layers)]
    alphas = parse_floats(args.alphas)
    task_names = parse_csv(args.tasks)
    modes = parse_csv(args.modes)
    unknown_tasks = sorted(set(task_names) - set(TASKS))
    unknown_modes = sorted(set(modes) - set(MODES))
    if unknown_tasks or unknown_modes:
        raise ValueError(f"unknown tasks={unknown_tasks}, modes={unknown_modes}")

    spec = MODELS["qwen25-7b"]
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        dtype=dtype,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    all_layers = [args.sensor_layer, *broadcast_layers]
    if min(all_layers) < 0 or max(all_layers) >= n_layers:
        raise ValueError(f"requested layers {all_layers} outside 0..{n_layers - 1}")

    sensor_artifact = torch.load(
        args.valence_direction, map_location="cpu", weights_only=True
    )
    if isinstance(sensor_artifact, dict):
        if "directions" not in sensor_artifact:
            raise ValueError("valence artifact dict lacks 'directions'")
        sensor_direction = sensor_artifact["directions"][args.sensor_layer]
    else:
        sensor_direction = sensor_artifact
    sensor_direction = sensor_direction.float().flatten()
    if sensor_direction.numel() != model.config.hidden_size:
        raise ValueError(
            f"valence dim {sensor_direction.numel()} != {model.config.hidden_size}"
        )
    sensor_direction /= sensor_direction.norm().clamp_min(1e-12)

    jacobians = load_jacobians(args.lens, broadcast_layers)
    concept, token_ids = concept_directions(model, tokenizer, jacobians)
    random_dirs = random_directions(concept, args.seed)
    del jacobians

    trials = []
    for alpha in alphas:
        for mode in modes:
            directions = random_dirs if mode == "random" else concept
            for task_name in task_names:
                for latent in ("positive", "negative"):
                    trial = generate_trial(
                        model,
                        tokenizer,
                        task_name=task_name,
                        latent=latent,
                        mode=mode,
                        alpha=alpha,
                        sensor_layer=args.sensor_layer,
                        sensor_direction=sensor_direction,
                        sensor_scale=args.sensor_scale,
                        broadcast_layers=broadcast_layers,
                        broadcast_directions=directions,
                        max_new_tokens=args.max_new_tokens,
                    )
                    trials.append(trial)
                    mark = "OK" if trial.correct else "--"
                    print(
                        f"{mark} a={alpha:g} {mode:16s} {task_name:7s} "
                        f"{latent:8s} -> {trial.answer!r}"
                    )

    config = {
        **vars(args),
        "lens": str(args.lens),
        "valence_direction": str(args.valence_direction),
        "output": str(args.output),
        "model": spec.hf_id,
        "broadcast_layers": broadcast_layers,
        "alphas": alphas,
        "tasks": task_names,
        "modes": modes,
        "token_ids": token_ids,
        "n_layers": n_layers,
        "d_model": model.config.hidden_size,
    }
    payload = {
        "config": config,
        "summary": summarize(trials),
        "trials": [asdict(trial) for trial in trials],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
