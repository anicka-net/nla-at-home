#!/usr/bin/env python3
"""
Best-of-N reranking experiment for NLA.

Generates N descriptions per activation using the AV, scores each with the AR,
picks the best. Compares best-of-N vs single-sample vs greedy to determine
whether the SL model CAN produce better descriptions (GRPO could help) or
whether it's at its ceiling (GRPO won't help).

Usage:
  python3 scripts/rerank_experiment.py \
    --av-adapter output/nla-qwen25-7b-L20-av-tight \
    --ar-adapter output/nla-qwen25-7b-L20-ar-tight \
    --n-samples 50 --n-candidates 16 \
    --output evaluation/rerank_tight.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import decode_generated

REPO_ROOT = Path(__file__).parent.parent

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
}

INJECTION_SCALE = 150.0


def normalize_activation(v, target_scale):
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


def load_nla_meta(path):
    meta_path = Path(path) / "nla_meta.yaml"
    with open(meta_path) as f:
        return yaml.safe_load(f)


def is_peft_adapter(path):
    return (Path(path) / "adapter_config.json").exists()


def is_value_head_only(path):
    return (Path(path) / "value_head.safetensors").exists() and \
           not (Path(path) / "adapter_config.json").exists() and \
           not (Path(path) / "model.safetensors.index.json").exists()


def load_examples(n_samples, seed=42):
    act_path = REPO_ROOT / "corpus" / "activations" / "qwen25-7b_L20.pt"
    data = torch.load(act_path, weights_only=True, map_location="cpu")
    activations = data["activations"]
    text_ids = data["ids"]

    desc_path = REPO_ROOT / "corpus" / "generated" / "descriptions_L71pct_tight.json"
    with open(desc_path) as f:
        descriptions = json.load(f)
    desc_map = {d["id"]: d for d in descriptions}

    examples = []
    for i, tid in enumerate(text_ids):
        if tid in desc_map:
            examples.append({
                "id": tid,
                "activation": activations[i],
                "target_desc": desc_map[tid]["description"],
                "category": desc_map[tid].get("category", "unknown"),
            })

    rng = random.Random(seed)
    selected = rng.sample(examples, min(n_samples, len(examples)))
    print(f"Selected {len(selected)} examples from {len(examples)} available")
    return selected, examples


def load_av(av_adapter, model_key, device):
    model_name = MODELS[model_key]
    meta = load_nla_meta(av_adapter)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

    if is_peft_adapter(av_adapter):
        model = PeftModel.from_pretrained(base, av_adapter)
    else:
        model = base
    model.eval()

    training_meta = meta.get("training", {})
    use_chat = training_meta.get("chat_template", not is_peft_adapter(av_adapter))
    use_normalize = training_meta.get("injection_mode") == "normalize" or \
                    not is_peft_adapter(av_adapter)

    injection_char = meta["tokens"]["injection_char"]
    template = meta["prompt_templates"]["av"]
    content = template.replace("{injection_char}", injection_char)

    if use_chat:
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    else:
        prompt_tokens = tokenizer.encode(content, add_special_tokens=True)

    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)
    injection_scale = float(meta.get("extraction", {}).get("injection_scale", INJECTION_SCALE))

    return {
        "model": model, "tokenizer": tokenizer,
        "prompt_tokens": prompt_tokens, "inject_pos": inject_pos,
        "injection_scale": injection_scale,
        "use_normalize": use_normalize,
    }


def load_ar(ar_adapter, model_key, device):
    model_name = MODELS[model_key]
    meta = load_nla_meta(ar_adapter)
    extraction_layer = int(meta.get("extraction_layer_index", 20))

    if is_value_head_only(ar_adapter):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        inner = model.model if hasattr(model, "model") else model
        if len(inner.layers) > extraction_layer + 1:
            inner.layers = inner.layers[:extraction_layer + 1]
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        model.lm_head = torch.nn.Identity()
    elif is_peft_adapter(ar_adapter):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ar_adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ar_adapter, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ar_adapter, trust_remote_code=True)
        inner = model.model if hasattr(model, "model") else model
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vh_path = Path(ar_adapter) / "value_head.safetensors"
    value_head = None
    if vh_path.exists():
        d_model = int(meta["d_model"])
        with safe_open(str(vh_path), framework="pt") as f:
            vh_weight = f.get_tensor("weight")
        value_head = torch.nn.Linear(d_model, d_model, bias=False, dtype=vh_weight.dtype)
        value_head.weight = torch.nn.Parameter(vh_weight)
        value_head = value_head.to(device).eval()

    template = meta["prompt_templates"]["ar"]
    return {
        "model": model, "tokenizer": tokenizer,
        "template": template, "value_head": value_head,
        "device": device,
    }


def generate_one(av, activation, device, temperature=0.7, max_new_tokens=200):
    model = av["model"]
    tokenizer = av["tokenizer"]
    prompt_tokens = av["prompt_tokens"]
    inject_pos = av["inject_pos"]
    embed_layer = model.get_input_embeddings()

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    embeddings = embed_layer(input_ids)
    act = activation.to(device)
    if av["use_normalize"]:
        act = normalize_activation(act, av["injection_scale"])
    else:
        act = act * av["injection_scale"]
    embeddings[0, inject_pos, :] = act.to(embeddings.dtype)

    generate_kwargs = {
        "inputs_embeds": embeddings.to(model.dtype),
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    return decode_generated(output, prompt_tokens, tokenizer)


def ar_score_one(ar, description, target_activation):
    model = ar["model"]
    tokenizer = ar["tokenizer"]
    device = ar["device"]
    value_head = ar["value_head"]

    prompt = ar["template"].replace("{explanation}", description)
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        inner = model.model if hasattr(model, "model") else model
        outputs = inner(input_ids=input_ids, use_cache=False)
        hidden = outputs.last_hidden_state[0, -1]
        if value_head is not None:
            reconstructed = value_head(hidden.unsqueeze(0)).squeeze(0)
        else:
            reconstructed = hidden

    reconstructed = reconstructed.float().cpu()
    target = target_activation.float()
    cos = torch.nn.functional.cosine_similarity(
        reconstructed.unsqueeze(0), target.unsqueeze(0)).item()
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-adapter", required=True)
    parser.add_argument("--model", default="qwen25-7b")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading examples...")
    selected, pool = load_examples(args.n_samples)

    print(f"Loading AV from {args.av_adapter}...")
    av = load_av(args.av_adapter, args.model, device)

    print(f"Loading AR from {args.ar_adapter}...")
    ar = load_ar(args.ar_adapter, args.model, device)

    results = []
    greedy_cosines = []
    single_cosines = []
    best_cosines = []
    worst_cosines = []
    mean_cosines = []
    group_stds = []

    print(f"\nGenerating {args.n_candidates} candidates per activation "
          f"(temp={args.temperature})...")

    for i, ex in enumerate(selected):
        # Generate greedy (temperature=0)
        greedy_desc = generate_one(av, ex["activation"], device,
                                   temperature=0, max_new_tokens=200)
        greedy_cos = ar_score_one(ar, greedy_desc, ex["activation"])

        # Generate N candidates with sampling
        candidates = []
        for _ in range(args.n_candidates):
            desc = generate_one(av, ex["activation"], device,
                               temperature=args.temperature, max_new_tokens=200)
            cos = ar_score_one(ar, desc, ex["activation"])
            candidates.append({"description": desc, "cosine": cos})

        cosines = [c["cosine"] for c in candidates]
        best_idx = max(range(len(candidates)), key=lambda j: candidates[j]["cosine"])
        worst_idx = min(range(len(candidates)), key=lambda j: candidates[j]["cosine"])

        greedy_cosines.append(greedy_cos)
        single_cosines.append(cosines[0])
        best_cosines.append(cosines[best_idx])
        worst_cosines.append(cosines[worst_idx])
        mean_cosines.append(np.mean(cosines))
        group_stds.append(np.std(cosines))

        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "greedy_cosine": greedy_cos,
            "best_cosine": cosines[best_idx],
            "worst_cosine": cosines[worst_idx],
            "mean_cosine": np.mean(cosines),
            "std_cosine": np.std(cosines),
            "best_description": candidates[best_idx]["description"],
            "greedy_description": greedy_desc,
            "n_candidates": args.n_candidates,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}] "
                  f"greedy={np.mean(greedy_cosines):.4f} "
                  f"single={np.mean(single_cosines):.4f} "
                  f"best-of-{args.n_candidates}={np.mean(best_cosines):.4f} "
                  f"group_std={np.mean(group_stds):.6f}")

    print(f"\n=== Reranking Results (N={len(selected)}, "
          f"candidates={args.n_candidates}, temp={args.temperature}) ===")
    print(f"  Greedy (temp=0):       {np.mean(greedy_cosines):.4f} "
          f"+/- {np.std(greedy_cosines):.4f}")
    print(f"  Single sample:         {np.mean(single_cosines):.4f} "
          f"+/- {np.std(single_cosines):.4f}")
    print(f"  Best-of-{args.n_candidates}:           "
          f"{np.mean(best_cosines):.4f} +/- {np.std(best_cosines):.4f}")
    print(f"  Worst-of-{args.n_candidates}:          "
          f"{np.mean(worst_cosines):.4f} +/- {np.std(worst_cosines):.4f}")
    print(f"  Mean of candidates:    {np.mean(mean_cosines):.4f}")
    print(f"  Within-group std:      {np.mean(group_stds):.6f}")
    print(f"  Best-greedy gap:       {np.mean(best_cosines) - np.mean(greedy_cosines):+.4f}")

    headroom = np.mean(best_cosines) - np.mean(greedy_cosines)
    if headroom > 0.02:
        print(f"\n  >> GRPO has room: best-of-N is {headroom:.3f} above greedy")
    elif headroom > 0.005:
        print(f"\n  >> Marginal: best-of-N only {headroom:.3f} above greedy")
    else:
        print(f"\n  >> Ceiling reached: best-of-N ≈ greedy, GRPO won't help")

    output = {
        "config": {
            "av_adapter": args.av_adapter,
            "ar_adapter": args.ar_adapter,
            "n_samples": len(selected),
            "n_candidates": args.n_candidates,
            "temperature": args.temperature,
        },
        "summary": {
            "greedy_cosine": float(np.mean(greedy_cosines)),
            "single_cosine": float(np.mean(single_cosines)),
            "best_cosine": float(np.mean(best_cosines)),
            "worst_cosine": float(np.mean(worst_cosines)),
            "mean_cosine": float(np.mean(mean_cosines)),
            "group_std": float(np.mean(group_stds)),
            "headroom": float(headroom),
        },
        "per_example": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
