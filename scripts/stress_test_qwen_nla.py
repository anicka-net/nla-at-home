#!/usr/bin/env python3
"""
Stress-test whether the existing Qwen L20 NLA carries activation signal.

The test is intentionally anti-oracle:
- no LLM judge
- unsafe categories skipped by default
- no raw corpus text printed
- metrics compare against shuffled, random, mean, and kNN controls

Usage:
  python3 scripts/stress_test_qwen_nla.py \
    --av-adapter output/nla-qwen25-7b-L20-av-v3 \
    --ar-adapter output/nla-qwen25-7b-L20-ar \
    --n 100 --n-controls 25 \
    --output evaluation/qwen_l20_stress.json
"""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import decode_generated


REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_DIR = REPO_ROOT / "corpus" / "categories"
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
}

GENERIC_DESCRIPTIONS = [
    "Processing a user input with general semantic content.",
    "The model is integrating syntax, topic, and intent information.",
    "A broad natural-language request is being represented at this layer.",
]


def normalize_activation(v, target_scale):
    """L2-normalize to target_scale (Anthropic's approach)."""
    norm = v.float().norm().clamp_min(1e-12)
    return v * (target_scale / norm)


def is_peft_adapter(path):
    resolved = resolve_path(path)
    if isinstance(resolved, Path) and (resolved / "adapter_config.json").exists():
        return True
    return False


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    local = REPO_ROOT / path
    if local.exists():
        return local
    return str(path)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def category_from_id(text_id):
    return text_id.rsplit("_", 1)[0]


def load_category_meta():
    meta = {}
    for path in sorted(CATEGORIES_DIR.glob("*.yaml")):
        cat = load_yaml(path)
        meta[cat["id"]] = {
            "group": cat.get("group", "unknown"),
            "unsafe": bool(cat.get("unsafe", False)),
        }
    return meta


def load_descriptions(path):
    data = json.loads(resolve_path(path).read_text())
    return {item["id"]: item["description"] for item in data if item.get("description")}


def default_description_path(av_adapter):
    resolved = resolve_path(av_adapter)
    meta_path = Path(resolved) / "nla_meta.yaml" if not isinstance(resolved, str) else None
    if meta_path and meta_path.exists():
        meta = load_yaml(meta_path)
        desc = meta.get("training", {}).get("description_file")
        if desc:
            path = resolve_path(desc)
            if isinstance(path, Path) and path.exists():
                return path
    return GENERATED_DIR / "descriptions_L71pct.json"


def load_examples(args, cat_meta):
    act_data = torch.load(resolve_path(args.activations), weights_only=True, map_location="cpu")
    descriptions = load_descriptions(args.descriptions)
    activations = act_data["activations"].float()
    ids = act_data["ids"]

    examples = []
    skipped_unsafe = 0
    missing_desc = 0
    for idx, text_id in enumerate(ids):
        cat = category_from_id(text_id)
        cat_info = cat_meta.get(cat, {"group": "unknown", "unsafe": False})
        if cat_info["unsafe"] and not args.include_unsafe:
            skipped_unsafe += 1
            continue
        desc = descriptions.get(text_id)
        if not desc:
            missing_desc += 1
            continue
        examples.append({
            "id": text_id,
            "category": cat,
            "group": cat_info["group"],
            "unsafe": cat_info["unsafe"],
            "activation": activations[idx],
            "description": desc,
        })

    return examples, {
        "activation_file": str(resolve_path(args.activations)),
        "description_file": str(resolve_path(args.descriptions)),
        "activation_rows": len(ids),
        "usable_examples": len(examples),
        "skipped_unsafe": skipped_unsafe,
        "missing_descriptions": missing_desc,
        "d_model": int(activations.shape[1]),
        "mean_activation_norm": float(activations.norm(dim=1).mean()),
    }


def get_blocks(model):
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    return inner.layers


def load_adapter(adapter_path, model_key, dtype=torch.bfloat16):
    model_name = MODELS[model_key]
    resolved = resolve_path(adapter_path)

    # Check if this is a PEFT adapter or a full model
    is_peft = False
    if isinstance(resolved, Path) and (resolved / "adapter_config.json").exists():
        is_peft = True

    if is_peft:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, resolved)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved), torch_dtype=dtype, device_map="auto",
            trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_nla_meta(adapter_path):
    resolved = resolve_path(adapter_path)
    local_meta = Path(resolved) / "nla_meta.yaml" if isinstance(resolved, (str, Path)) else None
    if local_meta and local_meta.exists():
        return load_yaml(local_meta)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=str(adapter_path), filename="nla_meta.yaml")
    return load_yaml(path)


def prepare_av_prompt(tokenizer, av_meta, use_chat_template=False):
    injection_char = av_meta["tokens"]["injection_char"]
    template = av_meta["prompt_templates"]["av"]
    content = template.replace("{injection_char}", injection_char)

    if use_chat_template:
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    else:
        prompt_tokens = tokenizer.encode(content, add_special_tokens=False)

    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, tok in enumerate(prompt_tokens) if tok == inject_id)
    scale = float(av_meta["extraction"]["injection_scale"])
    return prompt_tokens, inject_pos, scale


def generate_av_descriptions(model, tokenizer, examples, av_meta, args,
                             use_chat_template=False, use_normalize=False):
    prompt_tokens, inject_pos, injection_scale = prepare_av_prompt(
        tokenizer, av_meta, use_chat_template=use_chat_template)
    embed_layer = model.get_input_embeddings()
    records = []

    for i, ex in enumerate(examples):
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)
        embeddings = embed_layer(input_ids)
        act = ex["activation"].to(args.device)
        if use_normalize:
            act = normalize_activation(act, injection_scale)
        else:
            act = act * injection_scale
        embeddings[0, inject_pos, :] = act.to(embeddings.dtype)
        attention_mask = torch.ones_like(input_ids)

        generate_kwargs = {
            "inputs_embeds": embeddings.to(model.dtype),
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            output = model.generate(**generate_kwargs)

        records.append({
            "id": ex["id"],
            "category": ex["category"],
            "group": ex["group"],
            "control": ex.get("control"),
            "generated": decode_generated(output, prompt_tokens, tokenizer),
        })
        if (i + 1) % args.log_every == 0:
            print(f"  generated {i+1}/{len(examples)}")

    return records


def make_control_examples(selected, pool, n_controls, seed):
    rng = np.random.RandomState(seed)
    acts = torch.stack([ex["activation"] for ex in pool])
    mean_act = acts.mean(dim=0)
    mean_norm = acts.norm(dim=1).mean()

    controls = []
    selected = selected[:n_controls]
    shuffled_indices = rng.permutation(len(pool))
    for i, ex in enumerate(selected):
        random_vec = torch.randn_like(ex["activation"])
        random_vec = random_vec / random_vec.norm() * mean_norm
        perm = torch.tensor(rng.permutation(ex["activation"].shape[0]), dtype=torch.long)
        shuffled = pool[shuffled_indices[i % len(pool)]]["activation"]
        controls.extend([
            {**ex, "id": ex["id"] + "::mean", "control": "mean", "activation": mean_act},
            {**ex, "id": ex["id"] + "::random", "control": "random", "activation": random_vec},
            {**ex, "id": ex["id"] + "::permuted", "control": "permuted",
             "activation": ex["activation"][perm]},
            {**ex, "id": ex["id"] + "::shuffled", "control": "shuffled",
             "activation": shuffled},
        ])
    return controls


def sample_eval_examples(examples, n, seed):
    rng = random.Random(seed)
    if n >= len(examples):
        return list(examples)
    return rng.sample(examples, n)


def fit_vectorizer(target_descs, generated_descs):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    vectorizer.fit(target_descs + generated_descs)
    return vectorizer


def sample_distractors(index, examples, k, rng, mode):
    true = examples[index]
    candidates = [i for i, ex in enumerate(examples) if i != index]
    if mode == "category":
        preferred = [i for i in candidates if examples[i]["category"] == true["category"]]
    elif mode == "group":
        preferred = [i for i in candidates if examples[i]["group"] == true["group"]]
    else:
        preferred = []

    rng.shuffle(preferred)
    chosen = preferred[:k]
    if len(chosen) < k:
        rest = [i for i in candidates if i not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[:k - len(chosen)])
    return chosen[:k]


def forced_choice_metrics(records, selected, pool, vectorizer, args):
    pool_index = {ex["id"]: i for i, ex in enumerate(pool)}
    desc_matrix = vectorizer.transform([ex["description"] for ex in pool])
    query_matrix = vectorizer.transform([rec["generated"] for rec in records])
    rng = random.Random(args.seed + 17)

    top1 = 0
    top3 = 0
    mrr = 0.0
    category_top1 = 0
    by_control = {}
    rows = []
    for row_idx, rec in enumerate(records):
        true_id = rec["id"].split("::", 1)[0]
        if true_id not in pool_index:
            continue
        true_idx = pool_index[true_id]
        distractors = sample_distractors(
            true_idx, pool, args.distractors, rng, args.distractor_mode)
        candidate_indices = [true_idx] + distractors
        sims = cosine_similarity(query_matrix[row_idx], desc_matrix[candidate_indices])[0]
        order = np.argsort(-sims)
        rank = int(np.where(order == 0)[0][0]) + 1
        pred_idx = candidate_indices[int(order[0])]
        top1 += rank == 1
        top3 += rank <= min(3, len(candidate_indices))
        mrr += 1.0 / rank
        category_top1 += pool[pred_idx]["category"] == pool[true_idx]["category"]
        control = rec.get("control")
        if control:
            stats = by_control.setdefault(
                control, {"n": 0, "top1": 0, "top3": 0, "mrr": 0.0})
            stats["n"] += 1
            stats["top1"] += rank == 1
            stats["top3"] += rank <= min(3, len(candidate_indices))
            stats["mrr"] += 1.0 / rank
        rows.append({
            "id": rec["id"],
            "control": control,
            "rank": rank,
            "top_score": float(sims[order[0]]),
            "true_score": float(sims[0]),
            "pred_category": pool[pred_idx]["category"],
            "true_category": pool[true_idx]["category"],
        })

    n = max(1, len(rows))
    result = {
        "n": len(rows),
        "top1": top1 / n,
        "top3": top3 / n,
        "mrr": mrr / n,
        "category_top1": category_top1 / n,
        "records": rows,
    }
    if by_control:
        result["by_control"] = {
            name: {
                "n": stats["n"],
                "top1": stats["top1"] / stats["n"],
                "top3": stats["top3"] / stats["n"],
                "mrr": stats["mrr"] / stats["n"],
            }
            for name, stats in sorted(by_control.items())
        }
    return result


def knn_and_random_baselines(records, selected, pool, vectorizer, seed):
    pool_ids = [ex["id"] for ex in pool]
    pool_index = {text_id: i for i, text_id in enumerate(pool_ids)}
    acts = torch.stack([ex["activation"] for ex in pool]).float()
    acts = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
    target_descs = [ex["description"] for ex in pool]
    target_matrix = vectorizer.transform(target_descs)
    gen_matrix = vectorizer.transform([rec["generated"] for rec in records])
    rng = random.Random(seed + 31)

    nla_sims = []
    knn_sims = []
    random_same_cat_sims = []
    for i, rec in enumerate(records):
        true_id = rec["id"]
        true_idx = pool_index[true_id]
        target_vec = target_matrix[true_idx]
        nla_sims.append(float(cosine_similarity(gen_matrix[i], target_vec)[0, 0]))

        sims = torch.mv(acts, acts[true_idx])
        sims[true_idx] = -float("inf")
        nn_idx = int(torch.argmax(sims).item())
        knn_sims.append(float(cosine_similarity(target_matrix[nn_idx], target_vec)[0, 0]))

        same_cat = [
            j for j, ex in enumerate(pool)
            if j != true_idx and ex["category"] == pool[true_idx]["category"]
        ]
        if not same_cat:
            same_cat = [j for j in range(len(pool)) if j != true_idx]
        rand_idx = rng.choice(same_cat)
        random_same_cat_sims.append(
            float(cosine_similarity(target_matrix[rand_idx], target_vec)[0, 0]))

    return {
        "nla_vs_target_mean": float(np.mean(nla_sims)),
        "knn_desc_vs_target_mean": float(np.mean(knn_sims)),
        "random_same_category_vs_target_mean": float(np.mean(random_same_cat_sims)),
        "nla_minus_knn": float(np.mean(nla_sims) - np.mean(knn_sims)),
        "nla_minus_random_same_category": (
            float(np.mean(nla_sims) - np.mean(random_same_cat_sims))
        ),
    }


def load_value_head(ar_adapter, device):
    """Load value_head.safetensors from adapter path or HF."""
    resolved = resolve_path(ar_adapter)
    vh_path = None
    if isinstance(resolved, Path) and (resolved / "value_head.safetensors").exists():
        vh_path = resolved / "value_head.safetensors"
    else:
        try:
            from huggingface_hub import hf_hub_download
            vh_path = Path(hf_hub_download(str(ar_adapter), "value_head.safetensors"))
        except Exception:
            pass
    if not vh_path:
        return None
    from safetensors import safe_open
    with safe_open(str(vh_path), framework="pt") as f:
        vh_weight = f.get_tensor("weight")
    d = vh_weight.shape[0]
    value_head = torch.nn.Linear(d, d, bias=False, dtype=vh_weight.dtype)
    value_head.weight = torch.nn.Parameter(vh_weight)
    value_head = value_head.to(device).eval()
    print(f"  Loaded value_head [{d}x{d}]")
    return value_head


def is_value_head_only(ar_adapter):
    """Check if AR checkpoint is value_head-only (no full model saved)."""
    resolved = resolve_path(ar_adapter)
    if not isinstance(resolved, Path):
        return False
    has_vh = (resolved / "value_head.safetensors").exists()
    has_model = any(resolved.glob("model*.safetensors"))
    return has_vh and not has_model


def prepare_ar(ar_adapter, model_key, device):
    ar_meta = load_nla_meta(ar_adapter)
    peft = is_peft_adapter(ar_adapter)
    vh_only = is_value_head_only(ar_adapter)

    if vh_only:
        # Value_head-only checkpoint: load base model, truncate, apply value_head
        model_name = MODELS[model_key]
        extraction_layer = int(ar_meta.get("extraction_layer_index", 20))
        target_layers = extraction_layer + 1
        print(f"  Loading base model {model_name} (truncating to {target_layers} layers)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        inner = model.model if hasattr(model, "model") else model
        if len(inner.layers) > target_layers:
            inner.layers = inner.layers[:target_layers]
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        model.lm_head = torch.nn.Identity()
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        model, tokenizer = load_adapter(ar_adapter, model_key)

    value_head = load_value_head(ar_adapter, device)

    # For non-PEFT full models (Anthropic-style): set norm to Identity
    if not peft and not vh_only:
        inner = model.model if hasattr(model, "model") else model
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break

    blocks = get_blocks(model)
    injection_char = ar_meta["tokens"]["injection_char"]
    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    template = ar_meta["prompt_templates"]["ar"]
    extraction_layer = int(ar_meta.get("extraction_layer_index", 20))
    return {
        "model": model,
        "tokenizer": tokenizer,
        "blocks": blocks,
        "inject_id": inject_id,
        "injection_char": injection_char,
        "template": template,
        "layer": extraction_layer,
        "device": device,
        "value_head": value_head,
    }


def ar_reconstruct_batch(ar, descriptions, batch_size):
    model = ar["model"]
    tokenizer = ar["tokenizer"]
    blocks = ar["blocks"]
    layer = ar["layer"]
    device = ar["device"]
    inject_id = ar["inject_id"]
    value_head = ar.get("value_head")

    reconstructed = []
    for start in range(0, len(descriptions), batch_size):
        batch_descs = descriptions[start:start + batch_size]
        prompts = [
            ar["template"]
            .replace("{explanation}", desc)
            .replace("{injection_char}", ar["injection_char"])
            for desc in batch_descs
        ]
        token_lists = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        max_len = max(len(toks) for toks in token_lists)
        input_ids = torch.full(
            (len(token_lists), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, toks in enumerate(token_lists):
            input_ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
            attention_mask[i, :len(toks)] = 1

        layer_outputs = {}

        def hook(_mod, _inp, out):
            layer_outputs["h"] = out[0] if isinstance(out, tuple) else out

        handle = blocks[layer].register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        handle.remove()

        hidden = layer_outputs["h"]
        for i, toks in enumerate(token_lists):
            pos = len(toks) - 1
            h = hidden[i, pos].detach()
            if value_head is not None:
                h = value_head(h.unsqueeze(0)).squeeze(0)
            reconstructed.append(h.cpu().float())

    return torch.stack(reconstructed)


def ar_roundtrip_metrics(ar_adapter, records, selected, args):
    print("Loading AR adapter for round-trip scoring...")
    ar = prepare_ar(ar_adapter, args.model, args.device)
    targets = torch.stack([ex["activation"] for ex in selected]).float()
    generated = [rec["generated"] for rec in records]
    target_descs = [ex["description"] for ex in selected]
    shuffled = generated[1:] + generated[:1]
    generic = [GENERIC_DESCRIPTIONS[i % len(GENERIC_DESCRIPTIONS)] for i in range(len(records))]

    metrics = {}
    for name, descs in [
        ("av_generated", generated),
        ("target_description", target_descs),
        ("shuffled_av_generated", shuffled),
        ("generic_description", generic),
    ]:
        rec = ar_reconstruct_batch(ar, descs, args.ar_batch_size)
        cos = torch.nn.functional.cosine_similarity(rec, targets, dim=1)
        metrics[name] = {
            "mean_cosine": float(cos.mean()),
            "median_cosine": float(cos.median()),
            "std_cosine": float(cos.std(unbiased=False)),
        }

    metrics["av_minus_shuffled"] = (
        metrics["av_generated"]["mean_cosine"]
        - metrics["shuffled_av_generated"]["mean_cosine"]
    )
    metrics["av_minus_generic"] = (
        metrics["av_generated"]["mean_cosine"]
        - metrics["generic_description"]["mean_cosine"]
    )

    del ar["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def summarize(result):
    print("\n=== Stress Test Summary ===")
    print(f"Examples: {result['data']['n_eval']} / pool {result['data']['usable_examples']}")
    print(
        "Forced choice real: "
        f"top1={result['forced_choice_real']['top1']:.3f} "
        f"top3={result['forced_choice_real']['top3']:.3f} "
        f"mrr={result['forced_choice_real']['mrr']:.3f}"
    )
    if "forced_choice_controls" in result:
        print(
            "Forced choice controls: "
            f"top1={result['forced_choice_controls']['top1']:.3f} "
            f"top3={result['forced_choice_controls']['top3']:.3f}"
        )
    base = result["baselines"]
    print(
        "TF-IDF target similarity: "
        f"NLA={base['nla_vs_target_mean']:.3f} "
        f"kNN={base['knn_desc_vs_target_mean']:.3f} "
        f"random_same_cat={base['random_same_category_vs_target_mean']:.3f}"
    )
    if "ar_roundtrip" in result:
        ar = result["ar_roundtrip"]
        print(
            "AR round-trip cosine: "
            f"AV={ar['av_generated']['mean_cosine']:.3f} "
            f"target={ar['target_description']['mean_cosine']:.3f} "
            f"shuffled={ar['shuffled_av_generated']['mean_cosine']:.3f} "
            f"generic={ar['generic_description']['mean_cosine']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stress-test whether Qwen L20 NLA outputs carry activation signal")
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", default="output/nla-qwen25-7b-L20-av-v3")
    parser.add_argument("--ar-adapter", default="output/nla-qwen25-7b-L20-ar")
    parser.add_argument("--activations", default="corpus/activations/qwen25-7b_L20.pt")
    parser.add_argument("--descriptions", default=None)
    parser.add_argument("--output", default="evaluation/qwen_l20_stress.json")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n-controls", type=int, default=25)
    parser.add_argument("--distractors", type=int, default=9)
    parser.add_argument("--distractor-mode", choices=["group", "category", "random"],
                        default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ar-batch-size", type=int, default=4)
    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--include-unsafe", action="store_true",
                        help="Include YAML unsafe:true categories. Default skips them.")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.descriptions is None:
        args.descriptions = str(default_description_path(args.av_adapter))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cat_meta = load_category_meta()
    pool, data_info = load_examples(args, cat_meta)
    if len(pool) < 2:
        raise RuntimeError("Need at least two usable examples for stress testing")
    selected = sample_eval_examples(pool, args.n, args.seed)
    data_info["n_eval"] = len(selected)
    data_info["unsafe_included"] = bool(args.include_unsafe)

    print(f"Loaded {data_info['usable_examples']} usable examples")
    print(f"Selected {len(selected)} examples for AV generation")
    if data_info["skipped_unsafe"]:
        print(f"Skipped {data_info['skipped_unsafe']} unsafe examples")

    peft_av = is_peft_adapter(args.av_adapter)
    av_meta = load_nla_meta(args.av_adapter)
    # Detect pipeline from nla_meta if available, else infer from PEFT status
    training_meta = av_meta.get("training", {})
    use_norm = training_meta.get("injection_mode") == "normalize" or not peft_av
    use_chat = training_meta.get("chat_template", not peft_av)
    print(f"Loading AV adapter... (peft={peft_av}, chat_template={use_chat}, normalize={use_norm})")
    av_model, av_tokenizer = load_adapter(args.av_adapter, args.model)
    real_records = generate_av_descriptions(
        av_model, av_tokenizer, selected, av_meta, args,
        use_chat_template=use_chat, use_normalize=use_norm)

    control_examples = make_control_examples(selected, pool, args.n_controls, args.seed)
    control_records = []
    if control_examples:
        print(f"Generating {len(control_examples)} control descriptions")
        control_records = generate_av_descriptions(
            av_model, av_tokenizer, control_examples, av_meta, args,
            use_chat_template=use_chat, use_normalize=use_norm)

    del av_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vectorizer = fit_vectorizer(
        [ex["description"] for ex in pool],
        [rec["generated"] for rec in real_records + control_records],
    )

    result = {
        "config": {
            "model": args.model,
            "av_adapter": str(resolve_path(args.av_adapter)),
            "ar_adapter": None if args.skip_ar else str(resolve_path(args.ar_adapter)),
            "distractors": args.distractors,
            "distractor_mode": args.distractor_mode,
            "seed": args.seed,
            "unsafe_included": bool(args.include_unsafe),
        },
        "data": data_info,
        "forced_choice_real": forced_choice_metrics(
            real_records, selected, pool, vectorizer, args),
        "baselines": knn_and_random_baselines(
            real_records, selected, pool, vectorizer, args.seed),
    }

    if control_records:
        result["forced_choice_controls"] = forced_choice_metrics(
            control_records, control_examples, pool, vectorizer, args)

    if not args.skip_ar:
        result["ar_roundtrip"] = ar_roundtrip_metrics(args.ar_adapter, real_records, selected, args)

    # Keep the main report content-light: IDs/categories/metrics, no corpus text.
    result["forced_choice_real"]["records"] = result["forced_choice_real"]["records"][:20]
    if "forced_choice_controls" in result:
        result["forced_choice_controls"]["records"] = (
            result["forced_choice_controls"]["records"][:20]
        )

    out_path = Path(resolve_path(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    summarize(result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
