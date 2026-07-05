#!/usr/bin/env python3
"""
PCA on activation vectors, then NLA-describe each principal component.

1. Load 5557 activation vectors (L20, 3584-dim)
2. PCA → top N components (directions of maximum variance)
3. Inject each component (+ and -) into both NLAs
4. Save descriptions + eigenvalues + cosines with known axes

Usage:
  python3 scripts/pca_nla_manifold.py --n-components 50
  python3 scripts/pca_nla_manifold.py --n-components 50 --nla anthropic
"""
import torch
import json
import yaml
import argparse
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda")
INJECTION_SCALE = 150.0

KNOWN_AXES = {
    "vchip": "~/tone-experiment/results/vchip-directions/qwen25-7b_L14_unit.pt",
    "valence": "~/tone-experiment/results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt",
    "frame_integrity": "~/tone-experiment/results/frame-integrity-directions/qwen25-7b_frame_L26_unit.pt",
    "arousal": "~/tone-experiment/results/arousal-directions/qwen25-7b_arousal_L17_unit.pt",
    "agency": "~/tone-experiment/results/agency-directions/qwen25-7b_agency_L15_unit.pt",
    "continuity": "~/tone-experiment/results/continuity-directions/qwen25-7b_continuity_L19_unit.pt",
    "intimacy": "~/tone-experiment/results/intimacy-directions/qwen25-7b_intimacy_L20_unit.pt",
    "restraint": "~/tone-experiment/results/restraint-directions/qwen25-7b_restraint_L18_unit.pt",
}

MODELS = {
    "anthropic": {
        "path": "~/nla-qwen25-7b-av",
        "type": "merged",
    },
    "ours": {
        "path": "~/playground/nla-at-home/output/nla-qwen25-7b-L20-av-all-sonnet",
        "type": "lora",
    },
}


def run_pca(act_path, n_components):
    data = torch.load(act_path, weights_only=True, map_location="cpu")
    activations = data["activations"].float()
    ids = data["ids"]
    print(f"Activations: {activations.shape}")

    mean = activations.mean(dim=0)
    centered = activations - mean

    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S ** 2) / (len(activations) - 1)
    total_var = eigenvalues.sum()
    explained = eigenvalues[:n_components] / total_var

    components = Vt[:n_components]
    components = components / components.norm(dim=1, keepdim=True)

    print(f"Top {n_components} components explain {explained.sum()*100:.1f}% of variance")
    print(f"PC1: {explained[0]*100:.2f}%, PC10: {explained[9]*100:.2f}%, PC50: {explained[min(49,n_components-1)]*100:.3f}%")

    return components, eigenvalues[:n_components], explained, mean


def cosine_with_known_axes(components):
    known = {}
    for name, path in KNOWN_AXES.items():
        d = torch.load(Path(path).expanduser(), weights_only=True, map_location="cpu")
        if isinstance(d, dict):
            d = d.get("direction", d.get("unit", list(d.values())[0]))
        known[name] = d.squeeze().float()

    cosines = {}
    for i, pc in enumerate(components):
        pc_cos = {}
        for name, axis in known.items():
            c = torch.nn.functional.cosine_similarity(
                pc.unsqueeze(0), axis.unsqueeze(0)
            ).item()
            pc_cos[name] = round(c, 4)
        cosines[f"PC{i+1}"] = pc_cos
    return cosines


def load_nla(model_info, tokenizer, base_model_name):
    path = Path(model_info["path"]).expanduser()
    meta_path = path / "nla_meta.yaml"
    meta = yaml.safe_load(open(meta_path)) if meta_path.exists() else {}

    if model_info["type"] == "merged":
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(path))

    model.eval()

    injection_char = meta.get("tokens", {}).get("injection_char", "㈎")
    template = meta.get("prompt_templates", {}).get("av", "")
    if not template:
        template = (
            "You are a meticulous AI researcher conducting an important investigation "
            "into activation vectors from a language model. Your overall task is to "
            "describe the semantic content of that activation vector.\n\n"
            "We will pass the vector enclosed in <concept> tags into your context. "
            "You must then produce an explanation for the vector, enclosed within "
            "<explanation> tags. The explanation consists of 2-3 text snippets "
            "describing that vector.\n\n"
            "Here is the vector:\n\n"
            "<concept>{injection_char}</concept>\n\n"
            "Please provide an explanation.\n\n"
            "<explanation>"
        )

    content = template.replace("{injection_char}", injection_char)
    use_chat = meta.get("training", {}).get("chat_template", True)

    if use_chat:
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    else:
        prompt_tokens = tokenizer.encode(content, add_special_tokens=False)

    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == injection_token_id)

    scale = float(meta.get("extraction", {}).get("injection_scale", INJECTION_SCALE))

    return model, prompt_tokens, inject_pos, scale


def generate_description(model, tokenizer, prompt_tokens, inject_pos, direction, scale):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(input_ids)

    d = direction.to(device).float()
    d = d / d.norm() * scale
    embeddings[0, inject_pos, :] = d.to(embeddings.dtype)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=embeddings.to(model.dtype),
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0][len(prompt_tokens):], skip_special_tokens=True)
    if "</explanation>" in generated:
        generated = generated[:generated.index("</explanation>")]
    return generated.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--activations", default="corpus/activations/qwen25-7b_L20.pt")
    parser.add_argument("--nla", default="both", choices=["anthropic", "ours", "both"])
    parser.add_argument("--output", default="data/pca_manifold.json")
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    act_path = repo / args.activations
    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== PCA ===")
    components, eigenvalues, explained, mean = run_pca(act_path, args.n_components)

    print("\n=== Cosines with known axes ===")
    cosines = cosine_with_known_axes(components)
    for pc_name, pc_cos in list(cosines.items())[:10]:
        best = max(pc_cos.items(), key=lambda x: abs(x[1]))
        print(f"  {pc_name}: best match = {best[0]} ({best[1]:+.3f})")

    print("\n=== NLA descriptions ===")
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    nla_names = ["anthropic", "ours"] if args.nla == "both" else [args.nla]
    results = {
        "pca": {
            "n_components": args.n_components,
            "explained_variance": [round(v.item(), 6) for v in explained],
            "total_explained": round(explained.sum().item(), 4),
        },
        "cosines_with_known_axes": cosines,
        "descriptions": {},
    }

    for nla_name in nla_names:
        model_info = MODELS[nla_name]
        print(f"\nLoading NLA: {nla_name}")
        model, prompt_tokens, inject_pos, scale = load_nla(
            model_info, tokenizer, base_model_name)
        print(f"  Prompt: {len(prompt_tokens)} tokens, inject at {inject_pos}")

        for i in range(args.n_components):
            pc = components[i]
            t0 = time.time()

            desc_pos = generate_description(
                model, tokenizer, prompt_tokens, inject_pos, pc, scale)
            desc_neg = generate_description(
                model, tokenizer, prompt_tokens, inject_pos, -pc, scale)

            pc_key = f"PC{i+1}"
            results["descriptions"].setdefault(pc_key, {})[nla_name] = {
                "positive": desc_pos,
                "negative": desc_neg,
            }

            best_axis = max(cosines[pc_key].items(), key=lambda x: abs(x[1]))
            elapsed = time.time() - t0
            print(f"  PC{i+1} ({explained[i]*100:.2f}%, best={best_axis[0]} {best_axis[1]:+.3f}) [{elapsed:.1f}s]")
            print(f"    (+) {desc_pos[:150]}")
            print(f"    (-) {desc_neg[:150]}")

            if (i + 1) % 10 == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  [checkpoint saved at PC{i+1}]")

        del model
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY: Top 20 PCs ===")
    for i in range(min(20, args.n_components)):
        pc_key = f"PC{i+1}"
        best = max(cosines[pc_key].items(), key=lambda x: abs(x[1]))
        print(f"\n{pc_key} ({explained[i]*100:.2f}% var, closest axis: {best[0]} {best[1]:+.3f})")
        for nla_name in nla_names:
            d = results["descriptions"][pc_key][nla_name]
            print(f"  {nla_name} (+): {d['positive'][:200]}")
            print(f"  {nla_name} (-): {d['negative'][:200]}")


if __name__ == "__main__":
    main()
