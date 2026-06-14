#!/usr/bin/env python3
"""
Compare Anthropic NLA vs our NLA descriptions for known axis directions.

For each direction (valence, frame integrity, arousal, agency, continuity, intimacy),
inject the unit vector into both NLAs at L20 and compare the descriptions.

Usage:
  python3 scripts/compare_axis_descriptions.py
"""
import torch
import json
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu"))
INJECTION_SCALE = 150.0
# Canonical Qwen L20 adapters (anicka/nla-qwen2.5-7b-L20-av-v2 [GRPO hard-neg],
# kitft/nla-qwen2.5-7b-L20-av) use the NORMALIZE convention, verified empirically on known
# activations: inject normalize(vec) -> INJECTION_SCALE, with a DEPTH-aware prompt (a
# single-layer adapter sees every vector as if from its own layer, L20 -> 73%) and
# add_special_tokens=True. This matches brain_in_jar_qwen.py / space/app.py. (raw*scale +
# a non-depth prompt yields garbage on -v2 and prompt-leak on kitft.) The older single-layer
# SFT adapters (train_av.py) instead expect raw*scale + a non-depth prompt.
DEPTH_PCT = round(100 * (20 + 0.5) / 28)  # 73 — the adapter's own layer

DIRECTIONS = {
    "vchip_honest_denial": "~/tone-experiment/results/vchip-directions/qwen25-7b_L14_unit.pt",
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
    # DEPTH-aware template (the canonical convention). -v2's meta carries no template, so we
    # build it here; depth is the adapter's own layer (DEPTH_PCT), not the direction's source.
    template = (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. You must then produce "
        "an explanation for the vector, enclosed within <explanation> tags. The "
        "explanation consists of 2-3 text snippets describing that vector.\n\n"
        f"Here is the vector from depth {DEPTH_PCT}% of the network:\n\n"
        f"<concept>{{injection_char}}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )

    content = template.replace("{injection_char}", injection_char)
    # Raw template encode WITH special tokens (BOS), no chat wrap — matches
    # brain_in_jar_qwen.py's verbalize(). Chat-wrapping shifts the injection off-distribution.
    prompt_tokens = tokenizer.encode(content, add_special_tokens=True)

    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == injection_token_id)

    scale = float(meta.get("extraction", {}).get("injection_scale", INJECTION_SCALE))

    return model, prompt_tokens, inject_pos, scale


def generate_description(model, tokenizer, prompt_tokens, inject_pos, activation, scale):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(input_ids)

    # NORMALIZE convention: scale the vector to norm=injection_scale (a unit direction -> 150).
    act_scaled = activation.to(device).float()
    act_scaled = act_scaled / act_scaled.norm().clamp_min(1e-12) * scale
    embeddings[0, inject_pos, :] = act_scaled.to(embeddings.dtype)

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
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for nla_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading NLA: {nla_name} ({model_info['path']})")
        print(f"{'='*60}")

        model, prompt_tokens, inject_pos, scale = load_nla(
            model_info, tokenizer, base_model_name)
        print(f"  Prompt: {len(prompt_tokens)} tokens, inject at {inject_pos}, scale={scale}")

        for axis_name, dir_path in DIRECTIONS.items():
            direction = torch.load(
                Path(dir_path).expanduser(), weights_only=True,
                map_location="cpu")
            if isinstance(direction, dict):
                direction = direction.get("direction", direction.get("unit", list(direction.values())[0]))
            if direction.dim() > 1:
                direction = direction.squeeze()

            # Positive direction
            desc_pos = generate_description(
                model, tokenizer, prompt_tokens, inject_pos, direction, scale)

            # Negative direction
            desc_neg = generate_description(
                model, tokenizer, prompt_tokens, inject_pos, -direction, scale)

            results.setdefault(axis_name, {})[nla_name] = {
                "positive": desc_pos,
                "negative": desc_neg,
            }

            print(f"\n  --- {axis_name} ({nla_name}) ---")
            print(f"  (+) {desc_pos[:200]}")
            print(f"  (-) {desc_neg[:200]}")

        del model
        torch.cuda.empty_cache()

    # Save results
    out_path = Path("data/axis_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print side-by-side summary
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    for axis_name in DIRECTIONS:
        print(f"\n### {axis_name.upper().replace('_', ' ')}")
        for nla_name in MODELS:
            r = results[axis_name][nla_name]
            print(f"\n  {nla_name} (+): {r['positive'][:300]}")
            print(f"  {nla_name} (-): {r['negative'][:300]}")


if __name__ == "__main__":
    main()
