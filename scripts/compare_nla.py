#!/usr/bin/env python3
"""
Compare our NLA-at-home AV against Anthropic's kitft NLA.
Feed the same direction vectors to both, compare descriptions.

Usage:
  python3 scripts/compare_nla.py \
    --ours output/nla-qwen25-7b-L20-av-v2 \
    --directions ~/playground/karma-electric/data/directions
"""
import torch
import yaml
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from generation_utils import decode_generated

device = torch.device("cuda")

DIRECTION_FILES = {
    "valence": "valence/qwen25-7b_vedana_L20_unit.pt",
    "arousal": "arousal/qwen25-7b_arousal_L17_unit.pt",
    "agency": "agency/qwen25-7b_agency_L15_unit.pt",
    "continuity": "continuity/qwen25-7b_continuity_L19_unit.pt",
    "frame-integrity": "frame-integrity/qwen25-7b_frame_L26_unit.pt",
    "intimacy": "intimacy/qwen25-7b_intimacy_L20_unit.pt",
    "restraint": "restraint/qwen25-7b_restraint_L18_unit.pt",
    "assistant": "assistant/qwen25-7b_assistant_L19_unit.pt",
}


def load_nla(model_name_or_path, adapter_path=None):
    """Load an NLA model. If adapter_path is given, load base + LoRA."""
    meta_path = Path(model_name_or_path) / "nla_meta.yaml"
    if not meta_path.exists():
        from huggingface_hub import hf_hub_download
        meta_path = hf_hub_download(model_name_or_path, "nla_meta.yaml")
    meta = yaml.safe_load(open(meta_path))

    templates = meta["prompt_templates"]
    template = templates.get("av") or templates.get("actor")
    injection_char = meta["tokens"]["injection_char"]
    injection_scale = meta["extraction"]["injection_scale"]

    if adapter_path:
        base_model = "Qwen/Qwen2.5-7B-Instruct"
        print(f"  Loading base {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        print(f"  Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"  Loading {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

    model.eval()
    return model, tokenizer, template, injection_char, injection_scale


def verbalize(model, tokenizer, template, injection_char, injection_scale,
              direction, sign=+1, activation_norm=1.0):
    """Generate NLA description for a direction vector.

    Direction vectors are unit norm. Real activations have norm ~activation_norm.
    We scale the direction to match the training distribution before applying
    the injection scale.
    """
    prompt = template.replace("{injection_char}", injection_char)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(tokens) if t == inject_id)

    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(input_ids)

    norm = direction.norm()
    if not torch.isfinite(direction).all() or not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError("direction must be finite and non-zero")

    dir_normalized = direction / norm
    vec = dir_normalized.to(device).float() * sign * activation_norm
    embeddings[0, inject_pos, :] = vec * injection_scale
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=embeddings.to(model.dtype),
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    return decode_generated(output, tokens, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="Path to our NLA adapter")
    parser.add_argument("--anthropic", default="kitft/nla-qwen2.5-7b-L20-av",
                        help="Anthropic NLA model")
    parser.add_argument("--directions", required=True, help="Path to directions/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)

    print("Loading Anthropic NLA...")
    anth_model, anth_tok, anth_tmpl, anth_char, anth_scale = load_nla(args.anthropic)

    print("\nLoading our NLA...")
    ours_model, ours_tok, ours_tmpl, ours_char, ours_scale = load_nla(
        args.ours, adapter_path=args.ours)

    directions_dir = Path(args.directions)

    # Compute mean activation norm for proper direction scaling
    act_path = Path(args.ours).parent.parent / "corpus" / "activations" / "qwen25-7b_L20.pt"
    if act_path.exists():
        act_data = torch.load(act_path, weights_only=True, map_location="cpu")
        mean_norm = float(act_data["activations"].norm(dim=1).mean())
        print(f"\nMean activation norm: {mean_norm:.1f}")
        print(f"Direction vectors will be scaled by {mean_norm:.1f} before injection")
    else:
        mean_norm = 122.8
        print(f"\nUsing default activation norm: {mean_norm}")

    print(f"\n{'='*70}")
    print(f"Direction comparison: Anthropic NLA vs NLA-at-home")
    print(f"{'='*70}")

    for name, rel_path in DIRECTION_FILES.items():
        path = directions_dir / rel_path
        if not path.exists():
            print(f"\n[{name}] direction file not found, skipping")
            continue

        direction = torch.load(path, weights_only=True, map_location="cpu")
        if isinstance(direction, dict):
            direction = direction.get("direction", list(direction.values())[0])
        direction = direction.float()

        d_model = 3584
        if direction.shape[0] != d_model:
            print(f"\n[{name}] dimension mismatch ({direction.shape[0]} vs {d_model}), skipping")
            continue
        if not torch.isfinite(direction).all() or direction.norm() < 1e-8:
            print(f"\n[{name}] invalid or zero-norm direction, skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"Direction: {name} (+)")
        print(f"{'─'*70}")

        anth_desc = verbalize(anth_model, anth_tok, anth_tmpl, anth_char, anth_scale,
                              direction, sign=+1, activation_norm=mean_norm)
        ours_desc = verbalize(ours_model, ours_tok, ours_tmpl, ours_char, ours_scale,
                              direction, sign=+1, activation_norm=mean_norm)

        print(f"  Anthropic: {anth_desc[:300]}")
        print(f"  Ours:      {ours_desc[:300]}")

        print(f"\nDirection: {name} (−)")
        anth_neg = verbalize(anth_model, anth_tok, anth_tmpl, anth_char, anth_scale,
                             direction, sign=-1, activation_norm=mean_norm)
        ours_neg = verbalize(ours_model, ours_tok, ours_tmpl, ours_char, ours_scale,
                             direction, sign=-1, activation_norm=mean_norm)

        print(f"  Anthropic: {anth_neg[:300]}")
        print(f"  Ours:      {ours_neg[:300]}")


if __name__ == "__main__":
    main()
