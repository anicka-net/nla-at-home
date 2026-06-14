#!/usr/bin/env python3
"""
NLA Stress Test: cross-validate AV/AR pairs.

Supports two AR types:
- LoRA adapter (our AR): hooks at extraction layer
- Full model + value_head (Anthropic AR): truncated model + linear head

Usage:
  python3 experiments/stress_test_nla.py \
    --our-av ../output/nla-qwen25-7b-L20-av-twin-grpo \
    --our-ar ../output/nla-qwen25-7b-L20-ar-twin \
    --anthropic-ar ~/.cache/huggingface/hub/models--kitft--nla-qwen2.5-7b-L20-ar/snapshots/... \
    --activations ../corpus/activations/qwen25-7b_L20.pt \
    --n-samples 50
"""
import torch
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from generation_utils import decode_generated

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INJECTION_CHAR = "㈎"
INJECTION_SCALE = 150.0


def load_av(adapter_path, device):
    meta = yaml.safe_load(open(Path(adapter_path) / "nla_meta.yaml"))
    template = meta["prompt_templates"]["av"]
    content = template.replace("{injection_char}", INJECTION_CHAR)

    has_adapter = (Path(adapter_path) / "adapter_config.json").exists()

    if has_adapter:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    inject_id = tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)

    model.eval()
    return model, tokenizer, prompt_tokens, inject_pos


class LoraAR:
    def __init__(self, checkpoint_path, device):
        meta = yaml.safe_load(open(Path(checkpoint_path) / "nla_meta.yaml"))
        self.extraction_layer = int(meta.get("extraction_layer_index", 20))
        self.ar_template = meta.get("prompt_templates", {}).get("ar",
            "Summary of the following text: <text>{explanation}</text> <summary>{injection_char}")
        self.ar_template = self.ar_template.replace("{injection_char}", INJECTION_CHAR)

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.model = PeftModel.from_pretrained(base, checkpoint_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.name = "LoRA AR"

    def score(self, description, target_act):
        inject_id = self.tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)[0]
        prompt = self.ar_template.replace("{explanation}", description) + INJECTION_CHAR
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        inject_pos = next((i for i, t in enumerate(tokens) if t == inject_id), len(tokens) - 1)
        input_ids = torch.tensor([tokens], device=self.device)

        inner = self.model.model if hasattr(self.model, 'model') else self.model
        blocks = inner.model.layers

        layer_out = {}
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out['h'] = h
        handle = blocks[self.extraction_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        handle.remove()

        hidden = layer_out['h'][0, inject_pos].float().cpu()
        return torch.nn.functional.cosine_similarity(
            hidden.unsqueeze(0), target_act.float().unsqueeze(0)).item()

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


class ValueHeadAR:
    def __init__(self, checkpoint_path, device):
        meta = yaml.safe_load(open(Path(checkpoint_path) / "nla_meta.yaml"))
        self.extraction_layer = int(meta.get("extraction_layer_index", 20))
        self.ar_template = meta.get("prompt_templates", {}).get("ar",
            "Summary of the following text: <text>{explanation}</text> <summary>{injection_char}")
        self.ar_template = self.ar_template.replace("{injection_char}", INJECTION_CHAR)

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        inner = self.model.model if hasattr(self.model, "model") else self.model
        if len(inner.layers) > self.extraction_layer + 1:
            inner.layers = inner.layers[:self.extraction_layer + 1]
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        self.model.lm_head = torch.nn.Identity()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        from safetensors import safe_open
        vh_path = Path(checkpoint_path) / "value_head.safetensors"
        with safe_open(str(vh_path), framework="pt") as f:
            vh_weight = f.get_tensor("weight")
        self.value_head = torch.nn.Linear(vh_weight.shape[1], vh_weight.shape[0],
                                          bias=False, dtype=vh_weight.dtype)
        self.value_head.weight = torch.nn.Parameter(vh_weight)
        self.value_head = self.value_head.to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.name = "ValueHead AR (Anthropic)"

    def score(self, description, target_act):
        prompt = self.ar_template.replace("{explanation}", description)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)
        with torch.no_grad():
            inner = self.model.model if hasattr(self.model, "model") else self.model
            outputs = inner(input_ids=input_ids, use_cache=False)
            hidden = outputs.last_hidden_state[0, -1]
            reconstructed = self.value_head(hidden.unsqueeze(0)).squeeze(0).float().cpu()
        return torch.nn.functional.cosine_similarity(
            reconstructed.unsqueeze(0), target_act.float().unsqueeze(0)).item()

    def cleanup(self):
        del self.model, self.value_head
        torch.cuda.empty_cache()


def load_ar(checkpoint_path, device):
    has_vh = (Path(checkpoint_path) / "value_head.safetensors").exists()
    has_adapter = (Path(checkpoint_path) / "adapter_config.json").exists()
    if has_adapter:
        return LoraAR(checkpoint_path, device)
    elif has_vh:
        return ValueHeadAR(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown AR format at {checkpoint_path}")


def generate_description(av_model, tokenizer, activation, prompt_tokens, inject_pos, device):
    embed_layer = av_model.get_input_embeddings()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    embeddings = embed_layer(input_ids)
    norm = activation.float().norm().clamp_min(1e-12)
    scaled = activation.to(device) * (INJECTION_SCALE / norm)
    embeddings[0, inject_pos, :] = scaled.to(embeddings.dtype)

    with torch.no_grad():
        output = av_model.generate(
            inputs_embeds=embeddings.to(av_model.dtype),
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True)

    return decode_generated(output, prompt_tokens, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-av", required=True)
    parser.add_argument("--our-ar", required=True)
    parser.add_argument("--anthropic-av", default=None)
    parser.add_argument("--anthropic-ar", default=None)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.RandomState(42)

    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    all_acts = act_data["activations"].float()
    text_ids = act_data["ids"]
    indices = rng.choice(len(all_acts), min(args.n_samples, len(all_acts)), replace=False)
    acts = [all_acts[i] for i in indices]
    ids = [text_ids[i] for i in indices]
    print(f"  {len(acts)} samples")

    # Generate descriptions
    print(f"\nLoading AV: {args.our_av}...")
    av, av_tok, prompt_tokens, inject_pos = load_av(args.our_av, device)
    print("Generating descriptions...")
    descs = []
    for i, act in enumerate(acts):
        descs.append(generate_description(av, av_tok, act, prompt_tokens, inject_pos, device))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(acts)}")
    del av
    torch.cuda.empty_cache()

    # Generate descriptions with Anthropic AV (if provided)
    anth_descs = None
    if args.anthropic_av:
        print(f"\nLoading Anthropic AV: {args.anthropic_av}...")
        anth_av, anth_av_tok, anth_prompt, anth_inject = load_av(args.anthropic_av, device)
        print("Generating Anthropic descriptions...")
        anth_descs = []
        for i, act in enumerate(acts):
            anth_descs.append(generate_description(anth_av, anth_av_tok, act, anth_prompt, anth_inject, device))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(acts)}")
        del anth_av
        torch.cuda.empty_cache()

    # Score with our AR
    print(f"\nLoading our AR: {args.our_ar}...")
    our_ar = load_ar(args.our_ar, device)
    print(f"  Type: {our_ar.name}")
    print("Scoring our descs...")
    our_correct, our_garbage = [], []
    anth_descs_on_our_ar_correct, anth_descs_on_our_ar_garbage = [], []
    for i in range(len(acts)):
        our_correct.append(our_ar.score(descs[i], acts[i]))
        wrong_idx = (i + 1) % len(acts)
        our_garbage.append(our_ar.score(descs[i], acts[wrong_idx]))
        if anth_descs:
            anth_descs_on_our_ar_correct.append(our_ar.score(anth_descs[i], acts[i]))
            anth_descs_on_our_ar_garbage.append(our_ar.score(anth_descs[i], acts[wrong_idx]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(acts)}")
    our_ar.cleanup()

    # Score with Anthropic AR
    anth_correct, anth_garbage = [], []
    anth_descs_on_anth_ar_correct, anth_descs_on_anth_ar_garbage = [], []
    if args.anthropic_ar:
        print(f"\nLoading Anthropic AR: {args.anthropic_ar}...")
        anth_ar = load_ar(args.anthropic_ar, device)
        print(f"  Type: {anth_ar.name}")
        print("Scoring...")
        for i in range(len(acts)):
            anth_correct.append(anth_ar.score(descs[i], acts[i]))
            wrong_idx = (i + 1) % len(acts)
            anth_garbage.append(anth_ar.score(descs[i], acts[wrong_idx]))
            if anth_descs:
                anth_descs_on_anth_ar_correct.append(anth_ar.score(anth_descs[i], acts[i]))
                anth_descs_on_anth_ar_garbage.append(anth_ar.score(anth_descs[i], acts[wrong_idx]))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(acts)}")
        anth_ar.cleanup()

    # Report
    def mean(v): return sum(v) / len(v) if v else 0

    print(f"\n{'='*60}")
    print(f"NLA Stress Test ({len(acts)} samples)")
    print(f"{'='*60}")

    our_delta = mean(our_correct) - mean(our_garbage)
    our_top1 = sum(1 for c, g in zip(our_correct, our_garbage) if c > g)
    print(f"\nOUR AR:")
    print(f"  Correct:  {mean(our_correct):.4f}")
    print(f"  Garbage:  {mean(our_garbage):.4f}")
    print(f"  Delta:    {our_delta:.4f}")
    print(f"  Top-1:    {our_top1}/{len(acts)} ({100*our_top1/len(acts):.0f}%)")

    if anth_correct:
        anth_delta = mean(anth_correct) - mean(anth_garbage)
        anth_top1 = sum(1 for c, g in zip(anth_correct, anth_garbage) if c > g)
        print(f"\nANTHROPIC AR (scoring our descs):")
        print(f"  Correct:  {mean(anth_correct):.4f}")
        print(f"  Garbage:  {mean(anth_garbage):.4f}")
        print(f"  Delta:    {anth_delta:.4f}")
        print(f"  Top-1:    {anth_top1}/{len(acts)} ({100*anth_top1/len(acts):.0f}%)")

    if anth_descs_on_our_ar_correct:
        d = mean(anth_descs_on_our_ar_correct) - mean(anth_descs_on_our_ar_garbage)
        t1 = sum(1 for c, g in zip(anth_descs_on_our_ar_correct, anth_descs_on_our_ar_garbage) if c > g)
        print(f"\nOUR AR (scoring Anthropic descs):")
        print(f"  Correct:  {mean(anth_descs_on_our_ar_correct):.4f}")
        print(f"  Garbage:  {mean(anth_descs_on_our_ar_garbage):.4f}")
        print(f"  Delta:    {d:.4f}")
        print(f"  Top-1:    {t1}/{len(acts)} ({100*t1/len(acts):.0f}%)")

    if anth_descs_on_anth_ar_correct:
        d = mean(anth_descs_on_anth_ar_correct) - mean(anth_descs_on_anth_ar_garbage)
        t1 = sum(1 for c, g in zip(anth_descs_on_anth_ar_correct, anth_descs_on_anth_ar_garbage) if c > g)
        print(f"\nANTHROPIC AR (scoring Anthropic descs):")
        print(f"  Correct:  {mean(anth_descs_on_anth_ar_correct):.4f}")
        print(f"  Garbage:  {mean(anth_descs_on_anth_ar_garbage):.4f}")
        print(f"  Delta:    {d:.4f}")
        print(f"  Top-1:    {t1}/{len(acts)} ({100*t1/len(acts):.0f}%)")

    if anth_descs:
        print(f"\n2x2 CROSS-SCORING MATRIX (delta / top-1):")
        print(f"  {'Scored by →':<25} {'Our AR':>15} {'Anthropic AR':>15}")
        our_d = our_delta
        our_t = our_top1
        print(f"  {'Our AV descs':<25} {our_d:.4f}/{100*our_t/len(acts):.0f}%", end="")
        if anth_correct:
            print(f"   {anth_delta:.4f}/{100*anth_top1/len(acts):.0f}%", end="")
        print()
        if anth_descs_on_our_ar_correct:
            d1 = mean(anth_descs_on_our_ar_correct) - mean(anth_descs_on_our_ar_garbage)
            t1 = sum(1 for c, g in zip(anth_descs_on_our_ar_correct, anth_descs_on_our_ar_garbage) if c > g)
            print(f"  {'Anthropic AV descs':<25} {d1:.4f}/{100*t1/len(acts):.0f}%", end="")
            if anth_descs_on_anth_ar_correct:
                d2 = mean(anth_descs_on_anth_ar_correct) - mean(anth_descs_on_anth_ar_garbage)
                t2 = sum(1 for c, g in zip(anth_descs_on_anth_ar_correct, anth_descs_on_anth_ar_garbage) if c > g)
                print(f"   {d2:.4f}/{100*t2/len(acts):.0f}%", end="")
            print()

    print(f"\nSAMPLE DESCRIPTIONS:")
    for i in range(min(3, len(acts))):
        print(f"\n  [{ids[i]}] our_cos={our_correct[i]:.3f}", end="")
        if anth_correct:
            print(f" anth_cos={anth_correct[i]:.3f}", end="")
        print(f"\n  {descs[i][:200]}")

    if args.output:
        results = {
            "n_samples": len(acts),
            "our_ar": {"correct": mean(our_correct), "garbage": mean(our_garbage),
                       "delta": our_delta, "top1": our_top1 / len(acts)},
        }
        if anth_correct:
            results["anthropic_ar"] = {
                "correct": mean(anth_correct), "garbage": mean(anth_garbage),
                "delta": anth_delta, "top1": anth_top1 / len(acts)}
        results["samples"] = [
            {"id": ids[i], "desc": descs[i], "our_cos": our_correct[i],
             "anth_cos": anth_correct[i] if anth_correct else None}
            for i in range(min(10, len(acts)))]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
