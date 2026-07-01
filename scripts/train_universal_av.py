#!/usr/bin/env python3
"""
Train a Universal Activation Verbalizer — one adapter for ALL layers.

The prompt includes a depth tag so the model learns to describe different
processing stages: syntax at 10%, semantics at 50%, intent at 90%.

Takes: all-layers activation file + merged descriptions at 6 depth percentages
Produces: single LoRA adapter conditioned on depth

Usage:
  python3 scripts/train_universal_av.py \
    --model gemma3-1b \
    --activations corpus/activations/gemma3-1b_all_layers.pt \
    --output output/nla-gemma3-1b-universal-av \
    --epochs 5 --lr 8e-6
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from generation_utils import decode_generated

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

MODELS = {
    "gemma3-1b": "google/gemma-3-1b-it",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4": "microsoft/phi-4",
}

INJECTION_CHARS = {
    "gemma3-1b": "⎝",   # U+239D, token_id=251266
    "qwen25-7b": "㈎",   # U+320E, token_id=149705
    "qwen3-4b": "㈎",
    "phi4-mini": "★",    # U+2605, token_id=12087
    "phi4": "★",         # U+2605, token_id=27347
}
INJECTION_SCALE = 150.0

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]


def normalize_activation(v, target_scale):
    """L2-normalize activation to target_scale (Anthropic's approach)."""
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * (target_scale / norm)


def make_prompt(depth_pct, injection_char):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        f"Here is the vector from depth {depth_pct}% of the network:\n\n"
        f"<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )


def find_inject_pos(prompt_tokens, injection_token_id):
    for i, tid in enumerate(prompt_tokens):
        if tid == injection_token_id:
            return i
    raise ValueError("Injection token not found in prompt")


def nearest_depth_pct(layer, n_layers):
    depth = layer * 100 / n_layers
    return min(DEPTH_PCTS, key=lambda p: abs(p - depth))


def load_descriptions(suffix="", strict=False, mix=False):
    """Load descriptions at all depth percentages, with optional suffix.

    If strict=True, only load files matching the exact suffix — no fallback
    to _merged.json or unsuffixed files.

    If mix=True, load ALL available files per depth and merge them. Each text
    gets the description from the first file found (GPT-4o preferred), but
    texts only in merged/sonnet files are also included.
    """
    descs = {}
    for pct in DEPTH_PCTS:
        if mix:
            all_by_id = {}
            sources = []
            for pattern_suffix in [suffix, "_merged", "_sonnet"]:
                path = GENERATED_DIR / f"descriptions_L{pct}pct{pattern_suffix}.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    n = 0
                    for d in data:
                        if d["id"] not in all_by_id:
                            all_by_id[d["id"]] = []
                        all_by_id[d["id"]].append(d["description"])
                        n += 1
                    sources.append(f"{path.name}({n})")
            if all_by_id:
                merged = {}
                for tid, desc_list in all_by_id.items():
                    merged[tid] = np.random.choice(desc_list)
                descs[pct] = merged
                multi = sum(1 for v in all_by_id.values() if len(v) > 1)
                print(f"  L{pct}%: {len(merged)} texts ({multi} with multiple styles) from {', '.join(sources)}")
            else:
                print(f"  L{pct}%: NO FILES FOUND")
        elif strict and suffix:
            candidates = [
                GENERATED_DIR / f"descriptions_L{pct}pct{suffix}.json",
            ]
            path = None
            for c in candidates:
                if c.exists():
                    path = c
                    break
            if path is None:
                print(f"  L{pct}%: NO FILE FOUND (tried suffix='{suffix}')")
                continue
            data = json.loads(path.read_text())
            descs[pct] = {d["id"]: d["description"] for d in data}
            print(f"  L{pct}%: {len(descs[pct])} descriptions from {path.name}")
        else:
            candidates = [
                GENERATED_DIR / f"descriptions_L{pct}pct{suffix}.json",
                GENERATED_DIR / f"descriptions_L{pct}pct{suffix}_merged.json",
            ]
            if suffix:
                candidates += [
                    GENERATED_DIR / f"descriptions_L{pct}pct_merged.json",
                    GENERATED_DIR / f"descriptions_L{pct}pct.json",
                ]
            else:
                candidates += [
                    GENERATED_DIR / f"descriptions_L{pct}pct_merged.json",
                ]
            path = None
            for c in candidates:
                if c.exists():
                    path = c
                    break
            if path is None:
                print(f"  L{pct}%: NO FILE FOUND (tried suffix='{suffix}')")
                continue
            data = json.loads(path.read_text())
            descs[pct] = {d["id"]: d["description"] for d in data}
            print(f"  L{pct}%: {len(descs[pct])} descriptions from {path.name}")
    return descs


class UniversalNLADataset(Dataset):
    def __init__(self, examples, tokenizer, injection_char, max_length=512):
        self.max_length = max_length

        injection_token_id = tokenizer.encode(
            injection_char, add_special_tokens=False
        )
        assert len(injection_token_id) == 1, (
            f"{injection_char} encodes to {len(injection_token_id)} tokens on this tokenizer"
        )
        injection_token_id = injection_token_id[0]

        prompt_cache = {}
        for pct in DEPTH_PCTS:
            content = make_prompt(pct, injection_char)
            chat_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            tokens = tokenizer.encode(chat_str, add_special_tokens=False)
            inject_pos = find_inject_pos(tokens, injection_token_id)
            prompt_cache[pct] = (tokens, inject_pos)

        self.items = []
        for ex in examples:
            prompt_tokens, inject_pos = prompt_cache[ex["depth_pct"]]
            prompt_len = len(prompt_tokens)

            desc_text = ex["description"] + "</explanation>"
            desc_tokens = tokenizer.encode(desc_text, add_special_tokens=False)

            input_ids = prompt_tokens + desc_tokens
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) > max_length:
                labels = labels[:max_length]

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "activation": ex["activation"],
                "inject_pos": inject_pos,
            })
        print(f"    Pre-tokenized {len(self.items)} examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    inject_positions = [b["inject_pos"] for b in batch]

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "activations": activations,
        "inject_positions": inject_positions,
    }


def build_examples(act_data, descriptions_by_depth):
    """Build training examples: one per (text, layer) pair."""
    layer_acts = act_data["activations"]
    ids = act_data["ids"]
    n_layers = act_data["n_layers"]

    examples = []
    for layer_idx in range(n_layers):
        depth_pct = nearest_depth_pct(layer_idx, n_layers)
        if depth_pct not in descriptions_by_depth:
            print(f"  Layer {layer_idx:2d} ({depth_pct:2d}%): SKIPPED (no descriptions)")
            continue
        desc_map = descriptions_by_depth[depth_pct]
        acts = layer_acts[layer_idx]

        matched = 0
        for text_idx, text_id in enumerate(ids):
            if text_id in desc_map:
                examples.append({
                    "activation": acts[text_idx],
                    "description": desc_map[text_id],
                    "depth_pct": depth_pct,
                    "layer": layer_idx,
                    "text_id": text_id,
                })
                matched += 1

        print(f"  Layer {layer_idx:2d} ({depth_pct:2d}%): {matched} examples")

    print(f"Total: {len(examples)} training examples")
    return examples


def train(model, tokenizer, train_dataset, val_dataset, args, device):
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    embed_layer = model.get_input_embeddings()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activations = batch["activations"].to(device)
            inject_positions = batch["inject_positions"]

            embeddings = embed_layer(input_ids)
            for i, pos in enumerate(inject_positions):
                embeddings[i, pos, :] = normalize_activation(activations[i], INJECTION_SCALE)

            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 200 == 0:
                avg = total_loss / n_batches
                print(f"  epoch {epoch+1} step {batch_idx+1}: loss={avg:.3f}")

        train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                activations = batch["activations"].to(device)
                inject_positions = batch["inject_positions"]

                embeddings = embed_layer(input_ids)
                for i, pos in enumerate(inject_positions):
                    embeddings[i, pos, :] = normalize_activation(activations[i], INJECTION_SCALE)

                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        print(f"  Epoch {epoch+1}/{args.epochs}: train={train_loss:.3f} val={val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best val={best_val_loss:.3f})")

    return best_val_loss


def evaluate(model, tokenizer, examples, injection_char, device, n_samples=12):
    """Generate descriptions for sample activations across different depths."""
    model.eval()
    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]

    depths_seen = set()
    selected = []
    np.random.seed(42)
    shuffled = np.random.permutation(len(examples))
    for idx in shuffled:
        ex = examples[idx]
        if ex["depth_pct"] not in depths_seen or len(selected) < n_samples:
            selected.append(ex)
            depths_seen.add(ex["depth_pct"])
        if len(selected) >= n_samples:
            break

    print(f"\n{'='*60}")
    print(f"Sample generations (Universal NLA)")
    print(f"{'='*60}")

    for ex in sorted(selected, key=lambda e: e["depth_pct"]):
        content = make_prompt(ex["depth_pct"], injection_char)
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
        inject_pos = find_inject_pos(prompt_tokens, injection_token_id)

        embed_layer = model.get_input_embeddings()
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        embeddings = embed_layer(input_ids)
        embeddings[0, inject_pos, :] = normalize_activation(
            ex["activation"].to(device), INJECTION_SCALE)

        with torch.no_grad():
            output = model.generate(
                inputs_embeds=embeddings.to(model.dtype),
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated = decode_generated(output, prompt_tokens, tokenizer)
        print(f"\n[L{ex['layer']:2d} {ex['depth_pct']:2d}%] {ex['text_id']}")
        print(f"  Ground truth: {ex['description'][:150]}")
        print(f"  Generated:    {generated[:150]}")


def main():
    parser = argparse.ArgumentParser(description="Train Universal NLA (all layers)")
    parser.add_argument("--model", default="gemma3-1b", choices=list(MODELS.keys()))
    parser.add_argument("--activations", type=str, required=True,
                        help="Path to all-layers activation file (.pt)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--desc-suffix", default="",
                        help="Description file suffix (e.g. _prediction, _tight)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.15)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--force-holdout-json", default="",
                        help="path to a JSON list of text ids to FORCE into the val "
                             "split (never trained on); keeps an eval holdout leak-free")
    parser.add_argument("--strict", action="store_true",
                        help="Only load descriptions matching exact suffix (no fallback)")
    parser.add_argument("--mix", action="store_true",
                        help="Load ALL available description files per depth and merge")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading descriptions at all depths (suffix='{args.desc_suffix}', strict={args.strict}, mix={args.mix})...")
    descriptions = load_descriptions(suffix=args.desc_suffix, strict=args.strict, mix=args.mix)

    print(f"\nLoading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=True)
    n_layers = act_data["n_layers"]
    d_model = act_data["d_model"]
    n_texts = act_data["n_texts"]
    print(f"  {n_layers} layers, {n_texts} texts, d={d_model}")

    print("\nBuilding examples...")
    examples = build_examples(act_data, descriptions)

    injection_char = INJECTION_CHARS.get(args.model)
    if injection_char is None:
        raise ValueError(
            f"No injection char configured for {args.model}. "
            f"Run find_injection_token.py and add to INJECTION_CHARS."
        )

    model_name = MODELS[args.model]
    trust_remote = "phi" not in args.model.lower()
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=trust_remote
    )

    if args.eval_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.output)
        evaluate(model, tokenizer, examples, injection_char, device)
        return

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")

    # Split by TEXT ID to prevent leakage (same text at different layers in both splits)
    all_text_ids = sorted(set(ex["text_id"] for ex in examples))
    n_val_texts = max(1, int(len(all_text_ids) * args.val_split))
    rng = np.random.RandomState(42)
    val_text_ids = set(rng.choice(all_text_ids, n_val_texts, replace=False))

    if args.force_holdout_json:
        forced = set(json.load(open(args.force_holdout_json)))
        present = forced & set(all_text_ids)
        val_text_ids = set(val_text_ids) | present
        print(f"  Forced {len(present)}/{len(forced)} holdout ids into val "
              f"(excluded from training, leak-free eval)")

    train_examples = [ex for ex in examples if ex["text_id"] not in val_text_ids]
    val_examples = [ex for ex in examples if ex["text_id"] in val_text_ids]

    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)} ({n_val_texts} held-out texts)")

    train_dataset = UniversalNLADataset(train_examples, tokenizer, injection_char)
    val_dataset = UniversalNLADataset(val_examples, tokenizer, injection_char)

    print(f"\nTraining for {args.epochs} epochs...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    # Record the held-out text ids so downstream GRPO can exclude them
    # (--exclude-ids-file) and never fine-tune on the eval set.
    json.dump(sorted(val_text_ids),
              open(Path(args.output) / "val_text_ids.json", "w"))
    best = train(model, tokenizer, train_dataset, val_dataset, args, device)
    print(f"\nBest val loss: {best:.3f}")

    import yaml
    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "variant": "universal",
        "stage": "sft",
        "d_model": d_model,
        "n_layers": n_layers,
        "extraction": {"injection_scale": INJECTION_SCALE},
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(injection_token_id),
        },
        "prompt_templates": {
            "av": make_prompt("{depth_pct}", "{injection_char}"),
        },
        "depth_percentages": DEPTH_PCTS,
        "training": {
            "method": "lora_sft",
            "injection_mode": "normalize",
            "chat_template": True,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "best_val_loss": float(best),
            "corpus": "nla-at-home",
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    evaluate(model, tokenizer, val_examples, injection_char, device)


if __name__ == "__main__":
    main()
