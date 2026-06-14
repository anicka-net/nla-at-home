#!/usr/bin/env python3
"""
Train an Activation Verbalizer (AV) for NLA-at-home.

Takes: corpus texts + descriptions + extracted activations
Produces: LoRA adapter that verbalizes activation vectors

Architecture matches Anthropic's kitft/nla-qwen2.5-7b-L20-av:
- Injection token at a known position in the prompt
- Embedding at that position replaced with scaled activation vector
- Model trained to produce description in <explanation> tags

Usage:
  python3 scripts/train_av.py \
    --model qwen25-7b \
    --layer 20 \
    --epochs 20 \
    --output output/nla-qwen25-7b-L20-av
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
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

INJECTION_CHARS = {
    "qwen25-7b": "㈎",
    "qwen3-4b": "㈎",
    "phi4-mini": "★",
}
INJECTION_CHAR = "㈎"
INJECTION_SCALE = 150.0

def make_prompt_template(inject_char):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        "Here is the vector:\n\n"
        f"<concept>{inject_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )

PROMPT_TEMPLATE = make_prompt_template(INJECTION_CHAR)

device = torch.device("cuda")


class NLADataset(Dataset):
    def __init__(self, activations, descriptions, tokenizer, max_length=512):
        self.activations = activations
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.injection_token_id = tokenizer.encode(
            INJECTION_CHAR, add_special_tokens=False
        )
        assert len(self.injection_token_id) == 1, (
            f"Injection char encodes to {len(self.injection_token_id)} tokens, need exactly 1"
        )
        self.injection_token_id = self.injection_token_id[0]

        self.prompt_tokens = tokenizer.encode(
            PROMPT_TEMPLATE, add_special_tokens=False
        )
        self.prompt_len = len(self.prompt_tokens)

        self.inject_pos = None
        for i, tid in enumerate(self.prompt_tokens):
            if tid == self.injection_token_id:
                self.inject_pos = i
                break
        assert self.inject_pos is not None, "Injection token not found in prompt template"

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        desc = self.descriptions[idx]
        act = self.activations[idx]

        desc_text = desc + "</explanation>"
        desc_tokens = self.tokenizer.encode(desc_text, add_special_tokens=False)

        input_ids = self.prompt_tokens + desc_tokens
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        labels = [-100] * self.prompt_len + input_ids[self.prompt_len:]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "activation": act,
            "inject_pos": self.inject_pos,
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    inject_pos = batch[0]["inject_pos"]

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
        "inject_pos": inject_pos,
    }


def expected_description_pct(act_data, layer):
    n_layers = act_data.get("n_layers")
    if n_layers is None:
        raise KeyError(
            "Activation file does not contain n_layers; pass --description-file explicitly"
        )
    return int(layer * 100 / n_layers)


def resolve_description_path(act_data, layer, description_file=None):
    if description_file is not None:
        path = Path(description_file)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Description file not found: {path}")
        return path

    layer_pct = expected_description_pct(act_data, layer)
    desc_path = GENERATED_DIR / f"descriptions_L{layer_pct}pct.json"
    if desc_path.exists():
        return desc_path

    candidates = sorted(p.name for p in GENERATED_DIR.glob("descriptions_L*pct*.json"))
    raise FileNotFoundError(
        f"Expected descriptions for layer {layer} at {layer_pct}% depth: "
        f"{desc_path.name}. Available description files: {candidates or 'none'}. "
        "Pass --description-file for merged or custom files."
    )


def load_data(model_key, layer, description_file=None):
    act_path = ACTIVATIONS_DIR / f"{model_key}_L{layer}.pt"
    act_data = torch.load(act_path, weights_only=True)
    activations = act_data["activations"]
    ids = act_data["ids"]
    print(f"Loaded {len(ids)} activations, shape {activations.shape}")

    desc_path = resolve_description_path(act_data, layer, description_file)
    all_descs = json.loads(desc_path.read_text())
    print(f"Loaded {len(all_descs)} descriptions from {desc_path.name}")

    id_to_desc = {d["id"]: d["description"] for d in all_descs}

    matched_acts = []
    matched_descs = []
    for i, text_id in enumerate(ids):
        if text_id in id_to_desc:
            matched_acts.append(activations[i])
            matched_descs.append(id_to_desc[text_id])

    print(f"Matched {len(matched_descs)}/{len(ids)} (activation, description) pairs")
    return torch.stack(matched_acts), matched_descs, desc_path


def load_augmented(aug_path):
    """Load augmented direction/sparse training data."""
    aug_data = torch.load(aug_path, weights_only=False)
    vectors = aug_data["vectors"]
    metas = aug_data["metas"]

    acts = []
    descs = []
    for i, meta in enumerate(metas):
        if "description" in meta and meta["description"]:
            acts.append(vectors[i])
            descs.append(meta["description"])

    print(f"Loaded {len(descs)}/{len(metas)} augmented examples from {Path(aug_path).name}")
    return torch.stack(acts) if acts else torch.empty(0), descs


def train(model, tokenizer, train_acts, train_descs, val_acts, val_descs, args):
    train_dataset = NLADataset(train_acts, train_descs, tokenizer)
    val_dataset = NLADataset(val_acts, val_descs, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
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

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activations = batch["activations"].to(device)
            inject_pos = batch["inject_pos"]

            embeddings = embed_layer(input_ids)
            embeddings[:, inject_pos, :] = activations * INJECTION_SCALE

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
                inject_pos = batch["inject_pos"]

                embeddings = embed_layer(input_ids)
                embeddings[:, inject_pos, :] = activations * INJECTION_SCALE

                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.3f} val_loss={val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best val_loss={best_val_loss:.3f})")

    return best_val_loss


def generate_from_activation(model, tokenizer, activation, injection_scale,
                             prompt_tokens, inject_pos, max_new_tokens=150,
                             do_sample=False, temperature=1.0):
    """Generate a description from an activation vector."""
    embed_layer = model.get_input_embeddings()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    embeddings = embed_layer(input_ids)
    embeddings[0, inject_pos, :] = activation.to(device) * injection_scale
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=embeddings.to(model.dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    return decode_generated(output, prompt_tokens, tokenizer)


def evaluate(model, tokenizer, activations, descriptions, n_samples=10):
    """Generate descriptions for sample activations and compare."""
    prompt_tokens = tokenizer.encode(PROMPT_TEMPLATE, add_special_tokens=False)
    inject_id = tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)[0]
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)

    model.eval()
    indices = np.random.choice(len(descriptions), min(n_samples, len(descriptions)), replace=False)

    print(f"\n{'='*60}")
    print(f"Sample generations vs ground truth")
    print(f"{'='*60}")

    for idx in indices:
        generated = generate_from_activation(
            model, tokenizer, activations[idx], INJECTION_SCALE,
            prompt_tokens, inject_pos)

        print(f"\n[{idx}] Ground truth: {descriptions[idx][:200]}")
        print(f"    Generated:    {generated[:200]}")
        print(f"    Gen length:   {len(generated)} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augmented", type=str, default=None,
                        help="Path to augmented direction/sparse data (.pt)")
    parser.add_argument(
        "--description-file",
        default=None,
        help=(
            "Description JSON file to use. Defaults to "
            "corpus/generated/descriptions_L{layer_pct}pct.json derived from "
            "the activation file's n_layers metadata."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global device, INJECTION_CHAR, PROMPT_TEMPLATE
    device = torch.device(args.device)
    INJECTION_CHAR = INJECTION_CHARS[args.model]
    PROMPT_TEMPLATE = make_prompt_template(INJECTION_CHAR)

    model_name = MODELS[args.model]
    trust_remote = "phi" not in args.model.lower()
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote
    ).to(device)

    if args.eval_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.output)
        acts, descs, _ = load_data(args.model, args.layer, args.description_file)
        evaluate(model, tokenizer, acts, descs)
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

    acts, descs, desc_path = load_data(args.model, args.layer, args.description_file)

    n_val = int(len(descs) * args.val_split)
    indices = np.random.RandomState(42).permutation(len(descs))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_acts = acts[train_idx]
    train_descs = [descs[i] for i in train_idx]
    val_acts = acts[val_idx]
    val_descs = [descs[i] for i in val_idx]

    if args.augmented:
        aug_acts, aug_descs = load_augmented(args.augmented)
        if len(aug_descs) > 0:
            train_acts = torch.cat([train_acts, aug_acts])
            train_descs = train_descs + aug_descs

    print(f"  Train: {len(train_descs)}, Val: {len(val_descs)}")

    print(f"\nTraining for {args.epochs} epochs...")
    best = train(model, tokenizer, train_acts, train_descs, val_acts, val_descs, args)
    print(f"\nBest val loss: {best:.3f}")

    # Save NLA metadata
    import yaml
    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "stage": "sft",
        "d_model": int(acts.shape[1]),
        "extraction": {
            "injection_scale": INJECTION_SCALE,
        },
        "tokens": {
            "injection_char": INJECTION_CHAR,
            "injection_token_id": int(tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)[0]),
        },
        "prompt_templates": {
            "av": PROMPT_TEMPLATE.replace(INJECTION_CHAR, "{injection_char}"),
        },
        "extraction_layer_index": args.layer,
        "training": {
            "method": "lora_sft",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": len(train_descs),
            "n_val": len(val_descs),
            "best_val_loss": float(best),
            "corpus": "nla-at-home",
            "description_file": str(desc_path.relative_to(REPO_ROOT)),
        },
    }
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    evaluate(model, tokenizer, val_acts, val_descs, n_samples=10)


if __name__ == "__main__":
    main()
