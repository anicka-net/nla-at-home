#!/usr/bin/env python3
"""
Train an Activation Reconstructor (AR) for NLA-at-home.

The AR is the reverse of the AV: it takes a natural language description
and reconstructs the activation vector. The model processes the description
text, and the hidden state at the extraction layer at the injection token
position is trained (MSE loss) to match the original activation.

Usage:
  python3 scripts/train_ar.py \
    --model qwen25-7b \
    --layer 20 \
    --epochs 10 \
    --output output/nla-qwen25-7b-L20-ar
"""
import torch
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

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

def ar_template(inject_char):
    return (
        "Summary of the following text: <text>{explanation}</text> <summary>"
        + inject_char
    )

device = torch.device("cuda")


def expected_description_pct(act_data, layer):
    n_layers = act_data.get("n_layers")
    if n_layers is None:
        raise KeyError("Activation file missing n_layers")
    return int(layer * 100 / n_layers)


def resolve_description_path(act_data, layer, description_file=None):
    if description_file is not None:
        path = Path(description_file)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    layer_pct = expected_description_pct(act_data, layer)
    desc_path = GENERATED_DIR / f"descriptions_L{layer_pct}pct.json"
    if desc_path.exists():
        return desc_path
    raise FileNotFoundError(f"Expected {desc_path.name}")


class ARDataset(Dataset):
    def __init__(self, activations, descriptions, tokenizer, max_length=512):
        self.activations = activations
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.inject_id = tokenizer.encode(
            INJECTION_CHAR, add_special_tokens=False
        )
        assert len(self.inject_id) == 1
        self.inject_id = self.inject_id[0]

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        desc = self.descriptions[idx]
        act = self.activations[idx]

        prompt = AR_TEMPLATE.replace("{explanation}", desc)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        inject_pos = None
        for i, t in enumerate(tokens):
            if t == self.inject_id:
                inject_pos = i
                break

        if inject_pos is None:
            inject_pos = len(tokens) - 1

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "activation": act,
            "inject_pos": inject_pos,
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    inject_positions = torch.tensor([b["inject_pos"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "activations": activations,
        "inject_positions": inject_positions,
    }


def load_data(model_key, layer, description_file=None):
    act_path = ACTIVATIONS_DIR / f"{model_key}_L{layer}.pt"
    act_data = torch.load(act_path, weights_only=True)
    activations = act_data["activations"]
    ids = act_data["ids"]

    desc_path = resolve_description_path(act_data, layer, description_file)
    all_descs = json.loads(desc_path.read_text())
    id_to_desc = {d["id"]: d["description"] for d in all_descs}

    matched_acts = []
    matched_descs = []
    for i, text_id in enumerate(ids):
        if text_id in id_to_desc:
            matched_acts.append(activations[i])
            matched_descs.append(id_to_desc[text_id])

    print(f"Matched {len(matched_descs)}/{len(ids)} pairs")
    return torch.stack(matched_acts), matched_descs


def train(model, tokenizer, train_acts, train_descs, val_acts, val_descs,
          extraction_layer, args):
    train_dataset = ARDataset(train_acts, train_descs, tokenizer)
    val_dataset = ARDataset(val_acts, val_descs, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    blocks = inner.layers

    best_val_loss = float("inf")
    mse_scale = args.mse_scale

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_acts = batch["activations"].to(device).float()
            inject_positions = batch["inject_positions"]

            layer_outputs = {}
            def make_hook(layer_idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_outputs[layer_idx] = h
                return hook
            handle = blocks[extraction_layer].register_forward_hook(
                make_hook(extraction_layer))

            model(input_ids=input_ids, attention_mask=attention_mask)
            handle.remove()

            hidden = layer_outputs[extraction_layer]
            reconstructed = torch.stack([
                hidden[i, inject_positions[i]].float()
                for i in range(len(inject_positions))
            ])

            loss = torch.nn.functional.mse_loss(
                reconstructed * mse_scale, target_acts * mse_scale)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0
        val_cos = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_acts = batch["activations"].to(device).float()
                inject_positions = batch["inject_positions"]

                layer_outputs = {}
                handle = blocks[extraction_layer].register_forward_hook(
                    make_hook(extraction_layer))
                model(input_ids=input_ids, attention_mask=attention_mask)
                handle.remove()

                hidden = layer_outputs[extraction_layer]
                reconstructed = torch.stack([
                    hidden[i, inject_positions[i]].float()
                    for i in range(len(inject_positions))
                ])

                loss = torch.nn.functional.mse_loss(
                    reconstructed * mse_scale, target_acts * mse_scale)
                cos = torch.nn.functional.cosine_similarity(
                    reconstructed, target_acts, dim=1).mean()

                val_loss += loss.item()
                val_cos += cos.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        val_cos = val_cos / val_batches
        print(f"  Epoch {epoch+1}/{args.epochs}: train_mse={train_loss:.4f} "
              f"val_mse={val_loss:.4f} val_cosine={val_cos:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"    -> saved (best val_mse={best_val_loss:.4f})")

    return best_val_loss, val_cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--mse-scale", type=float, default=59.87)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--description-file", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global device, INJECTION_CHAR, AR_TEMPLATE
    device = torch.device(args.device)
    INJECTION_CHAR = INJECTION_CHARS[args.model]
    AR_TEMPLATE = ar_template(INJECTION_CHAR)

    model_name = MODELS[args.model]
    trust_remote = "phi" not in args.model.lower()
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=trust_remote)

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

    acts, descs = load_data(args.model, args.layer, args.description_file)
    print(f"  {len(descs)} examples, d_model={acts.shape[1]}")

    n_val = int(len(descs) * args.val_split)
    indices = np.random.RandomState(42).permutation(len(descs))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_acts = acts[train_idx]
    train_descs = [descs[i] for i in train_idx]
    val_acts = acts[val_idx]
    val_descs = [descs[i] for i in val_idx]
    print(f"  Train: {len(train_descs)}, Val: {len(val_descs)}")

    print(f"\nTraining AR for {args.epochs} epochs...")
    best_mse, best_cos = train(model, tokenizer, train_acts, train_descs,
                                val_acts, val_descs, args.layer, args)
    print(f"\nBest: val_mse={best_mse:.4f}, val_cosine={best_cos:.4f}")

    meta = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "ar",
        "stage": "sl",
        "d_model": int(acts.shape[1]),
        "extraction": {
            "injection_scale": None,
            "mse_scale": args.mse_scale,
        },
        "tokens": {
            "injection_char": INJECTION_CHAR,
            "injection_token_id": int(tokenizer.encode(
                INJECTION_CHAR, add_special_tokens=False)[0]),
        },
        "prompt_templates": {
            "ar": AR_TEMPLATE.replace(INJECTION_CHAR, "{injection_char}"),
        },
        "extraction_layer_index": args.layer,
        "training": {
            "method": "lora_sl",
            "lora_r": args.lora_r,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": len(train_descs),
            "n_val": len(val_descs),
            "best_val_mse": float(best_mse),
            "best_val_cosine": float(best_cos),
            "mse_scale": args.mse_scale,
            "corpus": "nla-at-home",
        },
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "nla_meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
