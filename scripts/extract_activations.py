#!/usr/bin/env python3
"""
Extract activations from a target model at a target layer for the NLA corpus.

Loads all generated texts, runs each through the model, saves the
last-token residual stream vector at the target layer.

Output: corpus/activations/{model_short}_{layer}.pt
  - Dict with keys: "activations" (N×d tensor), "ids" (list of text IDs),
    "model", "layer", "d_model"
"""
import torch
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"

# Extraction works on any registered model (no injection token needed),
# hence MODELS_HF rather than INJECTABLE_MODELS_HF.
from nla_lib import MODELS_HF as MODELS, get_model  # noqa: E402

device = torch.device("cuda")


def load_corpus(input_file=None):
    texts = []
    seen_ids = set()
    if input_file:
        sources = [Path(input_file)]
    else:
        sources = sorted(GENERATED_DIR.glob("*.json"))
    for path in sources:
        if path.name.startswith("descriptions_") or "descriptions_" in path.name:
            continue
        if path.name == "texts_needing_activations.json":
            continue
        data = json.loads(path.read_text())
        for item in data:
            if "text" in item and item.get("id") not in seen_ids:
                seen_ids.add(item["id"])
                texts.append(item)
    return texts


def get_blocks(model):
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    return inner.layers


def extract(model, tokenizer, texts, layer, model_key):
    blocks = get_blocks(model)
    activations = []
    ids = []

    for i, item in enumerate(texts):
        text = item["text"]
        messages = [{"role": "user", "content": text}]

        kwargs = {}
        if "qwen3" in model_key.lower():
            kwargs["enable_thinking"] = False

        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        inputs = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        seq_len = inputs["attention_mask"].sum() - 1

        activation = {}

        def make_hook(layer_idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activation[layer_idx] = h[0, seq_len].detach().cpu().float()
            return hook

        handle = blocks[layer].register_forward_hook(make_hook(layer))

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        activations.append(activation[layer])
        ids.append(item["id"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)}")

    return torch.stack(activations), ids


def extract_all_layers(model, tokenizer, texts, model_key):
    """Extract activations at ALL layers in a single forward pass per text."""
    blocks = get_blocks(model)
    n_layers = len(blocks)

    per_layer = {l: [] for l in range(n_layers)}
    ids = []

    for i, item in enumerate(texts):
        text = item["text"]
        messages = [{"role": "user", "content": text}]

        kwargs = {}
        if "qwen3" in model_key.lower():
            kwargs["enable_thinking"] = False

        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        inputs = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        seq_len = inputs["attention_mask"].sum() - 1

        activation = {}

        def make_hook(layer_idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activation[layer_idx] = h[0, seq_len].detach().cpu().float()
            return hook

        handles = []
        for l in range(n_layers):
            handles.append(blocks[l].register_forward_hook(make_hook(l)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for l in range(n_layers):
            per_layer[l].append(activation[l])
        ids.append(item["id"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)}")

    stacked = {l: torch.stack(per_layer[l]) for l in range(n_layers)}
    return stacked, ids


def pca_summary(acts, label=""):
    centered = acts - acts.mean(0)
    U, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var_explained = (S ** 2) / (S ** 2).sum()
    cumvar = var_explained.cumsum(0)
    prefix = f"  {label} " if label else "  "
    for k in [1, 10, 50]:
        if k <= len(cumvar):
            print(f"{prefix}PCA top-{k}: {cumvar[k-1]:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int)
    layer_group.add_argument("--all-layers", action="store_true",
                             help="Extract all layers in one pass")
    parser.add_argument("--input", type=str, default=None,
                        help="Specific JSON file to extract from (default: all corpus)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output filename (e.g. '_orig' → qwen25-7b_L20_orig.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting an existing activations file")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)

    model_name = MODELS[args.model]
    print(f"Loading corpus...")
    texts = load_corpus(args.input)
    print(f"  {len(texts)} texts")

    print(f"Loading {model_name}...")
    trust_remote = get_model(args.model).trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
        device_map={"": str(device)},
        trust_remote_code=trust_remote
    )
    model.eval()

    blocks = get_blocks(model)
    n_layers = len(blocks)
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Overwrite guard BEFORE any extraction compute. 2026-07-08: the
    # all-layers path used to ignore --output-suffix — a probe extraction
    # silently overwrote the main 5857-text corpus file (recovered from an
    # off-host copy). Fail in seconds, not after an hour of GPU work.
    out_name = (f"{args.model}_all_layers{args.output_suffix}.pt"
                if args.all_layers
                else f"{args.model}_L{args.layer}{args.output_suffix}.pt")
    planned_out = ACTIVATIONS_DIR / out_name
    if planned_out.exists() and planned_out.stat().st_size > 0 and not args.force:
        raise SystemExit(
            f"REFUSING to overwrite existing {planned_out} "
            f"({planned_out.stat().st_size / 1e6:.0f} MB). Pass --force if "
            f"you really mean it, or use --output-suffix.")

    if args.all_layers:
        print(f"  {n_layers} layers, extracting ALL")
        print(f"Extracting activations at all {n_layers} layers...")
        layer_acts, ids = extract_all_layers(model, tokenizer, texts, args.model)

        out_path = planned_out
        torch.save({
            "activations": {l: layer_acts[l] for l in range(n_layers)},
            "ids": ids,
            "model": model_name,
            "n_layers": n_layers,
            "d_model": layer_acts[0].shape[1],
            "n_texts": len(ids),
        }, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Saved {n_layers} layers × {len(ids)} texts × {layer_acts[0].shape[1]}d to {out_path} ({size_mb:.0f}MB)")

        for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            pca_summary(layer_acts[l], f"L{l}")
    else:
        print(f"  {n_layers} layers, extracting L{args.layer}")
        assert args.layer < n_layers, f"Layer {args.layer} >= {n_layers}"

        print(f"Extracting activations...")
        acts, ids = extract(model, tokenizer, texts, args.layer, args.model)

        out_path = planned_out
        torch.save({
            "activations": acts,
            "ids": ids,
            "model": model_name,
            "layer": args.layer,
            "n_layers": n_layers,
            "d_model": acts.shape[1],
            "n_texts": len(ids),
        }, out_path)
        print(f"Saved {acts.shape} to {out_path}")
        pca_summary(acts)


if __name__ == "__main__":
    main()
