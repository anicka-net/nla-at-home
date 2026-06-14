#!/usr/bin/env python3
"""
Download pre-computed NLA corpus and activations from HuggingFace,
set up local directory structure, and optionally launch training.

Usage:
  python3 scripts/quickstart.py                    # download data only
  python3 scripts/quickstart.py --train gemma3-1b  # download + train
  python3 scripts/quickstart.py --train gemma3-1b --device cpu  # CPU (slow)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"

HF_DATASET = "anicka/nla-at-home-corpus"

ACTIVATION_URLS = {
    "gemma3-1b": "https://huggingface.co/datasets/anicka/nla-at-home-corpus/resolve/main/activations/gemma3-1b_all_layers.pt",
}

DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]


def download_corpus():
    """Download the NLA corpus from HuggingFace and split into per-depth files."""
    print("Downloading corpus from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  pip install huggingface_hub")
        sys.exit(1)

    corpus_path = hf_hub_download(
        repo_id=HF_DATASET, filename="nla_corpus.json", repo_type="dataset")
    texts_path = hf_hub_download(
        repo_id=HF_DATASET, filename="texts.json", repo_type="dataset")

    corpus = json.loads(Path(corpus_path).read_text())
    texts = json.loads(Path(texts_path).read_text())
    print(f"  {len(corpus)} description rows, {len(texts)} texts")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Write per-depth description files (the format training scripts expect)
    by_depth = {}
    for row in corpus:
        pct = row["depth_pct"]
        if pct not in by_depth:
            by_depth[pct] = []
        by_depth[pct].append(row)

    for pct in DEPTH_PCTS:
        out = GENERATED_DIR / f"descriptions_L{pct}pct_merged.json"
        if out.exists():
            existing = json.loads(out.read_text())
            print(f"  L{pct}%: already exists ({len(existing)} entries), skipping")
            continue
        rows = by_depth.get(pct, [])
        out.write_text(json.dumps(rows, indent=1))
        print(f"  L{pct}%: wrote {len(rows)} descriptions")

    # Write text files by category (for corpus loader)
    by_cat = {}
    for t in texts:
        cat = t.get("category", "unknown")
        cat_file = cat.split("_")[0] + "_" + "_".join(cat.split("_")[1:])
        if cat_file not in by_cat:
            by_cat[cat_file] = []
        by_cat[cat_file].append(t)

    existing_cats = list(GENERATED_DIR.glob("[A-H]*.json"))
    if existing_cats:
        print(f"  {len(existing_cats)} category files already exist, skipping")
    else:
        for cat_file, items in sorted(by_cat.items()):
            out = GENERATED_DIR / f"{cat_file}.json"
            out.write_text(json.dumps(items, indent=1))
        print(f"  Wrote {len(by_cat)} category files")

    print("  Corpus ready.")


def download_activations(model):
    """Download pre-extracted activations for a model."""
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if model == "gemma3-1b":
        out = ACTIVATIONS_DIR / "gemma3-1b_all_layers.pt"
        if out.exists():
            print(f"  Activations already exist: {out}")
            return str(out)

        print(f"  Downloading activations for {model}...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=HF_DATASET,
                filename="activations/gemma3-1b_all_layers.pt",
                repo_type="dataset")
            import shutil
            shutil.copy2(path, out)
            print(f"  Saved to {out}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print(f"  You can extract locally instead:")
            print(f"    python3 scripts/extract_activations.py --model {model} --all-layers")
            return None
        return str(out)
    else:
        print(f"  No pre-extracted activations for {model}.")
        print(f"  Extract locally:")
        print(f"    python3 scripts/extract_activations.py --model {model} --all-layers")
        return None


def train(model, activations_path, device):
    """Launch AV and AR training."""
    output_av = REPO_ROOT / "output" / f"nla-{model}-universal-av"
    output_ar = REPO_ROOT / "output" / f"nla-{model}-universal-ar"

    print(f"\nTraining Universal AV...")
    subprocess.run([
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train_universal_av.py"),
        "--model", model,
        "--activations", activations_path,
        "--output", str(output_av),
        "--epochs", "5", "--lr", "8e-6",
        "--device", device,
    ], check=True)

    print(f"\nTraining Universal AR (verification)...")
    subprocess.run([
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train_universal_ar.py"),
        "--model", model,
        "--activations", activations_path,
        "--output", str(output_ar),
        "--epochs", "5", "--lr", "7e-5",
        "--device", device,
    ], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download NLA corpus and optionally train adapters")
    parser.add_argument("--train", metavar="MODEL",
                        choices=["gemma3-1b", "qwen25-7b", "qwen3-4b"],
                        help="Train adapters for this model after downloading")
    parser.add_argument("--device", default="cuda",
                        help="Training device (default: cuda)")
    args = parser.parse_args()

    download_corpus()

    if args.train:
        act_path = download_activations(args.train)
        if act_path is None:
            print("\nCannot train without activations. Extract them first.")
            sys.exit(1)
        train(args.train, act_path, args.device)
    else:
        print("\nCorpus downloaded. To train adapters:")
        print("  python3 scripts/quickstart.py --train gemma3-1b")
        print("\nOr extract activations for a different model:")
        print("  python3 scripts/extract_activations.py --model qwen25-7b --all-layers")


if __name__ == "__main__":
    main()
