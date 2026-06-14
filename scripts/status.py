#!/usr/bin/env python3
"""
Show pipeline readiness status for a given model and layer.

Checks what data exists, what's missing, and what needs to be generated.

Usage:
  python3 scripts/status.py --model qwen25-7b --layer 20
  python3 scripts/status.py  # show everything
"""
import json
import yaml
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_DIR = REPO_ROOT / "corpus" / "categories"
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"
AUGMENTED_DIR = REPO_ROOT / "corpus" / "augmented"
OUTPUT_DIR = REPO_ROOT / "output"

MODELS = {
    "qwen25-7b": ("Qwen/Qwen2.5-7B-Instruct", 28),
    "qwen3-4b": ("Qwen/Qwen3-4B", 36),
    "gemma3-1b": ("google/gemma-3-1b-it", 26),
    "qwen25-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", 24),
}


def check_corpus():
    """Check text generation status."""
    categories = []
    total_texts = 0
    missing = []
    unsafe_count = 0

    for path in sorted(CATEGORIES_DIR.glob("*.yaml")):
        cat = yaml.safe_load(path.read_text())
        cat_id = cat["id"]
        expected = cat.get("count", 20)
        unsafe = cat.get("unsafe", False)
        if unsafe:
            unsafe_count += 1

        gen_path = GENERATED_DIR / f"{cat_id}.json"
        if gen_path.exists():
            actual = len(json.loads(gen_path.read_text()))
            total_texts += actual
            categories.append((cat_id, actual, expected, unsafe))
        else:
            missing.append(cat_id)
            categories.append((cat_id, 0, expected, unsafe))

    return categories, total_texts, missing, unsafe_count


def check_descriptions():
    """Check description coverage."""
    depths = {}
    depth_ids = {}
    for path in sorted(GENERATED_DIR.glob("descriptions_L*pct.json")):
        name = path.stem
        pct = name.split("_L")[1].split("pct")[0]
        data = json.loads(path.read_text())
        depths[int(pct)] = len(data)
        depth_ids[int(pct)] = {item["id"] for item in data}

    expansion = {}
    expansion_ids = {}
    for path in sorted(GENERATED_DIR.glob("descriptions_L*pct_expansion.json")):
        name = path.stem
        pct = name.split("_L")[1].split("pct")[0]
        data = json.loads(path.read_text())
        expansion[int(pct)] = len(data)
        expansion_ids[int(pct)] = {item["id"] for item in data}

    return depths, expansion, depth_ids, expansion_ids


def generated_corpus_ids():
    """Return IDs present in generated corpus files, excluding descriptions."""
    ids = set()
    for path in sorted(GENERATED_DIR.glob("*.json")):
        if path.name.startswith("descriptions_"):
            continue
        data = json.loads(path.read_text())
        ids.update(item["id"] for item in data)
    return ids


def check_description_coverage(depth_ids=None, expansion_ids=None, corpus_ids=None):
    """Compare merged description IDs against generated corpus IDs."""
    if depth_ids is None or expansion_ids is None:
        _, _, depth_ids, expansion_ids = check_descriptions()
    if corpus_ids is None:
        corpus_ids = generated_corpus_ids()

    coverage = {}
    for pct in sorted(set(depth_ids.keys()) | set(expansion_ids.keys())):
        merged_ids = depth_ids.get(pct, set()) | expansion_ids.get(pct, set())
        coverage[pct] = {
            "unique": len(merged_ids),
            "missing": sorted(corpus_ids - merged_ids),
            "extra": sorted(merged_ids - corpus_ids),
        }
    return coverage


def activation_info(data):
    """Extract metadata from single-layer and all-layer activation files."""
    acts = data["activations"]
    ids = data.get("ids", [])

    n_texts = data.get("n_texts")
    d_model = data.get("d_model")
    n_layers = data.get("n_layers")
    layer = data.get("layer")

    if isinstance(acts, dict):
        kind = "all_layers"
        if n_layers is None:
            n_layers = len(acts)
        if acts:
            first = next(iter(acts.values()))
            if n_texts is None:
                n_texts = first.shape[0]
            if d_model is None:
                d_model = first.shape[1]
    else:
        kind = "single_layer"
        if n_texts is None:
            n_texts = len(ids) if ids else acts.shape[0]
        if d_model is None:
            d_model = acts.shape[1]

    return {
        "n_texts": n_texts,
        "d_model": d_model,
        "layer": layer,
        "n_layers": n_layers,
        "kind": kind,
    }


def check_activations():
    """Check extracted activation files."""
    activations = {}
    if ACTIVATIONS_DIR.exists():
        for path in sorted(ACTIVATIONS_DIR.glob("*.pt")):
            import torch
            data = torch.load(path, weights_only=True, map_location="cpu")
            activations[path.stem] = activation_info(data)
    return activations


def check_activation_coverage(corpus_ids=None):
    """Compare activation IDs against generated corpus IDs."""
    if corpus_ids is None:
        corpus_ids = generated_corpus_ids()

    coverage = {}
    if ACTIVATIONS_DIR.exists():
        for path in sorted(ACTIVATIONS_DIR.glob("*.pt")):
            import torch
            data = torch.load(path, weights_only=True, map_location="cpu")
            ids = set(data.get("ids", []))
            coverage[path.stem] = {
                "missing": sorted(corpus_ids - ids),
                "extra": sorted(ids - corpus_ids),
            }
    return coverage


def check_augmented():
    """Check augmented direction data."""
    augmented = {}
    if AUGMENTED_DIR.exists():
        for path in sorted(AUGMENTED_DIR.glob("*.pt")):
            import torch
            data = torch.load(path, weights_only=False, map_location="cpu")
            augmented[path.stem] = {
                "n_contrastive": data.get("n_contrastive", 0),
                "n_sparse": data.get("n_sparse", 0),
                "total": len(data.get("vectors", [])),
            }
    return augmented


def check_trained_models():
    """Check trained AV/AR adapters."""
    trained = {}
    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.iterdir()):
            if not path.is_dir():
                continue
            meta_path = path / "nla_meta.yaml"
            if meta_path.exists():
                meta = yaml.safe_load(meta_path.read_text())
                trained[path.name] = {
                    "role": meta.get("role"),
                    "stage": meta.get("stage"),
                    "layer": meta.get("extraction_layer_index"),
                    "training": meta.get("training", {}),
                }
            elif (path / "adapter_model.safetensors").exists():
                trained[path.name] = {"role": "unknown", "stage": "in_progress"}
    return trained


def readiness_for(model_key, layer):
    """Check what's ready for a specific model+layer combination."""
    if model_key not in MODELS:
        return f"Unknown model: {model_key}"

    _, n_layers = MODELS[model_key]
    layer_pct = int(layer * 100 / n_layers)

    checks = []

    # Texts
    cats, total, missing, _ = check_corpus()
    if total > 0:
        checks.append(("Corpus texts", "OK", f"{total} texts"))
    else:
        checks.append(("Corpus texts", "MISSING", "Run: generate_corpus.py"))

    corpus_ids = generated_corpus_ids()

    # Descriptions at this depth
    depths, expansion, depth_ids, expansion_ids = check_descriptions()
    if layer_pct in depth_ids or layer_pct in expansion_ids:
        unique_ids = depth_ids.get(layer_pct, set()) | expansion_ids.get(layer_pct, set())
        missing_n = len(corpus_ids - unique_ids)
        extra_n = len(unique_ids - corpus_ids)
        status = "PARTIAL" if missing_n or extra_n else "OK"
        detail = (
            f"{len(unique_ids)} unique at {layer_pct}% "
            f"({depths.get(layer_pct, 0)} main + {expansion.get(layer_pct, 0)} expansion)"
        )
        if missing_n or extra_n:
            detail += f"; missing {missing_n}, stale {extra_n}"
        checks.append(("Descriptions", status, detail))
    else:
        checks.append(("Descriptions", "MISSING",
                       f"Run: generate_corpus.py --describe {layer_pct}"))

    # Activations
    activations = check_activations()
    activation_coverage = check_activation_coverage(corpus_ids)
    act_key = f"{model_key}_L{layer}"
    all_key = f"{model_key}_all_layers"
    if act_key in activations:
        a = activations[act_key]
        missing_n = len(activation_coverage.get(act_key, {}).get("missing", []))
        extra_n = len(activation_coverage.get(act_key, {}).get("extra", []))
        status = "PARTIAL" if missing_n or extra_n else "OK"
        detail = f"{a['n_texts']} × {a['d_model']}"
        if missing_n or extra_n:
            detail += f"; missing {missing_n}, stale {extra_n}"
        checks.append(("Activations", status, detail))
    elif all_key in activations and layer < (activations[all_key].get("n_layers") or 0):
        a = activations[all_key]
        missing_n = len(activation_coverage.get(all_key, {}).get("missing", []))
        extra_n = len(activation_coverage.get(all_key, {}).get("extra", []))
        status = "PARTIAL" if missing_n or extra_n else "OK"
        detail = f"{a['n_layers']} layers × {a['n_texts']} × {a['d_model']}"
        if missing_n or extra_n:
            detail += f"; missing {missing_n}, stale {extra_n}"
        checks.append(("Activations", status, detail))
    else:
        checks.append(("Activations", "MISSING",
                       f"Run: extract_activations.py --model {model_key} --layer {layer}"))

    # Augmented directions
    augmented = check_augmented()
    aug_key = f"{model_key}_L{layer}_directions"
    if aug_key in augmented:
        a = augmented[aug_key]
        checks.append(("Augmented", "OK", f"{a['total']} vectors"))
    else:
        checks.append(("Augmented", "OPTIONAL",
                       f"Run: augment_directions.py --describe"))

    # Trained models
    trained = check_trained_models()
    av_found = any("av" in k and model_key in k for k in trained)
    ar_found = any("ar" in k and model_key in k for k in trained)
    if av_found:
        checks.append(("AV adapter", "OK", "trained"))
    else:
        checks.append(("AV adapter", "MISSING", "Run: train_av.py"))
    if ar_found:
        checks.append(("AR adapter", "OK", "trained"))
    else:
        checks.append(("AR adapter", "MISSING", "Run: train_ar.py"))

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("NLA-at-home Pipeline Status")
    print("=" * 60)

    # Corpus
    cats, total, missing, unsafe_count = check_corpus()
    print(f"\nCorpus: {total} texts across {len(cats)} categories "
          f"({unsafe_count} unsafe)")
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    # Descriptions
    depths, expansion, depth_ids, expansion_ids = check_descriptions()
    desc_coverage = check_description_coverage(depth_ids, expansion_ids)
    print(f"\nDescriptions:")
    for pct in sorted(set(list(depths.keys()) + list(expansion.keys()))):
        main_n = depths.get(pct, 0)
        exp_n = expansion.get(pct, 0)
        unique_n = len(depth_ids.get(pct, set()) | expansion_ids.get(pct, set()))
        print(f"  L{pct}%: {main_n} main +{exp_n} expansion ({unique_n} unique after merge)")
        coverage = desc_coverage.get(pct)
        if coverage:
            missing_n = len(coverage["missing"])
            extra_n = len(coverage["extra"])
            if missing_n or extra_n:
                print(f"    coverage: missing {missing_n} corpus ids, stale {extra_n} ids")

    # Activations
    activations = check_activations()
    act_coverage = check_activation_coverage()
    print(f"\nActivations:")
    if activations:
        for name, info in activations.items():
            if info.get("kind") == "all_layers":
                print(f"  {name}: {info['n_layers']} layers × "
                      f"{info['n_texts']} × {info['d_model']}")
            else:
                print(f"  {name}: {info['n_texts']} × {info['d_model']}")
            coverage = act_coverage.get(name)
            if coverage:
                missing_n = len(coverage["missing"])
                extra_n = len(coverage["extra"])
                if missing_n or extra_n:
                    print(f"    coverage: missing {missing_n} corpus ids, stale {extra_n} ids")
    else:
        print("  None extracted yet")

    # Augmented
    augmented = check_augmented()
    if augmented:
        print(f"\nAugmented:")
        for name, info in augmented.items():
            print(f"  {name}: {info['n_contrastive']} contrastive + "
                  f"{info['n_sparse']} sparse = {info['total']}")

    # Trained models
    trained = check_trained_models()
    if trained:
        print(f"\nTrained adapters:")
        for name, info in trained.items():
            stage = info.get("stage", "?")
            role = info.get("role", "?")
            t = info.get("training", {})
            detail = ""
            if "best_val_loss" in t:
                detail = f" val_loss={t['best_val_loss']:.3f}"
            elif "best_val_mse" in t:
                detail = f" val_mse={t['best_val_mse']:.4f} cos={t.get('best_val_cosine', 0):.4f}"
            print(f"  {name}: {role}/{stage}{detail}")

    # Specific model readiness
    if args.model and args.layer:
        print(f"\n{'─' * 60}")
        print(f"Readiness: {args.model} L{args.layer}")
        print(f"{'─' * 60}")
        checks = readiness_for(args.model, args.layer)
        for name, status, detail in checks:
            icon = (
                "✓" if status == "OK"
                else "○" if status == "OPTIONAL"
                else "!" if status == "PARTIAL"
                else "✗"
            )
            print(f"  {icon} {name}: {detail}")

    # Available models
    print(f"\n{'─' * 60}")
    print("Supported models:")
    for key, (hf_name, n_layers) in MODELS.items():
        print(f"  {key}: {hf_name} ({n_layers} layers)")


if __name__ == "__main__":
    main()
