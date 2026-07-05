#!/usr/bin/env python3
"""
Generate direction-like training data from existing activations.

Creates three types of augmented training examples:
1. Contrastive directions: normalized difference between cross-category activation pairs
2. PCA-sparse activations: projections onto top-k principal components
3. Scaled unit directions: random directions at typical activation norm

Each augmented activation gets a DeepSeek-generated description
explaining the contrast or sparsity.

Usage:
  python3 scripts/augment_directions.py \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --layer-pct 71 \
    --n-contrastive 300 \
    --n-sparse 200 \
    --output corpus/augmented/qwen25-7b_L20_directions.pt
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from generate_corpus import BACKENDS, get_client

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"


def load_activations_with_categories(act_path):
    data = torch.load(act_path, weights_only=True)
    acts = data["activations"]
    ids = data["ids"]

    # Load texts to get categories
    id_to_cat = {}
    id_to_text = {}
    for path in sorted(GENERATED_DIR.glob("*.json")):
        if path.name.startswith("descriptions_"):
            continue
        items = json.loads(path.read_text())
        for item in items:
            id_to_cat[item["id"]] = item["category"]
            id_to_text[item["id"]] = item["text"]

    categories = [id_to_cat.get(i, "unknown") for i in ids]
    texts = [id_to_text.get(i, "") for i in ids]
    return acts, ids, categories, texts


def generate_contrastive_pairs(acts, ids, categories, texts, n_pairs, rng):
    """Generate normalized difference vectors between cross-category pairs."""
    cat_indices = {}
    for i, cat in enumerate(categories):
        cat_indices.setdefault(cat, []).append(i)

    cat_names = list(cat_indices.keys())
    pairs = []
    vectors = []

    for _ in range(n_pairs):
        cat_a, cat_b = rng.choice(cat_names, 2, replace=False)
        idx_a = rng.choice(cat_indices[cat_a])
        idx_b = rng.choice(cat_indices[cat_b])

        diff = acts[idx_a] - acts[idx_b]
        norm = diff.norm()
        if norm < 1e-6:
            continue
        direction = diff / norm

        pairs.append({
            "type": "contrastive",
            "text_a": texts[idx_a][:200],
            "text_b": texts[idx_b][:200],
            "cat_a": cat_a,
            "cat_b": cat_b,
            "id_a": ids[idx_a],
            "id_b": ids[idx_b],
        })
        vectors.append(direction)

    return vectors, pairs


def generate_pca_sparse(acts, ids, categories, texts, n_samples, rng):
    """Project activations onto top-k PCA components."""
    centered = acts - acts.mean(0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    vectors = []
    metas = []

    indices = rng.choice(len(acts), n_samples, replace=True)
    ks = rng.choice([1, 3, 5, 10], n_samples)

    for idx, k in zip(indices, ks):
        act = centered[idx]
        # Project onto top-k PCA directions
        top_k_dirs = Vt[:k]  # (k, d_model)
        coeffs = act @ top_k_dirs.T  # (k,)
        sparse = (coeffs.unsqueeze(1) * top_k_dirs).sum(0)  # (d_model,)
        sparse = sparse + acts.mean(0)  # add back mean

        vectors.append(sparse)
        metas.append({
            "type": "pca_sparse",
            "k": int(k),
            "source_id": ids[idx],
            "source_cat": categories[idx],
            "source_text": texts[idx][:200],
        })

    return vectors, metas


def generate_descriptions_for_augmented(client, metas, layer_pct, model_name):
    """Generate descriptions for augmented training examples."""
    desc_system = (REPO_ROOT / "prompts" / "describe_system.txt").read_text().strip()
    described = []

    for i, meta in enumerate(metas):
        if meta["type"] == "contrastive":
            user_msg = f"""Layer depth: {layer_pct}% of total network depth.

Two texts are being processed. This activation represents the DIRECTION from text B toward text A — what makes A's processing different from B's at this layer.

Text A ({meta['cat_a']}): "{meta['text_a']}"
Text B ({meta['cat_b']}): "{meta['text_b']}"

Write:
DESCRIPTION: [2-4 sentences about what processing quality this direction captures — what shifts when moving from B to A at this depth]
SUMMARY: [1 short sentence, under 20 words]"""

        elif meta["type"] == "pca_sparse":
            user_msg = f"""Layer depth: {layer_pct}% of total network depth.

This is a simplified view of the model's processing, showing only the {meta['k']} strongest processing dimensions. The input being processed:

"{meta['source_text']}"

Category: {meta['source_cat']}

Write:
DESCRIPTION: [2-4 sentences about the dominant processing quality at this layer depth, focusing on the {meta['k']} most salient features]
SUMMARY: [1 short sentence, under 20 words]"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": desc_system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            desc = raw.split("DESCRIPTION:")[-1].split("SUMMARY:")[0].strip()
            summ = raw.split("SUMMARY:")[-1].strip()
            meta["description"] = desc
            meta["summary"] = summ
        except Exception as e:
            print(f"  error on {i}: {e}")
            meta["description"] = f"Processing direction between {meta.get('cat_a', 'unknown')} and {meta.get('cat_b', 'unknown')}."
            meta["summary"] = "Direction vector."

        described.append(meta)
        if (i + 1) % 50 == 0:
            print(f"  described {i+1}/{len(metas)}")

    return described


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--layer-pct", type=int, required=True)
    parser.add_argument("--n-contrastive", type=int, default=300)
    parser.add_argument("--n-sparse", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--describe", action="store_true",
                        help="Generate descriptions via the selected LLM backend")
    parser.add_argument("--backend", default="deepseek", choices=list(BACKENDS.keys()))
    parser.add_argument("--api-url", default=None, help="Override backend URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default=None,
                        help="Override model name (default: from backend config)")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    act_path = Path(args.activations)
    if not act_path.is_absolute():
        act_path = REPO_ROOT / act_path

    print("Loading activations...")
    acts, ids, categories, texts = load_activations_with_categories(act_path)
    mean_norm = float(acts.norm(dim=1).mean())
    print(f"  {len(acts)} activations, mean norm {mean_norm:.1f}")

    print(f"\nGenerating {args.n_contrastive} contrastive directions...")
    cont_vecs, cont_metas = generate_contrastive_pairs(
        acts, ids, categories, texts, args.n_contrastive, rng)
    # Scale to typical activation norm
    cont_vecs = [v * mean_norm for v in cont_vecs]
    print(f"  got {len(cont_vecs)} directions")

    print(f"\nGenerating {args.n_sparse} PCA-sparse activations...")
    sparse_vecs, sparse_metas = generate_pca_sparse(
        acts, ids, categories, texts, args.n_sparse, rng)
    print(f"  got {len(sparse_vecs)} sparse activations")

    all_vecs = cont_vecs + sparse_vecs
    all_metas = cont_metas + sparse_metas

    if args.describe:
        client, default_model = get_client(args.api_key, args.api_url, args.backend)
        model_name = args.model_name or default_model
        print(f"Backend: {args.backend}, model: {model_name}")
        print(f"\nGenerating descriptions for {len(all_metas)} augmented examples...")
        all_metas = generate_descriptions_for_augmented(
            client, all_metas, args.layer_pct, model_name)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "vectors": torch.stack(all_vecs),
        "metas": all_metas,
        "source_activations": str(act_path),
        "mean_norm": mean_norm,
        "n_contrastive": len(cont_vecs),
        "n_sparse": len(sparse_vecs),
    }, out_path)

    desc_path = out_path.with_suffix(".json")
    json.dump(all_metas, open(desc_path, "w"), indent=2)

    print(f"\nSaved {len(all_vecs)} augmented vectors to {out_path}")
    print(f"Saved {len(all_metas)} metas to {desc_path}")

    # Quick stats
    cont_norms = torch.stack(cont_vecs).norm(dim=1)
    sparse_norms = torch.stack(sparse_vecs).norm(dim=1)
    print(f"\nContrastive norms: mean={cont_norms.mean():.1f} (target: {mean_norm:.1f})")
    print(f"PCA-sparse norms: mean={sparse_norms.mean():.1f}")


if __name__ == "__main__":
    main()
