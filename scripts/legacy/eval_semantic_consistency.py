#!/usr/bin/env python3
"""
Evaluate semantic consistency of the NLA.

Measures how well the NLA preserves activation geometry by comparing
the cosine similarity of activation pairs with the semantic similarity
of their generated descriptions.

Metric: Spearman correlation between activation-cosine and description-cosine.
Visual: Scatter plot of activation-cosine vs description-cosine.

Usage:
  python3 scripts/eval_semantic_consistency.py \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --descriptions corpus/generated/descriptions_L71pct.json \
    --output evaluation/semantic_consistency_L20.png \
    --n-pairs 1000
"""
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

REPO_ROOT = Path(__file__).parent.parent


def load_data(act_path, desc_path):
    print(f"Loading activations from {act_path}...")
    act_data = torch.load(act_path, weights_only=False)
    activations = act_data["activations"]
    ids = act_data["ids"]


    print(f"Loading descriptions from {desc_path}...")
    all_descs = json.loads(Path(desc_path).read_text())
    
    # Map ID to description
    # Handle both main generated files and description files which might have different keys
    id_to_desc = {}
    for item in all_descs:
        # Priority: 'description' field (NLA output), then 'text' (source corpus)
        desc = item.get("description") or item.get("summary")
        if desc:
            id_to_desc[item["id"]] = desc

    matched_acts = []
    matched_descs = []
    matched_ids = []
    
    for i, text_id in enumerate(ids):
        if text_id in id_to_desc:
            matched_acts.append(activations[i])
            matched_descs.append(id_to_desc[text_id])
            matched_ids.append(text_id)

    print(f"Matched {len(matched_descs)} examples.")
    return torch.stack(matched_acts), matched_descs, matched_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True, help="Path to activation .pt file")
    parser.add_argument("--descriptions", required=True, help="Path to description .json file")
    parser.add_argument("--output", default="evaluation/semantic_consistency.png", help="Path to save plot")
    parser.add_argument("--n-pairs", type=int, default=1000, help="Number of random pairs to sample")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    
    acts, descs, ids = load_data(args.activations, args.descriptions)
    n_samples = len(acts)
    
    if n_samples < 2:
        print("Not enough samples to evaluate.")
        return

    print(f"Loading SentenceTransformer {args.model}...")
    model = SentenceTransformer(args.model)

    print(f"Sampling {args.n_pairs} pairs...")
    pair_indices = []
    for _ in range(args.n_pairs):
        i, j = rng.choice(n_samples, 2, replace=False)
        pair_indices.append((i, j))

    idx_a = [p[0] for p in pair_indices]
    idx_b = [p[1] for p in pair_indices]

    # Compute activation similarities
    print("Computing activation similarities...")
    act_sims = cosine_similarity(acts[idx_a], acts[idx_b]).numpy()

    # Compute description similarities
    print("Computing description embeddings...")
    # Get unique descriptions to embed (save computation)
    unique_descs = list(set(descs))
    desc_to_idx = {d: i for i, d in enumerate(unique_descs)}
    
    embeddings = model.encode(unique_descs, convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing description similarities...")
    # Map back to pairs
    emb_a = embeddings[[desc_to_idx[descs[i]] for i in idx_a]]
    emb_b = embeddings[[desc_to_idx[descs[i]] for i in idx_b]]
    desc_sims = cosine_similarity(emb_a, emb_b).cpu().numpy()

    # Calculate correlation
    rho, p_val = spearmanr(act_sims, desc_sims)
    print(f"\nResults:")
    print(f"  Spearman correlation: {rho:.4f}")
    print(f"  p-value: {p_val:.4e}")

    # Plot
    print(f"Saving plot to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(act_sims, desc_sims, alpha=0.3, s=10)
    
    # Add trend line
    m, b = np.polyfit(act_sims, desc_sims, 1)
    plt.plot(act_sims, m*act_sims + b, color='red', linestyle='--', label=f'Trend (r={rho:.3f})')
    
    plt.xlabel("Activation Cosine Similarity")
    plt.ylabel("Description Semantic Similarity")
    plt.title(f"NLA Semantic Consistency (N={args.n_pairs})\nSpearman ρ = {rho:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(args.output, dpi=300)
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
