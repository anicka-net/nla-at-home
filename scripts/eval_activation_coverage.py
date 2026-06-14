#!/usr/bin/env python3
"""
Evaluate activation space coverage and inter-layer similarity.

Analyzes how well the corpus covers the activation space at each layer
and identifies redundant or novel transition layers.

Outputs:
- Console summary table
- UMAP plots for representative layers
- Inter-layer similarity heatmap
- JSON metrics file

Usage:
  python3 scripts/eval_activation_coverage.py \
    --activations corpus/activations/gemma3-1b_all_layers.pt \
    --output-dir evaluation/coverage/
"""
import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
import umap
from tabulate import tabulate

def get_category(text_id):
    """Extract category from ID (e.g., A01_code_003 -> A01_code)."""
    parts = text_id.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "unknown"

def analyze_layer(activations, categories, layer_idx, n_clusters=20):
    """Perform PCA, k-means, and similarity analysis on a single layer."""
    acts_np = activations.numpy()
    
    # PCA
    pca = PCA(n_components=min(50, acts_np.shape[0], acts_np.shape[1]))
    pca.fit(acts_np)
    exp_var = pca.explained_variance_ratio_
    pca_metrics = {
        "top1": float(exp_var[0]) if len(exp_var) >= 1 else 0,
        "top5": float(exp_var[:5].sum()) if len(exp_var) >= 5 else 0,
        "top10": float(exp_var[:10].sum()) if len(exp_var) >= 10 else 0,
        "top50": float(exp_var[:50].sum()) if len(exp_var) >= 50 else 0,
    }
    
    # k-means gap detection
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(acts_np)
    cluster_counts = np.bincount(clusters, minlength=n_clusters)
    gaps = [int(i) for i, count in enumerate(cluster_counts) if count < 10]
    
    # Inter-text cosine similarity
    # To save time/memory, if N is large, we sample. But for ~1200 texts it's fine.
    sim_matrix = cosine_similarity(acts_np)
    # Mask diagonal
    np.fill_diagonal(sim_matrix, np.nan)
    sim_flat = sim_matrix[~np.isnan(sim_matrix)]
    
    sim_metrics = {
        "mean": float(np.mean(sim_flat)),
        "std": float(np.std(sim_flat)),
        "min": float(np.min(sim_flat)),
        "max": float(np.max(sim_flat)),
    }
    
    return {
        "pca": pca_metrics,
        "gaps": {
            "indices": gaps,
            "count": len(gaps),
            "small_clusters_info": [int(cluster_counts[i]) for i in gaps]
        },
        "similarity": sim_metrics
    }

def plot_umap(activations, categories, layer_idx, output_path):
    """Generate and save UMAP plot for a layer."""
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(activations.numpy())
    
    unique_cats = sorted(list(set(categories)))
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    colors = [cat_to_idx[cat] for cat in categories]
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='Spectral', s=15, alpha=0.6)
    plt.title(f"UMAP Projection - Layer {layer_idx}")
    plt.colorbar(scatter, label='Category Index')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True, help="Path to gemma3-1b_all_layers.pt")
    parser.add_argument("--output-dir", default="evaluation/coverage/", help="Output directory")
    parser.add_argument("--umap-layers", type=int, nargs='+', default=[0, 6, 13, 19, 25], help="Layers to plot UMAP")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading activations from {args.activations}...")
    # Use weights_only=False because the file contains lists (ids)
    data = torch.load(args.activations, weights_only=False)
    
    all_acts = data["activations"]
    ids = data["ids"]
    n_layers = data["n_layers"]
    d_model = data["d_model"]
    n_texts = data["n_texts"]
    
    categories = [get_category(tid) for tid in ids]
    
    layer_metrics = {}
    table_data = []
    
    print(f"Analyzing {n_layers} layers...")
    for l in range(n_layers):
        metrics = analyze_layer(all_acts[l], categories, l)
        layer_metrics[l] = metrics
        
        pca = metrics["pca"]
        table_data.append([
            l, 
            f"{pca['top1']:.1%}", 
            f"{pca['top5']:.1%}", 
            f"{pca['top10']:.1%}", 
            f"{pca['top50']:.1%}",
            metrics["gaps"]["count"],
            f"{metrics['similarity']['mean']:.3f}"
        ])
        
        if l in args.umap_layers:
            print(f"  Generating UMAP for layer {l}...")
            plot_umap(all_acts[l], categories, l, output_dir / f"layer_{l}_umap.png")

    # Console Summary Table
    headers = ["Layer", "PCA top-1", "PCA top-5", "PCA top-10", "PCA top-50", "Gaps (<10)", "Sim Mean"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    # Inter-layer similarity matrix
    print("\nComputing inter-layer similarity matrix...")
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    # Convert to float32 for faster cosine sim if needed, but torch is fast
    # For each text, compute cosine between its layer-i and layer-j activations, then average
    # This is slightly different from computing cosine between entire layer matrices
    
    # Optimize: pre-normalize activations
    norm_acts = {}
    for l in range(n_layers):
        l_act = all_acts[l].float()
        norm_acts[l] = l_act / (l_act.norm(dim=1, keepdim=True) + 1e-8)

    for i in range(n_layers):
        for j in range(i, n_layers):
            # Mean of dot products of normalized vectors
            cos_sim = (norm_acts[i] * norm_acts[j]).sum(dim=1).mean().item()
            similarity_matrix[i, j] = cos_sim
            similarity_matrix[j, i] = cos_sim

    # Identify redundant layers
    redundant = []
    for i in range(n_layers - 1):
        if similarity_matrix[i, i+1] > 0.95:
            redundant.append((i, i+1, similarity_matrix[i, i+1]))

    if redundant:
        print("\nRedundant Layer Pairs (Cosine > 0.95):")
        for i, j, val in redundant:
            print(f"  Layer {i} & {j}: {val:.4f}")
    else:
        print("\nNo redundant layer pairs (Cosine > 0.95) found.")

    # Identify novel layers (lowest mean similarity to adjacent layers)
    novelty = []
    for i in range(1, n_layers - 1):
        # Average similarity to previous and next layer
        adj_sim = (similarity_matrix[i, i-1] + similarity_matrix[i, i+1]) / 2
        novelty.append((i, adj_sim))
    
    novelty.sort(key=lambda x: x[1])
    print("\nTop 5 Most 'Novel' Transition Layers (lowest similarity to neighbors):")
    for l, score in novelty[:5]:
        print(f"  Layer {l}: {score:.4f}")

    # Plot Inter-layer Similarity Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis', xticklabels=5, yticklabels=5)
    plt.title("Inter-Layer Cosine Similarity Matrix")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.savefig(output_dir / "inter_layer_similarity.png", dpi=300)
    plt.close()

    # Save summary JSON
    summary = {
        "model_info": {
            "n_layers": n_layers,
            "d_model": d_model,
            "n_texts": n_texts
        },
        "layer_metrics": layer_metrics,
        "inter_layer_similarity": similarity_matrix.tolist(),
        "redundant_pairs": redundant,
        "novel_layers": novelty
    }
    
    with open(output_dir / "coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
