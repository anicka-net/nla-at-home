#!/usr/bin/env python3
"""
SAE Direction Validator — standard post-extraction health check.

Given an extracted direction vector and matching SAE, produces a validation
report answering: "Is this direction capturing what we think it captures?"

Runs three analyses:
1. Decomposition: cosine similarity against all SAE decoder columns
2. Subspace capture: how many SAE features to explain the direction?
3. Tuning curves: do top features fire selectively on the expected stimuli?

Usage:
  python3 sae_validate_direction.py \
    --direction path/to/direction_unit.pt \
    --stimuli path/to/stimuli.yaml \
    --sae-release llama_scope_lxr_32x \
    --sae-id l20r_32x \
    --model meta-llama/Llama-3.1-8B \
    --layer 20 \
    --axis-name vedana \
    --output-dir data/sae-decomposition/llama-8b/

Stimuli YAML formats supported:
  - vedana: {pleasant: [...], unpleasant: [...], neutral: [...]}
  - factorial: {cells: [{id, identity_pressure, harmful_intent, register, prompts}]}
  - generic: {stimuli: [{id, text, category}]}
"""
import torch
import torch.nn.functional as F
import yaml
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sae(release, sae_id):
    from sae_lens import SAE
    return SAE.from_pretrained(release=release, sae_id=sae_id)


def load_stimuli(path):
    data = yaml.safe_load(open(path))
    prompts = []
    if 'vedana' in data:
        for item in data['vedana'].get('pleasant', []):
            prompts.append({'id': item['id'], 'text': item['text'], 'category': 'pleasant'})
        for item in data['vedana'].get('unpleasant', []):
            prompts.append({'id': item['id'], 'text': item['text'], 'category': 'unpleasant'})
        for item in data['vedana'].get('neutral', []):
            prompts.append({'id': item['id'], 'text': item['text'], 'category': 'neutral'})
    elif 'cells' in data:
        for cell in data['cells']:
            cell_id = cell['id']
            ip = cell.get('identity_pressure', 'low')
            hi = cell.get('harmful_intent', 'low')
            reg = cell.get('register', 'neutral')
            category = f'ip={ip}_hi={hi}_{reg}'
            for j, text in enumerate(cell.get('prompts', [])):
                prompts.append({'id': f'{cell_id}_{j}', 'text': text, 'category': category})
    elif 'stimuli' in data:
        for item in data['stimuli']:
            prompts.append({'id': item['id'], 'text': item['text'],
                           'category': item.get('category', 'unknown')})
    return prompts


# ── Phase 1: Decomposition ──────────────────────────────────────────

def decompose_direction(direction, sae, top_k=50):
    W_dec = sae.W_dec.data.float()
    d_vec = direction.float()
    cos_sims = F.cosine_similarity(d_vec.unsqueeze(0), W_dec, dim=1)

    top_pos = cos_sims.topk(top_k)
    top_neg = (-cos_sims).topk(top_k)

    results = []
    for idx, cos in zip(top_pos.indices.tolist(), top_pos.values.tolist()):
        results.append({"feature_id": idx, "cosine": round(cos, 4), "sign": "positive"})
    for idx, cos in zip(top_neg.indices.tolist(), top_neg.values.tolist()):
        results.append({"feature_id": idx, "cosine": round(-cos, 4), "sign": "negative"})

    results.sort(key=lambda x: abs(x["cosine"]), reverse=True)

    cos_stats = {
        "max": round(cos_sims.max().item(), 4),
        "min": round(cos_sims.min().item(), 4),
        "mean_abs": round(cos_sims.abs().mean().item(), 4),
        "n_above_0.1": int((cos_sims.abs() > 0.1).sum().item()),
        "n_above_0.2": int((cos_sims.abs() > 0.2).sum().item()),
        "n_above_0.3": int((cos_sims.abs() > 0.3).sum().item()),
    }
    return results[:top_k], cos_stats, cos_sims


def greedy_subspace_capture(direction, sae, max_features=100):
    W_dec = sae.W_dec.data.float()
    d_vec = direction.float().clone()
    original_norm_sq = d_vec.norm() ** 2

    captured = []
    residual = d_vec.clone()

    for i in range(max_features):
        projections = residual @ W_dec.T
        best_idx = projections.abs().argmax().item()
        best_decoder = W_dec[best_idx]
        best_decoder_norm = best_decoder / best_decoder.norm()
        proj_scalar = residual @ best_decoder_norm
        residual = residual - proj_scalar * best_decoder_norm
        variance_explained = 1 - (residual.norm() ** 2 / original_norm_sq).item()
        captured.append({
            "step": i + 1,
            "feature_id": best_idx,
            "variance_explained": round(variance_explained, 6),
        })
        if variance_explained > 0.99:
            break

    return captured


def random_baseline(d_model, sae, max_features=100, n_trials=10):
    curves = []
    for _ in range(n_trials):
        rand_dir = torch.randn(d_model)
        rand_dir = rand_dir / rand_dir.norm()
        captured = greedy_subspace_capture(rand_dir, sae, max_features)
        curve = [c["variance_explained"] for c in captured]
        curves.append(curve)
    max_len = max(len(c) for c in curves)
    mean_curve = []
    for i in range(max_len):
        vals = [c[i] for c in curves if i < len(c)]
        mean_curve.append(round(sum(vals) / len(vals), 6))
    return mean_curve


# ── Phase 2: Tuning curves ──────────────────────────────────────────

def extract_activations(model, tokenizer, prompts, layer, device):
    blocks = model.model.layers
    results = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt['text'], return_tensors='pt', truncation=True,
                          max_length=512).to(device)
        layer_out = {}
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out['h'] = h.detach()
        handle = blocks[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        activation = layer_out['h'][0, -1, :].cpu().float()
        results.append({**prompt, 'activation': activation})
        if (i + 1) % 20 == 0:
            print(f'  Extracted {i+1}/{len(prompts)}')
    return results


def compute_tuning_data(results, direction, sae, top_feature_ids):
    output = []
    for r in results:
        act = r['activation']
        proj = float(act @ direction)
        sae_feats = sae.encode(act.unsqueeze(0)).squeeze(0).detach()
        top_acts = {str(fid): round(float(sae_feats[fid]), 4) for fid in top_feature_ids}

        recon = sae.decode(sae.encode(act.unsqueeze(0))).squeeze(0).detach()
        recon_cos = float(F.cosine_similarity(act.unsqueeze(0), recon.unsqueeze(0)))

        output.append({
            'id': r['id'], 'category': r['category'],
            'projection': round(proj, 4),
            'top_feature_activations': top_acts,
            'sae_recon_cosine': round(recon_cos, 4),
        })
    output.sort(key=lambda x: x['projection'])
    return output


# ── Report generation ────────────────────────────────────────────────

def generate_report(axis_name, decomp_features, cos_stats, capture,
                    baseline, tuning_data, top_feature_ids):
    lines = []
    lines.append(f"# SAE Validation Report: {axis_name}")
    lines.append("")

    lines.append("## 1. Decomposition")
    lines.append(f"- Max cosine with any SAE feature: {cos_stats['max']}")
    lines.append(f"- Features with |cos| > 0.1: {cos_stats['n_above_0.1']}")
    lines.append(f"- Features with |cos| > 0.2: {cos_stats['n_above_0.2']}")
    lines.append(f"- Features with |cos| > 0.3: {cos_stats['n_above_0.3']}")
    lines.append("")

    lines.append("Top 10 aligned features:")
    lines.append(f"{'Feature':>10} {'Cosine':>8} {'Sign':>10}")
    for f in decomp_features[:10]:
        lines.append(f"{'#'+str(f['feature_id']):>10} {f['cosine']:>+8.4f} {f['sign']:>10}")
    lines.append("")

    lines.append("## 2. Subspace Capture")
    milestones = [1, 5, 10, 20, 50, 100]
    lines.append(f"{'Features':>10} {'Direction':>10} {'Random':>10} {'Ratio':>8}")
    for m in milestones:
        if m <= len(capture):
            d_val = capture[m-1]['variance_explained']
        else:
            d_val = capture[-1]['variance_explained'] if capture else 0
        r_val = baseline[m-1] if m <= len(baseline) else baseline[-1] if baseline else 0
        ratio = d_val / r_val if r_val > 0 else float('inf')
        lines.append(f"{m:>10} {d_val*100:>9.1f}% {r_val*100:>9.1f}% {ratio:>7.1f}x")
    lines.append("")

    concentration = "high" if cos_stats['n_above_0.2'] >= 5 else "medium" if cos_stats['n_above_0.1'] >= 50 else "low"
    ratio_100 = capture[-1]['variance_explained'] / baseline[-1] if baseline and baseline[-1] > 0 else 0
    reality = "strong" if ratio_100 > 2.0 else "moderate" if ratio_100 > 1.5 else "weak"
    lines.append(f"**Concentration**: {concentration} (max cos {cos_stats['max']:.3f})")
    lines.append(f"**Non-randomness**: {reality} ({ratio_100:.1f}x random at 100 features)")
    lines.append("")

    if tuning_data:
        lines.append("## 3. Tuning Curves")
        categories = defaultdict(list)
        for item in tuning_data:
            categories[item['category']].append(item)

        lines.append("")
        lines.append("### Projection by category")
        for cat, items in sorted(categories.items(),
                                  key=lambda x: sum(i['projection'] for i in x[1])/len(x[1])):
            vals = [i['projection'] for i in items]
            mean_v = sum(vals) / len(vals)
            lines.append(f"  {cat}: mean={mean_v:+.3f}, n={len(vals)}, "
                        f"range=[{min(vals):+.2f}, {max(vals):+.2f}]")

        lines.append("")
        lines.append("### Top feature selectivity")
        for fid in top_feature_ids[:6]:
            fid_str = str(fid)
            lines.append(f"  Feature #{fid}:")
            for cat, items in sorted(categories.items()):
                vals = [i['top_feature_activations'].get(fid_str, 0) for i in items]
                n_fire = sum(1 for v in vals if v > 0.01)
                mean_v = sum(vals) / len(vals)
                lines.append(f"    {cat}: fires {n_fire}/{len(items)}, mean={mean_v:.3f}")

        if len(categories) >= 2:
            lines.append("")
            lines.append("### Factor analysis")
            all_items = tuning_data
            factors = defaultdict(lambda: defaultdict(list))
            for item in all_items:
                cat = item['category']
                for part in cat.replace('=', '_').split('_'):
                    pass
                if 'ip=high' in cat:
                    factors['identity_pressure']['high'].append(item['projection'])
                elif 'ip=low' in cat:
                    factors['identity_pressure']['low'].append(item['projection'])
                if 'hi=high' in cat:
                    factors['harmful_intent']['high'].append(item['projection'])
                elif 'hi=low' in cat:
                    factors['harmful_intent']['low'].append(item['projection'])
                if 'adversarial' in cat:
                    factors['register']['adversarial'].append(item['projection'])
                elif 'neutral' in cat:
                    factors['register']['neutral'].append(item['projection'])
                if cat in ('pleasant', 'unpleasant', 'neutral'):
                    factors['valence'][cat].append(item['projection'])

            for factor_name, levels in factors.items():
                if len(levels) >= 2:
                    level_means = {k: sum(v)/len(v) for k, v in levels.items()}
                    sorted_levels = sorted(level_means.items(), key=lambda x: x[1])
                    spread = sorted_levels[-1][1] - sorted_levels[0][1]
                    level_str = ", ".join(f"{k}={v:+.3f}" for k, v in sorted_levels)
                    lines.append(f"  {factor_name}: delta={spread:.3f} ({level_str})")

    lines.append("")
    lines.append("## 4. Assessment")
    lines.append("")

    if concentration == "high":
        lines.append("This direction aligns strongly with a small number of SAE features.")
        lines.append("It likely corresponds to a specific computational mechanism.")
    elif concentration == "medium" and reality in ("strong", "moderate"):
        lines.append("This direction is distributed across many SAE features but clearly non-random.")
        lines.append("It captures a real computational pattern, represented as a population code.")
    elif reality == "weak":
        lines.append("WARNING: This direction is barely distinguishable from random in SAE space.")
        lines.append("It may be an extraction artifact rather than a real computational feature.")
    else:
        lines.append("This direction is diffuse but non-random. Interpret with caution.")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAE Direction Validator")
    parser.add_argument("--direction", required=True, help="Path to unit direction .pt file")
    parser.add_argument("--stimuli", default=None, help="Stimuli YAML for tuning curves (optional)")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--model", default=None, help="HF model name (required if --stimuli given)")
    parser.add_argument("--layer", type=int, default=None, help="Extraction layer (required if --stimuli given)")
    parser.add_argument("--axis-name", default="direction", help="Name for the report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-capture", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== SAE Validation: {args.axis_name} ===\n")

    print("Loading direction...")
    direction = torch.load(args.direction, weights_only=True).float()
    print(f"  dim={direction.shape[0]}, norm={direction.norm():.4f}")

    print(f"Loading SAE: {args.sae_release}/{args.sae_id}...")
    sae = load_sae(args.sae_release, args.sae_id)
    W_dec = sae.W_dec.data
    print(f"  {W_dec.shape[0]} features, d_model={W_dec.shape[1]}")

    assert direction.shape[0] == W_dec.shape[1], \
        f"Dimension mismatch: direction {direction.shape[0]} vs SAE {W_dec.shape[1]}"

    # Phase 1: Decomposition
    print("\nPhase 1: Decomposing direction...")
    top_features, cos_stats, _ = decompose_direction(direction, sae, args.top_k)
    print(f"  max cos={cos_stats['max']}, n>0.1={cos_stats['n_above_0.1']}, "
          f"n>0.2={cos_stats['n_above_0.2']}")

    print("  Greedy subspace capture...")
    capture = greedy_subspace_capture(direction, sae, args.max_capture)
    at_10 = capture[9]['variance_explained'] if len(capture) >= 10 else 0
    at_100 = capture[-1]['variance_explained'] if capture else 0
    print(f"  10 features: {at_10*100:.1f}%, 100 features: {at_100*100:.1f}%")

    print("  Random baseline...")
    baseline = random_baseline(direction.shape[0], sae, args.max_capture, n_trials=10)
    print(f"  Random 100 features: {baseline[-1]*100:.1f}%")

    # Phase 2: Tuning curves (optional)
    tuning_data = None
    top_feature_ids = [f['feature_id'] for f in top_features[:20]]
    if args.stimuli and args.model and args.layer is not None:
        print(f"\nPhase 2: Tuning curves on {args.stimuli}...")
        prompts = load_stimuli(args.stimuli)
        print(f"  {len(prompts)} stimuli loaded")

        device = torch.device(args.device)
        print(f"  Loading model: {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()

        print(f"  Extracting activations at layer {args.layer}...")
        results = extract_activations(model, tokenizer, prompts, args.layer, device)
        del model
        torch.cuda.empty_cache()

        print("  Computing projections and SAE encodings...")
        sae = sae.to('cpu')
        tuning_data = compute_tuning_data(results, direction, sae, top_feature_ids)

    # Generate report
    report = generate_report(args.axis_name, top_features, cos_stats, capture,
                            baseline, tuning_data, top_feature_ids)

    elapsed = time.time() - t0
    report += f"\n\n---\nGenerated in {elapsed:.0f}s on {time.strftime('%Y-%m-%d %H:%M')}\n"

    # Save outputs
    prefix = args.axis_name.lower().replace(' ', '_')
    report_path = out_dir / f"{prefix}_validation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    data_path = out_dir / f"{prefix}_validation_data.json"
    save_data = {
        "axis_name": args.axis_name,
        "direction_file": str(args.direction),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "d_model": int(direction.shape[0]),
        "n_sae_features": int(W_dec.shape[0]),
        "cosine_statistics": cos_stats,
        "top_features": top_features,
        "subspace_capture": capture,
        "random_baseline": baseline,
    }
    if tuning_data:
        save_data["tuning_data"] = tuning_data
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Data:   {data_path}")

    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
