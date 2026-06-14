#!/usr/bin/env python3
"""
SAE Tuning Curves: measure how top SAE features respond along the valence axis.

For each stimulus, extracts the residual stream, computes valence projection,
and records SAE feature activations. Then plots feature activation vs valence
projection to see if features "tile" the axis sequentially.

Usage:
  python3 scripts/experiments/sae_tuning_curves.py \
    --direction ~/tone-experiment/results/vedana-vs-rc/llama-8b_vedana_L20_unit.pt \
    --decomposition data/sae-decomposition/llama-8b/vedana_L20_decomposition.json \
    --sae-release llama_scope_lxr_32x \
    --sae-id l20r_32x \
    --stimuli ~/playground/ungag/prompts/vedana_prompts_n50.yaml \
    --model meta-llama/Llama-3.1-8B \
    --layer 20 \
    --output data/sae-decomposition/llama-8b/vedana_L20_tuning_curves.json
"""
import torch
import yaml
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_stimuli(stimuli_path):
    data = yaml.safe_load(open(stimuli_path))
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
        results.append({
            'id': prompt['id'],
            'text': prompt['text'],
            'category': prompt['category'],
            'activation': activation,
        })

        if (i + 1) % 20 == 0:
            print(f'  Extracted {i+1}/{len(prompts)}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', required=True)
    parser.add_argument('--decomposition', required=True)
    parser.add_argument('--sae-release', required=True)
    parser.add_argument('--sae-id', required=True)
    parser.add_argument('--stimuli', required=True)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading direction...')
    direction = torch.load(args.direction, weights_only=True).float()

    print('Loading decomposition results...')
    decomp = json.load(open(args.decomposition))
    top_feature_ids = [f['feature_id'] for f in decomp['top_features'][:20]]

    print('Loading stimuli...')
    prompts = load_stimuli(args.stimuli)
    print(f'  {len(prompts)} stimuli loaded')

    print(f'Loading SAE: {args.sae_release}/{args.sae_id}...')
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=args.sae_release, sae_id=args.sae_id)
    sae = sae.to('cpu')

    print(f'Loading model: {args.model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    print(f'Extracting activations at layer {args.layer}...')
    results = extract_activations(model, tokenizer, prompts, args.layer, device)

    del model
    torch.cuda.empty_cache()

    print('Computing projections and SAE encodings...')
    output_data = []
    for r in results:
        act = r['activation']
        valence_proj = float(act @ direction)

        sae_features = sae.encode(act.unsqueeze(0))
        sae_features = sae_features.squeeze(0)

        top_activations = {}
        for fid in top_feature_ids:
            top_activations[str(fid)] = round(float(sae_features[fid]), 4)

        recon = sae.decode(sae.encode(act.unsqueeze(0))).squeeze(0)
        recon_error = float((act - recon).norm())
        recon_cos = float(torch.nn.functional.cosine_similarity(
            act.unsqueeze(0), recon.unsqueeze(0)))

        output_data.append({
            'id': r['id'],
            'category': r['category'],
            'valence_projection': round(valence_proj, 4),
            'top_feature_activations': top_activations,
            'sae_reconstruction_error': round(recon_error, 4),
            'sae_reconstruction_cosine': round(recon_cos, 4),
        })

    output_data.sort(key=lambda x: x['valence_projection'])

    categories = {}
    for item in output_data:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item['valence_projection'])

    print('\n=== Valence projection by category ===')
    for cat, vals in sorted(categories.items()):
        mean_v = sum(vals) / len(vals)
        print(f'  {cat}: mean={mean_v:.2f}, n={len(vals)}, range=[{min(vals):.2f}, {max(vals):.2f}]')

    print(f'\n=== Top feature responses (sorted by valence) ===')
    top_3_feats = top_feature_ids[:3]
    print(f'  {"Category":<12} {"Valence":>8}  ', end='')
    for fid in top_3_feats:
        print(f'  F#{fid:>6}', end='')
    print()

    for item in output_data[:5] + [{'sep': True}] + output_data[-5:]:
        if 'sep' in item:
            print('  ...')
            continue
        print(f'  {item["category"]:<12} {item["valence_projection"]:>8.2f}  ', end='')
        for fid in top_3_feats:
            val = item['top_feature_activations'].get(str(fid), 0)
            print(f'  {val:>8.3f}', end='')
        print()

    full_output = {
        'direction_file': args.direction,
        'decomposition_file': args.decomposition,
        'sae_release': args.sae_release,
        'model': args.model,
        'layer': args.layer,
        'n_stimuli': len(output_data),
        'category_stats': {cat: {'mean': round(sum(v)/len(v), 4), 'n': len(v)}
                          for cat, v in categories.items()},
        'top_feature_ids': top_feature_ids,
        'stimuli_results': output_data,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
