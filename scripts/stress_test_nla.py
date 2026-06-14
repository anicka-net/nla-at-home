#!/usr/bin/env python3
"""
Stress test for NLA: does the AV output carry activation-specific
information, or is it DeepSeek-flavored fabulation?

Six tests, no LLM judge, mostly forced-choice and controls.

Usage:
  python3 scripts/stress_test_nla.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-L20-av-v3 \
    --ar-adapter output/nla-qwen25-7b-L20-ar \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --n-samples 50 \
    --output stress_test_results.json
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma3-1b": "google/gemma-3-1b-it",
}
INJECTION_CHARS = {
    "qwen25-7b": "㈎",
    "gemma3-1b": "⎝",
}
INJECTION_SCALE = 150.0


def load_model_and_adapter(model_key, adapter_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def make_av_prompt(depth_pct, injection_char):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        f"Here is the vector from depth {depth_pct}% of the network:\n\n"
        f"<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )


def generate_av_description(model, tokenizer, activation, injection_char,
                            depth_pct, device):
    from generation_utils import decode_generated

    prompt = make_av_prompt(depth_pct, injection_char)
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(chat_str, add_special_tokens=False)
    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]

    inject_pos = None
    for i, t in enumerate(prompt_tokens):
        if t == injection_token_id:
            inject_pos = i
            break

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(input_ids)
    embeddings[0, inject_pos, :] = activation.to(device).float() * INJECTION_SCALE

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=embeddings.to(model.dtype),
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=150, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    return decode_generated(output, prompt_tokens, tokenizer)


def ar_reconstruct(model, tokenizer, description, injection_char,
                   target_layer, device):
    injection_token_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    ar_text = f"Summary of the following text: <text>{description}</text> <summary>{injection_char}"
    tokens = tokenizer.encode(ar_text, add_special_tokens=False)

    inject_pos = None
    for i, t in enumerate(tokens):
        if t == injection_token_id:
            inject_pos = i
            break
    if inject_pos is None:
        inject_pos = len(tokens) - 1

    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    blocks = inner.layers

    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    layer_outputs = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        layer_outputs[0] = h

    handle = blocks[target_layer].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()

    return layer_outputs[0][0, inject_pos].float()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)).item()


# ============================================================
# Test 1: Round-trip AR control
# ============================================================
def test_roundtrip_ar(av_model, ar_model, tokenizer, acts, ids, desc_map,
                      injection_char, target_layer, device, n=20):
    print("\n[Test 1] Round-trip AR control")
    rng = np.random.RandomState(42)
    sample_idxs = rng.choice(len(ids), min(n, len(ids)), replace=False)

    results = {"real_own": [], "real_shuffled": [], "gt_desc": [],
               "generic": [], "random_act": []}

    for i, idx in enumerate(sample_idxs):
        text_id = ids[idx]
        act = acts[idx].to(device).float()
        gt = desc_map.get(text_id, "")
        if not gt:
            continue

        # Generate AV description from real activation
        nla_desc = generate_av_description(
            av_model, tokenizer, acts[idx], injection_char, 71, device)
        if not nla_desc:
            continue

        # AR reconstruct from NLA description → compare to own activation
        recon = ar_reconstruct(ar_model, tokenizer, nla_desc, injection_char,
                               target_layer, device)
        results["real_own"].append(cosine(recon, act))

        # AR reconstruct from NLA description → compare to shuffled activation
        shuf_idx = rng.choice(len(ids))
        shuf_act = acts[shuf_idx].to(device).float()
        results["real_shuffled"].append(cosine(recon, shuf_act))

        # AR reconstruct from GT description
        gt_recon = ar_reconstruct(ar_model, tokenizer, gt, injection_char,
                                  target_layer, device)
        results["gt_desc"].append(cosine(gt_recon, act))

        # AR reconstruct from generic description
        generic = "Processing general text input at moderate depth."
        gen_recon = ar_reconstruct(ar_model, tokenizer, generic, injection_char,
                                   target_layer, device)
        results["generic"].append(cosine(gen_recon, act))

        # AV from random activation → AR → compare to random's own
        rand_act = torch.randn_like(act) * act.norm() / act.shape[0] ** 0.5
        rand_act = rand_act * (act.norm() / rand_act.norm())
        rand_desc = generate_av_description(
            av_model, tokenizer, rand_act.cpu(), injection_char, 71, device)
        rand_recon = ar_reconstruct(ar_model, tokenizer, rand_desc, injection_char,
                                     target_layer, device)
        results["random_act"].append(cosine(rand_recon, rand_act))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(sample_idxs)}")

    for k, v in results.items():
        print(f"  {k:20s}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    return results


# ============================================================
# Test 2: Forced-choice retrieval (CPU-only, no model needed)
# ============================================================
def test_forced_choice(nla_descriptions, corpus_texts, n_distractors=9):
    print("\n[Test 2] Forced-choice retrieval")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return {}

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {"top1": 0, "top3": 0, "category": 0, "total": 0}
    rng = np.random.RandomState(42)

    all_texts = list(corpus_texts.items())

    for text_id, nla_desc in nla_descriptions.items():
        if text_id not in corpus_texts:
            continue

        true_text = corpus_texts[text_id]
        true_cat = text_id.rsplit("_", 1)[0]

        # Pick distractors: same-length texts from different categories
        candidates = [(text_id, true_text)]
        attempts = 0
        while len(candidates) < n_distractors + 1 and attempts < 100:
            rid, rtext = all_texts[rng.randint(len(all_texts))]
            if rid != text_id and rid not in [c[0] for c in candidates]:
                candidates.append((rid, rtext))
            attempts += 1

        rng.shuffle(candidates)

        # Embed NLA description and all candidate texts
        nla_emb = st_model.encode([nla_desc])
        cand_embs = st_model.encode([c[1] for c in candidates])

        sims = np.dot(cand_embs, nla_emb.T).flatten()
        ranked = np.argsort(-sims)

        true_pos = [i for i, (cid, _) in enumerate(candidates) if cid == text_id][0]
        rank = list(ranked).index(true_pos)

        results["total"] += 1
        if rank == 0:
            results["top1"] += 1
        if rank < 3:
            results["top3"] += 1

        matched_cat = candidates[ranked[0]][0].rsplit("_", 1)[0]
        if matched_cat == true_cat:
            results["category"] += 1

    n = results["total"]
    if n > 0:
        print(f"  Top-1 accuracy: {results['top1']}/{n} ({results['top1']/n:.1%}) [chance: 10%]")
        print(f"  Top-3 accuracy: {results['top3']}/{n} ({results['top3']/n:.1%}) [chance: 30%]")
        print(f"  Category match: {results['category']}/{n} ({results['category']/n:.1%})")
    return results


# ============================================================
# Test 3: kNN baseline (CPU-only)
# ============================================================
def test_knn_baseline(acts, ids, desc_map, nla_descriptions, corpus_texts):
    print("\n[Test 3] kNN baseline")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return {}

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    nla_sims = []
    knn_sims = []
    random_sims = []
    rng = np.random.RandomState(42)

    act_tensor = acts if acts.dim() == 2 else acts[20]

    for text_id, nla_desc in nla_descriptions.items():
        if text_id not in corpus_texts or text_id not in desc_map:
            continue

        idx = ids.index(text_id)
        target_act = act_tensor[idx]

        # Find nearest neighbor by activation cosine
        all_cos = torch.nn.functional.cosine_similarity(
            target_act.unsqueeze(0), act_tensor)
        all_cos[idx] = -1
        nn_idx = all_cos.argmax().item()
        nn_id = ids[nn_idx]
        nn_desc = desc_map.get(nn_id, "")

        # Random same-category description
        cat = text_id.rsplit("_", 1)[0]
        same_cat = [tid for tid in desc_map if tid.startswith(cat) and tid != text_id]
        rand_id = same_cat[rng.randint(len(same_cat))] if same_cat else text_id
        rand_desc = desc_map[rand_id]

        true_text = corpus_texts[text_id]

        # Embed and compare
        embs = st_model.encode([true_text, nla_desc, nn_desc, rand_desc])
        nla_sim = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))
        knn_sim = float(np.dot(embs[0], embs[2]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[2])))
        rand_sim = float(np.dot(embs[0], embs[3]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[3])))

        nla_sims.append(nla_sim)
        knn_sims.append(knn_sim)
        random_sims.append(rand_sim)

    results = {
        "nla_vs_source": {"mean": np.mean(nla_sims), "std": np.std(nla_sims)},
        "knn_vs_source": {"mean": np.mean(knn_sims), "std": np.std(knn_sims)},
        "random_vs_source": {"mean": np.mean(random_sims), "std": np.std(random_sims)},
    }
    for k, v in results.items():
        print(f"  {k:20s}: {v['mean']:.4f} +/- {v['std']:.4f}")
    return results


# ============================================================
# Test 4: Activation ablation controls
# ============================================================
def test_ablation_controls(av_model, tokenizer, acts, ids, corpus_texts,
                           injection_char, device, n=10):
    print("\n[Test 4] Activation ablation controls")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return {}

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = np.random.RandomState(42)
    sample_idxs = rng.choice(len(ids), min(n, len(ids)), replace=False)

    mean_act = acts.mean(dim=0) if acts.dim() == 2 else acts[20].mean(dim=0)
    mean_norm = acts.norm(dim=1).mean().item() if acts.dim() == 2 else acts[20].norm(dim=1).mean().item()

    conditions = {
        "real": [], "zero": [], "mean": [], "random_gaussian": [],
        "permuted": [], "shuffled": [],
    }

    for idx in sample_idxs:
        text_id = ids[idx]
        if text_id not in corpus_texts:
            continue
        true_text = corpus_texts[text_id]
        real_act = acts[idx] if acts.dim() == 2 else acts[20][idx]

        ablations = {
            "real": real_act,
            "zero": torch.zeros_like(real_act),
            "mean": mean_act,
            "random_gaussian": torch.randn_like(real_act) * (mean_norm / real_act.shape[0] ** 0.5),
            "permuted": real_act[torch.randperm(real_act.shape[0])],
            "shuffled": acts[rng.randint(len(ids))] if acts.dim() == 2 else acts[20][rng.randint(len(ids))],
        }
        # Normalize random to match mean norm
        for k in ["random_gaussian"]:
            ablations[k] = ablations[k] * (mean_norm / ablations[k].norm().item())

        descs = {}
        for name, act in ablations.items():
            descs[name] = generate_av_description(
                av_model, tokenizer, act, injection_char, 71, device)

        # Compare each description to the true source text
        all_descs = [descs[k] for k in conditions.keys()]
        embs = st_model.encode([true_text] + all_descs)
        for i, name in enumerate(conditions.keys()):
            sim = float(np.dot(embs[0], embs[i+1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[i+1])))
            conditions[name].append(sim)

    results = {}
    for name, sims in conditions.items():
        results[name] = {"mean": np.mean(sims), "std": np.std(sims)}
        print(f"  {name:20s}: {results[name]['mean']:.4f} +/- {results[name]['std']:.4f}")
    return results


# ============================================================
# Test 5: Perturbation stability
# ============================================================
def test_perturbation(av_model, tokenizer, acts, ids, injection_char,
                      device, n=10):
    print("\n[Test 5] Perturbation stability")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return {}

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = np.random.RandomState(42)
    sample_idxs = rng.choice(len(ids), min(n, len(ids)), replace=False)

    noise_levels = [0.0, 0.01, 0.05, 0.20, 1.0]
    results = {f"noise_{int(l*100)}pct": [] for l in noise_levels}

    for idx in sample_idxs:
        real_act = acts[idx] if acts.dim() == 2 else acts[20][idx]
        act_norm = real_act.norm().item()

        descs = []
        for level in noise_levels:
            if level == 1.0:
                noisy = torch.randn_like(real_act)
                noisy = noisy * (act_norm / noisy.norm().item())
            else:
                noise = torch.randn_like(real_act) * act_norm * level
                noisy = real_act + noise

            desc = generate_av_description(
                av_model, tokenizer, noisy, injection_char, 71, device)
            descs.append(desc)

        embs = st_model.encode(descs)
        base_emb = embs[0]
        for i, level in enumerate(noise_levels):
            sim = float(np.dot(base_emb, embs[i]) / (np.linalg.norm(base_emb) * np.linalg.norm(embs[i])))
            results[f"noise_{int(level*100)}pct"].append(sim)

    for name, sims in results.items():
        print(f"  {name:20s}: {np.mean(sims):.4f} +/- {np.std(sims):.4f}")
    return results


# ============================================================
# Test 6: Interpolation
# ============================================================
def test_interpolation(av_model, tokenizer, acts, ids, corpus_texts,
                       injection_char, device, n_pairs=5):
    print("\n[Test 6] Interpolation")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return {}

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pick contrasting pairs by category
    cat_to_idxs = defaultdict(list)
    for i, tid in enumerate(ids):
        cat = tid.rsplit("_", 1)[0]
        cat_to_idxs[cat].append(i)

    pairs = [
        ("A01_code", "B12_grief_loss"),
        ("A02_math", "A07_medicine_health"),
        ("F31_completely_benign", "F35_clearly_harmful"),
        ("B11_joy_gratitude", "B13_anger_frustration"),
        ("D21_extremely_formal", "D22_casual_colloquial"),
    ][:n_pairs]

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for cat_a, cat_b in pairs:
        if cat_a not in cat_to_idxs or cat_b not in cat_to_idxs:
            continue
        idx_a = cat_to_idxs[cat_a][0]
        idx_b = cat_to_idxs[cat_b][0]
        act_a = acts[idx_a] if acts.dim() == 2 else acts[20][idx_a]
        act_b = acts[idx_b] if acts.dim() == 2 else acts[20][idx_b]

        descs = []
        for t in t_values:
            interp = (1 - t) * act_a.float() + t * act_b.float()
            desc = generate_av_description(
                av_model, tokenizer, interp, injection_char, 71, device)
            descs.append(desc)

        embs = st_model.encode(descs)
        # Check monotonic drift from A to B
        sims_to_a = [float(np.dot(embs[0], embs[i]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[i])))
                     for i in range(len(t_values))]
        sims_to_b = [float(np.dot(embs[-1], embs[i]) / (np.linalg.norm(embs[-1]) * np.linalg.norm(embs[i])))
                     for i in range(len(t_values))]

        pair_result = {
            "pair": f"{cat_a} -> {cat_b}",
            "sims_to_a": sims_to_a,
            "sims_to_b": sims_to_b,
            "monotonic_a": all(sims_to_a[i] >= sims_to_a[i+1] - 0.05 for i in range(len(sims_to_a)-1)),
            "monotonic_b": all(sims_to_b[i] <= sims_to_b[i+1] + 0.05 for i in range(len(sims_to_b)-1)),
        }
        results.append(pair_result)
        a_str = " ".join(f"{s:.2f}" for s in sims_to_a)
        b_str = " ".join(f"{s:.2f}" for s in sims_to_b)
        mono = "OK" if pair_result["monotonic_a"] and pair_result["monotonic_b"] else "FAIL"
        print(f"  {cat_a:30s} -> {cat_b}: sim_A=[{a_str}] [{mono}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="NLA stress test battery")
    parser.add_argument("--model", default="qwen25-7b", choices=list(MODELS.keys()))
    parser.add_argument("--av-adapter", required=True)
    parser.add_argument("--ar-adapter", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--target-layer", type=int, default=20)
    parser.add_argument("--depth-pct", type=int, default=71)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--output", default="stress_test_results.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip tests requiring model inference")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading activations...")
    act_data = torch.load(args.activations, weights_only=True, map_location="cpu")
    acts = act_data["activations"]
    ids = act_data["ids"]
    injection_char = INJECTION_CHARS[args.model]

    # Load corpus texts
    corpus_texts = {}
    for f in sorted(GENERATED_DIR.glob("[A-Z]*.json")):
        for item in json.loads(f.read_text()):
            corpus_texts[item["id"]] = item["text"]

    # Load GT descriptions
    desc_path = GENERATED_DIR / "descriptions_L71pct_merged.json"
    if not desc_path.exists():
        desc_path = GENERATED_DIR / "descriptions_L71pct.json"
    descs = json.loads(desc_path.read_text())
    desc_map = {d["id"]: d["description"] for d in descs}

    all_results = {}

    if not args.skip_gpu:
        print("Loading AV adapter...")
        av_model, tokenizer = load_model_and_adapter(args.model, args.av_adapter, device)
        device = next(av_model.parameters()).device

        # Generate NLA descriptions for samples (reused across tests)
        print("\nPre-generating NLA descriptions...")
        rng = np.random.RandomState(42)
        sample_idxs = rng.choice(len(ids), min(args.n_samples, len(ids)), replace=False)
        nla_descriptions = {}
        for i, idx in enumerate(sample_idxs):
            text_id = ids[idx]
            act = acts[idx] if acts.dim() == 2 else acts[20][idx]
            nla_descriptions[text_id] = generate_av_description(
                av_model, tokenizer, act, injection_char, args.depth_pct, device)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample_idxs)}")

        # Test 4: Ablation controls
        all_results["test4_ablation"] = test_ablation_controls(
            av_model, tokenizer, acts, ids, corpus_texts,
            injection_char, device, n=min(10, args.n_samples))

        # Test 5: Perturbation stability
        all_results["test5_perturbation"] = test_perturbation(
            av_model, tokenizer, acts, ids, injection_char,
            device, n=min(10, args.n_samples))

        # Test 6: Interpolation
        all_results["test6_interpolation"] = test_interpolation(
            av_model, tokenizer, acts, ids, corpus_texts,
            injection_char, device)

        # Load AR alongside AV for round-trip test (both fit on GB10 128GB)
        print("\nLoading AR adapter...")
        ar_model, tokenizer = load_model_and_adapter(args.model, args.ar_adapter, device)

        all_results["test1_roundtrip"] = test_roundtrip_ar(
            av_model, ar_model, tokenizer, acts, ids, desc_map,
            injection_char, args.target_layer, device, n=args.n_samples)

        del av_model, ar_model
        torch.cuda.empty_cache()
    else:
        nla_descriptions = {}
        print("  Skipping GPU tests. Using GT descriptions for CPU-only tests.")
        rng = np.random.RandomState(42)
        sample_idxs = rng.choice(len(ids), min(args.n_samples, len(ids)), replace=False)
        for idx in sample_idxs:
            text_id = ids[idx]
            if text_id in desc_map:
                nla_descriptions[text_id] = desc_map[text_id]

    # Test 2: Forced-choice (CPU-only)
    all_results["test2_forced_choice"] = test_forced_choice(
        nla_descriptions, corpus_texts)

    # Test 3: kNN baseline (CPU-only)
    all_results["test3_knn"] = test_knn_baseline(
        acts, ids, desc_map, nla_descriptions, corpus_texts)

    # Summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)

    if "test1_roundtrip" in all_results:
        r = all_results["test1_roundtrip"]
        print(f"  [1] AR roundtrip: own={np.mean(r['real_own']):.3f} vs shuffled={np.mean(r['real_shuffled']):.3f} "
              f"(delta={np.mean(r['real_own'])-np.mean(r['real_shuffled']):+.3f})")

    if "test2_forced_choice" in all_results:
        r = all_results["test2_forced_choice"]
        n = r.get("total", 0)
        if n:
            print(f"  [2] Forced-choice: top1={r['top1']/n:.0%} top3={r['top3']/n:.0%} "
                  f"cat={r['category']/n:.0%} (chance=10%/30%)")

    if "test3_knn" in all_results:
        r = all_results["test3_knn"]
        if "nla_vs_source" in r:
            print(f"  [3] kNN baseline: NLA={r['nla_vs_source']['mean']:.3f} "
                  f"kNN={r['knn_vs_source']['mean']:.3f} "
                  f"random={r['random_vs_source']['mean']:.3f}")

    if "test4_ablation" in all_results:
        r = all_results["test4_ablation"]
        print(f"  [4] Ablation: real={r['real']['mean']:.3f} "
              f"zero={r['zero']['mean']:.3f} "
              f"random={r['random_gaussian']['mean']:.3f} "
              f"permuted={r['permuted']['mean']:.3f}")

    if "test5_perturbation" in all_results:
        r = all_results["test5_perturbation"]
        vals = " ".join(f"{np.mean(r[k]):.2f}" for k in sorted(r.keys()))
        print(f"  [5] Perturbation: [{vals}]")

    if "test6_interpolation" in all_results:
        r = all_results["test6_interpolation"]
        n_mono = sum(1 for p in r if p["monotonic_a"] and p["monotonic_b"])
        print(f"  [6] Interpolation: {n_mono}/{len(r)} pairs monotonic")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
