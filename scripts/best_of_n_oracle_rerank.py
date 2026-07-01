#!/usr/bin/env python3
r"""
best_of_n_oracle_rerank.py — DISCRIMINATOR within the READOUT bucket. GPU.

=============================================================================
WHY THIS EXISTS
=============================================================================
The diagnostics pinned the AV's confabulation on the READOUT (not the
representation, not the injection mechanism): identity is linearly decodable
from the activation at ~0.83, yet the greedy AV scores only ~0.406 input-text
top1. This script splits the readout bucket into decoding vs capacity/training
with ONE cheap experiment.

For each holdout (layer, id) we draw N sampled AV descriptions plus the 1 greedy
description, then compute the *exact* per-layer 1-in-n input-text top1 the AV
faith-eval uses (so the numbers are directly comparable to 0.406), under several
"which sample do we keep" strategies:

  greedy        do_sample=False                         (sanity, ~0.406)
  sample0       first sample at temperature             (sampling vs greedy)
  oracle_rerank keep argmax_j cos(MiniLM(sample_j), t*),
                t* = oracle W_L·a (the linear compass)  (DEPLOYABLE, no retrain)
  truth_rerank  keep argmax_j cos(MiniLM(sample_j),
                MiniLM(input_text))                      (CHEATING upper bound:
                                                          do faithful descs even
                                                          EXIST in the N samples?)

=============================================================================
HOW TO READ THE RESULT
=============================================================================
  truth_rerank >> 0.406  -> faithful descriptions DO exist in the sample set ->
     the bottleneck is DECODING/confabulation; reranking is a viable fix.
     Then oracle_rerank's lift over greedy = what the deployable method buys.
  truth_rerank ~ 0.406   -> the AV never samples a faithful description ->
     bottleneck is CAPACITY/TRAINING; reranking can't help -> raise AV LoRA
     rank / re-SFT (no point spending on decoding).

The oracle compass W_L is the *same* ridge map as probe_activation_faithfulness
(activation -> MiniLM input-text space), fit on TRAIN ids, applied to holdout.

=============================================================================
EXPECTED COMMAND (deepthought; GPU for the 14B AV, CPU for MiniLM + ridge)
=============================================================================
  ~/venv/bin/python scripts/best_of_n_oracle_rerank.py \
    --av-adapter output/nla-phi4-universal-av-v2 \
    --acts ~/phi4_ar/phi4_13depths.pt \
    --holdout output/roundtrip_v2corpus/holdout.json \
    --layers 4,16,25 --n-samples 8 --temperature 0.9 \
    --out output/best_of_n_oracle_rerank.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from probe_activation_faithfulness import (  # noqa: E402
    l2norm, fit_predict_oracle, load_corpus_texts)
from train_universal_av import (  # noqa: E402
    INJECTION_SCALE, make_prompt, find_inject_pos, normalize_activation)

BASE = "microsoft/phi-4"
INJECTION_CHAR = "★"  # matches train_joint_grpo_phi4.INJECTION_CHAR
N_LAYERS = 40
DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]


def nearest_pct(layer):
    d = layer * 100 / N_LAYERS
    return min(DEPTH_PCTS, key=lambda p: abs(p - d))


def top1_np(query, keys):
    """query/keys: (n,d) row-aligned & L2-normalized; correct key of row i is i."""
    if query.shape[0] == 0:
        return float("nan")
    sims = query @ keys.T
    return float((sims.argmax(axis=1) == np.arange(query.shape[0])).mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av-adapter", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--corpus-glob", default=str(REPO / "corpus/generated/*.json"))
    ap.add_argument("--layers", default="4,16,25")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-holdout", type=int, default=0)
    ap.add_argument("--faith-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--alphas", default="0.001,0.01,0.1,1,10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "output/best_of_n_oracle_rerank.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    device = torch.device(args.device)

    print("=" * 70)
    print("BEST-OF-N ORACLE-RERANK DISCRIMINATOR (decoding vs capacity)")
    print(f"  layers={layers}  N={args.n_samples}  temp={args.temperature} "
          f"top_p={args.top_p}")
    print("=" * 70, flush=True)

    # ---- data (CPU) ----
    D = torch.load(args.acts, weights_only=True, map_location="cpu")
    acts, ids = D["activations"], D["ids"]
    id2idx = {t: i for i, t in enumerate(ids)}
    holdout = json.load(open(args.holdout))["holdout"]
    if args.max_holdout:
        holdout = holdout[: args.max_holdout]
    ho_set = set(holdout)
    texts = load_corpus_texts(args.corpus_glob)
    print(f"  acts ids={len(ids)}  d={acts[layers[0]].shape[1]}  "
          f"texts={len(texts)}  holdout={len(holdout)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.faith_model, device="cpu")

    def embed(strs):
        if not strs:
            return np.zeros((0, enc.get_sentence_embedding_dimension()))
        return enc.encode(list(strs), normalize_embeddings=True,
                          convert_to_numpy=True, batch_size=128,
                          show_progress_bar=False).astype(np.float64)

    txt_emb = {i: e for i, e in zip(texts.keys(), embed(list(texts.values())))}

    # ---- oracle compass t* per (layer, holdout id) (CPU ridge) ----
    print("fitting oracle compass W_L (activation -> MiniLM text) per layer...",
          flush=True)
    tstar = {}      # (L, id) -> unit vector in MiniLM text space
    ho_at_layer = {}
    for L in layers:
        ho_ids = [t for t in holdout if t in id2idx and t in texts]
        tr_ids = [t for t in ids if t in texts and t not in ho_set]
        A_tr = acts[L][[id2idx[t] for t in tr_ids]].double().numpy()
        A_ho = acts[L][[id2idx[t] for t in ho_ids]].double().numpy()
        mu = A_tr.mean(0, keepdims=True)
        Y_tr = np.stack([txt_emb[t] for t in tr_ids])
        P_txt, a_txt, _ = fit_predict_oracle(A_tr - mu, Y_tr, A_ho - mu, alphas)
        for r, t in enumerate(ho_ids):
            tstar[(L, t)] = P_txt[r]
        ho_at_layer[L] = ho_ids
        print(f"  L{L}: oracle fit (alpha={a_txt}, n_ho={len(ho_ids)})", flush=True)

    # ---- load AV (GPU) ----
    print("loading AV (Phi-4 + av-v2 LoRA)...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    av_tok = AutoTokenizer.from_pretrained(args.av_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False).to(device)
    av = PeftModel.from_pretrained(base, args.av_adapter)
    av.eval()
    inj_id = av_tok.encode(INJECTION_CHAR, add_special_tokens=False)[0]
    emb_layer = av.get_input_embeddings()

    prompt_cache = {}
    for L in layers:
        pct = nearest_pct(L)
        if pct in prompt_cache:
            continue
        chat = av_tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(pct, INJECTION_CHAR)}],
            tokenize=False, add_generation_prompt=True)
        toks = av_tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (toks, find_inject_pos(toks, inj_id))

    def clean(s):
        return s.split("</explanation>")[0].strip()

    @torch.no_grad()
    def gen(ptoks, ipos, a, n, sample):
        input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
        emb = emb_layer(input_ids)
        emb[0, ipos, :] = normalize_activation(
            a.to(device), INJECTION_SCALE).to(emb.dtype)
        embN = emb.expand(n, -1, -1).contiguous()
        kw = dict(inputs_embeds=embN.to(av.dtype),
                  attention_mask=torch.ones(n, embN.shape[1], dtype=torch.long,
                                            device=device),
                  max_new_tokens=args.max_new_tokens,
                  pad_token_id=av_tok.eos_token_id,
                  return_dict_in_generate=True)
        if sample:
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            kw.update(do_sample=False)
        out = av.generate(**kw)
        return [clean(av_tok.decode(s, skip_special_tokens=True))
                for s in out.sequences]

    # ---- generate greedy + N samples per (layer, id) ----
    greedy_desc, sample_descs = {}, {}
    for L in layers:
        ptoks, ipos = prompt_cache[nearest_pct(L)]
        for t in ho_at_layer[L]:
            a = acts[L][id2idx[t]].float()
            greedy_desc[(L, t)] = gen(ptoks, ipos, a, 1, sample=False)[0]
            sample_descs[(L, t)] = gen(ptoks, ipos, a, args.n_samples, sample=True)
        print(f"  L{L}: generated greedy + {args.n_samples} samples for "
              f"{len(ho_at_layer[L])} ids", flush=True)

    # ---- score: per-layer 1-in-n input-text top1 under each strategy ----
    strat_layer = {k: [] for k in ("greedy", "sample0", "oracle_rerank",
                                   "truth_rerank")}
    per_layer = {}
    for L in layers:
        ho_ids = ho_at_layer[L]
        keys = l2norm(np.stack([txt_emb[t] for t in ho_ids]))
        # cache sample embeddings per id
        samp_emb = {t: l2norm(embed(sample_descs[(L, t)])) for t in ho_ids}
        q_greedy = l2norm(embed([greedy_desc[(L, t)] for t in ho_ids]))
        q_s0 = l2norm(np.stack([samp_emb[t][0] for t in ho_ids]))
        q_oracle, q_truth = [], []
        for t in ho_ids:
            S = samp_emb[t]                          # (N, d) unit rows
            j_o = int((S @ tstar[(L, t)]).argmax())  # closest to oracle compass
            j_t = int((S @ txt_emb[t]).argmax())     # closest to its own text
            q_oracle.append(S[j_o])
            q_truth.append(S[j_t])
        q_oracle = l2norm(np.stack(q_oracle))
        q_truth = l2norm(np.stack(q_truth))
        row = {
            "greedy": top1_np(q_greedy, keys),
            "sample0": top1_np(q_s0, keys),
            "oracle_rerank": top1_np(q_oracle, keys),
            "truth_rerank": top1_np(q_truth, keys),
            "n": len(ho_ids),
        }
        per_layer[L] = row
        for k in strat_layer:
            strat_layer[k].append(row[k])
        print(f"  L{L:>2}: greedy={row['greedy']:.3f} sample0={row['sample0']:.3f} "
              f"oracle_rerank={row['oracle_rerank']:.3f} "
              f"truth_rerank={row['truth_rerank']:.3f}", flush=True)

    summ = {k: float(np.nanmean(v)) for k, v in strat_layer.items()}
    summ["AV_baseline_inputtext_top1"] = 0.406
    summ["n_samples"] = args.n_samples
    out = {"summary": summ, "per_layer": per_layer,
           "config": {"layers": layers, "n_samples": args.n_samples,
                      "temperature": args.temperature, "top_p": args.top_p}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("-" * 70)
    print("SUMMARY (mean over layers; compare to AV greedy faith-eval 0.406)")
    for k in ("greedy", "sample0", "oracle_rerank", "truth_rerank"):
        print(f"  {k:<14} = {summ[k]:.3f}")
    print("-" * 70)
    print("VERDICT:")
    tr, gr, orc = summ["truth_rerank"], summ["greedy"], summ["oracle_rerank"]
    if tr >= gr + 0.12:
        print(f"  truth_rerank ({tr:.3f}) >> greedy ({gr:.3f}): faithful descriptions")
        print("  EXIST in the sample set -> bottleneck is DECODING/confabulation.")
        if orc >= gr + 0.06:
            print(f"  oracle_rerank ({orc:.3f}) already lifts greedy by "
                  f"{orc - gr:+.3f} -> the deployable compass works; push it "
                  "(more samples / better compass / train on it).")
        else:
            print(f"  but oracle_rerank ({orc:.3f}) ~ greedy: the linear compass is")
            print("  too weak to PICK the good sample; need a stronger reranker.")
    else:
        print(f"  truth_rerank ({tr:.3f}) ~ greedy ({gr:.3f}): the AV NEVER samples a")
        print("  faithful description -> bottleneck is CAPACITY/TRAINING, not")
        print("  decoding. Reranking can't help; raise AV LoRA rank / re-SFT.")
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
