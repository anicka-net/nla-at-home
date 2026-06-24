#!/usr/bin/env python3
r"""
Linear-oracle faithfulness probe — DIAGNOSTIC, CPU-only, no model training.

=============================================================================
WHY THIS EXISTS
=============================================================================
We have changed the AV's TARGETS (twin_clean re-SFT -> null) and its OBJECTIVE
(faithfulness-reward GRPO -> running) without ever confirming that the
information we want is even *recoverable* from the single activation vector we
extract. This probe closes that gap. It asks one question:

    How much source-identity is LINEARLY decodable from the extraction-position
    activation, independent of the AV?

Method: fit a ridge map W: activation -> all-MiniLM text-embedding space on the
TRAIN ids, then run the *exact same* holdout retrieval the AV faith-eval runs
(1-in-50 top1), but from W·a instead of from an AV description. This upper-
bounds what any linear read-out of that activation can achieve.

It forks the roadmap:
  * oracle input-text top1 ~= AV's 0.406  -> info is NOT linearly there; the
    bottleneck is REPRESENTATIONAL (extraction position/layer), not the AV.
    Next lever = re-extract (lever #7), not more reward/target engineering.
  * oracle input-text top1 >> 0.406        -> the info IS there and the AV is
    leaving it on the table -> READOUT problem worth spending on (capacity,
    stronger reward embedder, decoding).

It ALSO runs the INJECTION-MECHANISM test. The AV never sees the raw activation:
normalize_activation() L2-normalizes it to a fixed norm (INJECTION_SCALE=150),
discarding magnitude, before it is placed in the embedding slot. So we re-run the
exact input-text oracle on the *direction-only* (per-sample L2-normalized)
activation -- the literal vector the model receives. (Ridge is scale-invariant,
so unit-norm == norm-150 here.) This isolates the one explicitly lossy step in
the injection path:
  * dir-oracle ~= raw-oracle (~0.83)   -> magnitude is NOT needed; the injection
    mechanism preserves identity -> the gap is training/decoding/capacity.
  * dir-oracle collapses toward 0.406  -> the discarded magnitude carried the
    identity -> the INJECTION MECHANISM is the bottleneck; lever = carry
    magnitude (separate scale token / un-normalized inject) [HUMAN REVIEW].
Caveat: a linear probe on the injected vector lower-bounds what the model (which
can transform it nonlinearly) could extract; it cleanly upper-bounds a *linear*
read-out and isolates the magnitude-discard effect vs the raw oracle.

It also reports two reference ceilings:
  * GT-desc -> input-text top1: how well the *teacher* gpt4o descriptions the AV
    imitates retrieve their own input text. This caps any description-based
    method; if it's ~0.406 the AV has almost nothing to gain in input-text space.
  * per-layer curve: which layers carry the identity (entity confab is known to
    concentrate at mid-depth L16-25) -> informs layer-selection/weighting.

=============================================================================
EXPECTED COMMAND (run on deepthought; CPU + RAM only, safe alongside a GPU job)
=============================================================================
  ~/venv/bin/python scripts/probe_activation_faithfulness.py \
    --acts ~/phi4_ar/phi4_13depths.pt \
    --holdout output/roundtrip_v2corpus/holdout.json \
    --desc-json ~/phi4_ar/descriptions_phi4_tokenpred_gpt4o.json \
    --desc-key-map 4:4,10:10,16:16,19:20,25:26,32:32,38:38 \
    --layers 4,10,16,19,25,32,38 \
    --out output/probe_activation_faithfulness.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent


def load_corpus_texts(corpus_glob):
    texts = {}
    for fp in glob.glob(corpus_glob):
        try:
            for x in json.load(open(fp)):
                if isinstance(x, dict) and x.get("id") and x.get("text"):
                    texts[x["id"]] = x["text"]
        except Exception:
            continue
    return texts


def load_gt_descs(desc_json, layers, keymap):
    recs = json.load(open(desc_json))
    out = {}
    for L in layers:
        dk = keymap.get(L, L)
        out[L] = {x["id"]: x["description"] for x in recs
                  if x["layer"] == dk and x.get("description")}
    return out


def top1(P, K):
    """P,K: (n,d) row-aligned & L2-normalized. Fraction whose nearest key is self."""
    if P.shape[0] == 0:
        return float("nan")
    sims = P @ K.T
    return float((sims.argmax(axis=1) == np.arange(P.shape[0])).mean())


def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def ridge_fit(A, Y, alpha_rel):
    """A:(n,d) centered, Y:(n,k). Ridge with alpha scaled to mean diag of A^T A.
    Returns W:(d,k)."""
    d = A.shape[1]
    G = A.T @ A
    diag_mean = np.trace(G) / d
    AtY = A.T @ Y
    W = np.linalg.solve(G + alpha_rel * diag_mean * np.eye(d, dtype=G.dtype), AtY)
    return W


def fit_compass(A, Y, alphas, val_frac=0.1, seed=0):
    """Fit a DEPLOYABLE oracle compass W: (a - mu) -> MiniLM text space.

    Same math as fit_predict_oracle but returns the artifact (mu, W, alpha)
    instead of holdout predictions, so it can be saved and reused at inference
    to rerank AV samples by cos(MiniLM(sample), l2norm((a-mu) @ W)). Picks alpha
    by a held-out val split (self-retrieval top1), then refits W on all rows.
    Returns (mu(d,), W(d,k), alpha, val_top1)."""
    mu = A.mean(0, keepdims=True)
    Ac = A - mu
    n = Ac.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, fi = perm[:nv], perm[nv:]
    Yv = l2norm(Y[vi])
    best_a, best_s = alphas[0], -1.0
    for a in alphas:
        W = ridge_fit(Ac[fi], Y[fi], a)
        s = top1(l2norm(Ac[vi] @ W), Yv)
        if s > best_s:
            best_s, best_a = s, a
    W = ridge_fit(Ac, Y, best_a)
    return mu[0], W, best_a, best_s


def save_compass(path, layers, acts, id2idx, txt_emb, alphas, ids, ho_set,
                 model_name, all_ids):
    """Fit + save the per-layer oracle compass for inference-time reranking.

    By default fits on TRAIN ids only (excludes the diagnostic holdout) so a
    rerank eval on that holdout stays leak-free. Pass all_ids=True to fit on
    every paired id for a max-data deployment compass (correct when reranking
    only NOVEL user text, as describe_live.py does)."""
    compass = {"layers": [], "mu": {}, "W": {}, "alpha": {}, "val_top1": {},
               "faith_model": model_name, "centered": True,
               "fit_on": "all_ids" if all_ids else "train_only"}
    for L in layers:
        pair_ids = [t for t in ids if t in txt_emb and t in id2idx
                    and (all_ids or t not in ho_set)]
        A = acts[L][[id2idx[t] for t in pair_ids]].double().numpy()
        Y = np.stack([txt_emb[t] for t in pair_ids])
        mu, W, a, vs = fit_compass(A, Y, alphas)
        compass["layers"].append(L)
        compass["mu"][L] = torch.tensor(mu, dtype=torch.float32)
        compass["W"][L] = torch.tensor(W, dtype=torch.float32)
        compass["alpha"][L] = float(a)
        compass["val_top1"][L] = float(vs)
        print(f"  compass L{L}: fit on {len(pair_ids)} ids "
              f"(alpha={a}, val_top1={vs:.3f})", flush=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(compass, path)
    print(f"saved oracle compass ({len(compass['layers'])} layers) -> {path}",
          flush=True)


def fit_predict_oracle(A_tr, Y_tr, A_ho, alphas, val_frac=0.1, seed=0):
    """Pick alpha by a held-out val split (maximize self-retrieval top1 in the
    predicted space against Y_tr keys), refit on all train, predict holdout."""
    n = A_tr.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, fi = perm[:nv], perm[nv:]
    best_a, best_s = alphas[0], -1.0
    Yv_keys = l2norm(Y_tr[vi])
    for a in alphas:
        W = ridge_fit(A_tr[fi], Y_tr[fi], a)
        Pv = l2norm(A_tr[vi] @ W)
        s = top1(Pv, Yv_keys)
        if s > best_s:
            best_s, best_a = s, a
    W = ridge_fit(A_tr, Y_tr, best_a)
    return l2norm(A_ho @ W), best_a, best_s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--desc-json", default=None,
                    help="GT descriptions for the matched-GT/teacher diagnostics. "
                         "Required for the diagnostic; not needed for --save-compass.")
    ap.add_argument("--desc-key-map", default="4:4,10:10,16:16,19:20,25:26,32:32,38:38")
    ap.add_argument("--layers", default="4,10,16,19,25,32,38")
    ap.add_argument("--corpus-glob", default=str(REPO / "corpus/generated/*.json"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--alphas", default="0.001,0.01,0.1,1,10")
    ap.add_argument("--save-compass", metavar="PATH", default=None,
                    help="also fit+save the per-layer oracle compass (W: a->MiniLM "
                         "text space) for inference-time reranking, then exit after "
                         "the diagnostic. CPU-only.")
    ap.add_argument("--compass-all-ids", action="store_true",
                    help="fit the saved compass on ALL paired ids incl. the holdout "
                         "(max-data deployment compass for reranking NOVEL text). "
                         "Default fits train-only = leak-free vs the holdout eval.")
    ap.add_argument("--out", default=str(REPO / "output/probe_activation_faithfulness.json"))
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    if not args.save_compass and not args.desc_json:
        ap.error("--desc-json is required for the diagnostic (omit only with --save-compass)")
    keymap = {int(a): int(b) for a, b in
              (kv.split(":") for kv in args.desc_key_map.split(","))} if args.desc_key_map else {}
    alphas = [float(x) for x in args.alphas.split(",")]

    print("=" * 70)
    print("LINEAR-ORACLE FAITHFULNESS PROBE (diagnostic, CPU)")
    print(f"  layers={layers}  alphas={alphas}")
    print("=" * 70, flush=True)

    D = torch.load(args.acts, weights_only=True, map_location="cpu")
    acts = D["activations"]
    ids = D["ids"]
    id2idx = {t: i for i, t in enumerate(ids)}
    texts = load_corpus_texts(args.corpus_glob)
    gt = load_gt_descs(args.desc_json, layers, keymap) if args.desc_json else {}
    holdout = json.load(open(args.holdout))["holdout"]
    ho_set = set(holdout)
    print(f"  acts ids={len(ids)}  d={acts[layers[0]].shape[1]}  "
          f"texts={len(texts)}  holdout={len(holdout)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model, device="cpu")

    def embed(strs):
        return enc.encode(list(strs), normalize_embeddings=True,
                          convert_to_numpy=True, batch_size=128,
                          show_progress_bar=False).astype(np.float64)

    # cache text + GT-desc embeddings
    print("encoding corpus + GT-desc embeddings (CPU)...", flush=True)
    txt_emb = {i: e for i, e in zip(texts.keys(), embed(list(texts.values())))}

    if args.save_compass:
        print("-" * 70)
        print(f"SAVE-COMPASS mode (fit_on={'all_ids' if args.compass_all_ids else 'train_only'})",
              flush=True)
        save_compass(args.save_compass, layers, acts, id2idx, txt_emb, alphas,
                     ids, ho_set, args.model, args.compass_all_ids)
        return

    results = {"layers": {}, "config": {"layers": layers, "alphas": alphas,
               "model": args.model}}
    per_layer_inputtext, per_layer_matched = [], []
    per_layer_inputtext_dir = []  # injection-faithful (direction-only) oracle
    gtdesc_inputtext_layers = []
    ho_pred_inputtext = []  # for the multi-layer ensemble

    for L in layers:
        # holdout ids present at this layer with text
        ho_ids = [t for t in holdout if t in id2idx and t in texts]
        tr_ids = [t for t in ids if t in texts and t not in ho_set]
        idxL = id2idx
        A_tr_raw = acts[L][[idxL[t] for t in tr_ids]].double().numpy()
        A_ho_raw = acts[L][[idxL[t] for t in ho_ids]].double().numpy()
        mu = A_tr_raw.mean(0, keepdims=True)
        A_tr = A_tr_raw - mu
        A_ho = A_ho_raw - mu

        # ---- input-text oracle: W: activation -> MiniLM(text) ----
        Y_tr_txt = np.stack([txt_emb[t] for t in tr_ids])
        K_txt = l2norm(np.stack([txt_emb[t] for t in ho_ids]))
        P_txt, a_txt, val_txt = fit_predict_oracle(A_tr, Y_tr_txt, A_ho, alphas)
        it_top1 = top1(P_txt, K_txt)
        ho_pred_inputtext.append((ho_ids, P_txt))

        # ---- injection-faithful oracle: SAME but on per-sample L2-normalized
        # (direction-only) activations = the literal vector the model receives
        # after normalize_activation() discards magnitude. Scale-invariant for
        # ridge, so unit-norm == INJECTION_SCALE here. Isolates magnitude-discard.
        A_tr_dir = l2norm(A_tr_raw)
        A_ho_dir = l2norm(A_ho_raw)
        mu_dir = A_tr_dir.mean(0, keepdims=True)
        P_dir, a_dir, _ = fit_predict_oracle(
            A_tr_dir - mu_dir, Y_tr_txt, A_ho_dir - mu_dir, alphas)
        it_dir_top1 = top1(P_dir, K_txt)

        # ---- matched-GT oracle: W: activation -> MiniLM(GT desc) ----
        mt_top1 = float("nan")
        gt_ho = [t for t in ho_ids if t in gt[L]]
        tr_gt = [t for t in tr_ids if t in gt[L]]
        if gt_ho and tr_gt:
            Y_tr_gt = embed([gt[L][t] for t in tr_gt])
            A_tr_gt = acts[L][[idxL[t] for t in tr_gt]].double().numpy() - mu
            A_ho_gt = acts[L][[idxL[t] for t in gt_ho]].double().numpy() - mu
            K_gt = l2norm(embed([gt[L][t] for t in gt_ho]))
            P_gt, a_gt, _ = fit_predict_oracle(A_tr_gt, Y_tr_gt, A_ho_gt, alphas)
            mt_top1 = top1(P_gt, K_gt)

        # ---- reference ceiling: GT-desc text -> input-text retrieval ----
        gt_it = float("nan")
        if gt_ho:
            Qg = l2norm(embed([gt[L][t] for t in gt_ho]))
            Kt = l2norm(np.stack([txt_emb[t] for t in gt_ho]))
            gt_it = top1(Qg, Kt)
            gtdesc_inputtext_layers.append(gt_it)

        results["layers"][L] = {
            "input_text_oracle_top1": it_top1, "alpha_txt": a_txt, "val_txt": val_txt,
            "input_text_oracle_dir_top1": it_dir_top1, "alpha_dir": a_dir,
            "matched_gt_oracle_top1": mt_top1,
            "gtdesc_to_inputtext_top1": gt_it,
            "n_holdout": len(ho_ids)}
        per_layer_inputtext.append(it_top1)
        per_layer_inputtext_dir.append(it_dir_top1)
        if not np.isnan(mt_top1):
            per_layer_matched.append(mt_top1)
        print(f"  L{L:>2}: input-text oracle top1={it_top1:.3f} (a={a_txt}) | "
              f"dir-only(injected)={it_dir_top1:.3f} | "
              f"matched-GT oracle top1={mt_top1:.3f} | "
              f"GT-desc->input-text={gt_it:.3f}", flush=True)

    # ---- multi-layer ensemble: average normalized predictions over layers ----
    # (poor-man's multi-position pooling; tells us if combining layers helps)
    common = set.intersection(*[set(h) for h, _ in ho_pred_inputtext])
    common = [t for t in holdout if t in common]
    ens_top1 = float("nan")
    if common:
        Pens = np.zeros((len(common), ho_pred_inputtext[0][1].shape[1]))
        for ho_ids, P in ho_pred_inputtext:
            pos = {t: k for k, t in enumerate(ho_ids)}
            Pens += l2norm(P)[[pos[t] for t in common]]
        Pens = l2norm(Pens)
        Kc = l2norm(np.stack([txt_emb[t] for t in common]))
        ens_top1 = top1(Pens, Kc)

    summ = {
        "input_text_oracle_mean": float(np.nanmean(per_layer_inputtext)),
        "input_text_oracle_dir_mean": float(np.nanmean(per_layer_inputtext_dir)),
        "input_text_oracle_dir_best_layer": float(np.nanmax(per_layer_inputtext_dir)),
        "matched_gt_oracle_mean": float(np.nanmean(per_layer_matched)) if per_layer_matched else None,
        "gtdesc_to_inputtext_mean": float(np.nanmean(gtdesc_inputtext_layers)) if gtdesc_inputtext_layers else None,
        "input_text_oracle_multilayer_ensemble": ens_top1,
        "input_text_oracle_best_layer": float(np.nanmax(per_layer_inputtext)),
        "AV_baseline_inputtext_top1": 0.406,
        "AV_baseline_matched_top1": 0.463,
        "random_top1": round(1.0 / max(1, len(holdout)), 4),
    }
    results["summary"] = summ
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    print("-" * 70)
    print("SUMMARY (compare to AV faith-eval: input-text 0.406 | matched 0.463)")
    print(f"  input-text oracle   mean over layers = {summ['input_text_oracle_mean']:.3f}"
          f"   best layer = {summ['input_text_oracle_best_layer']:.3f}"
          f"   multilayer-ensemble = {ens_top1:.3f}")
    print(f"  matched-GT oracle   mean over layers = {summ['matched_gt_oracle_mean']}")
    print(f"  GT-desc->input-text mean (teacher ceiling) = {summ['gtdesc_to_inputtext_mean']}")
    print(f"  random baseline = {summ['random_top1']}")
    print("-" * 70)
    print("READOUT:")
    m = summ["input_text_oracle_mean"]
    if m <= 0.45:
        print("  oracle ~ AV (0.406): identity is NOT linearly in this activation ->")
        print("  REPRESENTATIONAL bottleneck. Next = re-extract (lever #7), not reward/target.")
    elif m >= 0.55:
        print("  oracle >> AV (0.406): identity IS recoverable -> READOUT bottleneck.")
        print("  The AV is leaving signal on the table; capacity/reward/decoding worth it.")
    else:
        print("  oracle modestly > AV: partial headroom; inspect the per-layer curve +")
        print("  multilayer ensemble to decide between layer-selection and re-extraction.")
    print("-" * 70)
    print("INJECTION (magnitude-discard) TEST:")
    md = summ["input_text_oracle_dir_mean"]
    drop = m - md
    print(f"  raw-activation oracle mean       = {m:.3f}")
    print(f"  direction-only (injected) oracle = {md:.3f}   (drop {drop:+.3f} from raw)")
    if md <= 0.45:
        print("  -> normalizing to fixed norm COLLAPSES identity toward the AV's 0.406:")
        print("     the INJECTION MECHANISM discards the magnitude that carries identity.")
        print("     Lever = carry magnitude (separate scale token / un-normalized inject)")
        print("     -- this is a mechanism change, HUMAN REVIEW required.")
    elif drop <= 0.10:
        print("  -> direction alone preserves identity: injection is NOT the bottleneck.")
        print("     The gap is training/decoding/capacity, not the mechanism.")
    else:
        print("  -> partial magnitude dependence: injection loses some identity but not")
        print("     all; weigh a magnitude-carrying tweak against readout levers.")
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
