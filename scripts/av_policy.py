#!/usr/bin/env python3
r"""
av_policy.py — pragmatic test-time decoding policy for the AV.  GPU for --eval.

GOAL (operator, 2026-06-24): not perfect, "good enough" — (1) reduce *disturbing*
hallucinations (confident, specific, WRONG entity) and (2) avoid *useless
templated* outputs (generic boilerplate). Training to move the greedy metric has
failed 3x (twin_clean re-SFT, faith-GRPO, RAFT all null/inconclusive), but
inference reranking against the activation-derived oracle compass is a robust,
deployable win (+0.16 at N=16). This module turns that into a usable policy.

THE POLICY (compass-only v1, no retraining):
  per (activation, layer):
    1. sample N descriptions from the AV
    2. score each by compass faithfulness  c_i = cos(MiniLM(desc_i), t*),
       t* = l2norm((a - mu_L) @ W_L)                       (the +0.16 reranker)
    3. (optional) subtract a genericness penalty cos(desc_i, generic_centroid)
       so empty templates lose to specific candidates
    4. confidence = faithfulness of the chosen sample
    5. GATE:  conf >= tau  -> emit the specific faithful pick
              conf <  tau  -> emit an honest HEDGE instead of a confident
                              confabulation  (the "not disturbing" lever)

WHY THIS HITS BOTH FAILURE MODES:
  * disturbing hallucination = confident + specific + wrong. The gate converts the
    low-faithfulness tail (where confabulation lives) into honest hedges, so we
    stop emitting confident-wrong.
  * useless template = generic. Templates have generic embeddings -> they score
    LOW against a specific t*, so the reranker already down-ranks them; the
    genericness penalty makes that explicit. (When the activation truly is
    diffuse, hedging is the correct answer anyway.)

The right dashboard is NOT top1 (hedging trades faithfulness-on-hard-items for
honesty, by design). It is the per-item decomposition swept over tau:
    {confident-right, confident-WRONG (=disturbing), hedged}
plus the genericness of emitted outputs (template proxy). --eval prints exactly
that, with the greedy baseline, so you can pick an operating point.

EXPECTED COMMAND (deepthought; GPU; ONE 14B job at a time):
  ~/venv/bin/python scripts/av_policy.py --eval \
    --av-adapter output/nla-phi4-universal-av-v2 \
    --compass output/av_oracle_compass.pt \
    --acts ~/phi4_ar/phi4_13depths.pt \
    --holdout output/roundtrip_v2corpus/holdout.json \
    --layers 16,25 --n-samples 12 \
    --out output/av_policy_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Pure, CPU-testable policy logic (no torch / no model).
# ---------------------------------------------------------------------------
HEDGE_PREFIX = "[uncertain — weak/diffuse signal; tentative] "


def l2norm_rows(M):
    M = np.asarray(M, dtype=np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def compass_target(a, mu, W):
    """Predicted unit text-embedding for an activation: l2norm((a - mu) @ W)."""
    t = (np.asarray(a, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) \
        @ np.asarray(W, dtype=np.float64)
    n = np.linalg.norm(t)
    return t / n if n else t


def select_policy(sample_embs, tstar, tau, generic_centroid=None, gen_penalty=0.0):
    """Choose among N sampled-description embeddings (L2-normalized rows).

    Returns dict: idx (chosen), confidence (= chosen faithfulness), decision
    ('specific'|'hedge'), faith (N,), generic (N,|None), agreement (mean pairwise
    cos among samples = how consistent the model is on this activation)."""
    S = np.asarray(sample_embs, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] == 0:
        raise ValueError("sample_embs must be (N,d) with N>=1")
    faith = S @ np.asarray(tstar, dtype=np.float64)
    score = faith.copy()
    generic = None
    if generic_centroid is not None and gen_penalty:
        generic = S @ np.asarray(generic_centroid, dtype=np.float64)
        score = faith - gen_penalty * generic
    j = int(np.argmax(score))
    conf = float(faith[j])
    # inter-sample agreement (off-diagonal mean cosine)
    if S.shape[0] > 1:
        G = S @ S.T
        n = S.shape[0]
        agreement = float((G.sum() - np.trace(G)) / (n * (n - 1)))
    else:
        agreement = 1.0
    return {"idx": j, "confidence": conf,
            "decision": "specific" if conf >= tau else "hedge",
            "faith": faith, "generic": generic, "agreement": agreement}


def apply_policy_text(descs, sel):
    """Render the emitted string for a select_policy() result."""
    best = descs[sel["idx"]]
    return best if sel["decision"] == "specific" else HEDGE_PREFIX + best


def load_corpus_meta(corpus_glob):
    """id -> (category, group) from the corpus JSONs (for safety-gating dumps)."""
    import glob
    meta = {}
    for fp in glob.glob(corpus_glob):
        try:
            for x in json.load(open(fp)):
                if isinstance(x, dict) and x.get("id"):
                    meta[x["id"]] = (x.get("category"), x.get("group"))
        except Exception:
            continue
    return meta


# ---------------------------------------------------------------------------
# GPU batch evaluator: generate N samples on the holdout, sweep tau, report the
# {confident-right, confident-wrong, hedged} decomposition + genericness.
# ---------------------------------------------------------------------------
def run_eval(args):
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from train_universal_av import (
        INJECTION_SCALE, INJECTION_CHARS, make_prompt, find_inject_pos,
        normalize_activation, nearest_depth_pct)
    from probe_activation_faithfulness import load_corpus_texts
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    penalties = [float(x) for x in args.gen_penalties.split(",")]

    compass = torch.load(args.compass, weights_only=False, map_location="cpu")
    cov = set(compass["layers"])
    layers = [L for L in layers if L in cov]
    if not layers:
        raise SystemExit(f"none of --layers are in the compass {sorted(cov)}")

    D = torch.load(args.acts, weights_only=True, map_location="cpu")
    acts, ids = D["activations"], D["ids"]
    id2idx = {t: i for i, t in enumerate(ids)}
    holdout = json.load(open(args.holdout))["holdout"]
    texts = load_corpus_texts(args.corpus_glob)
    enc = SentenceTransformer(args.faith_model, device="cpu")

    def embed(strs):
        if not strs:
            return np.zeros((0, enc.get_sentence_embedding_dimension()))
        return enc.encode(list(strs), normalize_embeddings=True,
                          convert_to_numpy=True, batch_size=128,
                          show_progress_bar=False).astype(np.float64)

    print("loading AV (Phi-4 + LoRA)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.av_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-4", torch_dtype=torch.bfloat16).to(device)
    av = PeftModel.from_pretrained(base, args.av_adapter).eval()
    inj_char = INJECTION_CHARS["phi4"]
    inj_id = tok.encode(inj_char, add_special_tokens=False)[0]
    emb_layer = av.get_input_embeddings()
    n_layers = D.get("n_layers", 40)

    pcache = {}
    for L in layers:
        pct = nearest_depth_pct(L, n_layers)
        if pct in pcache:
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(pct, inj_char)}],
            tokenize=False, add_generation_prompt=True)
        toks = tok.encode(chat, add_special_tokens=False)
        pcache[pct] = (toks, find_inject_pos(toks, inj_id))

    def clean(s):
        return s.split("</explanation>")[0].strip()

    @torch.no_grad()
    def gen(L, t, n, sample):
        ptoks, ipos = pcache[nearest_depth_pct(L, n_layers)]
        a = acts[L][id2idx[t]].float()
        input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
        emb = emb_layer(input_ids)
        emb[0, ipos, :] = normalize_activation(a.to(device), INJECTION_SCALE).to(emb.dtype)
        embN = emb.expand(n, -1, -1).contiguous()
        kw = dict(inputs_embeds=embN.to(av.dtype),
                  attention_mask=torch.ones(n, embN.shape[1], dtype=torch.long, device=device),
                  max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id,
                  return_dict_in_generate=True)
        if sample:
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            kw.update(do_sample=False)
        out = av.generate(**kw)
        return [clean(tok.decode(s, skip_special_tokens=True)) for s in out.sequences]

    # ---- generate greedy + N samples per (layer, holdout id) ----
    per_layer_records = {}
    pooled_sample_embs = []
    for L in layers:
        ho_ids = [t for t in holdout if t in id2idx and t in texts]
        if args.max_items:
            ho_ids = ho_ids[:args.max_items]
        mu = compass["mu"][L].numpy()
        W = compass["W"][L].numpy()
        recs = []
        for t in ho_ids:
            greedy = gen(L, t, 1, sample=False)[0]
            samples = gen(L, t, args.n_samples, sample=True)
            s_emb = l2norm_rows(embed(samples))
            g_emb = l2norm_rows(embed([greedy]))[0]
            a = acts[L][id2idx[t]].float().numpy()
            tstar = compass_target(a, mu, W)
            recs.append({"id": t, "greedy": greedy, "g_emb": g_emb,
                         "samples": samples, "s_emb": s_emb, "tstar": tstar})
            pooled_sample_embs.append(s_emb)
        per_layer_records[L] = (ho_ids, recs)
        print(f"  L{L}: generated greedy + {args.n_samples} samples for {len(ho_ids)} ids",
              flush=True)

    generic_centroid = l2norm_rows(
        np.concatenate(pooled_sample_embs).mean(0, keepdims=True))[0]
    if args.save_centroid:
        torch.save({"centroid": torch.tensor(generic_centroid),
                    "faith_model": args.faith_model, "layers": layers,
                    "fit_on": "holdout sample pool"}, args.save_centroid)
        print(f"saved generic_centroid -> {args.save_centroid}", flush=True)

    cat_meta = load_corpus_meta(args.corpus_glob) if args.dump_texts else {}
    dump = {"config": {"dump_tau": args.dump_tau, "dump_penalty": args.dump_penalty,
                       "n_samples": args.n_samples}, "items": []} \
        if args.dump_texts else None

    # ---- score: per-item decomposition over tau ----
    report = {"layers": layers, "n_samples": args.n_samples, "taus": taus,
              "config": {k: v for k, v in vars(args).items()
                         if k not in ("device",)},
              "per_layer": {}}
    for L in layers:
        ho_ids, recs = per_layer_records[L]
        keys = l2norm_rows(np.stack([embed([texts[t]])[0] for t in ho_ids]))
        self_idx = {t: i for i, t in enumerate(ho_ids)}

        def is_self(emb, t):
            return int((emb @ keys.T).argmax()) == self_idx[t]

        # greedy baseline
        g_right = np.mean([is_self(r["g_emb"], r["id"]) for r in recs])
        g_generic = np.mean([float(r["g_emb"] @ generic_centroid) for r in recs])

        def sweep(gp):
            rows = []
            for tau in taus:
                cr = cw = hed = 0
                spec_generic = []
                for r in recs:
                    sel = select_policy(r["s_emb"], r["tstar"], tau,
                                        generic_centroid, gp)
                    chosen_emb = r["s_emb"][sel["idx"]]
                    if sel["decision"] == "hedge":
                        hed += 1
                    else:
                        spec_generic.append(float(chosen_emb @ generic_centroid))
                        if is_self(chosen_emb, r["id"]):
                            cr += 1
                        else:
                            cw += 1
                n = len(recs)
                rows.append({
                    "tau": tau,
                    "confident_right": round(cr / n, 3),
                    "confident_wrong": round(cw / n, 3),   # the "disturbing" rate
                    "hedged": round(hed / n, 3),
                    "specific_genericness": round(float(np.mean(spec_generic)), 3)
                    if spec_generic else None,
                })
            return rows

        penalty_sweeps = {gp: sweep(gp) for gp in penalties}
        report["per_layer"][L] = {
            "n": len(recs),
            "greedy_confident_right": round(float(g_right), 3),
            "greedy_confident_wrong": round(float(1 - g_right), 3),
            "greedy_genericness": round(float(g_generic), 3),
            "penalty_sweeps": {str(gp): rows for gp, rows in penalty_sweeps.items()},
            "tau_sweep": penalty_sweeps[penalties[0]],   # back-compat (first penalty)
        }
        print(f"  L{L}: greedy right={g_right:.3f} wrong={1-g_right:.3f} "
              f"generic={g_generic:.3f}", flush=True)
        for gp in penalties:
            print(f"   [gen_penalty={gp}]", flush=True)
            for row in penalty_sweeps[gp]:
                print(f"     tau={row['tau']:.2f}: right={row['confident_right']:.3f} "
                      f"WRONG={row['confident_wrong']:.3f} hedged={row['hedged']:.3f} "
                      f"spec_generic={row['specific_genericness']}", flush=True)

        # ---- per-item text dump for subjective review ----
        if dump is not None:
            UNSAFE = {"F35_clearly_harmful", "F36_harmful_obfuscated",
                      "I44_emotional_manipulation", "L59_nsfw_explicit"}
            for r in recs:
                t = r["id"]
                cat, grp = cat_meta.get(t, (t.rsplit("_", 1)[0], None))
                unsafe = cat in UNSAFE
                free = select_policy(r["s_emb"], r["tstar"], args.dump_tau,
                                     generic_centroid, 0.0)
                pen = select_policy(r["s_emb"], r["tstar"], args.dump_tau,
                                    generic_centroid, args.dump_penalty)
                dump["items"].append({
                    "layer": L, "id": t, "category": cat, "group": grp,
                    "unsafe": unsafe,
                    "input_text": ("[unsafe category %s — input withheld]" % cat)
                    if unsafe else texts[t],
                    "greedy": r["greedy"],
                    "greedy_correct": bool(is_self(r["g_emb"], t)),
                    "policy_nopen": apply_policy_text(r["samples"], free),
                    "policy_nopen_decision": free["decision"],
                    "policy_nopen_conf": round(free["confidence"], 3),
                    "policy_nopen_correct":
                        bool(is_self(r["s_emb"][free["idx"]], t)),
                    "policy_pen": apply_policy_text(r["samples"], pen),
                    "policy_pen_decision": pen["decision"],
                    "policy_pen_conf": round(pen["confidence"], 3),
                    "policy_pen_correct":
                        bool(is_self(r["s_emb"][pen["idx"]], t)),
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)
    if dump is not None:
        Path(args.dump_texts).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump, open(args.dump_texts, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {args.dump_texts}  ({len(dump['items'])} items)", flush=True)


def build_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="run the GPU batch evaluator")
    ap.add_argument("--av-adapter", default="output/nla-phi4-universal-av-v2")
    ap.add_argument("--compass", default="output/av_oracle_compass.pt")
    ap.add_argument("--acts", default=str(Path.home() / "phi4_ar/phi4_13depths.pt"))
    ap.add_argument("--holdout", default="output/roundtrip_v2corpus/holdout.json")
    ap.add_argument("--corpus-glob", default=str(REPO / "corpus/generated/*.json"))
    ap.add_argument("--layers", default="16,25")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gen-penalty", type=float, default=0.0,
                    help="weight on the genericness penalty in selection (0=off)")
    ap.add_argument("--gen-penalties", default="0.0,0.3,0.5",
                    help="comma list of genericness-penalty weights to sweep in "
                         "--eval (all reuse the same generated samples)")
    ap.add_argument("--taus", default="0.20,0.25,0.30,0.35,0.40",
                    help="confidence thresholds to sweep")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=0,
                    help="cap holdout items per layer (0=all); use for fast dumps")
    ap.add_argument("--save-centroid", default=None, metavar="PATH",
                    help="persist the data-driven generic_centroid (so "
                         "describe_live --policy can apply the genericness penalty)")
    ap.add_argument("--dump-texts", default=None, metavar="PATH",
                    help="also write a per-item side-by-side text dump "
                         "(greedy vs policy pick) for subjective review")
    ap.add_argument("--dump-tau", type=float, default=0.30,
                    help="tau used for the --dump-texts policy pick")
    ap.add_argument("--dump-penalty", type=float, default=0.30,
                    help="gen_penalty used for the --dump-texts penalized pick")
    ap.add_argument("--faith-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "output/av_policy_eval.json"))
    return ap.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.eval:
        run_eval(args)
    else:
        print("nothing to do; pass --eval for the GPU batch evaluator "
              "(or import the pure policy functions).")
