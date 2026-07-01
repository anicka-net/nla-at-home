#!/usr/bin/env python3
r"""
train_av_raft.py — RAFT / best-of-N distillation for the AV.  GPU.  *** UNTESTED ***

=============================================================================
*** HUMAN REVIEW REQUIRED BEFORE LAUNCH ***
This changes the training OBJECTIVE (AGENTS.md "What Requires Human Review":
"Changes to the training objective or reward signal"). The code is written and
the selection/scoring logic is unit-tested on CPU (tests/test_raft_select.py),
but it has NOT been run on GPU. Do not launch without operator approval.
=============================================================================

WHY THIS EXISTS
---------------
The diagnostic arc proved the AV's confabulation is a DECODING problem, not a
representational, injection, or capacity one:
  * linear oracle recovers identity from the activation at ~0.83  (info is there)
  * direction-only (injected) oracle ~0.84                        (survives inject)
  * best-of-8 truth-rerank 0.64 vs greedy 0.38                    (faithful descs
    already EXIST in the AV's own sample distribution; greedy just picks a
    confabulated mode)

RAFT (Reward-rAnked Fine-Tuning / rejection-sampling FT) bakes that selection
into the greedy mode: for each TRAIN activation, sample N descriptions from the
current policy, keep the one that best matches the activation's TRUE input text,
and SFT the AV on those self-generated faithful descriptions. Iterate. This moves
the greedy mode toward faithful using only supervised cross-entropy on the model's
OWN best outputs — no brittle RL reward.

It reframes the faithfulness-reward GRPO null: SELECTION on the identity signal
clearly works (best-of-N), even though the in-batch contrastive REWARD didn't
generalize. RAFT distills the selection instead of rewarding it.

SELECTION (TRAIN only — input text is known for train ids):
    s(d) = cos( MiniLM(d), MiniLM(input_text) )           # truth-grounded
    keep argmax_d over the N samples; DROP the example if max s < --min-score
    (no faithful sample this round -> never distill a confabulation).
At INFERENCE there is no input text, so faithfulness is read out with the ORACLE
COMPASS reranker from best_of_n_oracle_rerank.py (pick sample closest to W*a);
RAFT's job is to lift the greedy baseline that reranker starts from.

LOOP
----
  for round in 1..R:
    GENERATE   N samples per (layer,id) from the CURRENT policy, over a random
               subset of --samples-per-round (layer,id) pairs (cost control)
    SELECT     best desc per pair by s(d); filter by --min-score; optional top-k
    SFT        train the policy on {(prompt + injected activation) -> kept desc}
               for --sft-epochs (standard AV cross-entropy; prompt masked)
    EVAL       (optional) greedy holdout input-text top1, for a progress curve
    SAVE       round adapter

EXPECTED COMMAND (deepthought; ONE 14B job at a time; AWAIT HUMAN APPROVAL)
--------------------------------------------------------------------------
  ~/venv/bin/python scripts/train_av_raft.py \
    --av-adapter output/nla-phi4-universal-av-v2 \
    --acts ~/phi4_ar/phi4_13depths.pt \
    --holdout output/roundtrip_v2corpus/holdout.json \
    --layers 4,10,16,19,25,32,38 --n-samples 8 --temperature 0.9 \
    --samples-per-round 1200 --rounds 3 --sft-epochs 1 \
    --min-score 0.30 --out-dir output/nla-phi4-universal-av-raft
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from probe_activation_faithfulness import load_corpus_texts  # noqa: E402
from train_universal_av import (  # noqa: E402
    INJECTION_SCALE, INJECTION_CHARS, DEPTH_PCTS, make_prompt, find_inject_pos,
    normalize_activation, nearest_depth_pct, UniversalNLADataset, collate_fn)

BASE = "microsoft/phi-4"
N_LAYERS = 40


def nearest_pct(layer):
    return nearest_depth_pct(layer, N_LAYERS)


# ---------------------------------------------------------------------------
# Pure, unit-testable selection: pick the sampled description whose MiniLM
# embedding is closest to the activation's TRUE input-text embedding.
# ---------------------------------------------------------------------------
def select_best(descs, sample_embs, text_emb, min_score):
    """descs: list[str] (len N). sample_embs: (N,d) L2-normalized. text_emb: (d,)
    L2-normalized true input-text embedding. Returns (best_desc|None, best_score).
    None when the best sample fails the faithfulness floor."""
    if not descs:
        return None, float("-inf")
    sims = sample_embs @ text_emb
    j = int(np.argmax(sims))
    best = float(sims[j])
    return (descs[j] if best >= min_score else None), best


def l2norm_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _top1(Q, K):
    if Q.shape[0] == 0:
        return float("nan")
    sims = Q @ K.T
    return float((sims.argmax(axis=1) == np.arange(Q.shape[0])).mean())


def build_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av-adapter", required=True, help="starting SFT AV adapter")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--corpus-glob", default=str(REPO / "corpus/generated/*.json"))
    ap.add_argument("--layers", default="4,10,16,19,25,32,38")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--keep-top-k", type=int, default=1,
                    help="distill the K best samples per (layer,id)")
    ap.add_argument("--min-score", type=float, default=0.30,
                    help="drop a (layer,id) if its best sample's cos < this")
    ap.add_argument("--samples-per-round", type=int, default=1200,
                    help="random # of (layer,id) pairs to generate per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--sft-lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--faith-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--eval-each-round", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "output/nla-phi4-universal-av-raft"))
    return ap.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAFT / best-of-N distillation for the AV (UNTESTED; HUMAN-REVIEW gated)")
    print(f"  layers={layers} N={args.n_samples} rounds={args.rounds} "
          f"per_round={args.samples_per_round} min_score={args.min_score}")
    print("=" * 70, flush=True)

    # ---- data ----
    D = torch.load(args.acts, weights_only=True, map_location="cpu")
    acts, ids = D["activations"], D["ids"]
    id2idx = {t: i for i, t in enumerate(ids)}
    holdout = json.load(open(args.holdout))["holdout"]
    ho_set = set(holdout)
    texts = load_corpus_texts(args.corpus_glob)
    train_ids = [t for t in ids if t in texts and t not in ho_set]
    pairs = [(L, t) for L in layers for t in train_ids]
    print(f"  acts={len(ids)} train_ids={len(train_ids)} pairs={len(pairs)} "
          f"holdout={len(holdout)} texts={len(texts)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.faith_model, device="cpu")

    def embed(strs):
        if not strs:
            return np.zeros((0, enc.get_sentence_embedding_dimension()))
        return enc.encode(list(strs), normalize_embeddings=True,
                          convert_to_numpy=True, batch_size=128,
                          show_progress_bar=False).astype(np.float64)

    txt_emb = {t: e for t, e in zip(texts.keys(), embed(list(texts.values())))}

    # ---- model (GPU) ----
    print("loading AV policy (Phi-4 + LoRA, trainable)...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.av_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False).to(device)
    av = PeftModel.from_pretrained(base, args.av_adapter, is_trainable=True)
    inj_char = INJECTION_CHARS["phi4"]
    inj_id = tok.encode(inj_char, add_special_tokens=False)[0]
    emb_layer = av.get_input_embeddings()

    prompt_cache = {}
    for L in layers:
        pct = nearest_pct(L)
        if pct in prompt_cache:
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(pct, inj_char)}],
            tokenize=False, add_generation_prompt=True)
        toks = tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (toks, find_inject_pos(toks, inj_id))

    def clean(s):
        return s.split("</explanation>")[0].strip()

    @torch.no_grad()
    def sample_descs(L, t, n):
        ptoks, ipos = prompt_cache[nearest_pct(L)]
        a = acts[L][id2idx[t]].float()
        input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
        emb = emb_layer(input_ids)
        emb[0, ipos, :] = normalize_activation(a.to(device), INJECTION_SCALE).to(emb.dtype)
        embN = emb.expand(n, -1, -1).contiguous()
        out = av.generate(
            inputs_embeds=embN.to(av.dtype),
            attention_mask=torch.ones(n, embN.shape[1], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
        return [clean(tok.decode(s, skip_special_tokens=True)) for s in out.sequences]

    @torch.no_grad()
    def greedy_desc(L, t):
        ptoks, ipos = prompt_cache[nearest_pct(L)]
        a = acts[L][id2idx[t]].float()
        input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
        emb = emb_layer(input_ids)
        emb[0, ipos, :] = normalize_activation(a.to(device), INJECTION_SCALE).to(emb.dtype)
        out = av.generate(
            inputs_embeds=emb.to(av.dtype), attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
        return clean(tok.decode(out.sequences[0], skip_special_tokens=True))

    def eval_holdout():
        """Greedy holdout input-text top1, mean over layers — comparable to 0.406."""
        av.eval()
        per_layer = []
        for L in layers:
            ho_ids = [t for t in holdout if t in id2idx and t in texts]
            descs = [greedy_desc(L, t) for t in ho_ids]
            Q = l2norm_rows(embed(descs))
            K = l2norm_rows(np.stack([txt_emb[t] for t in ho_ids]))
            per_layer.append(_top1(Q, K))
        return float(np.nanmean(per_layer))

    def sft(examples):
        """SFT the policy on kept self-generated descriptions (AV CE, prompt masked)."""
        ds = UniversalNLADataset(examples, tok, inj_char, max_length=args.max_len)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        opt = torch.optim.AdamW([p for p in av.parameters() if p.requires_grad],
                                lr=args.sft_lr, weight_decay=0.01)
        av.train()
        for ep in range(args.sft_epochs):
            tot, nb = 0.0, 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn = batch["attention_mask"].to(device)
                activations = batch["activations"].to(device)
                emb = emb_layer(input_ids)
                for i, pos in enumerate(batch["inject_positions"]):
                    emb[i, pos, :] = normalize_activation(activations[i], INJECTION_SCALE)
                o = av(inputs_embeds=emb, attention_mask=attn, labels=labels)
                o.loss.backward()
                torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                tot += o.loss.item(); nb += 1
            print(f"    sft epoch {ep+1}/{args.sft_epochs}: loss={tot/max(1,nb):.3f}",
                  flush=True)

    # ---- RAFT rounds ----
    history = []
    if args.eval_each_round:
        base_top1 = eval_holdout()
        print(f"[round 0] greedy holdout input-text top1 = {base_top1:.3f}", flush=True)
        history.append({"round": 0, "inputtext_top1": base_top1})

    for r in range(1, args.rounds + 1):
        sel = rng.permutation(len(pairs))[: args.samples_per_round]
        round_pairs = [pairs[i] for i in sel]
        av.eval()
        examples, kept, dropped, scores = [], 0, 0, []
        for k, (L, t) in enumerate(round_pairs):
            descs = sample_descs(L, t, args.n_samples)
            S = l2norm_rows(embed(descs))
            sims = S @ txt_emb[t]
            order = np.argsort(-sims)[: args.keep_top_k]
            took = False
            for j in order:
                if float(sims[j]) >= args.min_score:
                    examples.append({
                        "activation": acts[L][id2idx[t]],
                        "description": descs[int(j)],
                        "depth_pct": nearest_pct(L),
                        "layer": L, "text_id": t})
                    took = True
            scores.append(float(sims[order[0]]))
            kept += int(took); dropped += int(not took)
            if (k + 1) % 200 == 0:
                print(f"  round {r}: generated {k+1}/{len(round_pairs)} "
                      f"(kept {kept} dropped {dropped} "
                      f"meanbest {np.mean(scores):.3f})", flush=True)
        print(f"[round {r}] selected {len(examples)} examples from "
              f"{len(round_pairs)} pairs (kept {kept} dropped {dropped}, "
              f"mean best-sample cos {np.mean(scores):.3f})", flush=True)
        if not examples:
            print("  no examples passed the floor — stopping.", flush=True)
            break
        sft(examples)
        av.save_pretrained(str(out / f"round{r}"))
        tok.save_pretrained(str(out / f"round{r}"))
        rec = {"round": r, "n_examples": len(examples), "kept": kept,
               "dropped": dropped, "mean_best_cos": float(np.mean(scores))}
        if args.eval_each_round:
            top1 = eval_holdout()
            rec["inputtext_top1"] = top1
            print(f"[round {r}] greedy holdout input-text top1 = {top1:.3f}", flush=True)
        history.append(rec)
        json.dump({"history": history, "config": vars(args)},
                  open(out / "raft_history.json", "w"), indent=2)

    av.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"saved final RAFT adapter -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
