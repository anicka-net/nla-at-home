#!/usr/bin/env python3
r"""
Joint GRPO co-training of AV + AR for the Phi-4 v2 line (UNTESTED — needs GB10).

=============================================================================
STATUS / REVIEW BANNER
=============================================================================
This script introduces a NEW training objective (round-trip reward + KL
legibility anchor + AR co-refinement). Per AGENTS.md, "Changes to the training
objective or reward signal" REQUIRE HUMAN REVIEW before any produced adapter is
trusted or published. It is written but has NOT been run on GPU. The GPU code
paths are marked `# UNTESTED`. Run only as ONE 14B job at a time on the GB10
(see DESIGN.md / LOCAL.md: never co-run two 14B generation jobs).

=============================================================================
WHY THIS EXISTS — "improve AV and AR simultaneously"
=============================================================================
The round trip is a loop:

    text -> activation a@L -> [AV] -> description d -> [AR] -> reconstructed â@L
    faithfulness = centered-cos(a, â)

Today AV and AR are trained separately and only SHARE a description
distribution. `scripts/train_universal_grpo.py` already rewards AV with a
FROZEN AR's cosine (phi4-mini line) — but (1) AR never improves from the loop,
and (2) there is NO KL anchor, so the AV policy is free to drift toward an
AR-readable private code (illegible descriptions). That is mode collapse
(DESIGN.md Failure 1) wearing a cleverer mask: an NLA whose descriptions a
human cannot read is useless.

This trainer closes BOTH gaps, safely:

  (A) AV step (GRPO): reward = contrastive round-trip cos, with an explicit
      per-token KL leash to the SFT AV policy (the "legibility anchor").
      reward_recon = cos(â_correct, a) - cos(â_correct, a_wrongtext)
      loss = mean_t[ -advantage * logπ_t  +  β * KL(π_t ‖ π_ref_t) ]
      The KL term holds descriptions in natural-language space; the contrastive
      term forces the description to be activation-SPECIFIC, not generic.

  (B) AR step (collusion-safe co-improvement): AR refines on GROUND-TRUTH
      (human / gpt4o) descriptions ONLY — never on AV's sampled outputs. So AR
      stays a *human-language decoder*; it cannot open a private channel with
      AV. Each AR improvement raises the reward ceiling AV chases. This is the
      sound form of "simultaneous": the reward couples them, the training data
      does not.

  (C) Judge: every epoch we score the LEAK-FREE double-holdout built by
      scripts/eval_roundtrip_phi4.py --phase split (AR-val ∩ AV-val). You
      cannot safely optimize a metric you can leak into; the holdout is the
      held-out judge that makes (A)+(B) honest. We keep the AV+AR that maximize
      holdout round-trip cos.

  (D) FAITHFULNESS REWARD (lever #6, --faith-coef λ>0; operator-approved
      2026-06-21 after the twin_clean re-SFT null). The round-trip cosine of
      (A) is a WEAK proxy — the joint win gamed it (+0.022 cos, ZERO
      faithfulness gain): the AV still CONFABULATES the text. This term rewards
      the description in all-MiniLM TEXT space directly, contrastively against
      K in-batch GT-desc negatives (mirrors the validated identification
      metric):
        R_faith(d) = cos(E(d), E(gt_correct)) - mean_j cos(E(d), E(gt_neg_j))
      Mixed in advantage space, renormalized to unit variance (lr-stable):
        adv = z( (1-λ)·z(R_rt) + λ·z(R_faith) )
      ANTI-HACK: (1) reward uses TRAIN-id GT + sampled negatives, eval uses the
      held-out 50; (2) each epoch reports the held-out STYLE-FAIR input-text
      top1 (bar = av-v2 0.406) alongside matched-GT top1 — if the reward games
      GT-desc STYLE, input-text top1 won't move (the same discriminator that
      caught the twin_clean null); (3) keeping R_rt (λ<1) + the KL anchor
      prevents faithful-sounding-but-unreconstructable / illegible text.
      λ=0 (default) reduces EXACTLY to the round-trip-only trainer above.

If --no-train-ar is passed, (B) is skipped and this reduces to "AV GRPO with a
KL anchor against a frozen v2 AR" — already a strict improvement over the
existing anchorless trainer, and a good first ablation.

=============================================================================
EXPECTED COMMAND (after the round-trip --phase split has produced holdout.json)
=============================================================================
  source ~/venv/bin/activate            # torch 2.11+cu130 — NOT system python3
  python scripts/train_joint_grpo_phi4.py \
    --av-adapter   output/nla-phi4-universal-av-v2 \
    --ar-best      ~/phi4_ar/stage2_v2corpus_best.pt \
    --ar-value-heads ~/phi4_ar/stage2_v2corpus_value_heads.pt \
    --ar-acts      ~/phi4_ar/phi4_13depths.pt \
    --ar-layers    4,10,16,19,25,32,38 \
    --ar-desc-json ~/phi4_ar/descriptions_phi4_tokenpred_gpt4o.json \
    --ar-desc-key-map 4:4,10:10,16:16,19:20,25:26,32:32,38:38 \
    --holdout      output/roundtrip_v2corpus/holdout.json \
    --epochs 6 --samples-per-epoch 200 --group-size 4 \
    --av-lr 5e-6 --kl-coef 0.05 --rep-penalty 0.2 --contrastive \
    --train-ar --ar-lr 1e-4 --ar-steps-per-epoch 50 \
    --faith-coef 0.5 --faith-distractors 8 \
    --out-dir output/nla-phi4-universal-av-v2-joint-faith

Quick load/eval sanity (cheap-ish, GPU; no weight updates):
  python scripts/train_joint_grpo_phi4.py ... --phase eval
  (faith eval runs even at λ=0, so --phase eval reports the faith top1 bars.)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_universal_av import (  # noqa: E402
    INJECTION_SCALE, make_prompt, find_inject_pos, normalize_activation,
)

BASE = "microsoft/phi-4"
INJECTION_CHAR = "★"
AR_PROMPT = "Summary of the following text: <text>{e}</text> <summary>"
DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]
N_LAYERS = 40
AR_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def nearest_pct(layer):
    d = layer * 100 / N_LAYERS
    return min(DEPTH_PCTS, key=lambda p: abs(p - d))


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="train", choices=["train", "eval"],
                    help="train = joint GRPO; eval = load + score holdout only (no updates)")
    # AV side
    ap.add_argument("--av-adapter", default=str(REPO / "output/nla-phi4-universal-av-v2"))
    # AR side (v2 phi-4 format: LoRA state-dict + value_heads.pt)
    ap.add_argument("--ar-best", required=True,
                    help="{tag}_best.pt — dict with 'lora' state + 'args' (lora_r, dropout)")
    ap.add_argument("--ar-value-heads", required=True,
                    help="{tag}_value_heads.pt — {str(L): weight}")
    ap.add_argument("--ar-acts", required=True,
                    help="phi4_13depths.pt — torch dict 'activations'{L: NxD}, 'ids'")
    ap.add_argument("--ar-layers", default="4,10,16,19,25,32,38")
    ap.add_argument("--ar-desc-json", required=True,
                    help="GT descriptions json (records id/layer/description) for AR refine + ceiling")
    ap.add_argument("--ar-desc-key-map", default="",
                    help="a:b layer remap into --ar-desc-json (e.g. 19:20,25:26); identity if absent")
    # holdout + io
    ap.add_argument("--holdout", required=True,
                    help="holdout.json from eval_roundtrip_phi4.py --phase split")
    ap.add_argument("--max-holdout", type=int, default=0,
                    help="cap holdout texts used for scoring (0=all). Use a small "
                         "value (e.g. 8) for a fast --phase eval sanity load.")
    ap.add_argument("--out-dir", default=str(REPO / "output/nla-phi4-universal-av-v2-joint"))
    ap.add_argument("--device", default="cuda")
    # GRPO knobs
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--av-lr", type=float, default=5e-6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--kl-coef", type=float, default=0.05,
                    help="β: per-token KL(policy ‖ SFT-AV) legibility anchor")
    ap.add_argument("--rep-penalty", type=float, default=0.2)
    ap.add_argument("--contrastive", action="store_true",
                    help="reward = cos_correct - cos_wrong-text (forces specificity)")
    ap.add_argument("--center-reward", action="store_true", default=True,
                    help="subtract train-set layer mean before cosine (matches centered-cos metric)")
    # AR refinement (collusion-safe co-training)
    ap.add_argument("--train-ar", action="store_true",
                    help="alternate AR refinement on GT descriptions only")
    ap.add_argument("--no-train-ar", dest="train_ar", action="store_false")
    ap.add_argument("--ar-lr", type=float, default=1e-4)
    ap.add_argument("--ar-steps-per-epoch", type=int, default=50)
    ap.add_argument("--ar-batch", type=int, default=8)
    ap.add_argument("--ar-max-len", type=int, default=320)
    # faithfulness reward (lever #6 — operator-approved objective change)
    ap.add_argument("--faith-coef", type=float, default=0.0,
                    help="λ: mix weight of the faithfulness reward in advantage "
                         "space: adv = (1-λ)·z(R_roundtrip) + λ·z(R_faith). "
                         "0 (default) = legacy round-trip-only behavior.")
    ap.add_argument("--faith-distractors", type=int, default=8,
                    help="K: # of random other-train-id GT descs used as "
                         "in-batch negatives for the contrastive faith reward")
    ap.add_argument("--faith-model",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-transformers encoder for the faith reward + "
                         "faith eval (same space as the validated metric)")
    ap.add_argument("--corpus-glob", default=str(REPO / "corpus/generated/*.json"),
                    help="glob of {id,text,...} json for the STYLE-FAIR input-text "
                         "faith eval (the honest judge; bar = av-v2 0.406)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_gt_descs(args, ar_layers):
    """GT descriptions per AR layer from the single json, via key-map."""
    keymap = ({int(a): int(b) for a, b in
               (kv.split(":") for kv in args.ar_desc_key_map.split(","))}
              if args.ar_desc_key_map else {})
    recs = json.load(open(args.ar_desc_json))
    descmap = {}
    for L in ar_layers:
        dk = keymap.get(L, L)
        descmap[L] = {x["id"]: x["description"] for x in recs
                      if x["layer"] == dk and x.get("description")}
    return descmap


def load_corpus_texts(args):
    """id -> input text, for the STYLE-FAIR input-text faith eval. Best-effort:
    returns {} if the glob matches nothing (eval just skips input-text top1)."""
    import glob
    texts = {}
    for fp in glob.glob(args.corpus_glob):
        try:
            for x in json.load(open(fp)):
                if isinstance(x, dict) and x.get("id") and x.get("text"):
                    texts[x["id"]] = x["text"]
        except Exception:
            continue
    return texts


# ---------------------------------------------------------------------------
# faithfulness reward (lever #6) — text-semantic-space scoring of AV descs
# ---------------------------------------------------------------------------
class FaithScorer:
    """Wraps the all-MiniLM encoder used by both the faith reward and the faith
    eval. GT-desc embeddings are precomputed once (they are fixed); only the
    freshly generated AV descriptions are encoded per step."""

    def __init__(self, model_name, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=str(device))
        self.cache = {}  # text -> normalized vector (torch.float32, on device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        """list[str] -> (N, d) L2-normalized float32 tensor on self.device."""
        v = self.model.encode(list(texts), normalize_embeddings=True,
                              convert_to_numpy=True, batch_size=64,
                              show_progress_bar=False)
        return torch.from_numpy(v).float().to(self.device)

    def precompute(self, gt_descs, ar_layers):
        """Embed every GT description once -> self.gt_emb[L][id] = vector."""
        self.gt_emb = {}
        for L in ar_layers:
            ids = list(gt_descs[L].keys())
            if not ids:
                self.gt_emb[L] = {}
                continue
            embs = self.encode([gt_descs[L][i] for i in ids])
            self.gt_emb[L] = {i: embs[k] for k, i in enumerate(ids)}

    def faith_reward(self, descs, L, t, distractor_ids):
        """Contrastive identification reward (one scalar per generated desc):
        cos(E(d), E(gt_t)) - mean_j cos(E(d), E(gt_distractor_j)).
        Returns a python list[float]; falls back to 0.0 if the GT is missing."""
        gt_t = self.gt_emb.get(L, {}).get(t)
        if gt_t is None:
            return [0.0] * len(descs)
        D = self.encode(descs)                       # (G, d)
        pos = D @ gt_t                               # (G,)
        negs = [self.gt_emb[L][j] for j in distractor_ids
                if j in self.gt_emb.get(L, {})]
        if negs:
            N = torch.stack(negs)                    # (K, d)
            neg = (D @ N.T).mean(dim=1)              # (G,)
        else:
            neg = torch.zeros_like(pos)
        return (pos - neg).tolist()


def load_ar(args, ar_layers, device):
    """Rebuild the v2 phi-4 AR: base + LoRA (from {tag}_best.pt) + value_heads.

    Returns (ar_model, value_heads, tok). AR backbone keeps its norm/lm_head;
    recon reads hidden_states[L+1] directly (NOT lm_head), per phi_ar_pilot.py.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        from peft.utils import set_peft_model_state_dict

    ckpt = torch.load(args.ar_best, weights_only=False, map_location="cpu")
    ck_args = ckpt.get("args", {})
    r = int(ck_args.get("lora_r", 16))
    dropout = float(ck_args.get("dropout", 0.0))

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False)
    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, bias="none",
                      target_modules=AR_TARGET_MODULES, task_type="CAUSAL_LM")
    ar_model = get_peft_model(backbone, lora).to(device)
    set_peft_model_state_dict(ar_model, {k: v.to(device) for k, v in ckpt["lora"].items()})

    vh_state = torch.load(args.ar_value_heads, weights_only=True, map_location=device)
    d_model = vh_state[str(ar_layers[0])].shape[1]
    value_heads = torch.nn.ModuleDict({
        str(L): torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        for L in ar_layers
    })
    for L in ar_layers:
        value_heads[str(L)].weight = torch.nn.Parameter(vh_state[str(L)].float())
    value_heads = value_heads.to(device)
    ar_model.eval()  # deterministic recon; refinement loop toggles train/eval itself
    return ar_model, value_heads, tok, d_model, ck_args


def ar_recon(ar_model, value_heads, ar_tok, descs, layer, device, max_len, no_grad=True):
    """descs: list[str] -> reconstructed activations (len(descs) x d), value-head output."""
    prompts = [AR_PROMPT.format(e=d) for d in descs]
    enc = ar_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=max_len).to(device)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        h = ar_model(**enc, output_hidden_states=True).hidden_states[layer + 1][:, -1, :].float()
        return value_heads[str(layer)](h)


# ---------------------------------------------------------------------------
# AV policy (+ frozen SFT reference adapter for the KL anchor)
# ---------------------------------------------------------------------------
def load_av(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.av_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False).to(device)
    # policy = trainable copy of the SFT AV adapter
    av = PeftModel.from_pretrained(base, args.av_adapter, adapter_name="policy",
                                   is_trainable=True)
    # ref = frozen SFT AV weights on the SAME base (no third 14B copy). KL leash.
    av.load_adapter(args.av_adapter, adapter_name="ref")  # UNTESTED: multi-adapter
    av.set_adapter("policy")
    return av, tok


def av_logprobs(av, tok, prompt_tokens, inject_pos, act, gen_ids, device, adapter,
                do_set=True):
    """Per-token logprobs of gen_ids under the given adapter, activation injected
    at inject_pos. grad-capable iff adapter == 'policy'. If do_set is False the
    caller is responsible for having activated the right adapter (so we don't
    flip requires_grad on the policy params mid-update)."""
    if do_set:
        av.set_adapter(adapter)
    full = prompt_tokens + gen_ids
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    emb_layer = av.get_input_embeddings()
    emb = emb_layer(input_ids)
    emb[0, inject_pos, :] = normalize_activation(act.to(device), INJECTION_SCALE).to(emb.dtype)
    grad = (adapter == "policy")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if grad and not emb.requires_grad:
            emb.requires_grad_(True)
        logits = av(inputs_embeds=emb.to(av.dtype),
                    attention_mask=torch.ones_like(input_ids)).logits[0]
    pl = len(prompt_tokens)
    gen_logits = logits[pl - 1: pl - 1 + len(gen_ids)]
    lp = torch.nn.functional.log_softmax(gen_logits.float(), dim=-1)
    gt = torch.tensor(gen_ids, dtype=torch.long, device=device)
    return lp[torch.arange(len(gen_ids), device=device), gt]


# ---------------------------------------------------------------------------
# holdout round-trip eval (the leak-free judge)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_holdout(av, av_tok, ar_model, value_heads, ar_tok, acts, id2idx,
                 layer_mu, holdout, ar_layers, prompt_cache, args, device):
    """Greedy AV -> AR recon -> centered round-trip cos on holdout; plus the
    GT-desc ceiling and (if a FaithScorer is attached) the faithfulness top1
    metrics. Returns (rt_mean, ceil_mean, faith_dict|None)."""
    av.set_adapter("policy")
    emb_layer = av.get_input_embeddings()
    gt = eval_holdout.gt_descs  # injected by caller
    faith = getattr(eval_holdout, "faith", None)
    corpus_texts = getattr(eval_holdout, "corpus_texts", {}) or {}
    rt_layers, ceil_layers = [], []
    descs_by_layer = {}  # L -> (ids, av_descs) aligned, for faith top1
    for L in ar_layers:
        ptoks, ipos = prompt_cache[nearest_pct(L)]
        P_rt, P_ceil, T = [], [], []
        ev_ids, ev_descs = [], []
        for t in holdout:
            if t not in id2idx or t not in gt[L]:
                continue
            a = acts[L][id2idx[t]].float()
            input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
            emb = emb_layer(input_ids)
            emb[0, ipos, :] = normalize_activation(a.to(device), INJECTION_SCALE).to(emb.dtype)
            out = av.generate(inputs_embeds=emb.to(av.dtype),
                              attention_mask=torch.ones_like(input_ids),
                              max_new_tokens=args.max_new_tokens, do_sample=False,
                              pad_token_id=av_tok.eos_token_id,
                              return_dict_in_generate=True)
            desc = av_tok.decode(out.sequences[0], skip_special_tokens=True)
            desc = desc.split("</explanation>")[0].strip()
            P_rt.append(ar_recon(ar_model, value_heads, ar_tok, [desc], L, device,
                                 args.ar_max_len)[0].cpu().double())
            P_ceil.append(ar_recon(ar_model, value_heads, ar_tok, [gt[L][t]], L, device,
                                   args.ar_max_len)[0].cpu().double())
            T.append(a.double())
            ev_ids.append(t)
            ev_descs.append(desc)
        descs_by_layer[L] = (ev_ids, ev_descs)
        if not T:
            continue
        T = torch.stack(T)
        # CANONICAL metric: batch-mean-centered cos over the holdout, per layer,
        # then mean over layers — identical to eval_roundtrip_phi4.centered_cos
        # so the judge is directly comparable to the 0.587 baseline. (The reward
        # uses per-sample train-mean centering via _cc, since batch-centering is
        # not available for a single sample.)
        rt_layers.append(_centered_cos(torch.stack(P_rt), T))
        ceil_layers.append(_centered_cos(torch.stack(P_ceil), T))
    f = lambda xs: sum(xs) / max(1, len(xs))
    faith_dict = None
    if faith is not None:
        faith_dict = _faith_top1(faith, descs_by_layer, gt, corpus_texts, ar_layers)
    return f(rt_layers), f(ceil_layers), faith_dict


def _top1(query, keys):
    """query/keys: (n, d) aligned by row (i-th query's correct key is row i).
    Returns fraction of rows whose argmax-cosine over keys equals i."""
    if query.shape[0] == 0:
        return float("nan")
    sims = query @ keys.T                       # (n, n), rows already normalized
    return (sims.argmax(dim=1) == torch.arange(query.shape[0],
            device=sims.device)).float().mean().item()


def _faith_top1(faith, descs_by_layer, gt, corpus_texts, ar_layers):
    """Per-layer identification top1, averaged over layers.
    - matched_top1: AV desc retrieves its own GT desc (style-confounded).
    - inputtext_top1: AV desc retrieves its own INPUT TEXT (style-fair judge).
    NOTE: this is a training MONITOR; the authoritative comparison to the
    av-v2 0.406 bar is the standalone faithfulness eval script."""
    matched, inputt = [], []
    for L in ar_layers:
        ids, av_descs = descs_by_layer.get(L, ([], []))
        if not ids:
            continue
        Q = faith.encode(av_descs)                                  # (n, d)
        gt_keys = faith.encode([gt[L][t] for t in ids])             # (n, d)
        matched.append(_top1(Q, gt_keys))
        txt_ids = [t for t in ids if t in corpus_texts]
        if len(txt_ids) == len(ids) and ids:
            txt_keys = faith.encode([corpus_texts[t] for t in ids])
            inputt.append(_top1(Q, txt_keys))
    g = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"matched_top1": g(matched), "inputtext_top1": g(inputt)}


def _centered_cos(P, T):
    Pc = P - P.mean(0, keepdim=True)
    Tc = T - T.mean(0, keepdim=True)
    return torch.nn.functional.cosine_similarity(Pc, Tc, dim=1).mean().item()


def _cc(pred, tgt, mu, center):
    if center:
        pred = pred - mu
        tgt = tgt - mu
    return torch.nn.functional.cosine_similarity(
        pred.float().unsqueeze(0), tgt.float().unsqueeze(0)).item()


def _z(x):
    """Standardize a 1-D reward tensor to ~unit variance (0 if degenerate)."""
    s = x.std()
    if s.item() < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / (s + 1e-8)


def _fmt_faith(fd):
    if not fd:
        return "n/a"
    m = "%.3f" % fd["matched_top1"] if fd.get("matched_top1") is not None else "n/a"
    it = "%.3f" % fd["inputtext_top1"] if fd.get("inputtext_top1") is not None else "n/a"
    return f"{m}/{it}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = build_args()
    device = torch.device(args.device)
    ar_layers = [int(x) for x in args.ar_layers.split(",")]
    torch.manual_seed(args.seed)
    import numpy as np
    rng = np.random.RandomState(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JOINT GRPO (UNTESTED) — AV policy + KL anchor + AR-on-GT refine")
    print(f"  ar_layers={ar_layers}  train_ar={args.train_ar}  kl={args.kl_coef}")
    print("=" * 70, flush=True)

    # ---- data ----
    D = torch.load(args.ar_acts, weights_only=True, map_location="cpu")
    acts = D["activations"]
    id2idx = {t: i for i, t in enumerate(D["ids"])}
    gt_descs = load_gt_descs(args, ar_layers)
    holdout = json.load(open(args.holdout))["holdout"]
    if args.max_holdout:
        holdout = holdout[: args.max_holdout]
    print(f"  acts ids={len(D['ids'])}  holdout={len(holdout)}  "
          f"gt per-layer={[len(gt_descs[L]) for L in ar_layers]}", flush=True)

    # train-set layer means (centered-cos reference). Mirror AR: 90/10 by sorted common.
    common = set(D["ids"])
    for L in ar_layers:
        common &= set(gt_descs[L])
    common = sorted(common)
    ntr = int(len(common) * 0.9)
    train_ids = common[:ntr]
    train_idx = [id2idx[t] for t in train_ids]
    layer_mu = {L: acts[L][train_idx].float().mean(0).to(device) for L in ar_layers}

    # ---- models (UNTESTED past this point — GPU) ----
    print("loading AR (LoRA + value_heads)...", flush=True)
    ar_model, value_heads, ar_tok, d_model, ar_ck_args = load_ar(args, ar_layers, device)
    print("loading AV (policy + frozen ref adapter)...", flush=True)
    av, av_tok = load_av(args, device)
    av.eval()  # deterministic logprobs (LoRA dropout off); grads still flow
    inj_id = av_tok.encode(INJECTION_CHAR, add_special_tokens=False)[0]

    prompt_cache = {}
    for L in ar_layers:
        pct = nearest_pct(L)
        if pct in prompt_cache:
            continue
        chat = av_tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(pct, INJECTION_CHAR)}],
            tokenize=False, add_generation_prompt=True)
        toks = av_tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (toks, find_inject_pos(toks, inj_id))

    eval_holdout.gt_descs = gt_descs

    # ---- faithfulness scorer (lever #6): always loaded so the faith eval runs
    # even at λ=0 (lets us watch the honest judge without changing the reward).
    faith = FaithScorer(args.faith_model, device)
    faith.precompute(gt_descs, ar_layers)
    eval_holdout.faith = faith
    eval_holdout.corpus_texts = load_corpus_texts(args)
    print(f"  faith encoder={args.faith_model}  λ={args.faith_coef}  "
          f"K={args.faith_distractors}  "
          f"corpus_texts={len(eval_holdout.corpus_texts)}", flush=True)

    rt0, ceil0, faith0 = eval_holdout(av, av_tok, ar_model, value_heads, ar_tok,
                                      acts, id2idx, layer_mu, holdout, ar_layers,
                                      prompt_cache, args, device)
    print(f"[init] holdout round-trip={rt0:.4f}  gt_ceiling={ceil0:.4f}  "
          f"faith(matched/inputtext)={_fmt_faith(faith0)}", flush=True)
    if args.phase == "eval":
        json.dump({"round_trip": rt0, "gt_ceiling": ceil0, "faith": faith0},
                  open(out / "joint_eval_init.json", "w"), indent=1)
        return

    # ---- optimizers ----
    av_params = [p for p in av.parameters() if p.requires_grad]
    av_opt = torch.optim.AdamW(av_params, lr=args.av_lr, weight_decay=0.01)
    if args.train_ar:
        ar_params = ([p for p in ar_model.parameters() if p.requires_grad]
                     + list(value_heads.parameters()))
        ar_opt = torch.optim.AdamW(ar_params, lr=args.ar_lr, weight_decay=0.01)
        mse_scale = math.sqrt(d_model)

    samples = [(L, t) for L in ar_layers for t in train_ids]
    best_rt = rt0

    for epoch in range(args.epochs):
        # ============ (B) AR refinement on GT descriptions ONLY ============
        if args.train_ar:
            ar_model.train()
            ar_losses = []
            for step in range(args.ar_steps_per_epoch):
                L = ar_layers[rng.randint(len(ar_layers))]
                ids_L = [t for t in train_ids if t in gt_descs[L]]
                batch = [ids_L[i] for i in rng.choice(len(ids_L), args.ar_batch, replace=False)]
                descs = [gt_descs[L][t] for t in batch]
                pred = ar_recon(ar_model, value_heads, ar_tok, descs, L, device,
                                args.ar_max_len, no_grad=False)
                tgt = torch.stack([acts[L][id2idx[t]].float() for t in batch]).to(device)
                mu = layer_mu[L]
                pn = torch.nn.functional.normalize(pred - mu, dim=1) * mse_scale
                tn = torch.nn.functional.normalize(tgt - mu, dim=1) * mse_scale
                ar_loss = ((pn - tn) ** 2).mean()  # centered dir-MSE (phi_ar v3 core)
                ar_opt.zero_grad()
                ar_loss.backward()
                gn_ar = torch.nn.utils.clip_grad_norm_(ar_params, 1.0)
                ar_opt.step()
                ar_losses.append(float(ar_loss.detach()))
            ar_model.eval()
            print(f"  ep{epoch+1} AR-refine: {len(ar_losses)} steps "
                  f"loss {ar_losses[0]:.4f}->{ar_losses[-1]:.4f} "
                  f"mean={sum(ar_losses)/len(ar_losses):.4f} "
                  f"grad_norm={float(gn_ar):.4f}", flush=True)

        # ============ (A) AV GRPO with KL legibility anchor ============
        idx = rng.choice(len(samples), min(args.samples_per_epoch, len(samples)),
                         replace=False)
        ep_reward = ep_kl = ep_faith = n = 0
        for si in idx:
            L, t = samples[si]
            ptoks, ipos = prompt_cache[nearest_pct(L)]
            a = acts[L][id2idx[t]].float()

            # --- sample a group of descriptions from the policy ---
            gen_groups, descs = [], []
            av.set_adapter("policy")
            emb_layer = av.get_input_embeddings()
            for _ in range(args.group_size):
                input_ids = torch.tensor([ptoks], dtype=torch.long, device=device)
                emb = emb_layer(input_ids)
                emb[0, ipos, :] = normalize_activation(a.to(device), INJECTION_SCALE).to(emb.dtype)
                with torch.no_grad():
                    o = av.generate(inputs_embeds=emb.to(av.dtype),
                                    attention_mask=torch.ones_like(input_ids),
                                    max_new_tokens=args.max_new_tokens, do_sample=True,
                                    temperature=args.temperature,
                                    pad_token_id=av_tok.eos_token_id,
                                    return_dict_in_generate=True)
                gids = o.sequences[0].tolist()
                while gids and gids[-1] in {av_tok.eos_token_id, av_tok.pad_token_id}:
                    gids.pop()
                stop = av_tok.encode("</explanation>", add_special_tokens=False)
                for i in range(len(gids) - len(stop) + 1):
                    if gids[i:i + len(stop)] == stop:
                        gids = gids[:i]
                        break
                gen_groups.append(gids)
                descs.append(av_tok.decode(o.sequences[0], skip_special_tokens=True)
                             .split("</explanation>")[0].strip())

            # --- reward: contrastive centered round-trip cos (R_rt) ---
            mu = layer_mu[L]
            preds = [ar_recon(ar_model, value_heads, ar_tok, [d], L, device,
                              args.ar_max_len)[0] for d in descs]
            tgt = a.to(device)
            correct = [_cc(p, tgt, mu, args.center_reward) for p in preds]
            if args.contrastive:
                wt = train_ids[(train_ids.index(t) + 1) % len(train_ids)] \
                    if t in train_ids else t
                a_wrong = acts[L][id2idx[wt]].float().to(device)
                wrong = [_cc(p, a_wrong, mu, args.center_reward) for p in preds]
                rewards = [c - w for c, w in zip(correct, wrong)]
            else:
                rewards = list(correct)
            if args.rep_penalty > 0:
                for i, g in enumerate(gen_groups):
                    if len(g) >= 4:
                        tri = [tuple(g[j:j + 3]) for j in range(len(g) - 2)]
                        rewards[i] -= args.rep_penalty * (1 - len(set(tri)) / len(tri))

            r_rt = torch.tensor(rewards, dtype=torch.float32)
            # --- faithfulness reward (R_faith): contrastive identification in
            # all-MiniLM text space against K in-batch GT-desc negatives ---
            r_faith_mean = 0.0
            if args.faith_coef > 0:
                pool = [i for i in train_ids
                        if i != t and i in faith.gt_emb.get(L, {})]
                k = min(args.faith_distractors, len(pool))
                distractors = ([pool[j] for j in
                                rng.choice(len(pool), k, replace=False)]
                               if k > 0 else [])
                r_faith = torch.tensor(
                    faith.faith_reward(descs, L, t, distractors),
                    dtype=torch.float32)
                r_faith_mean = r_faith.mean().item()
                adv = _z(r_rt).mul(1 - args.faith_coef) + _z(r_faith).mul(args.faith_coef)
            else:
                adv = _z(r_rt)
            if adv.std().item() < 1e-6:
                ep_reward += r_rt.mean().item()
                ep_faith += r_faith_mean
                n += 1
                continue
            adv = _z(adv)  # renormalize mix to unit variance (lr-stable across λ)

            # --- policy update with per-token KL to frozen SFT ref ---
            # IMPORTANT: compute ALL reference logprobs first (adapter='ref'),
            # THEN switch to 'policy' and keep it active through the policy
            # forwards AND backward — otherwise set_adapter('ref') would flip the
            # policy LoRA params to requires_grad=False and kill the gradient.
            av.set_adapter("ref")
            ref_lps = []
            with torch.no_grad():
                for g in gen_groups:
                    ref_lps.append(
                        av_logprobs(av, av_tok, ptoks, ipos, a, g, device, "ref",
                                    do_set=False) if g else None)

            av.set_adapter("policy")
            av_opt.zero_grad()
            loss = 0.0
            for g, A, lp_ref in zip(gen_groups, adv, ref_lps):
                if not g:
                    continue
                lp = av_logprobs(av, av_tok, ptoks, ipos, a, g, device, "policy",
                                 do_set=False)
                logr = lp_ref - lp                      # log(π_ref/π)
                kl = torch.exp(logr) - logr - 1.0       # Schulman k3, ≥0
                pol = -(A.to(device) * lp.sum())
                loss = loss + (pol + args.kl_coef * kl.sum()) / (len(idx) * args.group_size)
                ep_kl += kl.sum().item()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(av_params, 1.0)
            av_opt.step()
            ep_reward += r_rt.mean().item()
            ep_faith += r_faith_mean
            n += 1
            if n % 8 == 0:
                print(f"  ep{epoch+1} [{n}/{len(idx)}] mean_reward={ep_reward/n:.4f} "
                      f"mean_faith={ep_faith/n:.4f} "
                      f"mean_kl={ep_kl/max(1,n):.3f} grad_norm={float(gnorm):.4f} "
                      f"loss={float(loss.detach()):.4f}", flush=True)

        rt, ceil, faithep = eval_holdout(av, av_tok, ar_model, value_heads, ar_tok,
                                         acts, id2idx, layer_mu, holdout, ar_layers,
                                         prompt_cache, args, device)
        print(f"[epoch {epoch+1}] mean_reward={ep_reward/max(1,n):.4f} "
              f"mean_faith={ep_faith/max(1,n):.4f} "
              f"mean_kl={ep_kl/max(1,n):.3f} | holdout round-trip={rt:.4f} "
              f"ceiling={ceil:.4f} faith(matched/inputtext)={_fmt_faith(faithep)}",
              flush=True)
        json.dump({"epoch": epoch + 1, "round_trip": rt, "gt_ceiling": ceil,
                   "mean_reward": ep_reward / max(1, n),
                   "mean_faith_reward": ep_faith / max(1, n), "faith": faithep},
                  open(out / f"joint_epoch{epoch+1}.json", "w"), indent=1)

        if rt > best_rt:
            best_rt = rt
            av.save_pretrained(str(out), selected_adapters=["policy"])
            # make the saved policy adapter self-contained for eval: PEFT writes
            # only adapter weights+config, NOT the tokenizer (incl chat_template),
            # so a later --phase eval on this dir would crash at tokenizer load.
            av_tok.save_pretrained(str(out / "policy"))
            torch.save({str(L): value_heads[str(L)].weight.data.cpu() for L in ar_layers},
                       out / "value_heads_joint.pt")
            if args.train_ar:
                # persist refined AR LoRA in load_ar-compatible format so the
                # improved AR can actually be reconstructed for the final eval.
                try:
                    from peft import get_peft_model_state_dict
                except ImportError:
                    from peft.utils import get_peft_model_state_dict
                ar_lora = {k: v.detach().cpu() for k, v in
                           get_peft_model_state_dict(ar_model).items()}
                torch.save({"args": ar_ck_args, "lora": ar_lora},
                           out / "ar_joint_best.pt")
            print(f"    -> saved best (holdout round-trip={best_rt:.4f})", flush=True)

    print(f"\n=== JOINT GRPO DONE | best holdout round-trip={best_rt:.4f} "
          f"(init {rt0:.4f}, ceiling≈{ceil0:.4f}, kitft bar 0.769) ===")


if __name__ == "__main__":
    main()
