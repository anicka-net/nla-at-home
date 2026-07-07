#!/usr/bin/env python3
"""Audit: what does the AR actually read from a description?

Motivation (Anička, 2026-07-07): the AR is trained on synthetic descriptions
(GPT-4o/Sonnet guesses written from TEXTS, not from activations). Systematic
describer conventions could become retrieval keys — the AR might identify
"which text was this" from convention + topic and emit its centroid, rather
than read the description's content. GRPO reward would then enforce
describer-convention reproduction, not activation faithfulness.

Design: for each sampled (text, layer) pair, score SIX description variants
against the TRUE activation, with the exact reward scoring path used by GRPO
(imported from train_universal_grpo_hard — same load_ar, same ar_reconstruct,
same centered_cosine, same AR_TEMPLATE):

  A  matched      — the twin_clean description the AR family was trained on
  B  alt-describer— tokenpred_gpt4o_clean description of the SAME text+depth
                    (same content, different describer conventions)
  C  shuffled     — A with word order destroyed (keyword bag kept)
  D  same-cat     — twin_clean description of a DIFFERENT text, same category
                    and depth (the plausible-confabulation / topic-centroid probe)
  E  cross-cat    — description of a different-category text, same depth
  F  generic      — one constant content-free string

Readouts (all reported in BOTH raw and centered cosine):
  * B vs A  — style sensitivity. B ≈ A ⇒ AR reads content; B << A ⇒ it keys
              on trained conventions (the "learned on hallucinated
              scaffolding" fear).
  * A vs D  — within-topic discrimination. A >> D ⇒ reads specifics;
              A ≈ D ⇒ topic-centroid retrieval (reward blind past topic).
              Also reported as discrimination rate P(A > D) per layer.
  * C vs A  — word order vs keyword bag.
  * E, F    — sanity floor; F's raw column shows the mean-baseline freebie.

Honest-use rules: run on texts the AR did NOT train on (--ids-file with its
val split); refuses to run without one unless --allow-any-texts (loud).

Usage (any CUDA box with the repo + data synced):
  python3 scripts/audit_ar_faithfulness.py \
    --ar-checkpoint output/nla-qwen25-7b-universal-ar \
    --activations corpus/activations/qwen25-7b_all_layers.pt \
    --ids-file output/nla-qwen25-7b-universal-ar/val_text_ids.json \
    --layers 4,9,14,19,24,27 --n-per-layer 150 \
    --out audit_ar_faithfulness.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

# Single source of truth: constants from nla_lib, scoring path from the GRPO
# script itself (byte-identical reward — see differential-test rule).
from nla_lib import nearest_depth_pct, INJECTABLE_MODELS_HF, get_model  # noqa: E402
from train_universal_grpo_hard import (  # noqa: E402
    AR_TEMPLATE, centered_cosine, load_ar, ar_reconstruct,
)

GENERIC_DESC = (
    "- Routine processing of the input text\n"
    "- Mid-network representation of the prompt\n"
    "- Response preparation underway")


def load_desc_map(gen_dir, pct, suffix):
    p = gen_dir / f"descriptions_L{pct}pct{suffix}.json"
    if not p.exists():
        return {}
    items = json.loads(p.read_text())
    return {it["id"]: it for it in items if it.get("description")}


def shuffle_words(text, rng):
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def cat_of(text_id):
    """Category from the id itself (desc files carry only {id, description}).
    'A01_code_000' -> 'A01_code'; 'WC_02874' -> 'WC' (one mixed web pool —
    same-cat there means same-pool, which only makes the D probe harder)."""
    return text_id.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-checkpoint", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--generated-dir", default="corpus/generated")
    ap.add_argument("--trained-suffix", default="_twin_clean")
    ap.add_argument("--alt-suffix", default="_tokenpred_gpt4o_clean")
    ap.add_argument("--ids-file", default=None,
                    help="JSON list of text ids the AR did NOT train on")
    ap.add_argument("--allow-any-texts", action="store_true",
                    help="LOUD override: audit on unrestricted texts "
                         "(scores will be optimistic on AR train texts)")
    ap.add_argument("--layers", default="4,9,14,19,24,27")
    ap.add_argument("--n-per-layer", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="audit_ar_faithfulness.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    gen_dir = Path(args.generated_dir)

    if not args.ids_file and not args.allow_any_texts:
        sys.exit("REFUSING: no --ids-file (AR val split). The audit must run "
                 "on texts the AR never trained on, or its numbers flatter "
                 "the AR. Override consciously with --allow-any-texts.")
    allowed = None
    if args.ids_file:
        allowed = set(json.loads(Path(args.ids_file).read_text()))
        print(f"restricting to {len(allowed)} held-out text ids")

    print("loading activations...")
    blob = torch.load(args.activations, map_location="cpu")
    acts, ids = blob["activations"], blob["ids"]
    if not isinstance(acts, dict):
        sys.exit(f"expected all_layers dict format, got {type(acts)}")
    n_layers = blob["n_layers"]
    id_row = {t: i for i, t in enumerate(ids)}
    # blob["model"] is the full HF id in all_layers files; find the spec for
    # trust_remote_code, but never guess a different model than the file says
    hf_name = blob["model"]
    short_key = next((k for k, v in INJECTABLE_MODELS_HF.items()
                      if v == hf_name), None)
    trust = get_model(short_key).trust_remote_code if short_key else False
    print(f"model {hf_name}, {n_layers} layers, "
          f"{len(ids)} texts, d={blob['d_model']}")

    layers = [int(x) for x in args.layers.split(",")]
    bad = [L for L in layers if not 0 <= L < n_layers]
    if bad:
        sys.exit(f"layers {bad} out of range for {n_layers}")

    # per-layer means over the WHOLE corpus (same convention as GRPO reward)
    means = {L: acts[L].float().mean(0).to(device) for L in layers}

    ar_model, value_heads, ar_tok = load_ar(
        args.ar_checkpoint, hf_name, device, trust_remote=trust)

    # ---- build jobs -------------------------------------------------------
    jobs = []          # (layer, row, condition, description)
    per_layer_pairs = {}
    for L in layers:
        pct = nearest_depth_pct(L, n_layers)
        trained = load_desc_map(gen_dir, pct, args.trained_suffix)
        alt = load_desc_map(gen_dir, pct, args.alt_suffix)
        usable = [t for t in trained
                  if t in alt and t in id_row
                  and (allowed is None or t in allowed)]
        rng.shuffle(usable)
        usable = usable[:args.n_per_layer]
        if len(usable) < 20:
            print(f"  L{L} ({pct}%): only {len(usable)} usable texts — skipping")
            continue
        by_cat = defaultdict(list)
        for t in trained:
            by_cat[cat_of(t)].append(t)
        per_layer_pairs[L] = len(usable)
        for tid in usable:
            row = id_row[tid]
            cat = cat_of(tid)
            desc_a = trained[tid]["description"]
            desc_b = alt[tid]["description"]
            desc_c = shuffle_words(desc_a, rng)
            same_cat = [x for x in by_cat[cat] if x != tid]
            cross = [x for c, xs in by_cat.items() if c != cat for x in xs]
            desc_d = trained[rng.choice(same_cat)]["description"] if same_cat else None
            desc_e = trained[rng.choice(cross)]["description"] if cross else None
            for cond, d in [("A_matched", desc_a), ("B_alt_style", desc_b),
                            ("C_shuffled", desc_c), ("D_same_cat", desc_d),
                            ("E_cross_cat", desc_e), ("F_generic", GENERIC_DESC)]:
                if d is not None:
                    jobs.append((L, row, cond, d))
        print(f"  L{L} ({pct}%): {len(usable)} texts x 6 conditions")

    if not jobs:
        sys.exit("no jobs built — check desc files / ids-file overlap")
    print(f"{len(jobs)} scoring jobs, batch {args.batch_size}")

    # ---- score ------------------------------------------------------------
    raw = defaultdict(lambda: defaultdict(list))       # [L][cond] -> cosines
    cen = defaultdict(lambda: defaultdict(list))
    per_text = defaultdict(dict)                       # (L, row) -> cond -> centered
    for i in range(0, len(jobs), args.batch_size):
        chunk = jobs[i:i + args.batch_size]
        descs = [j[3] for j in chunk]
        layers_needed = sorted({j[0] for j in chunk})
        recons = ar_reconstruct(ar_model, value_heads, ar_tok, descs,
                                layers_needed, device)
        for k, (L, row, cond, _) in enumerate(chunk):
            pred = recons[L][k]
            target = acts[L][row].float().to(device)
            r = torch.nn.functional.cosine_similarity(
                pred.unsqueeze(0), target.unsqueeze(0)).item()
            c = centered_cosine(pred, target, means[L]).item()
            raw[L][cond].append(r)
            cen[L][cond].append(c)
            per_text[(L, row)][cond] = c
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i}/{len(jobs)}", flush=True)

    # ---- report -----------------------------------------------------------
    conds = ["A_matched", "B_alt_style", "C_shuffled",
             "D_same_cat", "E_cross_cat", "F_generic"]
    out = {"args": vars(args), "n_layers": n_layers, "per_layer": {}}
    print("\n=== centered cosine (the reward's metric) | raw in parens ===")
    hdr = "layer " + "".join(f"{c[:9]:>12s}" for c in conds) + "   P(A>D)  P(A>B)"
    print(hdr)
    for L in layers:
        if L not in cen or not cen[L].get("A_matched"):
            continue
        cells, layer_rec = [], {}
        for c in conds:
            cs, rs = cen[L].get(c, []), raw[L].get(c, [])
            m_c = sum(cs) / len(cs) if cs else float("nan")
            m_r = sum(rs) / len(rs) if rs else float("nan")
            layer_rec[c] = {"centered": m_c, "raw": m_r, "n": len(cs)}
            cells.append(f"{m_c:6.3f}({m_r:4.2f})")
        ad = [(v["A_matched"], v["D_same_cat"])
              for (l2, _), v in per_text.items() if l2 == L
              and "A_matched" in v and "D_same_cat" in v]
        ab = [(v["A_matched"], v["B_alt_style"])
              for (l2, _), v in per_text.items() if l2 == L
              and "A_matched" in v and "B_alt_style" in v]
        p_ad = sum(a > d for a, d in ad) / len(ad) if ad else float("nan")
        p_ab = sum(a > b for a, b in ab) / len(ab) if ab else float("nan")
        layer_rec["P_A_gt_D"], layer_rec["P_A_gt_B"] = p_ad, p_ab
        out["per_layer"][L] = layer_rec
        print(f"L{L:4d} " + "".join(f"{c:>12s}" for c in cells)
              + f"   {p_ad:5.2f}   {p_ab:5.2f}")

    print("\nHow to read this:")
    print("  B far below A          -> AR keys on trained describer conventions")
    print("  A ~ D (P(A>D) ~ 0.5)   -> topic-centroid retrieval; reward blind past topic")
    print("  F raw high, centered ~0 -> the mean-baseline freebie, quantified")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
