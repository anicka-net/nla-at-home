"""Matched round-trip eval: Phi-4 universal AV (twinclean) -> Phi-4 universal AR (stage2 v2mid).

The definitive pre-GRPO faithfulness number: take ground-truth activations on a
DOUBLE holdout (texts unseen by BOTH the AV and the AR during training), have
the AV verbalize each activation (injection per nla_meta.yaml), feed the
generated description to the AR, and measure mean-subtracted centered cosine
between the reconstruction and the ground truth, per layer. Benchmark bar:
kitft/nla-qwen2.5-7b round-trip = 0.769 (measured the same way).

Also reports the AR-only ceiling on the same holdout (ground-truth twin_clean
descriptions through the AR instead of AV generations), so the round-trip drop
decomposes into "AR reconstruction limit" vs "AV verbalization loss".

Splits are RECONSTRUCTED, not guessed, and verified:
  AV val = np.random.RandomState(42).choice over the sorted text-id universe of
           the AV example set (10%), exactly as train_universal_av.py does.
           Verified by asserting the reproduced example counts match the
           n_train/n_val recorded in the adapter's nla_meta.yaml.
  AR val = alphabetical tail (last 10%) of the first --n-texts ids common to
           the activation file and all 13 twin_clean description files,
           exactly as phi_ar_stage2.py does. Verified against the n_texts and
           val count printed in the training log.
  holdout = AV val ∩ AR val.

Phases (run separately; each loads only one 14B model):
  --phase split   reproduce + verify splits, write the holdout JSON (CPU only)
  --phase av      generate AV descriptions for holdout x AR layers (GPU)
  --phase ar      reconstruct from AV generations AND from ground-truth
                  descriptions; write final metrics (GPU)

Usage:
  python3 scripts/eval_roundtrip_phi4.py --phase split
  python3 scripts/eval_roundtrip_phi4.py --phase av
  python3 scripts/eval_roundtrip_phi4.py --phase ar

GPU phases written for cuda bf16; untested until first run (per repo policy).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_universal_av import (  # noqa: E402
    INJECTION_SCALE, make_prompt, find_inject_pos, normalize_activation,
)

BASE = "microsoft/phi-4"
INJECTION_CHAR = "★"
AR_PROMPT = "Summary of the following text: <text>{e}</text> <summary>"
AR_LAYERS = [13, 16, 19, 22, 25, 28, 32, 36, 38]
DEPTH_PCTS = [4, 10, 17, 25, 32, 40, 47, 55, 63, 71, 80, 90, 96]
N_LAYERS = 40

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["split", "av", "ar"])
ap.add_argument("--av-adapter", default=str(REPO / "output/nla-phi4-universal-av-twinclean"))
ap.add_argument("--av-activations", default=str(REPO / "corpus/activations/phi4_all_layers.pt"))
ap.add_argument("--av-desc-dir", default=str(REPO / "corpus/generated"))
ap.add_argument("--ar-root", default=os.environ.get("AR_ROOT", str(Path.home() / "phi4_ar")))
ap.add_argument("--ar-tag", default="stage2_v2mid")
ap.add_argument("--n-texts", type=int, default=6000, help="AR run's --n-texts")
ap.add_argument("--out-dir", default=str(REPO / "output/roundtrip_v2mid"))
ap.add_argument("--max-new-tokens", type=int, default=200)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT_F = OUT / "holdout.json"
GEN_F = OUT / "av_generations.json"
RESULT_F = OUT / "roundtrip_result.json"


def nearest_pct(layer):
    d = layer * 100 / N_LAYERS
    return min(DEPTH_PCTS, key=lambda p: abs(p - d))


def load_av_desc(pct):
    f = Path(args.av_desc_dir) / f"descriptions_L{pct}pct_twin_clean.json"
    return {x["id"]: x["description"] for x in json.load(open(f)) if x.get("description")}


def reproduce_splits():
    """Reproduce AV and AR val splits; verify against recorded counts."""
    import yaml
    meta = yaml.safe_load(open(Path(args.av_adapter) / "nla_meta.yaml"))
    want_train = meta["training"]["n_train"]
    want_val = meta["training"]["n_val"]

    # --- AV example universe (mirror build_examples over the AV activation file)
    act = torch.load(args.av_activations, weights_only=True, map_location="cpu")
    layer_acts = act["activations"]
    ids = act["ids"]
    n_layers = act.get("n_layers", N_LAYERS)
    desc_by_pct = {p: load_av_desc(p) for p in DEPTH_PCTS}
    pairs = []  # (text_id, layer)
    avail_layers = sorted(layer_acts.keys()) if isinstance(layer_acts, dict) else range(n_layers)
    for L in avail_layers:
        pct = nearest_pct(int(L))
        dm = desc_by_pct.get(pct)
        if not dm:
            continue
        for t in ids:
            if t in dm:
                pairs.append((t, int(L)))
    all_text_ids = sorted(set(t for t, _ in pairs))
    n_val_texts = max(1, int(len(all_text_ids) * 0.1))
    rng = np.random.RandomState(42)
    av_val_ids = set(rng.choice(all_text_ids, n_val_texts, replace=False))
    n_val = sum(1 for t, _ in pairs if t in av_val_ids)
    n_train = len(pairs) - n_val
    print(f"[split] AV examples reproduced: train={n_train} val={n_val} "
          f"(meta: {want_train}/{want_val})")
    assert (n_train, n_val) == (want_train, want_val), (
        "AV split reproduction MISMATCH — wrong activations/descs/val_split; "
        "do not trust the holdout")

    # --- AR common-id universe (mirror phi_ar_stage2.py)
    D = torch.load(f"{args.ar_root}/phi4_13depths.pt", weights_only=True,
                   map_location="cpu")
    ar_ids = D["ids"]
    common = set(ar_ids)
    for L in AR_LAYERS:
        p = nearest_pct(L)
        f = f"{args.ar_root}/descs/descriptions_L{p}pct_twin_clean.json"
        dm = {x["id"] for x in json.load(open(f)) if x.get("description")}
        common &= dm
    common = sorted(common)[: args.n_texts]
    ntr = int(len(common) * 0.9)
    ar_val = common[ntr:]
    print(f"[split] AR common={len(common)} train={ntr} val={len(ar_val)}")

    holdout = sorted(set(ar_val) & av_val_ids)
    print(f"[split] DOUBLE holdout (AR val ∩ AV val): {len(holdout)} texts")
    if len(holdout) < 20:
        print("[split] WARNING: holdout < 20 texts — variance will be high")
    json.dump({"holdout": holdout, "ar_val": ar_val,
               "n_av_val_texts": len(av_val_ids)}, open(HOLDOUT_F, "w"), indent=1)
    print(f"[split] wrote {HOLDOUT_F}")


def phase_av():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    holdout = json.load(open(HOLDOUT_F))["holdout"]
    D = torch.load(f"{args.ar_root}/phi4_13depths.pt", weights_only=True,
                   map_location="cpu")
    acts = D["activations"]
    id2idx = {t: i for i, t in enumerate(D["ids"])}

    tok = AutoTokenizer.from_pretrained(args.av_adapter)
    print(f"loading {BASE} + AV adapter...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, args.av_adapter).to(args.device).eval()

    inj_id = tok.encode(INJECTION_CHAR, add_special_tokens=False)
    assert len(inj_id) == 1
    inj_id = inj_id[0]
    prompt_cache = {}
    for L in AR_LAYERS:
        pct = nearest_pct(L)
        if pct in prompt_cache:
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(pct, INJECTION_CHAR)}],
            tokenize=False, add_generation_prompt=True)
        toks = tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (toks, find_inject_pos(toks, inj_id))

    rows = []
    if GEN_F.exists():
        rows = json.load(open(GEN_F))["rows"]
    done = {(r["text_id"], r["layer"]) for r in rows}
    k, total = len(rows), len(holdout) * len(AR_LAYERS)
    embed = model.get_input_embeddings()
    for L in AR_LAYERS:
        pct = nearest_pct(L)
        ptoks, ipos = prompt_cache[pct]
        for t in holdout:
            if (t, L) in done:
                continue
            a = acts[L][id2idx[t]]
            input_ids = torch.tensor([ptoks], dtype=torch.long, device=args.device)
            with torch.no_grad():
                emb = embed(input_ids)
                emb[0, ipos, :] = normalize_activation(
                    a.to(args.device), INJECTION_SCALE).to(emb.dtype)
                out = model.generate(
                    inputs_embeds=emb, attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
            txt = tok.decode(out.sequences[0], skip_special_tokens=True)
            txt = txt.split("</explanation>")[0].strip()
            rows.append({"text_id": t, "layer": L, "depth_pct": pct,
                         "description": txt})
            k += 1
            if k % 10 == 0:
                print(f"    [av] {k}/{total}", flush=True)
                json.dump({"rows": rows}, open(GEN_F, "w"), indent=1)
    json.dump({"rows": rows}, open(GEN_F, "w"), indent=1)
    print(f"[av] wrote {GEN_F} ({len(rows)} generations)", flush=True)


def phase_ar():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        from peft.utils import set_peft_model_state_dict

    holdout = json.load(open(HOLDOUT_F))["holdout"]
    gen = json.load(open(GEN_F))["rows"]
    gen_map = {(r["text_id"], r["layer"]): r["description"] for r in gen}
    D = torch.load(f"{args.ar_root}/phi4_13depths.pt", weights_only=True,
                   map_location="cpu")
    acts = D["activations"]
    id2idx = {t: i for i, t in enumerate(D["ids"])}
    gt_desc = {}
    for L in AR_LAYERS:
        p = nearest_pct(L)
        f = f"{args.ar_root}/descs/descriptions_L{p}pct_twin_clean.json"
        gt_desc[L] = {x["id"]: x["description"] for x in json.load(open(f))
                      if x.get("description")}

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loading {BASE} + AR LoRA ({args.ar_tag})...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=False)
    best = torch.load(f"{args.ar_root}/{args.ar_tag}_best.pt", map_location="cpu",
                      weights_only=False)
    lora_args = best["args"]
    lora = LoraConfig(r=lora_args["lora_r"], lora_alpha=2 * lora_args["lora_r"],
                      lora_dropout=0.0, bias="none",
                      target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(backbone, lora)
    set_peft_model_state_dict(model, best["lora"])
    model = model.to(args.device).eval()
    vh_w = torch.load(f"{args.ar_root}/{args.ar_tag}_value_heads.pt",
                      map_location="cpu", weights_only=True)
    d_model = acts[AR_LAYERS[0]].shape[1]
    value_heads = {L: torch.nn.Linear(d_model, d_model, bias=False,
                                      dtype=torch.float32) for L in AR_LAYERS}
    for L in AR_LAYERS:
        value_heads[L].weight.data = vh_w[str(L)]
        value_heads[L] = value_heads[L].to(args.device)
    print(f"[ar] best mean from training: {best.get('best'):.3f}", flush=True)

    @torch.no_grad()
    def recon(L, descs):
        prompts = [AR_PROMPT.format(e=e) for e in descs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(args.device)
        h = model(**enc, output_hidden_states=True).hidden_states[L + 1][:, -1, :].float()
        return value_heads[L](h)

    def centered_cos(P, T):
        Pc = P - P.mean(0, keepdim=True)
        Tc = T - T.mean(0, keepdim=True)
        return torch.nn.functional.cosine_similarity(Pc, Tc, dim=1).mean().item()

    result = {"holdout_n": len(holdout), "arms": {}}
    for arm in ("roundtrip", "gt_ceiling"):
        per_layer = {}
        for L in AR_LAYERS:
            ids_L, descs = [], []
            for t in holdout:
                e = gen_map.get((t, L)) if arm == "roundtrip" else gt_desc[L].get(t)
                if e:
                    ids_L.append(t)
                    descs.append(e)
            P = []
            for i in range(0, len(descs), 4):
                P.append(recon(L, descs[i:i + 4]).cpu().double())
            P = torch.cat(P)
            T = torch.stack([acts[L][id2idx[t]].double() for t in ids_L])
            per_layer[f"L{L}({nearest_pct(L)}%)"] = round(centered_cos(P, T), 3)
            print(f"    [{arm}] L{L}: {per_layer[f'L{L}({nearest_pct(L)}%)']} "
                  f"(n={len(ids_L)})", flush=True)
        vals = list(per_layer.values())
        result["arms"][arm] = {"per_layer": per_layer,
                               "mean": round(sum(vals) / len(vals), 3)}
    result["kitft_bar"] = 0.769
    json.dump(result, open(RESULT_F, "w"), indent=1)
    print(json.dumps({k: v if k != "arms" else
                      {a: r["mean"] for a, r in v.items()}
                      for k, v in result.items()}, indent=1), flush=True)
    print(f"[ar] wrote {RESULT_F}", flush=True)


if args.phase == "split":
    reproduce_splits()
elif args.phase == "av":
    phase_av()
else:
    phase_ar()
