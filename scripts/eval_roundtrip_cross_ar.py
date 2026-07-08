#!/usr/bin/env python3
"""Cross-AR validation: does the GRPO AV's edge survive an independent reader?

The GRPO AV was optimized against the universal lora_sl AR. If its round-trip
advantage over the SFT AV is a property of the DESCRIPTIONS, a different AR —
trained in a different era, different format, different template — should
reproduce the ranking. If the advantage vanishes under the independent
reader, it was reward hacking (AV exploiting reward-AR idiosyncrasies).
Final rung of the falsification ladder in DESIGN.md § AR faithfulness audit.

Independent reader: the published single-layer L20 AR v2
(output/nla-qwen25-7b-L20-ar-v2) — frozen truncated-backbone + value_head
design. Its protocol is reimplemented live here from the era-correct
sources (scripts/legacy/train_ar_truncated.py is frozen; per contract we
reimplement, never import):
  - prompt   = nla_meta prompt_templates.ar (NOT hardcoded — the template is
               a frozen interface of the shipped checkpoint)
  - tokens   = add_special_tokens=True, max_length 512 with
               suffix-preserving truncation ("</text> <summary>")
  - position = true last token via per-item seq_lens (right padding)
  - state    = RAW residual after block 20. The trained backbone was
               truncated with model.norm -> Identity, so the equivalent
               full-model read is hidden_states[21] (pre-final-RMSNorm; the
               post-norm hidden_states[-1] would be the classic trap)
  - head     = Linear(d, d, bias=False) from value_head.safetensors

Built-in differential guard: before any verdict, the independent reader
must reproduce the audit's sanity gradient (matched twin_clean description
beats a same-category wrong-text one) on the SAME texts it will judge.
If that fails, the loader is wrong or the reader is noise — the script
refuses to print a verdict rather than report an artifact.

Usage (after eval_roundtrip_universal.py produced its .records.jsonl):
  python3 scripts/eval_roundtrip_cross_ar.py \
    --records output/roundtrip_qwen_universal.records.jsonl \
    --ar-dir output/nla-qwen25-7b-L20-ar-v2 \
    --activations corpus/activations/qwen25-7b_all_layers.pt \
    --layer 20 --out working-docs/cross_ar_verdict.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

from nla_lib import nearest_depth_pct, read_nla_meta  # noqa: E402


def load_truncated_head_ar(ar_dir, hf_name, device):
    """Live reimplementation of the value_head+frozen (stage: sl) reader."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = read_nla_meta(ar_dir)
    if not meta or meta.get("training", {}).get("method") != "value_head+frozen":
        sys.exit(f"{ar_dir} is not a value_head+frozen checkpoint "
                 f"(meta: {meta and meta.get('training', {}).get('method')})")
    template = meta["prompt_templates"]["ar"]
    layer = meta["extraction_layer_index"]
    n_backbone = meta["training"]["backbone_layers"]
    assert n_backbone == layer + 1, (n_backbone, layer)

    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    head_w = load_file(str(Path(ar_dir) / "value_head.safetensors"))
    (key, w), = head_w.items()
    head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False,
                           dtype=torch.float32)
    head.weight = torch.nn.Parameter(w.float(), requires_grad=False)
    head = head.to(device).eval()
    print(f"  independent AR: {Path(ar_dir).name} (layer {layer}, "
          f"head {key} {tuple(w.shape)}, template from nla_meta)")

    suffix_tokens = tok.encode("</text> <summary>", add_special_tokens=False)

    @torch.no_grad()
    def reconstruct(descriptions, batch=16, max_length=512):
        preds = []
        for s in range(0, len(descriptions), batch):
            chunk = descriptions[s:s + batch]
            rows = []
            for d in chunk:
                t = tok.encode(template.replace("{explanation}", d),
                               add_special_tokens=True)
                if len(t) > max_length:
                    t = t[:max_length - len(suffix_tokens)] + suffix_tokens
                rows.append(t)
            lens = [len(r) for r in rows]
            width = max(lens)
            pad = tok.pad_token_id or tok.eos_token_id
            input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
            attn = torch.zeros((len(rows), width), dtype=torch.long)
            for i, r in enumerate(rows):          # RIGHT padding, per training
                input_ids[i, :len(r)] = torch.tensor(r)
                attn[i, :len(r)] = 1
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attn.to(device),
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer + 1]      # RAW residual after block L
            last = torch.stack([h[i, lens[i] - 1] for i in range(len(rows))])
            preds.append(head(last.float()).cpu())
        return torch.cat(preds), layer

    return reconstruct, layer


def centered(v, mean):
    v = v.float() - mean
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help=".records.jsonl from eval_roundtrip_universal.py")
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--hf-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--generated-dir", default="corpus/generated")
    ap.add_argument("--sanity-n", type=int, default=60)
    ap.add_argument("--out", default="working-docs/cross_ar_verdict.json")
    args = ap.parse_args()
    device = "cuda"
    L = args.layer

    rows = [json.loads(l) for l in open(args.records)]
    rows = [r for r in rows if r["layer"] == L]
    by_tag = defaultdict(dict)
    for r in rows:
        by_tag[r["tag"]][r["text_id"]] = r
    tags = sorted(by_tag)
    if len(tags) < 2:
        sys.exit(f"need >=2 tags (test+baseline) at layer {L}, got {tags}")
    common = sorted(set.intersection(*(set(by_tag[t]) for t in tags)))
    print(f"{len(rows)} records at L{L}, tags {tags}, {len(common)} common texts")

    blob = torch.load(args.activations, map_location="cpu")
    id_row = {t: i for i, t in enumerate(blob["ids"])}
    mean = blob["activations"][L].float().mean(0)
    targets = {t: blob["activations"][L][id_row[t]] for t in common
               if t in id_row}
    missing = [t for t in common if t not in id_row]
    if missing:
        sys.exit(f"{len(missing)} record texts missing from activations "
                 f"(e.g. {missing[:3]}) — wrong activations file?")

    reconstruct, ar_layer = load_truncated_head_ar(args.ar_dir, args.hf_name,
                                                   device)
    if ar_layer != L:
        sys.exit(f"independent AR reads layer {ar_layer}, records are L{L}")

    # ---- differential guard: sanity gradient on the same texts ------------
    pct = nearest_depth_pct(L, blob["n_layers"])
    desc_path = Path(args.generated_dir) / f"descriptions_L{pct}pct_twin_clean.json"
    dmap = {it["id"]: it["description"]
            for it in json.loads(desc_path.read_text()) if it.get("description")}
    sane = [t for t in common if t in dmap][:args.sanity_n]
    if len(sane) < 20:
        sys.exit(f"only {len(sane)} sanity texts with twin_clean descs — "
                 f"check {desc_path}")
    import random
    rng = random.Random(0)
    wrong = [dmap[rng.choice([x for x in sane if x != t])] for t in sane]
    pred_a, _ = reconstruct([dmap[t] for t in sane])
    pred_d, _ = reconstruct(wrong)
    tgt = torch.stack([targets[t] for t in sane])
    ca = (centered(pred_a, mean) * centered(tgt, mean)).sum(-1)
    cd = (centered(pred_d, mean) * centered(tgt, mean)).sum(-1)
    p_ad = (ca > cd).float().mean().item()
    print(f"GUARD  matched {ca.mean():.3f} vs wrong-text {cd.mean():.3f}, "
          f"P(A>D)={p_ad:.2f}  (n={len(sane)})")
    if not (ca.mean() > cd.mean() + 0.05 and p_ad > 0.7):
        sys.exit("GUARD FAILED: the independent reader does not reproduce "
                 "the basic content gradient — loader bug or useless reader. "
                 "REFUSING to emit a cross-AR verdict.")

    # ---- the verdict -------------------------------------------------------
    result = {"layer": L, "n_texts": len(common), "tags": {},
              "guard": {"matched": ca.mean().item(), "wrong": cd.mean().item(),
                        "p_ad": p_ad}}
    cos_by_tag = {}
    for tag in tags:
        descs = [by_tag[tag][t]["desc"] for t in common]
        pred, _ = reconstruct(descs)
        tgt = torch.stack([targets[t] for t in common])
        cos = (centered(pred, mean) * centered(tgt, mean)).sum(-1)
        cos_by_tag[tag] = cos
        orig = torch.tensor([by_tag[tag][t]["cos"] for t in common])
        corr = torch.corrcoef(torch.stack([cos, orig]))[0, 1].item()
        result["tags"][tag] = {
            "indep_centered_cos": cos.mean().item(),
            "orig_ar_centered_cos": orig.mean().item(),
            "per_text_corr_indep_vs_orig": corr}
        print(f"[{tag:9s}] independent AR: {cos.mean():.4f}   "
              f"original AR: {orig.mean():.4f}   corr: {corr:.3f}")

    if len(tags) == 2:
        t_test, t_base = tags if "test" in tags[0] else tags[::-1]
        d = cos_by_tag[t_test] - cos_by_tag[t_base]
        p_win = (d > 0).float().mean().item()
        result["verdict"] = {
            "edge_under_independent_ar": d.mean().item(),
            "p_test_beats_baseline": p_win}
        print(f"\nVERDICT: {t_test} - {t_base} edge under independent AR = "
              f"{d.mean():+.4f}, wins {p_win:.0%} of texts")
        print("  edge holds  -> the GRPO gain is a property of the "
              "descriptions (not reward hacking)")
        print("  edge ~ 0    -> the gain was specific to the reward AR")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
