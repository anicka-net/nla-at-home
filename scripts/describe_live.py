#!/usr/bin/env python3
"""Interactive NLA: type a text, see what the AV says about its activations.

Loads ONE Phi-4 backbone serving two roles: extraction (adapter disabled —
identical to extract_activations.py protocol: chat template + generation
prompt, last-token hook) and description (AV LoRA adapter on, injection
prompt/scale identical to training via train_universal_grpo_hard helpers).

Usage:
  python3 scripts/describe_live.py \
      --av-adapter output/nla-phi4-universal-av-grpo-hard \
      [--layers 13,22,28,36] [--max-new-tokens 150]
Then type a text and Enter; empty line or Ctrl-D quits.
Compare adapters by pointing --av-adapter at ...-twinclean (SFT baseline).

Oracle-guided reranking (deploys the +0.133 best-of-N decoding fix):
  1. save a compass (CPU):
     python3 scripts/probe_activation_faithfulness.py --save-compass \
         output/av_oracle_compass.pt --acts ~/phi4_ar/phi4_13depths.pt \
         --holdout output/roundtrip_v2corpus/holdout.json --compass-all-ids \
         --layers 4,10,16,19,25,32,38
  2. run with reranking on the SAME layers the compass covers:
     python3 scripts/describe_live.py --av-adapter ... \
         --layers 4,10,16,19,25,32,38 --rerank-best-of 16 \
         --compass output/av_oracle_compass.pt
  Layers absent from the compass fall back to greedy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_universal_grpo_hard import (  # noqa: E402
    INJECTION_CHARS, INJECTION_SCALE, MODELS, make_av_prompt,
    nearest_depth_pct, normalize_activation, strip_generated_row)
import av_policy  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def get_blocks(model):
    base = model
    while not hasattr(base, "layers"):
        base = base.model
    return base.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-adapter", required=True)
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--layers", default="4,10,16,25,32,38")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerank-best-of", type=int, default=1, metavar="N",
                    help="oracle-guided reranking: sample N descriptions and keep "
                         "the one closest to the compass prediction W*a. Requires "
                         "--compass. N=1 (default) = plain greedy.")
    ap.add_argument("--compass", default=None, metavar="PATH",
                    help="oracle compass artifact from "
                         "probe_activation_faithfulness.py --save-compass. Layers "
                         "absent from it fall back to greedy.")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--policy", action="store_true",
                    help="confidence-gated policy: if the best reranked sample's "
                         "compass agreement < --tau, emit an honest HEDGE instead "
                         "of a confident (possibly confabulated) description. "
                         "Implies reranking; requires --compass.")
    ap.add_argument("--tau", type=float, default=0.30,
                    help="confidence threshold for --policy (compass cosine).")
    ap.add_argument("--generic-centroid", default=None, metavar="PATH",
                    help="generic_centroid artifact (av_policy.py --save-centroid); "
                         "enables the genericness penalty under --policy.")
    ap.add_argument("--gen-penalty", type=float, default=0.0,
                    help="genericness-penalty weight for --policy (needs "
                         "--generic-centroid). 0 = off.")
    ap.add_argument("--faith-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    base_name = MODELS[args.model]
    inj_char = INJECTION_CHARS[args.model]

    rerank = args.rerank_best_of > 1 or args.policy
    compass = enc = gen_centroid = None
    rerank_layers = set()
    if args.policy and args.rerank_best_of <= 1:
        ap.error("--policy needs a sample pool; pass --rerank-best-of N (N>1)")
    if rerank:
        if not args.compass:
            ap.error("--rerank-best-of > 1 / --policy requires --compass")
        compass = torch.load(args.compass, weights_only=False, map_location="cpu")
        rerank_layers = set(compass["layers"])
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(args.faith_model, device="cpu")
        miss = [L for L in layers if L not in rerank_layers]
        mode = f"policy(tau={args.tau})" if args.policy else "rerank"
        print(f"[{mode}] best-of-{args.rerank_best_of} via compass {args.compass} "
              f"(layers {sorted(rerank_layers & set(layers))}); "
              f"greedy fallback for {miss}", flush=True)
        if args.policy and args.generic_centroid and args.gen_penalty:
            gc = torch.load(args.generic_centroid, weights_only=False,
                            map_location="cpu")
            gen_centroid = np.asarray(gc["centroid"], dtype=np.float64)
            print(f"[policy] genericness penalty {args.gen_penalty} via "
                  f"{args.generic_centroid}", flush=True)

    print(f"[load] {base_name} + adapter {args.av_adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16).to(device).eval()
    model = PeftModel.from_pretrained(model, args.av_adapter).eval()
    n_layers = model.config.num_hidden_layers
    blocks = get_blocks(model)
    embed = model.get_input_embeddings()

    inject_id = tok.encode(inj_char, add_special_tokens=False)[0]
    eos_ids = {tok.eos_token_id, tok.pad_token_id}
    stop_ids = tok.encode("</explanation>", add_special_tokens=False)

    prompt_cache = {}
    for L in layers:
        pct = nearest_depth_pct(L, n_layers)
        if pct in prompt_cache:
            continue
        content = make_av_prompt(pct, inj_char)
        chat = tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True)
        tokens = tok.encode(chat, add_special_tokens=False)
        prompt_cache[pct] = (tokens, tokens.index(inject_id))

    def extract(text):
        chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        seq_len = inputs["attention_mask"].sum() - 1
        grabbed = {}

        def make_hook(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                grabbed[L] = h[0, seq_len].detach().float()
            return hook

        handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
        try:
            with torch.no_grad(), model.disable_adapter():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    def clip(s):
        """Mirror the batch evaluator: cut at the explanation close tag so the
        reranker scores (and the demo shows) only the description, not the
        post-tag bleed (e.g. '</explanation>assistant...')."""
        for marker in ("</explanation>", "<|system|>", "<|user|>", "<|end|>"):
            s = s.split(marker)[0]
        return s.strip()

    def describe(act, L):
        pct = nearest_depth_pct(L, n_layers)
        tokens, inject_pos = prompt_cache[pct]
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        emb = embed(ids)
        emb[0, inject_pos, :] = normalize_activation(
            act.to(device), INJECTION_SCALE).to(emb.dtype)
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=emb.to(model.dtype),
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True)
        gen = strip_generated_row(out.sequences[0], tokens, eos_ids, stop_ids)
        return clip(tok.decode(gen, skip_special_tokens=True))

    def compass_target(act, L):
        """Predicted MiniLM text-embedding for this activation: l2norm((a-mu)@W)."""
        a = act.detach().float().cpu().numpy()
        mu = compass["mu"][L].numpy()
        W = compass["W"][L].numpy()
        t = (a - mu) @ W
        n = np.linalg.norm(t)
        return t / n if n else t

    def describe_rerank(act, L, n):
        """Sample n descriptions and keep the one whose MiniLM embedding is
        closest to the compass prediction W*a (oracle-guided, no GT needed)."""
        pct = nearest_depth_pct(L, n_layers)
        tokens, inject_pos = prompt_cache[pct]
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        emb = embed(ids)
        emb[0, inject_pos, :] = normalize_activation(
            act.to(device), INJECTION_SCALE).to(emb.dtype)
        embN = emb.expand(n, -1, -1).contiguous()
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embN.to(model.dtype),
                attention_mask=torch.ones(n, embN.shape[1], dtype=torch.long,
                                          device=device),
                max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
        cands = []
        for s in out.sequences:
            gen = strip_generated_row(s, tokens, eos_ids, stop_ids)
            cands.append(clip(tok.decode(gen, skip_special_tokens=True)))
        embs = enc.encode(cands, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype(np.float64)
        tstar = compass_target(act, L)
        if args.policy:
            sel = av_policy.select_policy(embs, tstar, tau=args.tau,
                                          generic_centroid=gen_centroid,
                                          gen_penalty=args.gen_penalty)
            return av_policy.apply_policy_text(cands, sel), sel
        j = int((embs @ tstar).argmax())
        return cands[j], None

    print(f"[ready] layers {layers} — type a text, empty line quits", flush=True)
    while True:
        try:
            text = input("\ntext> ").strip()
        except EOFError:
            break
        if not text:
            break
        acts = extract(text)
        for L in layers:
            pct = nearest_depth_pct(L, n_layers)
            tag = ""
            if rerank and L in rerank_layers:
                desc, meta = describe_rerank(acts[L], L, args.rerank_best_of)
                if args.policy and meta is not None:
                    tag = (f" [{meta['decision']} conf={meta['confidence']:.2f} "
                           f"agree={meta['agreement']:.2f}]")
                else:
                    tag = f" [rerank best-of-{args.rerank_best_of}]"
            else:
                desc = describe(acts[L], L)
            print(f"\n--- L{L} ({pct}%){tag} ---")
            print(desc, flush=True)


if __name__ == "__main__":
    main()
