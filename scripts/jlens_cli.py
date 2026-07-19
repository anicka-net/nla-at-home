#!/usr/bin/env python3
"""Jacobian-lens CLI — the geometry screen for the live demo.

One prompt in, three things out, all on the SAME captured hidden state h:
  * logit lens   — jm.unembed(h): pretends h already uses the final basis.
  * J-lens       — jm.unembed(lens.transport(h, L)): estimates what later
                   layers will DO with h, then applies the same output head.
  * NC controls  — the same lens on a random vector and on the WRONG layer's
                   Jacobian, so a plausible top-k is never trusted by itself.

And one manipulation (:ablate): transpose answer tokens back to hidden-space
directions with J̄ᵀ, project them out of the residual stream across a band of
layers, regenerate, and show the answer move. A broad causal test, not a claim
of a single neuron.

The average Jacobians are PRE-FITTED and published — this tool only loads them
(anicka/jlens-qwen2.5-7b-instruct); nothing is estimated at runtime.

Requires the jacobian-lens package (Apache-2.0):
  pip install "git+https://github.com/anthropics/jacobian-lens"
(or clone it and pass --jlens-repo /path/to/jacobian-lens).

Usage (GPU host):
  python scripts/jlens_cli.py --device cuda
  python scripts/jlens_cli.py --device cuda --layer 21 \
      --prompt "Fact: the currency used in the country shaped like a boot is"

REPL commands:
  <text>            read <text> at the current layer (logit lens + J-lens)
  :layer N          switch the read/transport layer
  :nc               random-vector + wrong-layer controls for the last prompt
  :ablate T1,T2     J̄ᵀ-project the surface forms out and regenerate (band)
  :band L1,L2,..    set the layers used by :ablate (default = current layer)
  :reset            layer -> startup default, clear the ablation band
  :help  /  :quit
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# Prefer a pip-installed `jlens`; fall back to a local clone of anthropics/
# jacobian-lens (override the clone location with --jlens-repo).
JLENS_REPO = os.environ.get(
    "JLENS_REPO", os.path.expanduser("~/playground/jacobian-lens"))
try:
    import jlens  # noqa: F401
except ImportError:
    if os.path.isdir(JLENS_REPO) and JLENS_REPO not in sys.path:
        sys.path.insert(0, JLENS_REPO)

from nla_lib import get_model  # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "magenta": "\033[35m"}


def c(text, color):
    return f"{C.get(color, '')}{text}{C['reset']}"


def topk_toks(tok, logits, k=5):
    return [tok.decode([i]).strip() or repr(tok.decode([i]))
            for i in logits.topk(k).indices]


def capture_h(model, tok, prompt, layer, device, use_chat):
    """Last-token hidden state after `layer`. Raw prompt = the lens's web-text
    home distribution (default); chat template optional for parity with NLA."""
    if use_chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt").to(device)
    grab = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grab["h"] = h[:, -1, :].detach()

    layers = model.model.layers if hasattr(model, "model") else model.layers
    handle = layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return grab["h"].squeeze(0).float()


def read(model, tok, jm, lens, prompt, layer, device, use_chat, k=5):
    if layer not in lens.source_layers:
        print(c(f"  lens not fitted at layer {layer}; fitted: "
                f"{sorted(lens.source_layers)}", "red"))
        return None
    h = capture_h(model, tok, prompt, layer, device, use_chat)
    hb = h.unsqueeze(0)
    ll = topk_toks(tok, jm.unembed(hb)[0], k)
    jl = topk_toks(tok, jm.unembed(lens.transport(hb, layer))[0], k)
    print(f"\n  {c('PROMPT', 'bold')}   : {prompt}")
    print(f"  {c('layer', 'dim')}    : L{layer}")
    print(f"  {c('logit lens', 'yellow')}: {ll}")
    print(f"  {c('J-lens', 'green')}    : {jl}")
    print(f"  {c('(what if h were already output)', 'dim')} vs "
          f"{c('(what h will become downstream)', 'dim')}\n")
    return h


def nc_controls(tok, jm, lens, h, layer, k=5):
    if h is None:
        print(c("  read a prompt first.", "red"))
        return
    rand = torch.randn_like(h)
    rand = rand / rand.norm() * h.norm()
    wrong = sorted(x for x in lens.source_layers if x != layer)
    wl = wrong[0] if wrong else layer
    print(f"\n  {c('NC controls at L%d' % layer, 'bold')} "
          f"(a plausible top-k is not evidence by itself):")
    print(f"  {c('random vector', 'red')} : "
          f"{topk_toks(tok, jm.unembed(lens.transport(rand.unsqueeze(0), layer))[0], k)}")
    print(f"  {c('wrong layer J_%d' % wl, 'red')}: "
          f"{topk_toks(tok, jm.unembed(lens.transport(h.unsqueeze(0), wl))[0], k)}")
    print(f"  {c('matched J_%d' % layer, 'green')}  : "
          f"{topk_toks(tok, jm.unembed(lens.transport(h.unsqueeze(0), layer))[0], k)}")
    print(c("  only the matched condition should read the answer.\n", "dim"))


def token_span(lens, base, tok, tokens, layer, device, tol=1e-4):
    """SVD basis of {J̄_layerᵀ · w_token} with a singular-value floor."""
    J = lens.jacobians[layer].float()
    W = base.get_output_embeddings().weight
    dirs = []
    for t in tokens:
        ids = tok.encode(t, add_special_tokens=False)
        if not ids:
            continue
        dirs.append(J.T @ W[ids[0]].float().cpu())
    if not dirs:
        return None, 0
    M = torch.stack(dirs, dim=1)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = int((S > S.max() * tol).sum())
    return U[:, :keep].to(device), keep


def ablate(model, tok, base, lens, prompt, tokens, band, device):
    inp = tok(prompt, return_tensors="pt").to(device)
    layers = model.model.layers if hasattr(model, "model") else model.layers

    def gen():
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    base_ans = gen()
    spans = {}
    for L in band:
        Q, r = token_span(lens, base, tok, tokens, L, device)
        if Q is not None:
            spans[L] = Q
            print(c(f"  L{L}: retained rank {r}", "dim"))

    def make_hook(Q):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            coef = h.float() @ Q
            h_new = h - (coef @ Q.T).to(h.dtype)
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
        return hook

    handles = [layers[L].register_forward_hook(make_hook(Q))
               for L, Q in spans.items()]
    try:
        edit_ans = gen()
    finally:
        for hd in handles:
            hd.remove()
    print(f"\n  {c('baseline', 'bold')}      : {base_ans}")
    print(f"  {c('after ablation', 'magenta')}: {edit_ans}")
    print(c(f"  (removed {tokens} directions across L{band})\n", "dim"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat-template", action="store_true",
                    help="capture at the chat generation position instead of "
                         "the raw last token (default = lens's web-text home)")
    ap.add_argument("--jlens-repo", default=JLENS_REPO)
    ap.add_argument("--jlens-id", default="anicka/jlens-qwen2.5-7b-instruct")
    ap.add_argument("--jlens-file", default="qwen2.5-7b-instruct_jlens.pt")
    args = ap.parse_args()

    if args.jlens_repo not in sys.path:
        sys.path.insert(0, args.jlens_repo)
    import jlens

    spec = get_model("qwen25-7b")
    device = args.device
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading Qwen 2.5 7B...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=dtype, trust_remote_code=spec.trust_remote_code).to(device).eval()
    print("done")

    print("loading fitted J-lens (no runtime estimation)...", end=" ", flush=True)
    lens = jlens.JacobianLens.from_pretrained(args.jlens_id, filename=args.jlens_file)
    jm = jlens.from_hf(base, tok)
    print("done")
    print(c(f"fitted layers: {sorted(lens.source_layers)}", "dim"))

    default_layer = args.layer
    layer = args.layer
    band = [layer]
    last_h = None

    if args.prompt:
        last_h = read(base, tok, jm, lens, args.prompt, layer, device, args.chat_template)
        return

    print(f"\n{c('Jacobian-lens CLI', 'bold')} — logit lens vs J-lens, NC controls, ablation")
    print(c("commands: :layer N  :nc  :ablate T1,T2  :band L1,L2  :help  :quit\n", "dim"))
    last_prompt = None
    while True:
        try:
            line = input(c(f"jlens[L{layer}]> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\nGoodbye.", "dim"))
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line == ":help":
            print(__doc__)
            continue
        if line == ":reset":
            layer = default_layer
            band = [layer]
            last_h = None
            print(c(f"  reset -> layer L{layer}, band cleared", "dim"))
            continue
        if line.startswith(":layer"):
            try:
                layer = int(line.split()[1]); band = [layer]
                print(c(f"  layer -> L{layer}", "dim"))
            except (IndexError, ValueError):
                print(c("  usage: :layer N", "red"))
            continue
        if line.startswith(":band"):
            try:
                band = [int(x) for x in line.split(None, 1)[1].replace(",", " ").split()]
                print(c(f"  ablation band -> L{band}", "dim"))
            except (IndexError, ValueError):
                print(c("  usage: :band L1,L2,..", "red"))
            continue
        if line == ":nc":
            nc_controls(tok, jm, lens, last_h, layer)
            continue
        if line.startswith(":ablate"):
            if last_prompt is None:
                print(c("  read a prompt first.", "red")); continue
            try:
                toks = [t if t.startswith(" ") else " " + t
                        for t in line.split(None, 1)[1].replace(",", " ").split()]
            except IndexError:
                print(c("  usage: :ablate token1,token2", "red")); continue
            ablate(base, tok, base, lens, last_prompt, toks, band, device)
            continue
        last_prompt = line
        last_h = read(base, tok, jm, lens, line, layer, device, args.chat_template)


if __name__ == "__main__":
    main()
