#!/usr/bin/env python3
"""
Workshop live-demo driver — demos 2-4 of the HAAISS run-of-show in ONE
warm session (load the model once, keep it in tmux, survive an ssh drop).

Mirrors the *tested* notebook code paths (NB01-NB03), but imports every
constant/template from nla_lib instead of carrying copies:

  read <sentence>   capture the clean-base activation at --layer, show the
                    model's own reply next to the NLA caption   (demo 2)
  depth             the same vector read at several claimed depths (demo 2/3)
  bug               normalize-TO-150 vs multiply-BY-150, side by side (demo 3)
  gap               round-trip + curry-recipe gap: naive (fails) then
                    centered ranking over distractors               (demo 4)
  help / quit

Run on any CUDA box (bf16 default; --load-4bit mirrors the Colab T4 path):

  python3 scripts/workshop_live_demo.py                 # interactive
  python3 scripts/workshop_live_demo.py --smoke         # preflight, then exit

Presenter notes live in working-docs/workshop-runbook.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nla_lib import (  # noqa: E402
    AR_TEMPLATE_RECONSTRUCT,
    INJECTION_SCALE,
    MODELS_HF,
    get_model,
    make_av_prompt,
    nearest_depth_pct,
    normalize_activation,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

MODEL_KEY = "qwen25-7b"
AV_ADAPTER = "anicka/nla-qwen2.5-7b-L20-av-v2"
AR_ADAPTER = "anicka/nla-qwen2.5-7b-L20-ar-v2"

# NB03's distractor set, verbatim — the centered-gap demo depends on having
# a few reconstructions to define "generic" before the true caption can win.
DISTRACTORS = [
    "- Recipe for Thai green curry with coconut milk and basil\n"
    "- Step-by-step cooking instructions for dinner",
    "- Legal contract clause about liability limitation active\n"
    "- Formal register, defined terms",
    "- Football match commentary, goal celebration active\n"
    "- Present-tense excited sports narration",
    "- Romantic poetry about moonlight and longing\n"
    "- Metaphor-dense lyrical register",
    "- Python exception traceback analysis active\n"
    "- Debugging context, error-message vocabulary",
]

RULE = "─" * 72


def get_layers(m):
    b = m.base_model.model if hasattr(m, "base_model") else m
    inner = b.model if hasattr(b, "model") else b
    return inner.layers


class Demo:
    def __init__(self, args):
        self.layer = args.layer
        hf_id = MODELS_HF[MODEL_KEY]
        spec = get_model(MODEL_KEY)
        self.inject_char = spec.injection_char
        print(f"loading {hf_id} ({'4-bit' if args.load_4bit else 'bf16'})...")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
            base = AutoModelForCausalLM.from_pretrained(
                hf_id, quantization_config=bnb, device_map={"": 0})
        else:
            base = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.bfloat16, device_map={"": 0})
        self.model = PeftModel.from_pretrained(base, args.av_adapter).eval()
        self.model.load_adapter(args.ar_adapter, adapter_name="ar")
        self.model.set_adapter("default")
        self.device = next(self.model.parameters()).device

        ids = self.tok.encode(self.inject_char, add_special_tokens=False)
        assert len(ids) == 1, f"injection char must be ONE token, got {ids}"
        self.inject_id = ids[0]

        n_layers = self.model.config.num_hidden_layers
        assert self.layer < n_layers, f"--layer {self.layer} >= {n_layers}"
        self.depth_pct = nearest_depth_pct(self.layer, n_layers)
        print(f"ready — layer {self.layer}/{n_layers} = depth {self.depth_pct}%, "
              f"adapters: {list(self.model.peft_config.keys())}")

        self.activation = None   # last captured vector
        self.prompt = None
        self.caption = None

    # -- capture (clean base: the mind we read is the model's, not the AV's) --
    def read(self, prompt, max_new_tokens=80):
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inp = self.tok(chat, return_tensors="pt").to(self.device)
        grab = {}

        def hook(mod, inpt, out):
            h = out[0] if isinstance(out, tuple) else out
            if "h" not in grab:              # FIRST forward pass only
                grab["h"] = h[:, -1, :].detach()

        handle = get_layers(self.model)[self.layer].register_forward_hook(hook)
        with torch.no_grad(), self.model.disable_adapter():
            out = self.model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        handle.remove()
        reply = self.tok.decode(out[0][inp.input_ids.shape[1]:],
                                skip_special_tokens=True)
        self.activation = grab["h"].squeeze(0)
        self.prompt = prompt
        self.caption = self.describe(self.activation)
        print(f"\n{RULE}\nMODEL SAYS (base, no adapter):\n  {reply.strip()}")
        print(f"{RULE}\nNLA READS (layer {self.layer}, depth {self.depth_pct}%):"
              f"\n  {self.caption}\n{RULE}")

    # -- the injection, importable-constants edition --
    def describe(self, activation, depth=None, scale_fn=None,
                 max_new_tokens=120):
        depth = self.depth_pct if depth is None else depth
        scale_fn = normalize_activation if scale_fn is None else scale_fn
        self.model.set_adapter("default")
        ids = self.tok.encode(make_av_prompt(depth, self.inject_char),
                              add_special_tokens=True)
        pos = ids.index(self.inject_id)
        emb = self.model.get_input_embeddings()(
            torch.tensor([ids], device=self.device)).clone()
        emb[0, pos, :] = scale_fn(activation.to(emb.dtype))
        with torch.no_grad():
            out = self.model.generate(
                inputs_embeds=emb, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tok.eos_token_id)
        seq = out[0]
        gen = seq[len(ids):] if seq.shape[0] > len(ids) else seq
        return (self.tok.decode(gen, skip_special_tokens=True)
                .split("</explanation>")[0].strip())

    def _need_activation(self):
        if self.activation is None:
            print("no activation yet — `read <sentence>` first")
            return True
        return False

    # -- demo 3a: the multiply-vs-normalize break --
    def bug(self):
        if self._need_activation():
            return
        v = self.activation
        ok = normalize_activation(v)
        broken = v * INJECTION_SCALE
        print(f"\nraw norm {v.float().norm():.0f}  →  "
              f"normalize-TO-{INJECTION_SCALE:.0f}: {ok.float().norm():.0f}"
              f"   |   multiply-BY-{INJECTION_SCALE:.0f}: "
              f"{broken.float().norm():.0f}")
        print(f"\n{RULE}\nCORRECT (normalize TO {INJECTION_SCALE:.0f}):\n  "
              f"{self.describe(v)}")
        print(f"{RULE}\nBUG (multiply BY {INJECTION_SCALE:.0f}):\n  "
              f"{self.describe(v, scale_fn=lambda x: x * INJECTION_SCALE)}"
              f"\n{RULE}")

    # -- demo 3b: depth is an input --
    def depth(self, depths=(71, 40, 96, 10)):
        if self._need_activation():
            return
        print()
        for d in depths:
            tag = " (trained)" if d == self.depth_pct else ""
            print(f"depth told = {d:>2}%{tag}  →  "
                  f"{self.describe(self.activation, depth=d)}")

    # -- demo 4: round-trip + gap --
    def reconstruct(self, description):
        self.model.set_adapter("ar")
        ids = self.tok.encode(
            AR_TEMPLATE_RECONSTRUCT.format(explanation=description),
            add_special_tokens=True)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device),
                             output_hidden_states=True, use_cache=False)
        self.model.set_adapter("default")
        return out.hidden_states[self.layer + 1][0, -1].float().cpu()

    def gap(self):
        if self._need_activation():
            return
        act = self.activation.float().cpu()
        if self.caption is None:
            self.caption = self.describe(self.activation)
        print(f"\ncaption: {self.caption}")

        def cos(a, b):
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

        true_recon = self.reconstruct(self.caption)
        print(f"round-trip cos (raw): {cos(act, true_recon):.3f}")

        wrong = DISTRACTORS[0]
        c_wrong = cos(act, self.reconstruct(wrong))
        print(f"\nNAIVE — raw cosine, real vs curry recipe:")
        print(f"  cos(real)  = {cos(act, true_recon):.3f}")
        print(f"  cos(wrong) = {c_wrong:.3f}   ← both ~same. Raw cosine is "
              f"mostly the shared mean. Why we center:")

        recons = [self.reconstruct(d) for d in DISTRACTORS]
        mean_recon = torch.stack(recons).mean(0)
        a_dev = act - mean_recon
        scores = {"TRUE caption": cos(a_dev, true_recon - mean_recon)}
        for d, r in zip(DISTRACTORS, recons):
            scores[d.split("\n")[0][:48]] = cos(a_dev, r - mean_recon)
        print(f"\nCENTERED (deviation from the mean reconstruction):")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        for name, s in ranked:
            mark = "  ← the vector votes for this one" \
                if name == "TRUE caption" else ""
            print(f"  {s:+.3f}  {name}{mark}")
        centered_gap = scores["TRUE caption"] - max(
            v for k, v in scores.items() if k != "TRUE caption")
        print(f"\ncentered gap = {centered_gap:+.3f}  "
              f"(positive ⇒ the vector prefers the truth)")
        return ranked[0][0], centered_gap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=20,
                    help="layer the single-layer adapters live at")
    ap.add_argument("--av-adapter", default=AV_ADAPTER)
    ap.add_argument("--ar-adapter", default=AR_ADAPTER)
    ap.add_argument("--load-4bit", action="store_true",
                    help="mirror the Colab T4 load path (default: bf16)")
    ap.add_argument("--smoke", action="store_true",
                    help="run one canned pass of all demos, assert, exit")
    args = ap.parse_args()

    demo = Demo(args)

    if args.smoke:
        demo.read("Explain how a hash map handles collisions.")
        demo.bug()
        demo.depth()
        winner, cgap = demo.gap()
        assert winner == "TRUE caption", "centered ranking failed"
        assert cgap > 0.03, f"centered gap suspiciously small: {cgap:+.3f}"
        norm = normalize_activation(demo.activation).float().norm().item()
        assert abs(norm - INJECTION_SCALE) < 1.0
        print("\nsmoke: all demos ran, TRUE caption ranks #1, "
              f"centered gap {cgap:+.3f} ✓")
        return

    print("\ncommands: read <sentence> | bug | depth | gap | help | quit")
    while True:
        try:
            line = input("\ndemo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "quit":
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "read":
            if rest:
                demo.read(rest)
            else:
                print("usage: read <sentence>")
        elif cmd == "bug":
            demo.bug()
        elif cmd == "depth":
            demo.depth()
        elif cmd == "gap":
            demo.gap()
        else:
            print(f"unknown command {cmd!r} — read/bug/depth/gap/help/quit")


if __name__ == "__main__":
    main()
