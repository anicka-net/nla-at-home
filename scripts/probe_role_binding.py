#!/usr/bin/env python3
"""Probe: does the last-token residual encode WHO-did-WHAT-to-WHOM?

Premise test for fixing the AR's bag-of-words ceiling (DESIGN.md § AR
faithfulness audit, finding 4). Before teaching any reader to use word
order, check whether the substrate distinguishes role binding at all at
our extraction position. If it doesn't, no reader can recover it and
joint training could only fake it.

Design: N scenarios, each a triple sharing one event schema:
  orig  "The dog chased the cat across the yard."
  swap  "The cat chased the dog across the yard."   (same bag of words,
                                                     opposite meaning)
  pass  "The cat was chased by the dog across the yard."
                                                    (surface order of SWAP,
                                                     meaning of ORIG)

The passive is the arbiter. Per layer, with centered cosine (means from
the MAIN corpus, same convention as the GRPO reward):

  c_swap = cos(orig, swap)   high if representation tracks surface/bag
  c_pass = cos(orig, pass)   high if representation tracks meaning
  role_index = c_pass - c_swap
      > 0  -> semantics wins: roles are encoded beyond word identity;
              a structure-aware AR has something to read (proceed with
              relational corpus category + order-swap hard negatives)
      <= 0 -> surface wins: the bag-of-words ceiling is the substrate's,
              not the AR's, at that depth
  floor  = mean cos(orig_i, orig_j!=i)  cross-scenario baseline

Swapped versions are chosen to stay PLAUSIBLE (named people, animal
pairs) so surprise doesn't masquerade as role signal.

Two-step usage (extraction reuses extract_activations.py byte-for-byte):

  1. python3 scripts/probe_role_binding.py --make-texts
  2. python3 scripts/extract_activations.py --model qwen25-7b --all-layers \
       --input working-docs/role_binding_texts.json --output-suffix _rolebind
  3. python3 scripts/probe_role_binding.py --analyze \
       --probe-activations corpus/activations/qwen25-7b_all_layers_rolebind.pt \
       --corpus-activations corpus/activations/qwen25-7b_all_layers.pt
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).parent.parent

# (agent, verb_active, verb_passive, patient, suffix) — both directions
# plausible; people are named so the swap stays natural.
SCENARIOS = [
    ("the dog", "chased", "was chased by", "the cat", "across the yard"),
    ("the goalkeeper", "pushed", "was pushed by", "the striker", "during the match"),
    ("Anna", "called", "was called by", "Peter", "on Sunday evening"),
    ("Marta", "hired", "was hired by", "Jonas", "last spring"),
    ("the teacher", "praised", "was praised by", "the student", "after the lecture"),
    ("Lena", "photographed", "was photographed by", "Tomas", "at the wedding"),
    ("the champion", "defeated", "was defeated by", "the challenger", "in the final"),
    ("Eva", "interviewed", "was interviewed by", "Martin", "for the podcast"),
    ("the toddler", "followed", "was followed by", "the puppy", "around the garden"),
    ("Nora", "invited", "was invited by", "Filip", "to the opening"),
    ("the detective", "questioned", "was questioned by", "the journalist", "about the case"),
    ("Klara", "hugged", "was hugged by", "her brother", "at the station"),
    ("the mentor", "recommended", "was recommended by", "the intern", "for the award"),
    ("Ivan", "beat", "was beaten by", "his neighbour", "at chess"),
    ("the nurse", "woke", "was woken by", "the patient", "before dawn"),
    ("Sofie", "sketched", "was sketched by", "the painter", "in the atelier"),
    ("the guide", "warned", "was warned by", "the climber", "about the storm"),
    ("Adam", "texted", "was texted by", "Lucie", "around midnight"),
    ("the editor", "criticized", "was criticized by", "the author", "in the reply"),
    ("Hana", "taught", "was taught by", "Viktor", "over the summer"),
    ("the fox", "watched", "was watched by", "the rabbit", "from the hedge"),
    ("Milan", "paid", "was paid by", "Ondrej", "for the repairs"),
    ("the coach", "encouraged", "was encouraged by", "the runner", "before the race"),
    ("Tereza", "surprised", "was surprised by", "her colleague", "with the news"),
    ("the landlord", "sued", "was sued by", "the tenant", "over the deposit"),
    ("Jakub", "greeted", "was greeted by", "the mayor", "at the ceremony"),
    ("the analyst", "briefed", "was briefed by", "the director", "on Monday"),
    ("Alice", "visited", "was visited by", "her grandmother", "every Friday"),
    ("the referee", "cautioned", "was cautioned by", "the captain", "in extra time"),
    ("Petra", "quoted", "was quoted by", "the historian", "in the article"),
    ("the drummer", "replaced", "was replaced by", "the guitarist", "for one song"),
    ("Radek", "carried", "was carried by", "his teammate", "off the field"),
    ("the translator", "corrected", "was corrected by", "the poet", "twice"),
    ("Zuzana", "trained", "was trained by", "the veteran", "all winter"),
    ("the owl", "startled", "was startled by", "the hare", "at dusk"),
    ("Daniel", "sponsored", "was sponsored by", "the bakery", "this season"),
]

TEXTS_PATH = REPO / "working-docs" / "role_binding_texts.json"


def cap(s):
    return s[0].upper() + s[1:]


def build_texts():
    items = []
    for i, (a, v_act, v_pas, b, suf) in enumerate(SCENARIOS):
        sid = f"RB_{i:03d}"
        items.append({"id": f"{sid}_orig", "text": f"{cap(a)} {v_act} {b} {suf}.",
                      "category": "RB_probe", "group": "probe"})
        items.append({"id": f"{sid}_swap", "text": f"{cap(b)} {v_act} {a} {suf}.",
                      "category": "RB_probe", "group": "probe"})
        items.append({"id": f"{sid}_pass", "text": f"{cap(b)} {v_pas} {a} {suf}.",
                      "category": "RB_probe", "group": "probe"})
    return items


def make_texts():
    items = build_texts()
    TEXTS_PATH.parent.mkdir(exist_ok=True)
    TEXTS_PATH.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} texts ({len(SCENARIOS)} scenarios) -> {TEXTS_PATH}")
    print("NOTE: kept OUT of corpus/generated on purpose — probe texts must "
          "never leak into training corpora.")


def analyze(args):
    import torch

    probe = torch.load(args.probe_activations, map_location="cpu")
    corpus = torch.load(args.corpus_activations, map_location="cpu")
    acts, ids = probe["activations"], probe["ids"]
    n_layers = probe["n_layers"]
    if corpus["n_layers"] != n_layers or corpus["model"] != probe["model"]:
        raise SystemExit("probe/corpus activation files disagree on model or "
                         f"layers: {probe['model']}/{n_layers} vs "
                         f"{corpus['model']}/{corpus['n_layers']}")
    row = {t: i for i, t in enumerate(ids)}
    n_sc = len(SCENARIOS)

    def cvec(L, tid, mean):
        v = acts[L][row[tid]].float() - mean
        return v / v.norm().clamp_min(1e-12)

    print(f"{len(ids)} probe texts, {n_sc} scenarios, {n_layers} layers "
          f"({probe['model']})")
    print(f"{'layer':>5s} {'c_swap':>8s} {'c_pass':>8s} {'ROLE_IDX':>9s} "
          f"{'floor':>7s} {'P(pass>swap)':>13s}")
    out = {}
    for L in range(n_layers):
        mean = corpus["activations"][L].float().mean(0)
        o = torch.stack([cvec(L, f"RB_{i:03d}_orig", mean) for i in range(n_sc)])
        s = torch.stack([cvec(L, f"RB_{i:03d}_swap", mean) for i in range(n_sc)])
        p = torch.stack([cvec(L, f"RB_{i:03d}_pass", mean) for i in range(n_sc)])
        c_swap = (o * s).sum(-1)
        c_pass = (o * p).sum(-1)
        cross = o @ o.T
        floor = (cross.sum() - cross.trace()) / (n_sc * (n_sc - 1))
        role_idx = (c_pass - c_swap).mean().item()
        p_win = (c_pass > c_swap).float().mean().item()
        out[L] = {"c_swap": c_swap.mean().item(), "c_pass": c_pass.mean().item(),
                  "role_index": role_idx, "floor": floor.item(),
                  "p_pass_gt_swap": p_win}
        print(f"L{L:4d} {out[L]['c_swap']:8.3f} {out[L]['c_pass']:8.3f} "
              f"{role_idx:9.3f} {out[L]['floor']:7.3f} {p_win:13.2f}")
    print("\nrole_index > 0 & P(pass>swap) >> 0.5  -> roles encoded beyond "
          "bag of words; the AR fix path (relational category + order-swap "
          "negatives) has something to learn from.")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-texts", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    ap.add_argument("--probe-activations")
    ap.add_argument("--corpus-activations")
    ap.add_argument("--out", default="working-docs/role_binding_result.json")
    args = ap.parse_args()
    if args.make_texts:
        make_texts()
    else:
        if not (args.probe_activations and args.corpus_activations):
            ap.error("--analyze needs --probe-activations and --corpus-activations")
        analyze(args)


if __name__ == "__main__":
    main()
