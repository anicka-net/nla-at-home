"""Unit tests for the faithfulness-reward additions to train_joint_grpo_phi4.py
(lever #6 — operator-approved objective change, 2026-06-21).

Covers the GPU-free pieces: advantage standardization (`_z`), top1 retrieval
(`_top1`), the contrastive faith reward (`FaithScorer.faith_reward`), and the
holdout faith metric (`_faith_top1`). The end-to-end GRPO loop needs a GPU and
is not covered here.

The FaithScorer tests load the real all-MiniLM-L6-v2 encoder on CPU (the same
model the validated faithfulness metric and the kb tool use), so they exercise
the actual text-semantic space, not a mock.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_joint_grpo_phi4 as J


# ---- _z : reward standardization in advantage space ----

def test_z_standardizes_to_unit_variance():
    z = J._z(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert abs(z.mean().item()) < 1e-6
    assert z.std().item() == pytest.approx(1.0, abs=1e-4)


def test_z_degenerate_group_returns_zero():
    # all-equal rewards -> zero advantage (no spurious gradient direction)
    z = J._z(torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(z, torch.zeros(3), atol=1e-6)


def test_lambda_zero_reduces_to_legacy_roundtrip():
    # adv = (1-λ)·z(R_rt) + λ·z(R_faith) must equal legacy z(R_rt) at λ=0,
    # so the faith reward is a strict, opt-in superset of prior behavior.
    r_rt = torch.tensor([0.1, -0.2, 0.3, 0.0])
    lam = 0.0
    mix = J._z(r_rt).mul(1 - lam) + J._z(torch.zeros(4)).mul(lam)
    assert torch.allclose(J._z(mix), J._z(r_rt), atol=1e-5)


# ---- _top1 : aligned self-retrieval ----

def test_top1_self_retrieval_is_one():
    A = F.normalize(torch.eye(5) + 0.001 * torch.randn(5, 5), dim=1)
    assert J._top1(A, A) == pytest.approx(1.0)


def test_top1_empty_is_nan():
    import math
    assert math.isnan(J._top1(torch.zeros(0, 8), torch.zeros(0, 8)))


# ---- FaithScorer : contrastive identification reward (real encoder) ----

@pytest.fixture(scope="module")
def scorer():
    fs = J.FaithScorer("sentence-transformers/all-MiniLM-L6-v2",
                       torch.device("cpu"))
    gt = {16: {
        "cake": "a recipe for chocolate cake with flour and sugar",
        "code": "python code that sorts a list using quicksort",
        "rome": "the history of the roman empire and its emperors",
    }}
    fs.precompute(gt, [16])
    return fs, gt


def test_precompute_embeds_every_gt(scorer):
    fs, _ = scorer
    assert set(fs.gt_emb[16].keys()) == {"cake", "code", "rome"}
    for v in fs.gt_emb[16].values():
        assert v.shape == (384,)
        assert float(v.norm()) == pytest.approx(1.0, abs=1e-4)


def test_faith_reward_prefers_on_topic_description(scorer):
    fs, _ = scorer
    descs = [
        "baking a cake: mix flour sugar and cocoa",  # close to GT 'cake'
        "dessert with chocolate and sugar",          # close to GT 'cake'
        "sorting algorithms in python",              # off-topic (code)
        "ancient rome senate",                       # off-topic (rome)
    ]
    r = fs.faith_reward(descs, 16, "cake", ["code", "rome"])
    assert len(r) == 4
    # an on-topic description must out-reward both off-topic ones
    assert r[0] > r[2] and r[0] > r[3]
    assert r[1] > r[2]


def test_faith_reward_missing_gt_returns_zeros(scorer):
    fs, _ = scorer
    assert fs.faith_reward(["x", "y"], 16, "does-not-exist", ["code"]) == [0.0, 0.0]


def test_faith_reward_no_distractors_is_bare_similarity(scorer):
    fs, _ = scorer
    # with no negatives the reward is just cos(desc, gt_correct) in [-1, 1]
    r = fs.faith_reward(["a chocolate cake recipe"], 16, "cake", [])
    assert -1.0 <= r[0] <= 1.0


# ---- _faith_top1 : holdout monitor (matched + style-fair input-text) ----

def test_faith_top1_returns_both_metrics_in_range(scorer):
    fs, gt = scorer
    corpus = {"cake": "chocolate cake recipe",
              "code": "quicksort in python",
              "rome": "roman empire history"}
    descs_by_layer = {16: (
        ["cake", "code", "rome"],
        ["a cake with chocolate", "sorting a list in python", "the roman emperors"],
    )}
    ft = J._faith_top1(fs, descs_by_layer, gt, corpus, [16])
    assert 0.0 <= ft["matched_top1"] <= 1.0
    assert 0.0 <= ft["inputtext_top1"] <= 1.0


def test_faith_top1_skips_inputtext_when_corpus_missing(scorer):
    fs, gt = scorer
    descs_by_layer = {16: (["cake", "code"], ["a cake", "some code"])}
    ft = J._faith_top1(fs, descs_by_layer, gt, {}, [16])
    assert ft["matched_top1"] is not None
    assert ft["inputtext_top1"] is None  # no input texts -> style-fair skipped


# ---- _fmt_faith : log formatting ----

def test_fmt_faith_handles_none():
    assert J._fmt_faith(None) == "n/a"
    assert J._fmt_faith({"matched_top1": 0.46, "inputtext_top1": None}) == "0.460/n/a"
    assert J._fmt_faith({"matched_top1": 0.46, "inputtext_top1": 0.41}) == "0.460/0.410"
