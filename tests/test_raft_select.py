"""CPU unit tests for the RAFT/best-of-N distillation selection logic
(scripts/train_av_raft.py — lever, HUMAN-REVIEW gated, GPU-untested).

Covers the GPU-free, decision-critical pieces: `select_best` (the rejection-
sampling selector that picks the sampled description closest to the activation's
TRUE input text and enforces the faithfulness floor), `l2norm_rows`, and `_top1`.
The generation + SFT loop needs a GPU and is not covered here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_av_raft as R  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_select_best_picks_closest_above_floor():
    text = _unit([1.0, 0.0, 0.0])
    # sample 1 nearly orthogonal, sample 2 aligned, sample 0 anti-aligned
    embs = np.stack([_unit([-1.0, 0.0, 0.0]),
                     _unit([0.0, 1.0, 0.0]),
                     _unit([0.95, 0.05, 0.0])])
    desc, score = R.select_best(["bad", "ortho", "good"], embs, text, min_score=0.3)
    assert desc == "good"
    assert score == pytest.approx(float(embs[2] @ text))


def test_select_best_rejects_when_all_below_floor():
    text = _unit([1.0, 0.0, 0.0])
    embs = np.stack([_unit([0.0, 1.0, 0.0]), _unit([0.0, 0.0, 1.0])])
    desc, score = R.select_best(["a", "b"], embs, text, min_score=0.3)
    assert desc is None          # nothing clears the floor -> never distill confab
    assert score < 0.3


def test_select_best_floor_is_inclusive():
    text = _unit([1.0, 0.0, 0.0])
    # exactly at the floor should be kept
    c = 0.5
    embs = np.stack([_unit([c, (1 - c**2) ** 0.5, 0.0])])
    desc, score = R.select_best(["edge"], embs, text, min_score=0.5)
    assert desc == "edge"
    assert score == pytest.approx(0.5, abs=1e-9)


def test_select_best_empty():
    desc, score = R.select_best([], np.zeros((0, 3)), _unit([1, 0, 0]), 0.3)
    assert desc is None
    assert score == float("-inf")


def test_l2norm_rows_unit_and_zero_safe():
    M = np.array([[3.0, 4.0], [0.0, 0.0]])
    N = R.l2norm_rows(M)
    assert np.linalg.norm(N[0]) == pytest.approx(1.0)
    assert np.all(N[1] == 0.0)   # zero row stays zero, no NaN


def test_top1_identity_and_permuted():
    K = R.l2norm_rows(np.eye(4) + 1e-6)
    assert R._top1(K, K) == pytest.approx(1.0)        # each query nearest to itself
    Q = K[::-1]                                        # every query points at wrong key
    assert R._top1(Q, K) == pytest.approx(0.0)


def test_top1_empty_is_nan():
    assert np.isnan(R._top1(np.zeros((0, 4)), np.eye(4)))
