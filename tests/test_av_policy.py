"""CPU unit tests for the pragmatic AV decoding policy (scripts/av_policy.py):
compass target, confidence-gated specific-vs-hedge selection, the genericness
penalty, inter-sample agreement, and emitted-text rendering. The GPU batch
evaluator (run_eval) is not covered here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import av_policy as A  # noqa: E402


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_compass_target_unit_and_centered():
    a = np.array([2.0, 0.0, 0.0])
    mu = np.array([1.0, 0.0, 0.0])
    W = np.eye(3)
    t = A.compass_target(a, mu, W)         # (a-mu)@W = [1,0,0] -> unit
    assert t.shape == (3,)
    assert np.allclose(t, [1.0, 0.0, 0.0])
    assert abs(np.linalg.norm(t) - 1.0) < 1e-9


def test_select_picks_most_faithful_and_gates_specific():
    tstar = _unit([1.0, 0.0, 0.0])
    S = np.stack([_unit([0.95, 0.05, 0.0]),     # faithful
                  _unit([0.0, 1.0, 0.0])])       # off
    sel = A.select_policy(S, tstar, tau=0.3)
    assert sel["idx"] == 0
    assert sel["decision"] == "specific"          # conf ~0.95 >= 0.3
    assert sel["confidence"] == pytest.approx(float(S[0] @ tstar))


def test_select_hedges_when_best_below_tau():
    tstar = _unit([1.0, 0.0, 0.0])
    S = np.stack([_unit([0.2, 1.0, 0.0]), _unit([0.1, 0.0, 1.0])])
    sel = A.select_policy(S, tstar, tau=0.5)
    assert sel["decision"] == "hedge"             # best faith < 0.5
    assert sel["confidence"] < 0.5


def test_genericness_penalty_demotes_template():
    tstar = _unit([1.0, 0.0, 0.0])
    centroid = _unit([0.0, 1.0, 0.0])             # "template" direction
    # sample 0: faith .7 but very generic (.7); sample 1: faith .55, generic 0.
    # without penalty the more-faithful sample 0 wins; a big genericness penalty
    # flips the choice to the specific sample 1.
    S = np.stack([_unit([0.7, 0.7, 0.141]),       # faith .7, generic .7
                  _unit([0.55, 0.0, 0.835])])      # faith .55, generic 0
    no_pen = A.select_policy(S, tstar, tau=0.0)
    assert no_pen["idx"] == 0                      # faithful pick wins w/o penalty
    pen = A.select_policy(S, tstar, tau=0.0,
                          generic_centroid=centroid, gen_penalty=1.0)
    assert pen["idx"] == 1                         # penalty flips to specific one
    assert pen["generic"] is not None


def test_agreement_high_when_samples_aligned():
    tstar = _unit([1.0, 0.0, 0.0])
    aligned = np.stack([_unit([1, 0.01, 0]), _unit([1, 0.02, 0]), _unit([1, 0, 0.01])])
    spread = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    a_hi = A.select_policy(aligned, tstar, tau=0.0)["agreement"]
    a_lo = A.select_policy(spread, tstar, tau=0.0)["agreement"]
    assert a_hi > 0.9 and a_lo < 0.1


def test_apply_policy_text_prefixes_only_hedge():
    descs = ["a specific cat fact", "generic filler"]
    spec = {"idx": 0, "decision": "specific"}
    hed = {"idx": 1, "decision": "hedge"}
    assert A.apply_policy_text(descs, spec) == "a specific cat fact"
    out = A.apply_policy_text(descs, hed)
    assert out.startswith(A.HEDGE_PREFIX) and out.endswith("generic filler")


def test_select_rejects_empty():
    with pytest.raises(ValueError):
        A.select_policy(np.zeros((0, 3)), _unit([1, 0, 0]), tau=0.3)
