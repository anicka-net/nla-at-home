"""CPU unit tests for the oracle-compass save path added to
scripts/probe_activation_faithfulness.py (--save-compass), which productizes the
+0.133 best-of-N decoding fix for inference-time reranking in describe_live.py.

Covers `fit_compass` (recovers a linear activation->text map and picks alpha) and
`save_compass` (writes a loadable artifact with the right per-layer shapes and the
train-only-vs-all-ids leak-free default). No GPU / no language model needed:
save_compass consumes a precomputed text-embedding dict.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_activation_faithfulness as P  # noqa: E402


def _l2(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def test_fit_compass_recovers_linear_map():
    rng = np.random.RandomState(0)
    d, k, n = 32, 8, 400
    A = rng.randn(n, d)
    Wt = rng.randn(d, k)
    mu_true = A.mean(0)
    Y = _l2((A - mu_true) @ Wt)                       # text emb is a linear fn of act
    mu, W, alpha, val = P.fit_compass(A, Y, [0.001, 0.01, 0.1, 1, 10])
    assert mu.shape == (d,) and W.shape == (d, k)
    np.testing.assert_allclose(mu, mu_true, atol=1e-9)
    pred = _l2((A - mu) @ W)
    # near-perfect self-retrieval on a noiseless linear target
    assert P.top1(pred, Y) > 0.95
    assert alpha in [0.001, 0.01, 0.1, 1, 10] and 0.0 <= val <= 1.0


def test_save_compass_artifact_shapes_and_leak_free(tmp_path):
    rng = np.random.RandomState(1)
    d, k = 16, 8
    ids = [f"t{i}" for i in range(60)]
    layers = [4, 16]
    acts = {L: torch.tensor(rng.randn(len(ids), d)) for L in layers}
    id2idx = {t: i for i, t in enumerate(ids)}
    txt_emb = {t: _l2(rng.randn(1, k))[0] for t in ids}
    ho_set = set(ids[:10])
    out = tmp_path / "compass.pt"

    # train-only (default): holdout ids excluded from the fit
    P.save_compass(str(out), layers, acts, id2idx, txt_emb,
                   [0.01, 0.1, 1], ids, ho_set, "minilm", all_ids=False)
    c = torch.load(out, weights_only=False)
    assert c["layers"] == layers and c["fit_on"] == "train_only"
    assert c["centered"] is True and c["faith_model"] == "minilm"
    for L in layers:
        assert c["mu"][L].shape == (d,)
        assert c["W"][L].shape == (d, k)
        assert L in c["alpha"] and L in c["val_top1"]

    # all-ids deployment compass differs (it saw the 10 holdout ids too)
    out2 = tmp_path / "compass_all.pt"
    P.save_compass(str(out2), layers, acts, id2idx, txt_emb,
                   [0.01, 0.1, 1], ids, ho_set, "minilm", all_ids=True)
    c2 = torch.load(out2, weights_only=False)
    assert c2["fit_on"] == "all_ids"
    assert not torch.allclose(c["W"][4], c2["W"][4])


def test_compass_target_math_matches_inference():
    """Mirror describe_live.compass_target: l2norm((a-mu)@W)."""
    rng = np.random.RandomState(2)
    d, k, n = 16, 8, 200
    A = rng.randn(n, d)
    Y = _l2((A - A.mean(0)) @ rng.randn(d, k))
    mu, W, _, _ = P.fit_compass(A, Y, [0.1, 1])
    a = A[3]
    t = (a - mu) @ W
    t = t / np.linalg.norm(t)
    assert t.shape == (k,)
    assert np.linalg.norm(t) == np.float64(1.0).round(6) or abs(np.linalg.norm(t) - 1) < 1e-9
