"""sink_fix: the Gemma massive-activation outlier treatment.

Synthetic reproduction of the measured pathology (one dim carrying ~97%
of energy pins all pairwise cosines near 1.0) and verification that
center+drop-top-PC repairs the geometry without deleting the signal.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sink_fix  # noqa: E402


def synth_layer(n=400, d=64, sink_dim=10, sink_scale=300.0, seed=0):
    """Real Gemma sink has huge MEAN and huge VARIANCE (centering alone
    kills only the constant part; the PC drop removes the fluctuation)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X[:, sink_dim] = sink_scale + 50.0 * torch.randn(n, generator=g)
    return X


def mean_pairwise_cos(X, cap=100):
    Xn = torch.nn.functional.normalize(X[:cap], dim=1)
    sims = Xn @ Xn.T
    n = sims.shape[0]
    return (sims.sum() - n) / (n * (n - 1))


def test_outlier_pins_cosine_and_fix_repairs_it():
    X = synth_layer()
    assert mean_pairwise_cos(X) > 0.98  # pathology reproduced
    params = sink_fix.fit([X], drop_top_pc=1)
    Xf = sink_fix.apply(params, 0, X)
    assert abs(mean_pairwise_cos(Xf)) < 0.1  # geometry repaired


def test_fix_preserves_class_signal():
    # two classes separated in a non-sink direction must stay separable
    g = torch.Generator().manual_seed(1)
    A = synth_layer(seed=1)
    B = synth_layer(seed=2)
    B[:, 3] += 4.0  # class signal on dim 3
    params = sink_fix.fit([torch.cat([A, B])], drop_top_pc=1)
    Af = sink_fix.apply(params, 0, A)
    Bf = sink_fix.apply(params, 0, B)
    gap = (Bf[:, 3].mean() - Af[:, 3].mean()).item()
    assert gap > 3.0  # signal survives the projection


def test_roundtrip_save_load(tmp_path):
    X = synth_layer()
    params = sink_fix.fit([X], drop_top_pc=2)
    p = tmp_path / "sink_fix.pt"
    sink_fix.save(params, p)
    loaded = sink_fix.load(p)
    assert loaded["drop_top_pc"] == 2
    a = sink_fix.apply(params, 0, X[:5])
    b = sink_fix.apply(loaded, 0, X[:5])
    assert torch.allclose(a, b, atol=1e-5)


def test_single_vector_apply_matches_batch():
    X = synth_layer()
    params = sink_fix.fit([X], drop_top_pc=1)
    batch = sink_fix.apply(params, 0, X[:3])
    single = torch.stack([sink_fix.apply(params, 0, X[i]) for i in range(3)])
    assert torch.allclose(batch, single, atol=1e-5)
