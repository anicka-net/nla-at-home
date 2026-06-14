"""Unit tests for train_universal_grpo_hard.py — the GPU-free pieces.

Covers the 2026-06-09 rewrite for the Phi-4 14B AV/AR pair: centered-cosine
reward, centered hard-negative index, phi_ar_stage2 value-head loading, and
generated-row stripping. The end-to-end GRPO loop needs a GPU and is not
covered here.
"""
import sys
import torch
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_universal_grpo_hard as grpo


# ---- centered_cosine ----

def test_centered_cosine_identity():
    mean = torch.zeros(4)
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = grpo.centered_cosine(v, v[0], mean)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(1.0, abs=1e-6)


def test_centered_cosine_removes_shared_mean():
    # pred and target share a huge mean offset; their centered parts are
    # orthogonal. Raw cosine would be ~1, centered must be ~0.
    mean = torch.full((4,), 100.0)
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    pred = (mean + a).unsqueeze(0)
    target = mean + b
    raw = torch.nn.functional.cosine_similarity(pred, target.unsqueeze(0)).item()
    centered = grpo.centered_cosine(pred, target, mean).item()
    assert raw > 0.99
    assert abs(centered) < 1e-5


def test_centered_cosine_matrix_shape():
    mean = torch.zeros(8)
    pred = torch.randn(3, 8)
    targets = torch.randn(5, 8)
    out = grpo.centered_cosine(pred, targets, mean)
    assert out.shape == (3, 5)
    # spot-check one entry against the 1-D path
    single = grpo.centered_cosine(pred[1:2], targets[2], mean)
    assert out[1, 2].item() == pytest.approx(single.item(), abs=1e-6)


# ---- strip_generated_row ----

def test_strip_generated_row_prompt_prefixed():
    prompt = [10, 11, 12]
    seq = torch.tensor(prompt + [20, 21, 22, 0])
    out = grpo.strip_generated_row(seq, prompt, eos_ids={0}, stop_ids=[])
    assert out == [20, 21, 22]


def test_strip_generated_row_generation_only():
    prompt = [10, 11, 12]
    seq = torch.tensor([20, 21, 22])
    out = grpo.strip_generated_row(seq, prompt, eos_ids={0}, stop_ids=[])
    assert out == [20, 21, 22]


def test_strip_generated_row_stop_sequence():
    prompt = [10]
    seq = torch.tensor([20, 21, 7, 8, 30, 31])
    out = grpo.strip_generated_row(seq, prompt, eos_ids={0}, stop_ids=[7, 8])
    assert out == [20, 21]


# ---- load_value_head_weights ----

def test_load_value_heads_pt_format(tmp_path):
    # phi_ar_stage2 format: {str(layer): weight}
    w13 = torch.randn(8, 8)
    w22 = torch.randn(8, 8)
    path = tmp_path / "stage2_v2mid_value_heads.pt"
    torch.save({"13": w13, "22": w22}, path)
    out = grpo.load_value_head_weights(path)
    assert set(out.keys()) == {13, 22}
    assert torch.equal(out[13], w13)
    assert torch.equal(out[22], w22)


# ---- build_hard_negative_index ----

def test_hard_negative_index_excludes_self_and_respects_k():
    torch.manual_seed(0)
    acts = {5: torch.randn(10, 6)}
    means = {5: acts[5].mean(0)}
    idx = grpo.build_hard_negative_index(acts, means, [5], n_texts=10, k=3)
    assert set(idx.keys()) == {5}
    assert len(idx[5]) == 10
    for i, neigh in enumerate(idx[5]):
        assert len(neigh) == 3
        assert i not in neigh


def test_hard_negative_index_is_centered_not_raw():
    # All vectors share a giant mean. After centering, item 0's nearest
    # neighbor is item 1 (same direction); raw cosine would rank everything
    # as ~equally similar (mean-dominated) so this is the discriminating case.
    mean_vec = torch.full((4,), 50.0)
    centered = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],   # nearest to item 0 after centering
        [-1.0, 0.0, 0.0, 0.0],  # farthest after centering
        [0.0, 1.0, 0.0, 0.0],
    ])
    acts = {0: mean_vec + centered}
    means = {0: acts[0].mean(0)}
    idx = grpo.build_hard_negative_index(acts, means, [0], n_texts=4, k=1)
    assert idx[0][0] == [1]


# ---- nearest_depth_pct / wrong-layer mapping ----

def test_nearest_depth_pct_phi4_ladder():
    # Phi-4: 40 layers, the v2mid AR ladder
    expected = {13: 32, 16: 40, 19: 47, 22: 55, 25: 63,
                28: 71, 32: 80, 36: 90, 38: 96}
    for layer, pct in expected.items():
        assert grpo.nearest_depth_pct(layer, 40) == pct


def test_wrong_layer_partner_is_farthest_trained_layer():
    ar_layers = [13, 16, 19, 22, 25, 28, 32, 36, 38]
    wrong = {L: max(ar_layers, key=lambda M: abs(M - L)) for L in ar_layers}
    assert wrong[13] == 38
    assert wrong[38] == 13
    assert wrong[25] == 38  # |25-13|=12 < |25-38|=13
    assert all(w != L for L, w in wrong.items())
