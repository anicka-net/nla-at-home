"""Tests for activation files and augmented data."""
import torch
import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"
AUGMENTED_DIR = REPO_ROOT / "corpus" / "augmented"


def all_activation_files():
    if not ACTIVATIONS_DIR.exists():
        return []
    return sorted(ACTIVATIONS_DIR.glob("*.pt"))


def all_augmented_files():
    if not AUGMENTED_DIR.exists():
        return []
    return sorted(AUGMENTED_DIR.glob("*.pt"))


REQUIRED_SINGLE_KEYS = {"activations", "ids", "model", "layer", "d_model", "n_texts"}
REQUIRED_ALLLAYER_KEYS = {"activations", "ids", "model", "n_layers", "d_model", "n_texts"}


def is_all_layers(data):
    return "n_layers" in data and isinstance(data["activations"], dict)


@pytest.mark.parametrize("path", all_activation_files(), ids=lambda p: p.stem)
def test_activation_file_has_required_keys(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    required = REQUIRED_ALLLAYER_KEYS if is_all_layers(data) else REQUIRED_SINGLE_KEYS
    missing = required - set(data.keys())
    assert not missing, f"Missing keys: {missing}"


@pytest.mark.parametrize("path", all_activation_files(), ids=lambda p: p.stem)
def test_activation_shapes_consistent(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    ids = data["ids"]
    if is_all_layers(data):
        for layer_idx, acts in data["activations"].items():
            assert acts.shape[0] == len(ids), (
                f"Layer {layer_idx}: count {acts.shape[0]} != id count {len(ids)}")
            assert acts.shape[1] == data["d_model"], (
                f"Layer {layer_idx}: d_model {acts.shape[1]} != {data['d_model']}")
    else:
        acts = data["activations"]
        assert acts.shape[0] == len(ids), (
            f"Activation count {acts.shape[0]} != id count {len(ids)}")
        assert acts.shape[1] == data["d_model"], (
            f"d_model mismatch: tensor {acts.shape[1]} != metadata {data['d_model']}")


def _get_all_acts(data):
    if is_all_layers(data):
        return torch.cat(list(data["activations"].values()))
    return data["activations"]


@pytest.mark.parametrize("path", all_activation_files(), ids=lambda p: p.stem)
def test_activation_norms_reasonable(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    acts = _get_all_acts(data)
    norms = acts.norm(dim=1)
    assert norms.min() > 1.0, f"Suspiciously small norm: {norms.min():.2f}"
    assert norms.max() < 50000, f"Suspiciously large norm: {norms.max():.2f}"  # Gemma-3 late-layer norms legitimately reach ~16k


@pytest.mark.parametrize("path", all_activation_files(), ids=lambda p: p.stem)
def test_activation_no_nans(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    acts = _get_all_acts(data)
    assert not torch.isnan(acts).any(), "NaN values in activations"
    assert not torch.isinf(acts).any(), "Inf values in activations"


@pytest.mark.parametrize("path", all_augmented_files(), ids=lambda p: p.stem)
def test_augmented_vectors_and_metas_match(path):
    data = torch.load(path, weights_only=False, map_location="cpu")
    vectors = data["vectors"]
    metas = data["metas"]
    assert len(vectors) == len(metas), (
        f"Vector count {len(vectors)} != meta count {len(metas)}")


@pytest.mark.parametrize("path", all_augmented_files(), ids=lambda p: p.stem)
def test_augmented_has_descriptions(path):
    data = torch.load(path, weights_only=False, map_location="cpu")
    metas = data["metas"]
    described = sum(1 for m in metas if m.get("description"))
    assert described > len(metas) * 0.9, (
        f"Only {described}/{len(metas)} augmented examples have descriptions")


@pytest.mark.parametrize("path", all_augmented_files(), ids=lambda p: p.stem)
def test_augmented_contrastive_norms(path):
    data = torch.load(path, weights_only=False, map_location="cpu")
    vectors = data["vectors"]
    mean_norm = data.get("mean_norm", 100)
    n_cont = data.get("n_contrastive", 0)
    if n_cont > 0:
        cont_norms = vectors[:n_cont].norm(dim=1)
        ratio = cont_norms.mean() / mean_norm
        assert 0.8 < ratio < 1.2, (
            f"Contrastive norms {cont_norms.mean():.1f} too far from "
            f"target {mean_norm:.1f} (ratio {ratio:.2f})")
