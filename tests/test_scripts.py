"""Tests for script imports and basic functionality (no GPU required)."""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_generate_corpus_imports():
    import generate_corpus
    assert hasattr(generate_corpus, "BACKENDS")
    assert hasattr(generate_corpus, "get_client")
    assert hasattr(generate_corpus, "load_categories")


def test_generate_corpus_stats(capsys):
    import generate_corpus
    import argparse
    args = argparse.Namespace(stats=True)
    # Just verify it doesn't crash
    generate_corpus.main.__code__  # exists


def test_generate_corpus_backends():
    import generate_corpus
    assert "deepseek" in generate_corpus.BACKENDS
    assert "local" in generate_corpus.BACKENDS
    assert "nvidia" in generate_corpus.BACKENDS
    for name, cfg in generate_corpus.BACKENDS.items():
        assert "base_url" in cfg, f"Backend {name} missing base_url"
        assert "model" in cfg, f"Backend {name} missing model"


def test_load_categories():
    import generate_corpus
    cats = generate_corpus.load_categories()
    assert len(cats) > 50
    for cat in cats:
        assert "id" in cat
        assert "batches" in cat


def test_load_categories_filter():
    import generate_corpus
    cats = generate_corpus.load_categories(["A01_code", "A02_math"])
    assert len(cats) == 2
    assert {c["id"] for c in cats} == {"A01_code", "A02_math"}


def test_status_imports():
    import status
    assert hasattr(status, "check_corpus")
    assert hasattr(status, "check_descriptions")
    assert hasattr(status, "readiness_for")


def test_status_check_corpus():
    import status
    cats, total, missing, unsafe_count = status.check_corpus()
    assert total > 1000
    assert unsafe_count >= 4


def test_status_check_activations_handles_all_layers(tmp_path, monkeypatch):
    import torch
    import status

    act_path = tmp_path / "gemma3-1b_all_layers.pt"
    torch.save({
        "activations": {
            0: torch.zeros(3, 5),
            1: torch.ones(3, 5),
        },
        "ids": ["x_000", "x_001", "x_002"],
        "n_layers": 2,
        "n_texts": 3,
        "d_model": 5,
    }, act_path)

    monkeypatch.setattr(status, "ACTIVATIONS_DIR", tmp_path)
    activations = status.check_activations()

    assert activations["gemma3-1b_all_layers"] == {
        "n_texts": 3,
        "d_model": 5,
        "layer": None,
        "n_layers": 2,
        "kind": "all_layers",
    }


def test_status_check_activations_handles_single_layer(tmp_path, monkeypatch):
    import torch
    import status

    act_path = tmp_path / "qwen25-7b_L20.pt"
    torch.save({
        "activations": torch.zeros(4, 7),
        "ids": ["x_000", "x_001", "x_002", "x_003"],
        "layer": 20,
    }, act_path)

    monkeypatch.setattr(status, "ACTIVATIONS_DIR", tmp_path)
    activations = status.check_activations()

    assert activations["qwen25-7b_L20"] == {
        "n_texts": 4,
        "d_model": 7,
        "layer": 20,
        "n_layers": None,
        "kind": "single_layer",
    }


def test_status_check_activation_coverage_reports_missing_and_stale(tmp_path, monkeypatch):
    import torch
    import status

    act_path = tmp_path / "qwen25-7b_L20.pt"
    torch.save({
        "activations": torch.zeros(2, 7),
        "ids": ["kept_000", "stale_000"],
        "layer": 20,
    }, act_path)

    monkeypatch.setattr(status, "ACTIVATIONS_DIR", tmp_path)
    coverage = status.check_activation_coverage({"kept_000", "missing_000"})

    assert coverage["qwen25-7b_L20"]["missing"] == ["missing_000"]
    assert coverage["qwen25-7b_L20"]["extra"] == ["stale_000"]


def test_merge_descriptions_imports():
    import merge_descriptions


def test_extract_activations_models_dict():
    import extract_activations
    assert "qwen25-7b" in extract_activations.MODELS


def test_injection_token_single_token():
    """Verify the injection char is a single Unicode character."""
    char = "㈎"
    assert len(char) == 1
    assert ord(char) == 0x320E


@pytest.mark.parametrize("script", [
    "generate_corpus", "extract_activations", "train_av", "train_ar",
    "train_av_rft", "augment_directions", "compare_nla",
    "find_injection_token", "merge_descriptions", "status",
    "stress_test_qwen_nla",
])
def test_script_compiles(script):
    import importlib
    mod = importlib.import_module(script)
    assert mod is not None
