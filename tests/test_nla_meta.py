"""Tests for NLA metadata files in trained adapters."""
import yaml
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

REQUIRED_META_FIELDS = {"kind", "role", "d_model", "extraction_layer_index", "tokens"}
REQUIRED_TOKEN_FIELDS = {"injection_char", "injection_token_id"}


def all_meta_files():
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*/nla_meta.yaml"))


@pytest.mark.parametrize("path", all_meta_files(), ids=lambda p: p.parent.name)
def test_meta_has_required_fields(path):
    meta = yaml.safe_load(path.read_text())
    missing = REQUIRED_META_FIELDS - set(meta.keys())
    assert not missing, f"Missing fields: {missing}"


@pytest.mark.parametrize("path", all_meta_files(), ids=lambda p: p.parent.name)
def test_meta_tokens_complete(path):
    meta = yaml.safe_load(path.read_text())
    tokens = meta.get("tokens", {})
    missing = REQUIRED_TOKEN_FIELDS - set(tokens.keys())
    assert not missing, f"Missing token fields: {missing}"


@pytest.mark.parametrize("path", all_meta_files(), ids=lambda p: p.parent.name)
def test_meta_role_valid(path):
    meta = yaml.safe_load(path.read_text())
    assert meta["role"] in ("av", "ar"), f"Invalid role: {meta['role']}"


@pytest.mark.parametrize("path", all_meta_files(), ids=lambda p: p.parent.name)
def test_meta_has_prompt_template(path):
    meta = yaml.safe_load(path.read_text())
    templates = meta.get("prompt_templates", {})
    role = meta["role"]
    assert role in templates, f"No template for role '{role}'"
