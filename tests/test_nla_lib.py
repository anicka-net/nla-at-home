"""Tests for nla_lib — the single source of truth for NLA constants,
templates, and adapter-format handling.

Two jobs:
1. Pin the FROZEN interfaces (templates, injection chars, scale semantics)
   byte-exactly. Shipped adapters were trained against these strings; a
   failing golden test means a silently-broken published checkpoint.
2. Scan the legacy scripts that still carry local copies and fail loudly if
   any copy drifts from the library definition (this is how we found an AV
   prompt that had silently lost its depth sentence).
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nla_lib  # noqa: E402


# ---------------------------------------------------------------------------
# Golden templates
# ---------------------------------------------------------------------------

GOLDEN_AV_PROMPT_47 = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context, "
    "along with the network depth where it was extracted. "
    "You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n\n"
    "Here is the vector from depth 47% of the network:\n\n"
    "<concept>㈎</concept>\n\n"
    "Please provide an explanation.\n\n"
    "<explanation>"
)


def test_av_prompt_golden():
    assert nla_lib.make_av_prompt(47, "㈎") == GOLDEN_AV_PROMPT_47


def test_ar_prompt_nodepth_golden():
    assert nla_lib.make_ar_prompt_nodepth("XYZ") == (
        "Summary of the following text: <text>XYZ</text> <summary>")


def test_ar_prompt_depth_sl_golden():
    assert nla_lib.make_ar_prompt_depth_sl("XYZ", 32, "㈎") == (
        "Summary of the following text from depth 32%: "
        "<text>XYZ</text> <summary>㈎")


def test_ar_depth_sl_matches_train_universal_ar():
    """The depth_sl template must produce exactly what
    train_universal_ar.py's make_ar_template trained the adapters on."""
    from train_universal_ar import make_ar_template
    template = make_ar_template(32, "㈎")
    ours = nla_lib.AR_TEMPLATE_DEPTH_SL.replace("{depth}", "32").replace(
        "{inj}", "㈎").replace("{explanation}", "{explanation}")
    assert template == ours


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_injection_chars():
    assert nla_lib.INJECTION_CHARS["qwen25-7b"] == "㈎"
    assert nla_lib.INJECTION_CHARS["qwen3-4b"] == "㈎"
    assert nla_lib.INJECTION_CHARS["gemma3-1b"] == "⎝"
    assert nla_lib.INJECTION_CHARS["phi4-mini"] == "★"
    assert nla_lib.INJECTION_CHARS["phi4"] == "★"
    assert "llama-8b" not in nla_lib.INJECTION_CHARS


def test_registry_hf_ids():
    assert nla_lib.MODELS_HF["qwen25-7b"] == "Qwen/Qwen2.5-7B-Instruct"
    assert nla_lib.MODELS_HF["phi4"] == "microsoft/phi-4"
    assert nla_lib.MODELS_HF["phi4-mini"] == "microsoft/Phi-4-mini-instruct"


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        nla_lib.get_model("gpt2")


# ---------------------------------------------------------------------------
# Depth mapping — the hardcoded-40 regression
# ---------------------------------------------------------------------------

def test_nearest_depth_pct_qwen_vs_phi4():
    # Same layer index, different models, different depths. Hardcoding
    # phi4's 40 into a qwen (28-layer) run shifted every depth prompt.
    assert nla_lib.nearest_depth_pct(14, 28) == 47
    assert nla_lib.nearest_depth_pct(14, 40) == 32


def test_nearest_depth_pct_bounds():
    assert nla_lib.nearest_depth_pct(0, 28) == nla_lib.DEPTH_PCTS[0]
    assert nla_lib.nearest_depth_pct(27, 28) == 96


# ---------------------------------------------------------------------------
# Injection semantics — normalize TO, never multiply BY
# ---------------------------------------------------------------------------

def test_normalize_activation_semantics():
    torch = pytest.importorskip("torch")
    v = torch.randn(3584) * 37.0
    out = nla_lib.normalize_activation(v, 150.0)
    assert abs(out.float().norm().item() - 150.0) < 1e-3
    # And the direction is preserved
    cos = torch.nn.functional.cosine_similarity(
        v.float().unsqueeze(0), out.float().unsqueeze(0)).item()
    assert cos > 0.9999


# ---------------------------------------------------------------------------
# AR format detection
# ---------------------------------------------------------------------------

def test_detect_ar_format(tmp_path):
    pt = tmp_path / "stage2_v2mid_best.pt"
    pt.write_bytes(b"x")
    assert nla_lib.detect_ar_format(pt) == nla_lib.AR_FORMAT_STAGE2_PT

    heads = tmp_path / "heads_ar"
    heads.mkdir()
    (heads / "value_heads.safetensors").write_bytes(b"x")
    assert nla_lib.detect_ar_format(heads) == nla_lib.AR_FORMAT_HEADS_DIR

    lora = tmp_path / "lora_ar"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}")
    assert nla_lib.detect_ar_format(lora) == nla_lib.AR_FORMAT_LORA_SL

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        nla_lib.detect_ar_format(empty)


def test_heads_dir_wins_over_adapter_config(tmp_path):
    """A dir with BOTH value heads and an adapter must be treated as
    heads_dir — the heads are the reconstruction interface there."""
    both = tmp_path / "both"
    both.mkdir()
    (both / "value_heads.safetensors").write_bytes(b"x")
    (both / "adapter_config.json").write_text("{}")
    assert nla_lib.detect_ar_format(both) == nla_lib.AR_FORMAT_HEADS_DIR


# ---------------------------------------------------------------------------
# Drift scan: legacy scripts carrying local copies must match the library
# ---------------------------------------------------------------------------

SCRIPTS = REPO_ROOT / "scripts"

# Scripts allowed to carry the AV prompt text literally: only the library
# itself. (brain_in_jar_qwen.py used to be exempt on the theory that it
# served an older prompt variant — its literals turned out to be
# byte-identical to nla_lib's, so it now imports them; 2026-07-05.)
AV_LITERAL_ALLOWED = {"nla_lib.py"}


def _scripts_with(pattern):
    hits = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == "nla_lib.py":
            continue
        text = f.read_text(errors="replace")
        if re.search(pattern, text):
            hits.append((f.name, text))
    return hits


def test_no_live_av_prompt_copies():
    """No live script may carry a literal copy of the AV prompt — copies
    drift (one lost its depth sentence before nla_lib existed). Import
    make_av_prompt from nla_lib instead."""
    offenders = [
        f.name for f in sorted(SCRIPTS.glob("*.py"))
        if f.name not in AV_LITERAL_ALLOWED
        and "meticulous AI researcher" in f.read_text(errors="replace")
    ]
    assert not offenders, (
        f"Literal AV prompt copies in live scripts: {offenders} — "
        f"import make_av_prompt from nla_lib instead")


def test_no_live_ar_template_copies():
    """Same rule for the AR summary templates."""
    offenders = [
        f.name for f in sorted(SCRIPTS.glob("*.py"))
        if f.name != "nla_lib.py"
        and re.search(r'"Summary of the following text', f.read_text(errors="replace"))
    ]
    assert not offenders, (
        f"Literal AR template copies in live scripts: {offenders} — "
        f"import from nla_lib instead")


def test_no_injection_char_dict_copies():
    """No live script may declare a model->injection-char mapping at all
    (consistency scans are not enough: a consistent copy still diverges the
    day the registry changes). Import INJECTION_CHARS from nla_lib."""
    pat = re.compile(
        r'"(qwen25-7b|qwen3-4b|gemma3-1b|phi4-mini|phi4)"\s*:\s*"(.)"')
    offenders = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == "nla_lib.py":
            continue
        for m in pat.finditer(f.read_text(errors="replace")):
            offenders.append((f.name, m.group(1), m.group(2)))
    assert not offenders, (
        f"Model->char dict literals in live scripts: {offenders} — "
        f"import INJECTION_CHARS from nla_lib instead")


def test_no_hf_id_dict_copies():
    """Same rule for model->HF-id mappings: import MODELS_HF (or
    INJECTABLE_MODELS_HF) from nla_lib."""
    pat = re.compile(
        r'"(qwen25-7b|qwen3-4b|gemma3-1b|phi4-mini|phi4|llama-8b)"'
        r'\s*:\s*"([A-Za-z0-9./_-]+/[A-Za-z0-9./_-]+)"')
    offenders = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == "nla_lib.py":
            continue
        for m in pat.finditer(f.read_text(errors="replace")):
            offenders.append((f.name, m.group(1), m.group(2)))
    assert not offenders, (
        f"Model->HF-id dict literals in live scripts: {offenders} — "
        f"import MODELS_HF from nla_lib instead")


def test_no_nla_lib_function_redeclarations():
    """No live script may re-declare a function that nla_lib owns. The drift
    scans above only catch string literals; this catches the 2026-07-05
    Seventh finding (nearest_depth_pct re-declared in train_universal_av.py,
    invisible to the template scans)."""
    owned = ("nearest_depth_pct", "normalize_activation", "make_av_prompt",
             "make_ar_prompt_nodepth", "make_ar_prompt_depth_sl",
             "detect_ar_format", "load_ar_lora_sl", "get_model",
             "read_nla_meta")
    pat = re.compile(r"^\s*def (" + "|".join(owned) + r")\s*\(", re.M)
    offenders = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == "nla_lib.py":
            continue
        for m in pat.finditer(f.read_text(errors="replace")):
            offenders.append((f.name, m.group(1)))
    assert not offenders, (
        f"nla_lib functions re-declared in live scripts: {offenders} — "
        f"import from nla_lib instead")


def test_no_trust_remote_heuristic():
    """trust_remote_code comes from get_model(key).trust_remote_code, not
    from substring heuristics on the model name (the '\"phi\" not in' rule
    was scattered across 5 scripts and contradicted the registry)."""
    offenders = [
        f.name for f in sorted(SCRIPTS.glob("*.py"))
        if f.name != "nla_lib.py"
        and re.search(r'''["'']phi["'']\s+(not\s+)?in\s''',
                      f.read_text(errors="replace"))
    ]
    assert not offenders, (
        f"trust_remote_code name-substring heuristic in: {offenders} — "
        f"use get_model(args.model).trust_remote_code")
