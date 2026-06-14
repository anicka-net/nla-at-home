"""Tests for corpus integrity — categories, generated texts, descriptions."""
import json
import re
import yaml
import pytest
import hashlib
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_DIR = REPO_ROOT / "corpus" / "categories"
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

REQUIRED_YAML_FIELDS = {"id", "name", "group", "count", "preamble", "batches"}
UNSAFE_CATEGORIES = {"F35_clearly_harmful", "F36_harmful_obfuscated",
                     "I44_emotional_manipulation", "L59_nsfw_explicit"}


def all_category_yamls():
    return sorted(CATEGORIES_DIR.glob("*.yaml"))


# Category files are named {GroupLetter}{number}_{name}.json. Other JSONs in
# generated/ are auxiliary collections (wildchat_*, expansion_*, texts_*) with
# their own id/category conventions — the category-shape tests below do not
# apply to them.
CATEGORY_FILE_RE = re.compile(r"^[A-L]\d+_.*\.json$")


def all_generated_jsons():
    return sorted(
        p for p in GENERATED_DIR.glob("*.json")
        if CATEGORY_FILE_RE.match(p.name)
    )


def all_description_files():
    return sorted(GENERATED_DIR.glob("descriptions_L*pct.json"))


def all_description_jsons():
    return sorted(GENERATED_DIR.glob("descriptions_L*pct*.json"))


def corpus_text_ids():
    ids = set()
    for path in all_generated_jsons():
        data = json.loads(path.read_text())
        ids.update(item["id"] for item in data)
    return ids


def merged_description_ids_by_depth():
    by_depth = {}
    for path in all_description_jsons():
        pct = int(path.stem.split("_L")[1].split("pct")[0])
        data = json.loads(path.read_text())
        by_depth.setdefault(pct, set()).update(item["id"] for item in data)
    return by_depth


# --- Category YAML tests ---

@pytest.mark.parametrize("path", all_category_yamls(), ids=lambda p: p.stem)
def test_yaml_parses(path):
    cat = yaml.safe_load(path.read_text())
    assert isinstance(cat, dict)


@pytest.mark.parametrize("path", all_category_yamls(), ids=lambda p: p.stem)
def test_yaml_has_required_fields(path):
    cat = yaml.safe_load(path.read_text())
    missing = REQUIRED_YAML_FIELDS - set(cat.keys())
    assert not missing, f"Missing fields: {missing}"


@pytest.mark.parametrize("path", all_category_yamls(), ids=lambda p: p.stem)
def test_yaml_id_matches_filename(path):
    cat = yaml.safe_load(path.read_text())
    assert cat["id"] == path.stem


@pytest.mark.parametrize("path", all_category_yamls(), ids=lambda p: p.stem)
def test_yaml_batches_have_instructions(path):
    cat = yaml.safe_load(path.read_text())
    for i, batch in enumerate(cat["batches"]):
        assert "instruction" in batch, f"Batch {i} missing instruction"


def test_unsafe_categories_flagged():
    for path in all_category_yamls():
        cat = yaml.safe_load(path.read_text())
        if cat["id"] in UNSAFE_CATEGORIES:
            assert cat.get("unsafe") is True, (
                f"{cat['id']} should have unsafe: true")
            assert cat.get("content_warning"), (
                f"{cat['id']} should have content_warning")


def test_category_ids_unique():
    ids = []
    for path in all_category_yamls():
        cat = yaml.safe_load(path.read_text())
        ids.append(cat["id"])
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    assert not dupes, f"Duplicate category IDs: {dupes}"


# --- Generated text tests ---

@pytest.mark.parametrize("path", all_generated_jsons(), ids=lambda p: p.stem)
def test_generated_json_valid(path):
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.parametrize("path", all_generated_jsons(), ids=lambda p: p.stem)
def test_generated_items_have_fields(path):
    data = json.loads(path.read_text())
    for item in data:
        assert "id" in item, f"Missing id in {path.stem}"
        assert "text" in item, f"Missing text in {path.stem}"
        assert "category" in item, f"Missing category in {path.stem}"


@pytest.mark.parametrize("path", all_generated_jsons(), ids=lambda p: p.stem)
def test_generated_ids_match_category(path):
    data = json.loads(path.read_text())
    expected_cat = path.stem
    for item in data:
        assert item["category"] == expected_cat, (
            f"Item {item['id']} has category {item['category']}, "
            f"expected {expected_cat}")


# Known cross-category collisions, left in place deliberately: the text is
# valid in both categories, and activations + descriptions are already keyed
# to both ids — removing or editing one would silently desync those files.
KNOWN_DUPLICATE_ID_SETS = [
    {"H40_ultra_short_001", "L54_ambiguous_underspecified_006"},  # "Help"
]


def test_no_exact_duplicate_texts_across_corpus():
    all_texts = []
    for path in all_generated_jsons():
        data = json.loads(path.read_text())
        for item in data:
            all_texts.append((item["id"], item["text"]))

    text_to_ids = {}
    for item_id, text in all_texts:
        text_to_ids.setdefault(text, []).append(item_id)
    dupes = [
        {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "ids": ids,
            "count": len(ids),
        }
        for text, ids in text_to_ids.items()
        if len(ids) > 1 and set(ids) not in KNOWN_DUPLICATE_ID_SETS
    ]
    assert not dupes, f"Duplicate texts found: {dupes[:5]}"


def test_all_categories_have_generated_texts():
    yaml_ids = set()
    for path in all_category_yamls():
        cat = yaml.safe_load(path.read_text())
        yaml_ids.add(cat["id"])

    generated_ids = {p.stem for p in all_generated_jsons()}
    # Unsafe categories' generated texts are withheld from the public repo
    # (see CORPUS.md "Public release"); only their YAML definitions ship, so
    # they are expected to have no generated-text file here.
    missing = yaml_ids - generated_ids - UNSAFE_CATEGORIES
    assert not missing, f"Categories without generated texts: {missing}"


def test_activation_ids_match_generated_corpus_when_complete():
    """Progress check: activation files should eventually cover corpus IDs."""
    import torch

    activation_dir = REPO_ROOT / "corpus" / "activations"
    activation_files = sorted(activation_dir.glob("*.pt"))
    if not activation_files:
        pytest.skip("No activation files extracted yet")

    corpus_ids = corpus_text_ids()
    problems = {}
    for path in activation_files:
        data = torch.load(path, weights_only=True, map_location="cpu")
        ids = set(data.get("ids", []))
        missing_n = len(corpus_ids - ids)
        stale_n = len(ids - corpus_ids)
        if missing_n or stale_n:
            problems[path.name] = (missing_n, stale_n)

    if problems:
        detail = ", ".join(
            f"{name}: missing={missing}, stale={stale}"
            for name, (missing, stale) in sorted(problems.items())
        )
        pytest.xfail(f"Activation extraction still in progress: {detail}")


# --- Description tests ---

@pytest.mark.parametrize("path", all_description_files(), ids=lambda p: p.stem)
def test_description_json_valid(path):
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.parametrize("path", all_description_files(), ids=lambda p: p.stem)
def test_descriptions_have_content(path):
    data = json.loads(path.read_text())
    fallback_count = sum(
        1 for d in data
        if d.get("description", "").startswith("Processing a ")
    )
    assert fallback_count < len(data) * 0.05, (
        f"{fallback_count}/{len(data)} fallback descriptions (>5%)")


@pytest.mark.parametrize("path", all_description_files(), ids=lambda p: p.stem)
def test_descriptions_not_empty(path):
    data = json.loads(path.read_text())
    empty = [d["id"] for d in data if not d.get("description", "").strip()]
    assert not empty, f"Empty descriptions: {empty[:10]}"


def test_description_ids_match_generated_corpus_when_complete():
    """Progress check: merged descriptions should eventually cover corpus IDs."""
    corpus_ids = corpus_text_ids()
    by_depth = merged_description_ids_by_depth()
    assert by_depth, "No description files found"

    problems = {}
    for pct, ids in by_depth.items():
        missing_n = len(corpus_ids - ids)
        stale_n = len(ids - corpus_ids)
        if missing_n or stale_n:
            problems[pct] = (missing_n, stale_n)

    if problems:
        detail = ", ".join(
            f"L{pct}%: missing={missing}, stale={stale}"
            for pct, (missing, stale) in sorted(problems.items())
        )
        pytest.xfail(f"Description generation still in progress: {detail}")
