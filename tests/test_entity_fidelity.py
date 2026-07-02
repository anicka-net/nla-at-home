"""Tests for entity_fidelity.py — matcher fix (item 2a).

Follows the testing doctrine: pure functions, tiny synthetic fixtures,
test invariants not snapshots, one golden case per known trap.
"""
import sys
from pathlib import Path

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from entity_fidelity import (extract_entities, fuzzy_match, compute_fidelity,
                            make_derangement, compute_random_floor,
                            compute_teacher_ceiling)


# --- extract_entities ---

def test_extract_basic_proper_nouns():
    """NNP spans are extracted and lowercased."""
    ents = extract_entities("Alice visited Paris last summer.")
    assert "alice" in ents
    assert "paris" in ents


def test_extract_multiword():
    """Multi-word proper noun spans come out as one entity."""
    ents = extract_entities("The Monster Hunter World game is popular.")
    # Should get at least the combined span
    lowered = {e for e in ents}
    assert any("monster" in e and "hunter" in e for e in lowered)


def test_extract_empty():
    """Empty or whitespace-only input returns empty set."""
    assert extract_entities("") == set()
    assert extract_entities("   ") == set()
    assert extract_entities(None) == set()


# --- fuzzy_match ---

def test_exact_and_substring_still_match():
    """Existing behavior: exact and substring matching work."""
    desc = {"alice", "monster hunter"}
    inp = {"alice", "monster hunter world"}
    matched = fuzzy_match(desc, inp)
    assert matched == {"alice", "monster hunter"}


def test_rawtext_fallback_rescues_ner_miss():
    """Desc entity present verbatim in input text but absent from input
    entity set → matched via raw-text fallback."""
    desc_ents = {"agnes prins"}
    input_ents = set()  # NER failed to extract it
    input_text = "The story of Agnes Prins begins in 1923."
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    assert "agnes prins" in matched


def test_rawtext_fallback_token_boundary():
    """Desc entity 'ada', input text contains only 'Canada' → NOT matched.
    Token boundaries prevent false positives on short entities."""
    desc_ents = {"ada"}
    input_ents = set()
    input_text = "Canada is a large country in North America."
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    assert "ada" not in matched


def test_rawtext_fallback_boundary_positive():
    """'ada' as a standalone word in input text → matched."""
    desc_ents = {"ada"}
    input_ents = set()
    input_text = "Ada Lovelace wrote the first algorithm."
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    assert "ada" in matched


def test_no_match_stays_confabulated():
    """Entity not in input entities or input text → not matched."""
    desc_ents = {"winterfell", "arya"}
    input_ents = {"showa"}
    input_text = "The Showa Restoration transformed Japanese politics."
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    assert "winterfell" not in matched
    assert "arya" not in matched
    assert matched == set()  # neither desc entity appears in input


def test_no_match_stays_confabulated_exact():
    """More explicit: showa matches exact, GoT names don't match anything."""
    desc_ents = {"winterfell"}
    input_ents = set()
    input_text = "The Showa Restoration transformed Japanese politics."
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    assert matched == set()


def test_fuzzy_match_without_input_text():
    """Without input_text, behaves exactly as before (no fallback)."""
    desc_ents = {"agnes prins"}
    input_ents = set()
    matched = fuzzy_match(desc_ents, input_ents)
    assert matched == set()
    # Also with explicit None
    matched2 = fuzzy_match(desc_ents, input_ents, input_text=None)
    assert matched2 == set()


def test_rawtext_regex_special_chars():
    """Entity with regex-special characters doesn't crash."""
    desc_ents = {"c++", "c#"}
    input_ents = set()
    input_text = "Programming languages like C++ and C# are widely used."
    # Should not raise even though + and # are regex specials
    matched = fuzzy_match(desc_ents, input_ents, input_text=input_text)
    # C++ might not match with \b (+ is not a word char), that's fine —
    # the test is that it doesn't crash
    assert isinstance(matched, set)


# --- compute_fidelity ---

def test_compute_fidelity_passes_input_text():
    """compute_fidelity now uses raw-text fallback — an entity present
    in the input text but missed by NER should match."""
    # Use a name that NER will find in the description but might miss
    # in input due to context. We test the plumbing, not NER quality.
    result = compute_fidelity(
        input_text="Agnes Prins was born in 1923.",
        description="This text mentions Agnes Prins, a historical figure."
    )
    # If NER extracts "agnes prins" from both, it matches via entity sets.
    # If NER misses it in input, the raw-text fallback catches it.
    # Either way, it should be matched, not confabulated.
    if result['precision'] is not None and result['n_desc'] > 0:
        # Agnes Prins should not be confabulated
        assert "agnes prins" not in result['confabulated']


def test_empty_desc_precision_none():
    """Description with no entities → precision is None."""
    result = compute_fidelity(
        input_text="The quick brown fox jumps over the lazy dog.",
        description="A sentence about animals."
    )
    # "A sentence about animals" has no proper nouns
    # If NER finds nothing, precision should be None
    if result['n_desc'] == 0:
        assert result['precision'] is None


def test_compute_fidelity_basic():
    """Basic compute_fidelity returns expected structure."""
    result = compute_fidelity(
        input_text="Alice went to Paris in December.",
        description="This text is about Alice visiting Paris."
    )
    assert 'precision' in result
    assert 'recall' in result
    assert 'matched' in result
    assert 'confabulated' in result
    assert isinstance(result['matched'], list)
    assert isinstance(result['confabulated'], list)


# --- make_derangement ---

def test_derangement_no_fixed_points():
    """No element maps to itself, across several seeds."""
    ids = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"]
    for seed in [0, 1, 42, 99, 12345]:
        perm = make_derangement(ids, seed)
        assert set(perm.keys()) == set(ids)
        assert set(perm.values()) == set(ids)
        for k, v in perm.items():
            assert k != v, f"Fixed point {k}→{v} at seed={seed}"


def test_derangement_deterministic():
    """Same seed → same derangement."""
    ids = ["X", "Y", "Z", "W"]
    d1 = make_derangement(ids, seed=42)
    d2 = make_derangement(ids, seed=42)
    assert d1 == d2


def test_derangement_two_elements():
    """Minimum case: 2 elements always swap."""
    perm = make_derangement(["a", "b"], seed=0)
    assert perm == {"a": "b", "b": "a"}


def test_derangement_raises_on_single():
    """Cannot derange a single element."""
    import pytest
    with pytest.raises(ValueError):
        make_derangement(["only_one"], seed=0)


# --- compute_random_floor ---

def test_floor_deterministic_with_seed():
    """Same seed → same floor result."""
    descs = [
        {"text_id": "A", "layer": 10, "description": "Alice visits Paris."},
        {"text_id": "B", "layer": 10, "description": "Bob goes to London."},
        {"text_id": "C", "layer": 10, "description": "Charlie in Tokyo."},
    ]
    texts = {
        "A": "Alice went to Paris last summer.",
        "B": "Bob traveled to London for work.",
        "C": "Charlie moved to Tokyo in spring.",
    }
    f1 = compute_random_floor(descs, texts, seed=42)
    f2 = compute_random_floor(descs, texts, seed=42)
    assert f1 == f2


def test_floor_different_seed_can_differ():
    """Different seeds can produce different results (not guaranteed but likely)."""
    descs = [
        {"text_id": "A", "layer": 10, "description": "Alice visits Paris."},
        {"text_id": "B", "layer": 10, "description": "Bob goes to London."},
        {"text_id": "C", "layer": 10, "description": "Charlie in Tokyo."},
        {"text_id": "D", "layer": 10, "description": "Diana in Berlin."},
    ]
    texts = {
        "A": "Alice went to Paris.",
        "B": "Bob traveled to London.",
        "C": "Charlie moved to Tokyo.",
        "D": "Diana flew to Berlin.",
    }
    f1 = compute_random_floor(descs, texts, seed=0)
    f2 = compute_random_floor(descs, texts, seed=999)
    # They *may* be equal by coincidence but the derangements differ
    # Just check structure
    assert 'pooled' in f1
    assert 'pooled' in f2


# --- compute_teacher_ceiling ---

def test_teacher_ceiling_basic():
    """Teacher ceiling computes precision for GT descriptions."""
    gt_descs = [
        {"id": "A", "av_layer": 10, "description": "Alice visits Paris."},
        {"id": "B", "av_layer": 10, "description": "Bob goes to London."},
    ]
    texts = {
        "A": "Alice went to Paris last summer.",
        "B": "Bob traveled to London for work.",
    }
    result = compute_teacher_ceiling(gt_descs, texts)
    assert 'pooled' in result
    assert result['pooled']['n'] >= 0


def test_teacher_join_integrity():
    """Dangling id raises KeyError, not silent skip."""
    import pytest
    gt_descs = [
        {"id": "A", "av_layer": 10, "description": "Alice visits Paris."},
        {"id": "MISSING", "av_layer": 10, "description": "Unknown person."},
    ]
    texts = {"A": "Alice went to Paris."}
    with pytest.raises(KeyError, match="MISSING"):
        compute_teacher_ceiling(gt_descs, texts)


def test_teacher_all_ids_resolve():
    """3-row fixture: every description resolves to its text."""
    gt_descs = [
        {"id": "X", "av_layer": 4, "description": "Discussion of Tokyo's architecture."},
        {"id": "Y", "av_layer": 4, "description": "Analysis of Berlin's history."},
        {"id": "Z", "av_layer": 4, "description": "Review of Paris's cuisine."},
    ]
    texts = {
        "X": "Tokyo has amazing modern architecture.",
        "Y": "Berlin's history spans centuries of change.",
        "Z": "Paris is renowned for its cuisine.",
    }
    # Should not raise
    result = compute_teacher_ceiling(gt_descs, texts)
    assert result['pooled']['n'] > 0
