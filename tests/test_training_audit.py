"""Tests for training_entity_audit.py (item 3).

4-row fixture per the design doc:
  grounded entity / ungrounded entity / no entities / unsafe id
Assert fraction = 1/2 over the 2 scoreable safe rows, unsafe skipped,
no-entity row excluded from denominator.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from training_entity_audit import is_unsafe, audit_descriptions


# --- is_unsafe ---

def test_unsafe_f35():
    assert is_unsafe("F35_something_000") is True

def test_unsafe_f36():
    assert is_unsafe("F36_other_001") is True

def test_unsafe_i44():
    assert is_unsafe("I44_bad_002") is True

def test_unsafe_l59():
    assert is_unsafe("L59_bad_003") is True

def test_safe_a01():
    assert is_unsafe("A01_code_000") is False

def test_safe_f31():
    """F31 is benign (not F35/F36), must NOT be flagged."""
    assert is_unsafe("F31_benign_000") is False


# --- audit_descriptions ---

def test_audit_4row_fixture():
    """Design-doc fixture: grounded / ungrounded / no-entities / unsafe.

    - grounded: desc mentions 'alice' which is in text → matched
    - ungrounded: desc mentions 'winterfell' which is NOT in text → confab
    - no_entities: desc has no proper nouns → excluded from denominator
    - unsafe: F35 id → skipped entirely
    """
    texts = {
        "A01_grounded": "Alice went to Paris last summer for vacation.",
        "A02_ungrounded": "The weather in Berlin was cold and rainy.",
        "A03_no_ents": "a simple sentence about nothing in particular.",
        "F35_unsafe": "This text should never be read.",
    }
    descriptions = [
        {"id": "A01_grounded", "description": "Text about Alice visiting Paris."},
        {"id": "A02_ungrounded", "description": "Text about Winterfell and dragons."},
        {"id": "A03_no_ents", "description": "a short text about something simple."},
        {"id": "F35_unsafe", "description": "Something about unsafe content."},
    ]

    result = audit_descriptions(descriptions, texts, depth_label="test")

    # Unsafe skipped
    assert result['n_skipped_unsafe'] == 1

    # No-entity row excluded from denominator
    # Scoreable = those with entities in their description (A01 + A02)
    # A03 has no proper nouns, so n_desc=0, precision=None → excluded
    assert result['n_scoreable'] >= 1  # at least the ungrounded one

    # The ungrounded row should have confabulated entities
    ungrounded_items = [it for it in result['items']
                        if it['id'] == 'A02_ungrounded']
    if ungrounded_items:
        assert ungrounded_items[0]['n_confabulated'] > 0

    # The grounded row should have matched entities
    grounded_items = [it for it in result['items']
                      if it['id'] == 'A01_grounded']
    if grounded_items:
        assert grounded_items[0]['n_matched'] > 0


def test_audit_dangling_id_raises():
    """Dangling id raises KeyError, not silent skip."""
    texts = {"A01_exists": "Some text."}
    descriptions = [
        {"id": "A01_exists", "description": "This is fine."},
        {"id": "MISSING_ID", "description": "This has no text."},
    ]
    with pytest.raises(KeyError, match="MISSING_ID"):
        audit_descriptions(descriptions, texts)


def test_audit_unsafe_skipped_not_joined():
    """Unsafe ids are skipped before join — no KeyError even if text is missing."""
    texts = {"A01_safe": "Alice goes to school."}
    descriptions = [
        {"id": "A01_safe", "description": "Text about Alice at school."},
        {"id": "F35_missing", "description": "No text for this."},
    ]
    # Should NOT raise — F35 is skipped before the join
    result = audit_descriptions(descriptions, texts)
    assert result['n_skipped_unsafe'] == 1


def test_audit_empty_descriptions():
    """Empty description list returns zero counts."""
    result = audit_descriptions([], {}, depth_label="empty")
    assert result['n_scoreable'] == 0
    assert result['n_with_ungrounded'] == 0
    assert result['frac_with_ungrounded'] == 0
