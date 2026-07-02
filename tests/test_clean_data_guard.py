"""Regression tests for the NLA clean-data guard (scripts/clean_data_guard.py).

These lock in the enforcement that prevents the 2026-07-02 contamination: a
training run that silently loaded _merged (verbose prose) + raw _sonnet. If
someone weakens the guard, these fail. No torch/GPU — pure logic, fast.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import clean_data_guard as g  # noqa: E402


# ----------------------------------------------------------- L1: filename check
CLEAN = [
    "descriptions_L71pct_twin_clean.json",
    "descriptions_L4pct_sonnet_clean.json",
    "descriptions_L96pct_tokenpred_gpt4o_clean.json",
    "descriptions_L47pct_stripped.json",
]
VERBOSE = [
    "descriptions_L71pct_merged.json",     # literary prose — the cardinal sin
    "descriptions_L71pct.json",            # unsuffixed base — also verbose
    "descriptions_L71pct_sonnet.json",     # raw sonnet, unfiltered
    "descriptions_L71pct_tokenpred_gpt4o.json",  # raw tokenpred, unfiltered
]


@pytest.mark.parametrize("name", CLEAN)
def test_clean_filenames_pass(name):
    assert g.is_clean_filename(name) is True


@pytest.mark.parametrize("name", VERBOSE)
def test_verbose_filenames_fail(name):
    assert g.is_clean_filename(name) is False


def test_raw_sonnet_not_confused_with_clean():
    # substring trap: "_sonnet" is in "_sonnet_clean" — make sure raw still fails
    assert g.is_clean_filename("descriptions_L10pct_sonnet.json") is False
    assert g.is_clean_filename("descriptions_L10pct_sonnet_clean.json") is True


def test_assert_clean_sources_exits_on_verbose():
    with pytest.raises(SystemExit) as e:
        g.assert_clean_sources(CLEAN + ["descriptions_L71pct_merged.json"],
                               allow_verbose=False)
    assert e.value.code == 1


def test_assert_clean_sources_passes_on_all_clean():
    # no exit, returns empty offender list
    assert g.assert_clean_sources(CLEAN, allow_verbose=False) == []


def test_allow_verbose_bypasses_filename_guard():
    # explicit opt-in: does not exit, reports the offenders
    bad = g.assert_clean_sources(["descriptions_L71pct_merged.json"],
                                 allow_verbose=True)
    assert bad == ["descriptions_L71pct_merged.json"]


# ------------------------------------------------------------ L2: content check
def _descs(median_len):
    text = "x" * median_len
    return {71: {f"id{i}": text for i in range(10)}}


def test_assert_terse_exits_on_prose():
    # merged/base run ~548 chars — well over the 400 threshold
    with pytest.raises(SystemExit) as e:
        g.assert_terse(_descs(548), allow_verbose=False)
    assert e.value.code == 1


def test_assert_terse_passes_on_bullets():
    # clean bullets run ~221 chars
    assert g.assert_terse(_descs(221), allow_verbose=False) == 221


def test_allow_verbose_bypasses_content_guard():
    assert g.assert_terse(_descs(548), allow_verbose=True) == 548


def test_empty_descriptions_no_crash():
    assert g.assert_terse({}, allow_verbose=False) is None
