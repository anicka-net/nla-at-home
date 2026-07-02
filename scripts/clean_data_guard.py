#!/usr/bin/env python3
"""
Data-integrity guard for the NLA training loaders.

THE CARDINAL RULE (see the nla-training skill and DESIGN.md):
    NEVER train an NLA on verbose-prose descriptions.

`_merged.json` (literary prose: "the model is humming with focused
pattern-matching activity"), the unsuffixed base `descriptions_L{pct}pct.json`,
and raw `_sonnet.json` / `_tokenpred_gpt4o.json` (unfiltered) all teach a small
model a *template style* instead of matching descriptions to activations. The
result is fluent hallucination — the SpongeBob failure (Qwen, 2026-06-01) and
the "Chinese-language response tokens active" failure (Phi-4, 2026-06-03). Both
were GREEN ON PAPER: contaminated models train to a low loss and read fluently.
Only `*_twin_clean` / `*_sonnet_clean` / `*_tokenpred_gpt4o_clean` / `*_stripped`
are safe.

This module makes that rule enforceable in CODE, not just documented, because a
rule that lives only in prose gets skipped under load. It happened once
(2026-07-02: a `--mix` run without `--strict` silently pulled `_merged` +
raw `_sonnet` at every depth). The fix is to disarm the tool, not to add
another sign next to the footgun.

Two layers:
  L1  assert_clean_sources()  — checks the FILENAMES the loader chose.
  L2  assert_terse()          — checks the CONTENT it actually loaded, so a
                                novel bad file (clean-looking name, prose text)
                                still gets caught.

Both are bypassable only with an explicit `allow_verbose=True` (wired to a
`--allow-verbose` CLI flag) — the safe path is the default, the dangerous path
is loud and opt-in.
"""
import sys
import statistics

# Suffixes that mark a KNOWN-CLEAN description file (terse bullets).
CLEAN_MARKERS = (
    "_twin_clean", "_sonnet_clean", "_tokenpred_gpt4o_clean", "_stripped",
)

# Median char length above which descriptions look like prose, not bullets.
# Calibrated 2026-07-02 on real files at L71/L47:
#   clean _stripped 221 · clean _v2 286 · verbose _merged/base 548 · raw _sonnet 1005
VERBOSE_LEN_THRESHOLD = 400

_FIX_HINT = (
    "How to fix:\n"
    "    AV:  --desc-suffix _twin_clean --strict          (and DROP --mix)\n"
    "    AR:  --desc-suffix _tokenpred_gpt4o_clean\n"
    "If you REALLY intend verbose/raw prose (you almost never do), pass\n"
    "    --allow-verbose\n"
    "to acknowledge it explicitly and bypass this guard."
)


def is_clean_filename(name):
    """Default-DENY: a description file is clean only if it carries a clean
    marker and no verbose marker. Base `descriptions_L{pct}pct.json`, `_merged`,
    and raw `_sonnet` / `_tokenpred_gpt4o` (without `_clean`) all return False."""
    if "_merged" in name:
        return False
    # raw sonnet / raw tokenpred: the family name present but NOT the _clean form
    if "_sonnet" in name and "_sonnet_clean" not in name:
        return False
    if "_tokenpred_gpt4o" in name and "_tokenpred_gpt4o_clean" not in name:
        return False
    return any(marker in name for marker in CLEAN_MARKERS)


def _die(title, detail):
    bar = "=" * 72
    print(f"\n{bar}\nREFUSING TO TRAIN: {title}\n{bar}", file=sys.stderr)
    print(detail, file=sys.stderr)
    print(f"\n{_FIX_HINT}\n{bar}", file=sys.stderr)
    sys.exit(1)


def assert_clean_sources(source_names, allow_verbose):
    """Layer 1 — filename check. `source_names` = iterable of the base filenames
    the loader actually opened. Exits non-zero if any is verbose/raw and
    `allow_verbose` is False. Returns the list of offenders (for logging)."""
    bad = sorted({n for n in source_names if not is_clean_filename(n)})
    if bad and not allow_verbose:
        listing = "\n".join(f"    {n}" for n in bad)
        _die(
            "verbose / unfiltered description files were selected.",
            "These are literary prose, not terse bullets — training on them\n"
            "produces fluent hallucination (the SpongeBob / Chinese-tokens\n"
            f"failures). Offending files:\n{listing}",
        )
    if bad:
        print(f"[clean_data_guard] --allow-verbose set; training on VERBOSE "
              f"files anyway: {bad}", file=sys.stderr)
    return bad


def assert_terse(descriptions_by_depth, allow_verbose):
    """Layer 2 — content net. `descriptions_by_depth` = {depth_pct: {id: text}}.
    Catches a file that passed the filename check but holds prose. Exits
    non-zero if the median description length exceeds VERBOSE_LEN_THRESHOLD.
    Returns the observed median (or None if empty)."""
    lengths = [len(text)
               for depth_map in descriptions_by_depth.values()
               for text in depth_map.values()]
    if not lengths:
        return None
    median = statistics.median(lengths)
    if median > VERBOSE_LEN_THRESHOLD and not allow_verbose:
        _die(
            "descriptions look like verbose prose (content check).",
            f"Median description length = {median:.0f} chars; clean bullets are\n"
            f"below {VERBOSE_LEN_THRESHOLD}. A file passed the filename check but\n"
            "its text is long-form prose — this is the content net firing.",
        )
    if median > VERBOSE_LEN_THRESHOLD:
        print(f"[clean_data_guard] --allow-verbose set; median length "
              f"{median:.0f} > {VERBOSE_LEN_THRESHOLD} (prose) allowed anyway.",
              file=sys.stderr)
    return median
