#!/usr/bin/env bash
# Canonical launcher for the universal NLA pipeline (AV + AR).
#
# WHY THIS EXISTS: hand-rolling flags for train_universal_av.py is how the
# 2026-07-02 contamination happened (someone passed --mix without --strict and
# silently trained on _merged verbose prose). This script hardwires the SAFE,
# clean-data flags so nobody has to remember them. Prefer it over calling the
# python scripts directly.
#
# Usage:
#   scripts/train_universal.sh <model> <stage> [extra args...]
#     model : a key from the MODELS dict (qwen25-7b, phi4, phi4-mini, gemma3-1b, ...)
#     stage : av | ar
#
# Examples:
#   scripts/train_universal.sh qwen25-7b av
#   scripts/train_universal.sh qwen25-7b ar --mean-subtract
#
# The clean-data guard (scripts/clean_data_guard.py) still enforces at runtime;
# this script just makes the safe path the easy path. To train on verbose data
# on purpose (you almost never should), call the python script directly with
# --allow-verbose — this launcher will not do it for you.
set -euo pipefail

MODEL="${1:?usage: train_universal.sh <model> <av|ar> [extra args...]}"
STAGE="${2:?usage: train_universal.sh <model> <av|ar> [extra args...]}"
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTS="corpus/activations/${MODEL}_all_layers.pt"
cd "$REPO"

if [[ ! -f "$ACTS" ]]; then
  echo "ERROR: activations not found: $ACTS" >&2
  echo "Run first:  python3 scripts/extract_activations.py --model $MODEL --all-layers" >&2
  exit 1
fi

case "$STAGE" in
  av)
    echo ">> AV SFT (clean twin_clean, strict) for $MODEL"
    # ${PYTHON:-python3}: bare python3 crashed a 2026-07-07 launch on a host
    # where torch lives in a venv. -u: unbuffered — DESIGN.md known issue #2
    # (buffered logs look like a hang for the whole first epoch).
    exec "${PYTHON:-python3}" -u scripts/train_universal_av.py \
      --model "$MODEL" \
      --activations "$ACTS" \
      --desc-suffix _twin_clean --strict \
      --output "output/nla-${MODEL}-universal-av" \
      --epochs 5 --lr 8e-6 --batch-size 4 \
      "$@"
    ;;
  ar)
    echo ">> AR (clean tokenpred_gpt4o_clean) for $MODEL"
    exec "${PYTHON:-python3}" -u scripts/train_universal_ar.py \
      --model "$MODEL" \
      --activations "$ACTS" \
      --desc-suffix _tokenpred_gpt4o_clean \
      --output "output/nla-${MODEL}-universal-ar" \
      --epochs 5 --lr 7e-5 --contrastive-weight 1.0 --contrastive-temp 20 \
      "$@"
    ;;
  *)
    echo "ERROR: unknown stage '$STAGE' (expected av|ar)" >&2
    exit 1
    ;;
esac
