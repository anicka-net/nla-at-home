#!/bin/bash
# Full pipeline: merge → sync → extract → train AV → train AR → stress test
# Designed to run unattended after WildChat descriptions finish.
set -eo pipefail

REPO="$HOME/playground/nla-at-home"
REMOTE="deepthought"
REMOTE_REPO="playground/nla-at-home"
LOG="/tmp/expanded_pipeline.log"
PYTHON="~/venv/bin/python3"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_remote() {
    log "  Running: $1"
    ssh "$REMOTE" "cd ~/$REMOTE_REPO && $PYTHON $1" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        log "ERROR: Remote command failed with exit code $rc"
        exit "$rc"
    fi
}

log "=== EXPANDED PIPELINE START ==="

# Step 1: Merge
log "Step 1: Merging corpus..."
cd "$REPO"
python3 scripts/merge_expanded_corpus.py 2>&1 | tee -a "$LOG"

MERGED=$(python3 -c "import json; print(len(json.load(open('corpus/generated/descriptions_L71pct_expanded.json'))))")
log "Merged descriptions: $MERGED"
if [ "$MERGED" -lt 2000 ]; then
    log "ERROR: Only $MERGED descriptions merged, expected 2000+. Aborting."
    exit 1
fi

# Step 2: Sync
log "Step 2: Syncing to deepthought..."
rsync -avz "$REPO/corpus/generated/" "$REMOTE:~/$REMOTE_REPO/corpus/generated/" 2>&1 | tee -a "$LOG"
rsync -avz "$REPO/scripts/" "$REMOTE:~/$REMOTE_REPO/scripts/" 2>&1 | tee -a "$LOG"

# Step 3: Extract activations
log "Step 3: Extracting activations on deepthought (Qwen 2.5 7B, L20)..."
run_remote "scripts/extract_activations.py --model qwen25-7b --layer 20"

log "Syncing activations back..."
rsync -avz "$REMOTE:~/$REMOTE_REPO/corpus/activations/" "$REPO/corpus/activations/" 2>&1 | tee -a "$LOG"

# Step 4: Train AV
log "Step 4: Training AV on expanded dataset..."
run_remote "scripts/train_av_single_layer.py \
    --model qwen25-7b \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --descriptions corpus/generated/descriptions_L71pct_expanded.json \
    --output output/nla-qwen25-7b-L20-av-sonnet-expanded \
    --epochs 5 --lr 8e-6 --lora-r 16 --batch-size 4"

# Step 5: Train AR
log "Step 5: Training AR on expanded dataset..."
run_remote "scripts/train_ar.py \
    --model qwen25-7b --layer 20 \
    --description-file corpus/generated/descriptions_L71pct_expanded.json \
    --output output/nla-qwen25-7b-L20-ar-sonnet-expanded \
    --epochs 10 --lr 7e-5 --lora-r 16 --batch-size 4"

# Step 6: Stress test
log "Step 6: Stress test..."
run_remote "scripts/stress_test_nla.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-L20-av-sonnet-expanded \
    --ar-adapter output/nla-qwen25-7b-L20-ar-sonnet-expanded \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --target-layer 20 --n-samples 50"

log "=== EXPANDED PIPELINE COMPLETE ==="
log "Full log: $LOG"
