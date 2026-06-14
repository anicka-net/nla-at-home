#!/bin/bash
# Pipeline: sync → train AV → train AR → stress test
# All-Sonnet descriptions, 5540 examples
set -eo pipefail

REPO="$HOME/playground/nla-at-home"
REMOTE="deepthought"
REMOTE_REPO="playground/nla-at-home"
LOG="/tmp/sonnet_pipeline.log"
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

log "=== ALL-SONNET PIPELINE START ==="

# Step 1: Sync
log "Step 1: Syncing to deepthought..."
rsync -avz "$REPO/corpus/generated/descriptions_L71pct_all_sonnet.json" "$REMOTE:~/$REMOTE_REPO/corpus/generated/" 2>&1 | tee -a "$LOG"
rsync -avz "$REPO/scripts/" "$REMOTE:~/$REMOTE_REPO/scripts/" 2>&1 | tee -a "$LOG"

# Step 2: Train AV (LoRA r=16, 5 epochs)
log "Step 2: Training AV on all-Sonnet dataset (5540 examples)..."
run_remote "scripts/train_av_single_layer.py \
    --model qwen25-7b \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --descriptions corpus/generated/descriptions_L71pct_all_sonnet.json \
    --output output/nla-qwen25-7b-L20-av-all-sonnet \
    --epochs 5 --lr 8e-6 --lora-r 16 --batch-size 4"

# Step 3: Train AR
log "Step 3: Training AR on all-Sonnet dataset..."
run_remote "scripts/train_ar.py \
    --model qwen25-7b --layer 20 \
    --description-file corpus/generated/descriptions_L71pct_all_sonnet.json \
    --output output/nla-qwen25-7b-L20-ar-all-sonnet \
    --epochs 10 --lr 7e-5 --lora-r 16 --batch-size 4"

# Step 4: Stress test
log "Step 4: Stress test..."
run_remote "scripts/stress_test_nla.py \
    --model qwen25-7b \
    --av-adapter output/nla-qwen25-7b-L20-av-all-sonnet \
    --ar-adapter output/nla-qwen25-7b-L20-ar-all-sonnet \
    --activations corpus/activations/qwen25-7b_L20.pt \
    --target-layer 20 --n-samples 100"

log "=== ALL-SONNET PIPELINE COMPLETE ==="
log "Full log: $LOG"
