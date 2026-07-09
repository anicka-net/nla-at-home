#!/bin/bash
# Launch multi-turn HARC training on idun (M4 Max, MPS)
#
# Usage:
#   ./run_idun.sh              # full training (2000 steps)
#   ./run_idun.sh --smoke      # 5-step smoke test
#   ./run_idun.sh --max_steps 100  # short run
#
# Prerequisites:
#   1. Run prepare_data.py first to download and format the dataset
#   2. Qwen 2.5 7B must be cached in HF hub (auto-downloads on first run)

set -euo pipefail
cd "$(dirname "$0")"

# Activate venv
source ~/venv/bin/activate

# Step 1: Prepare data if needed
if [ ! -f data/train_harmful.jsonl ]; then
    echo "=== Preparing training data ==="
    python3 prepare_data.py
fi

# Step 2: Train
echo "=== Starting multi-turn HARC training ==="
python3 train.py --config config_idun.yaml "$@"
