#!/bin/bash
# Multi-turn HARC v2 overnight training on idun
# Fixed: context 2048, lr 1e-4, r=32, response coupling
# Estimated: ~35-55s/step × 2000 = 19-31h (check smoke test timing)
set -euo pipefail
cd ~/multiturn_harc
source ~/venv/bin/activate

# Move v1 final aside for reference
if [ -d "runs/multiturn_harc/final" ]; then
    mv runs/multiturn_harc runs/multiturn_harc_v1 2>/dev/null || true
    echo "[info] v1 results moved to runs/multiturn_harc_v1/"
fi

echo "=== Smoke test (5 steps) ==="
python3 -u train.py --config config_idun_v2.yaml --smoke 2>&1 | tee runs_v2_smoke.log
echo ""
echo "=== Smoke timing ==="
tail -1 runs_v2_smoke.log
echo ""

read -p "Start full training? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo "=== Full v2 training (2000 steps) ==="
echo "Started at: $(date)"
nohup python3 -u train.py --config config_idun_v2.yaml > runs_v2_stdout.log 2>&1 &
echo "PID: $!"
echo "Monitor: tail -f ~/multiturn_harc/runs_v2_stdout.log"
echo "Log: tail -1 ~/multiturn_harc/runs/multiturn_harc_v2/train_log.jsonl"
