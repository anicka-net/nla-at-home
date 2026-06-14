#!/bin/bash
# Periodic commit of growing description files.
# Run via: nohup bash scripts/periodic_commit.sh &
cd "$(dirname "$0")/.."

INTERVAL=${1:-3600}  # default: every hour

while true; do
    sleep "$INTERVAL"

    AZURE_COUNT=$(python3 -c "import json; print(len(json.load(open('corpus/generated/descriptions_gemma3_tokenpred_gpt4o_extra.json'))))" 2>/dev/null || echo "?")
    COPILOT_COUNT=$(python3 -c "import json; print(len(json.load(open('corpus/generated/descriptions_gemma3_copilot_sonnet.json'))))" 2>/dev/null || echo "?")

    git add corpus/generated/descriptions_gemma3_tokenpred_gpt4o_extra.json \
            corpus/generated/descriptions_gemma3_copilot_sonnet.json 2>/dev/null

    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "checkpoint: Azure GPT-4o ${AZURE_COUNT}, Copilot Sonnet ${COPILOT_COUNT} descriptions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        echo "$(date): committed (azure=${AZURE_COUNT}, copilot=${COPILOT_COUNT})"
    else
        echo "$(date): no changes"
    fi
done
