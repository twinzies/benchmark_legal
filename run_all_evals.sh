#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/closed_source_eval.py"
MODE="full-run"

MODELS=(
    "GPT-5.5"
    "o4"
    "Claude-Opus"
    "Gemini-flash"
    "SaulLM-141B"
    "Mixtral-141B"
    "GLM-5.2"
    "Qwen3.6-35B"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=============================="
    echo "Running: $MODEL ($MODE)"
    echo "=============================="
    python "$EVAL_SCRIPT" --model "$MODEL" --mode "$MODE"
done

echo ""
echo "All runs complete."
