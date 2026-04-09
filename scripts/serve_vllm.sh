#!/usr/bin/env bash

# Scaffold launcher for a future vLLM serving workflow.
# Keep orchestration thin here and place reusable logic in src/json_ft/inference.py.

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-placeholder-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/drive/MyDrive/json-ft-runs}"

cat <<EOF
vLLM serving scaffold
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
HOST=${HOST}
PORT=${PORT}
RUNTIME_ROOT=${RUNTIME_ROOT}

This path is intended as the secondary Colab serving mode.
The recommended primary benchmark path remains in-process vLLM execution.
EOF

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME_OR_PATH}" \
  --host "${HOST}" \
  --port "${PORT}"

