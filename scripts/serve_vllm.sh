#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
REQUEST_MODEL_NAME="${REQUEST_MODEL_NAME:-$MODEL_NAME_OR_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
ENABLE_LORA="${ENABLE_LORA:-0}"
LORA_ALIAS="${LORA_ALIAS:-support-ticket-ft}"
LORA_PATH="${LORA_PATH:-}"
MAX_LORAS="${MAX_LORAS:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-}"
PID_PATH="${PID_PATH:-runtime/vllm-server.pid}"

mkdir -p "$(dirname "$PID_PATH")"

echo "vLLM server startup"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "REQUEST_MODEL_NAME=${REQUEST_MODEL_NAME}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "GENERATION_CONFIG=${GENERATION_CONFIG}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "ENABLE_LORA=${ENABLE_LORA}"
if [[ "${ENABLE_LORA}" == "1" ]]; then
  echo "LORA_ALIAS=${LORA_ALIAS}"
  echo "LORA_PATH=${LORA_PATH}"
fi

command=(
  vllm serve "${MODEL_NAME_OR_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --generation-config "${GENERATION_CONFIG}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ "${ENABLE_LORA}" == "1" ]]; then
  if [[ -z "${LORA_PATH}" ]]; then
    echo "ENABLE_LORA=1 requires LORA_PATH" >&2
    exit 1
  fi
  command+=(--enable-lora --lora-modules "${LORA_ALIAS}=${LORA_PATH}")
  if [[ -n "${MAX_LORAS}" ]]; then
    command+=(--max-loras "${MAX_LORAS}")
  fi
  if [[ -n "${MAX_LORA_RANK}" ]]; then
    command+=(--max-lora-rank "${MAX_LORA_RANK}")
  fi
fi

printf '%s\n' "$$" > "${PID_PATH}"
echo "PID_PATH=${PID_PATH}"
echo "Executing: ${command[*]}"

exec "${command[@]}"
