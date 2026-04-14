#!/usr/bin/env bash

set -euo pipefail

PID_PATH="${PID_PATH:-runtime/vllm-server.pid}"

if [[ ! -f "${PID_PATH}" ]]; then
  echo "No PID file found at ${PID_PATH}"
  exit 0
fi

PID="$(cat "${PID_PATH}")"
if [[ -z "${PID}" ]]; then
  echo "PID file is empty: ${PID_PATH}"
  exit 0
fi

if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}"
  echo "Sent SIGTERM to vLLM process ${PID}"
else
  echo "Process ${PID} is not running"
fi

rm -f "${PID_PATH}"
