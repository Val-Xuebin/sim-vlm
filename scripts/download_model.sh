#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${SIM_VLM_ENV_PREFIX:-$PROJECT_DIR/.venv}"
MODEL_ID="${1:-Qwen/Qwen3-VL-2B-Instruct}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Stale localhost proxy variables made direct downloads hang on this machine.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
mkdir -p "$HF_HOME"
exec "$ENV_PREFIX/bin/hf" download "$MODEL_ID"
