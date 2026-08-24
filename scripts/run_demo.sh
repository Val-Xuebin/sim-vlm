#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${SIM_VLM_ENV_PREFIX:-$PROJECT_DIR/.venv}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export AI2THOR_PLATFORM="${AI2THOR_PLATFORM:-Linux64}"
cd "$PROJECT_DIR"
if [[ "$AI2THOR_PLATFORM" == "Linux64" && -z "${DISPLAY:-}" ]]; then
  exec xvfb-run -a -s "-screen 0 1280x720x24" "$ENV_PREFIX/bin/vlm-sim" demo "$@"
fi
exec "$ENV_PREFIX/bin/vlm-sim" demo "$@"
