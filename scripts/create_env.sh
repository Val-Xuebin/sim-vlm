#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${SIM_VLM_ENV_PREFIX:-$PROJECT_DIR/.venv}"
TMP_ROOT="${SIM_VLM_TMP_ROOT:-$PROJECT_DIR/.tmp}"
if [[ -n "${SIM_VLM_BOOTSTRAP_PYTHON:-}" ]]; then
  BOOTSTRAP_PYTHON="$SIM_VLM_BOOTSTRAP_PYTHON"
elif [[ -x /base/mambaforge/bin/python ]]; then
  BOOTSTRAP_PYTHON=/base/mambaforge/bin/python
else
  BOOTSTRAP_PYTHON="$(command -v python3)"
fi

if [[ -z "$BOOTSTRAP_PYTHON" || ! -x "$BOOTSTRAP_PYTHON" ]]; then
  echo "ERROR: Python 3.10 or 3.11 is required." >&2
  exit 2
fi

python_minor="$($BOOTSTRAP_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_minor" != "3.10" && "$python_minor" != "3.11" ]]; then
  echo "ERROR: Python 3.10 or 3.11 is required; found $python_minor." >&2
  echo "Set SIM_VLM_BOOTSTRAP_PYTHON to a compatible interpreter." >&2
  exit 2
fi

mkdir -p "$TMP_ROOT" "$PROJECT_DIR/.cache/huggingface" "$PROJECT_DIR/.cache/torch"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$ENV_PREFIX"
fi

export TMPDIR="$TMP_ROOT"
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir -e "$PROJECT_DIR[dev]"

if [[ "${INSTALL_LOCAL_VLM:-0}" == "1" ]]; then
  available_kb="$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')"
  required_kb=$((12 * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "ERROR: local Qwen3-VL setup needs at least 12 GiB free on the project filesystem." >&2
    echo "Available: $((available_kb / 1024 / 1024)) GiB at $PROJECT_DIR" >&2
    echo "Set SIM_VLM_ENV_PREFIX and HF_HOME to a filesystem with at least 15 GiB free." >&2
    exit 2
  fi
  "$ENV_PREFIX/bin/python" -m pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
  "$ENV_PREFIX/bin/python" -m pip install --no-cache-dir -e "$PROJECT_DIR[local,dev]"
fi

cat <<EOF
Environment ready.
Activate: source $ENV_PREFIX/bin/activate
Cache:    export HF_HOME=$PROJECT_DIR/.cache/huggingface
Smoke:    cd $PROJECT_DIR && vlm-sim demo --backend metadata --run-id smoke
GPU VLM:  cd $PROJECT_DIR && vlm-sim demo --backend qwen --run-id qwen-demo
Install GPU dependencies later: INSTALL_LOCAL_VLM=1 bash scripts/create_env.sh
EOF
