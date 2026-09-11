#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "${PYTHON_BIN}" -c 'import torch, orjson' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m venv "${PACKAGE_ROOT}/.venv"
  PYTHON_BIN="${PACKAGE_ROOT}/.venv/bin/python"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${PACKAGE_ROOT}/portable_8gpu_requirements.txt"
fi

"${PYTHON_BIN}" "${PACKAGE_ROOT}/tools/selfcheck_portable_8gpu.py" --require-gpus 8
exec "${PYTHON_BIN}" "${PACKAGE_ROOT}/tools/run_8gpu_ppo.py" "$@"
