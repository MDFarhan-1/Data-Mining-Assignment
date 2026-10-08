#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if [[ -x "/tmp/dmdm-gpu-venv/bin/python" ]]; then
	PYTHON_BIN="/tmp/dmdm-gpu-venv/bin/python"
	GPU_LIBS="$(find /tmp/dmdm-gpu-venv/lib64/python3.11/site-packages/nvidia -type d -name lib -print | paste -sd: -)"
	export LD_LIBRARY_PATH="${GPU_LIBS}:${LD_LIBRARY_PATH:-}"
elif [[ -x ".venv/bin/python" ]]; then
	PYTHON_BIN=".venv/bin/python"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
mkdir -p outputs/logs
nohup "$PYTHON_BIN" run_all.py > outputs/logs/run_all.log 2>&1 &
echo "Started PID $!; follow with: tail -f outputs/logs/run_all.log"