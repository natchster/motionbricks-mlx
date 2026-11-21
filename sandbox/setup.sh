#!/usr/bin/env bash
# One-shot bootstrap of the CUDA-emulation sandbox + reference oracle. Idempotent.
#
#   sandbox/setup.sh            # env, upstream @ pinned commit, checkpoints, ONNX export, golden fixtures
#   sandbox/setup.sh --no-golden --no-onnx
#
# Everything lands inside the repo (.venv/, sandbox/upstream/, sandbox/assets/,
# sandbox/golden/, checkpoints/) and every download is SHA-256 verified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

onnx=1 golden=1
for arg in "$@"; do
  case "$arg" in
    --no-onnx) onnx=0 ;;
    --no-golden) golden=0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || echo "warning: built for Apple silicon macOS" >&2

echo "==> uv environment (Python 3.12: MLX, PyTorch/MPS, ONNX Runtime, Warp)"
uv sync --all-packages

echo "==> upstream source, rig assets, Hugging Face checkpoints (~1.6 GB, verified)"
uv run --all-packages python sandbox/scripts/fetch_assets.py

if (( onnx )); then
  echo "==> ONNX graphs via NVIDIA's exporter (TensorRT-shim / ONNX Runtime baseline)"
  uv run --all-packages python sandbox/scripts/export_onnx.py
fi
if (( golden )); then
  echo "==> golden fixtures from the upstream PyTorch model"
  uv run --all-packages python sandbox/scripts/make_golden.py
fi

echo "==> doctor"
uv run --all-packages cudashim-doctor
