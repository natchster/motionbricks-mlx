#!/usr/bin/env bash
# One-shot bootstrap of the CUDA-emulation sandbox + reference oracle. Idempotent: verified
# downloads are skipped, and the ONNX export and fixtures are only produced when missing.
#
#   sandbox/setup.sh            # env, upstream @ pinned commit, checkpoints, ONNX export, golden + runtime fixtures
#   sandbox/setup.sh --no-golden --no-onnx
#   sandbox/setup.sh --force    # re-export the ONNX graphs and regenerate every fixture
#
# Everything lands inside the repo (.venv/, sandbox/upstream/, sandbox/assets/,
# sandbox/golden/, checkpoints/) and every download is SHA-256 verified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

onnx=1 golden=1 force=0
for arg in "$@"; do
  case "$arg" in
    --no-onnx) onnx=0 ;;
    --no-golden) golden=0 ;;
    --force) force=1 ;;
    *) echo "unknown option $arg (expected --no-onnx, --no-golden, --force)" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || echo "warning: built for Apple silicon macOS" >&2

missing() {  # true when --force is given or any of the files is absent
  (( force )) && return 0
  local file
  for file in "$@"; do [[ -f "$file" ]] || return 0; done
  return 1
}

echo "==> uv environment (Python 3.12: MLX, PyTorch/MPS, ONNX Runtime, Warp)"
uv sync --all-packages

echo "==> upstream source, rig assets, Hugging Face checkpoints (~1.6 GB, verified)"
uv run --all-packages python sandbox/scripts/fetch_assets.py

if (( onnx )); then
  if missing sandbox/assets/onnx/mmm_v6_{g1,soma}_fp32.onnx; then
    echo "==> ONNX graphs via NVIDIA's exporter (TensorRT-shim / ONNX Runtime baseline)"
    uv run --all-packages python sandbox/scripts/export_onnx.py
  else
    echo "==> ONNX graphs present in sandbox/assets/onnx (--force re-exports)"
  fi
fi
if (( golden )); then
  if missing sandbox/golden/{unitree_g1,soma}/{trace,random,one_target,restricted,chained}.npz; then
    echo "==> golden fixtures from the upstream PyTorch model"
    uv run --all-packages python sandbox/scripts/make_golden.py
  else
    echo "==> golden fixtures present in sandbox/golden (--force regenerates)"
  fi
  if missing sandbox/golden/runtime/{soma,unitree_g1}.npz; then
    echo "==> Runtime API fixtures from upstream's Warp kernels (CPU) and the PyTorch model"
    uv run --all-packages python sandbox/scripts/make_runtime_golden.py
  else
    echo "==> Runtime API fixtures present in sandbox/golden/runtime (--force regenerates)"
  fi
fi

echo "==> doctor"
uv run --all-packages cudashim-doctor
