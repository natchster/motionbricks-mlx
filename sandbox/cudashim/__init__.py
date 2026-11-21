# SPDX-License-Identifier: Apache-2.0
"""cudashim: a CUDA-emulation sandbox for Apple silicon (M1-M5).

There is no way to execute NVIDIA's binaries (TensorRT, cuDNN, libtorch_cuda) on macOS;
what works is substituting each CUDA layer at its API boundary:

=================  ==========================================  ===============================
CUDA layer          Substitute on Apple silicon                 Module
=================  ==========================================  ===============================
torch.cuda          PyTorch MPS (Metal) + CPU fallback          :mod:`cudashim.torch_cuda`
TensorRT            ONNX Runtime (CPU EP / CoreML EP)           :mod:`cudashim.tensorrt_ort`
Warp CUDA kernels   Warp's LLVM CPU backend (same kernel src)   :mod:`cudashim.warp_cpu`
raw CUDA C++        CuMetal (CUDA->Metal compiler), optional    ``sandbox/native``
=================  ==========================================  ===============================

:mod:`cudashim.reference` runs NVIDIA's upstream PyTorch MotionBricks model as the oracle
for the MLX port in ``model/``.
"""

from __future__ import annotations

import sys

__all__ = ["install_tensorrt", "enable_torch_cuda", "enable_all"]


def install_tensorrt(force: bool = False):
    """Make ``import tensorrt`` resolve to the ONNX Runtime-backed shim."""
    from . import tensorrt_ort

    if force or "tensorrt" not in sys.modules:
        sys.modules["tensorrt"] = tensorrt_ort
    return sys.modules["tensorrt"]


def enable_torch_cuda():
    """Redirect ``cuda`` devices and the ``torch.cuda`` API to MPS. Returns the target device."""
    from .torch_cuda import enable

    return enable()


def enable_all():
    install_tensorrt()
    return enable_torch_cuda()
