# SPDX-License-Identifier: Apache-2.0
"""Custom Metal kernels for the rig geometry (``mx.fast.metal_kernel``).

The rig math (G1 forward kinematics and joint-angle decoding, the SOMA T-pose retarget) is
hundreds of tiny batched 3x3 matrix products and gathers when written as MLX ops. Each one is
padded into a GEMM tile and dispatched separately: about 1 ms per call for G1 at batch 1 and
3-4 ms at batch 32. One thread per pose (or per joint) does the same math in microseconds.

The kernels are transcriptions of the op-based reference implementations in
:mod:`motionbricks_mlx.rig`, which remain the fallback on the CPU and are what the tests compare
against. The source lives in ``assets/rig_kernels.metal``, shared with the Rust mlx-rs port.
"""

from __future__ import annotations

import re
from importlib import resources

import mlx.core as mx

def _sections() -> dict[str, str]:
    text = resources.files("motionbricks_mlx").joinpath("assets", "rig_kernels.metal").read_text()
    parts = re.split(r"^// @@ (\w+)\n", text, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


_SRC = _sections()
HEADER = _SRC["HEADER"]
G1_PREPROCESS = _SRC["G1_PREPROCESS"]
G1_POSTPROCESS = _SRC["G1_POSTPROCESS"]
SOMA_RETARGET = _SRC["SOMA_RETARGET"]

_cache: dict[str, object] = {}


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str):
    if name not in _cache:
        _cache[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source, header=HEADER)
    return _cache[name]


def available() -> bool:
    """Custom kernels run on the GPU; on the CPU stream the op-based reference is used."""
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def _threads(n: int, per_group: int = 64):
    return (max(n, 1), 1, 1), (min(per_group, max(n, 1)), 1, 1)


def g1_preprocess(context_qpos, target_qpos, rf, ri):
    """``G1Rig.preprocess`` -> (global_root (B,8,5), local_root (B,8,4), local_poses (B,8,303))."""
    b = context_qpos.shape[0]
    grid, group = _threads(b, 32)
    k = _kernel("mb_g1_preprocess", ["context", "target", "rf", "ri"], ["global_root", "local_root", "local_poses"], G1_PREPROCESS)
    return k(
        inputs=[context_qpos.astype(mx.float32), target_qpos.astype(mx.float32), rf, ri],
        template=[("B", b)], grid=grid, threadgroup=group,
        output_shapes=[(b, 8, 5), (b, 8, 4), (b, 8, 303)], output_dtypes=[mx.float32] * 3,
    )


def g1_postprocess(features, rf, ri):
    """``G1Rig.postprocess``: global features (B,T,414) -> (qpos (B,T,36), foot contacts (B,T,4))."""
    b, t = features.shape[:2]
    grid, group = _threads(b * t)
    k = _kernel("mb_g1_postprocess", ["features", "rf", "ri"], ["qpos", "contacts"], G1_POSTPROCESS)
    return k(
        inputs=[features.astype(mx.float32), rf, ri], template=[("ROWS", b * t)], grid=grid, threadgroup=group,
        output_shapes=[(b, t, 36), (b, t, 4)], output_dtypes=[mx.float32] * 2,
    )


def soma_retarget(six_flat, src, dst, num_joints: int):
    """``SomaRetarget._apply`` on (..., J*6) 6D rotations with offset matrices (J,3,3)."""
    rows = six_flat.size // (num_joints * 6)
    grid, group = _threads(rows * num_joints)
    k = _kernel("mb_soma_retarget", ["six", "src", "dst"], ["out"], SOMA_RETARGET)
    return k(
        inputs=[six_flat, src, dst], template=[("ROWS", rows), ("J", num_joints)], grid=grid, threadgroup=group,
        output_shapes=[six_flat.shape], output_dtypes=[mx.float32],
    )[0]
