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

# The fixed G1 layout the kernels are written for. G1Rig checks the rig arrays and the feature
# layout against it, and test_layers checks the F_* / I_* offsets in rig_kernels.metal.
G1_QPOS = 36  # root position, root quaternion (XYZW), 29 hinge angles
G1_FEATURES = 414  # global features of the 34-joint skeleton
G1_ROTATIONS_AT, G1_CONTACTS_AT = 104, 410  # where global_rot_data and foot_contacts start in them
G1_LOCAL_POSE = 303  # 33 joint positions * 3 + 34 rotations * 6
# Packed rig: G1Rig.rf (floats) and G1Rig.ri (ints), in this order and with these sizes.
G1_RF_LAYOUT = (("BASIS", 9), ("AXES", 29 * 3), ("Q2T", 29 * 9), ("REST", 30 * 3), ("NEUTRAL", 34 * 3),
                ("F2Q", 34 * 9), ("MAXES", 29 * 3))
G1_RI_LAYOUT = (("PARENTS", 30), ("MPARENTS", 34), ("REVERSE", 34), ("MAPPING", 29))


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
    # The kernel indexes raw buffers, so a wrong shape would read out of bounds. Under mx.compile
    # these checks run once, at trace time.
    shape = context_qpos.shape
    if len(shape) != 3 or shape[1:] != (4, G1_QPOS) or target_qpos.shape != shape:
        raise ValueError(f"G1 rig kernel: context_qpos and target_qpos must both be (B, 4, {G1_QPOS}); "
                         f"got {shape} and {target_qpos.shape}.")
    b = shape[0]
    grid, group = _threads(b, 32)
    k = _kernel("mb_g1_preprocess", ["context", "target", "rf", "ri"], ["global_root", "local_root", "local_poses"], G1_PREPROCESS)
    return k(
        inputs=[context_qpos.astype(mx.float32), target_qpos.astype(mx.float32), rf, ri],
        template=[("B", b)], grid=grid, threadgroup=group,
        output_shapes=[(b, 8, 5), (b, 8, 4), (b, 8, G1_LOCAL_POSE)], output_dtypes=[mx.float32] * 3,
    )


def g1_postprocess(features, rf, ri):
    """``G1Rig.postprocess``: global features (B,T,414) -> (qpos (B,T,36), foot contacts (B,T,4))."""
    if features.ndim != 3 or features.shape[-1] != G1_FEATURES:
        raise ValueError(f"G1 rig kernel: features must be (B, T, {G1_FEATURES}); got {features.shape}.")
    b, t = features.shape[:2]
    grid, group = _threads(b * t)
    k = _kernel("mb_g1_postprocess", ["features", "rf", "ri"], ["qpos", "contacts"], G1_POSTPROCESS)
    return k(
        inputs=[features.astype(mx.float32), rf, ri], template=[("ROWS", b * t)], grid=grid, threadgroup=group,
        output_shapes=[(b, t, G1_QPOS), (b, t, 4)], output_dtypes=[mx.float32] * 2,
    )


def soma_retarget(six_flat, src, dst, num_joints: int):
    """``SomaRetarget._apply`` on (..., J*6) 6D rotations with offset matrices (J,3,3)."""
    if six_flat.ndim < 1 or six_flat.shape[-1] != num_joints * 6 or src.shape != (num_joints, 3, 3) \
            or dst.shape != src.shape:
        raise ValueError(f"SOMA retarget kernel: expected (..., {num_joints * 6}) rotations and ({num_joints}, 3, 3) "
                         f"offsets; got {six_flat.shape}, {src.shape} and {dst.shape}.")
    rows = six_flat.size // (num_joints * 6)
    grid, group = _threads(rows * num_joints)
    k = _kernel("mb_soma_retarget", ["six", "src", "dst"], ["out"], SOMA_RETARGET)
    return k(
        inputs=[six_flat, src, dst], template=[("ROWS", rows), ("J", num_joints)], grid=grid, threadgroup=group,
        output_shapes=[six_flat.shape], output_dtypes=[mx.float32],
    )[0]
