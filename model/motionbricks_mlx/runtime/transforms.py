# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (motionbricks/utils/*).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Rigid transforms and quaternions as NumPy arrays, with Warp's conventions.

A quaternion is ``[x, y, z, w]`` and a transform is ``[px, py, pz, qx, qy, qz, qw]``, the memory
layout of ``wp.quat`` and ``wp.transform``. Every function broadcasts over leading axes and
computes in float32. The formulas follow Warp's native code (``quat_rotate``, ``quat_slerp``,
``quat_from_matrix``, ...) so results match upstream's kernels to float32 rounding, including
which of ``q`` and ``-q`` comes out.
"""

from __future__ import annotations

import numpy as np

F = np.float32
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], F)
IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], F)
# Shared defaults of many signatures: read-only, so no caller can change them by accident.
IDENTITY_QUAT.setflags(write=False)
IDENTITY.setflags(write=False)


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=F)


def transform(p, q) -> np.ndarray:
    """Build transforms from positions ``[..., 3]`` and rotations ``[..., 4]``."""
    p, q = _f(p), _f(q)
    shape = p.shape[:-1]
    if shape != q.shape[:-1]:
        shape = np.broadcast_shapes(shape, q.shape[:-1])
    out = np.empty((*shape, 7), F)
    out[..., :3] = p
    out[..., 3:] = q
    return out


def pos(t) -> np.ndarray:
    return _f(t)[..., :3]


def rot(t) -> np.ndarray:
    return _f(t)[..., 3:7]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``np.cross`` of 3-vectors with its exact operations (``a1 b2 - a2 b1``, ...), without its
    per-call overhead, which dominates for the small arrays here."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], -1)


def cross(a, b) -> np.ndarray:
    return _cross(_f(a), _f(b))


def dot(a, b) -> np.ndarray:
    return (_f(a) * _f(b)).sum(-1)


def normalize(v) -> np.ndarray:
    """Vector normalize; zero length gives zero (Warp's ``kEps`` is 0)."""
    v = _f(v)
    n = np.sqrt((v * v).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, v / np.where(n > 0, n, 1), 0).astype(F)


def quat_normalize(q) -> np.ndarray:
    """Quaternion normalize; zero length gives the identity."""
    q = _f(q)
    n = np.sqrt((q * q).sum(-1, keepdims=True))
    out = q / np.where(n > 0, n, 1)
    return np.where(n > 0, out, IDENTITY_QUAT).astype(F)


def quat_mul(a, b) -> np.ndarray:
    a, b = _f(a), _f(b)
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    v = aw * bv + bw * av + _cross(av, bv)
    w = aw * bw - (av * bv).sum(-1, keepdims=True)
    return np.concatenate([v, w], -1).astype(F)


def quat_inverse(q) -> np.ndarray:
    """Conjugate, like ``wp.quat_inverse`` (not renormalized)."""
    q = _f(q).copy()
    q[..., :3] *= -1
    return q


def quat_rotate(q, x) -> np.ndarray:
    q, x = _f(q), _f(x)
    qv, w = q[..., :3], q[..., 3:]
    c = 2.0 * w * w - 1.0
    d = 2.0 * (qv * x).sum(-1, keepdims=True)
    return (x * c + qv * d + _cross(qv, x) * w * 2.0).astype(F)


def quat_rotate_inv(q, x) -> np.ndarray:
    q, x = _f(q), _f(x)
    qv, w = q[..., :3], q[..., 3:]
    c = 2.0 * w * w - 1.0
    d = 2.0 * (qv * x).sum(-1, keepdims=True)
    return (x * c + qv * d - _cross(qv, x) * w * 2.0).astype(F)


def quat_from_axis_angle(axis, angle) -> np.ndarray:
    axis, angle = _f(axis), _f(angle)
    half = angle * 0.5
    return np.concatenate([axis * np.sin(half)[..., None], np.cos(half)[..., None]], -1).astype(F)


def quat_to_axis_angle(q):
    q = _f(q)
    v = q[..., :3]
    axis = normalize(v)
    axis = np.where(q[..., 3:] < 0, -axis, axis)
    angle = 2.0 * np.arctan2(np.sqrt((v * v).sum(-1)), np.abs(q[..., 3]))
    return axis.astype(F), angle.astype(F)


def quat_slerp(q0, q1, t) -> np.ndarray:
    """``wp.quat_slerp``: ``q0 * axis_angle(inverse(q0) * q1) ^ t`` along the shorter arc."""
    axis, angle = quat_to_axis_angle(quat_mul(quat_inverse(q0), q1))
    return quat_mul(q0, quat_from_axis_angle(axis, _f(t) * angle))


def quat_from_matrix(m) -> np.ndarray:
    """``wp.quat_from_matrix`` for ``[..., 3, 3]`` rotation matrices (same branches).

    All four of Warp's branches are evaluated with its float32 operation order, and each matrix
    keeps the one Warp would take, so the result is bitwise the same as the per-matrix code.
    """
    m = _f(m)
    shape = m.shape[:-2]
    r = m.reshape(-1, 3, 3)
    r00, r01, r02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    r10, r11, r12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    r20, r21, r22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    one, half = F(1), F(0.5)
    tr = r00 + r11 + r22
    with np.errstate(invalid="ignore", divide="ignore"):  # branches that are not taken may be NaN
        h = np.sqrt(tr + one)
        s = half / h
        trace_branch = np.stack([(r21 - r12) * s, (r02 - r20) * s, (r10 - r01) * s, half * h], -1)
        h = np.sqrt((r00 - (r11 + r22)) + one)
        s = half / h
        x_branch = np.stack([half * h, (r01 + r10) * s, (r20 + r02) * s, (r21 - r12) * s], -1)
        h = np.sqrt((r11 - (r22 + r00)) + one)
        s = half / h
        y_branch = np.stack([(r01 + r10) * s, half * h, (r12 + r21) * s, (r02 - r20) * s], -1)
        h = np.sqrt((r22 - (r00 + r11)) + one)
        s = half / h
        z_branch = np.stack([(r20 + r02) * s, (r12 + r21) * s, half * h, (r10 - r01) * s], -1)
    k = np.where(r11 > r00, 1, 0)
    k = np.where(r22 > np.where(k == 1, r11, r00), 2, k)
    out = np.where((k == 0)[:, None], x_branch, np.where((k == 1)[:, None], y_branch, z_branch))
    out = np.where((tr >= 0)[:, None], trace_branch, out)
    return quat_normalize(out).reshape((*shape, 4))


def quat_between_vectors(a, b) -> np.ndarray:
    a, b = normalize(a), normalize(b)
    c = _cross(a, b)
    d = (a * b).sum(-1, keepdims=True)
    return quat_normalize(np.concatenate([c, 1.0 + d], -1))


def transform_mul(a, b) -> np.ndarray:
    a, b = _f(a), _f(b)
    return transform(a[..., :3] + quat_rotate(a[..., 3:], b[..., :3]), quat_mul(a[..., 3:], b[..., 3:]))


def transform_inverse(t) -> np.ndarray:
    t = _f(t)
    q = quat_inverse(t[..., 3:])
    return transform(-quat_rotate(q, t[..., :3]), q)


def transform_point(t, x) -> np.ndarray:
    t = _f(t)
    return (t[..., :3] + quat_rotate(t[..., 3:], x)).astype(F)


def transform_vector(t, x) -> np.ndarray:
    return quat_rotate(_f(t)[..., 3:], x)


def blend_transform(t0, t1, blend) -> np.ndarray:
    """Positions lerp, rotations slerp (``pose_utils.blend_transform``)."""
    t0, t1, blend = _f(t0), _f(t1), _f(blend)
    b = blend[..., None]
    return transform(t0[..., :3] + (t1[..., :3] - t0[..., :3]) * b, quat_slerp(t0[..., 3:], t1[..., 3:], blend))


def compute_delta_transform(src, dst) -> np.ndarray:
    return transform_mul(transform_inverse(src), dst)


def apply_delta_transform(t, delta, weight) -> np.ndarray:
    return transform_mul(t, blend_transform(IDENTITY, delta, weight))


def compute_root_endpoint_delta(generated_root, guide_root) -> np.ndarray:
    """World-space position offset and post-multiplied rotation from a generated root to a guide."""
    g, t = _f(generated_root), _f(guide_root)
    r = quat_mul(quat_inverse(quat_normalize(g[..., 3:])), quat_normalize(t[..., 3:]))
    return transform(t[..., :3] - g[..., :3], quat_normalize(r))


def apply_root_endpoint_correction(root, start_delta, end_delta, end_weight) -> np.ndarray:
    root, sd, ed, w = _f(root), _f(start_delta), _f(end_delta), _f(end_weight)
    sw = 1.0 - w
    r = quat_mul(quat_mul(root[..., 3:], quat_slerp(IDENTITY_QUAT, sd[..., 3:], sw)), quat_slerp(IDENTITY_QUAT, ed[..., 3:], w))
    p = root[..., :3] + sd[..., :3] * sw[..., None] + ed[..., :3] * w[..., None]
    return transform(p, quat_normalize(r))


def quadratic_fade_weight(value) -> np.ndarray:
    t = np.clip(_f(value), 0.0, 1.0)
    rem = 1.0 - t
    return np.where(t < 0.5, 1.0 - 2.0 * t * t, 2.0 * rem * rem).astype(F)


def matrix_from_cols(x, y, z) -> np.ndarray:
    return np.stack([_f(x), _f(y), _f(z)], -1)


def check_parent_indices(parent_indices) -> np.ndarray:
    """Parent indices as int32, checked to list parents first: joint 0 is the only root."""
    parents = np.asarray(parent_indices, dtype=np.int32)
    if parents.ndim != 1 or parents.size == 0:
        raise ValueError("parent_indices must be a non-empty vector.")
    if parents[0] >= 0:
        raise ValueError(f"Joint 0 must be the root (parent -1), got parent {parents[0]}.")
    for joint in range(1, parents.size):
        if not 0 <= parents[joint] < joint:
            raise ValueError(f"Joint {joint} has parent {parents[joint]}; each parent must come before its children "
                             "and only joint 0 may be a root.")
    return parents


def hierarchy_levels(parent_indices) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The joints below the root grouped by depth, as ``(joints, their parents)`` per level."""
    parents = check_parent_indices(parent_indices)
    depth = np.zeros(parents.size, np.int64)
    for joint in range(1, parents.size):
        depth[joint] = depth[parents[joint]] + 1
    levels = []
    for d in range(1, int(depth.max(initial=0)) + 1):
        joints = np.flatnonzero(depth == d)
        levels.append((joints, parents[joints]))
    return tuple(levels)


def compute_world_poses(local_poses, parent_indices, root_tx=IDENTITY, levels=None) -> np.ndarray:
    """Compose local joint transforms ``[..., J, 7]`` through the hierarchy (parents first).

    The joints of one depth are composed together, which gives the same values as joint by joint.
    ``levels`` is ``hierarchy_levels(parent_indices)``, which ``Skeleton`` keeps; ``None``
    computes it.
    """
    local = _f(local_poses)
    if levels is None:
        levels = hierarchy_levels(parent_indices)
    world = np.empty_like(local)
    world[..., 0, :] = transform_mul(root_tx, local[..., 0, :])
    for joints, parents in levels:
        world[..., joints, :] = transform_mul(world[..., parents, :], local[..., joints, :])
    return world
