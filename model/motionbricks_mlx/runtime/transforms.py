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


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=F)


def transform(p, q) -> np.ndarray:
    """Build transforms from positions ``[..., 3]`` and rotations ``[..., 4]``."""
    p, q = _f(p), _f(q)
    shape = np.broadcast_shapes(p.shape[:-1], q.shape[:-1])
    out = np.empty(shape + (7,), F)
    out[..., :3] = p
    out[..., 3:] = q
    return out


def pos(t) -> np.ndarray:
    return _f(t)[..., :3]


def rot(t) -> np.ndarray:
    return _f(t)[..., 3:7]


def cross(a, b) -> np.ndarray:
    return np.cross(_f(a), _f(b)).astype(F)


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
    v = aw * bv + bw * av + np.cross(av, bv)
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
    return (x * c + qv * d + np.cross(qv, x) * w * 2.0).astype(F)


def quat_rotate_inv(q, x) -> np.ndarray:
    q, x = _f(q), _f(x)
    qv, w = q[..., :3], q[..., 3:]
    c = 2.0 * w * w - 1.0
    d = 2.0 * (qv * x).sum(-1, keepdims=True)
    return (x * c + qv * d - np.cross(qv, x) * w * 2.0).astype(F)


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
    """``wp.quat_from_matrix`` for ``[..., 3, 3]`` rotation matrices (same branches)."""
    m = _f(m)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4), F)
    for i, r in enumerate(m):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr >= 0:
            h = np.sqrt(tr + F(1))
            w = F(0.5) * h
            h = F(0.5) / h
            x, y, z = (r[2, 1] - r[1, 2]) * h, (r[0, 2] - r[2, 0]) * h, (r[1, 0] - r[0, 1]) * h
        else:
            k = 0
            if r[1, 1] > r[0, 0]:
                k = 1
            if r[2, 2] > r[k, k]:
                k = 2
            if k == 0:
                h = np.sqrt((r[0, 0] - (r[1, 1] + r[2, 2])) + F(1))
                x = F(0.5) * h
                h = F(0.5) / h
                y, z, w = (r[0, 1] + r[1, 0]) * h, (r[2, 0] + r[0, 2]) * h, (r[2, 1] - r[1, 2]) * h
            elif k == 1:
                h = np.sqrt((r[1, 1] - (r[2, 2] + r[0, 0])) + F(1))
                y = F(0.5) * h
                h = F(0.5) / h
                z, x, w = (r[1, 2] + r[2, 1]) * h, (r[0, 1] + r[1, 0]) * h, (r[0, 2] - r[2, 0]) * h
            else:
                h = np.sqrt((r[2, 2] - (r[0, 0] + r[1, 1])) + F(1))
                z = F(0.5) * h
                h = F(0.5) / h
                x, y, w = (r[2, 0] + r[0, 2]) * h, (r[1, 2] + r[2, 1]) * h, (r[1, 0] - r[0, 1]) * h
        out[i] = (x, y, z, w)
    return quat_normalize(out).reshape(shape + (4,))


def quat_between_vectors(a, b) -> np.ndarray:
    a, b = normalize(a), normalize(b)
    c = np.cross(a, b)
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


def compute_world_poses(local_poses, parent_indices, root_tx=IDENTITY) -> np.ndarray:
    """Compose local joint transforms ``[..., J, 7]`` through the hierarchy (parents first)."""
    local = _f(local_poses)
    world = np.empty_like(local)
    world[..., 0, :] = transform_mul(root_tx, local[..., 0, :])
    for j in range(1, local.shape[-2]):
        world[..., j, :] = transform_mul(world[..., parent_indices[j], :], local[..., j, :])
    return world


def compute_local_poses(world_poses, parent_indices, root_tx=IDENTITY) -> np.ndarray:
    world = _f(world_poses)
    local = np.empty_like(world)
    for j in range(world.shape[-2]):
        parent = parent_indices[j]
        parent_tx = np.broadcast_to(_f(root_tx), world[..., j, :].shape) if parent < 0 else world[..., parent, :]
        local[..., j, :] = transform_mul(transform_inverse(parent_tx), world[..., j, :])
    return local
