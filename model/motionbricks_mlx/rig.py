# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/g1/rig.py and
# tools/onnx/models/soma/rotations.py). Original: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""Rig geometry around the network: G1 forward kinematics and joint-angle decoding, and
the SOMA capture <-> standard T-pose rotation retarget. All float32 (the upstream mixed-FP16
export also keeps geometry in FP32)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import kernels


def cross(a, b):
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def normalize(v):
    return v / mx.sqrt((v * v).sum(axis=-1, keepdims=True))


def cont6d_to_matrix(six):
    """6D rotation (first two columns) -> 3x3 matrix with columns (x, y, z)."""
    x = normalize(six[..., 0:3])
    z = normalize(cross(x, six[..., 3:6]))
    y = cross(z, x)
    return mx.stack([x, y, z], axis=-1)


def matrix_to_cont6d(m):
    return mx.concatenate([m[..., 0], m[..., 1]], axis=-1)


def quat_matrix(q):
    """WXYZ quaternion -> rotation matrix (normalizing like upstream's ``2 / |q|^2``)."""
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    s = 2 / (q * q).sum(axis=-1)
    rows = [
        1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
        s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r),
        s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j),
    ]
    return mx.stack(rows, axis=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_quat(m):
    """Rotation matrix -> WXYZ quaternion, choosing the best-conditioned of 4 candidates."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    a = mx.sqrt(mx.maximum(mx.stack([
        1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22,
    ], axis=-1), 0))
    rows = mx.stack([
        mx.stack([a[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
        mx.stack([m21 - m12, a[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
        mx.stack([m02 - m20, m10 + m01, a[..., 2] ** 2, m12 + m21], axis=-1),
        mx.stack([m10 - m01, m20 + m02, m21 + m12, a[..., 3] ** 2], axis=-1),
    ], axis=-2) / (2 * mx.maximum(a, 0.1)[..., :, None])
    best = mx.argmax(a, axis=-1)[..., None, None]
    return mx.take_along_axis(rows, mx.broadcast_to(best, (*a.shape[:-1], 1, 4)), axis=-2)[..., 0, :]


def rotate(m, p):
    return (m @ p[..., None])[..., 0]


def wrap(angle):
    """Wrap radians to [-pi, pi)."""
    return mx.remainder(angle + math.pi, 2 * math.pi) - math.pi


def _check_g1_layout(floats: list[np.ndarray], ints: list[np.ndarray], indices: dict[str, np.ndarray]) -> None:
    """The Metal kernels hard-code the packed rig and feature layout; refuse anything else."""
    got = [part.size for part in floats], [part.size for part in ints]
    want = [size for _, size in kernels.G1_RF_LAYOUT], [size for _, size in kernels.G1_RI_LAYOUT]
    layout = (int(indices["all"].size), int(indices["global_rot_data"][0]), int(indices["foot_contacts"][0]))
    want_layout = (kernels.G1_FEATURES, kernels.G1_ROTATIONS_AT, kernels.G1_CONTACTS_AT)
    if got != want or layout != want_layout:
        raise ValueError(f"G1 rig does not match the layout of the Metal kernels: packed sizes {got} vs {want}, "
                         f"(features, rotations at, contacts at) {layout} vs {want_layout}.")


class G1Rig:
    """Unitree G1: MuJoCo joint coordinates <-> the model's 34-joint skeleton features.

    ``arrays`` is the upstream ``g1_model_rig.npz``; ``indices`` the global layout table.
    """

    def __init__(self, arrays: dict[str, np.ndarray], indices: dict[str, np.ndarray]):
        f32 = lambda k: mx.array(np.asarray(arrays[k], dtype=np.float32))  # noqa: E731
        self.basis = f32("mujoco_to_motion_matrix")
        self.axes = f32("mujoco_joint_axis_values_mujoco_space")
        self.model_axes = f32("mujoco_joint_axis_values_motion_space")
        self.q2t = f32("rot_offsets_q2t")
        self.f2q = f32("rot_offsets_f2q")
        self.rest = f32("capture_neutral_joints_mujoco")
        self.neutral = f32("neutral_joints")
        self.parents = arrays["mujoco_joint_including_root_parent_list"].tolist()
        self.model_parents = arrays["joint_parents"].tolist()
        self.reverse = arrays["motion_indices_to_mujoco_indices"].tolist()
        mapping = np.asarray(arrays["mujoco_indices_to_motion_indices"], dtype=np.int32)
        q2t_mapped = np.asarray(arrays["rot_offsets_q2t"], np.float32)[mapping]
        self.mapping = mx.array(mapping)
        self.q2t_mapped = mx.array(q2t_mapped)
        self.rotation_indices = mx.array(indices["global_rot_data"].astype(np.int32))
        self.root_indices = mx.array(indices["global_root_pos"].astype(np.int32))
        self.contact_indices = mx.array(indices["foot_contacts"].astype(np.int32))
        self.parent_indices = mx.array(np.asarray(self.model_parents[1:], dtype=np.int32))
        self.identity = mx.eye(3)
        self.num_joints = len(self.model_parents)
        # Packed constants for the custom Metal kernels, in the order of kernels.G1_RF_LAYOUT /
        # G1_RI_LAYOUT (the F_* / I_* offsets in rig_kernels.metal).
        floats = [np.asarray(arrays[k], np.float32).ravel() for k in (
            "mujoco_to_motion_matrix", "mujoco_joint_axis_values_mujoco_space")] + [q2t_mapped.ravel()] + [
            np.asarray(arrays[k], np.float32).ravel() for k in (
                "capture_neutral_joints_mujoco", "neutral_joints", "rot_offsets_f2q", "mujoco_joint_axis_values_motion_space")]
        ints = [np.asarray(arrays[k]).astype(np.int32).ravel() for k in (
            "mujoco_joint_including_root_parent_list", "joint_parents", "motion_indices_to_mujoco_indices",
            "mujoco_indices_to_motion_indices")]
        _check_g1_layout(floats, ints, indices)
        self.rf = mx.array(np.concatenate(floats))
        self.ri = mx.array(np.concatenate(ints))

    def fk(self, qpos):
        """``qpos (..., 36)`` with a WXYZ root quaternion -> joint positions ``(..., 34, 3)``
        and rotations ``(..., 34, 3, 3)`` in model space."""
        dofs = qpos[..., 7:]
        quats = mx.concatenate([mx.cos(dofs / 2)[..., None], mx.sin(dofs / 2)[..., None] * self.axes], axis=-1)
        local = self.q2t_mapped @ quat_matrix(quats)
        rotations = [quat_matrix(qpos[..., 3:7])]
        positions = [qpos[..., :3] + self.rest[0]]
        for joint in range(1, len(self.parents)):
            parent = self.parents[joint]
            positions.append(positions[parent] + rotate(rotations[parent], self.rest[joint] - self.rest[parent]))
            rotations.append(rotations[parent] @ local[..., joint - 1, :, :])
        # Change of basis for all 30 robot joints at once (2 batched matmuls, not 90 tiny ones).
        p_all = rotate(self.basis, mx.stack(positions, axis=-2))
        r_all = self.basis @ mx.stack(rotations, axis=-3) @ self.basis.T
        model_p, model_r = [], []
        for joint, robot_index in enumerate(self.reverse):
            if robot_index >= 0:
                model_p.append(p_all[..., robot_index, :])
                model_r.append(r_all[..., robot_index, :, :])
            else:
                parent = self.model_parents[joint]
                model_p.append(model_p[parent] + rotate(model_r[parent], self.neutral[joint] - self.neutral[parent]))
                model_r.append(mx.broadcast_to(self.identity, model_r[parent].shape))
        return mx.stack(model_p, axis=-2), mx.stack(model_r, axis=-3)

    def decode(self, features):
        """Global features ``(B, T, D)`` -> root position, root rotation, 29 joint angles."""
        b, t = features.shape[:2]
        r = cont6d_to_matrix(features[..., self.rotation_indices].reshape(b, t, self.num_joints, 6))
        parents = mx.concatenate(
            [mx.broadcast_to(self.identity, r[..., :1, :, :].shape), r[..., self.parent_indices, :, :]], axis=-3
        )
        local = self.f2q @ (parents.swapaxes(-1, -2) @ r)
        root_p = rotate(self.basis.T, features[..., self.root_indices])
        root_r = self.basis.T @ local[..., 0, :, :] @ self.basis
        hinge = local[..., self.mapping, :, :]
        angles = mx.stack([
            mx.arctan2(hinge[..., 2, 1], hinge[..., 2, 2]),
            mx.arctan2(hinge[..., 0, 2], hinge[..., 0, 0]),
            mx.arctan2(hinge[..., 1, 0], hinge[..., 1, 1]),
        ], axis=-1)
        return root_p, root_r, (angles * self.model_axes).sum(axis=-1)

    def preprocess(self, context_qpos, target_qpos):
        """Placed G1 poses -> ``(global_root (B,8,5), local_root (B,8,4), local_poses (B,8,303))``."""
        if kernels.available():
            return tuple(kernels.g1_preprocess(context_qpos, target_qpos, self.rf, self.ri))
        return self.preprocess_ops(context_qpos, target_qpos)

    def preprocess_ops(self, context_qpos, target_qpos):
        """Op-based reference of :meth:`preprocess` (CPU fallback, test oracle)."""
        poses = mx.concatenate([context_qpos, target_qpos], axis=1)
        poses = mx.concatenate([poses[..., :3], poses[..., 6:7], poses[..., 3:6], poses[..., 7:]], axis=-1)  # XYZW -> WXYZ
        p, r = self.fk(poses)
        root = p[:, :, 0]
        heading = mx.arctan2(r[:, :, 0, 0, 2], r[:, :, 0, 2, 2])
        global_root = mx.concatenate([root, mx.cos(heading)[..., None], mx.sin(heading)[..., None]], axis=-1)
        windows = []
        for start in (0, 4):
            yaw_vel = wrap(heading[:, start + 1:start + 4] - heading[:, start:start + 3]) * 30
            xz = mx.stack([root[..., 0], root[..., 2]], axis=-1)
            velocity = (xz[:, start + 1:start + 4] - xz[:, start:start + 3]) * 30
            first = mx.concatenate([yaw_vel[..., None], velocity, root[:, start:start + 3, 1:2]], axis=-1)
            last = mx.concatenate([mx.zeros_like(first[:, :1, :3]), root[:, start + 3:start + 4, 1:2]], axis=-1)
            windows.append(mx.concatenate([first, last], axis=1))
        nonroot = p[:, :, 1:] - root[:, :, None] * mx.array([1.0, 0.0, 1.0])
        six = mx.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)
        b = poses.shape[0]
        local_poses = mx.concatenate([nonroot.reshape(b, 8, -1), six.reshape(b, 8, -1)], axis=-1)
        return global_root, mx.concatenate(windows, axis=1), local_poses

    def postprocess(self, pred_global):
        """Global features -> ``(pred_qpos (B,T,36), foot contacts (B,T,4))``."""
        if kernels.available():
            return tuple(kernels.g1_postprocess(pred_global, self.rf, self.ri))
        return self.postprocess_ops(pred_global)

    def postprocess_ops(self, pred_global):
        """Op-based reference of :meth:`postprocess` (CPU fallback, test oracle)."""
        root_p, root_r, hinges = self.decode(pred_global)
        q = matrix_quat(root_r)
        quat_xyzw = mx.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
        return mx.concatenate([root_p, quat_xyzw, hinges], axis=-1), pred_global[..., self.contact_indices]


class SomaRetarget:
    """Moves global joint rotations between the capture and the standard T pose."""

    def __init__(self, arrays: dict[str, np.ndarray], num_joints: int):
        self.capture = mx.array(np.asarray(arrays["capture_offsets"], dtype=np.float32))
        self.standard = mx.array(np.asarray(arrays["standard_offsets"], dtype=np.float32))
        self.num_joints = num_joints

    def _apply(self, six_flat, offset_from, offset_to):
        if kernels.available():
            return kernels.soma_retarget(six_flat, offset_from, offset_to, self.num_joints)
        return self.apply_ops(six_flat, offset_from, offset_to)

    def apply_ops(self, six_flat, offset_from, offset_to):
        """Op-based reference of the retarget (CPU fallback, test oracle)."""
        shape = six_flat.shape
        m = cont6d_to_matrix(six_flat.reshape(*shape[:-1], self.num_joints, 6))
        m = (m @ offset_from) @ offset_to.swapaxes(-1, -2)
        return matrix_to_cont6d(m).reshape(shape)

    def capture_to_standard(self, six_flat):
        return self._apply(six_flat, self.capture, self.standard)

    def standard_to_capture(self, six_flat):
        return self._apply(six_flat, self.standard, self.capture)
