# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/{representation,features}.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""Motion feature layouts, normalization, and global <-> local root conversion.

MotionBricks describes a frame twice: a *global* layout (root position + heading as
cos/sin) and a *local* layout (root yaw rate, planar velocity, height). Both share the
same body features. The checkpoint's statistics cover the *dual* layout, which is the
global root, then the local root, then the body.

All index arithmetic here is done once, in NumPy, at construction time; the hot path
only does gathers and elementwise math, so it compiles into a few fused kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

FPS = 30.0
STATS_EPS = 1e-5

GLOBAL_ROOT_KEYS = {"global_root_pos": 3, "global_root_heading": 2}
LOCAL_ROOT_KEYS = {"local_root_rot_vel": 1, "local_root_vel": 2, "global_root_y": 1}


def body_keys(num_joints: int) -> dict[str, int]:
    return {
        "ric_data": (num_joints - 1) * 3,
        "global_rot_data": num_joints * 6,
        "local_vel": num_joints * 3,
        "foot_contacts": 4,
    }


@dataclass(frozen=True)
class Layout:
    """Index table of one feature layout (same semantics as upstream ``motion_rep.indices``)."""

    indices: dict[str, np.ndarray]
    dim: int
    root_dim: int

    @staticmethod
    def build(root: dict[str, int], body: dict[str, int]) -> Layout:
        indices, offset = {}, 0
        for key, size in (root | body).items():
            indices[key] = np.arange(offset, offset + size)
            offset += size
        root_dim = sum(root.values())
        indices["all"] = np.arange(offset)
        indices["root"] = np.arange(root_dim)
        indices["body"] = np.arange(root_dim, offset)
        if "global_root_pos" in indices:
            indices["global_root_pos_2d"] = indices["global_root_pos"][[0, 2]]
        return Layout(indices, offset, root_dim)


class MotionRepresentation:
    """Dual global/local layout with the checkpoint's normalization statistics."""

    def __init__(self, num_joints: int, mean: np.ndarray, std: np.ndarray, fps: float = FPS):
        self.num_joints, self.fps = num_joints, fps
        body = body_keys(num_joints)
        self.global_ = Layout.build(GLOBAL_ROOT_KEYS, body)
        self.local = Layout.build(LOCAL_ROOT_KEYS, body)
        self.dual = Layout.build(GLOBAL_ROOT_KEYS | LOCAL_ROOT_KEYS, body)
        if mean.shape != (self.dual.dim,) or std.shape != (self.dual.dim,):
            raise ValueError(f"stats must have shape ({self.dual.dim},), got {mean.shape} / {std.shape}")
        g_root = np.arange(self.global_.root_dim)
        l_root = np.arange(self.global_.root_dim, self.dual.root_dim)
        d_body = self.dual.indices["body"]
        # Map each layout's feature index to the dual index that holds its statistics.
        self.global_to_dual = np.concatenate([g_root, d_body])
        self.local_to_dual = np.concatenate([l_root, d_body])
        self.mean = mean.astype(np.float32)
        self.scale = np.sqrt(std.astype(np.float32) ** 2 + np.float32(STATS_EPS))

        li = self.local.indices
        gi = self.global_.indices
        # "joint_positions_and_rotations_and_hip_height" in the local layout.
        self.local_pose_feature = np.concatenate([li["global_root_y"], li["ric_data"], li["global_rot_data"]])
        # "root_without_hip_height" in the global layout (pose-network root condition).
        self.global_root_no_height = np.concatenate([gi["global_root_pos_2d"], gi["global_root_heading"]])
        # "root_without_hip_height_without_heading" in the local layout (decoder condition).
        self.local_root_velocity = li["local_root_vel"]
        # MLX gathers need mx.array indices.
        self.global_root_no_height_mx = mx.array(self.global_root_no_height.astype(np.int32))
        self.local_root_velocity_mx = mx.array(self.local_root_velocity.astype(np.int32))

        self._g_root = self._stats(self.global_to_dual[gi["root"]])
        self._l_root = self._stats(self.local_to_dual[li["root"]])
        self._l_pose = self._stats(self.local_to_dual[self.local_pose_feature])
        self._l_body = self._stats(self.local_to_dual[li["body"]])

    def _stats(self, dual_index: np.ndarray) -> tuple[mx.array, mx.array]:
        return mx.array(self.mean[dual_index]), mx.array(self.scale[dual_index])

    @staticmethod
    def _norm(x, stats):
        return (x - stats[0]) / stats[1]

    @staticmethod
    def _unnorm(x, stats):
        return x * stats[1] + stats[0]

    # -- normalization of the model inputs -------------------------------------------
    def normalize_global_root(self, x):
        return self._norm(x, self._g_root)

    def unnormalize_global_root(self, x):
        return self._unnorm(x, self._g_root)

    def normalize_local_root(self, x):
        return self._norm(x, self._l_root)

    def unnormalize_local_root(self, x):
        return self._unnorm(x, self._l_root)

    def normalize_local_pose(self, x):
        """Normalize ``[hip height, joint positions, joint rotations]`` features."""
        return self._norm(x, self._l_pose)

    # -- root conversions --------------------------------------------------------------
    def global_root_to_local(self, normalized_global_root: mx.array, lengths: mx.array) -> mx.array:
        """``global_root_to_local_onnx_compatible``: normalized global root ``(B, T, 5)`` ->
        normalized local root ``(B, T, 4)``. The velocity at each request's last frame
        (``lengths - 1``) repeats the frame before it."""
        root = self.unnormalize_global_root(normalized_global_root)
        positions = root[..., :3]
        heading = mx.arctan2(root[..., 4], root[..., 3])
        # Round-trip through the half-angle quaternion like the original, for parity.
        angles = mx.arctan2(mx.sin(heading / 2), mx.cos(heading / 2)) * 2
        cos, sin = mx.cos(angles), mx.sin(angles)
        cos_diff = cos[:, 1:] * cos[:, :-1] + sin[:, 1:] * sin[:, :-1]
        sin_diff = sin[:, 1:] * cos[:, :-1] - cos[:, 1:] * sin[:, :-1]
        angular = self.fps * mx.arctan2(sin_diff, cos_diff)
        angular = mx.concatenate([angular, mx.zeros_like(angular[:, :1])], axis=1)
        velocity = self.fps * (positions[:, 1:] - positions[:, :-1])
        velocity = mx.concatenate([velocity, mx.zeros_like(velocity[:, :1])], axis=1)

        last = (lengths.astype(mx.int32) - 1)[:, None]  # (B, 1)
        is_last = mx.arange(root.shape[1])[None, :] == last
        prev_angular = mx.take_along_axis(angular, last - 1, axis=1)
        prev_velocity = mx.take_along_axis(velocity, (last - 1)[..., None], axis=1)
        angular = mx.where(is_last, prev_angular, angular)
        velocity = mx.where(is_last[..., None], prev_velocity, velocity)
        # COMPUTE_KWARGS["removing_heading"] is False: velocities stay in the global frame.
        local_root = mx.concatenate([angular[..., None], velocity[..., 0:1], velocity[..., 2:3], positions[..., 1:2]], -1)
        return self.normalize_local_root(local_root)

    def local_to_global(self, normalized_local: mx.array) -> mx.array:
        """Decoded local features (normalized) -> global features (unnormalized).

        Integrates the root yaw rate and planar velocity (``LocalRootLocalBody``), then
        rebuilds the global root as ``[x, y, z, cos(heading), sin(heading)]``."""
        root = self.unnormalize_local_root(normalized_local[..., :4])
        body = self._unnorm(normalized_local[..., 4:], self._l_body)
        yaw_rate = root[..., 0] / self.fps
        planar = root[..., 1:3] / self.fps
        zeros = mx.zeros_like(yaw_rate[:, :1])
        yaw = mx.cumsum(mx.concatenate([zeros, yaw_rate[:, :-1]], axis=1), axis=1)
        xz = mx.cumsum(mx.concatenate([mx.zeros_like(planar[:, :1]), planar[:, :-1]], axis=1), axis=1)
        heading = mx.arctan2(mx.sin(yaw / 2), mx.cos(yaw / 2)) * 2
        global_root = mx.stack(
            [xz[..., 0], root[..., 3], xz[..., 1], mx.cos(heading), mx.sin(heading)], axis=-1
        )
        return mx.concatenate([global_root, body], axis=-1)
