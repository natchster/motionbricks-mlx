# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (model/soma/buffers.py,
# model/unitree_g1/buffers.py, model/generation.py, runtime/motion_recorder.py, utils/time_utils.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Context/target tokens and the recorder of shown poses that feeds them."""

from __future__ import annotations

import numpy as np

from . import transforms as T
from .animation import read_root, write_root
from .contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ROOT_COORDINATE_COUNT

F = np.float32


def history_sample_indices(delta_times: np.ndarray, num_samples: int, sample_rate: float):
    """Samples at a fixed rate ending at the latest recorded pose: lower frame, upper frame, blend.

    Samples before the available history clamp to its oldest pose.
    """
    pose_times = np.zeros_like(delta_times)
    for frame in range(len(pose_times) - 2, -1, -1):
        pose_times[frame] = pose_times[frame + 1] - delta_times[frame]
    lower_frames = np.empty(num_samples, np.int32)
    upper_frames = np.empty(num_samples, np.int32)
    blends = np.empty(num_samples, F)
    first_sample = -(num_samples - 1) / sample_rate
    for sample in range(num_samples):
        sample_time = first_sample + sample / sample_rate
        upper = min(int(np.searchsorted(pose_times, sample_time, side="right")), len(pose_times) - 1)
        lower = max(upper - 1, 0)
        frame_time = float(pose_times[upper] - pose_times[lower])
        if frame_time > 0.0:
            blend = (sample_time - float(pose_times[lower])) / frame_time
        else:
            lower = upper
            blend = 0.0
        lower_frames[sample] = lower
        upper_frames[sample] = upper
        blends[sample] = np.clip(blend, 0.0, 1.0)
    return lower_frames, upper_frames, blends


def write_token_poses(poses, token_poses: np.ndarray) -> int:
    """Copy one to four poses into a token and pad the unused rows with the last pose."""
    poses = np.asarray(poses, dtype=F)
    per_pose = token_poses.shape[1:]
    if poses.shape == per_pose:
        poses = poses[None]
    if poses.shape[1:] != per_pose:
        raise ValueError(f"Each token pose must have shape {per_pose}, got {poses.shape[1:]}.")
    num_poses = poses.shape[0]
    if not 1 <= num_poses <= MODEL_POSES_PER_TOKEN:
        raise ValueError(f"A context or target token requires 1 to {MODEL_POSES_PER_TOKEN} poses.")
    for row in range(MODEL_POSES_PER_TOKEN):
        token_poses[row] = poses[min(row, num_poses - 1)]
    return num_poses


class MotionRecorder:
    """A fixed number of recent poses (oldest first) and the time from each to the next.

    ``poses`` keeps the reference pose's layout: local joint transforms ``[J, 7]`` for SOMA or joint
    coordinates ``[36]`` for G1. ``delta_times[-1]`` is always zero. Unrecorded slots hold the pose
    of the last ``fill()`` with zero duration.
    """

    def __init__(self, reference_pose, capacity: int = 32, stream=None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        reference_pose = np.asarray(reference_pose, dtype=F)
        self.capacity = capacity
        self.poses = np.empty((capacity,) + reference_pose.shape, F)
        self.delta_times = np.zeros(capacity, F)
        self.fill(reference_pose)

    def fill(self, pose, stream=None) -> None:
        """Every slot holds one pose with zero durations, as if the character had held it."""
        pose = self._check(pose)
        self.poses[:] = pose
        self.delta_times.fill(0.0)

    def record(self, delta_time: float, pose, stream=None) -> None:
        """Store a pose sampled ``delta_time`` (finite, > 0) seconds after the previous one."""
        dt = F(delta_time)
        pose = self._check(pose)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("delta_time must be finite and positive.")
        self.delta_times[-1] = dt
        self.delta_times[:-1] = self.delta_times[1:].copy()
        self.delta_times[-1] = 0.0
        self.poses[:-1] = self.poses[1:].copy()
        self.poses[-1] = pose

    def _check(self, pose) -> np.ndarray:
        pose = np.asarray(pose, dtype=F)
        if pose.shape != self.poses.shape[1:]:
            raise ValueError("pose must match the reference pose's shape.")
        return pose


class PoseToken:
    """Up to four SOMA poses (local joint transforms ``[4, J, 7]``) for context or a target."""

    def __init__(self, num_joints: int, *, device=None) -> None:
        if num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        self.num_joints = num_joints
        self.poses = np.zeros((MODEL_POSES_PER_TOKEN, num_joints, 7), F)
        self.poses[..., 6] = 1.0
        self._num_poses = 0

    @property
    def num_poses(self) -> int:
        """Supplied poses, 1 to 4, or 0 before the first update."""
        return self._num_poses

    def update_from_poses(self, local_poses, stream=None) -> None:
        """One to four poses at 30 Hz (``[n, J, 7]`` or ``[J, 7]``); the last pads the rest."""
        self._num_poses = write_token_poses(local_poses, self.poses)

    def update_from_recorder(self, motion_history: MotionRecorder, stream=None) -> None:
        """Four poses at 30 Hz ending at the latest recorded pose (neighbors blended)."""
        if motion_history.poses.shape[1:] != (self.num_joints, 7):
            raise ValueError("History and token joint counts do not match.")
        lower, upper, blends = history_sample_indices(motion_history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE)
        h = motion_history.poses
        for k in range(MODEL_POSES_PER_TOKEN):
            self.poses[k] = T.blend_transform(h[lower[k]], h[upper[k]], np.full(self.num_joints, blends[k], F))
        self._num_poses = MODEL_POSES_PER_TOKEN


class DOFToken:
    """Up to four G1 poses (joint coordinates ``[4, 7 + num_dofs]``) for context or a target."""

    def __init__(self, num_dofs: int, *, device=None) -> None:
        if num_dofs <= 0:
            raise ValueError("num_dofs must be positive.")
        self.num_dofs = num_dofs
        self.num_coordinates = ROOT_COORDINATE_COUNT + num_dofs
        self.poses = np.zeros((MODEL_POSES_PER_TOKEN, self.num_coordinates), F)
        self.poses[:, 6] = 1.0
        self._num_poses = 0

    @property
    def num_poses(self) -> int:
        return self._num_poses

    def update_from_poses(self, joint_q, stream=None) -> None:
        """One to four poses of joint coordinates (``[n, 36]`` or ``[36]``); the last pads the rest."""
        self._num_poses = write_token_poses(joint_q, self.poses)

    def update_from_recorder(self, motion_history: MotionRecorder, stream=None) -> None:
        if motion_history.poses.shape[1:] != (self.num_coordinates,):
            raise ValueError("History and token coordinate counts do not match.")
        lower, upper, blends = history_sample_indices(motion_history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE)
        h = motion_history.poses
        for k in range(MODEL_POSES_PER_TOKEN):
            a, b, w = h[lower[k]], h[upper[k]], F(blends[k])
            out = (a + (b - a) * w).astype(F)
            write_root(T.blend_transform(read_root(a), read_root(b), w), out)
            self.poses[k] = out
        self._num_poses = MODEL_POSES_PER_TOKEN
