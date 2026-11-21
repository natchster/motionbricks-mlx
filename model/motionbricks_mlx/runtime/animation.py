# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (animation/skeleton.py,
# animation/animation_buffer.py, animation/foot_contacts.py, robotics/dof_animation_buffer.py,
# utils/time_utils.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Skeletons and animation buffers: local joint transforms (SOMA) and joint coordinates (G1)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import transforms as T
from .contract import FOOT_CONTACT_COUNT, ROOT_COORDINATE_COUNT

F = np.float32


def frame_index_from_time(time: float, sample_rate: float, num_frames: int) -> tuple[int, int, float]:
    """The frame at a time, the next frame, and the blend toward it; both clamped to the clip."""
    last = max(0, num_frames - 1)
    fractional = time * sample_rate
    frame = min(max(int(math.floor(fractional)), 0), last)
    blend = 0.0
    if frame < last:
        blend = min(max(fractional - frame, 0.0), 1.0)
    return frame, min(frame + 1, last), blend


class Skeleton:
    """Joint hierarchy and reference pose (one local transform ``[7]`` per joint, parents first)."""

    def __init__(self, joint_names: Sequence[str], parent_indices, local_transforms) -> None:
        self.joint_names = list(joint_names)
        if not self.joint_names:
            raise ValueError("A skeleton needs at least one joint.")
        self.parent_indices = np.asarray(parent_indices, dtype=np.int32)
        if self.parent_indices.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {len(self.joint_names)} parent indices, got {self.parent_indices.size}.")
        local = np.asarray(local_transforms, dtype=F)
        if local.shape != (len(self.joint_names), 7):
            raise ValueError(f"Expected {len(self.joint_names)} local transforms.")
        self._reference_local_transforms = local
        self.up_axis = np.array([0.0, 0.0, 1.0], F)
        self.forward_axis = np.array([0.0, -1.0, 0.0], F)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def reference_local_transforms(self) -> np.ndarray:
        """Reference pose, one local joint transform per joint ``[J, 7]``."""
        return self._reference_local_transforms

    def joint_index(self, joint_name: str) -> int:
        """Index of a joint, or -1."""
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            return -1

    def joint_name(self, index: int) -> str:
        if not 0 <= index < self.num_joints:
            raise ValueError(f"Joint index {index} must be in [0, {self.num_joints}).")
        return self.joint_names[index]

    def joint_parent(self, index: int) -> int:
        if not 0 <= index < self.num_joints:
            raise ValueError(f"Joint index {index} must be in [0, {self.num_joints}).")
        return int(self.parent_indices[index])

    def joint_mask(self, joint_names, weight: float = 1.0) -> np.ndarray:
        """``weight`` (clamped to 0..1) on the named joints, zero elsewhere."""
        weights = np.zeros(self.num_joints, F)
        for name in joint_names:
            index = self.joint_index(name)
            if index >= 0:
                weights[index] = min(max(float(weight), 0.0), 1.0)
        return weights

    def compute_world_transforms(self, local_transforms, root_tx=T.IDENTITY) -> np.ndarray:
        """World transforms of one pose ``[J, 7]`` (or several ``[N, J, 7]``)."""
        return T.compute_world_poses(local_transforms, self.parent_indices, root_tx)


def _validate_foot_contacts(foot_contacts, capacity: int):
    if foot_contacts is None:
        return None
    foot_contacts = np.asarray(foot_contacts, dtype=F)
    if foot_contacts.shape != (capacity, FOOT_CONTACT_COUNT):
        raise ValueError(f"foot_contacts must be float32 with shape [{capacity}, {FOOT_CONTACT_COUNT}].")
    return foot_contacts


def sample_foot_contacts(buffer, time: float, output: np.ndarray | None = None) -> np.ndarray:
    """The four foot contacts at a time, blending the two nearest frames (written into ``output``)."""
    if buffer.foot_contacts is None:
        raise ValueError("Cannot sample foot contacts from an animation buffer that has none.")
    if buffer.num_frames == 0:
        raise ValueError("Cannot sample an empty animation buffer.")
    frame, nxt, blend = frame_index_from_time(time, buffer.sample_rate, buffer.num_frames)
    c = buffer.foot_contacts
    value = (c[frame] + (c[nxt] - c[frame]) * F(blend)).astype(F)
    if output is not None:
        output[...] = value
        return output
    return value


class _Buffer:
    """Shared frame bookkeeping of the two buffer types."""

    sample_rate: float
    poses: np.ndarray
    foot_contacts: np.ndarray | None

    @property
    def capacity(self) -> int:
        return self.poses.shape[0]

    @property
    def num_frames(self) -> int:
        """Frames in use, 0 to capacity."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, frame_count: int) -> None:
        if not 0 <= frame_count <= self.capacity:
            raise ValueError(f"num_frames must be in [0, {self.capacity}].")
        self._num_frames = int(frame_count)

    @property
    def duration(self) -> float:
        return max(0, self.num_frames - 1) / self.sample_rate

    def sample_foot_contacts(self, time: float, output: np.ndarray | None = None, stream=None) -> np.ndarray:
        return sample_foot_contacts(self, time, output)


class AnimationBuffer(_Buffer):
    """Animation frames as local joint transforms ``[capacity, J, 7]`` at a fixed sample rate.

    ``num_frames`` counts the frames in use. ``foot_contacts`` is ``[capacity, 4]`` for generated
    motion and ``None`` for clips.
    """

    def __init__(self, skeleton: Skeleton, sample_rate: float, poses, *, num_frames: int | None = None, foot_contacts=None):
        self.skeleton = skeleton
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        self.poses = np.asarray(poses, dtype=F)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (skeleton.num_joints, 7):
            raise ValueError("poses must have shape [capacity, num_joints, 7].")
        self.foot_contacts = _validate_foot_contacts(foot_contacts, self.capacity)
        self.num_frames = self.capacity if num_frames is None else num_frames

    def root_transforms(self) -> np.ndarray:
        """Root of each frame in use, ``[num_frames, 7]`` (position, then XYZW rotation)."""
        return self.poses[: self.num_frames, 0].copy()

    def sample(self, time: float, output_pose: np.ndarray | None = None, *, root_offset=T.IDENTITY, stream=None) -> np.ndarray:
        """The pose at a time (two nearest frames blended; times clamped), root moved by ``root_offset``.

        ``output_pose`` ``[J, 7]`` is written in place when given. ``stream`` is accepted for
        source compatibility with upstream and ignored.
        """
        if self.num_frames == 0:
            raise ValueError("Cannot sample an empty animation buffer.")
        frame, nxt, blend = frame_index_from_time(time, self.sample_rate, self.num_frames)
        pose = T.blend_transform(self.poses[frame], self.poses[nxt], np.full(self.skeleton.num_joints, blend, F))
        pose[0] = T.transform_mul(root_offset, pose[0])
        if output_pose is not None:
            if output_pose.shape != (self.skeleton.num_joints, 7):
                raise ValueError("output_pose must contain one transform per joint.")
            output_pose[...] = pose
            return output_pose
        return pose

    def copy(self) -> AnimationBuffer:
        contacts = None if self.foot_contacts is None else self.foot_contacts[: self.num_frames].copy()
        return AnimationBuffer(self.skeleton, self.sample_rate, self.poses[: self.num_frames].copy(), foot_contacts=contacts)

    def make_loopable(self, stream=None) -> None:
        """Adjust the frames in use so the first and last poses match (correction grows linearly)."""
        if self.num_frames < 2:
            return
        last = self.num_frames - 1
        p = self.poses
        root_delta = T.compute_root_endpoint_delta(p[last, 0], p[0, 0])
        deltas = T.compute_delta_transform(p[last, 1:], p[0, 1:])
        for frame in range(self.num_frames):
            w = F(frame / last)
            p[frame, 0] = T.apply_root_endpoint_correction(p[frame, 0], T.IDENTITY, root_delta, w)
            p[frame, 1:] = T.apply_delta_transform(p[frame, 1:], deltas, np.full(len(deltas), w, F))


def create_animation_buffer_for_skeleton(animation: AnimationBuffer, new_skeleton: Skeleton) -> AnimationBuffer:
    """Remap an animation to another skeleton by joint name; missing joints keep its reference pose."""
    if animation.skeleton is new_skeleton:
        return animation
    poses = np.broadcast_to(new_skeleton.reference_local_transforms, (animation.num_frames, new_skeleton.num_joints, 7)).copy()
    for source, name in enumerate(animation.skeleton.joint_names):
        target = new_skeleton.joint_index(name)
        if target >= 0:
            poses[:, target] = animation.poses[: animation.num_frames, source]
    return AnimationBuffer(new_skeleton, animation.sample_rate, poses)


class DOFAnimationBuffer(_Buffer):
    """Animation frames as joint coordinates ``[capacity, 7 + num_dofs]``.

    Each frame holds the root position (m), root rotation (XYZW) and the joint angles (rad), in
    ``joint_names`` order. Sampling lerps positions and angles and slerps the root rotation.
    """

    def __init__(self, joint_names: Sequence[str], sample_rate: float, poses, *, num_frames: int | None = None, foot_contacts=None):
        self.joint_names = tuple(joint_names)
        self.num_dofs = len(self.joint_names)
        self.num_coordinates = ROOT_COORDINATE_COUNT + self.num_dofs
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        self.poses = np.asarray(poses, dtype=F)
        if self.poses.ndim != 2 or self.poses.shape[1] != self.num_coordinates:
            raise ValueError(f"poses must have shape [capacity, {self.num_coordinates}].")
        self.foot_contacts = _validate_foot_contacts(foot_contacts, self.capacity)
        self.num_frames = self.capacity if num_frames is None else num_frames

    def root_transforms(self) -> np.ndarray:
        return self.poses[: self.num_frames, :ROOT_COORDINATE_COUNT].copy()

    def sample(self, time: float, output_pose: np.ndarray | None = None, *, root_offset=T.IDENTITY, stream=None) -> np.ndarray:
        """Joint coordinates at a time; the root rotation is renormalized like upstream."""
        if self.num_frames == 0:
            raise ValueError("Cannot sample an empty animation buffer.")
        frame, nxt, blend = frame_index_from_time(time, self.sample_rate, self.num_frames)
        a, b = self.poses[frame], self.poses[nxt]
        out = (a + (b - a) * F(blend)).astype(F)  # joint angles lerp, also across +-pi
        root = T.blend_transform(read_root(a), read_root(b), F(blend))
        write_root(T.transform_mul(root_offset, root), out)
        if output_pose is not None:
            if output_pose.shape != (self.num_coordinates,):
                raise ValueError(f"output_pose must be a vector of {self.num_coordinates} coordinates.")
            output_pose[...] = out
            return output_pose
        return out

    def copy(self) -> DOFAnimationBuffer:
        contacts = None if self.foot_contacts is None else self.foot_contacts[: self.num_frames].copy()
        return DOFAnimationBuffer(self.joint_names, self.sample_rate, self.poses[: self.num_frames].copy(), foot_contacts=contacts)

    def make_loopable(self, stream=None) -> None:
        if self.num_frames < 2:
            return
        last = self.num_frames - 1
        p = self.poses
        root_delta = T.compute_root_endpoint_delta(read_root(p[last]), read_root(p[0]))
        angle_deltas = p[0, ROOT_COORDINATE_COUNT:] - p[last, ROOT_COORDINATE_COUNT:]
        for frame in range(self.num_frames):
            w = F(frame / last)
            write_root(T.apply_root_endpoint_correction(read_root(p[frame]), T.IDENTITY, root_delta, w), p[frame])
            p[frame, ROOT_COORDINATE_COUNT:] += angle_deltas * w


def read_root(joint_q) -> np.ndarray:
    """Root transform of joint coordinates ``[..., 7+]``, with a normalized rotation."""
    q = np.asarray(joint_q, dtype=F)
    return T.transform(q[..., :3], T.quat_normalize(q[..., 3:7]))


def write_root(root, joint_q) -> None:
    """Write a root transform (rotation normalized) into joint coordinates in place."""
    root = np.asarray(root, dtype=F)
    joint_q[..., :3] = root[..., :3]
    joint_q[..., 3:7] = T.quat_normalize(root[..., 3:7])
