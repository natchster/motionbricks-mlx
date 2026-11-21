# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (runtime/target_pose.py,
# animation/ik.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# The two-bone IK solver derives from Daniel Holden's ik_two_bone (controller.cpp in
# https://github.com/orangeduck/Motion-Matching, MIT; see licenses/Daniel-Holden-MIT-LICENSE.txt).
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Per-target generation settings, target-pose IK pins, and locomotion target placement."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from . import transforms as T
from .animation import Skeleton, read_root, write_root
from .contract import MODEL_SAMPLE_RATE, ROOT_COORDINATE_COUNT

F = np.float32
MAX_TRANSITION_EARLY_BY_FRAMES = 20


class IKPin(IntFlag):
    """Hands and feet that a target pose can pin with IK."""

    NONE = 0
    LEFT_HAND = 1 << 0
    RIGHT_HAND = 1 << 1
    LEFT_FOOT = 1 << 2
    RIGHT_FOOT = 1 << 3
    ALL = LEFT_HAND | RIGHT_HAND | LEFT_FOOT | RIGHT_FOOT


IK_PINS = (IKPin.LEFT_HAND, IKPin.RIGHT_HAND, IKPin.LEFT_FOOT, IKPin.RIGHT_FOOT)


@dataclass(frozen=True, slots=True)
class TargetPoseInferenceParams:
    """Generation settings for reaching one target.

    Attributes:
        transition_early_by_frames: Frames before the end of the generated motion where the
            transition point sits (a larger value makes the target a guide).
        num_poses_to_use: How many of the target's four poses inform the model, 1 to 4.
        ik_pin_mask: ``IKPin`` flags for the hands and feet that target pose IK pins (SOMA).
        allowed_token_counts: Token counts that replace the generation config's; ``None`` keeps them.
    """

    transition_early_by_frames: int
    num_poses_to_use: int
    ik_pin_mask: IKPin = IKPin.NONE
    allowed_token_counts: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class TwoBoneIKChainConfig:
    """Joint indices of a two-bone chain (start, middle, tip) and its bend hint (middle-joint space)."""

    pin: IKPin
    root_index: int
    mid_index: int
    tip_index: int
    hint_offset_local: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.pin not in IK_PINS:
            raise ValueError("pin must identify exactly one supported IK pin.")

    @classmethod
    def from_joint_names(cls, pin: IKPin, skeleton: Skeleton, root_joint: str, mid_joint: str, tip_joint: str,
                         hint_offset_local: tuple[float, float, float]) -> TwoBoneIKChainConfig:
        names = (root_joint, mid_joint, tip_joint)
        indices = tuple(skeleton.joint_index(n) for n in names)
        missing = [n for n, i in zip(names, indices) if i < 0]
        if missing:
            raise ValueError(f"{pin.name} IK chain joints are missing from the skeleton: {', '.join(map(repr, missing))}.")
        return cls(pin, *indices, hint_offset_local)


_SOMA_IK_CHAINS = (
    (IKPin.LEFT_HAND, "LeftArm", "LeftForeArm", "LeftHand", (0.0, -0.25, 0.0)),
    (IKPin.RIGHT_HAND, "RightArm", "RightForeArm", "RightHand", (0.0, 0.25, 0.0)),
    (IKPin.LEFT_FOOT, "LeftLeg", "LeftShin", "LeftFoot", (0.0, -0.25, 0.0)),
    (IKPin.RIGHT_FOOT, "RightLeg", "RightShin", "RightFoot", (0.0, 0.25, 0.0)),
)


def create_soma_target_pose_ik_rig(skeleton: Skeleton) -> tuple[TwoBoneIKChainConfig, ...]:
    """The SOMA arm and leg chains with their bend hints."""
    return tuple(TwoBoneIKChainConfig.from_joint_names(pin, skeleton, r, m, t, h) for pin, r, m, t, h in _SOMA_IK_CHAINS)


_EPSILON = F(1e-3)


def _acos(x):
    return np.arccos(np.clip(x, -1.0, 1.0)).astype(F)


def solve_two_bone_ik(weight, a_parent, a, b, c, t_world, use_hint: bool, hint_world):
    """Solved world transforms of the chain start, middle joint and tip so the tip reaches ``t_world``.

    ``weight`` blends from the input pose (0) to the solved pose (1); with ``use_hint`` the
    chain bends toward ``hint_world``.
    """
    weight = F(min(max(weight, 0.0), 1.0))
    pa, pb, pc = a[:3], b[:3], c[:3]
    t = pc + (t_world[:3] - pc) * weight
    l_ab = np.linalg.norm(pb - pa).astype(F)
    l_cb = np.linalg.norm(pb - pc).astype(F)
    l_at = F(min(max(np.linalg.norm(t - pa), _EPSILON), l_ab + l_cb - _EPSILON))
    axis = T.normalize(np.cross(pc - pa, hint_world - pa) if use_hint else np.cross(pc - pa, pb - pa))
    ac_ab_0 = np.arctan2(T.dot(axis, np.cross(pc - pa, pb - pa)), T.dot(pc - pa, pb - pa))
    ba_bc_0 = np.arctan2(T.dot(axis, np.cross(pa - pb, pc - pb)), T.dot(pa - pb, pc - pb))
    ac_ab_1 = _acos((l_ab * l_ab + l_at * l_at - l_cb * l_cb) / (2.0 * l_ab * l_at))
    ba_bc_1 = _acos((l_ab * l_ab + l_cb * l_cb - l_at * l_at) / (2.0 * l_ab * l_cb))
    rot_a = T.quat_from_axis_angle(T.quat_rotate_inv(a[3:], axis), F(ac_ab_1 - ac_ab_0))
    rot_b = T.quat_from_axis_angle(T.quat_rotate_inv(b[3:], axis), F(ba_bc_1 - ba_bc_0))
    zero = np.zeros(3, F)
    a_local = T.transform_mul(T.transform_mul(T.transform_inverse(a_parent), a), T.transform(zero, rot_a))
    a_world = T.transform_mul(a_parent, a_local)
    ac_local = T.quat_rotate_inv(a_world[3:], pc - pa)
    at_local = T.quat_rotate_inv(a_world[3:], t - pa)
    a_local = T.transform_mul(a_local, T.transform(zero, T.quat_between_vectors(ac_local, at_local)))
    a_world = T.transform_mul(a_parent, a_local)
    b_local = T.transform_mul(T.transform_mul(T.transform_inverse(a), b), T.transform(zero, rot_b))
    b_world = T.transform_mul(a_world, b_local)
    c_world = T.transform_mul(b_world, T.transform_mul(T.transform_inverse(b), c))
    # wp.quat_inverse is the conjugate, so renormalize to keep errors from building up.
    return (
        T.transform(a_world[:3], T.quat_normalize(a_world[3:])),
        T.transform(b_world[:3], T.quat_normalize(b_world[3:])),
        T.transform(c_world[:3], T.quat_normalize(T.quat_slerp(c_world[3:], t_world[3:], weight))),
    )


class TargetPoseIKCorrector:
    """Ease pinned hands and feet onto a target pose with two-bone IK (SOMA)."""

    def __init__(self, skeleton: Skeleton, chains: Sequence[TwoBoneIKChainConfig]) -> None:
        if not chains:
            raise ValueError("Target pose IK chains cannot be empty.")
        for chain in chains:
            if skeleton.joint_parent(chain.mid_index) != chain.root_index or skeleton.joint_parent(chain.tip_index) != chain.mid_index:
                raise ValueError("An IK chain's mid joint must be a child of its chain start, and its tip a child of its mid joint.")
        self._skeleton = skeleton
        self._chains = tuple(chains)
        self._pin_mask = IKPin.NONE
        self._pinned = np.zeros((len(chains), 7), F)

    @property
    def pin_mask(self) -> IKPin:
        return self._pin_mask

    def capture_pins(self, target_pose, pin_mask: IKPin, stream=None) -> None:
        """Pin the world transforms of the chains' tips in a target's local pose (``NONE`` clears)."""
        if pin_mask == IKPin.NONE:
            self.clear_pins()
            return
        world = self._skeleton.compute_world_transforms(self._check(target_pose))
        for i, chain in enumerate(self._chains):
            self._pinned[i] = world[chain.tip_index]
        self._pin_mask = pin_mask

    def apply(self, current_pose: np.ndarray, weight: float, stream=None) -> None:
        """Correct the pinned chains of a sampled local pose in place (``weight`` 0..1)."""
        if self._pin_mask == IKPin.NONE or weight <= 0.0:
            return
        pose = self._check(current_pose)
        world = self._skeleton.compute_world_transforms(pose)
        parents = self._skeleton.parent_indices
        for i, chain in enumerate(self._chains):
            if not (int(self._pin_mask) & int(chain.pin)):
                continue
            parent = parents[chain.root_index]
            parent_world = world[parent] if parent >= 0 else T.IDENTITY
            mid = world[chain.mid_index]
            hint = T.transform_point(mid, np.asarray(chain.hint_offset_local, F))
            ra, rb, rc = solve_two_bone_ik(weight, parent_world, world[chain.root_index], mid, world[chain.tip_index],
                                           self._pinned[i], True, hint)
            current_pose[chain.root_index] = T.transform_mul(T.transform_inverse(parent_world), ra)
            current_pose[chain.mid_index] = T.transform_mul(T.transform_inverse(ra), rb)
            current_pose[chain.tip_index] = T.transform_mul(T.transform_inverse(rb), rc)

    def clear_pins(self) -> None:
        self._pin_mask = IKPin.NONE

    def _check(self, pose) -> np.ndarray:
        pose = np.asarray(pose, F)
        if pose.shape != (self._skeleton.num_joints, 7):
            raise ValueError("A pose must contain one transform per skeleton joint.")
        return pose


def _ground_speed_scale(start, end, pose_count: int, desired_speed: float) -> F:
    velocity = (end - start) * F(MODEL_SAMPLE_RATE / (pose_count - 1))
    speed = math.hypot(float(velocity[0]), float(velocity[1]))
    return F(1.0) if speed <= 1.0e-6 else F(desired_speed / speed)


def _check_speed(desired_speed: float) -> None:
    if not math.isfinite(desired_speed) or desired_speed < 0.0:
        raise ValueError("desired_speed must be finite and non-negative.")


def adjust_target_pose_speed(target_poses: np.ndarray, desired_speed: float, stream=None) -> None:
    """Scale the root's ground motion of 30 Hz target poses ``[n >= 2, J, 7]`` to a mean speed, in place."""
    if target_poses.ndim != 3 or target_poses.shape[0] < 2 or target_poses.shape[2] != 7:
        raise ValueError("target_poses must contain at least two poses.")
    _check_speed(desired_speed)
    start = target_poses[0, 0, :3].copy()
    scale = _ground_speed_scale(start, target_poses[-1, 0, :3], target_poses.shape[0], desired_speed)
    target_poses[1:, 0, :2] = start[:2] + (target_poses[1:, 0, :2] - start[:2]) * scale


def adjust_target_dof_speed(target_joint_q: np.ndarray, desired_speed: float, stream=None) -> None:
    """The same for joint coordinates ``[n >= 2, 36]``; height, rotation and angles are kept."""
    if target_joint_q.ndim != 2 or target_joint_q.shape[0] < 2 or target_joint_q.shape[1] < ROOT_COORDINATE_COUNT:
        raise ValueError("target_joint_q must contain at least two poses with root XYZ and XYZW values.")
    _check_speed(desired_speed)
    start = target_joint_q[0, :3].copy()
    scale = _ground_speed_scale(start, target_joint_q[-1, :3], target_joint_q.shape[0], desired_speed)
    target_joint_q[1:, :2] = start[:2] + (target_joint_q[1:, :2] - start[:2]) * scale


def _ground_placement(ground_position, facing_direction) -> np.ndarray:
    yaw = F(math.atan2(float(facing_direction[1]), float(facing_direction[0])))
    return T.transform(np.asarray(ground_position, F), T.quat_from_axis_angle([0.0, 0.0, 1.0], yaw))


def place_target_poses(ground_position, facing_direction, blended_poses, target_poses: np.ndarray | None = None, stream=None) -> np.ndarray:
    """Move the roots of local poses ``[n, J, 7]`` to a ground position, turned to face a direction (Z up)."""
    out = np.array(blended_poses, F, copy=True)
    out[:, 0] = T.transform_mul(_ground_placement(ground_position, facing_direction), out[:, 0])
    if target_poses is not None:
        target_poses[...] = out
        return target_poses
    return out


def place_target_dof(ground_position, facing_direction, blended_joint_q, target_joint_q: np.ndarray | None = None, stream=None) -> np.ndarray:
    """The same for joint coordinates ``[n, 36]``; the joint angles are kept."""
    out = np.array(blended_joint_q, F, copy=True)
    write_root(T.transform_mul(_ground_placement(ground_position, facing_direction), read_root(out)), out)
    if target_joint_q is not None:
        target_joint_q[...] = out
        return target_joint_q
    return out
