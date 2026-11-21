# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (motionbricks/model/contract.py,
# config.py, soma/contract.py, unitree_g1/contract.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Rigs, timing, joint layouts and generation settings shared by the engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

MODEL_SAMPLE_RATE = 30.0
MODEL_POSES_PER_TOKEN = 4
MODEL_CONTEXT_AND_TARGET_POSES = 2 * MODEL_POSES_PER_TOKEN
MODEL_MIN_TOKEN_COUNT = 6
MODEL_MAX_TOKEN_COUNT = 16
MODEL_BOUNDARY_FRAMES = MODEL_POSES_PER_TOKEN - 1
FOOT_CONTACT_COUNT = 4
ROOT_COORDINATE_COUNT = 7  # root position (XYZ) and rotation (XYZW) at the start of joint coordinates


def valid_frame_count(token_count: int) -> int:
    """Frames left after removing the boundary frames: 18 to 58 for 6 to 16 tokens."""
    if not MODEL_MIN_TOKEN_COUNT <= token_count <= MODEL_MAX_TOKEN_COUNT:
        raise ValueError(f"token_count must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}, got {token_count}.")
    return token_count * MODEL_POSES_PER_TOKEN - 2 * MODEL_BOUNDARY_FRAMES


MODEL_MAX_VALID_FRAMES = valid_frame_count(MODEL_MAX_TOKEN_COUNT)


class ModelRig(StrEnum):
    """SOMA poses are local joint transforms; Unitree G1 poses are joint coordinates."""

    SOMA = "soma"
    UNITREE_G1 = "unitree_g1"


SOMA_JOINT_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)
SOMA_FOOT_CONTACT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
# The right and left leg joints whose positions give a pose's heading, as in training.
SOMA_HEADING_JOINTS = ("RightLeg", "LeftLeg")

UNITREE_G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
UNITREE_G1_FOOT_CONTACT_BODIES = ("left_ankle_roll_link",) * 2 + ("right_ankle_roll_link",) * 2
UNITREE_G1_FOOT_CONTACT_OFFSETS = ((0.0, 0.0, 0.0), (0.14, 0.0, -0.035)) * 2

DEFAULT_TOKEN_COUNTS = tuple(range(8, 15))


@dataclass(frozen=True, slots=True)
class MotionBricksInferenceConfig:
    """Generation settings for output length, heading alignment and endpoint correction.

    Attributes:
        allowed_token_counts: Token counts the model may choose from, each 6 to 16 (default 8 to 14).
        correct_generated_diff: Spread small corrections over the motion so it starts at the last
            context pose and ends at the first target pose.
        heading_alignment: Ground direction in model space that inference turns the last context
            pose to face: (x, z) for SOMA, (x, y) for G1. (1, 0) faces +X.
    """

    allowed_token_counts: tuple[int, ...] = DEFAULT_TOKEN_COUNTS
    correct_generated_diff: bool = True
    heading_alignment: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if not self.allowed_token_counts:
            raise ValueError("allowed_token_counts cannot be empty.")
        if not all(MODEL_MIN_TOKEN_COUNT <= c <= MODEL_MAX_TOKEN_COUNT for c in self.allowed_token_counts):
            raise ValueError(f"allowed_token_counts values must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}.")
        if len(self.heading_alignment) != 2:
            raise ValueError("heading_alignment must contain two values.")
        if not np.isfinite(self.heading_alignment).all() or np.linalg.norm(self.heading_alignment) <= 1.0e-6:
            raise ValueError("heading_alignment must be a finite, non-zero direction.")

    def allowed_token_bits(self) -> int:
        """The allowed token counts as a bit mask, bit 0 for 6 tokens."""
        bits = 0
        for value in self.allowed_token_counts:
            bits |= 1 << (value - MODEL_MIN_TOKEN_COUNT)
        return bits
