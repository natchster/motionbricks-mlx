# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (motionbricks/model,
# motionbricks/runtime, motionbricks/animation, motionbricks/io).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""The MotionBricks Runtime API on MLX: engines, tokens, animation buffers, recorders, motion
players, target-pose helpers and clip loading, with upstream's names and behavior.

Poses are NumPy arrays: SOMA poses are local joint transforms ``[J, 7]`` (position, then XYZW
rotation, root first); G1 poses are 36 joint coordinates. See ``docs/runtime_api.md``.
"""

from . import transforms
from .animation import AnimationBuffer, DOFAnimationBuffer, Skeleton, create_animation_buffer_for_skeleton, frame_index_from_time
from .contract import (
    DEFAULT_TOKEN_COUNTS,
    FOOT_CONTACT_COUNT,
    MODEL_BOUNDARY_FRAMES,
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MAX_VALID_FRAMES,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    ROOT_COORDINATE_COUNT,
    SOMA_FOOT_CONTACT_JOINTS,
    SOMA_HEADING_JOINTS,
    SOMA_JOINT_NAMES,
    UNITREE_G1_FOOT_CONTACT_BODIES,
    UNITREE_G1_FOOT_CONTACT_OFFSETS,
    UNITREE_G1_JOINT_NAMES,
    ModelRig,
    MotionBricksInferenceConfig,
    valid_frame_count,
)
from .engine import MotionBricksEngine, MotionBricksSOMAEngine, MotionBricksUnitreeG1Engine
from .io import FacingDirectionType, SpaceConverter, load_bvh, load_csv, load_skeleton_from_bvh, model_to_runtime, save_csv
from .player import GeneratedDOFMotionPlayer, GeneratedMotionPlayer
from .target_pose import (
    IK_PINS,
    MAX_TRANSITION_EARLY_BY_FRAMES,
    IKPin,
    TargetPoseIKCorrector,
    TargetPoseInferenceParams,
    TwoBoneIKChainConfig,
    adjust_target_dof_speed,
    adjust_target_pose_speed,
    create_soma_target_pose_ik_rig,
    place_target_dof,
    place_target_poses,
)
from .tokens import DOFToken, MotionRecorder, PoseToken, history_sample_indices

__all__ = [
    "DEFAULT_TOKEN_COUNTS",
    "FOOT_CONTACT_COUNT",
    "IK_PINS",
    "MAX_TRANSITION_EARLY_BY_FRAMES",
    "MODEL_BOUNDARY_FRAMES",
    "MODEL_MAX_TOKEN_COUNT",
    "MODEL_MAX_VALID_FRAMES",
    "MODEL_MIN_TOKEN_COUNT",
    "MODEL_POSES_PER_TOKEN",
    "MODEL_SAMPLE_RATE",
    "ROOT_COORDINATE_COUNT",
    "SOMA_FOOT_CONTACT_JOINTS",
    "SOMA_HEADING_JOINTS",
    "SOMA_JOINT_NAMES",
    "UNITREE_G1_FOOT_CONTACT_BODIES",
    "UNITREE_G1_FOOT_CONTACT_OFFSETS",
    "UNITREE_G1_JOINT_NAMES",
    "AnimationBuffer",
    "DOFAnimationBuffer",
    "DOFToken",
    "FacingDirectionType",
    "GeneratedDOFMotionPlayer",
    "GeneratedMotionPlayer",
    "IKPin",
    "ModelRig",
    "MotionBricksEngine",
    "MotionBricksInferenceConfig",
    "MotionBricksSOMAEngine",
    "MotionBricksUnitreeG1Engine",
    "MotionRecorder",
    "PoseToken",
    "Skeleton",
    "SpaceConverter",
    "TargetPoseIKCorrector",
    "TargetPoseInferenceParams",
    "TwoBoneIKChainConfig",
    "adjust_target_dof_speed",
    "adjust_target_pose_speed",
    "create_animation_buffer_for_skeleton",
    "create_soma_target_pose_ik_rig",
    "frame_index_from_time",
    "history_sample_indices",
    "load_bvh",
    "load_csv",
    "load_skeleton_from_bvh",
    "model_to_runtime",
    "place_target_dof",
    "place_target_poses",
    "save_csv",
    "transforms",
    "valid_frame_count",
]
