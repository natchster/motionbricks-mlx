# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (runtime/motion_player.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Motion players: generate from recorded poses, play the result, record what was shown."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .animation import frame_index_from_time
from .contract import MotionBricksInferenceConfig
from .target_pose import TargetPoseInferenceParams
from .tokens import MotionRecorder


class _GeneratedMotionPlayer:
    def __init__(self, engine, context, target, output, motion_history: MotionRecorder) -> None:
        self.engine = engine
        self.context = context
        self.target = target
        self.output = output
        self.motion_history = motion_history
        self.playback_time = 0.0
        # The current target's transition point, in frames before the end of its motion.
        self.transition_early_by_frames = 0
        self._generated_root_positions = np.zeros((1, 3), np.float32)

    @property
    def playback_duration(self) -> float:
        """Playback time of the transition point."""
        end_frame = max(0, self.output.num_frames - 1 - self.transition_early_by_frames)
        return end_frame / self.output.sample_rate

    @property
    def time_to_transition(self) -> float:
        """Seconds until the transition point. Sampling can continue past it."""
        return max(0.0, self.playback_duration - self.playback_time)

    def generate(self, target_poses, inference_params: TargetPoseInferenceParams, *,
                 config: MotionBricksInferenceConfig | None = None) -> None:
        """Generate toward a target from the recorded history, then restart playback.

        ``inference_params.num_poses_to_use`` target poses inform the model, its
        ``transition_early_by_frames`` sets the transition point, and its ``allowed_token_counts``
        (when set) replaces the counts of ``config``.
        """
        cfg = config if config is not None else MotionBricksInferenceConfig()
        if inference_params.allowed_token_counts is not None:
            cfg = replace(cfg, allowed_token_counts=inference_params.allowed_token_counts)
        self.target.update_from_poses(np.asarray(target_poses)[: inference_params.num_poses_to_use])
        self.transition_early_by_frames = inference_params.transition_early_by_frames
        self.refresh_context()
        self.engine.generate_animation(self.context, self.target, self.output, config=cfg)
        self.playback_time = 0.0
        self._generated_root_positions = self.output.root_transforms()[:, :3]

    def sample(self, delta_time: float, destination_pose: np.ndarray, destination_foot_contacts: np.ndarray) -> None:
        """Advance playback, then write the pose and its four foot contacts in place."""
        if not math.isfinite(delta_time) or delta_time < 0.0:
            raise ValueError("delta_time must be finite and non-negative.")
        self.playback_time += delta_time
        self.output.sample(self.playback_time, destination_pose)
        self.output.sample_foot_contacts(self.playback_time, destination_foot_contacts)

    def record_pose(self, delta_time: float, pose) -> None:
        """Record the pose that was shown, for the next generation's context."""
        self.motion_history.record(delta_time, pose)

    def seed_history_from_pose(self, pose) -> None:
        """Fill every history slot and the context with one pose."""
        self.motion_history.fill(pose)
        self.refresh_context()

    def refresh_context(self) -> None:
        self.context.update_from_recorder(self.motion_history)

    def current_root_position(self) -> np.ndarray:
        """The generated root position at the playback time."""
        positions = self._generated_root_positions
        frame, nxt, blend = frame_index_from_time(self.playback_time, self.output.sample_rate, len(positions))
        return (positions[frame] * (1.0 - blend) + positions[nxt] * blend).astype(np.float32)


class GeneratedMotionPlayer(_GeneratedMotionPlayer):
    """Play SOMA motion generated from recorded poses and targets."""

    def __init__(self, engine, skeleton, *, history_capacity: int = 32) -> None:
        super().__init__(engine, engine.create_input(skeleton), engine.create_input(skeleton), engine.create_output(skeleton),
                         MotionRecorder(skeleton.reference_local_transforms, capacity=history_capacity))


class GeneratedDOFMotionPlayer(_GeneratedMotionPlayer):
    """Play G1 motion generated from recorded poses and targets."""

    def __init__(self, engine, reference_joint_q, *, history_capacity: int = 32) -> None:
        super().__init__(engine, engine.create_input(), engine.create_input(), engine.create_output(),
                         MotionRecorder(reference_joint_q, capacity=history_capacity))
