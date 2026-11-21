# SPDX-License-Identifier: Apache-2.0
# Ported from NVlabs/motionbricks docs/examples/_rig.py. Original: Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
"""Shared setup for the examples: a rig's skeleton, engine and clip, without a viewer.

Needs the checkpoints and the sample clips: `motionbricks-mlx download --clips`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import motionbricks_mlx as mb
from motionbricks_mlx import runtime as rt

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"
CLIPS = ROOT / "checkpoints" / "motions"
WALK = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057"
JOG = "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534"


@dataclass
class Rig:
    """What an example needs from one rig.

    Attributes:
        engine: The engine. Use it as a context manager.
        clip: The clip that the example samples poses from.
        skeleton: The SOMA skeleton, or None for the Unitree G1.
        reference_pose: A pose with the rig's layout.
        model_to_runtime: Transform from the clip's source space to world space, which is Z up.
    """

    engine: rt.MotionBricksSOMAEngine | rt.MotionBricksUnitreeG1Engine
    clip: object
    skeleton: object
    reference_pose: np.ndarray
    model_to_runtime: np.ndarray


def load_rig(model: str, clip_name: str = WALK, dtype: str = "float32") -> Rig:
    """Load a clip of the bundled locomotion set and create the engine for "soma" or "unitree_g1"."""
    folder = mb.hub.checkpoint_dir(model, CHECKPOINTS)
    if rt.ModelRig(model) is rt.ModelRig.SOMA:
        path = CLIPS / "soma" / f"{clip_name}.bvh"
        # The BVH skeleton without its identity Root joint: Hips first, as the engine needs.
        skeleton = rt.load_skeleton_from_bvh(path, remove_root=True)
        clip = rt.load_bvh(path, skeleton)
        # SOMA model space and BVH clips are Y up. World space is Z up.
        to_runtime = rt.model_to_runtime("soma")
        engine = rt.MotionBricksSOMAEngine.from_checkpoints(folder, dtype=dtype, model_to_runtime=to_runtime)
        return Rig(engine, clip, skeleton, skeleton.reference_local_transforms, to_runtime)
    clip = rt.load_csv(CLIPS / "unitree_g1" / f"{clip_name}.csv")
    # G1 model space and CSV clips are already Z up.
    engine = rt.MotionBricksUnitreeG1Engine.from_checkpoints(folder, dtype=dtype)
    return Rig(engine, clip, None, clip.sample(0.0), rt.transforms.IDENTITY.copy())


def sample_window(rig: Rig, start_time: float, root_offset=None) -> np.ndarray:
    """Four clip poses at 30 Hz from start_time, their roots moved by root_offset."""
    offset = rt.transforms.transform_mul(root_offset, rig.model_to_runtime) if root_offset is not None else rig.model_to_runtime
    return np.stack([rig.clip.sample(start_time + i / rt.MODEL_SAMPLE_RATE, root_offset=offset)
                     for i in range(rt.MODEL_POSES_PER_TOKEN)])


def root_positions(poses: np.ndarray) -> np.ndarray:
    """The root XYZ of each pose of either layout. Both start with the root position."""
    return poses.reshape(poses.shape[0], -1)[:, :3]
