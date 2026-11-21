# SPDX-License-Identifier: Apache-2.0
# Ported from NVlabs/motionbricks docs/examples/locomotion_loop.py. Original: Copyright (c) 2026
# NVIDIA CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
"""Move a character along a steered path with a generation loop. No viewer, no apps.

This is the continuous locomotion loop of the Runtime API page, made complete. Each update plays
the generated motion, records the pose it showed, and decides whether to generate again. The
target poses are one set of jog poses, placed ahead of the character along the steering direction.
The steering turns from +X to +Y, so the path bends left. Run it from the repository root:

    uv run --package motionbricks-mlx python model/examples/locomotion_loop.py
    uv run --package motionbricks-mlx python model/examples/locomotion_loop.py --model unitree_g1
"""

import argparse
import math
import time

import numpy as np

from motionbricks_mlx import runtime as rt
from motionbricks_mlx.runtime import transforms as T

from _rig import JOG, load_rig, root_positions, sample_window

UPDATE_RATE = 60.0
DURATION = 9.0
# The jog poses that serve as the target. Any clip time in the jog works.
TARGET_TIME = 1.8
# The ground speed to move at, in m/s. The target poses are scaled to it.
DESIRED_SPEED = {"soma": 1.9, "unitree_g1": 1.45}
# The target sits this many seconds of travel ahead of the character.
LEAD_SECONDS = 1.0
# The transition point, in frames before the end of each generation.
TRANSITION_EARLY_BY = 4
# Generate again when the steering direction turns by more than this many degrees.
TURN_THRESHOLD = 15.0
# The steering direction. It holds +X, turns to +Y between these two times, then holds.
TURN_START, TURN_END = 2.0, 4.0


def steering(t: float) -> np.ndarray:
    """The desired ground direction at a time. An application reads its controls here."""
    blend = min(max((t - TURN_START) / (TURN_END - TURN_START), 0.0), 1.0)
    angle = 0.5 * math.pi * blend
    return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)


def canonical_target(rig, model: str) -> np.ndarray:
    """The jog target at the origin, facing +X, at the model's desired speed."""
    roots = root_positions(sample_window(rig, TARGET_TIME))
    offset = roots[-1, :2] - roots[0, :2]
    heading = math.atan2(offset[1], offset[0])
    origin = T.transform([roots[0, 0], roots[0, 1], 0.0], T.quat_from_axis_angle([0.0, 0.0, 1.0], np.float32(heading)))
    poses = sample_window(rig, TARGET_TIME, root_offset=T.transform_inverse(origin))
    (rt.adjust_target_pose_speed if rig.skeleton is not None else rt.adjust_target_dof_speed)(poses, DESIRED_SPEED[model])
    return poses


def run(model: str, dtype: str = "float32") -> list[np.ndarray]:
    """Move "soma" or "unitree_g1" along the steered path. Return the pose shown at each update."""
    rig = load_rig(model, JOG, dtype=dtype)
    shown = []
    with rig.engine as engine:
        if rig.skeleton is not None:
            player = rt.GeneratedMotionPlayer(engine, rig.skeleton)
            place = rt.place_target_poses
        else:
            player = rt.GeneratedDOFMotionPlayer(engine, rig.reference_pose)
            place = rt.place_target_dof
        jog_poses = canonical_target(rig, model)
        settings = rt.TargetPoseInferenceParams(transition_early_by_frames=TRANSITION_EARLY_BY, num_poses_to_use=4)

        # Start at the origin in the jog pose, as if the character had held it.
        current_pose = jog_poses[rt.MODEL_POSES_PER_TOKEN - 1].copy()
        player.seed_history_from_pose(current_pose)
        foot_contacts = np.zeros(4, np.float32)
        generated_direction = None
        generations, busy = 0, 0.0
        delta_time = 1.0 / UPDATE_RATE

        for step in range(round(DURATION * UPDATE_RATE)):
            t = step * delta_time
            direction = steering(t)
            position = root_positions(current_pose.reshape(1, -1))[0] * np.array([1.0, 1.0, 0.0], np.float32)
            turned = generated_direction is not None and math.degrees(
                math.acos(min(1.0, float(direction @ generated_direction)))) > TURN_THRESHOLD
            if generated_direction is None or turned or player.time_to_transition <= delta_time:
                ground_position = position + direction * DESIRED_SPEED[model] * LEAD_SECONDS
                start = time.perf_counter()
                player.generate(place(ground_position, direction, jog_poses), settings)
                busy += time.perf_counter() - start
                generated_direction = direction
                generations += 1

            player.sample(delta_time, current_pose, foot_contacts)
            # Apply target pose IK, physics, or other changes to current_pose here.
            player.record_pose(delta_time, current_pose)
            shown.append(current_pose.copy())

            if step % round(UPDATE_RATE) == 0:
                print(f"t = {t:4.1f} s  steering {direction[:2].round(2)}  root {position[:2].round(2)}")
        print(f"{generations} generations in {DURATION:.0f} s, {busy / generations * 1e3:.1f} ms each")
    return shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("soma", "unitree_g1"), default="soma")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    run(args.model, args.dtype)


if __name__ == "__main__":
    main()
