# SPDX-License-Identifier: Apache-2.0
# Ported from NVlabs/motionbricks docs/examples/generate_once.py. Original: Copyright (c) 2026
# NVIDIA CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
"""Generate motion between clip poses, then chain a second generation. No viewer, no apps.

Each generation takes four context poses and one to four target poses. The first context and both
targets come from one walk clip. The second generation reads its context from the last four poses
of the first, which is the loop that makes continuous motion. Run it from the repository root:

    uv run --package motionbricks-mlx python model/examples/generate_once.py
    uv run --package motionbricks-mlx python model/examples/generate_once.py --model unitree_g1 --dtype bfloat16
"""

import argparse

import numpy as np

from motionbricks_mlx import runtime as rt

from _rig import load_rig, sample_window

# The context ends at this clip time. Each generation targets the clip poses at the next time.
CONTEXT_END = 1.0
TARGET_TIMES = (1.8, 3.4)


def generate(model: str, dtype: str = "float32") -> list[np.ndarray]:
    """Run both generations for "soma" or "unitree_g1". Return the valid poses of each."""
    rig = load_rig(model, dtype=dtype)
    motions = []
    with rig.engine as engine:
        skeleton_args = (rig.skeleton,) if rig.skeleton is not None else ()
        context = engine.create_input(*skeleton_args)
        target = engine.create_input(*skeleton_args)
        output = engine.create_output(*skeleton_args)

        # The context is the four poses that end at CONTEXT_END.
        context_poses = sample_window(rig, CONTEXT_END - (rt.MODEL_POSES_PER_TOKEN - 1) / rt.MODEL_SAMPLE_RATE)
        for target_time in TARGET_TIMES:
            context.update_from_poses(context_poses)
            target.update_from_poses(sample_window(rig, target_time))
            engine.generate_animation(context, target, output)

            # The output buffer has room for the longest result. Only num_frames frames are valid.
            frames = output.num_frames
            motions.append(output.poses[:frames].copy())
            root = output.root_transforms()[:frames, :3]
            stages = ", ".join(f"{k} {v:.1f}" for k, v in engine.timer.milliseconds.items())
            print(f"generated {frames} frames ({frames / rt.MODEL_SAMPLE_RATE:.2f} s), "
                  f"root from {root[0].round(2)} to {root[-1].round(2)}  [ms: {stages}]")

            # Play the whole motion, so the last four poses are what the character showed.
            # An application that moves on early keeps the last four poses it played instead.
            context_poses = output.poses[frames - rt.MODEL_POSES_PER_TOKEN:frames].copy()
    return motions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("soma", "unitree_g1"), default="soma")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    generate(args.model, args.dtype)


if __name__ == "__main__":
    main()
