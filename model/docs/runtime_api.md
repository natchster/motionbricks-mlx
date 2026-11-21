# Runtime API (Python, `motionbricks_mlx.runtime`)

> Adapted from NVIDIA's [MotionBricks Runtime API documentation](https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/docs/runtime_api.md)
> (commit `54fc563`), © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: rewritten for the MLX port.
> TensorRT, CUDA streams, Warp arrays and the viewer apps are replaced by MLX inference and NumPy
> arrays. The demo-runtime sections are dropped, and the port's own setup, differences and
> verification are added. The Rust version is [model-rs/docs/runtime_api.md](../../model-rs/docs/runtime_api.md).

This page shows how to use MotionBricks in your own application through the Runtime API: set up
an engine, prepare poses, generate, and chain generations. The API keeps upstream's names and
behavior (`MotionBricksSOMAEngine`, `PoseToken`, `AnimationBuffer`, `MotionRecorder`,
`GeneratedMotionPlayer`, ...). Its engines run the MLX model instead of a TensorRT plan, and its
poses are NumPy arrays instead of Warp arrays.

```sh
uv sync --package motionbricks-mlx
uv run --package motionbricks-mlx motionbricks-mlx download --clips   # checkpoints + the sample clips of the examples
```

## Overview

### Generation flow

MotionBricks builds continuous motion one generation at a time. Each generation takes four context
poses and one to four target poses. Your app plays the generated motion up to the transition point
and records every pose it shows. The next generation reads its context from those records and
generates toward a new target.

| Step | API | Section |
| --- | --- | --- |
| Prepare poses | `load_bvh()`, `load_csv()`, `sample()` with `root_offset`, or poses from your application | [Place poses in the scene](#place-poses-in-the-scene), [Clips](#clips) |
| Fill the tokens | `create_input()`, `update_from_poses()`, `update_from_recorder()` | [Engines](#engines), [Record what you show](#record-what-you-show) |
| Generate | `generate_animation()`, `MotionBricksInferenceConfig`, `TargetPoseInferenceParams` | [Generate](#generate) |
| Play | `sample()`, `output.num_frames`, `sample_foot_contacts()` | [Engines](#engines), [Foot contacts](#foot-contacts) |
| Record and chain | `MotionRecorder`, `GeneratedMotionPlayer`, `time_to_transition` | [Chaining generations](#chaining-generations) |

### Runnable examples

Two scripts in [model/examples](../examples) call the API without a viewer, for either rig:

```sh
uv run --package motionbricks-mlx python model/examples/generate_once.py
uv run --package motionbricks-mlx python model/examples/locomotion_loop.py --model unitree_g1 --dtype bfloat16
```

- [generate_once.py](../examples/generate_once.py) generates from clip poses, then chains a second
  generation from the last four poses it played.
- [locomotion_loop.py](../examples/locomotion_loop.py) runs the
  [continuous locomotion loop](#a-continuous-locomotion-loop) with a motion player. It steers a jog
  around a left turn, scales the target poses to the desired speed, and places them ahead of the
  character.

[_rig.py](../examples/_rig.py) holds the setup both share.

### Modules

| Module | Contents |
| --- | --- |
| `runtime.engine` | `MotionBricksSOMAEngine`, `MotionBricksUnitreeG1Engine`: encoding, inference, decoding, endpoint correction |
| `runtime.tokens` | `PoseToken`, `DOFToken`, `MotionRecorder`, `history_sample_indices()` |
| `runtime.animation` | `Skeleton`, `AnimationBuffer`, `DOFAnimationBuffer`, foot contact sampling |
| `runtime.player` | `GeneratedMotionPlayer`, `GeneratedDOFMotionPlayer` |
| `runtime.target_pose` | `TargetPoseInferenceParams`, `IKPin`, `TargetPoseIKCorrector`, `place_target_poses()`, `adjust_target_pose_speed()`, and the G1 variants |
| `runtime.io` | `load_bvh()`, `load_skeleton_from_bvh()`, `load_csv()`, `save_csv()`, `SpaceConverter`, `model_to_runtime()` |
| `runtime.contract` | timing constants, joint layouts, `MotionBricksInferenceConfig`, `valid_frame_count()` |
| `runtime.transforms` | NumPy transforms and quaternions with Warp's conventions |

Everything is exported from `motionbricks_mlx.runtime`.

## Engines

An engine wraps a loaded model: any precision, `mb.load(...)` or `.quantize(8)`. It aligns the
context and target for inference, runs the model as one compiled MLX graph, and restores the world
placement afterward. It also removes the three [boundary frames](#generation-length) at each end of
the generated motion. Use it as a context manager, or call `close()` when you are done.

```python
import motionbricks_mlx as mb
from motionbricks_mlx import runtime as rt

skeleton = rt.load_skeleton_from_bvh("checkpoints/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh", remove_root=True)
with rt.MotionBricksSOMAEngine(mb.load("soma", "checkpoints/MotionBricks/soma", dtype="bfloat16"),
                               model_to_runtime=rt.model_to_runtime("soma")) as engine:
    context = engine.create_input(skeleton)
    target = engine.create_input(skeleton)
    output = engine.create_output(skeleton)

    context.update_from_poses(context_poses)   # four poses, [4, J, 7]
    target.update_from_poses(target_poses)     # one to four poses
    engine.generate_animation(context, target, output)

    # output.num_frames marks the valid range of the reused buffer.
    current_pose = output.sample(time)
```

```python
with rt.MotionBricksUnitreeG1Engine.from_checkpoints("checkpoints/MotionBricks/g1", dtype="float32") as engine:
    context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
    context.update_from_poses(context_joint_q)   # four poses, [4, 36]
    target.update_from_poses(target_joint_q)     # one to four poses
    engine.generate_animation(context, target, output)
    robot_joint_q = output.sample(time)
```

| | `MotionBricksSOMAEngine` | `MotionBricksUnitreeG1Engine` |
| --- | --- | --- |
| Context and target tokens | `PoseToken`, local joint transforms | `DOFToken`, joint coordinates |
| Output buffer | `AnimationBuffer`, the same type as loaded BVH clips | `DOFAnimationBuffer`, the same type as loaded CSV clips |
| Pose | `[J, 7]` float32: per joint an XYZ position and an XYZW quaternion, root first | `[36]` float32: root XYZ, root XYZW quaternion, then 29 joint angles (rad) |
| Model space | Y up, with XZ as the ground plane | Z up, with +X forward |
| World space | set by `model_to_runtime` | set by `model_to_runtime` |

- **SOMA joints:** a SOMA engine finds the model's 30 joints in a skeleton by name, and the root
  (`Hips`) must be at index zero. The skeleton can hold more joints, such as the fingers, which keep
  their reference transforms in generated motion. Upstream reads its skeleton from
  `soma_base_skel_minimal_no_root.usd`. Here `load_skeleton_from_bvh(path, remove_root=True)` builds
  the same `Hips`-rooted skeleton from any SOMA BVH clip, without USD. The clips' identity `Root`
  joint is dropped.
- **G1 joints:** the G1 engine works directly on the robot's joint coordinates.
- **Backends:** the `model` argument is anything with `run(inputs) -> outputs` on the ONNX tensor
  names, so another backend (ONNX Runtime, a server) can stand in. `model=None` builds an engine
  whose encode and decode stages still work, which the tests use.
- **Stages:** each stage is a public method, which makes them easy to inspect or replace:
  `encode_inputs()`, `run_model()`, `decode_prediction()`, `correct_endpoints()`.
  `engine.timer.milliseconds` holds the time of each stage of the last generation. On an
  M4 Pro, encode and decode take a few ms for SOMA and well under 1 ms for the G1. FP32 inference
  takes 11–15 ms at batch 1 after the first call, which also compiles the graph. The Rust runtime
  needs about 7 ms for the whole generation.
- **Source compatibility:** `stream=` arguments and `synchronize()` are accepted so upstream code
  ports with few edits. Generation is synchronous here.

## Pose layouts and model space

The examples work in a Z-up world space. SOMA model space and the BVH clips are Y up. G1 model space
and the CSV clips are already Z up. `rt.model_to_runtime("soma")` turns Y up into Z up, following
the MuJoCo convention, so SOMA faces the same way as the G1. `rt.model_to_runtime("unitree_g1")` is
the identity. Pass the transform to the engine and to clip sampling, so generated motion and clips
land in the same space. Both engines default to the identity, which suits an app that already works
in model space.

### Place poses in the scene

Your app places every context and target pose in world space. To place a pose, move its root by one
world transform and keep the other joint values. For SOMA the root is the first local joint
transform. For the G1 it is the first seven joint coordinates. When you sample a clip, pass the
placement as `root_offset`. A clip in source coordinates also needs `model_to_runtime`, applied
first:

```python
pose = clip.sample(sample_time, root_offset=rt.transforms.transform_mul(placement, model_to_runtime))
```

For poses you already have, `place_target_poses()` and `place_target_dof()` move the roots to a ground
position, turned to a facing direction. Because only the root moves, one set of sampled poses serves
every placement.

### Clips

`load_bvh()` reads SOMA BVH clips. Positions are converted from cm to m, and rotations are applied
in each joint's channel order. `load_csv()` reads G1 CSV clips, which store the root in centimeters
and angles in degrees at 120 Hz, and converts them to meters, XYZW quaternions and radians.
`load_bvh(path, skeleton)` remaps a clip onto a skeleton by joint name. Clips are only one source of
poses: any pose with the right layout works as a context or a target.

`motionbricks-mlx download --clips` fetches the four sample clips the examples use. They come from
upstream's repository at the pinned commit, and each one is SHA-256 verified. The clips are NVIDIA
sample data under the NVIDIA Sample Data Evaluation License and are not redistributed here.

## Generate

`generate_animation()` reads the context and target tokens and fills the output buffer with the
generated motion and its foot contacts.

### Generation settings

`generate_animation()` accepts an optional `MotionBricksInferenceConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `allowed_token_counts` | 8 to 14 | Lengths the model may choose, from 6 to 16 tokens of four poses |
| `correct_generated_diff` | `True` | Spreads small corrections over the motion so it starts at the last context pose and ends at the target |
| `heading_alignment` | (1, 0) | The ground direction in model space that inference turns the last context pose to face: (x, z) for SOMA and (x, y) for the G1 |

### Settings for each target

Each target carries its own settings in `TargetPoseInferenceParams`. The motion players read them in
`generate()`.

| Field | Effect |
| --- | --- |
| `num_poses_to_use` | How many of the target's four poses inform the model, 1 to 4. One pose sets the destination, with the posture and facing at which the character stops. A second adds velocity; a third and fourth add acceleration and momentum. |
| `transition_early_by_frames` | Moves the transition point this many frames before the end of the generated motion. A larger value makes the target a guide. |
| `allowed_token_counts` | Replaces the generation settings' counts for this target. A single count gives exact control over the length. |
| `ik_pin_mask` | SOMA only. `TargetPoseIKCorrector` eases the pinned hands and feet onto the target with two-bone IK as playback nears it. |

### Generation length

The model decides how long each transition takes. Each token holds four poses at 30 Hz, and one
generation produces 6 to 16 tokens, or 24 to 64 raw frames. The first token holds the context and
the last the target. The engine removes three boundary frames at each end, so the motion starts at
the last context pose and ends at the first target pose. That leaves 18 to 58 valid frames
(`valid_frame_count()`), or 0.6 to 1.9 s. `output.num_frames` holds the valid frame count of the
latest generation.

### Foot contacts

Both models predict four foot contacts per frame: left heel, left toe, right heel, right toe. Each is
a 0..1 confidence that the heel or toe is planted, and the engines clamp the raw values to that
range. They are SOMA's `LeftFoot`, `LeftToeBase`, `RightFoot` and `RightToeBase` joints
(`SOMA_FOOT_CONTACT_JOINTS`). For the G1 they are the ankle roll joints, plus points 0.14 m ahead and
0.035 m below them (`UNITREE_G1_FOOT_CONTACT_OFFSETS`). Generated motion keeps them in
`output.foot_contacts` (`[capacity, 4]`, the first `num_frames` valid), and
`output.sample_foot_contacts(time)` blends them like `sample()`. Clips loaded from files have
`foot_contacts = None`.

## Chaining generations

A single generation is short. Continuous motion comes from chaining generations. Each new context
must hold the poses that were actually shown. Otherwise the next generation starts from another pose,
and the character jumps.

### Chain by hand

At each transition point, copy the last four played frames of the generated motion into the context.
`generate_once.py` does this:

```python
context_poses = output.poses[output.num_frames - 4: output.num_frames].copy()
```

This works when each segment plays to its end and nothing changes the pose on the way.

### Record what you show

`MotionRecorder` keeps a fixed number of recent poses, oldest first, and the time from each pose to
the next. Record one pose at each update of your app; the update rate does not need to match the
model's 30 Hz:

```python
history = rt.MotionRecorder(skeleton.reference_local_transforms, capacity=32)
history.fill(start_pose)                       # as if the character had held it

history.record(delta_time, current_pose)       # at each update, after every change to the pose
context.update_from_recorder(history)          # before each generation
```

`update_from_recorder()` samples four context poses at 30 Hz, ending at the latest recorded pose,
and blends the two recorded poses around each sample time. Record the pose after every change your
app makes to it, such as clip frames, IK or physics. That lets a generation start from a clip or
from the middle of a segment.

### Play generated motion

`GeneratedMotionPlayer` (SOMA) and `GeneratedDOFMotionPlayer` (G1) pair an engine with a recorder:

| Member | Use |
| --- | --- |
| `seed_history_from_pose(pose)` | Fills the history and the context with one pose. Call it before the first generation. |
| `generate(target_poses, inference_params, config=None)` | Reads the context from the history, generates toward the target, and restarts playback |
| `sample(delta_time, pose, foot_contacts)` | Advances the playback time, then writes the pose and its foot contacts in place |
| `record_pose(delta_time, pose)` | Records the pose that you showed |
| `time_to_transition` | Seconds left until the transition point |

The player does not decide when to generate. Your code calls `generate()`, usually when
`time_to_transition` drops below the next update's time step. Sampling can continue past the
transition point, and at the end of the valid frames the pose holds.

### A continuous locomotion loop

```python
player = rt.GeneratedMotionPlayer(engine, skeleton)
player.seed_history_from_pose(start_pose)
current_pose, foot_contacts = start_pose.copy(), np.zeros(4, np.float32)
settings = rt.TargetPoseInferenceParams(transition_early_by_frames=10, num_poses_to_use=4)


def update(delta_time, desired_velocity):
    if player.time_to_transition <= delta_time or input_changed(desired_velocity):
        player.generate(place_target(desired_velocity), settings)
    player.sample(delta_time, current_pose, foot_contacts)
    # Apply target pose IK, physics, or other changes to current_pose here.
    player.record_pose(delta_time, current_pose)
```

[locomotion_loop.py](../examples/locomotion_loop.py) runs this loop with `input_changed()` and
`place_target()` filled in. Generate at the transition point at the latest, and earlier when the
movement input changes enough. Place target poses that match the desired speed
(`adjust_target_pose_speed()`) ahead of the character along the desired velocity.

## Differences from upstream

| Upstream | Here |
| --- | --- |
| TensorRT plan on a CUDA stream, CUDA graph, plan cache | the MLX model, one `mx.compile` graph (any precision, INT8 via `quantize(8)`) |
| Warp arrays on a device | NumPy float32 arrays: transforms are `[..., 7]`, the `wp.transform` layout |
| `MotionBricksSOMAEngine(onnx_path, ...)` | `MotionBricksSOMAEngine(model, ...)` or `.from_checkpoints(dir, dtype=...)` |
| skeleton from `soma_base_skel_minimal_no_root.usd` | `load_skeleton_from_bvh(clip, remove_root=True)` (the same hierarchy, from the clip) |
| `place_target_poses(..., target_poses)` writes into its last argument | also returns the placed poses (and writes into `target_poses` when given) |
| renderers, blend spaces, Smart Primitives, the demo apps | not ported: the API covers generation, playback, recording, IK pins and clips |

## Verification

`model/tests/test_runtime.py` compares every step with upstream's own code:
- **Fixtures:** `sandbox/scripts/make_runtime_golden.py` runs upstream's engine methods, recorder,
  target helpers, IK corrector and clip loaders on Warp's CPU backend. NVIDIA's PyTorch model stands
  in for TensorRT. The script records their inputs and outputs for four generations per rig: a
  default walk, a far-away one-pose target with a custom heading and allowed lengths, no endpoint
  correction, and a non-identity `model_to_runtime`.
- **Teacher-forced checks:** each step is checked against those fixtures. Encoding, decoding, endpoint
  correction, clip loading, recorder sampling, target placement and speed scaling, buffer sampling
  and looping all agree with upstream within 2e-5, and IK pins within 5e-5.
- **End to end:** every generation picks the same duration as upstream, and its poses stay within
  3e-3, the model's float noise after the rotation decoding.

The [Rust runtime](../../model-rs/docs/runtime_api.md) passes the same checks against the same
fixtures.
