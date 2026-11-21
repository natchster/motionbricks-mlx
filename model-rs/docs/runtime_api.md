# Runtime API (Rust, `motionbricks_rs::runtime`)

> Adapted from NVIDIA's [MotionBricks Runtime API documentation](https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/docs/runtime_api.md)
> (commit `54fc563`), © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: rewritten for the Rust port.
> TensorRT, CUDA streams, Warp arrays and the viewer apps are replaced by the Metal runtime or
> mlx-rs and plain Rust values. The demo-runtime sections are dropped, and the port's setup,
> differences and verification are added. The Python version, with the full explanation of each
> concept, is [model/docs/runtime_api.md](../../model/docs/runtime_api.md).

The Runtime API turns MotionBricks into continuous character motion. It covers:
- engines that generate from four context poses and one to four target poses;
- tokens, animation buffers with sampling and foot contacts;
- a recorder of shown poses, motion players;
- target placement, speed scaling and SOMA IK pins;
- BVH/CSV clip loading.

It follows upstream's library, and engines run on a `Backend`:

| Backend | What it is | Generation (batch 1, M4 Pro) |
| --- | --- | --- |
| `metal::model::MetalModel` | the model-specific Metal runtime: FP32, FP16, BF16 or INT8 | about 7 ms in FP32 and 5 ms in BF16, including encode, decode and endpoint correction (the examples' loops) |
| `runtime::MlxBackend` | the mlx-rs port, one compiled graph | the same as Python MLX |
| your type | `impl Backend` (`predict` on the ONNX tensor names) | — |

```sh
scripts/download_weights.sh all && scripts/download_weights.sh clips   # checkpoints + the examples' sample clips
cargo run --release --example generate_once -- soma bfloat16
cargo run --release --example locomotion_loop -- unitree_g1
```

## Generation flow

Each generation takes four context poses and one to four target poses. Your app plays the generated
motion up to the transition point and records every pose it shows. The next generation reads its
context from those records.

| Step | API |
| --- | --- |
| Prepare poses | `load_bvh()`, `load_csv()`, `sample(time, root_offset)`, or poses from your application |
| Fill the tokens | `create_input()`, `update_from_poses()`, `update_from_recorder()` |
| Generate | `generate_animation()`, `MotionBricksInferenceConfig`, `TargetPoseInferenceParams` |
| Play | `sample()` or `sample_into()`, `output.num_frames()`, `sample_foot_contacts()` |
| Record and chain | `MotionRecorder`, `GeneratedMotionPlayer`, `time_to_transition()` |

## Engines

```rust
use std::sync::Arc;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::runtime::*;   // includes the MotionBricksEngine trait (generate_animation, stages)

let clip_path = "checkpoints/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh";
let skeleton = Arc::new(load_skeleton_from_bvh(clip_path, true, true)?);   // Hips first, no USD needed
let clip = load_bvh(clip_path, Some(skeleton.clone()), true)?;
let model = MetalModel::from_checkpoints_with("soma", "checkpoints/MotionBricks/soma", 1, "bfloat16")?;
let mut engine = MotionBricksSomaEngine::new(Some(model), model_to_runtime(ModelRig::Soma))?;
let (mut context, mut target, mut output) = (engine.create_input(&skeleton)?, engine.create_input(&skeleton)?, engine.create_output(&skeleton)?);

context.update_from_poses(&context_poses)?;   // four poses: Vec<Vec<Transform>>
target.update_from_poses(&target_poses)?;     // one to four poses
engine.generate_animation(&context, &target, &mut output, &MotionBricksInferenceConfig::default())?;
let pose = output.sample(time, Transform::IDENTITY)?;   // output.num_frames() marks the valid range
```

```rust
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 1)?;
let mut engine = MotionBricksUnitreeG1Engine::new(Some(model), Transform::IDENTITY)?;
let (mut context, mut target, mut output) = (engine.create_input(), engine.create_input(), engine.create_output());
context.update_from_poses(&context_joint_q)?;   // four poses: Vec<Vec<f32>> of 36 coordinates
target.update_from_poses(&target_joint_q)?;
engine.generate_animation(&context, &target, &mut output, &MotionBricksInferenceConfig::default())?;
let robot_joint_q = output.sample(time, Transform::IDENTITY)?;
```

| | `MotionBricksSomaEngine` | `MotionBricksUnitreeG1Engine` |
| --- | --- | --- |
| Tokens | `PoseToken`, local joint transforms | `DofToken`, joint coordinates |
| Output | `AnimationBuffer` (same type as BVH clips) | `DofAnimationBuffer` (same type as CSV clips) |
| Pose | `Vec<Transform>`: position + XYZW quaternion per joint, root (`Hips`) first | `Vec<f32>` of 36: root XYZ, root XYZW, 29 joint angles (rad) |
| Model space | Y up, XZ ground plane | Z up, +X forward |

- **Engine contract:** the engine aligns the context and target to a heading, encodes the ONNX
  inputs, runs the backend, decodes back to world space through `model_to_runtime`, drops the three
  boundary frames at each end, and corrects the endpoints.
- **SOMA skeleton:** the SOMA engine finds the model's 30 joints by name. Extra joints, such as the
  fingers, keep their reference transforms.
- **Stages:** both engines implement the `MotionBricksEngine` trait (upstream's base class), whose
  stages are public (`encode_inputs`, `decode_prediction`, `correct_endpoints`). `engine.timer()`
  holds the milliseconds of `encode`, `inference`, `decode` and `post_process`. Malformed
  predictions from a custom `Backend` (wrong row width, too few frames for the token count, missing
  foot contacts) are errors.
- **No backend:** `MotionBricksSomaEngine::without_backend(m2r)` (likewise for the G1) builds an
  engine whose encode and decode stages work without inference.
- **One model, many characters:** `Rc<RefCell<B>>` implements `Backend`, so several engines (and
  players) can share one loaded model: `MotionBricksSomaEngine::new(Some(shared.clone()), m2r)`.

## Spaces, placement and clips

- **Spaces:** `model_to_runtime(ModelRig::Soma)` turns SOMA's Y-up model space into Z up, following
  the MuJoCo convention, and `ModelRig::UnitreeG1` is the identity. Pass the same transform to the
  engine and to clip sampling (`clip.sample(t, placement * m2r)`; `*` composes transforms).
- **Placement:** to place a pose, move only its root. `place_target_poses()` and
  `place_target_dof()` do that for a ground position and facing direction.
- **Clips:** `load_bvh()` converts cm to m and applies rotations in each joint's channel order.
  `load_csv()` converts the G1's cm, degrees and 120 Hz Euler angles into meters, quaternions and
  radians. Malformed files are errors, never panics.
- **Skeletons:** `Skeleton::new` requires joint 0 to be the only root (parent < 0) and every other
  joint's parent to come before it, the order world-transform composition relies on. Skeletons are
  shared as `Arc<Skeleton>`, so clips and buffers can move between threads.
- **Sample clips:** `scripts/download_weights.sh clips` fetches the four clips the examples use from
  upstream at the pinned commit, SHA-256 verified. They are NVIDIA sample data under the NVIDIA
  Sample Data Evaluation License.

## Generate

`MotionBricksInferenceConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `allowed_token_counts` | 8 to 14 | lengths the model may choose (6 to 16 tokens of four poses) |
| `correct_generated_diff` | true | start at the last context pose, end at the target |
| `heading_alignment` | [1, 0] | ground direction in model space the last context pose is turned to face |

Settings for each target are in `TargetPoseInferenceParams`: `num_poses_to_use` (1 to 4),
`transition_early_by_frames`, `allowed_token_counts` (replaces the config's), and `ik_pin_mask`
(`ik_pin::LEFT_HAND | ...`, SOMA). The players don't apply IK themselves: pass the mask to
`TargetPoseIkCorrector::capture_pins(&target_pose, params.ik_pin_mask)` when you generate, and call
`corrector.apply(&mut pose, weight)` on each sampled pose before recording it.

A generation yields 6 to 16 tokens, which is 18 to 58 valid frames at 30 Hz after the boundary frames
(`valid_frame_count`). Each frame carries four foot contacts, the left/right heel and toe, as 0..1
confidences. They are in `output.foot_contacts()` and `output.sample_foot_contacts(t)`; clips have none.

## Chaining generations

Each new context must hold the poses that were actually shown:
- **By hand:** copy the last four played frames into the context (`generate_once.rs`).
- **With a recorder:** `MotionRecorder::record(dt, &pose)` at every update and
  `context.update_from_recorder(&history)` before each generation. The recorder resamples four
  poses at 30 Hz from the history.
- **With a player:** `GeneratedMotionPlayer` and `GeneratedDofMotionPlayer` (aliases of the generic
  `GeneratedPlayer<E>`) combine the two. They own their engine (`player.engine()`) and its tokens.
  If a generation fails, the playing motion, its clock and its transition point are kept. The
  playback clock is `f64`, like Python's:

```rust
let mut player = GeneratedMotionPlayer::new(engine, &skeleton, 32)?;
player.seed_history_from_pose(&start_pose)?;
let settings = TargetPoseInferenceParams::new(10, 4);
let mut pose = start_pose.clone();   // reused every update: sample_into and record_pose don't allocate
// each update:
if player.time_to_transition() <= dt as f64 || input_changed {
    player.generate(&place_target_poses(ground, direction, &jog_poses), &settings, None)?;
}
let contacts = player.sample_into(dt, &mut pose)?;   // or `let (pose, contacts) = player.sample(dt)?`
// apply IK, physics or other changes to `pose` here
player.record_pose(dt, &pose)?;
```

`locomotion_loop.rs` runs this loop and steers a jog around a left turn. It makes 14 generations in
9 s of motion, like the Python example, at about 7 ms each on the Metal runtime in FP32 and 5 ms in
BF16.

## Differences from upstream

| Upstream (Python, Warp) | Here |
| --- | --- |
| TensorRT plan on a CUDA stream | a `Backend`: `MetalModel` (any precision) or `MlxBackend` |
| `MotionBricksSOMAEngine`, `TargetPoseIKCorrector`, `TwoBoneIKChainConfig` | `MotionBricksSomaEngine`, `TargetPoseIkCorrector`, `TwoBoneIkChainConfig` (Rust naming); the other names are unchanged or `Dof` instead of `DOF` |
| Warp arrays | `Transform` (position + XYZW quaternion) and `Vec<f32>`; buffers store frames flat |
| `IKPin` flags enum | `ik_pin::*` `u32` constants |
| players hold a reference to the engine | players own the engine (`player.engine()`); share one model through `Rc<RefCell<B>>` |
| `sample(dt, destination_pose, destination_contacts)` writes in place | `sample_into(dt, &mut pose)` returns the contacts; `sample(dt)` returns a new pose |
| a failed generation empties the playing motion (`output.num_frames = 0`) | the playing motion, clock and transition point are kept (as in the Python port) |
| skeleton from USD | `load_skeleton_from_bvh(clip, true, true)` (the same hierarchy, without the identity `Root`) |
| renderers, blend spaces, Smart Primitives, demo apps | not ported |

## Verification

`tests/runtime.rs` runs the same checks as the Python suite, against the same upstream-generated
fixtures (`sandbox/scripts/make_runtime_golden.py`):
- **Teacher-forced steps:** encoding, decoding, endpoint correction, clip loading, recorder
  sampling, target placement and speed, and buffer sampling and looping agree with upstream's
  kernels within 2e-5, and IK pins within 5e-5.
- **End to end:** every generation on both the Metal runtime and mlx-rs picks upstream's duration,
  and its poses stay within 3e-3.
- **Weight-free unit tests:** transforms, tokens, the recorder, buffers, settings, skeleton
  validation, both players with fixed backends (including a refused generation), a shared backend,
  malformed predictions, degenerate and NaN IK poses, malformed BVH and CSV files, and a CSV round
  trip.
- **Missing inputs:** tests that need fixtures, clips or checkpoints print `SKIPPED` to stderr and
  pass; `MOTIONBRICKS_REQUIRE_FIXTURES=1` turns each skip into a failure.
- **CPU cost:** `cargo test --release --test runtime bench -- --ignored --nocapture` times the
  per-frame player path (sample, IK, record) and a generation's encode, decode and correction.
