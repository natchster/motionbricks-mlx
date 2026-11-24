// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Tests for the app runtime (`src/runtime/`), porting the meaningful upstream
//! behavioral tests from `sandbox/upstream/tests/app/` headlessly: state machine
//! transitions, locomotion regeneration, blend-space sampling, probe-box overlap, the
//! sitting behavior's Sit -> loop -> Stand sequence, and character pose-history
//! chaining.
//!
//! Generation runs through a scripted `EngineAdapter` mock fed by a real clip from
//! `checkpoints/motions`, so tests never touch the GPU. Tests that need the clip skip
//! gracefully when it is missing.
//!
//! `src/lib.rs` does not declare the runtime module yet (that wiring is the
//! integrator's job), so the documents and runtime modules are included here directly
//! by path; `crate::documents` inside the runtime sources resolves against the
//! `documents` module below.

#[path = "../src/documents/mod.rs"]
pub mod documents;
#[path = "../src/runtime/mod.rs"]
pub mod runtime;

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use glam::{Vec2, Vec3};
use motionbricks_rs::runtime as rt;
use rt::animation::MotionBuffer;
use rt::math::{self as rtm, Transform};
use rt::{AnimationBuffer, FootContacts, MotionRecorder, PoseToken, Skeleton, TargetPoseInferenceParams};

use runtime::behaviors::{
    BehaviorRegistry, CharacterProperties, SmartPrimitiveAsset, SmartPrimitiveBehavior, SmartPrimitiveData, SmartPrimitiveSequence,
    VelocityMatchingSequenceBehavior,
};
use runtime::blender::{BlendSpace, BlendSpaceType, SomaTargetPoseBlender, TargetPoseBlender, TargetPoseContext};
use runtime::character::{Character, EngineAdapter, target_pose_ik_weight};
use runtime::interaction::{InteractionBox, boxes_overlap, compute_interaction_box_overlaps};
use runtime::motion::{
    self, AnimationClip, CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE, LocomotionCommand, LocomotionMode,
    LocomotionModeRuntime, LocomotionSettings, MovementControl, PlaybackSource, ground_direction, make_animation_clip,
};
use runtime::presets;
use runtime::state_machine::ActiveState;
use runtime::states::IdleState;

const DT: f32 = 1.0 / 30.0;
const SIT_EARLY: usize = 3;
const STAND_EARLY: usize = 5;

// --- Clips and fixtures --------------------------------------------------------------------------

fn motions_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../checkpoints/motions")
}

/// The real SOMA clip that scripts the mock adapter's predictions, if present.
fn load_soma_clip() -> Option<AnimationBuffer> {
    let path = motions_dir().join("soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh");
    rt::load_bvh(path, None, true).ok()
}

fn two_joint_skeleton() -> Arc<Skeleton> {
    let reference = vec![
        Transform::new(rtm::Vec3::new(0.0, 0.0, 1.0), rtm::QUAT_IDENTITY),
        Transform::new(rtm::Vec3::new(0.0, 0.5, 0.0), rtm::QUAT_IDENTITY),
    ];
    Arc::new(Skeleton::new(vec!["Hips".to_string(), "Child".to_string()], vec![-1, 0], reference).unwrap())
}

/// A synthetic 30 Hz clip: the root walks +X at 1.5 m/s, the child holds its offset.
fn synthetic_clip() -> AnimationBuffer {
    let skeleton = two_joint_skeleton();
    let mut poses = Vec::new();
    for frame in 0..60 {
        poses.push(Transform::new(rtm::Vec3::new(frame as f32 * 0.05, 0.0, 1.0), rtm::QUAT_IDENTITY));
        poses.push(Transform::new(rtm::Vec3::new(0.0, 0.5, 0.0), rtm::QUAT_IDENTITY));
    }
    AnimationBuffer::new(skeleton, 30.0, poses, None, None).unwrap()
}

fn make_idle_clip(clip: &AnimationBuffer, model_to_runtime: Transform) -> AnimationClip<AnimationBuffer> {
    let mut buffer = clip.copy();
    buffer.make_loopable();
    make_animation_clip(buffer, model_to_runtime).unwrap()
}

/// The scripted engine adapter: generations replay a real clip, and the context for
/// each generation comes from the recorded poses, as the real players do it.
struct MockAdapter {
    clip: AnimationBuffer,
    history: MotionRecorder<Transform>,
    context: PoseToken,
    playback_time: f64,
    num_frames: usize,
    transition_early_by_frames: usize,
    generates: Vec<(Vec<Vec<Transform>>, TargetPoseInferenceParams)>,
    context_snapshots: Vec<PoseToken>,
}

impl MockAdapter {
    fn new(clip: AnimationBuffer, reference_pose: &[Transform]) -> Self {
        MockAdapter {
            clip,
            history: MotionRecorder::new(reference_pose, 8).unwrap(),
            context: PoseToken::new(reference_pose.len()),
            playback_time: 0.0,
            num_frames: 0,
            transition_early_by_frames: 0,
            generates: Vec::new(),
            context_snapshots: Vec::new(),
        }
    }
}

impl EngineAdapter for MockAdapter {
    type Buffer = AnimationBuffer;

    fn generate(&mut self, target_poses: &[Vec<Transform>], params: &TargetPoseInferenceParams) -> Result<()> {
        // Like the real player: the next generation's context comes from the recorder.
        self.context.update_from_recorder(&self.history)?;
        self.context_snapshots.push(self.context.clone());
        self.generates.push((target_poses.to_vec(), params.clone()));
        self.num_frames = 30.min(self.clip.num_frames());
        self.transition_early_by_frames = params.transition_early_by_frames;
        self.playback_time = 0.0;
        Ok(())
    }

    fn sample(&mut self, delta_time: f32, pose: &mut [Transform]) -> Result<FootContacts> {
        self.playback_time += delta_time as f64;
        self.clip.sample_into_at(self.playback_time, Transform::IDENTITY, pose)?;
        Ok([0.0; 4])
    }

    fn record_pose(&mut self, delta_time: f32, pose: &[Transform]) -> Result<()> {
        self.history.record(delta_time, pose)
    }

    fn seed_history_from_pose(&mut self, pose: &[Transform]) -> Result<()> {
        self.history.fill(pose)?;
        self.context.update_from_recorder(&self.history)
    }

    fn time_to_transition(&self) -> f64 {
        (self.playback_duration() - self.playback_time).max(0.0)
    }

    fn playback_duration(&self) -> f64 {
        self.num_frames.saturating_sub(1 + self.transition_early_by_frames) as f64 / self.clip.sample_rate() as f64
    }

    fn current_root_position(&self) -> rtm::Vec3 {
        let (f, n, b) = rt::frame_index_from_time(self.playback_time as f32, self.clip.sample_rate(), self.clip.num_frames());
        self.clip.frame(f)[0].p * (1.0 - b) + self.clip.frame(n)[0].p * b
    }
}

/// The test modes: the three core modes plus two standing styles, with a single Idle
/// target each (upstream `test_state_machine.py`'s setup). The linear mode carries
/// distinct generation settings so tests can check they reach the adapter.
fn test_modes() -> Vec<LocomotionMode> {
    let mode = |name: &str, blend_space_type: BlendSpaceType, max_speed: f32, early: usize, tokens: Vec<i32>| {
        LocomotionMode::new(name, PathBuf::from("unused.smart_primitive.json"), blend_space_type, vec![("Idle".to_string(), Vec2::ZERO)], max_speed, 2.3_f32.min(max_speed), 0.0, early, tokens, 1.0, 1.0, "").unwrap()
    };
    vec![
        mode(LINEAR_LOCOMOTION_MODE, BlendSpaceType::Linear1D, 4.5, 5, vec![6, 7, 8]),
        mode(DIRECTIONAL_LOCOMOTION_MODE, BlendSpaceType::Directional2D, 4.5, 8, (8..=14).collect()),
        mode(CROUCH_LOCOMOTION_MODE, BlendSpaceType::Directional2D, 3.6, 8, (8..=14).collect()),
        mode("zombie_1d", BlendSpaceType::Linear1D, 1.0, 8, (8..=14).collect()),
        mode("skipping_1d", BlendSpaceType::Linear1D, 1.5, 8, (8..=14).collect()),
    ]
}

fn runtime_for(mode: &LocomotionMode, clip: &AnimationBuffer) -> LocomotionModeRuntime<Transform> {
    let mut blend_space = BlendSpace::new(mode.blend_space_type);
    let mut target_poses = Vec::new();
    for (index, (_, value)) in mode.target_values.iter().enumerate() {
        for frame in 0..4 {
            target_poses.push(clip.sample_at(index as f64 * 0.2 + frame as f64 / 30.0, Transform::IDENTITY).unwrap());
        }
        blend_space.add_target(*value, index).unwrap();
    }
    blend_space.compile().unwrap();
    LocomotionModeRuntime { definition: mode.clone(), blend_space, target_poses }
}

fn make_character(clip: &AnimationBuffer, initial_mode: &str, modes: &[LocomotionMode]) -> Character<MockAdapter> {
    let skeleton = clip.skeleton().clone();
    let reference = skeleton.reference_local_transforms().to_vec();
    let model_to_runtime = rt::model_to_runtime(rt::ModelRig::Soma);
    let runtimes: Vec<_> = modes.iter().map(|mode| runtime_for(mode, clip)).collect();
    let standing_idle = make_idle_clip(clip, model_to_runtime);
    let crouching_idle = make_idle_clip(clip, model_to_runtime);
    let player = MockAdapter::new(clip.clone(), &reference);
    let clip_for_loader = clip.clone();
    let clip_loader = Box::new(move |_: &Path, loop_: bool| -> Result<AnimationClip<AnimationBuffer>> {
        let mut buffer = clip_for_loader.copy();
        if loop_ {
            buffer.make_loopable();
        }
        make_animation_clip(buffer, model_to_runtime)
    });
    Character::new(
        player,
        Box::new(SomaTargetPoseBlender::new(reference.clone())),
        None,
        Vec3::Y,
        model_to_runtime,
        Transform::IDENTITY,
        runtimes,
        initial_mode,
        vec![standing_idle],
        vec![crouching_idle],
        presets::soma::interaction_probe(),
        &reference,
        clip_loader,
    )
    .unwrap()
}

fn idle_command() -> LocomotionCommand {
    LocomotionCommand { facing_direction: Vec3::X, movement: None, use_max_speed: false, requested_mode: None }
}

fn move_command(direction: Vec3) -> LocomotionCommand {
    LocomotionCommand { facing_direction: Vec3::X, movement: Some(direction), use_max_speed: false, requested_mode: None }
}

/// A Smart Primitive asset with Sit and Stand sequences (one target each), built
/// directly, mirroring upstream's synthetic assets.
fn sitting_asset(clip: &AnimationBuffer) -> Rc<SmartPrimitiveAsset<Transform>> {
    let poses = |start: f64| (0..4).map(|frame| clip.sample_at(start + frame as f64 / 30.0, Transform::IDENTITY).unwrap()).collect::<Vec<_>>();
    let params = |early: usize| TargetPoseInferenceParams { transition_early_by_frames: early, num_poses_to_use: 4, ik_pin_mask: rt::ik_pin::NONE, allowed_token_counts: Some((6..=14).collect()) };
    Rc::new(SmartPrimitiveAsset::new(SmartPrimitiveData {
        rig: rt::ModelRig::Soma,
        node_local_transforms: vec![Transform::IDENTITY; 3],
        node_parent_indices: vec![-1, 0, 0],
        node_names: vec!["Root".to_string(), "SitTarget".to_string(), "StandTarget".to_string()],
        target_node_indices: vec![1, 2],
        target_poses: [poses(0.0), poses(0.5)].concat(),
        target_inference_params: vec![params(SIT_EARLY), params(STAND_EARLY)],
        box_node_indices: vec![],
        box_half_extents: vec![],
        pivot_node_indices: vec![],
        pivots: vec![],
        sequences: vec![
            SmartPrimitiveSequence { name: "Sit".to_string(), node_indices: vec![1], start_speed: 0.0, pivot_node_index: -1 },
            SmartPrimitiveSequence { name: "Stand".to_string(), node_indices: vec![2], start_speed: 0.0, pivot_node_index: -1 },
        ],
    }))
}

fn assert_vec3_close(actual: Vec3, expected: [f32; 3], atol: f32) {
    assert!(
        (actual.x - expected[0]).abs() <= atol && (actual.y - expected[1]).abs() <= atol && (actual.z - expected[2]).abs() <= atol,
        "expected {expected:?}, got {actual:?}"
    );
}

fn assert_pose_close(actual: &[Transform], expected: &[Transform], atol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        for k in 0..3 {
            assert!((a.p.0[k] - e.p.0[k]).abs() <= atol, "root position differs: {:?} vs {:?}", a.p, e.p);
        }
        let dot: f32 = (0..4).map(|k| a.q[k] * e.q[k]).sum();
        assert!(dot.abs() >= 1.0 - 1.0e-3, "root rotation differs: {:?} vs {:?}", a.q, e.q);
    }
}

// --- Blend spaces (upstream test_locomotion_character_states.py) ---------------------------------

#[test]
fn blend_space_1d_sampling_weights() {
    // Upstream's loader test: "Fast" at 1 with data 0, "Slow" at 0 with data 1.
    let mut space = BlendSpace::new(BlendSpaceType::Linear1D);
    space.add_target(Vec2::new(1.0, 0.0), 0).unwrap();
    space.add_target(Vec2::new(0.0, 0.0), 1).unwrap();
    space.compile().unwrap();
    assert_eq!(space.data_indices(), &[0, 1]);

    let sample = space.evaluate(Vec2::new(0.25, 0.0)).unwrap();
    assert_eq!(sample.indices[..2], [1, 0]);
    assert!((sample.weights[0] - 0.75).abs() < 1.0e-6);
    assert!((sample.weights[1] - 0.25).abs() < 1.0e-6);

    // Values clamp to the target range.
    let high = space.evaluate(Vec2::new(2.0, 0.0)).unwrap();
    assert!((high.weights[1] - 1.0).abs() < 1.0e-6);
    let low = space.evaluate(Vec2::new(-1.0, 0.0)).unwrap();
    assert!((low.weights[0] - 1.0).abs() < 1.0e-6);

    // A single target always gets the full weight.
    let mut single = BlendSpace::new(BlendSpaceType::Linear1D);
    single.add_target(Vec2::new(0.5, 0.0), 3).unwrap();
    single.compile().unwrap();
    let sample = single.evaluate(Vec2::new(0.1, 0.0)).unwrap();
    assert_eq!(sample.weights, [1.0, 0.0, 0.0]);
}

#[test]
fn blend_space_directional_center_and_sectors() {
    // Idle at the center, Forward/Backward/Left/Right around it.
    let mut space = BlendSpace::new(BlendSpaceType::Directional2D);
    for (index, value) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)].into_iter().enumerate() {
        space.add_target(Vec2::new(value.0, value.1), index).unwrap();
    }
    space.compile().unwrap();

    // Near zero, the center target gets the full weight.
    let center = space.evaluate(Vec2::ZERO).unwrap();
    assert_eq!(center.indices[0], 0);
    assert!((center.weights[0] - 1.0).abs() < 1.0e-6);

    // On the Forward axis, Forward gets the full weight.
    let forward = space.evaluate(Vec2::X).unwrap();
    let contributions = forward.contributions();
    let forward_weight: f32 = contributions.iter().filter(|&&(data, _)| data == 1).map(|&(_, weight)| weight).sum();
    assert!((forward_weight - 1.0).abs() < 1.0e-6, "forward target should dominate: {contributions:?}");

    // Halfway between Forward and Left, the two share the weight.
    let diagonal = space.evaluate(Vec2::new(1.0, 1.0).normalize()).unwrap();
    let weight_of = |data: usize| diagonal.contributions().iter().filter(|&&(d, _)| d == data).map(|&(_, w)| w).sum::<f32>();
    assert!((weight_of(1) - 0.5).abs() < 1.0e-6, "Forward weight: {}", weight_of(1));
    assert!((weight_of(3) - 0.5).abs() < 1.0e-6, "Left weight: {}", weight_of(3));
}

// --- Probe box and overlap (upstream test_runtime_smart_primitive.py) -----------------------------

fn interaction_box(position: Vec3, half_extents: Vec3, yaw: f32) -> InteractionBox {
    InteractionBox {
        transform: Transform::new(motion::to_rt(position), rtm::quat_from_axis_angle(rtm::Vec3::new(0.0, 0.0, 1.0), yaw)),
        half_extents,
    }
}

#[test]
fn interaction_probe_segment_scales_with_speed() {
    let probe = presets::soma::interaction_probe();
    // At rest the box reaches near_distance + box_min_depth.
    let (near, far) = probe.segment(Vec3::ZERO, Vec3::X, 0.0);
    assert_vec3_close(near, [0.15, 0.0, 0.0], 1.0e-6);
    assert_vec3_close(far, [0.9, 0.0, 0.0], 1.0e-6);
    // At 2 m/s the box reaches the lookahead distance.
    let (_, far) = probe.segment(Vec3::ZERO, Vec3::X, 2.0);
    assert_vec3_close(far, [2.0, 0.0, 0.0], 1.0e-6);
    // Above extra_lookahead_min_speed (2.3) the box reaches further.
    let (_, far) = probe.segment(Vec3::ZERO, Vec3::X, 3.0);
    assert_vec3_close(far, [3.0 + 0.7 * 0.25, 0.0, 0.0], 1.0e-6);

    // The box covers the segment, standing on the ground; local Y is its depth.
    let probe_box = probe.box_for_segment((near, Vec3::new(2.0, 0.0, 0.0)), 1.0);
    assert_vec3_close(motion::to_glam(probe_box.transform.p), [1.075, 0.0, 0.8], 1.0e-6);
    assert_vec3_close(probe_box.half_extents, [0.5, 0.925, 0.8], 1.0e-6);
    // Crouching scales the height only.
    let crouched = probe.box_for_segment((near, Vec3::new(2.0, 0.0, 0.0)), 0.6);
    assert_vec3_close(motion::to_glam(crouched.transform.p), [1.075, 0.0, 0.48], 1.0e-6);
    assert_vec3_close(crouched.half_extents, [0.5, 0.925, 0.48], 1.0e-6);
}

#[test]
fn interaction_box_overlap_matches_upstream_cases() {
    let near = interaction_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    let far = interaction_box(Vec3::new(3.0, 0.0, 0.0), Vec3::splat(0.5), 0.0);
    let boxes = [near, far];
    let overlaps = |probe: InteractionBox| compute_interaction_box_overlaps(&boxes, &probe);
    // Touching boxes count as overlapping.
    assert_eq!(overlaps(interaction_box(Vec3::ZERO, Vec3::splat(0.5), 0.0)), [true, false]);
    assert_eq!(overlaps(interaction_box(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(0.5), 0.0)), [true, false]);
    assert_eq!(overlaps(interaction_box(Vec3::new(2.0, 0.0, 0.0), Vec3::splat(0.5), 0.0)), [false, true]);
    assert_eq!(overlaps(interaction_box(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5), 0.0)), [false, false]);
    // A 45-degree turn still reaches the near box.
    assert_eq!(overlaps(interaction_box(Vec3::ZERO, Vec3::new(0.75, 0.25, 0.5), std::f32::consts::FRAC_PI_4)), [true, false]);

    // Upstream's exact parallel-box case: the probe walks under an overhead box.
    let overhead = InteractionBox {
        transform: Transform::new(rtm::Vec3::new(3.1405947, 12.562379, 1.5375934), [0.0, 0.0, -0.70710677, 0.70710677]),
        half_extents: Vec3::new(0.19628717, 0.19628717, 0.39257434),
    };
    let probe_at = |half_height: f32| InteractionBox {
        transform: Transform::new(rtm::Vec3::new(3.0434444, 12.853794, half_height), [0.0, 0.0, -0.7071068, 0.70710677]),
        half_extents: Vec3::new(0.395, 0.5872061, half_height),
    };
    assert!(boxes_overlap(&overhead, &probe_at(0.63)));
    assert!(!boxes_overlap(&overhead, &probe_at(0.55)));
}

// --- Movement springs (upstream test_locomotion_character_states.py) ------------------------------

#[test]
fn ground_direction_returns_unit_xy_or_fallback() {
    let direction = ground_direction(Vec3::new(3.0, 4.0, 8.0), Vec3::X).unwrap();
    assert_vec3_close(direction, [0.6, 0.8, 0.0], 1.0e-6);
    let fallback = ground_direction(Vec3::ZERO, Vec3::new(0.0, -3.0, 4.0)).unwrap();
    assert_vec3_close(fallback, [0.0, -1.0, 0.0], 1.0e-6);
    assert!(ground_direction(Vec3::ZERO, Vec3::ZERO).is_err());
}

fn test_mode(name: &str, blend_space_type: BlendSpaceType, max_speed: f32) -> LocomotionMode {
    LocomotionMode::new(name, PathBuf::from("unused.smart_primitive.json"), blend_space_type, vec![("Idle".to_string(), Vec2::ZERO)], max_speed, 2.3, 0.0, 8, (8..=14).collect(), 1.0, 1.0, "").unwrap()
}

#[test]
fn locomotion_mode_rejects_invalid_settings() {
    let build = |default_speed: f32, minimum_speed: f32, tokens: Vec<i32>| {
        LocomotionMode::new("linear", PathBuf::from("x"), BlendSpaceType::Linear1D, vec![("Idle".to_string(), Vec2::ZERO)], 4.5, default_speed, minimum_speed, 8, tokens, 1.0, 1.0, "")
    };
    assert!(build(2.3, 0.0, vec![]).is_err(), "empty token counts");
    assert!(build(2.3, 0.0, vec![5, 6]).is_err(), "tokens below 6");
    assert!(build(2.3, 0.0, vec![16, 17]).is_err(), "tokens above 16");
    assert!(build(4.6, 0.0, vec![8]).is_err(), "default above max");
    assert!(build(2.3, -0.1, vec![8]).is_err(), "negative minimum");
    assert!(build(2.3, 2.4, vec![8]).is_err(), "minimum above default");
}

#[test]
fn mode_switch_reduces_speed_only_when_not_faster() {
    let linear = test_mode("linear", BlendSpaceType::Linear1D, 4.5);
    let directional = test_mode("directional", BlendSpaceType::Directional2D, 4.5);
    let crouch = test_mode("crouch", BlendSpaceType::Directional2D, 3.6);
    for (previous, current, speed, expected) in [(&linear, &directional, 4.5, 3.375), (&directional, &crouch, 4.5, 3.375), (&crouch, &directional, 3.6, 3.6)] {
        let mut movement = MovementControl::with_settings(Vec3::Y, LocomotionSettings::default());
        movement.desired_facing = Vec3::X;
        movement.desired_speed = speed;
        movement.ease_mode_switch(current, previous);
        assert!((movement.desired_speed - expected).abs() < 1.0e-4, "{speed} -> {expected}, got {}", movement.desired_speed);
        assert!(movement.desired_speed <= current.max_speed);
        assert_vec3_close(movement.current_facing, [movement.desired_facing.x, movement.desired_facing.y, 0.0], 1.0e-6);
    }
}

#[test]
fn analog_movement_scales_max_speed_with_a_floor() {
    for (movement_input, minimum_speed, expected_speed) in [(Some(Vec3::new(0.5, 0.0, 0.0)), 0.0, 2.25), (Some(Vec3::new(0.05, 0.0, 0.0)), 1.84, 1.84), (None, 1.84, 0.0)] {
        let mode = LocomotionMode::new("linear", PathBuf::from("x"), BlendSpaceType::Linear1D, vec![("Idle".to_string(), Vec2::ZERO)], 4.5, 2.3, minimum_speed, 8, (8..=14).collect(), 1.0, 1.0, "").unwrap();
        let mut control = MovementControl::new(Vec3::X);
        control.input_speed = 1.0;
        control.update(&LocomotionCommand { facing_direction: Vec3::X, movement: movement_input, use_max_speed: true, requested_mode: None }, &mode, DT, None);
        assert!((control.input_speed - expected_speed).abs() < 1.0e-6, "expected {expected_speed}, got {}", control.input_speed);
        assert_vec3_close(control.input_velocity_direction, [1.0, 0.0, 0.0], 1.0e-6);
    }
}

#[test]
fn realign_uses_measured_velocity_and_preserves_input() {
    let mode = test_mode("linear", BlendSpaceType::Linear1D, 4.5);
    let mut movement = MovementControl::new(Vec3::X);
    movement.input_speed = 4.5;
    movement.desired_speed = 4.5;
    movement.desired_speed_rate = 1.0;
    movement.desired_velocity_direction_rate = 1.0;
    movement.current_facing_rate = 1.0;
    movement.desired_velocity = Vec3::new(4.5, 0.0, 0.0);

    movement.realign(Vec3::new(0.0, 3.0, 0.0), &mode);

    assert!((movement.desired_speed - 3.0).abs() < 1.0e-6);
    assert_vec3_close(movement.desired_velocity, [0.0, 3.0, 0.0], 1.0e-6);
    assert_vec3_close(movement.current_facing, [0.0, 1.0, 0.0], 1.0e-6);
    assert!((movement.input_speed - 4.5).abs() < 1.0e-6);
    assert_vec3_close(movement.input_velocity_direction, [1.0, 0.0, 0.0], 1.0e-6);
}

#[test]
fn target_pose_ik_weight_uses_a_late_smooth_ramp() {
    assert_eq!(target_pose_ik_weight(1.0, 0.5), 0.0);
    assert!((target_pose_ik_weight(1.0, 0.25) - 0.5).abs() < 1.0e-6);
    assert_eq!(target_pose_ik_weight(1.0, 0.0), 1.0);
    assert_eq!(target_pose_ik_weight(0.0, 0.0), 1.0);
}

// --- Target evaluation (upstream LocomotionTargetEvaluationTests) ---------------------------------

/// Upstream's `_targets()`: two blend targets whose roots walk at different speeds and
/// heights, with distinct child offsets.
fn blend_targets() -> (Vec<Vec<Transform>>, Vec<Transform>) {
    let mut poses = Vec::new();
    for target in 0..2 {
        for frame in 0..4 {
            let root = Transform::new(
                rtm::Vec3::new(frame as f32 * if target == 0 { 0.1 } else { 0.3 }, 0.0, if target == 0 { 1.0 } else { 3.0 }),
                rtm::QUAT_IDENTITY,
            );
            let child = Transform::new(rtm::Vec3::new(if target == 0 { 0.5 } else { 1.5 }, 0.0, 0.0), rtm::QUAT_IDENTITY);
            poses.push(vec![root, child]);
        }
    }
    let reference = poses[0].clone();
    (poses, reference)
}

fn evaluate_mode(
    blend_space_type: BlendSpaceType,
    keys: &[(f32, f32)],
    speed: f32,
    facing: Vec3,
) -> (Vec<Vec<Transform>>, Vec<Vec<Transform>>, Vec<Vec<Transform>>) {
    let (poses, reference) = blend_targets();
    let mut blend_space = BlendSpace::new(blend_space_type);
    for (index, &(x, y)) in keys.iter().enumerate() {
        blend_space.add_target(Vec2::new(x, y), index).unwrap();
    }
    blend_space.compile().unwrap();
    let mode = LocomotionMode::new("test", PathBuf::from("x"), blend_space_type, vec![("Idle".to_string(), Vec2::ZERO)], 4.5, 2.3, 0.0, 8, (8..=14).collect(), 0.9, 0.8, "").unwrap();
    let runtime = LocomotionModeRuntime { definition: mode, blend_space, target_poses: poses.clone() };
    let blender = SomaTargetPoseBlender::new(reference);
    let context = TargetPoseContext { ground_position: Vec3::new(10.0, 2.0, 0.0), current_facing: facing, desired_velocity: facing * speed };
    let mut blended = vec![vec![Transform::IDENTITY; 2]; 4];
    let mut target = Vec::new();
    blender.evaluate(&context, &runtime, &mut blended, &mut target).unwrap();
    // The stored target poses after the evaluation, to check they are untouched.
    (runtime.target_poses, blended, target)
}

#[test]
fn modes_blend_then_scale_speed_and_place_only_the_root() {
    for (blend_space_type, keys, speed, facing, height, child_x) in [
        (BlendSpaceType::Linear1D, vec![(0.0, 0.0), (1.0, 0.0)], 2.25, Vec3::Y, 2.0, 1.0),
        (BlendSpaceType::Directional2D, vec![(0.0, 0.0), (1.0, 0.0), (0.0, -1.0)], 4.5, Vec3::X, 3.0, 1.5),
    ] {
        let (poses, _blended, target) = evaluate_mode(blend_space_type, &keys, speed, facing);
        let ground = Vec3::new(10.0, 2.0, 0.0) + facing * (speed * 0.9 * 0.8);
        for frame in 0..4 {
            let expected_root = ground + Vec3::new(0.0, 0.0, height) + facing * (speed * 0.9 * frame as f32 / 30.0);
            let actual_root = motion::to_glam(target[frame][0].p);
            assert_vec3_close(actual_root, [expected_root.x, expected_root.y, expected_root.z], 1.0e-5);
            assert!((target[frame][1].p.x() - child_x).abs() < 1.0e-6, "child x: {}", target[frame][1].p.x());
        }
        // The placement turns the roots to face the movement direction.
        let root_forward = motion::to_glam(target[0][0].vector(rtm::Vec3::new(1.0, 0.0, 0.0)));
        assert_vec3_close(root_forward, [facing.x, facing.y, 0.0], 1.0e-5);
        // The mode's stored target poses are untouched.
        let (expected_poses, _) = blend_targets();
        for (actual, expected) in poses.iter().zip(expected_poses.iter()) {
            assert_pose_close(actual, expected, 0.0);
        }
    }
}

// --- Idle state (upstream CharacterStateTests) -----------------------------------------------------

#[test]
fn idle_selection_excludes_the_previous_clip() {
    let clip = synthetic_clip();
    let model_to_runtime = rt::model_to_runtime(rt::ModelRig::Soma);
    let clip_a = make_idle_clip(&clip, model_to_runtime);
    let clip_b = make_idle_clip(&clip, model_to_runtime);
    let mut state = IdleState::new("Idle", vec![clip_a.clone(), clip_b.clone()]).unwrap();
    assert_eq!(state.choose_clip(Some(&clip_a)).id, clip_b.id);
    assert_eq!(state.choose_clip(Some(&clip_b)).id, clip_a.id);
    assert!(IdleState::<AnimationBuffer>::new("Idle", vec![]).is_err());
}

// --- Velocity matching (upstream test_runtime_smart_primitive.py) ----------------------------------

#[test]
fn velocity_matching_selects_the_closest_sequence_speed() {
    let clip = synthetic_clip();
    let poses = |start: f64| (0..4).map(|frame| clip.sample_at(start + frame as f64 / 30.0, Transform::IDENTITY).unwrap()).collect::<Vec<_>>();
    let params = |early: usize| TargetPoseInferenceParams { transition_early_by_frames: early, num_poses_to_use: 4, ik_pin_mask: rt::ik_pin::NONE, allowed_token_counts: Some((6..=14).collect()) };
    let asset = Rc::new(SmartPrimitiveAsset::new(SmartPrimitiveData {
        rig: rt::ModelRig::Soma,
        node_local_transforms: vec![Transform::IDENTITY; 3],
        node_parent_indices: vec![-1, 0, 0],
        node_names: vec!["Root".to_string(), "Slow".to_string(), "Fast".to_string()],
        target_node_indices: vec![1, 2],
        target_poses: [poses(0.0), poses(0.5)].concat(),
        target_inference_params: vec![params(0), params(5)],
        box_node_indices: vec![],
        box_half_extents: vec![],
        pivot_node_indices: vec![],
        pivots: vec![],
        sequences: vec![
            SmartPrimitiveSequence { name: "Stop".to_string(), node_indices: vec![1], start_speed: 0.0, pivot_node_index: -1 },
            SmartPrimitiveSequence { name: "Run".to_string(), node_indices: vec![2], start_speed: 2.0, pivot_node_index: -1 },
        ],
    }));
    let mut behavior = VelocityMatchingSequenceBehavior::new(asset).unwrap();
    let properties = |speed: f32| CharacterProperties {
        position: Vec3::ZERO,
        velocity: Vec3::new(speed, 0.0, 0.0),
        facing: Vec3::X,
        desired_velocity: Vec3::ZERO,
        probe_segment: (Vec3::ZERO, Vec3::ZERO),
    };
    // Speeds above every sequence pick the fastest one. A tie keeps the first.
    for (speed, expected_early) in [(0.0, 0), (2.0, 5), (4.0, 5), (1.0, 0), (0.4, 0)] {
        let (_poses, params) = behavior.get_targets(&properties(speed)).unwrap();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].transition_early_by_frames, expected_early, "speed {speed}");
    }
    // A behavior requires at least one sequence.
    let empty = Rc::new(SmartPrimitiveAsset::new(SmartPrimitiveData::<Transform> {
        rig: rt::ModelRig::Soma,
        node_local_transforms: vec![Transform::IDENTITY],
        node_parent_indices: vec![-1],
        node_names: vec!["Root".to_string()],
        target_node_indices: vec![],
        target_poses: vec![],
        target_inference_params: vec![],
        box_node_indices: vec![],
        box_half_extents: vec![],
        pivot_node_indices: vec![],
        pivots: vec![],
        sequences: vec![],
    }));
    assert!(VelocityMatchingSequenceBehavior::new(empty).is_err());
    // The registry knows the upstream class names and rejects unknown ones.
    let registry = BehaviorRegistry::<Transform>::new("sitting.bvh");
    assert!(registry.create("unknown", Rc::new(SmartPrimitiveAsset::new(SmartPrimitiveData::<Transform> {
        rig: rt::ModelRig::Soma,
        node_local_transforms: vec![Transform::IDENTITY],
        node_parent_indices: vec![-1],
        node_names: vec!["Root".to_string()],
        target_node_indices: vec![],
        target_poses: vec![],
        target_inference_params: vec![],
        box_node_indices: vec![],
        box_half_extents: vec![],
        pivot_node_indices: vec![],
        pivots: vec![],
        sequences: vec![],
    })))
    .is_err());
}

// --- SOMA pose facing (upstream PoseFacingTests) ----------------------------------------------------

#[test]
fn soma_faces_along_the_hips_y_axis_on_the_ground() {
    let clip = synthetic_clip();
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);
    // The hips turn 30 degrees and tilt forward, as when bending to stand up.
    let rotation = rtm::quat_mul(
        rtm::quat_from_axis_angle(rtm::Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_6),
        rtm::quat_from_axis_angle(rtm::Vec3::new(1.0, 0.0, 0.0), -0.4),
    );
    character.current_pose[0] = Transform::new(rtm::Vec3::new(1.0, 2.0, 0.9), rotation);
    let facing = character.pose_facing();
    assert_vec3_close(facing, [-0.5, 0.75_f32.sqrt(), 0.0], 1.0e-5);

    // face_pose() sets every facing to the pose's heading, with the turn rate reset.
    character.movement.current_facing_rate = 0.4;
    character.face_pose();
    assert_vec3_close(character.movement.current_facing, [-0.5, 0.75_f32.sqrt(), 0.0], 1.0e-5);
    assert_vec3_close(character.movement.desired_facing, [-0.5, 0.75_f32.sqrt(), 0.0], 1.0e-5);
    assert_vec3_close(character.movement.last_input_facing, [-0.5, 0.75_f32.sqrt(), 0.0], 1.0e-5);
    assert_eq!(character.movement.current_facing_rate, 0.0);
}

// --- Character-level tests with a real clip and the scripted adapter -------------------------------

#[test]
fn character_state_transitions_follow_the_three_source_model() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);

    // The character starts in the standing idle's clip loop.
    assert_eq!(character.state_machine.state(), ActiveState::StandingIdle);
    assert_eq!(character.playback_source(), PlaybackSource::Clip);

    // Movement input transitions idle -> locomotion and generates motion.
    character.step(DT, &move_command(Vec3::X)).unwrap();
    assert_eq!(character.state_machine.state(), ActiveState::StandingLocomotion);
    assert_eq!(character.playback_source(), PlaybackSource::Generated);
    assert_eq!(character.player().generates.len(), 1);
    // The generation uses the active mode's transition point and token counts.
    let (_, params) = &character.player().generates[0];
    assert_eq!(params.transition_early_by_frames, 5);
    assert_eq!(params.num_poses_to_use, 4);
    assert_eq!(params.allowed_token_counts, Some(vec![6, 7, 8]));

    // Walking on, the segment regenerates when it reaches its transition point.
    let mut regenerated = false;
    for _ in 0..60 {
        character.step(DT, &move_command(Vec3::X)).unwrap();
        if character.player().generates.len() > 1 {
            regenerated = true;
            break;
        }
    }
    assert!(regenerated, "locomotion should regenerate at the transition point");

    // Releasing the input transitions back to idle through a clip transition.
    character.step(DT, &idle_command()).unwrap();
    assert_eq!(character.state_machine.state(), ActiveState::StandingIdle);
    assert_eq!(character.playback_source(), PlaybackSource::ClipTransition);
    let (_, params) = character.player().generates.last().unwrap();
    // Clip transitions play out to the end (transition_early_by_frames = 0).
    assert_eq!(params.transition_early_by_frames, 0);
    assert_eq!(params.num_poses_to_use, 4);

    // The transition leads into the looping idle clip.
    for _ in 0..60 {
        character.step(DT, &idle_command()).unwrap();
        if character.playback_source() == PlaybackSource::Clip {
            break;
        }
    }
    assert_eq!(character.playback_source(), PlaybackSource::Clip);
    assert!(character.state_machine.state() == ActiveState::StandingIdle);
}

#[test]
fn locomotion_regenerates_on_a_large_direction_change() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);

    // Walk forward until a fresh generation starts, so the transition timer alone
    // cannot explain the next regeneration.
    for _ in 0..90 {
        let before = character.player().generates.len();
        character.step(DT, &move_command(Vec3::X)).unwrap();
        if character.player().generates.len() > before && character.player().generates.len() >= 2 {
            break;
        }
    }
    assert!(character.player().generates.len() >= 2, "walking should regenerate within 90 steps");
    assert!(character.state_machine.is_locomoting());
    // Walking straight, no regeneration is needed right after a generation.
    assert!(!character.state_machine.active_locomotion_state().needs_regeneration(&character.movement));

    // A 180-degree direction error asks for regeneration (the dot product test).
    let previous_speed = character.movement.desired_speed;
    character.movement.desired_velocity_direction = Vec3::NEG_X;
    character.movement.desired_velocity = Vec3::NEG_X * previous_speed;
    assert!(character.state_machine.active_locomotion_state().needs_regeneration(&character.movement));
    // A small direction error with the same speed does not.
    character.movement.desired_velocity_direction = Vec3::new(0.99, 0.05, 0.0).normalize();
    character.movement.desired_velocity = character.movement.desired_velocity_direction * previous_speed;
    assert!(!character.state_machine.active_locomotion_state().needs_regeneration(&character.movement));

    // End to end: turning around regenerates well before the next transition point
    // (21 frames away at 4 poses x 5 early frames).
    character.movement.face(Vec3::X);
    character.movement.reset_desired_velocity();
    character.movement.realign(Vec3::X * 2.3, &modes[0]);
    // Restore the straight-walk state, then flip the input.
    let before = character.player().generates.len();
    let mut frames_to_regenerate = None;
    for frame in 1..=20 {
        character.step(DT, &move_command(Vec3::NEG_X)).unwrap();
        if character.player().generates.len() > before {
            frames_to_regenerate = Some(frame);
            break;
        }
    }
    assert!(frames_to_regenerate.is_some(), "a 180-degree turn must regenerate");
    assert!(frames_to_regenerate.unwrap() < 21, "regeneration came {} frames after the turn, not direction-driven", frames_to_regenerate.unwrap());
}

#[test]
fn sitting_behavior_sits_loops_and_stands() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);

    let registry = BehaviorRegistry::<Transform>::new("sitting.bvh");
    let behavior = registry.create("SittingBehavior", sitting_asset(&clip)).unwrap();
    character.start_interaction(behavior).unwrap();

    // The Sit sequence generates its single target with the target's settings.
    assert!(character.state_machine.is_interacting());
    assert_eq!(character.player().generates.len(), 1);
    let (_, params) = &character.player().generates[0];
    assert_eq!(params.transition_early_by_frames, SIT_EARLY);

    // When the Sit target is reached, the sitting clip starts looping.
    let mut looping = false;
    for _ in 0..40 {
        character.step(DT, &idle_command()).unwrap();
        if character.state_machine.interaction_state().is_looping() {
            looping = true;
            break;
        }
    }
    assert!(looping, "the sitting clip should loop after the Sit sequence");
    assert_eq!(character.playback_source(), PlaybackSource::Clip);
    assert_eq!(character.player().generates.len(), 1, "a looping clip generates nothing");

    // Interacting again plays the Stand sequence.
    character.interact().unwrap();
    assert!(!character.state_machine.interaction_state().is_looping());
    assert_eq!(character.player().generates.len(), 2);
    let (_, params) = &character.player().generates[1];
    assert_eq!(params.transition_early_by_frames, STAND_EARLY);

    // When the Stand target is reached, the interaction finishes into idle.
    for _ in 0..40 {
        character.step(DT, &idle_command()).unwrap();
        if !character.state_machine.is_interacting() {
            break;
        }
    }
    assert!(!character.state_machine.is_interacting());
    assert_eq!(character.state_machine.state(), ActiveState::StandingIdle);
    assert!(character.state_machine.interaction_state().primitive().is_none());
    // The default completion resumes movement, which idles through a clip transition.
    assert_eq!(character.playback_source(), PlaybackSource::ClipTransition);
}

#[test]
fn character_chaining_uses_the_last_recorded_poses_as_context() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);

    // Walk forward, recording every shown pose and the step of each generation.
    let mut shown: Vec<Vec<Transform>> = Vec::new();
    let mut generate_steps: Vec<usize> = Vec::new();
    for step in 0..60 {
        let before = character.player().generates.len();
        character.step(DT, &move_command(Vec3::X)).unwrap();
        shown.push(character.current_pose.clone());
        if character.player().generates.len() > before {
            generate_steps.push(step);
        }
    }
    // The speed ramp triggers extra generations in the first steps (upstream's
    // "locomotion started" rule); use a later one, with a full history behind it.
    let Some(&k) = generate_steps.iter().rev().find(|&&step| step >= 4) else {
        panic!("walking should regenerate past step 4, got {generate_steps:?}");
    };

    // That generation's context is the last 4 recorded poses at its step: the poses
    // the character showed in the 4 steps before it.
    let context = &character.player().context_snapshots[generate_steps.iter().position(|&step| step == k).unwrap()];
    for j in 0..4 {
        assert_pose_close(&context.poses()[j], &shown[k - 4 + j], 1.0e-4);
    }
}

#[test]
fn mode_request_during_interaction_selects_the_resume_pair() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, LINEAR_LOCOMOTION_MODE, &modes);

    let behavior = VelocityMatchingSequenceBehavior::new(sitting_asset(&clip)).unwrap();
    character.start_interaction(Box::new(behavior)).unwrap();
    assert!(character.state_machine.is_interacting());

    // Requesting Crouching does not interrupt the interaction, but selects the pair
    // the character resumes to (and shrinks the probe box).
    let command = LocomotionCommand { requested_mode: Some(CROUCH_LOCOMOTION_MODE.to_string()), ..idle_command() };
    character.step(DT, &command).unwrap();
    assert!(character.state_machine.is_interacting());
    assert_eq!(character.state_machine.probe_box_height_scale(), 0.6);

    // When the behavior finishes, the character resumes in the crouching pair.
    for _ in 0..40 {
        character.step(DT, &idle_command()).unwrap();
        if !character.state_machine.is_interacting() {
            break;
        }
    }
    assert_eq!(character.state_machine.state(), ActiveState::CrouchingIdle);
}

#[test]
fn extra_modes_are_standing_styles_sharing_the_standing_idle() {
    let Some(clip) = load_soma_clip() else {
        eprintln!("skipping: checkpoints/motions SOMA clip is missing");
        return;
    };
    let modes = test_modes();
    let mut character = make_character(&clip, CROUCH_LOCOMOTION_MODE, &modes);

    assert_eq!(character.state_machine.state(), ActiveState::CrouchingIdle);
    assert_eq!(character.state_machine.style_modes(), &["zombie_1d", "skipping_1d"]);

    // Picking a style while crouched stands the character up into the standing idle.
    let command = LocomotionCommand { requested_mode: Some("zombie_1d".to_string()), ..idle_command() };
    character.step(DT, &command).unwrap();
    assert_eq!(character.state_machine.active_locomotion_state().active_mode().name, "zombie_1d");
    assert_eq!(character.state_machine.state(), ActiveState::StandingIdle);
}

// --- Presets match upstream's per-rig constants ----------------------------------------------------

#[test]
fn presets_match_upstream_constants() {
    let soma = presets::soma::default_locomotion_modes();
    let g1 = presets::unitree_g1::default_locomotion_modes();
    assert_eq!(soma.len(), 9);
    assert_eq!(g1.len(), 9);
    // Both rigs offer the same styles, each blending Idle into the style target.
    let labels = ["Skipping", "Vogue", "Injured Torso", "Injured Leg", "Zombie", "Combat"];
    for modes in [&soma, &g1] {
        let styles = &modes[3..];
        assert_eq!(styles.iter().map(|m| m.label.as_str()).collect::<Vec<_>>(), labels);
        for mode in styles {
            assert_eq!(mode.target_values, vec![("Idle".to_string(), Vec2::ZERO), (mode.label.clone(), Vec2::X)]);
            assert_eq!(mode.default_speed, mode.max_speed);
            assert!(mode.minimum_speed < mode.max_speed);
        }
    }
    assert_eq!(soma[3..].iter().map(|m| &m.name).collect::<Vec<_>>(), g1[3..].iter().map(|m| &m.name).collect::<Vec<_>>());
    // Core speeds.
    assert_eq!(soma[0].max_speed, 4.5);
    assert_eq!(soma[0].default_speed, 2.3);
    assert_eq!(g1[0].max_speed, 3.4);
    assert_eq!(g1[0].default_speed, 1.3);
    // Probes.
    let soma_probe = presets::soma::interaction_probe();
    assert_eq!((soma_probe.box_width, soma_probe.box_height), (1.0, 1.6));
    let g1_probe = presets::unitree_g1::interaction_probe();
    assert_eq!((g1_probe.box_width, g1_probe.box_height), (0.79, 1.26));
}
