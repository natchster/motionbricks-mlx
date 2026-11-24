// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Locomotion modes, movement smoothing, and clip data for the characters, ported from
//! upstream `app/runtime/common/motion.py`. The critically damped springs come from
//! `motionbricks/utils/math_utils.py`; [`sample_root_position`] from
//! `motionbricks/utils/pose_utils.py`.
//!
//! Movement math uses [`glam`] vectors; poses use the `motionbricks-rs` types.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};
use glam::{Vec2, Vec3};
use motionbricks_rs::runtime as rt;
use rt::animation::MotionBuffer;
use rt::math::{self as rtm, Transform};

use super::behaviors::SmartPrimitiveAsset;
use super::blender::{BlendSpace, BlendSpaceType};

/// Directions, speeds, and input lengths at or below this count as zero.
pub const MOVEMENT_EPSILON: f32 = 1.0e-6;

pub const LINEAR_LOCOMOTION_MODE: &str = "linear_1d";
pub const DIRECTIONAL_LOCOMOTION_MODE: &str = "directional_2d";
pub const CROUCH_LOCOMOTION_MODE: &str = "crouch_directional_2d";

/// Strafing and Crouching share these targets, with Idle first.
pub const DIRECTIONAL_TARGET_VALUES: [(&str, [f32; 2]); 5] = [
    ("Idle", [0.0, 0.0]),
    ("Forward", [1.0, 0.0]),
    ("Backward", [-1.0, 0.0]),
    ("Left", [0.0, 1.0]),
    ("Right", [0.0, -1.0]),
];

/// Clips face -Y at their start, once model_to_runtime is applied.
pub const CLIP_FACING_YAW: f32 = -0.5 * std::f32::consts::PI;

/// Poses per generated token (the model contract).
pub const POSES_PER_TOKEN: usize = rt::MODEL_POSES_PER_TOKEN;

/// One value of a pose: a joint [`Transform`] (SOMA) or a joint coordinate (G1).
///
/// Both layouts start with the root: position (XYZ), then rotation (XYZW).
pub trait PoseElem: Copy + Default {
    /// The root transform of a pose in this layout.
    fn root_of(pose: &[Self]) -> Transform;
    /// Replace the root of a pose in this layout.
    fn write_root_of(pose: &mut [Self], root: Transform);
    /// The pose as joint transforms, when this is the SOMA layout.
    fn as_transforms(pose: &[Self]) -> Option<&[Transform]> {
        let _ = pose;
        None
    }
    /// The pose as mutable joint transforms, when this is the SOMA layout.
    fn as_transforms_mut(pose: &mut [Self]) -> Option<&mut [Transform]> {
        let _ = pose;
        None
    }
    /// Give the pose's quaternions the sign that matches the reference pose's rotation
    /// per joint (upstream `prepare_directional_poses` for 2D directional blending).
    /// A no-op for joint coordinates.
    fn align_quaternion_signs(pose: &mut [Self], reference: &[Self]) {
        let _ = (pose, reference);
    }
    /// Whether every value of the pose is finite.
    fn values_are_finite(pose: &[Self]) -> bool;
}

impl PoseElem for Transform {
    fn root_of(pose: &[Self]) -> Transform {
        pose[0]
    }
    fn write_root_of(pose: &mut [Self], root: Transform) {
        pose[0] = root;
    }
    fn as_transforms(pose: &[Self]) -> Option<&[Transform]> {
        Some(pose)
    }
    fn as_transforms_mut(pose: &mut [Self]) -> Option<&mut [Transform]> {
        Some(pose)
    }
    fn align_quaternion_signs(pose: &mut [Self], reference: &[Self]) {
        for (joint, reference) in pose.iter_mut().zip(reference) {
            let dot = reference.q[0] * joint.q[0] + reference.q[1] * joint.q[1] + reference.q[2] * joint.q[2] + reference.q[3] * joint.q[3];
            if dot < 0.0 {
                joint.q = [-joint.q[0], -joint.q[1], -joint.q[2], -joint.q[3]];
            }
        }
    }
    fn values_are_finite(pose: &[Self]) -> bool {
        pose.iter().all(|t| t.p.0.iter().chain(t.q.iter()).all(|v| v.is_finite()))
    }
}

impl PoseElem for f32 {
    fn root_of(pose: &[Self]) -> Transform {
        rt::read_root(pose)
    }
    fn write_root_of(pose: &mut [Self], root: Transform) {
        rt::write_root(root, pose);
    }
    fn values_are_finite(pose: &[Self]) -> bool {
        pose.iter().all(|v| v.is_finite())
    }
}

/// Convert a glam vector to the runtime API's vector.
pub fn to_rt(v: Vec3) -> rtm::Vec3 {
    rtm::Vec3::new(v.x, v.y, v.z)
}

/// Convert the runtime API's vector to glam.
pub fn to_glam(v: rtm::Vec3) -> Vec3 {
    Vec3::new(v.x(), v.y(), v.z())
}

/// `wp.quat_twist`: the twist component of `q` around a unit axis.
pub(crate) fn quat_twist(axis: rtm::Vec3, q: rtm::Quat) -> rtm::Quat {
    let d = q[0] * axis.x() + q[1] * axis.y() + q[2] * axis.z();
    rtm::quat_normalize([axis.x() * d, axis.y() * d, axis.z() * d, q[3]])
}

/// Project a transform onto the plane through the origin with this normal
/// (upstream `project_to_plane`).
pub(crate) fn project_to_plane(transform: Transform, plane_normal: rtm::Vec3) -> Transform {
    let normal = plane_normal.normalize();
    let position = transform.p - normal * transform.p.dot(normal);
    Transform::new(position, quat_twist(normal, transform.q))
}

/// Wrap an angle to [-pi, pi), upstream's `wrap_angle` (Python's floor modulo).
pub(crate) fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Advance a critically damped spring, returning its new value and rate (upstream
/// `critical_spring_damper_exact`). A spring within 1e-4 of the goal and nearly still
/// snaps to the goal.
pub(crate) fn critical_spring_damper_exact(current: f32, current_rate: f32, goal: f32, halflife: f32, delta_time: f32) -> (f32, f32) {
    let damping = (4.0 * std::f32::consts::LN_2) / (halflife + 1.0e-5);
    let decay_rate = damping / 2.0;
    let position_offset = current - goal;
    let velocity_offset = current_rate + position_offset * decay_rate;
    let decay = (-decay_rate * delta_time).exp();
    let next_current = decay * (position_offset + velocity_offset * delta_time) + goal;
    let next_rate = decay * (current_rate - velocity_offset * decay_rate * delta_time);
    if (next_current - goal).abs() <= 1.0e-4 && next_rate.abs() <= 1.0e-4 {
        return (goal, 0.0);
    }
    (next_current, next_rate)
}

/// Advance an angle spring toward a 2D direction by the shortest arc, returning the
/// new angle (wrapped to [-pi, pi)) and its rate (upstream `critical_spring_angle`).
pub(crate) fn critical_spring_angle(current_angle: f32, current_rate: f32, desired_direction: Vec3, halflife: f32, delta_time: f32) -> (f32, f32) {
    let desired_angle = desired_direction.y.atan2(desired_direction.x);
    let goal_angle = current_angle + wrap_angle(desired_angle - current_angle);
    let (angle, rate) = critical_spring_damper_exact(current_angle, current_rate, goal_angle, halflife, delta_time);
    (wrap_angle(angle), rate)
}

/// Where the character's current pose comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackSource {
    Generated,
    ClipTransition,
    Clip,
}

/// What the controls ask the character to do this frame.
#[derive(Clone, Debug, Default)]
pub struct LocomotionCommand {
    /// Ground direction the character faces in the 2D modes, in world space.
    /// Scene Preview passes the camera direction.
    pub facing_direction: Vec3,
    /// Movement input along the ground in world space. Its length is the input
    /// strength from 0 to 1. `None`, or a length of about zero, means no movement.
    pub movement: Option<Vec3>,
    /// Base the speed on the mode's maximum speed instead of its default speed.
    pub use_max_speed: bool,
    /// Name of the mode to switch to. `None` keeps the current mode.
    pub requested_mode: Option<String>,
}

/// How quickly movement responds to input and when to generate new motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocomotionSettings {
    pub facing_halflife: f32,
    pub velocity_direction_halflife: f32,
    pub speed_acceleration_halflife: f32,
    pub speed_deceleration_halflife: f32,
    pub speed_regeneration_fraction: f32,
    pub minimum_speed_regeneration_fraction: f32,
    pub direction_regeneration_dot: f32,
    pub maximum_direction_change_speed_reduction: f32,
}

impl Default for LocomotionSettings {
    fn default() -> Self {
        LocomotionSettings {
            facing_halflife: 0.15,
            velocity_direction_halflife: 0.2,
            speed_acceleration_halflife: 0.6,
            speed_deceleration_halflife: 0.4,
            speed_regeneration_fraction: 0.3,
            minimum_speed_regeneration_fraction: 0.1,
            direction_regeneration_dot: 0.92,
            maximum_direction_change_speed_reduction: 0.25,
        }
    }
}

/// Blend targets, blend space settings, and speed limits for one locomotion mode.
///
/// `smart_primitive_path` is the Smart Primitive document that holds the blend targets,
/// relative to the app's asset root; the bins resolve it.
#[derive(Clone, Debug)]
pub struct LocomotionMode {
    pub name: String,
    pub smart_primitive_path: PathBuf,
    pub blend_space_type: BlendSpaceType,
    /// Name and blend value of each blend target.
    pub target_values: Vec<(String, Vec2)>,
    pub max_speed: f32,
    pub default_speed: f32,
    pub minimum_speed: f32,
    pub transition_early_by_frames: usize,
    pub allowed_token_counts: Vec<i32>,
    /// Scale on the desired velocity for target speed and placement.
    pub velocity_scale: f32,
    /// How far ahead the targets sit, in seconds of travel at the scaled desired velocity.
    pub position_scale: f32,
    /// Name shown in the apps.
    pub label: String,
}

impl LocomotionMode {
    /// Create a mode, checking its settings as upstream's `LocomotionMode.__post_init__`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        smart_primitive_path: PathBuf,
        blend_space_type: BlendSpaceType,
        target_values: Vec<(String, Vec2)>,
        max_speed: f32,
        default_speed: f32,
        minimum_speed: f32,
        transition_early_by_frames: usize,
        allowed_token_counts: Vec<i32>,
        velocity_scale: f32,
        position_scale: f32,
        label: &str,
    ) -> Result<Self> {
        if name.is_empty() {
            bail!("Locomotion mode name cannot be empty.");
        }
        if target_values.is_empty() {
            bail!("Locomotion mode target_values cannot be empty.");
        }
        if max_speed <= 0.0 {
            bail!("Locomotion mode max_speed must be positive.");
        }
        if !(0.0 < default_speed && default_speed <= max_speed) {
            bail!("Locomotion mode default_speed must be positive and at most max_speed.");
        }
        if !(0.0 <= minimum_speed && minimum_speed <= default_speed) {
            bail!("Locomotion mode minimum_speed must be 0 to default_speed.");
        }
        if velocity_scale < 0.0 {
            bail!("Locomotion mode velocity_scale cannot be negative.");
        }
        if position_scale < 0.0 {
            bail!("Locomotion mode position_scale cannot be negative.");
        }
        if allowed_token_counts.is_empty() {
            bail!("Locomotion mode allowed_token_counts cannot be empty.");
        }
        let (min_tokens, max_tokens) = (rt::MODEL_MIN_TOKEN_COUNT, rt::MODEL_MAX_TOKEN_COUNT);
        if !allowed_token_counts.iter().all(|count| (min_tokens..=max_tokens).contains(count)) {
            bail!("Locomotion mode allowed_token_counts must be {min_tokens} to {max_tokens}.");
        }
        Ok(LocomotionMode {
            name: name.to_string(),
            smart_primitive_path,
            blend_space_type,
            target_values,
            max_speed,
            default_speed,
            minimum_speed,
            transition_early_by_frames,
            allowed_token_counts,
            velocity_scale,
            position_scale,
            label: label.to_string(),
        })
    }
}

/// Tuning of [`style_locomotion_mode`] beyond its speeds.
#[derive(Clone, Debug)]
pub struct StyleModeOptions {
    pub velocity_scale: f32,
    pub position_scale: f32,
    pub transition_early_by_frames: usize,
    pub allowed_token_counts: Vec<i32>,
}

impl Default for StyleModeOptions {
    fn default() -> Self {
        StyleModeOptions {
            velocity_scale: 1.0,
            position_scale: 1.5,
            transition_early_by_frames: 2,
            allowed_token_counts: (6..=15).collect(),
        }
    }
}

/// A 1D mode that blends from the Idle target at 0 to the style target at 1 (upstream
/// `style_locomotion_mode`). `target` names both the blend target and the mode, as in
/// `skipping_1d`; the style pose shows in full at `max_speed`.
pub fn style_locomotion_mode(target: &str, smart_primitive_path: PathBuf, minimum_speed: f32, max_speed: f32, options: StyleModeOptions) -> Result<LocomotionMode> {
    LocomotionMode::new(
        &format!("{}_1d", target.to_lowercase().replace(' ', "_")),
        smart_primitive_path,
        BlendSpaceType::Linear1D,
        vec![("Idle".to_string(), Vec2::ZERO), (target.to_string(), Vec2::X)],
        max_speed,
        max_speed,
        minimum_speed,
        options.transition_early_by_frames,
        options.allowed_token_counts,
        options.velocity_scale,
        options.position_scale,
        target,
    )
}

static CLIP_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A loaded clip and the data used to place it in world space (upstream `AnimationClip`).
///
/// The animation and root positions are shared cheaply; `root_offset` is per clip use
/// (aligning a clip rewrites it). `id` identifies the clip asset across clones, the role
/// object identity plays upstream (the idle states avoid repeating the current clip).
#[derive(Debug)]
pub struct AnimationClip<B: MotionBuffer> {
    pub id: u64,
    pub animation: Rc<B>,
    /// Where the clip starts on the ground, in source coordinates.
    pub source_ground: Transform,
    pub duration: f32,
    /// World transform applied to the sampled root.
    pub root_offset: Transform,
    /// Root position of each frame, in source coordinates.
    pub root_positions: Rc<Vec<rtm::Vec3>>,
}

impl<B: MotionBuffer> Clone for AnimationClip<B> {
    fn clone(&self) -> Self {
        AnimationClip {
            id: self.id,
            animation: self.animation.clone(),
            source_ground: self.source_ground,
            duration: self.duration,
            root_offset: self.root_offset,
            root_positions: self.root_positions.clone(),
        }
    }
}

/// Describe a clip and where it starts on the ground (upstream `make_animation_clip`).
///
/// The roots are in source coordinates.
pub fn make_animation_clip<B: MotionBuffer>(animation: B, model_to_runtime: Transform) -> Result<AnimationClip<B>>
where
    B::Elem: PoseElem,
{
    if animation.num_frames() == 0 {
        bail!("an animation clip needs at least one frame");
    }
    let first_pose = animation.sample_at(0.0, Transform::IDENTITY)?;
    let source_ground = project_to_plane(model_to_runtime * B::Elem::root_of(&first_pose), rtm::Vec3::new(0.0, 0.0, 1.0));
    let duration = (animation.num_frames().saturating_sub(1)).max(1) as f32 / animation.sample_rate();
    Ok(AnimationClip {
        id: CLIP_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
        root_positions: Rc::new((0..animation.num_frames()).map(|frame| animation.root_position(frame)).collect()),
        animation: Rc::new(animation),
        source_ground,
        duration,
        root_offset: Transform::IDENTITY,
    })
}

/// Interpolate cached root positions and place them in world space (upstream
/// `sample_root_position`). Times outside the positions are clamped.
pub fn sample_root_position(positions: &[rtm::Vec3], time: f32, sample_rate: f32, root_offset: Transform) -> rtm::Vec3 {
    let (frame, next_frame, blend) = rt::frame_index_from_time(time, sample_rate, positions.len());
    let position = positions[frame] * (1.0 - blend) + positions[next_frame] * blend;
    root_offset.point(position)
}

/// The blend space and target poses used by a locomotion mode (upstream
/// `LocomotionModeRuntime`). `target_poses` holds [`POSES_PER_TOKEN`] poses per target,
/// target-major.
pub struct LocomotionModeRuntime<E> {
    pub definition: LocomotionMode,
    pub blend_space: BlendSpace,
    pub target_poses: Vec<Vec<E>>,
}

impl<E> LocomotionModeRuntime<E> {
    pub fn target_count(&self) -> usize {
        self.target_poses.len() / POSES_PER_TOKEN
    }
    /// The four poses of one blend target.
    pub fn target(&self, target: usize) -> &[Vec<E>] {
        &self.target_poses[target * POSES_PER_TOKEN..(target + 1) * POSES_PER_TOKEN]
    }
}

/// Load a mode's target poses from its Smart Primitive and build its blend space
/// (upstream `load_locomotion_mode`). Every locomotion target needs all four poses,
/// because blending mixes whole targets.
pub fn load_locomotion_mode<E: PoseElem>(mode: &LocomotionMode, asset: &Rc<SmartPrimitiveAsset<E>>, reference_pose: &[E]) -> Result<LocomotionModeRuntime<E>> {
    let smart_primitive = super::behaviors::SmartPrimitive::new(asset.clone());
    let mut target_poses: Vec<Vec<E>> = Vec::with_capacity(mode.target_values.len() * POSES_PER_TOKEN);
    let mut blend_space = BlendSpace::new(mode.blend_space_type);
    for (data_index, (node_name, value)) in mode.target_values.iter().enumerate() {
        let node_index = smart_primitive.node_index(node_name);
        if node_index < 0 || !smart_primitive.has_target_pose(node_index as usize) {
            bail!("The {:?} Smart Primitive has no target named {node_name:?}.", mode.name);
        }
        let (poses, params) = smart_primitive.get_local_target_poses(node_index as usize)?;
        if params.num_poses_to_use != POSES_PER_TOKEN {
            bail!("The {:?} target {node_name:?} must use {POSES_PER_TOKEN} poses.", mode.name);
        }
        target_poses.extend(poses);
        blend_space.add_target(*value, data_index)?;
    }
    blend_space.compile()?;
    if asset.rig() == rt::ModelRig::Soma && mode.blend_space_type == BlendSpaceType::Directional2D {
        for pose in &mut target_poses {
            E::align_quaternion_signs(pose, reference_pose);
        }
    }
    Ok(LocomotionModeRuntime { definition: mode.clone(), blend_space, target_poses })
}

/// Return a ground direction of length one, using `fallback` when needed.
pub fn ground_direction(direction: Vec3, fallback: Vec3) -> Result<Vec3> {
    for candidate in [direction, fallback] {
        let ground = Vec3::new(candidate.x, candidate.y, 0.0);
        let length = ground.length();
        if length.is_finite() && length > MOVEMENT_EPSILON {
            return Ok(ground / length);
        }
    }
    bail!("direction and fallback have no valid ground direction.");
}

/// Return the ground position and facing of a scene's CharacterStart.
pub fn character_start_position_and_facing(transform: Transform) -> (Vec3, Vec3) {
    let ground_transform = project_to_plane(transform, rtm::Vec3::new(0.0, 0.0, 1.0));
    let position = to_glam(ground_transform.p);
    let world_facing = ground_transform.vector(rtm::Vec3::new(1.0, 0.0, 0.0));
    let facing = ground_direction(to_glam(world_facing), Vec3::X).expect("the +X fallback is a valid ground direction");
    (position, facing)
}

/// Express movement as a forward and left blend value relative to facing.
pub fn directional_blend_space_value(facing_direction: Vec3, desired_velocity: Vec3, max_speed: f32) -> Vec2 {
    let normalized_velocity = desired_velocity / max_speed;
    let facing_left = Vec3::new(-facing_direction.y, facing_direction.x, 0.0);
    Vec2::new(normalized_velocity.dot(facing_direction).clamp(-1.0, 1.0), normalized_velocity.dot(facing_left).clamp(-1.0, 1.0))
}

fn speed_reduction_scale(speed: f32, max_speed: f32, maximum_reduction: f32) -> f32 {
    let speed_fraction = (speed / max_speed).clamp(0.0, 1.0);
    1.0 - maximum_reduction * speed_fraction
}

/// Smooth movement commands into the desired velocity and facing (upstream
/// `MovementControl`).
///
/// Critically damped springs ease the speed, the velocity direction, and the facing.
/// Directions lie on the ground in world space, speeds are in meters per second, and
/// angles in radians.
#[derive(Clone, Debug)]
pub struct MovementControl {
    pub settings: LocomotionSettings,
    /// Speed the movement input asks for. It is 0 without movement input.
    pub input_speed: f32,
    /// Direction of the latest movement input.
    pub input_velocity_direction: Vec3,
    /// Facing that `current_facing` turns toward.
    pub desired_facing: Vec3,
    /// Facing at the latest movement input. Idle clips face this way.
    pub last_input_facing: Vec3,
    /// Smoothed facing; `current_facing_angle` and `current_facing_rate` hold its angle
    /// and turn rate.
    pub current_facing: Vec3,
    pub current_facing_angle: f32,
    pub current_facing_rate: f32,
    /// Smoothed velocity direction; `desired_velocity_angle` and
    /// `desired_velocity_direction_rate` hold its angle and turn rate.
    pub desired_velocity_direction: Vec3,
    pub desired_velocity_angle: f32,
    pub desired_velocity_direction_rate: f32,
    /// Smoothed speed the character aims for; `desired_speed_rate` holds its rate of change.
    pub desired_speed: f32,
    pub desired_speed_rate: f32,
    /// `desired_velocity_direction` times `desired_speed`.
    pub desired_velocity: Vec3,
}

impl MovementControl {
    /// Start at rest, facing along the ground direction `facing`.
    pub fn new(facing: Vec3) -> Self {
        Self::with_settings(facing, LocomotionSettings::default())
    }

    pub fn with_settings(facing: Vec3, settings: LocomotionSettings) -> Self {
        let angle = facing.y.atan2(facing.x);
        MovementControl {
            settings,
            input_speed: 0.0,
            input_velocity_direction: facing,
            desired_facing: facing,
            last_input_facing: facing,
            current_facing: facing,
            current_facing_angle: angle,
            current_facing_rate: 0.0,
            desired_velocity_direction: facing,
            desired_velocity_angle: angle,
            desired_velocity_direction_rate: 0.0,
            desired_speed: 0.0,
            desired_speed_rate: 0.0,
            desired_velocity: Vec3::ZERO,
        }
    }

    /// Input velocity on the ground before smoothing, in meters per second.
    pub fn input_velocity(&self) -> Vec3 {
        self.input_velocity_direction * self.input_speed
    }

    /// Apply one frame of movement input for the active mode.
    ///
    /// `previous_mode` is the mode this frame switched away from, or `None`. The switch
    /// eases in.
    pub fn update(&mut self, command: &LocomotionCommand, mode: &LocomotionMode, delta_time: f32, previous_mode: Option<&LocomotionMode>) {
        self.read_input(command, mode);
        if let Some(previous_mode) = previous_mode {
            self.ease_mode_switch(mode, previous_mode);
        }
        self.smooth(mode, delta_time);
    }

    /// Resume locomotion from the character's actual velocity. The input stays as it is.
    pub fn realign(&mut self, velocity: Vec3, mode: &LocomotionMode) {
        let speed = velocity.truncate().length();
        let direction = ground_direction(velocity, self.current_facing).expect("current_facing is a valid ground direction");
        let angle = direction.y.atan2(direction.x);
        self.desired_speed = speed;
        self.desired_speed_rate = 0.0;
        self.desired_velocity_direction = direction;
        self.desired_velocity_angle = angle;
        self.desired_velocity_direction_rate = 0.0;
        self.desired_velocity = direction * speed;
        if mode.blend_space_type == BlendSpaceType::Linear1D {
            self.set_current_facing(direction, angle, 0.0);
        }
    }

    /// Set every facing to `direction` at once.
    pub fn face(&mut self, direction: Vec3) {
        self.desired_facing = direction;
        self.last_input_facing = direction;
        self.set_current_facing(direction, direction.y.atan2(direction.x), 0.0);
    }

    /// Reset the desired velocity and the desired speed to zero.
    pub fn reset_desired_velocity(&mut self) {
        self.desired_speed = 0.0;
        self.desired_speed_rate = 0.0;
        self.desired_velocity = Vec3::ZERO;
    }

    fn set_current_facing(&mut self, direction: Vec3, angle: f32, rate: f32) {
        self.current_facing = direction;
        self.current_facing_angle = angle;
        self.current_facing_rate = rate;
    }

    fn read_input(&mut self, command: &LocomotionCommand, mode: &LocomotionMode) {
        let facing_direction = ground_direction(command.facing_direction, self.desired_facing).expect("desired_facing is a valid ground direction");
        let movement_amount = command.movement.map_or(0.0, |movement| movement.truncate().length());
        // Written as "not above", so a NaN input also counts as no movement.
        if movement_amount.partial_cmp(&MOVEMENT_EPSILON) != Some(std::cmp::Ordering::Greater) {
            if mode.blend_space_type == BlendSpaceType::Directional2D || self.desired_speed <= 0.0 {
                self.desired_facing = facing_direction;
            }
            self.input_speed = 0.0;
            return;
        }

        let movement = command.movement.expect("movement_amount is positive");
        let movement_direction = ground_direction(movement, self.input_velocity_direction).expect("the input direction is a valid fallback");
        self.input_velocity_direction = movement_direction;
        // 1D modes face the way they move. The 2D modes, Strafing and Crouching, face
        // the camera direction.
        self.desired_facing = if mode.blend_space_type == BlendSpaceType::Linear1D { movement_direction } else { facing_direction };
        self.last_input_facing = self.desired_facing;
        let requested_speed = if command.use_max_speed { mode.max_speed } else { mode.default_speed };
        self.input_speed = (requested_speed * movement_amount.min(1.0)).max(mode.minimum_speed);
    }

    /// Ease a mode switch in: directional modes snap the facing, and a switch to a
    /// mode that is not faster slows down a little, as a sharp turn does (upstream
    /// `_ease_mode_switch`).
    pub fn ease_mode_switch(&mut self, mode: &LocomotionMode, previous_mode: &LocomotionMode) {
        if mode.blend_space_type == BlendSpaceType::Directional2D {
            let facing = self.desired_facing;
            self.set_current_facing(facing, facing.y.atan2(facing.x), 0.0);
        }
        // A switch to a mode that is not faster slows down a little, as a sharp turn does.
        if mode.max_speed <= previous_mode.max_speed {
            let speed_scale = speed_reduction_scale(self.desired_speed, mode.max_speed, self.settings.maximum_direction_change_speed_reduction);
            self.desired_speed = (self.desired_speed * speed_scale).min(mode.max_speed);
            self.desired_speed_rate *= speed_scale;
        }
    }

    fn smooth(&mut self, mode: &LocomotionMode, delta_time: f32) {
        let settings = self.settings;
        let (angle, rate) = critical_spring_angle(
            self.desired_velocity_angle,
            self.desired_velocity_direction_rate,
            self.input_velocity_direction,
            settings.velocity_direction_halflife,
            delta_time,
        );
        self.desired_velocity_angle = angle;
        self.desired_velocity_direction_rate = rate;
        self.desired_velocity_direction = Vec3::new(angle.cos(), angle.sin(), 0.0);

        if mode.blend_space_type == BlendSpaceType::Linear1D {
            let (direction, angle, rate) = (self.desired_velocity_direction, self.desired_velocity_angle, self.desired_velocity_direction_rate);
            self.set_current_facing(direction, angle, rate);
        } else {
            let (angle, rate) = critical_spring_angle(self.current_facing_angle, self.current_facing_rate, self.desired_facing, settings.facing_halflife, delta_time);
            self.current_facing_angle = angle;
            self.current_facing_rate = rate;
            self.current_facing = Vec3::new(angle.cos(), angle.sin(), 0.0);
        }

        // A sharp turn slows the character until the velocity direction catches up.
        let mut speed_goal = self.input_speed;
        let turn_dot = self.desired_velocity_direction.dot(self.input_velocity_direction);
        if turn_dot < settings.direction_regeneration_dot {
            speed_goal *= speed_reduction_scale(self.desired_speed, mode.max_speed, settings.maximum_direction_change_speed_reduction);
        }
        let speed_halflife = if speed_goal > self.desired_speed { settings.speed_acceleration_halflife } else { settings.speed_deceleration_halflife };
        let (speed, rate) = critical_spring_damper_exact(self.desired_speed, self.desired_speed_rate, speed_goal, speed_halflife, delta_time);
        self.desired_speed = speed;
        self.desired_speed_rate = rate;
        if self.desired_speed > mode.max_speed {
            self.desired_speed = mode.max_speed;
            self.desired_speed_rate = self.desired_speed_rate.min(0.0);
        }
        self.desired_velocity = self.desired_velocity_direction * self.desired_speed;
    }
}
