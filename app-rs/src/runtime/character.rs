// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Play generated motion and clips for a character, and keep recent poses for the next
//! generation, ported from upstream `app/runtime/common/character.py` and the per-rig
//! characters (`soma/character.py`, `unitree_g1/character.py`).
//!
//! Generation is behind the [`EngineAdapter`] trait, so the character and its state
//! machine drive playback without knowing the model (Metal or MLX). The concrete
//! [`PlayerAdapter`] over the `motionbricks-rs` engines lives here; the bins construct
//! it from a backend, and tests substitute a scripted adapter.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
use glam::Vec3;
use motionbricks_rs::runtime as rt;
use rt::animation::MotionBuffer;
use rt::math::{self as rtm, Transform};
use rt::{
    Backend, FootContacts, GeneratedDofMotionPlayer, GeneratedMotionPlayer, GeneratedPlayer, MotionBricksEngine, MotionBricksSomaEngine,
    MotionBricksUnitreeG1Engine, Skeleton, TargetPoseIkCorrector, TargetPoseInferenceParams,
};

use super::behaviors::CharacterProperties;
use super::blender::TargetPoseBlender;
use super::interaction::{InteractionBox, InteractionProbe};
use super::motion::{
    AnimationClip, CLIP_FACING_YAW, LocomotionCommand, LocomotionModeRuntime, MovementControl, POSES_PER_TOKEN, PlaybackSource, PoseElem,
    character_start_position_and_facing, ground_direction, sample_root_position, to_glam, to_rt,
};
use super::state_machine::StateMachine;

/// The pose element type a character plays: a joint [`Transform`] (SOMA) or a joint
/// coordinate (G1).
pub type ElemOf<A> = <<A as EngineAdapter>::Buffer as MotionBuffer>::Elem;

/// How the character generates and plays motion without knowing the model.
///
/// This mirrors how `character.py` uses its motion player: start a generation from the
/// recorded context toward target poses, sample the output, and record the shown poses
/// for the next generation. The concrete adapter over the `motionbricks-rs` engines is
/// [`PlayerAdapter`]; tests use scripted adapters, so they never touch the GPU.
pub trait EngineAdapter {
    /// The generated-motion buffer type, which also types the character's clips.
    type Buffer: MotionBuffer;

    /// Generate from recent motion toward target poses placed in the scene, and
    /// restart playback. On failure the playing motion is kept.
    fn generate(&mut self, target_poses: &[Vec<ElemOf<Self>>], params: &TargetPoseInferenceParams) -> Result<()>;
    /// Advance playback by `delta_time`, write the pose into `pose`, and return its
    /// foot contacts.
    fn sample(&mut self, delta_time: f32, pose: &mut [ElemOf<Self>]) -> Result<FootContacts>;
    /// Record the pose that was shown, for the next generation's context.
    fn record_pose(&mut self, delta_time: f32, pose: &[ElemOf<Self>]) -> Result<()>;
    /// Fill every history slot and the context with one pose.
    fn seed_history_from_pose(&mut self, pose: &[ElemOf<Self>]) -> Result<()>;
    /// Seconds until the transition point. Sampling can continue past it.
    fn time_to_transition(&self) -> f64;
    /// Playback time of the transition point.
    fn playback_duration(&self) -> f64;
    /// The generated root position at the playback time.
    fn current_root_position(&self) -> rtm::Vec3;
}

/// The [`EngineAdapter`] over a `motionbricks-rs` engine's generated-motion player.
///
/// The bins build it with [`PlayerAdapter::new_soma`] or [`PlayerAdapter::new_g1`]
/// from an engine on their backend (Metal or MLX); everything else in this module
/// works against the trait.
pub struct PlayerAdapter<E: MotionBricksEngine> {
    player: GeneratedPlayer<E>,
}

impl<B: Backend> PlayerAdapter<MotionBricksSomaEngine<B>> {
    /// A SOMA player: the engine, the skeleton, and the motion history capacity.
    pub fn new_soma(engine: MotionBricksSomaEngine<B>, skeleton: &Arc<Skeleton>, history_capacity: usize) -> Result<Self> {
        Ok(PlayerAdapter { player: GeneratedMotionPlayer::new(engine, skeleton, history_capacity)? })
    }
}

impl<B: Backend> PlayerAdapter<MotionBricksUnitreeG1Engine<B>> {
    /// A G1 player: the engine, the reference joint coordinates that fill the history,
    /// and the motion history capacity.
    pub fn new_g1(engine: MotionBricksUnitreeG1Engine<B>, reference_joint_q: &[f32], history_capacity: usize) -> Result<Self> {
        Ok(PlayerAdapter { player: GeneratedDofMotionPlayer::new(engine, reference_joint_q, history_capacity)? })
    }
}

impl<E: MotionBricksEngine> PlayerAdapter<E> {
    /// The underlying player (its context, target, and motion history), for debug UI.
    pub fn player(&self) -> &GeneratedPlayer<E> {
        &self.player
    }
    pub fn player_mut(&mut self) -> &mut GeneratedPlayer<E> {
        &mut self.player
    }
}

impl<E: MotionBricksEngine> EngineAdapter for PlayerAdapter<E> {
    type Buffer = E::Output;

    fn generate(&mut self, target_poses: &[Vec<ElemOf<Self>>], params: &TargetPoseInferenceParams) -> Result<()> {
        self.player.generate(target_poses, params, None)
    }
    fn sample(&mut self, delta_time: f32, pose: &mut [ElemOf<Self>]) -> Result<FootContacts> {
        self.player.sample_into(delta_time, pose)
    }
    fn record_pose(&mut self, delta_time: f32, pose: &[ElemOf<Self>]) -> Result<()> {
        self.player.record_pose(delta_time, pose)
    }
    fn seed_history_from_pose(&mut self, pose: &[ElemOf<Self>]) -> Result<()> {
        self.player.seed_history_from_pose(pose)
    }
    fn time_to_transition(&self) -> f64 {
        self.player.time_to_transition()
    }
    fn playback_duration(&self) -> f64 {
        self.player.playback_duration()
    }
    fn current_root_position(&self) -> rtm::Vec3 {
        self.player.current_root_position()
    }
}

/// Load a clip for a character: its path, and whether to join its ends for looping.
/// The bins wire this to the rig module's clip registry.
pub type ClipLoader<B> = Box<dyn FnMut(&Path, bool) -> Result<AnimationClip<B>>>;

/// Generation settings of a clip transition: play the motion out to its end, using
/// all four target poses (upstream `_CLIP_TRANSITION_INFERENCE_PARAMS`).
fn clip_transition_inference_params() -> TargetPoseInferenceParams {
    TargetPoseInferenceParams::new(0, POSES_PER_TOKEN)
}

/// Target pose IK fades in from this fraction of a generated segment until its
/// transition point.
const TARGET_POSE_IK_BLEND_START_FRACTION: f64 = 0.5;

/// Warp's `smoothstep` between two edges.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The target pose IK weight: a smoothstep from 50% of the segment to its transition
/// point (upstream `_target_pose_ik_weight`).
pub fn target_pose_ik_weight(playback_duration: f64, time_to_transition: f64) -> f32 {
    if playback_duration <= 0.0 {
        return 1.0;
    }
    let progress = 1.0 - time_to_transition / playback_duration;
    smoothstep(TARGET_POSE_IK_BLEND_START_FRACTION, 1.0, progress) as f32
}

/// Animate a character with generated motion and clips through its own state machine
/// (upstream `BaseCharacter` and its rig subclasses).
///
/// The rig supplies its motion player (any [`EngineAdapter`]), its target pose
/// blender, and its `root_forward_axis`. A rig with target pose IK (SOMA) also
/// supplies an IK corrector. Playback follows upstream's three-source model:
/// generated playback, a clip transition, or a clip; the pose that is shown is
/// recorded for the next generation.
pub struct Character<A: EngineAdapter> {
    /// Movement smoothing: input, desired velocity, and facings.
    pub movement: MovementControl,
    /// Where the character stands on the ground (z is always 0).
    pub character_ground_position: Vec3,
    /// Measured ground velocity from the last playback step, in m/s.
    pub actual_velocity: Vec3,
    /// The pose that is shown and recorded, including any IK adjustments.
    pub current_pose: Vec<ElemOf<A>>,
    /// Foot contacts of the shown pose. They stay at 0 while a clip plays.
    pub current_foot_contacts: FootContacts,
    /// The character's state machine.
    pub state_machine: StateMachine<A>,
    /// Size and reach of the probe box.
    pub interaction_probe: InteractionProbe,
    /// The character's current probe box in world space.
    pub probe_box: InteractionBox,
    player: A,
    target_pose_blender: Box<dyn TargetPoseBlender<ElemOf<A>>>,
    ik_corrector: Option<TargetPoseIkCorrector>,
    model_to_runtime: Transform,
    root_forward_axis: Vec3,
    playback_source: PlaybackSource,
    current_clip: Option<AnimationClip<A::Buffer>>,
    clip_playback_time: f32,
    clip_loop: bool,
    clip_target_poses: Vec<Vec<ElemOf<A>>>,
    clip_loader: ClipLoader<A::Buffer>,
}

impl<A: EngineAdapter> Character<A>
where
    ElemOf<A>: PoseElem,
{
    /// Start the character in an idle clip at `initial_transform`.
    ///
    /// The arguments mirror upstream's `BaseCharacter.__init__`: `player` plays the
    /// generated motion; `mode_runtimes` are the loaded locomotion modes (standing
    /// and crouching); `standing_idle_clips` and `crouching_idle_clips` are loaded,
    /// loop-corrected clips; `ik_corrector` is target pose IK for pinned hands and
    /// feet (`None` applies no IK, as on G1); `reference_pose` initializes the shown
    /// pose; `clip_loader` loads further clips (Smart Primitive behavior clips).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player: A,
        target_pose_blender: Box<dyn TargetPoseBlender<ElemOf<A>>>,
        ik_corrector: Option<TargetPoseIkCorrector>,
        root_forward_axis: Vec3,
        model_to_runtime: Transform,
        initial_transform: Transform,
        mode_runtimes: Vec<LocomotionModeRuntime<ElemOf<A>>>,
        initial_mode: &str,
        standing_idle_clips: Vec<AnimationClip<A::Buffer>>,
        crouching_idle_clips: Vec<AnimationClip<A::Buffer>>,
        interaction_probe: InteractionProbe,
        reference_pose: &[ElemOf<A>],
        clip_loader: ClipLoader<A::Buffer>,
    ) -> Result<Self> {
        let (position, facing) = character_start_position_and_facing(initial_transform);
        let state_machine = StateMachine::new(mode_runtimes, initial_mode, standing_idle_clips, crouching_idle_clips, reference_pose.len())?;
        let mut character = Character {
            movement: MovementControl::new(facing),
            character_ground_position: position,
            actual_velocity: Vec3::ZERO,
            current_pose: reference_pose.to_vec(),
            current_foot_contacts: [0.0; rt::FOOT_CONTACT_COUNT],
            state_machine,
            interaction_probe,
            probe_box: InteractionBox { transform: Transform::IDENTITY, half_extents: Vec3::ZERO },
            player,
            target_pose_blender,
            ik_corrector,
            model_to_runtime,
            root_forward_axis,
            playback_source: PlaybackSource::Generated,
            current_clip: None,
            clip_playback_time: 0.0,
            clip_loop: false,
            clip_target_poses: vec![reference_pose.to_vec(); POSES_PER_TOKEN],
            clip_loader,
        };
        let mut machine = std::mem::replace(&mut character.state_machine, StateMachine::placeholder());
        let result = machine.start(&mut character);
        character.state_machine = machine;
        result?;
        character.update_probe_box();
        Ok(character)
    }

    /// The engine adapter (the generated-motion player).
    pub fn player(&self) -> &A {
        &self.player
    }

    /// Load a clip through the character's clip loader. A looping clip gets its own
    /// copy with its ends joined.
    pub fn load_clip(&mut self, path: &Path, loop_: bool) -> Result<AnimationClip<A::Buffer>> {
        (self.clip_loader)(path, loop_)
    }

    /// Where the character's current pose comes from.
    pub fn playback_source(&self) -> PlaybackSource {
        self.playback_source
    }

    /// Latest clip that played or that a clip transition leads into.
    pub fn current_clip(&self) -> Option<&AnimationClip<A::Buffer>> {
        self.current_clip.as_ref()
    }

    /// The near and far ends of the character's probe box on the ground.
    pub fn probe_segment(&self) -> (Vec3, Vec3) {
        self.interaction_probe.segment(
            self.character_ground_position,
            self.movement.desired_velocity_direction,
            self.movement.desired_velocity.truncate().length(),
        )
    }

    /// The character's values for Smart Primitive behaviors, in world space.
    pub fn character_properties(&self) -> CharacterProperties {
        CharacterProperties {
            position: self.character_ground_position,
            velocity: self.actual_velocity,
            facing: self.movement.current_facing,
            desired_velocity: self.movement.desired_velocity,
            probe_segment: self.probe_segment(),
        }
    }

    /// Stop applying the current target's hand and foot pins.
    pub fn clear_ik_pins(&mut self) {
        if let Some(corrector) = &mut self.ik_corrector {
            corrector.clear_pins();
        }
    }

    /// Whether a clip transition reaches its transition point within `delta_time` (in
    /// seconds).
    pub fn clip_transition_ends_within(&self, delta_time: f32) -> bool {
        self.playback_source == PlaybackSource::ClipTransition && self.time_to_transition() <= delta_time as f64
    }

    /// Whether the current clip ends within `delta_time`. Looping clips never end.
    pub fn clip_ends_within(&self, delta_time: f32) -> bool {
        self.playback_source == PlaybackSource::Clip
            && !self.clip_loop
            && self.current_clip.as_ref().is_some_and(|clip| self.clip_playback_time + delta_time >= clip.duration)
    }

    /// Seconds left until the transition point. Clip playback returns 0.
    pub fn time_to_transition(&self) -> f64 {
        if self.playback_source == PlaybackSource::Clip { 0.0 } else { self.player.time_to_transition() }
    }

    /// Advance the state machine and playback by `delta_time` (in seconds).
    ///
    /// The pose that is shown is then recorded for the next generation.
    pub fn step(&mut self, delta_time: f32, command: &LocomotionCommand) -> Result<()> {
        self.with_state_machine(|machine, character| machine.update(character, delta_time, command))?;
        self.step_playback_and_record(delta_time)
    }

    fn with_state_machine(&mut self, run: impl FnOnce(&mut StateMachine<A>, &mut Self) -> Result<()>) -> Result<()> {
        let mut machine = std::mem::replace(&mut self.state_machine, StateMachine::placeholder());
        let result = run(&mut machine, self);
        self.state_machine = machine;
        result
    }

    /// Let a Smart Primitive control the character until its behavior calls finish.
    pub fn start_interaction(&mut self, primitive: Box<dyn super::behaviors::SmartPrimitiveBehavior<ElemOf<A>>>) -> Result<()> {
        self.with_state_machine(|machine, character| machine.start_interaction(character, primitive))
    }

    /// Pass an interaction press to the active Smart Primitive behavior.
    pub fn interact(&mut self) -> Result<()> {
        self.with_state_machine(|machine, character| machine.interact(character))
    }

    fn step_playback_and_record(&mut self, delta_time: f32) -> Result<()> {
        let previous_ground_position = self.character_ground_position;
        self.sample_playback(delta_time)?;
        self.actual_velocity = (self.character_ground_position - previous_ground_position) / delta_time;
        self.actual_velocity.z = 0.0;
        self.player.record_pose(delta_time, &self.current_pose)?;
        self.update_probe_box();
        Ok(())
    }

    fn update_probe_box(&mut self) {
        self.probe_box = self.interaction_probe.box_for_segment(self.probe_segment(), self.state_machine.probe_box_height_scale());
    }

    /// Advance playback by the full frame time and sample the resulting pose.
    fn sample_playback(&mut self, delta_time: f32) -> Result<()> {
        match self.playback_source {
            PlaybackSource::Clip => {
                let clip = self.current_clip.as_ref().expect("clip playback holds a clip");
                self.clip_playback_time += delta_time;
                if self.clip_loop {
                    self.clip_playback_time %= clip.duration;
                }
                clip.animation.sample_into_at(self.clip_playback_time as f64, clip.root_offset, &mut self.current_pose)?;
            }
            _ => {
                self.current_foot_contacts = self.player.sample(delta_time, &mut self.current_pose)?;
                self.apply_target_pose_ik()?;
            }
        }
        self.update_character_position();
        Ok(())
    }

    /// Generate from recent motion toward target poses placed in the scene.
    ///
    /// Target pose IK pins the hands and feet in the target's `ik_pin_mask` to its
    /// first pose. Other targets clear the pins. `playback_source` is
    /// [`PlaybackSource::ClipTransition`] when the motion leads into a clip.
    pub fn generate_to_target_poses(&mut self, target_poses: &[Vec<ElemOf<A>>], params: &TargetPoseInferenceParams, playback_source: PlaybackSource) -> Result<()> {
        if target_poses.is_empty() {
            bail!("generation needs at least one target pose");
        }
        if let Some(corrector) = &mut self.ik_corrector {
            if let Some(pose) = ElemOf::<A>::as_transforms(&target_poses[0]) {
                corrector.capture_pins(pose, params.ik_pin_mask)?;
            }
        }
        self.player.generate(target_poses, params)?;
        self.playback_source = playback_source;
        self.current_foot_contacts = self.player.sample(0.0, &mut self.current_pose)?;
        self.apply_target_pose_ik()?;
        self.update_character_position();
        Ok(())
    }

    /// Place a clip ahead of the character and generate motion to its first frame.
    pub fn generate_clip_transition(&mut self, clip: AnimationClip<A::Buffer>) -> Result<()> {
        let mut clip = clip;
        let clip_ground_position = self.character_ground_position + self.actual_velocity;
        Self::align_clip(&mut clip, clip_ground_position, self.movement.last_input_facing, self.model_to_runtime);
        for (frame, target) in self.clip_target_poses.iter_mut().enumerate().take(POSES_PER_TOKEN) {
            clip.animation.sample_into_at(frame as f64 / rt::MODEL_SAMPLE_RATE as f64, clip.root_offset, target)?;
        }
        self.current_clip = Some(clip);
        let target_poses = self.clip_target_poses.clone();
        self.generate_to_target_poses(&target_poses, &clip_transition_inference_params(), PlaybackSource::ClipTransition)
    }

    /// Turn the clip from the direction clips start facing to the requested facing.
    fn align_clip(clip: &mut AnimationClip<A::Buffer>, ground_position: Vec3, facing_direction: Vec3, model_to_runtime: Transform) {
        let yaw = facing_direction.y.atan2(facing_direction.x) - CLIP_FACING_YAW;
        let facing_rotation = rtm::quat_from_axis_angle(rtm::Vec3::new(0.0, 0.0, 1.0), yaw);
        let target_ground = Transform::new(rtm::Vec3::new(ground_position.x, ground_position.y, 0.0), rtm::quat_mul(facing_rotation, clip.source_ground.q));
        let alignment = target_ground * clip.source_ground.inverse();
        clip.root_offset = alignment * model_to_runtime;
    }

    /// Play a clip, optionally looping it or aligning its first root with the current
    /// root.
    ///
    /// The motion history is kept for the next generation. Aligning the roots also
    /// converts source coordinates to world space.
    pub fn start_clip(&mut self, mut clip: AnimationClip<A::Buffer>, loop_: bool, align_to_current_pose: bool) -> Result<()> {
        if align_to_current_pose {
            let first_pose = clip.animation.sample_at(0.0, Transform::IDENTITY)?;
            clip.root_offset = ElemOf::<A>::root_of(&self.current_pose) * ElemOf::<A>::root_of(&first_pose).inverse();
        }
        self.current_clip = Some(clip.clone());
        self.clip_playback_time = 0.0;
        self.clip_loop = loop_;
        self.playback_source = PlaybackSource::Clip;
        clip.animation.sample_into_at(0.0, clip.root_offset, &mut self.current_pose)?;
        self.current_foot_contacts = [0.0; rt::FOOT_CONTACT_COUNT];
        self.update_character_position();
        Ok(())
    }

    /// Start in an idle clip and fill the player's history with the starting pose.
    pub fn initialize_clip_loop(&mut self, clip: AnimationClip<A::Buffer>) -> Result<()> {
        let mut clip = clip;
        Self::align_clip(&mut clip, self.character_ground_position, self.movement.last_input_facing, self.model_to_runtime);
        self.start_clip(clip, true, false)?;
        self.player.seed_history_from_pose(&self.current_pose)
    }

    /// The current pose's heading, from its root rotation.
    pub fn pose_facing(&self) -> Vec3 {
        let root = ElemOf::<A>::root_of(&self.current_pose);
        let forward = root.vector(to_rt(self.root_forward_axis));
        ground_direction(to_glam(forward), self.movement.current_facing).unwrap_or(self.movement.current_facing)
    }

    /// Set every facing to the current pose's heading.
    pub fn face_pose(&mut self) {
        let facing = self.pose_facing();
        self.movement.face(facing);
    }

    /// Blend and place a locomotion mode's targets with the character's blender, for
    /// generation or debug display.
    pub fn evaluate_target_poses(&self, mode_runtime: &LocomotionModeRuntime<ElemOf<A>>, blended_poses: &mut Vec<Vec<ElemOf<A>>>, target_poses: &mut Vec<Vec<ElemOf<A>>>) -> Result<()> {
        let context = super::blender::TargetPoseContext {
            ground_position: self.character_ground_position,
            current_facing: self.movement.current_facing,
            desired_velocity: self.movement.desired_velocity,
        };
        self.target_pose_blender.evaluate(&context, mode_runtime, blended_poses, target_poses)
    }

    fn apply_target_pose_ik(&mut self) -> Result<()> {
        // The corrector does nothing while the current target pins no hands or feet.
        let Some(corrector) = &mut self.ik_corrector else {
            return Ok(());
        };
        let weight = target_pose_ik_weight(self.player.playback_duration(), self.player.time_to_transition());
        if let Some(pose) = ElemOf::<A>::as_transforms_mut(&mut self.current_pose) {
            corrector.apply(pose, weight)?;
        }
        Ok(())
    }

    fn update_character_position(&mut self) {
        let position = match self.playback_source {
            PlaybackSource::Clip => {
                let clip = self.current_clip.as_ref().expect("clip playback holds a clip");
                sample_root_position(&clip.root_positions, self.clip_playback_time, clip.animation.sample_rate(), clip.root_offset)
            }
            _ => self.player.current_root_position(),
        };
        self.character_ground_position = Vec3::new(position.x(), position.y(), 0.0);
    }
}
