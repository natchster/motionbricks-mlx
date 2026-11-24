// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Idle, locomotion, and interaction states shared by both rigs, ported from upstream
//! `app/runtime/common/states.py`.
//!
//! Rust's ownership changes the wiring, not the behavior: the state machine moves the
//! active state out while it runs (`State::update`), and a state asks for a transition
//! by writing a `deferred` target that the machine applies once the state is back in
//! place. Behavior callbacks queue [`InteractionCtl`] requests that the interaction
//! state applies after the callback returns.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use glam::Vec3;
use motionbricks_rs::runtime as rt;
use rt::TargetPoseInferenceParams;
use rt::animation::MotionBuffer;

use super::behaviors::{Completion, InteractionCtl, SmartPrimitiveBehavior};
use super::character::{Character, ElemOf, EngineAdapter};
use super::motion::{
    AnimationClip, LocomotionMode, LocomotionModeRuntime, LocomotionSettings, MOVEMENT_EPSILON, MovementControl, POSES_PER_TOKEN, PlaybackSource, PoseElem,
};
use super::state_machine::{ActiveState, StateMachine};

/// Generate a transition into an idle clip, then loop the clip (upstream `IdleState`).
pub struct IdleState<B: MotionBuffer> {
    name: &'static str,
    clips: Vec<AnimationClip<B>>,
    rng: u64,
}

impl<B: MotionBuffer> Default for IdleState<B> {
    /// The placeholder a state machine leaves behind while a state is moved out.
    fn default() -> Self {
        IdleState { name: "Idle", clips: Vec::new(), rng: 0 }
    }
}

impl<B: MotionBuffer> IdleState<B> {
    /// Create an idle state that picks from the given looping clips.
    pub fn new(name: &'static str, clips: Vec<AnimationClip<B>>) -> Result<Self> {
        if clips.is_empty() {
            bail!("Idle state requires at least one clip.");
        }
        Ok(IdleState { name, clips, rng: 0x9E37_79B9_7F4A_7C15 })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The looping clips this state picks from.
    pub fn clips(&self) -> &[AnimationClip<B>] {
        &self.clips
    }

    /// Pick a random idle clip, preferring one other than the current clip.
    pub fn choose_clip(&mut self, current_clip: Option<&AnimationClip<B>>) -> AnimationClip<B> {
        let candidates: Vec<usize> = (0..self.clips.len()).filter(|&index| current_clip.is_none_or(|clip| self.clips[index].id != clip.id)).collect();
        let candidates = if candidates.is_empty() { (0..self.clips.len()).collect() } else { candidates };
        // xorshift64* stands in for Python's `random`.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let pick = (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as usize % candidates.len();
        self.clips[candidates[pick]].clone()
    }

    /// Start in an idle loop, without a generated transition.
    pub fn start<A>(&mut self, character: &mut Character<A>) -> Result<()>
    where
        A: EngineAdapter<Buffer = B>,
        ElemOf<A>: PoseElem,
    {
        character.movement.reset_desired_velocity();
        let clip = self.choose_clip(character.current_clip());
        character.initialize_clip_loop(clip)
    }

    fn enter<A>(&mut self, character: &mut Character<A>) -> Result<()>
    where
        A: EngineAdapter<Buffer = B>,
        ElemOf<A>: PoseElem,
    {
        character.movement.reset_desired_velocity();
        let clip = self.choose_clip(character.current_clip());
        character.generate_clip_transition(clip)
    }

    fn update<A>(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, delta_time: f32, deferred: &mut Option<ActiveState>) -> Result<()>
    where
        A: EngineAdapter<Buffer = B>,
        ElemOf<A>: PoseElem,
    {
        if character.movement.input_speed > 0.0 {
            *deferred = Some(machine.locomotion_key());
            return Ok(());
        }
        if character.clip_transition_ends_within(delta_time) {
            let clip = character.current_clip().expect("a clip transition leads into a clip").clone();
            character.start_clip(clip, true, false)?;
        }
        Ok(())
    }
}

/// Place target poses ahead along the desired velocity, and generate toward them
/// (upstream `LocomotionState`).
///
/// The character's [`MovementControl`] smooths the input into that velocity.
pub struct LocomotionState<E> {
    name: &'static str,
    settings: LocomotionSettings,
    mode_runtimes: Vec<LocomotionModeRuntime<E>>,
    active_mode_index: usize,
    blended_target_poses: Vec<Vec<E>>,
    target_poses: Vec<Vec<E>>,
    previous_generation_desired_velocity: Vec3,
    force_regeneration: bool,
}

impl<E> Default for LocomotionState<E> {
    /// The placeholder a state machine leaves behind while a state is moved out.
    fn default() -> Self {
        LocomotionState {
            name: "Locomotion",
            settings: LocomotionSettings::default(),
            mode_runtimes: Vec::new(),
            active_mode_index: 0,
            blended_target_poses: Vec::new(),
            target_poses: Vec::new(),
            previous_generation_desired_velocity: Vec3::ZERO,
            force_regeneration: false,
        }
    }
}

impl<E: PoseElem> LocomotionState<E> {
    /// Create a locomotion state with its loaded modes, initial mode, and regeneration
    /// settings. `pose_len` sizes the target pose buffers.
    pub fn new(name: &'static str, mode_runtimes: Vec<LocomotionModeRuntime<E>>, initial_mode: &str, pose_len: usize, settings: LocomotionSettings) -> Result<Self> {
        let Some(active_mode_index) = mode_runtimes.iter().position(|runtime| runtime.definition.name == initial_mode) else {
            bail!("Unknown initial locomotion mode {initial_mode:?}.");
        };
        Ok(LocomotionState {
            name,
            settings,
            mode_runtimes,
            active_mode_index,
            blended_target_poses: vec![vec![E::default(); pose_len]; POSES_PER_TOKEN],
            target_poses: vec![vec![E::default(); pose_len]; POSES_PER_TOKEN],
            previous_generation_desired_velocity: Vec3::ZERO,
            force_regeneration: true,
        })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Definition of the active mode.
    pub fn active_mode(&self) -> &LocomotionMode {
        &self.mode_runtimes[self.active_mode_index].definition
    }

    /// The active mode's blend space and target poses.
    pub fn active_mode_runtime(&self) -> &LocomotionModeRuntime<E> {
        &self.mode_runtimes[self.active_mode_index]
    }

    /// The modes this state can switch between, in the order they were given.
    pub fn mode_names(&self) -> Vec<&str> {
        self.mode_runtimes.iter().map(|runtime| runtime.definition.name.as_str()).collect()
    }

    /// Make a mode active. A new mode generates new motion on the next update.
    pub fn select_mode(&mut self, name: &str) -> Result<()> {
        let Some(index) = self.mode_runtimes.iter().position(|runtime| runtime.definition.name == name) else {
            bail!("Unknown locomotion mode {name:?}.");
        };
        if index != self.active_mode_index {
            self.active_mode_index = index;
            self.force_regeneration = true;
        }
        Ok(())
    }

    fn enter<A: EngineAdapter>(&mut self, character: &mut Character<A>) -> Result<()>
    where
        ElemOf<A>: PoseElem,
        A: EngineAdapter<Buffer: MotionBuffer<Elem = E>>,
    {
        character.movement.realign(character.actual_velocity, self.active_mode());
        self.generate(character)
    }

    fn update<A: EngineAdapter>(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, delta_time: f32, deferred: &mut Option<ActiveState>) -> Result<()>
    where
        ElemOf<A>: PoseElem,
        A: EngineAdapter<Buffer: MotionBuffer<Elem = E>>,
    {
        if character.movement.input_speed <= 0.0 {
            *deferred = Some(machine.idle_key());
            return Ok(());
        }
        if self.force_regeneration || character.time_to_transition() <= delta_time as f64 || self.needs_regeneration(&character.movement) {
            self.generate(character)?;
        }
        Ok(())
    }

    /// Blend and place the active mode's targets for generation or debug display.
    pub fn evaluate_current_target_poses<A: EngineAdapter>(&self, character: &Character<A>, blended_poses: &mut Vec<Vec<ElemOf<A>>>, target_poses: &mut Vec<Vec<ElemOf<A>>>) -> Result<()>
    where
        ElemOf<A>: PoseElem,
        A: EngineAdapter<Buffer: MotionBuffer<Elem = E>>,
    {
        character.evaluate_target_poses(self.active_mode_runtime(), blended_poses, target_poses)
    }

    fn generate<A: EngineAdapter>(&mut self, character: &mut Character<A>) -> Result<()>
    where
        ElemOf<A>: PoseElem,
        A: EngineAdapter<Buffer: MotionBuffer<Elem = E>>,
    {
        let LocomotionState { mode_runtimes, active_mode_index, blended_target_poses, target_poses, .. } = self;
        let runtime = &mode_runtimes[*active_mode_index];
        character.evaluate_target_poses(runtime, blended_target_poses, target_poses)?;
        let mode = &runtime.definition;
        character.generate_to_target_poses(
            target_poses,
            &TargetPoseInferenceParams {
                transition_early_by_frames: mode.transition_early_by_frames,
                num_poses_to_use: POSES_PER_TOKEN,
                ik_pin_mask: rt::ik_pin::NONE,
                allowed_token_counts: Some(mode.allowed_token_counts.clone()),
            },
            PlaybackSource::Generated,
        )?;
        self.previous_generation_desired_velocity = character.movement.desired_velocity;
        self.force_regeneration = false;
        Ok(())
    }

    /// Whether the smoothed velocity moved far enough from the last generation's to
    /// generate new motion (upstream `_needs_regeneration`).
    pub fn needs_regeneration(&self, movement: &MovementControl) -> bool {
        let settings = &self.settings;
        let previous_speed = self.previous_generation_desired_velocity.length();
        let max_speed = self.active_mode().max_speed;
        let locomotion_started = movement.desired_speed > 0.0 && previous_speed <= MOVEMENT_EPSILON;
        let speed_delta = movement.desired_speed - previous_speed;
        let remaining_speed_range = if speed_delta > 0.0 { max_speed - previous_speed } else { previous_speed };
        let regeneration_threshold = (remaining_speed_range * settings.speed_regeneration_fraction).max(max_speed * settings.minimum_speed_regeneration_fraction);
        if locomotion_started || speed_delta.abs() > regeneration_threshold {
            return true;
        }
        if previous_speed <= MOVEMENT_EPSILON {
            return false;
        }
        let previous_direction = self.previous_generation_desired_velocity / previous_speed;
        movement.desired_velocity_direction.dot(previous_direction) < settings.direction_regeneration_dot
    }
}

/// Queued behavior requests, applied once the behavior's call returns.
enum InteractionRequest<E> {
    PlaySequence { name: Option<String>, on_complete: Option<Completion<E>> },
    PlayClip { path: PathBuf, loop_: bool, align: bool, on_complete: Option<Completion<E>> },
    Finish,
}

struct InteractionQueue<E> {
    is_looping: bool,
    requests: Vec<InteractionRequest<E>>,
}

impl<E> InteractionCtl<E> for InteractionQueue<E> {
    fn play_target_pose_sequence(&mut self, name: Option<&str>, on_complete: Option<Completion<E>>) {
        self.requests.push(InteractionRequest::PlaySequence { name: name.map(str::to_string), on_complete });
    }
    fn play_clip(&mut self, path: &Path, loop_: bool, align_to_current_pose: bool, on_complete: Option<Completion<E>>) {
        self.requests.push(InteractionRequest::PlayClip { path: path.to_path_buf(), loop_, align: align_to_current_pose, on_complete });
    }
    fn finish(&mut self) {
        self.requests.push(InteractionRequest::Finish);
    }
    fn is_looping(&self) -> bool {
        self.is_looping
    }
}

fn resolve_clip_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Run a Smart Primitive's sequences and clips until the behavior finishes (upstream
/// `InteractionState`).
///
/// The behavior chooses the next sequence or clip. This state generates motion toward
/// each target. The character plays that motion and records its recent poses.
pub struct InteractionState<A: EngineAdapter> {
    primitive: Option<Box<dyn SmartPrimitiveBehavior<ElemOf<A>>>>,
    target_poses: Vec<Vec<ElemOf<A>>>,
    inference_params: Vec<TargetPoseInferenceParams>,
    target_index: i64,
    playing_clip: bool,
    clip_loop: bool,
    clips: HashMap<(PathBuf, bool), AnimationClip<A::Buffer>>,
    on_complete: Option<Completion<ElemOf<A>>>,
}

impl<A: EngineAdapter> Default for InteractionState<A> {
    /// The placeholder a state machine leaves behind while a state is moved out.
    fn default() -> Self {
        InteractionState {
            primitive: None,
            target_poses: Vec::new(),
            inference_params: Vec::new(),
            target_index: -1,
            playing_clip: false,
            clip_loop: false,
            clips: HashMap::new(),
            on_complete: None,
        }
    }
}

impl<A: EngineAdapter> InteractionState<A>
where
    ElemOf<A>: PoseElem,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// The behavior running the active interaction, if any.
    pub fn primitive(&self) -> Option<&dyn SmartPrimitiveBehavior<ElemOf<A>>> {
        self.primitive.as_deref()
    }

    /// Whether a looping clip is playing.
    pub fn is_looping(&self) -> bool {
        self.playing_clip && self.clip_loop
    }

    /// Prepare a Smart Primitive behavior for the next interaction (upstream
    /// `configure` and `prepare`): load its clips, looping and not, and size the
    /// target pose storage for it.
    pub fn configure(&mut self, character: &mut Character<A>, primitive: Box<dyn SmartPrimitiveBehavior<ElemOf<A>>>) -> Result<()> {
        for path in primitive.clip_paths() {
            let path = resolve_clip_path(&path);
            for loop_ in [false, true] {
                if !self.clips.contains_key(&(path.clone(), loop_)) {
                    let clip = character.load_clip(&path, loop_)?;
                    self.clips.insert((path.clone(), loop_), clip);
                }
            }
        }
        self.target_poses.clear();
        self.target_poses.reserve(primitive.primitive().target_pose_count().max(primitive.primitive().max_sequence_length()).max(1) * POSES_PER_TOKEN);
        self.primitive = Some(primitive);
        Ok(())
    }

    fn enter(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, deferred: &mut Option<ActiveState>) -> Result<()> {
        let mut behavior = self.primitive.take().ok_or_else(|| anyhow!("Cannot enter an interaction without a configured behavior."))?;
        let mut queue = InteractionQueue { is_looping: false, requests: Vec::new() };
        let result = behavior.enter(&mut queue);
        self.primitive = Some(behavior);
        result?;
        self.apply_requests(character, machine, queue.requests, deferred)
    }

    fn exit(&mut self, character: &mut Character<A>) {
        self.stop_target_pose_sequence(character);
        // Behaviors can turn the character, so movement continues from the pose's heading.
        character.face_pose();
        self.primitive = None;
        self.inference_params = Vec::new();
        self.playing_clip = false;
    }

    /// Pass an interaction press to the active behavior.
    fn interact(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, deferred: &mut Option<ActiveState>) -> Result<()> {
        let mut behavior = self.primitive.take().ok_or_else(|| anyhow!("Cannot interact without an active interaction."))?;
        let mut queue = InteractionQueue { is_looping: self.is_looping(), requests: Vec::new() };
        behavior.interact(&mut queue);
        self.primitive = Some(behavior);
        self.apply_requests(character, machine, queue.requests, deferred)
    }

    /// Generate motion through a target pose sequence, then run `on_complete` (or
    /// return to locomotion or idle without one).
    fn play_target_pose_sequence(&mut self, character: &mut Character<A>, name: Option<String>, on_complete: Option<Completion<ElemOf<A>>>) -> Result<()> {
        let properties = character.character_properties();
        let mut behavior = self.primitive.take().ok_or_else(|| anyhow!("Cannot play a sequence without an active interaction."))?;
        let result = match name.as_deref() {
            None => behavior.get_targets(&properties),
            Some(name) => {
                let index = behavior.primitive().sequence_index(name);
                if index < 0 {
                    Err(anyhow!("Smart Primitive has no sequence named {name:?}."))
                } else {
                    behavior
                        .primitive_mut()
                        .align_sequence_pivot(index as usize, &properties)
                        .and_then(|()| behavior.primitive_mut().get_world_target_poses_for_sequence(index as usize))
                }
            }
        };
        self.primitive = Some(behavior);
        let (poses, params) = result?;
        // A behavior that does not override get_targets() returns no targets.
        if params.is_empty() {
            bail!("The Smart Primitive behavior returned no targets.");
        }

        self.stop_target_pose_sequence(character);
        self.playing_clip = false;
        self.inference_params = params;
        self.target_poses = poses;
        self.target_index = 0;
        self.on_complete = on_complete;
        self.generate_current_target(character)
    }

    /// Play a clip once, or keep repeating it when `loop_` is set.
    fn play_clip(&mut self, character: &mut Character<A>, path: &Path, loop_: bool, align_to_current_pose: bool, on_complete: Option<Completion<ElemOf<A>>>) -> Result<()> {
        let key = (resolve_clip_path(path), loop_);
        let Some(clip) = self.clips.get(&key) else {
            bail!("Cannot play {}. Declare it in the behavior's clip_paths first.", path.display());
        };
        let clip = clip.clone();
        self.stop_target_pose_sequence(character);
        character.start_clip(clip, loop_, align_to_current_pose)?;
        self.playing_clip = true;
        self.clip_loop = loop_;
        self.on_complete = on_complete;
        Ok(())
    }

    fn update(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, delta_time: f32, deferred: &mut Option<ActiveState>) -> Result<()> {
        if self.playing_clip {
            if character.clip_ends_within(delta_time) {
                self.complete_playback(character, machine, deferred)?;
            }
            return Ok(());
        }

        if self.target_index < 0 || character.time_to_transition() > delta_time as f64 {
            return Ok(());
        }

        self.target_index += 1;
        if (self.target_index as usize) < self.inference_params.len() {
            self.generate_current_target(character)?;
        } else {
            self.complete_playback(character, machine, deferred)?;
        }
        Ok(())
    }

    fn complete_playback(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, deferred: &mut Option<ActiveState>) -> Result<()> {
        let on_complete = self.on_complete.take();
        // Clear the finished playback first, because the callback may start a new
        // sequence or clip.
        self.playing_clip = false;
        self.stop_target_playback_state();
        character.clear_ik_pins();
        let mut behavior = self.primitive.take().ok_or_else(|| anyhow!("Cannot complete playback without an active interaction."))?;
        let mut queue = InteractionQueue { is_looping: false, requests: Vec::new() };
        let result = match on_complete {
            Some(callback) => callback(&mut *behavior, &mut queue),
            None => {
                queue.finish();
                Ok(())
            }
        };
        self.primitive = Some(behavior);
        result?;
        self.apply_requests(character, machine, queue.requests, deferred)
    }

    fn stop_target_playback_state(&mut self) {
        self.target_index = -1;
        self.on_complete = None;
    }

    fn stop_target_pose_sequence(&mut self, character: &mut Character<A>) {
        self.stop_target_playback_state();
        character.clear_ik_pins();
    }

    fn generate_current_target(&self, character: &mut Character<A>) -> Result<()> {
        let frame_offset = self.target_index as usize * POSES_PER_TOKEN;
        let params = self.inference_params[self.target_index as usize].clone();
        character.generate_to_target_poses(&self.target_poses[frame_offset..frame_offset + POSES_PER_TOKEN], &params, PlaybackSource::Generated)
    }

    fn apply_requests(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, requests: Vec<InteractionRequest<ElemOf<A>>>, deferred: &mut Option<ActiveState>) -> Result<()> {
        for request in requests {
            match request {
                InteractionRequest::PlaySequence { name, on_complete } => self.play_target_pose_sequence(character, name, on_complete)?,
                InteractionRequest::PlayClip { path, loop_, align, on_complete } => self.play_clip(character, &path, loop_, align, on_complete)?,
                InteractionRequest::Finish => *deferred = Some(machine.resume_key(character)),
            }
        }
        Ok(())
    }
}

/// One of the state machine's states: the idle and locomotion states of the standing
/// and crouching movement pairs, or the shared interaction state.
pub enum State<A: EngineAdapter> {
    Idle(IdleState<A::Buffer>),
    Locomotion(LocomotionState<ElemOf<A>>),
    Interaction(InteractionState<A>),
}

impl<A: EngineAdapter> State<A>
where
    ElemOf<A>: PoseElem,
{
    /// The placeholder a state machine leaves behind while this state is moved out.
    pub(crate) fn placeholder() -> Self {
        State::Idle(IdleState::default())
    }

    pub fn name(&self) -> &'static str {
        match self {
            State::Idle(state) => state.name(),
            State::Locomotion(state) => state.name(),
            State::Interaction(_) => "Interaction",
        }
    }

    /// Start the character in the initial idle loop. Only idle states act.
    pub(crate) fn start(&mut self, character: &mut Character<A>) -> Result<()> {
        match self {
            State::Idle(state) => state.start(character),
            _ => Ok(()),
        }
    }

    /// Run when the state machine switches to this state.
    pub(crate) fn enter(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, deferred: &mut Option<ActiveState>) -> Result<()> {
        match self {
            State::Idle(state) => state.enter(character),
            State::Locomotion(state) => state.enter(character),
            State::Interaction(state) => state.enter(character, machine, deferred),
        }
    }

    /// Run when the state machine leaves this state.
    pub(crate) fn exit(&mut self, character: &mut Character<A>) {
        if let State::Interaction(state) = self {
            state.exit(character);
        }
    }

    /// Advance the state by `delta_time` (in seconds).
    pub(crate) fn update(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, delta_time: f32, deferred: &mut Option<ActiveState>) -> Result<()> {
        match self {
            State::Idle(state) => state.update(character, machine, delta_time, deferred),
            State::Locomotion(state) => state.update(character, machine, delta_time, deferred),
            State::Interaction(state) => state.update(character, machine, delta_time, deferred),
        }
    }

    /// Pass an interaction press to the active Smart Primitive behavior.
    pub(crate) fn interact(&mut self, character: &mut Character<A>, machine: &mut StateMachine<A>, deferred: &mut Option<ActiveState>) -> Result<()> {
        match self {
            State::Interaction(state) => state.interact(character, machine, deferred),
            _ => Ok(()),
        }
    }
}
