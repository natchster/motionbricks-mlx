// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (runtime/motion_player.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Motion players: generate from recorded poses, play the result, record what was shown.

use std::sync::Arc;

use anyhow::{Result, bail};

use super::animation::{FootContacts, MotionBuffer, Skeleton, frame_index_at};
use super::contract::MotionBricksInferenceConfig;
use super::engine::{Backend, MotionBricksEngine, MotionBricksSomaEngine, MotionBricksUnitreeG1Engine};
use super::math::{Transform, Vec3};
use super::target_pose::TargetPoseInferenceParams;
use super::tokens::{MotionRecorder, MotionToken};

fn config_for(params: &TargetPoseInferenceParams, config: Option<&MotionBricksInferenceConfig>) -> MotionBricksInferenceConfig {
    let mut c = config.cloned().unwrap_or_default();
    if let Some(counts) = &params.allowed_token_counts {
        c.allowed_token_counts = counts.clone();
    }
    c
}

/// Play motion generated from recorded poses and targets, for either rig: see the aliases
/// [`GeneratedMotionPlayer`] (SOMA) and [`GeneratedDofMotionPlayer`] (G1).
///
/// The player owns its engine. To drive several characters with one model, give each player an
/// engine over a shared backend (`Rc<RefCell<B>>` implements [`Backend`]).
pub struct GeneratedPlayer<E: MotionBricksEngine> {
    engine: E,
    context: E::Token,
    target: E::Token,
    output: E::Output,
    // Generations run into this buffer, swapped in on success, so a failed generation keeps the
    // motion that is playing (as the Python port does).
    next_output: E::Output,
    motion_history: MotionRecorder<E::Elem>,
    playback_time: f64,
    transition_early_by_frames: usize,
    root_positions: Vec<Vec3>,
}

/// Play SOMA motion generated from recorded poses and targets.
pub type GeneratedMotionPlayer<B> = GeneratedPlayer<MotionBricksSomaEngine<B>>;
/// Play G1 motion generated from recorded poses and targets.
pub type GeneratedDofMotionPlayer<B> = GeneratedPlayer<MotionBricksUnitreeG1Engine<B>>;

impl<B: Backend> GeneratedPlayer<MotionBricksSomaEngine<B>> {
    pub fn new(mut engine: MotionBricksSomaEngine<B>, skeleton: &Arc<Skeleton>, history_capacity: usize) -> Result<Self> {
        let (context, target) = (engine.create_input(skeleton)?, engine.create_input(skeleton)?);
        let (output, next_output) = (engine.create_output(skeleton)?, engine.create_output(skeleton)?);
        let motion_history = MotionRecorder::new(skeleton.reference_local_transforms(), history_capacity)?;
        Ok(Self::assemble(engine, context, target, output, next_output, motion_history))
    }
}

impl<B: Backend> GeneratedPlayer<MotionBricksUnitreeG1Engine<B>> {
    /// `reference_joint_q`: the 36 joint coordinates that fill the history until `seed_history_from_pose`.
    pub fn new(engine: MotionBricksUnitreeG1Engine<B>, reference_joint_q: &[f32], history_capacity: usize) -> Result<Self> {
        let (context, target) = (engine.create_input(), engine.create_input());
        let (output, next_output) = (engine.create_output(), engine.create_output());
        let motion_history = MotionRecorder::new(reference_joint_q, history_capacity)?;
        Ok(Self::assemble(engine, context, target, output, next_output, motion_history))
    }
}

impl<E: MotionBricksEngine> GeneratedPlayer<E> {
    fn assemble(engine: E, context: E::Token, target: E::Token, output: E::Output, next_output: E::Output, motion_history: MotionRecorder<E::Elem>) -> Self {
        GeneratedPlayer { engine, context, target, output, next_output, motion_history, playback_time: 0.0, transition_early_by_frames: 0, root_positions: Vec::new() }
    }
    pub fn engine(&self) -> &E {
        &self.engine
    }
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }
    /// The context of the last generation (or of the last `refresh_context`).
    pub fn context(&self) -> &E::Token {
        &self.context
    }
    /// The target of the last generation.
    pub fn target(&self) -> &E::Token {
        &self.target
    }
    /// The motion that is playing.
    pub fn output(&self) -> &E::Output {
        &self.output
    }
    pub fn motion_history(&self) -> &MotionRecorder<E::Elem> {
        &self.motion_history
    }
    /// Seconds since the motion that is playing started.
    pub fn playback_time(&self) -> f64 {
        self.playback_time
    }
    /// Frames before the end of the playing motion where its transition point sits.
    pub fn transition_early_by_frames(&self) -> usize {
        self.transition_early_by_frames
    }
    /// Playback time of the transition point.
    pub fn playback_duration(&self) -> f64 {
        self.output.num_frames().saturating_sub(1 + self.transition_early_by_frames) as f64 / self.output.sample_rate() as f64
    }
    /// Seconds until the transition point. Sampling can continue past it.
    pub fn time_to_transition(&self) -> f64 {
        (self.playback_duration() - self.playback_time).max(0.0)
    }
    /// Generate toward a target from the recorded history, then restart playback. If the
    /// generation fails, the playing motion, its playback time and its transition point are kept.
    pub fn generate<P: AsRef<[E::Elem]>>(&mut self, target_poses: &[P], params: &TargetPoseInferenceParams, config: Option<&MotionBricksInferenceConfig>) -> Result<()> {
        let config = config_for(params, config);
        self.target.update_from_poses(&target_poses[..params.num_poses_to_use.min(target_poses.len())])?;
        self.refresh_context()?;
        self.engine.generate_animation(&self.context, &self.target, &mut self.next_output, &config)?;
        std::mem::swap(&mut self.output, &mut self.next_output);
        self.transition_early_by_frames = params.transition_early_by_frames;
        self.playback_time = 0.0;
        self.root_positions.clear();
        self.root_positions.extend((0..self.output.num_frames()).map(|f| self.output.root_position(f)));
        Ok(())
    }
    /// Advance playback, then return the pose and its foot contacts.
    pub fn sample(&mut self, delta_time: f32) -> Result<(Vec<E::Elem>, FootContacts)> {
        self.advance(delta_time)?;
        let t = self.playback_time;
        Ok((self.output.sample_at(t, Transform::IDENTITY)?, self.output.foot_contacts_at(t)?))
    }
    /// Advance playback, write the pose into `pose` (without allocating) and return its foot contacts.
    pub fn sample_into(&mut self, delta_time: f32, pose: &mut [E::Elem]) -> Result<FootContacts> {
        self.advance(delta_time)?;
        let t = self.playback_time;
        self.output.sample_into_at(t, Transform::IDENTITY, pose)?;
        self.output.foot_contacts_at(t)
    }
    /// Record the pose that was shown, for the next generation's context.
    pub fn record_pose(&mut self, delta_time: f32, pose: &[E::Elem]) -> Result<()> {
        self.motion_history.record(delta_time, pose)
    }
    /// Fill every history slot and the context with one pose.
    pub fn seed_history_from_pose(&mut self, pose: &[E::Elem]) -> Result<()> {
        self.motion_history.fill(pose)?;
        self.refresh_context()
    }
    pub fn refresh_context(&mut self) -> Result<()> {
        self.context.update_from_recorder(&self.motion_history)
    }
    /// The generated root position at the playback time.
    pub fn current_root_position(&self) -> Vec3 {
        if self.root_positions.is_empty() {
            return Vec3::ZERO;
        }
        let (f, n, b) = frame_index_at(self.playback_time, self.output.sample_rate(), self.root_positions.len());
        self.root_positions[f] * (1.0 - b) + self.root_positions[n] * b
    }
    fn advance(&mut self, delta_time: f32) -> Result<()> {
        if !delta_time.is_finite() || delta_time < 0.0 {
            bail!("delta_time must be finite and non-negative");
        }
        self.playback_time += delta_time as f64;
        Ok(())
    }
}
