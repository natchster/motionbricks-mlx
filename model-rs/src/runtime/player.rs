// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (runtime/motion_player.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Motion players: generate from recorded poses, play the result, record what was shown.

use std::rc::Rc;

use anyhow::{Result, bail};

use super::animation::{AnimationBuffer, DofAnimationBuffer, FootContacts, Skeleton, frame_index_from_time};
use super::contract::MotionBricksInferenceConfig;
use super::engine::{Backend, MotionBricksSomaEngine, MotionBricksUnitreeG1Engine};
use super::math::{Transform, Vec3};
use super::target_pose::TargetPoseInferenceParams;
use super::tokens::{DofToken, MotionRecorder, PoseToken};

/// Playback timing shared by both players.
#[derive(Clone, Debug, Default)]
struct Playback {
    playback_time: f32,
    transition_early_by_frames: usize,
    root_positions: Vec<Vec3>,
}

impl Playback {
    fn duration(&self, num_frames: usize, sample_rate: f32) -> f32 {
        num_frames.saturating_sub(1 + self.transition_early_by_frames) as f32 / sample_rate
    }
    fn advance(&mut self, delta_time: f32) -> Result<()> {
        if !delta_time.is_finite() || delta_time < 0.0 {
            bail!("delta_time must be finite and non-negative");
        }
        self.playback_time += delta_time;
        Ok(())
    }
    fn root_position(&self, sample_rate: f32) -> Vec3 {
        if self.root_positions.is_empty() {
            return Vec3::ZERO;
        }
        let (f, n, b) = frame_index_from_time(self.playback_time, sample_rate, self.root_positions.len());
        self.root_positions[f].lerp(self.root_positions[n], b)
    }
}

fn config_for(params: &TargetPoseInferenceParams, config: Option<&MotionBricksInferenceConfig>) -> MotionBricksInferenceConfig {
    let mut c = config.cloned().unwrap_or_default();
    if let Some(counts) = &params.allowed_token_counts {
        c.allowed_token_counts = counts.clone();
    }
    c
}

/// Play SOMA motion generated from recorded poses and targets.
pub struct GeneratedMotionPlayer<B: Backend> {
    pub engine: MotionBricksSomaEngine<B>,
    pub context: PoseToken,
    pub target: PoseToken,
    pub output: AnimationBuffer,
    pub motion_history: MotionRecorder<Vec<Transform>>,
    playback: Playback,
}

impl<B: Backend> GeneratedMotionPlayer<B> {
    pub fn new(mut engine: MotionBricksSomaEngine<B>, skeleton: &Rc<Skeleton>, history_capacity: usize) -> Result<Self> {
        let (context, target, output) = (engine.create_input(skeleton)?, engine.create_input(skeleton)?, engine.create_output(skeleton)?);
        let motion_history = MotionRecorder::new(skeleton.reference_local_transforms.clone(), history_capacity)?;
        Ok(GeneratedMotionPlayer { engine, context, target, output, motion_history, playback: Playback::default() })
    }
    pub fn playback_time(&self) -> f32 {
        self.playback.playback_time
    }
    /// Playback time of the transition point.
    pub fn playback_duration(&self) -> f32 {
        self.playback.duration(self.output.num_frames, self.output.sample_rate)
    }
    /// Seconds until the transition point. Sampling can continue past it.
    pub fn time_to_transition(&self) -> f32 {
        (self.playback_duration() - self.playback.playback_time).max(0.0)
    }
    /// Generate toward a target from the recorded history, then restart playback.
    pub fn generate(&mut self, target_poses: &[Vec<Transform>], params: &TargetPoseInferenceParams, config: Option<&MotionBricksInferenceConfig>) -> Result<()> {
        let config = config_for(params, config);
        self.target.update_from_poses(&target_poses[..params.num_poses_to_use.min(target_poses.len())])?;
        self.playback.transition_early_by_frames = params.transition_early_by_frames;
        self.refresh_context()?;
        self.engine.generate_animation(&self.context, &self.target, &mut self.output, &config)?;
        self.playback.playback_time = 0.0;
        self.playback.root_positions = self.output.root_transforms().iter().map(|t| t.p).collect();
        Ok(())
    }
    /// Advance playback, then return the pose and its foot contacts.
    pub fn sample(&mut self, delta_time: f32) -> Result<(Vec<Transform>, FootContacts)> {
        self.playback.advance(delta_time)?;
        let t = self.playback.playback_time;
        Ok((self.output.sample(t, Transform::IDENTITY)?, self.output.sample_foot_contacts(t)?))
    }
    /// Record the pose that was shown, for the next generation's context.
    pub fn record_pose(&mut self, delta_time: f32, pose: &Vec<Transform>) -> Result<()> {
        self.motion_history.record(delta_time, pose)
    }
    /// Fill every history slot and the context with one pose.
    pub fn seed_history_from_pose(&mut self, pose: &Vec<Transform>) -> Result<()> {
        self.motion_history.fill(pose)?;
        self.refresh_context()
    }
    pub fn refresh_context(&mut self) -> Result<()> {
        self.context.update_from_recorder(&self.motion_history)
    }
    pub fn current_root_position(&self) -> Vec3 {
        self.playback.root_position(self.output.sample_rate)
    }
}

/// Play G1 motion generated from recorded poses and targets.
pub struct GeneratedDofMotionPlayer<B: Backend> {
    pub engine: MotionBricksUnitreeG1Engine<B>,
    pub context: DofToken,
    pub target: DofToken,
    pub output: DofAnimationBuffer,
    pub motion_history: MotionRecorder<Vec<f32>>,
    playback: Playback,
}

impl<B: Backend> GeneratedDofMotionPlayer<B> {
    pub fn new(engine: MotionBricksUnitreeG1Engine<B>, reference_joint_q: Vec<f32>, history_capacity: usize) -> Result<Self> {
        let (context, target, output) = (engine.create_input(), engine.create_input(), engine.create_output());
        let motion_history = MotionRecorder::new(reference_joint_q, history_capacity)?;
        Ok(GeneratedDofMotionPlayer { engine, context, target, output, motion_history, playback: Playback::default() })
    }
    pub fn playback_time(&self) -> f32 {
        self.playback.playback_time
    }
    pub fn playback_duration(&self) -> f32 {
        self.playback.duration(self.output.num_frames, self.output.sample_rate)
    }
    pub fn time_to_transition(&self) -> f32 {
        (self.playback_duration() - self.playback.playback_time).max(0.0)
    }
    pub fn generate(&mut self, target_poses: &[Vec<f32>], params: &TargetPoseInferenceParams, config: Option<&MotionBricksInferenceConfig>) -> Result<()> {
        let config = config_for(params, config);
        self.target.update_from_poses(&target_poses[..params.num_poses_to_use.min(target_poses.len())])?;
        self.playback.transition_early_by_frames = params.transition_early_by_frames;
        self.refresh_context()?;
        self.engine.generate_animation(&self.context, &self.target, &mut self.output, &config)?;
        self.playback.playback_time = 0.0;
        self.playback.root_positions = self.output.root_transforms().iter().map(|t| t.p).collect();
        Ok(())
    }
    pub fn sample(&mut self, delta_time: f32) -> Result<(Vec<f32>, FootContacts)> {
        self.playback.advance(delta_time)?;
        let t = self.playback.playback_time;
        Ok((self.output.sample(t, Transform::IDENTITY)?, self.output.sample_foot_contacts(t)?))
    }
    pub fn record_pose(&mut self, delta_time: f32, pose: &Vec<f32>) -> Result<()> {
        self.motion_history.record(delta_time, pose)
    }
    pub fn seed_history_from_pose(&mut self, pose: &Vec<f32>) -> Result<()> {
        self.motion_history.fill(pose)?;
        self.refresh_context()
    }
    pub fn refresh_context(&mut self) -> Result<()> {
        self.context.update_from_recorder(&self.motion_history)
    }
    pub fn current_root_position(&self) -> Vec3 {
        self.playback.root_position(self.output.sample_rate)
    }
}
