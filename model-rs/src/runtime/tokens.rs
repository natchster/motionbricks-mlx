// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (model/soma/buffers.py,
// model/unitree_g1/buffers.py, model/generation.py, runtime/motion_recorder.py, utils/time_utils.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Context/target tokens and the recorder of shown poses that feeds them.

use anyhow::{Result, bail};

use super::animation::{blend_joint_coordinates, identity_joint_coordinates, write_root};
use super::contract::{MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ROOT_COORDINATE_COUNT};
use super::math::{Transform, blend_transform};

/// Samples at a fixed rate ending at the latest recorded pose: (lower frame, upper frame, blend)
/// per sample. Samples before the available history clamp to its oldest pose. An empty history
/// gives no samples.
pub fn history_sample_indices(delta_times: &[f32], num_samples: usize, sample_rate: f32) -> Vec<(usize, usize, f32)> {
    let n = delta_times.len();
    if n == 0 || num_samples == 0 {
        return Vec::new();
    }
    let mut pose_times = vec![0f32; n];
    for frame in (0..n - 1).rev() {
        pose_times[frame] = pose_times[frame + 1] - delta_times[frame];
    }
    let first = -((num_samples - 1) as f64) / sample_rate as f64;
    (0..num_samples)
        .map(|s| {
            let t = first + s as f64 / sample_rate as f64;
            let upper = pose_times.iter().take_while(|&&p| p as f64 <= t).count().min(n - 1);
            let mut lower = upper.saturating_sub(1);
            let frame_time = (pose_times[upper] - pose_times[lower]) as f64;
            let blend = if frame_time > 0.0 {
                (t - pose_times[lower] as f64) / frame_time
            } else {
                lower = upper;
                0.0
            };
            (lower, upper, blend.clamp(0.0, 1.0) as f32)
        })
        .collect()
}

/// A fixed number of recent poses (oldest first) and the time from each to the next.
///
/// `T` is one value of a pose: [`Transform`] (SOMA local joint transforms) or `f32` (G1 joint
/// coordinates). `delta_times()` ends with zero. Unrecorded slots hold the last `fill()` pose.
#[derive(Clone, Debug)]
pub struct MotionRecorder<T> {
    poses: Vec<Vec<T>>,
    delta_times: Vec<f32>,
}

impl<T: Copy> MotionRecorder<T> {
    pub fn new(reference_pose: &[T], capacity: usize) -> Result<Self> {
        if capacity == 0 {
            bail!("capacity must be positive");
        }
        Ok(MotionRecorder { poses: vec![reference_pose.to_vec(); capacity], delta_times: vec![0.0; capacity] })
    }
    pub fn capacity(&self) -> usize {
        self.poses.len()
    }
    /// Values per pose (the reference pose's length).
    pub fn pose_len(&self) -> usize {
        self.poses[0].len()
    }
    /// The recorded poses, oldest first.
    pub fn poses(&self) -> &[Vec<T>] {
        &self.poses
    }
    /// Seconds from each pose to the next; the last is zero.
    pub fn delta_times(&self) -> &[f32] {
        &self.delta_times
    }
    /// Every slot holds one pose with zero durations, as if the character had held it.
    pub fn fill(&mut self, pose: &[T]) -> Result<()> {
        self.check(pose)?;
        self.poses.iter_mut().for_each(|p| p.copy_from_slice(pose));
        self.delta_times.iter_mut().for_each(|d| *d = 0.0);
        Ok(())
    }
    /// Store a pose sampled `delta_time` (finite, > 0) seconds after the previous one. The oldest
    /// slot is reused, so recording does not allocate.
    pub fn record(&mut self, delta_time: f32, pose: &[T]) -> Result<()> {
        self.check(pose)?;
        if !delta_time.is_finite() || delta_time <= 0.0 {
            bail!("delta_time must be finite and positive");
        }
        let last = self.capacity() - 1;
        self.delta_times[last] = delta_time;
        self.delta_times.rotate_left(1);
        self.delta_times[last] = 0.0;
        self.poses.rotate_left(1);
        self.poses[last].copy_from_slice(pose);
        Ok(())
    }
    fn check(&self, pose: &[T]) -> Result<()> {
        if pose.len() != self.pose_len() {
            bail!("pose must match the reference pose's layout ({} values, got {})", self.pose_len(), pose.len());
        }
        Ok(())
    }
}

/// Copy one to four poses into a token's slots, padding the unused rows with the last pose.
fn pad<T: Copy, P: AsRef<[T]>>(poses: &[P], slots: &mut [Vec<T>], what: &str) -> Result<usize> {
    if poses.is_empty() || poses.len() > MODEL_POSES_PER_TOKEN {
        bail!("a context or target token requires 1 to {MODEL_POSES_PER_TOKEN} poses, got {}", poses.len());
    }
    let len = slots[0].len();
    if poses.iter().any(|p| p.as_ref().len() != len) {
        bail!("each token pose must hold {len} {what}");
    }
    for (row, slot) in slots.iter_mut().enumerate() {
        slot.copy_from_slice(poses[row.min(poses.len() - 1)].as_ref());
    }
    Ok(poses.len())
}

/// What the motion players need from both token types ([`PoseToken`], [`DofToken`]).
pub trait MotionToken {
    /// One value of a pose: a joint [`Transform`] (SOMA) or a joint coordinate (G1).
    type Elem: Copy;
    fn num_poses(&self) -> usize;
    fn update_from_poses<P: AsRef<[Self::Elem]>>(&mut self, poses: &[P]) -> Result<()>;
    fn update_from_recorder(&mut self, history: &MotionRecorder<Self::Elem>) -> Result<()>;
}

/// Up to four SOMA poses (local joint transforms) for context or a target.
#[derive(Clone, Debug)]
pub struct PoseToken {
    num_joints: usize,
    poses: Vec<Vec<Transform>>,
    num_poses: usize,
}

impl PoseToken {
    pub fn new(num_joints: usize) -> Self {
        PoseToken { num_joints, poses: vec![vec![Transform::IDENTITY; num_joints]; MODEL_POSES_PER_TOKEN], num_poses: 0 }
    }
    pub fn num_joints(&self) -> usize {
        self.num_joints
    }
    /// The four pose slots (unused ones repeat the last supplied pose).
    pub fn poses(&self) -> &[Vec<Transform>] {
        &self.poses
    }
    /// Supplied poses, 1 to 4, or 0 before the first update.
    pub fn num_poses(&self) -> usize {
        self.num_poses
    }
    /// One to four poses at 30 Hz; the last pads the unused rows.
    pub fn update_from_poses<P: AsRef<[Transform]>>(&mut self, poses: &[P]) -> Result<()> {
        self.num_poses = pad(poses, &mut self.poses, "joints")?;
        Ok(())
    }
    /// Four poses at 30 Hz ending at the latest recorded pose (neighbors blended).
    pub fn update_from_recorder(&mut self, history: &MotionRecorder<Transform>) -> Result<()> {
        if history.pose_len() != self.num_joints {
            bail!("history and token joint counts do not match");
        }
        for (k, (lo, up, b)) in history_sample_indices(&history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE).into_iter().enumerate() {
            let (a, c) = (&history.poses[lo], &history.poses[up]);
            for ((o, x), y) in self.poses[k].iter_mut().zip(a).zip(c) {
                *o = blend_transform(*x, *y, b);
            }
        }
        self.num_poses = MODEL_POSES_PER_TOKEN;
        Ok(())
    }
}

impl MotionToken for PoseToken {
    type Elem = Transform;
    fn num_poses(&self) -> usize {
        self.num_poses
    }
    fn update_from_poses<P: AsRef<[Transform]>>(&mut self, poses: &[P]) -> Result<()> {
        PoseToken::update_from_poses(self, poses)
    }
    fn update_from_recorder(&mut self, history: &MotionRecorder<Transform>) -> Result<()> {
        PoseToken::update_from_recorder(self, history)
    }
}

/// Up to four G1 poses (joint coordinates `[7 + num_dofs]`) for context or a target.
#[derive(Clone, Debug)]
pub struct DofToken {
    num_dofs: usize,
    poses: Vec<Vec<f32>>,
    num_poses: usize,
}

impl DofToken {
    pub fn new(num_dofs: usize) -> Self {
        DofToken { num_dofs, poses: vec![identity_joint_coordinates(num_dofs); MODEL_POSES_PER_TOKEN], num_poses: 0 }
    }
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }
    pub fn num_coordinates(&self) -> usize {
        ROOT_COORDINATE_COUNT + self.num_dofs
    }
    /// The four pose slots (unused ones repeat the last supplied pose).
    pub fn poses(&self) -> &[Vec<f32>] {
        &self.poses
    }
    pub fn num_poses(&self) -> usize {
        self.num_poses
    }
    pub fn update_from_poses<P: AsRef<[f32]>>(&mut self, poses: &[P]) -> Result<()> {
        self.num_poses = pad(poses, &mut self.poses, "coordinates")?;
        Ok(())
    }
    pub fn update_from_recorder(&mut self, history: &MotionRecorder<f32>) -> Result<()> {
        if history.pose_len() != self.num_coordinates() {
            bail!("history and token coordinate counts do not match");
        }
        for (k, (lo, up, b)) in history_sample_indices(&history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE).into_iter().enumerate() {
            let out = &mut self.poses[k];
            let root = blend_joint_coordinates(&history.poses[lo], &history.poses[up], b, out);
            write_root(root, out);
        }
        self.num_poses = MODEL_POSES_PER_TOKEN;
        Ok(())
    }
}

impl MotionToken for DofToken {
    type Elem = f32;
    fn num_poses(&self) -> usize {
        self.num_poses
    }
    fn update_from_poses<P: AsRef<[f32]>>(&mut self, poses: &[P]) -> Result<()> {
        DofToken::update_from_poses(self, poses)
    }
    fn update_from_recorder(&mut self, history: &MotionRecorder<f32>) -> Result<()> {
        DofToken::update_from_recorder(self, history)
    }
}
