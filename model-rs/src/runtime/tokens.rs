// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (model/soma/buffers.py,
// model/unitree_g1/buffers.py, model/generation.py, runtime/motion_recorder.py, utils/time_utils.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Context/target tokens and the recorder of shown poses that feeds them.

use anyhow::{Result, bail};

use super::animation::{read_root, write_root};
use super::contract::{MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ROOT_COORDINATE_COUNT};
use super::math::{Transform, blend_transform};

/// Samples at a fixed rate ending at the latest recorded pose: (lower frame, upper frame, blend)
/// per sample. Samples before the available history clamp to its oldest pose.
pub fn history_sample_indices(delta_times: &[f32], num_samples: usize, sample_rate: f32) -> Vec<(usize, usize, f32)> {
    let n = delta_times.len();
    let mut pose_times = vec![0f32; n];
    for frame in (0..n.saturating_sub(1)).rev() {
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
/// `P` is the pose layout: `Vec<Transform>` (SOMA local joint transforms) or `Vec<f32>` (G1 joint
/// coordinates). `delta_times` ends with zero. Unrecorded slots hold the last `fill()` pose.
#[derive(Clone, Debug)]
pub struct MotionRecorder<P: Clone + PoseLen> {
    pub capacity: usize,
    pub poses: Vec<P>,
    pub delta_times: Vec<f32>,
}

/// Number of values in a pose, for layout checks.
pub trait PoseLen {
    fn pose_len(&self) -> usize;
}
impl PoseLen for Vec<Transform> {
    fn pose_len(&self) -> usize {
        self.len()
    }
}
impl PoseLen for Vec<f32> {
    fn pose_len(&self) -> usize {
        self.len()
    }
}

impl<P: Clone + PoseLen> MotionRecorder<P> {
    pub fn new(reference_pose: P, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            bail!("capacity must be positive");
        }
        Ok(MotionRecorder { capacity, poses: vec![reference_pose; capacity], delta_times: vec![0.0; capacity] })
    }
    /// Every slot holds one pose with zero durations, as if the character had held it.
    pub fn fill(&mut self, pose: &P) -> Result<()> {
        self.check(pose)?;
        self.poses.iter_mut().for_each(|p| *p = pose.clone());
        self.delta_times.iter_mut().for_each(|d| *d = 0.0);
        Ok(())
    }
    /// Store a pose sampled `delta_time` (finite, > 0) seconds after the previous one.
    pub fn record(&mut self, delta_time: f32, pose: &P) -> Result<()> {
        self.check(pose)?;
        if !delta_time.is_finite() || delta_time <= 0.0 {
            bail!("delta_time must be finite and positive");
        }
        let last = self.capacity - 1;
        self.delta_times[last] = delta_time;
        self.delta_times.rotate_left(1);
        self.delta_times[last] = 0.0;
        self.poses.rotate_left(1);
        self.poses[last] = pose.clone();
        Ok(())
    }
    fn check(&self, pose: &P) -> Result<()> {
        if pose.pose_len() != self.poses[0].pose_len() {
            bail!("pose must match the reference pose's layout");
        }
        Ok(())
    }
}

fn pad<P: Clone>(poses: &[P], slots: &mut [P]) -> Result<usize> {
    if poses.is_empty() || poses.len() > MODEL_POSES_PER_TOKEN {
        bail!("a context or target token requires 1 to {MODEL_POSES_PER_TOKEN} poses");
    }
    for (row, slot) in slots.iter_mut().enumerate() {
        *slot = poses[row.min(poses.len() - 1)].clone();
    }
    Ok(poses.len())
}

/// Up to four SOMA poses (local joint transforms) for context or a target.
#[derive(Clone, Debug)]
pub struct PoseToken {
    pub num_joints: usize,
    pub poses: Vec<Vec<Transform>>,
    num_poses: usize,
}

impl PoseToken {
    pub fn new(num_joints: usize) -> Self {
        PoseToken { num_joints, poses: vec![vec![Transform::IDENTITY; num_joints]; MODEL_POSES_PER_TOKEN], num_poses: 0 }
    }
    /// Supplied poses, 1 to 4, or 0 before the first update.
    pub fn num_poses(&self) -> usize {
        self.num_poses
    }
    /// One to four poses at 30 Hz; the last pads the unused rows.
    pub fn update_from_poses(&mut self, poses: &[Vec<Transform>]) -> Result<()> {
        if poses.iter().any(|p| p.len() != self.num_joints) {
            bail!("each token pose must hold {} joints", self.num_joints);
        }
        self.num_poses = pad(poses, &mut self.poses)?;
        Ok(())
    }
    /// Four poses at 30 Hz ending at the latest recorded pose (neighbors blended).
    pub fn update_from_recorder(&mut self, history: &MotionRecorder<Vec<Transform>>) -> Result<()> {
        if history.poses[0].len() != self.num_joints {
            bail!("history and token joint counts do not match");
        }
        for (k, (lo, up, b)) in history_sample_indices(&history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE).into_iter().enumerate() {
            self.poses[k] = history.poses[lo].iter().zip(&history.poses[up]).map(|(x, y)| blend_transform(*x, *y, b)).collect();
        }
        self.num_poses = MODEL_POSES_PER_TOKEN;
        Ok(())
    }
}

/// Up to four G1 poses (joint coordinates `[7 + num_dofs]`) for context or a target.
#[derive(Clone, Debug)]
pub struct DofToken {
    pub num_dofs: usize,
    pub poses: Vec<Vec<f32>>,
    num_poses: usize,
}

impl DofToken {
    pub fn new(num_dofs: usize) -> Self {
        let mut identity = vec![0.0; ROOT_COORDINATE_COUNT + num_dofs];
        identity[6] = 1.0;
        DofToken { num_dofs, poses: vec![identity; MODEL_POSES_PER_TOKEN], num_poses: 0 }
    }
    pub fn num_coordinates(&self) -> usize {
        ROOT_COORDINATE_COUNT + self.num_dofs
    }
    pub fn num_poses(&self) -> usize {
        self.num_poses
    }
    pub fn update_from_poses(&mut self, poses: &[Vec<f32>]) -> Result<()> {
        if poses.iter().any(|p| p.len() != self.num_coordinates()) {
            bail!("each token pose must hold {} coordinates", self.num_coordinates());
        }
        self.num_poses = pad(poses, &mut self.poses)?;
        Ok(())
    }
    pub fn update_from_recorder(&mut self, history: &MotionRecorder<Vec<f32>>) -> Result<()> {
        if history.poses[0].len() != self.num_coordinates() {
            bail!("history and token coordinate counts do not match");
        }
        for (k, (lo, up, b)) in history_sample_indices(&history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE).into_iter().enumerate() {
            let (a, c) = (&history.poses[lo], &history.poses[up]);
            let mut out: Vec<f32> = a.iter().zip(c).map(|(x, y)| x + (y - x) * b).collect();
            write_root(blend_transform(read_root(a), read_root(c), b), &mut out);
            self.poses[k] = out;
        }
        self.num_poses = MODEL_POSES_PER_TOKEN;
        Ok(())
    }
}
