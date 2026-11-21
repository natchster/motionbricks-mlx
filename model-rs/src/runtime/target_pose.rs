// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (runtime/target_pose.py, animation/ik.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// The two-bone IK solver derives from Daniel Holden's ik_two_bone (controller.cpp in
// https://github.com/orangeduck/Motion-Matching, MIT; see licenses/Daniel-Holden-MIT-LICENSE.txt).
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Per-target generation settings, target-pose IK pins, and locomotion target placement.

use anyhow::{Result, bail};

use super::animation::{Skeleton, read_root, write_root};
use super::contract::{MODEL_SAMPLE_RATE, ROOT_COORDINATE_COUNT};
use super::math::*;

pub const MAX_TRANSITION_EARLY_BY_FRAMES: usize = 20;

/// Hands and feet that a target pose can pin with IK (bit flags, combine with `|`).
pub mod ik_pin {
    pub const NONE: u32 = 0;
    pub const LEFT_HAND: u32 = 1 << 0;
    pub const RIGHT_HAND: u32 = 1 << 1;
    pub const LEFT_FOOT: u32 = 1 << 2;
    pub const RIGHT_FOOT: u32 = 1 << 3;
    pub const ALL: u32 = LEFT_HAND | RIGHT_HAND | LEFT_FOOT | RIGHT_FOOT;
}

/// Generation settings for reaching one target.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetPoseInferenceParams {
    /// Frames before the end of the generated motion where the transition point sits.
    pub transition_early_by_frames: usize,
    /// How many of the target's four poses inform the model, 1 to 4.
    pub num_poses_to_use: usize,
    /// `ik_pin` flags for the hands and feet that target pose IK pins (SOMA).
    pub ik_pin_mask: u32,
    /// Token counts that replace the generation config's; `None` keeps them.
    pub allowed_token_counts: Option<Vec<i32>>,
}

impl TargetPoseInferenceParams {
    pub fn new(transition_early_by_frames: usize, num_poses_to_use: usize) -> Self {
        TargetPoseInferenceParams { transition_early_by_frames, num_poses_to_use, ik_pin_mask: ik_pin::NONE, allowed_token_counts: None }
    }
}

/// Joint indices of a two-bone chain (start, middle, tip) and its bend hint in middle-joint space.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoBoneIkChainConfig {
    pub pin: u32,
    pub root_index: usize,
    pub mid_index: usize,
    pub tip_index: usize,
    pub hint_offset_local: Vec3,
}

impl TwoBoneIkChainConfig {
    pub fn from_joint_names(pin: u32, skeleton: &Skeleton, root: &str, mid: &str, tip: &str, hint: Vec3) -> Result<Self> {
        let find = |n: &str| skeleton.joint_index(n).ok_or_else(|| anyhow::anyhow!("IK chain joint {n:?} is missing from the skeleton"));
        if ![ik_pin::LEFT_HAND, ik_pin::RIGHT_HAND, ik_pin::LEFT_FOOT, ik_pin::RIGHT_FOOT].contains(&pin) {
            bail!("pin must identify exactly one supported IK pin");
        }
        Ok(TwoBoneIkChainConfig { pin, root_index: find(root)?, mid_index: find(mid)?, tip_index: find(tip)?, hint_offset_local: hint })
    }
}

/// The SOMA arm and leg chains with their bend hints.
pub fn create_soma_target_pose_ik_rig(skeleton: &Skeleton) -> Result<Vec<TwoBoneIkChainConfig>> {
    [
        (ik_pin::LEFT_HAND, "LeftArm", "LeftForeArm", "LeftHand", -0.25),
        (ik_pin::RIGHT_HAND, "RightArm", "RightForeArm", "RightHand", 0.25),
        (ik_pin::LEFT_FOOT, "LeftLeg", "LeftShin", "LeftFoot", -0.25),
        (ik_pin::RIGHT_FOOT, "RightLeg", "RightShin", "RightFoot", 0.25),
    ]
    .into_iter()
    .map(|(pin, r, m, t, y)| TwoBoneIkChainConfig::from_joint_names(pin, skeleton, r, m, t, Vec3::new(0.0, y, 0.0)))
    .collect()
}

const EPSILON: f32 = 1e-3;

/// Solved world transforms of the chain start, middle joint and tip, so the tip reaches `t_world`.
/// `weight` blends from the input pose (0) to the solved pose (1); with `use_hint` the chain bends
/// toward `hint_world`.
#[allow(clippy::too_many_arguments)]
pub fn solve_two_bone_ik(weight: f32, a_parent: Transform, a: Transform, b: Transform, c: Transform, t_world: Transform, use_hint: bool, hint_world: Vec3) -> (Transform, Transform, Transform) {
    let weight = weight.clamp(0.0, 1.0);
    let (pa, pb, pc) = (a.p, b.p, c.p);
    let t = pc.lerp(t_world.p, weight);
    let l_ab = (pb - pa).length();
    let l_cb = (pb - pc).length();
    let l_at = (t - pa).length().clamp(EPSILON, l_ab + l_cb - EPSILON);
    let axis = if use_hint { (pc - pa).cross(hint_world - pa) } else { (pc - pa).cross(pb - pa) }.normalize();
    let ac_ab_0 = axis.dot((pc - pa).cross(pb - pa)).atan2((pc - pa).dot(pb - pa));
    let ba_bc_0 = axis.dot((pa - pb).cross(pc - pb)).atan2((pa - pb).dot(pc - pb));
    let acos = |x: f32| x.clamp(-1.0, 1.0).acos();
    let ac_ab_1 = acos((l_ab * l_ab + l_at * l_at - l_cb * l_cb) / (2.0 * l_ab * l_at));
    let ba_bc_1 = acos((l_ab * l_ab + l_cb * l_cb - l_at * l_at) / (2.0 * l_ab * l_cb));
    let rot_a = quat_from_axis_angle(quat_rotate_inv(a.q, axis), ac_ab_1 - ac_ab_0);
    let rot_b = quat_from_axis_angle(quat_rotate_inv(b.q, axis), ba_bc_1 - ba_bc_0);
    let rotation = |q: Quat| Transform::new(Vec3::ZERO, q);
    let mut a_local = a_parent.inverse().mul(a).mul(rotation(rot_a));
    let a_world = a_parent.mul(a_local);
    let ac_local = quat_rotate_inv(a_world.q, pc - pa);
    let at_local = quat_rotate_inv(a_world.q, t - pa);
    a_local = a_local.mul(rotation(quat_between_vectors(ac_local, at_local)));
    let a_world = a_parent.mul(a_local);
    let b_local = a.inverse().mul(b).mul(rotation(rot_b));
    let b_world = a_world.mul(b_local);
    let c_world = b_world.mul(b.inverse().mul(c));
    // The conjugate is the inverse only for unit quaternions, so renormalize.
    (
        Transform::new(a_world.p, quat_normalize(a_world.q)),
        Transform::new(b_world.p, quat_normalize(b_world.q)),
        Transform::new(c_world.p, quat_normalize(quat_slerp(c_world.q, t_world.q, weight))),
    )
}

/// Ease pinned hands and feet onto a target pose with two-bone IK (SOMA).
pub struct TargetPoseIkCorrector {
    skeleton: Skeleton,
    chains: Vec<TwoBoneIkChainConfig>,
    pin_mask: u32,
    pinned: Vec<Transform>,
}

impl TargetPoseIkCorrector {
    pub fn new(skeleton: &Skeleton, chains: Vec<TwoBoneIkChainConfig>) -> Result<Self> {
        if chains.is_empty() {
            bail!("target pose IK chains cannot be empty");
        }
        for c in &chains {
            if skeleton.joint_parent(c.mid_index) != c.root_index as i32 || skeleton.joint_parent(c.tip_index) != c.mid_index as i32 {
                bail!("an IK chain's mid joint must be a child of its start, and its tip a child of its mid joint");
            }
        }
        let n = chains.len();
        Ok(TargetPoseIkCorrector { skeleton: skeleton.clone(), chains, pin_mask: ik_pin::NONE, pinned: vec![Transform::IDENTITY; n] })
    }
    pub fn pin_mask(&self) -> u32 {
        self.pin_mask
    }
    /// Pin the world transforms of the chains' tips in a target's local pose (`NONE` clears).
    pub fn capture_pins(&mut self, target_pose: &[Transform], pin_mask: u32) -> Result<()> {
        if pin_mask == ik_pin::NONE {
            self.clear_pins();
            return Ok(());
        }
        self.check(target_pose)?;
        let world = self.skeleton.compute_world_transforms(target_pose, Transform::IDENTITY);
        for (i, c) in self.chains.iter().enumerate() {
            self.pinned[i] = world[c.tip_index];
        }
        self.pin_mask = pin_mask;
        Ok(())
    }
    /// Correct the pinned chains of a sampled local pose in place (`weight` 0..1).
    pub fn apply(&self, pose: &mut [Transform], weight: f32) -> Result<()> {
        if self.pin_mask == ik_pin::NONE || weight <= 0.0 {
            return Ok(());
        }
        self.check(pose)?;
        let world = self.skeleton.compute_world_transforms(pose, Transform::IDENTITY);
        for (i, c) in self.chains.iter().enumerate() {
            if self.pin_mask & c.pin == 0 {
                continue;
            }
            let parent = self.skeleton.parent_indices[c.root_index];
            let parent_world = if parent >= 0 { world[parent as usize] } else { Transform::IDENTITY };
            let mid = world[c.mid_index];
            let hint = mid.point(c.hint_offset_local);
            let (ra, rb, rc) = solve_two_bone_ik(weight, parent_world, world[c.root_index], mid, world[c.tip_index], self.pinned[i], true, hint);
            pose[c.root_index] = parent_world.inverse().mul(ra);
            pose[c.mid_index] = ra.inverse().mul(rb);
            pose[c.tip_index] = rb.inverse().mul(rc);
        }
        Ok(())
    }
    pub fn clear_pins(&mut self) {
        self.pin_mask = ik_pin::NONE;
    }
    fn check(&self, pose: &[Transform]) -> Result<()> {
        if pose.len() != self.skeleton.num_joints() {
            bail!("a pose must contain one transform per skeleton joint");
        }
        Ok(())
    }
}

fn ground_speed_scale(start: Vec3, end: Vec3, pose_count: usize, desired_speed: f32) -> f32 {
    let v = (end - start) * (MODEL_SAMPLE_RATE / (pose_count - 1) as f32);
    let speed = (v.x() * v.x() + v.y() * v.y()).sqrt();
    if speed <= 1.0e-6 { 1.0 } else { desired_speed / speed }
}

fn check_speed(desired_speed: f32) -> Result<()> {
    if !desired_speed.is_finite() || desired_speed < 0.0 {
        bail!("desired_speed must be finite and non-negative");
    }
    Ok(())
}

/// Scale the root's ground motion of 30 Hz target poses (at least two) to a mean speed, in place.
pub fn adjust_target_pose_speed(target_poses: &mut [Vec<Transform>], desired_speed: f32) -> Result<()> {
    if target_poses.len() < 2 {
        bail!("target_poses must contain at least two poses");
    }
    check_speed(desired_speed)?;
    let start = target_poses[0][0].p;
    let scale = ground_speed_scale(start, target_poses[target_poses.len() - 1][0].p, target_poses.len(), desired_speed);
    for pose in target_poses.iter_mut().skip(1) {
        let p = pose[0].p;
        pose[0].p = Vec3::new(start.x() + (p.x() - start.x()) * scale, start.y() + (p.y() - start.y()) * scale, p.z());
    }
    Ok(())
}

/// The same for joint coordinates; height, rotation and angles are kept.
pub fn adjust_target_dof_speed(target_joint_q: &mut [Vec<f32>], desired_speed: f32) -> Result<()> {
    if target_joint_q.len() < 2 || target_joint_q.iter().any(|q| q.len() < ROOT_COORDINATE_COUNT) {
        bail!("target_joint_q must contain at least two poses with root XYZ and XYZW values");
    }
    check_speed(desired_speed)?;
    let start = read_root(&target_joint_q[0]).p;
    let end = read_root(&target_joint_q[target_joint_q.len() - 1]).p;
    let scale = ground_speed_scale(start, end, target_joint_q.len(), desired_speed);
    for q in target_joint_q.iter_mut().skip(1) {
        q[0] = start.x() + (q[0] - start.x()) * scale;
        q[1] = start.y() + (q[1] - start.y()) * scale;
    }
    Ok(())
}

fn ground_placement(ground_position: Vec3, facing_direction: Vec3) -> Transform {
    let yaw = facing_direction.y().atan2(facing_direction.x());
    Transform::new(ground_position, quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), yaw))
}

/// Move the roots of local poses to a ground position, turned to face a direction (Z up).
pub fn place_target_poses(ground_position: Vec3, facing_direction: Vec3, blended_poses: &[Vec<Transform>]) -> Vec<Vec<Transform>> {
    let placement = ground_placement(ground_position, facing_direction);
    blended_poses
        .iter()
        .map(|pose| {
            let mut out = pose.clone();
            out[0] = placement.mul(out[0]);
            out
        })
        .collect()
}

/// The same for joint coordinates; the joint angles are kept.
pub fn place_target_dof(ground_position: Vec3, facing_direction: Vec3, blended_joint_q: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let placement = ground_placement(ground_position, facing_direction);
    blended_joint_q
        .iter()
        .map(|q| {
            let mut out = q.clone();
            write_root(placement.mul(read_root(q)), &mut out);
            out
        })
        .collect()
}
