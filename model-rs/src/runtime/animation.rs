// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (animation/skeleton.py,
// animation/animation_buffer.py, animation/foot_contacts.py, robotics/dof_animation_buffer.py,
// utils/time_utils.py, utils/dof_utils.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Skeletons and animation buffers: local joint transforms (SOMA) and joint coordinates (G1).

use std::rc::Rc;

use anyhow::{Result, bail};

use super::contract::{FOOT_CONTACT_COUNT, ROOT_COORDINATE_COUNT};
use super::math::*;

/// The frame at a time, the next frame, and the blend toward it; both clamped to the clip.
pub fn frame_index_from_time(time: f32, sample_rate: f32, num_frames: usize) -> (usize, usize, f32) {
    let last = num_frames.saturating_sub(1);
    let fractional = time as f64 * sample_rate as f64;
    let frame = (fractional.floor().max(0.0) as usize).min(last);
    let blend = if frame < last { (fractional - frame as f64).clamp(0.0, 1.0) as f32 } else { 0.0 };
    (frame, (frame + 1).min(last), blend)
}

/// Joint hierarchy and reference pose (one local transform per joint, parents first).
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub joint_names: Vec<String>,
    pub parent_indices: Vec<i32>,
    pub reference_local_transforms: Vec<Transform>,
}

impl Skeleton {
    pub fn new(joint_names: Vec<String>, parent_indices: Vec<i32>, reference_local_transforms: Vec<Transform>) -> Result<Self> {
        if joint_names.is_empty() {
            bail!("a skeleton needs at least one joint");
        }
        if parent_indices.len() != joint_names.len() || reference_local_transforms.len() != joint_names.len() {
            bail!("expected {} parent indices and local transforms", joint_names.len());
        }
        Ok(Skeleton { joint_names, parent_indices, reference_local_transforms })
    }
    pub fn num_joints(&self) -> usize {
        self.joint_names.len()
    }
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        self.joint_names.iter().position(|n| n == name)
    }
    /// Parent index, or -1 for the root.
    pub fn joint_parent(&self, index: usize) -> i32 {
        self.parent_indices[index]
    }
    /// `weight` (clamped to 0..1) on the named joints, zero elsewhere.
    pub fn joint_mask(&self, names: &[&str], weight: f32) -> Vec<f32> {
        let mut w = vec![0.0; self.num_joints()];
        for name in names {
            if let Some(i) = self.joint_index(name) {
                w[i] = weight.clamp(0.0, 1.0);
            }
        }
        w
    }
    pub fn compute_world_transforms(&self, local: &[Transform], root: Transform) -> Vec<Transform> {
        compute_world_poses(local, &self.parent_indices, root)
    }
}

/// Foot contacts of one frame: left heel, left toe, right heel, right toe (confidence 0..1).
pub type FootContacts = [f32; FOOT_CONTACT_COUNT];

fn lerp_contacts(a: FootContacts, b: FootContacts, t: f32) -> FootContacts {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// Animation frames as local joint transforms at a fixed sample rate (SOMA, clips).
///
/// `poses` is `[capacity][num_joints]` flattened; `num_frames` counts the frames in use;
/// `foot_contacts` holds `capacity` frames for generated motion and is `None` for clips.
#[derive(Clone, Debug)]
pub struct AnimationBuffer {
    pub skeleton: Rc<Skeleton>,
    pub sample_rate: f32,
    pub poses: Vec<Transform>,
    pub num_frames: usize,
    pub foot_contacts: Option<Vec<FootContacts>>,
}

impl AnimationBuffer {
    pub fn new(skeleton: Rc<Skeleton>, sample_rate: f32, poses: Vec<Transform>, num_frames: Option<usize>, foot_contacts: Option<Vec<FootContacts>>) -> Result<Self> {
        if sample_rate <= 0.0 {
            bail!("sample_rate must be positive");
        }
        let j = skeleton.num_joints();
        if poses.len() % j != 0 {
            bail!("poses must have shape [capacity, num_joints]");
        }
        let capacity = poses.len() / j;
        if foot_contacts.as_ref().is_some_and(|c| c.len() != capacity) {
            bail!("foot_contacts must have shape [{capacity}, 4]");
        }
        let num_frames = num_frames.unwrap_or(capacity);
        if num_frames > capacity {
            bail!("num_frames must be in [0, {capacity}]");
        }
        Ok(AnimationBuffer { skeleton, sample_rate, poses, num_frames, foot_contacts })
    }
    pub fn capacity(&self) -> usize {
        self.poses.len() / self.skeleton.num_joints()
    }
    pub fn duration(&self) -> f32 {
        self.num_frames.saturating_sub(1) as f32 / self.sample_rate
    }
    pub fn frame(&self, i: usize) -> &[Transform] {
        let j = self.skeleton.num_joints();
        &self.poses[i * j..(i + 1) * j]
    }
    pub fn frame_mut(&mut self, i: usize) -> &mut [Transform] {
        let j = self.skeleton.num_joints();
        &mut self.poses[i * j..(i + 1) * j]
    }
    /// The root of each frame in use.
    pub fn root_transforms(&self) -> Vec<Transform> {
        (0..self.num_frames).map(|i| self.frame(i)[0]).collect()
    }
    /// The pose at a time (two nearest frames blended; times clamped), root moved by `root_offset`.
    pub fn sample(&self, time: f32, root_offset: Transform) -> Result<Vec<Transform>> {
        if self.num_frames == 0 {
            bail!("cannot sample an empty animation buffer");
        }
        let (f, n, b) = frame_index_from_time(time, self.sample_rate, self.num_frames);
        let (a, c) = (self.frame(f), self.frame(n));
        let mut pose: Vec<Transform> = a.iter().zip(c).map(|(x, y)| blend_transform(*x, *y, b)).collect();
        pose[0] = root_offset.mul(pose[0]);
        Ok(pose)
    }
    pub fn sample_foot_contacts(&self, time: f32) -> Result<FootContacts> {
        sample_foot_contacts(self.foot_contacts.as_deref(), self.num_frames, self.sample_rate, time)
    }
    /// A copy of the frames in use (and their foot contacts).
    pub fn copy(&self) -> AnimationBuffer {
        let j = self.skeleton.num_joints();
        AnimationBuffer {
            skeleton: self.skeleton.clone(),
            sample_rate: self.sample_rate,
            poses: self.poses[..self.num_frames * j].to_vec(),
            num_frames: self.num_frames,
            foot_contacts: self.foot_contacts.as_ref().map(|c| c[..self.num_frames].to_vec()),
        }
    }
    /// Adjust the frames in use so the first and last poses match (the correction grows linearly).
    pub fn make_loopable(&mut self) {
        if self.num_frames < 2 {
            return;
        }
        let last = self.num_frames - 1;
        let (first, end) = (self.frame(0).to_vec(), self.frame(last).to_vec());
        let root_delta = compute_root_endpoint_delta(end[0], first[0]);
        let deltas: Vec<Transform> = (1..first.len()).map(|j| compute_delta_transform(end[j], first[j])).collect();
        for frame in 0..self.num_frames {
            let w = frame as f32 / last as f32;
            let pose = self.frame_mut(frame);
            pose[0] = apply_root_endpoint_correction(pose[0], Transform::IDENTITY, root_delta, w);
            for j in 1..pose.len() {
                pose[j] = apply_delta_transform(pose[j], deltas[j - 1], w);
            }
        }
    }
}

/// Remap an animation to another skeleton by joint name; missing joints keep its reference pose.
pub fn create_animation_buffer_for_skeleton(animation: &AnimationBuffer, skeleton: Rc<Skeleton>) -> AnimationBuffer {
    if Rc::ptr_eq(&animation.skeleton, &skeleton) {
        return animation.clone();
    }
    let j = skeleton.num_joints();
    let mapping: Vec<Option<usize>> = animation.skeleton.joint_names.iter().map(|n| skeleton.joint_index(n)).collect();
    let mut poses = Vec::with_capacity(animation.num_frames * j);
    for frame in 0..animation.num_frames {
        let mut pose = skeleton.reference_local_transforms.clone();
        for (source, target) in mapping.iter().enumerate() {
            if let Some(t) = target {
                pose[*t] = animation.frame(frame)[source];
            }
        }
        poses.extend(pose);
    }
    AnimationBuffer { skeleton, sample_rate: animation.sample_rate, poses, num_frames: animation.num_frames, foot_contacts: None }
}

/// Root transform of joint coordinates, with a normalized rotation.
pub fn read_root(q: &[f32]) -> Transform {
    Transform::new(Vec3([q[0], q[1], q[2]]), quat_normalize([q[3], q[4], q[5], q[6]]))
}

/// Write a root transform (rotation normalized) into joint coordinates.
pub fn write_root(root: Transform, q: &mut [f32]) {
    q[..3].copy_from_slice(&root.p.0);
    q[3..7].copy_from_slice(&quat_normalize(root.q));
}

/// Animation frames as joint coordinates: root position (m), root rotation (XYZW), joint angles
/// (rad) in `joint_names` order (G1). `poses` is `[capacity][7 + num_dofs]` flattened.
#[derive(Clone, Debug)]
pub struct DofAnimationBuffer {
    pub joint_names: Vec<String>,
    pub sample_rate: f32,
    pub poses: Vec<f32>,
    pub num_frames: usize,
    pub foot_contacts: Option<Vec<FootContacts>>,
}

impl DofAnimationBuffer {
    pub fn new(joint_names: Vec<String>, sample_rate: f32, poses: Vec<f32>, num_frames: Option<usize>, foot_contacts: Option<Vec<FootContacts>>) -> Result<Self> {
        if sample_rate <= 0.0 {
            bail!("sample_rate must be positive");
        }
        let nc = ROOT_COORDINATE_COUNT + joint_names.len();
        if poses.len() % nc != 0 {
            bail!("poses must have shape [capacity, {nc}]");
        }
        let capacity = poses.len() / nc;
        if foot_contacts.as_ref().is_some_and(|c| c.len() != capacity) {
            bail!("foot_contacts must have shape [{capacity}, 4]");
        }
        let num_frames = num_frames.unwrap_or(capacity);
        if num_frames > capacity {
            bail!("num_frames must be in [0, {capacity}]");
        }
        Ok(DofAnimationBuffer { joint_names, sample_rate, poses, num_frames, foot_contacts })
    }
    pub fn num_dofs(&self) -> usize {
        self.joint_names.len()
    }
    pub fn num_coordinates(&self) -> usize {
        ROOT_COORDINATE_COUNT + self.joint_names.len()
    }
    pub fn capacity(&self) -> usize {
        self.poses.len() / self.num_coordinates()
    }
    pub fn duration(&self) -> f32 {
        self.num_frames.saturating_sub(1) as f32 / self.sample_rate
    }
    pub fn frame(&self, i: usize) -> &[f32] {
        let nc = self.num_coordinates();
        &self.poses[i * nc..(i + 1) * nc]
    }
    pub fn frame_mut(&mut self, i: usize) -> &mut [f32] {
        let nc = self.num_coordinates();
        &mut self.poses[i * nc..(i + 1) * nc]
    }
    pub fn root_transforms(&self) -> Vec<Transform> {
        (0..self.num_frames).map(|i| read_root(self.frame(i))).collect()
    }
    /// Joint coordinates at a time; angles lerp (also across +-pi), the root slerps.
    pub fn sample(&self, time: f32, root_offset: Transform) -> Result<Vec<f32>> {
        if self.num_frames == 0 {
            bail!("cannot sample an empty animation buffer");
        }
        let (f, n, b) = frame_index_from_time(time, self.sample_rate, self.num_frames);
        let (a, c) = (self.frame(f), self.frame(n));
        let mut out: Vec<f32> = a.iter().zip(c).map(|(x, y)| x + (y - x) * b).collect();
        write_root(root_offset.mul(blend_transform(read_root(a), read_root(c), b)), &mut out);
        Ok(out)
    }
    pub fn sample_foot_contacts(&self, time: f32) -> Result<FootContacts> {
        sample_foot_contacts(self.foot_contacts.as_deref(), self.num_frames, self.sample_rate, time)
    }
    pub fn copy(&self) -> DofAnimationBuffer {
        let nc = self.num_coordinates();
        DofAnimationBuffer {
            joint_names: self.joint_names.clone(),
            sample_rate: self.sample_rate,
            poses: self.poses[..self.num_frames * nc].to_vec(),
            num_frames: self.num_frames,
            foot_contacts: self.foot_contacts.as_ref().map(|c| c[..self.num_frames].to_vec()),
        }
    }
    pub fn make_loopable(&mut self) {
        if self.num_frames < 2 {
            return;
        }
        let last = self.num_frames - 1;
        let (first, end) = (self.frame(0).to_vec(), self.frame(last).to_vec());
        let root_delta = compute_root_endpoint_delta(read_root(&end), read_root(&first));
        for frame in 0..self.num_frames {
            let w = frame as f32 / last as f32;
            let q = self.frame_mut(frame);
            write_root(apply_root_endpoint_correction(read_root(q), Transform::IDENTITY, root_delta, w), q);
            for c in ROOT_COORDINATE_COUNT..q.len() {
                q[c] += (first[c] - end[c]) * w;
            }
        }
    }
}

fn sample_foot_contacts(contacts: Option<&[FootContacts]>, num_frames: usize, sample_rate: f32, time: f32) -> Result<FootContacts> {
    let Some(c) = contacts else { bail!("cannot sample foot contacts from an animation buffer that has none") };
    if num_frames == 0 {
        bail!("cannot sample an empty animation buffer");
    }
    let (f, n, b) = frame_index_from_time(time, sample_rate, num_frames);
    Ok(lerp_contacts(c[f], c[n], b))
}
