// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Blend spaces and target-pose blending, ported from upstream
//! `motionbricks/runtime/blend_spaces.py` and `app/runtime/common/target_pose_blender.py`
//! with the per-rig blenders (`soma/target_pose_blender.py`, `unitree_g1/target_pose_blender.py`).
//! Warp kernels become plain loops with the same math.

use std::rc::Rc;

use anyhow::{Result, bail};
use glam::{Vec2, Vec3};
use motionbricks_rs::runtime as rt;
use rt::math::{self as rtm, Transform};

use super::motion::{LocomotionModeRuntime, PoseElem, directional_blend_space_value, to_rt};

const EPSILON: f32 = 1.0e-6;

/// The type of blend space used to compute target poses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendSpaceType {
    Linear1D,
    Directional2D,
}

/// Blend targets and blend weights that [`BlendSpace::evaluate`] selects (upstream
/// `BlendSample`).
///
/// `indices` tells which added targets were selected; `data_indices` tells where each
/// target's pose data is stored. A 2D space without a target at (0, 0) uses the average
/// of all targets as its center: `uniform_weight` is each target's share of the
/// center's weight.
#[derive(Clone, Debug)]
pub struct BlendSample {
    pub indices: [usize; 3],
    pub weights: [f32; 3],
    pub data_indices: Rc<Vec<usize>>,
    pub uniform_weight: f32,
}

impl BlendSample {
    /// Each contributing `(data_index, weight)` in slot order, upstream's
    /// `_contribution_count` / `_contribution`.
    pub fn contributions(&self) -> Vec<(usize, f32)> {
        if self.uniform_weight > 0.0 {
            (0..self.data_indices.len())
                .map(|slot| {
                    let mut weight = self.uniform_weight;
                    for selected in 0..3 {
                        if self.indices[selected] == slot {
                            weight += self.weights[selected];
                        }
                    }
                    (self.data_indices[slot], weight)
                })
                .collect()
        } else {
            (0..3).map(|slot| (self.data_indices[self.indices[slot]], self.weights[slot])).collect()
        }
    }
}

#[derive(Clone, Debug)]
struct Sector {
    inverse: [f32; 4],
    indices: [usize; 3],
    order: [usize; 3],
}

#[derive(Clone, Debug)]
enum CompiledSpace {
    Linear { keys: Vec<f32>, intervals: Vec<(usize, usize, f32, f32)> },
    Directional { center: i64, angles: Vec<f32>, sectors: Vec<Option<Sector>> },
}

/// Select targets and weights from a 1D value or a 2D direction (upstream `BlendSpace`).
///
/// Add targets with [`BlendSpace::add_target`], call [`BlendSpace::compile`] after setup,
/// then use [`BlendSpace::evaluate`] to get the targets and weights for each blend.
/// A 1D space blends between neighboring values. A 2D directional space blends between
/// neighboring directions and an optional center target at (0, 0).
#[derive(Clone, Debug)]
pub struct BlendSpace {
    space_type: BlendSpaceType,
    target_values: Vec<(f32, f32)>,
    target_data_indices: Vec<usize>,
    data_indices: Rc<Vec<usize>>,
    compiled: Option<CompiledSpace>,
}

fn value_components(value: Vec2) -> Result<(f32, f32)> {
    if !value.x.is_finite() || !value.y.is_finite() {
        bail!("Blend space values must contain two finite components.");
    }
    Ok((value.x, value.y))
}

fn bisect_left(keys: &[f32], x: f32) -> usize {
    keys.partition_point(|&key| key < x)
}

impl BlendSpace {
    pub fn new(blend_space_type: BlendSpaceType) -> Self {
        BlendSpace { space_type: blend_space_type, target_values: Vec::new(), target_data_indices: Vec::new(), data_indices: Rc::new(Vec::new()), compiled: None }
    }

    /// Kind of blend space, 1D or 2D directional.
    pub fn space_type(&self) -> BlendSpaceType {
        self.space_type
    }

    /// Number of added targets.
    pub fn target_count(&self) -> usize {
        self.target_values.len()
    }

    /// Source pose indices, in the order the targets were added.
    pub fn data_indices(&self) -> &[usize] {
        &self.data_indices
    }

    /// Add a blend target with its value and its index in the source pose array.
    ///
    /// A 1D space uses only `value.x`. A 2D space reads it as a direction, and (0, 0)
    /// marks the center target. `data_index` is the target's index in the source pose
    /// array.
    pub fn add_target(&mut self, value: Vec2, data_index: usize) -> Result<()> {
        self.target_values.push(value_components(value)?);
        self.target_data_indices.push(data_index);
        self.compiled = None;
        Ok(())
    }

    /// Prepare the target lookup used by [`BlendSpace::evaluate`]. Call this after
    /// adding the targets, and again if you add more.
    pub fn compile(&mut self) -> Result<()> {
        let values = &self.target_values;
        if values.is_empty() {
            bail!("Cannot compile a blend space without targets.");
        }
        self.compiled = Some(match self.space_type {
            BlendSpaceType::Linear1D => {
                let mut order: Vec<usize> = (0..values.len()).collect();
                order.sort_by(|&a, &b| values[a].0.total_cmp(&values[b].0));
                let keys: Vec<f32> = order.iter().map(|&index| values[index].0).collect();
                if keys.windows(2).any(|w| w[0] == w[1]) {
                    bail!("Blend target values in a 1D blend space must be unique.");
                }
                let intervals = order
                    .iter()
                    .zip(order.iter().skip(1))
                    .map(|(&a, &b)| (a, b, values[a].0, 1.0 / (values[b].0 - values[a].0)))
                    .collect();
                CompiledSpace::Linear { keys, intervals }
            }
            BlendSpaceType::Directional2D => {
                let centers: Vec<usize> = (0..values.len()).filter(|&index| values[index] == (0.0, 0.0)).collect();
                if centers.len() > 1 {
                    bail!("A directional blend space can have only one center target.");
                }
                let center = centers.first().map_or(-1, |&index| index as i64);
                let mut order: Vec<usize> = (0..values.len()).filter(|&index| Some(index as i64) != centers.first().copied().map(|c| c as i64)).collect();
                order.sort_by(|&a, &b| values[a].1.atan2(values[a].0).total_cmp(&values[b].1.atan2(values[b].0)));
                let angles: Vec<f32> = order.iter().map(|&index| values[index].1.atan2(values[index].0)).collect();
                let mut sectors = Vec::with_capacity(order.len());
                for upper in 0..order.len() {
                    let (a, b) = (order[(upper + order.len() - 1) % order.len()], order[upper]);
                    let gap = (angles[upper] - angles[(upper + order.len() - 1) % order.len()]).rem_euclid(std::f32::consts::TAU);
                    if order.len() > 1 && gap < EPSILON {
                        bail!("Directional targets must have distinct directions.");
                    }
                    let (ax, ay) = values[a];
                    let (bx, by) = values[b];
                    let determinant = ax * by - ay * bx;
                    // A valid sector needs two directions less than 180 degrees apart.
                    if gap >= std::f32::consts::PI || determinant <= 0.0 {
                        sectors.push(None);
                        continue;
                    }
                    let inverse = [by / determinant, -bx / determinant, -ay / determinant, ax / determinant];
                    let slots = [a, b, (center.max(0)) as usize];
                    let mut weight_order = [0usize, 1, 2];
                    weight_order.sort_by_key(|&slot| slots[slot]);
                    sectors.push(Some(Sector { inverse, indices: [slots[weight_order[0]], slots[weight_order[1]], slots[weight_order[2]]], order: weight_order }));
                }
                CompiledSpace::Directional { center, angles, sectors }
            }
        });
        self.data_indices = Rc::new(self.target_data_indices.clone());
        Ok(())
    }

    fn sample(&self, indices: [usize; 3], weights: [f32; 3], uniform_weight: f32) -> BlendSample {
        BlendSample { indices, weights, data_indices: self.data_indices.clone(), uniform_weight }
    }

    /// Select targets and compute their blend weights for the given input.
    ///
    /// 1D targets are ordered from the lower value to the higher one. 2D targets follow
    /// the order they were added. A 1D space clamps `value.x` to the target range. A 2D
    /// space reads `value` as a direction and a distance from the center. Near zero, it
    /// selects the center target; without a center target, every target then gets the
    /// same weight.
    pub fn evaluate(&self, value: Vec2) -> Result<BlendSample> {
        let Some(compiled) = &self.compiled else {
            bail!("Cannot evaluate before compile(). Call it after adding blend targets.");
        };
        let (x, y) = value_components(value)?;
        if self.target_count() == 1 {
            return Ok(self.sample([0, 0, 0], [1.0, 0.0, 0.0], 0.0));
        }
        match compiled {
            CompiledSpace::Linear { keys, intervals } => {
                let x = x.clamp(keys[0], keys[keys.len() - 1]);
                let interval = bisect_left(keys, x).saturating_sub(1).min(intervals.len() - 1);
                let (a, b, start, inverse_width) = intervals[interval];
                let blend = (x - start) * inverse_width;
                Ok(self.sample([a, b, 0], [1.0 - blend, blend, 0.0], 0.0))
            }
            CompiledSpace::Directional { center, angles, sectors } => {
                if (x.abs() > EPSILON || y.abs() > EPSILON)
                    && !angles.is_empty() {
                        let upper = bisect_left(angles, y.atan2(x)) % angles.len();
                        if let Some(sector) = &sectors[upper] {
                            let a = (sector.inverse[0] * x + sector.inverse[1] * y).max(0.0);
                            let b = (sector.inverse[2] * x + sector.inverse[3] * y).max(0.0);
                            let total = (a + b).max(1.0);
                            let (a, b) = (a / total, b / total);
                            let center_weight = (1.0 - a - b).max(0.0);
                            let weights = [a, b, if *center >= 0 { center_weight } else { 0.0 }];
                            return Ok(self.sample(
                                sector.indices,
                                [weights[sector.order[0]], weights[sector.order[1]], weights[sector.order[2]]],
                                if *center < 0 { center_weight / self.target_count() as f32 } else { 0.0 },
                            ));
                        }
                    }
                if *center >= 0 {
                    return Ok(self.sample([*center as usize, 0, 0], [1.0, 0.0, 0.0], 0.0));
                }
                Ok(self.sample([0, 0, 0], [0.0, 0.0, 0.0], 1.0 / self.target_count() as f32))
            }
        }
    }
}

/// Give target quaternions consistent signs before 2D directional blending (upstream
/// `prepare_directional_poses`). Each quaternion takes the sign that matches the
/// reference pose's rotation for its joint.
pub fn prepare_directional_poses(poses: &mut [Vec<Transform>], reference_pose: &[Transform]) {
    for pose in poses {
        PoseElem::align_quaternion_signs(pose, reference_pose);
    }
}

fn frames_and_joints<T>(poses: &[Vec<T>], output: &[Vec<T>]) -> Result<(usize, usize)> {
    let frames = output.len();
    if frames == 0 || output.iter().any(|pose| pose.len() != output[0].len()) {
        bail!("Blend output must have frame capacity and matching components.");
    }
    let joints = output[0].len();
    if poses.len() < frames || poses.len() % frames != 0 || poses.iter().any(|pose| pose.len() != joints) {
        bail!("Blend data must be [target, frame, component] and output [frame, component].");
    }
    Ok((frames, joints))
}

/// Blend the selected 1D targets into one sequence of local joint transforms.
///
/// `poses` holds the poses of every target, laid out as [target, frame] flattened;
/// `output` receives the blended frames.
pub fn linear_1d_pose_blend(poses: &[Vec<Transform>], sample: &BlendSample, output: &mut [Vec<Transform>]) -> Result<()> {
    let (frames, joints) = frames_and_joints(poses, output)?;
    let first = sample.data_indices[sample.indices[0]];
    let second = sample.data_indices[sample.indices[1]];
    let blend = sample.weights[1];
    for frame in 0..frames {
        for joint in 0..joints {
            let mut result = poses[first * frames + frame][joint];
            if blend >= 1.0 {
                result = poses[second * frames + frame][joint];
            } else if blend > 0.0 && first != second {
                result = rtm::blend_transform(result, poses[second * frames + frame][joint], blend);
            }
            output[frame][joint] = result;
        }
    }
    Ok(())
}

/// Blend the selected 2D targets into one sequence of local joint transforms.
///
/// Call [`prepare_directional_poses`] on `poses` once during setup. When the weights
/// add up to less than one, the reference pose supplies the rest.
pub fn directional_2d_pose_blend(poses: &[Vec<Transform>], sample: &BlendSample, reference_pose: &[Transform], output: &mut [Vec<Transform>]) -> Result<()> {
    let (frames, joints) = frames_and_joints(poses, output)?;
    if reference_pose.len() != joints {
        bail!("reference_pose must contain one transform per joint.");
    }
    let contributions = sample.contributions();
    for frame in 0..frames {
        for joint in 0..joints {
            let mut p = rtm::Vec3::ZERO;
            let mut q = [0.0f32; 4];
            let mut weight_sum = 0.0f32;
            for &(target, weight) in &contributions {
                if weight > 0.0 {
                    let pose = poses[target * frames + frame][joint];
                    p = p + pose.p * weight;
                    for k in 0..4 {
                        q[k] += pose.q[k] * weight;
                    }
                    weight_sum += weight;
                }
            }
            let reference = reference_pose[joint];
            let missing = (1.0 - weight_sum).max(0.0);
            for k in 0..4 {
                q[k] += reference.q[k] * missing;
            }
            output[frame][joint] = Transform::new(p + reference.p * missing, rtm::quat_normalize(q));
        }
    }
    Ok(())
}

/// Add a weighted transform to a running sum (upstream
/// `blend_additive_accumulate_transform`). The rotation is flipped onto the sum's side
/// of the quaternion sphere first.
fn blend_additive_accumulate_transform(accumulated: (rtm::Vec3, rtm::Quat), pose: Transform, weight: f32) -> (rtm::Vec3, rtm::Quat) {
    let dot: f32 = (0..4).map(|k| accumulated.1[k] * pose.q[k]).sum();
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    let mut q = [0.0f32; 4];
    for k in 0..4 {
        q[k] = accumulated.1[k] + pose.q[k] * sign * weight;
    }
    (accumulated.0 + pose.p * weight, q)
}

/// Finish an additive blend and normalize its rotation (upstream
/// `blend_additive_end_transform`). Weight missing below one comes from
/// `default_transform`.
fn blend_additive_end_transform(accumulated: (rtm::Vec3, rtm::Quat), default_transform: Transform, weight_sum: f32) -> Transform {
    let accumulated = if weight_sum < 1.0 { blend_additive_accumulate_transform(accumulated, default_transform, 1.0 - weight_sum) } else { accumulated };
    Transform::new(accumulated.0, rtm::quat_normalize(accumulated.1))
}

/// Blend the selected 1D targets into one sequence of joint coordinates.
///
/// `joint_q` holds the joint coordinates of every target, laid out as [target, frame]
/// flattened. Each frame starts with the root position and XYZW quaternion.
pub fn linear_1d_dof_blend(joint_q: &[Vec<f32>], sample: &BlendSample, output: &mut [Vec<f32>]) -> Result<()> {
    let (frames, nc) = frames_and_joints(joint_q, output)?;
    if nc < rt::ROOT_COORDINATE_COUNT {
        bail!("joint_q must contain the root position and XYZW quaternion before the joint angles.");
    }
    let first = sample.data_indices[sample.indices[0]];
    let second = sample.data_indices[sample.indices[1]];
    let blend = sample.weights[1];
    for frame in 0..frames {
        let (a, b) = (&joint_q[first * frames + frame], &joint_q[second * frames + frame]);
        let mut root = rt::read_root(a);
        if blend >= 1.0 {
            root = rt::read_root(b);
        } else if blend > 0.0 && first != second {
            root = rtm::blend_transform(root, rt::read_root(b), blend);
        }
        rt::write_root(root, &mut output[frame]);
        for coordinate in rt::ROOT_COORDINATE_COUNT..nc {
            let mut value = a[coordinate];
            if blend >= 1.0 {
                value = b[coordinate];
            } else if blend > 0.0 {
                value += (b[coordinate] - value) * blend;
            }
            output[frame][coordinate] = value;
        }
    }
    Ok(())
}

/// Blend the selected 2D targets into one sequence of joint coordinates.
///
/// When the weights add up to less than one, `reference_joint_q` supplies the rest.
pub fn directional_2d_dof_blend(joint_q: &[Vec<f32>], sample: &BlendSample, reference_joint_q: &[f32], output: &mut [Vec<f32>]) -> Result<()> {
    let (frames, nc) = frames_and_joints(joint_q, output)?;
    if nc < rt::ROOT_COORDINATE_COUNT || reference_joint_q.len() != nc {
        bail!("reference_joint_q must hold the root position, XYZW quaternion, and joint angles.");
    }
    let reference_root = rt::read_root(reference_joint_q);
    let contributions = sample.contributions();
    for frame in 0..frames {
        let mut accumulated = (rtm::Vec3::ZERO, [0.0f32; 4]);
        let mut weight_sum = 0.0f32;
        for &(target, weight) in &contributions {
            if weight > 0.0 {
                accumulated = blend_additive_accumulate_transform(accumulated, rt::read_root(&joint_q[target * frames + frame]), weight);
                weight_sum += weight;
            }
        }
        let root = blend_additive_end_transform(accumulated, reference_root, weight_sum);
        rt::write_root(root, &mut output[frame]);
        for coordinate in rt::ROOT_COORDINATE_COUNT..nc {
            let mut value = 0.0f32;
            let mut weight_sum = 0.0f32;
            for &(target, weight) in &contributions {
                if weight > 0.0 {
                    value += joint_q[target * frames + frame][coordinate] * weight;
                    weight_sum += weight;
                }
            }
            output[frame][coordinate] = value + reference_joint_q[coordinate] * (1.0 - weight_sum).max(0.0);
        }
    }
    Ok(())
}

/// What the target pose blender needs from the character: where it stands, where it
/// faces, and what velocity it aims for.
#[derive(Clone, Copy, Debug)]
pub struct TargetPoseContext {
    pub ground_position: Vec3,
    pub current_facing: Vec3,
    pub desired_velocity: Vec3,
}

/// Turn a locomotion mode and the character's desired velocity into target poses
/// (upstream `BaseTargetPoseBlender`).
///
/// [`TargetPoseBlender::evaluate`] runs the shared steps; each rig implements the
/// blend, speed, and placement steps for its pose layout.
pub trait TargetPoseBlender<E> {
    /// Blend the targets of a 1D blend space into `output`.
    fn blend_linear(&self, target_poses: &[Vec<E>], sample: &BlendSample, output: &mut [Vec<E>]) -> Result<()>;

    /// Blend the targets of a 2D blend space into `output`.
    ///
    /// When the blend weights add up to less than one, a base pose fills the rest. SOMA
    /// uses its reference pose, and G1 uses the Idle target.
    fn blend_directional(&self, target_poses: &[Vec<E>], sample: &BlendSample, output: &mut [Vec<E>]) -> Result<()>;

    /// Scale the root's motion along the ground to a mean speed.
    fn adjust_speed(&self, poses: &mut [Vec<E>], speed: f32) -> Result<()>;

    /// Copy the blended poses into `target_poses`, placed on the ground and turned to
    /// face a direction. One placement moves every root, so the roots keep their
    /// offsets from one another. Other joint values stay as they are.
    fn place(&self, ground_position: Vec3, facing_direction: Vec3, blended_poses: &[Vec<E>], target_poses: &mut Vec<Vec<E>>) -> Result<()>;

    /// Blend the mode's targets for the desired velocity and place them in
    /// `target_poses` (upstream `BaseTargetPoseBlender.evaluate`).
    fn evaluate(&self, context: &TargetPoseContext, mode_runtime: &LocomotionModeRuntime<E>, blended_poses: &mut Vec<Vec<E>>, target_poses: &mut Vec<Vec<E>>) -> Result<()> {
        let mode = &mode_runtime.definition;
        let speed = context.desired_velocity.length();
        if mode.blend_space_type == BlendSpaceType::Linear1D {
            let sample = mode_runtime.blend_space.evaluate(Vec2::new(speed / mode.max_speed, 0.0))?;
            self.blend_linear(&mode_runtime.target_poses, &sample, blended_poses)?;
        } else {
            let sample = mode_runtime.blend_space.evaluate(directional_blend_space_value(context.current_facing, context.desired_velocity, mode.max_speed))?;
            self.blend_directional(&mode_runtime.target_poses, &sample, blended_poses)?;
        }
        self.adjust_speed(blended_poses, speed * mode.velocity_scale)?;
        let mut ground_position = context.ground_position + context.desired_velocity * (mode.velocity_scale * mode.position_scale);
        ground_position.z = 0.0;
        self.place(ground_position, context.current_facing, blended_poses, target_poses)
    }
}

/// Blend SOMA local joint transforms around the skeleton's reference pose (upstream
/// `SOMATargetPoseBlender`).
pub struct SomaTargetPoseBlender {
    pub reference_pose: Vec<Transform>,
}

impl SomaTargetPoseBlender {
    pub fn new(reference_pose: Vec<Transform>) -> Self {
        SomaTargetPoseBlender { reference_pose }
    }
}

impl TargetPoseBlender<Transform> for SomaTargetPoseBlender {
    fn blend_linear(&self, target_poses: &[Vec<Transform>], sample: &BlendSample, output: &mut [Vec<Transform>]) -> Result<()> {
        linear_1d_pose_blend(target_poses, sample, output)
    }

    fn blend_directional(&self, target_poses: &[Vec<Transform>], sample: &BlendSample, output: &mut [Vec<Transform>]) -> Result<()> {
        directional_2d_pose_blend(target_poses, sample, &self.reference_pose, output)
    }

    fn adjust_speed(&self, poses: &mut [Vec<Transform>], speed: f32) -> Result<()> {
        rt::adjust_target_pose_speed(poses, speed)
    }

    fn place(&self, ground_position: Vec3, facing_direction: Vec3, blended_poses: &[Vec<Transform>], target_poses: &mut Vec<Vec<Transform>>) -> Result<()> {
        *target_poses = rt::place_target_poses(to_rt(ground_position), to_rt(facing_direction), blended_poses);
        Ok(())
    }
}

/// Blend G1 joint coordinates around the Idle target of each 2D directional blend
/// space (upstream `G1TargetPoseBlender`).
pub struct G1TargetPoseBlender;

impl TargetPoseBlender<f32> for G1TargetPoseBlender {
    fn blend_linear(&self, target_poses: &[Vec<f32>], sample: &BlendSample, output: &mut [Vec<f32>]) -> Result<()> {
        linear_1d_dof_blend(target_poses, sample, output)
    }

    fn blend_directional(&self, target_poses: &[Vec<f32>], sample: &BlendSample, output: &mut [Vec<f32>]) -> Result<()> {
        // DIRECTIONAL_TARGET_VALUES lists the Idle target first.
        directional_2d_dof_blend(target_poses, sample, &target_poses[0], output)
    }

    fn adjust_speed(&self, poses: &mut [Vec<f32>], speed: f32) -> Result<()> {
        rt::adjust_target_dof_speed(poses, speed)
    }

    fn place(&self, ground_position: Vec3, facing_direction: Vec3, blended_poses: &[Vec<f32>], target_poses: &mut Vec<Vec<f32>>) -> Result<()> {
        *target_poses = rt::place_target_dof(to_rt(ground_position), to_rt(facing_direction), blended_poses);
        Ok(())
    }
}

