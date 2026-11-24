// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Smart Primitive runtime assets and instances, pivots, and the behaviors scenes pick
//! by class name, ported from upstream `motionbricks/runtime/pivot.py`,
//! `motionbricks/runtime/smart_primitive.py`, and
//! `app/runtime/common/smart_primitive_behaviors.py` (plus the asset preparation of
//! `app/runtime/common/io/smart_primitive_document.py`). Warp kernels become plain
//! loops with the same math.
//!
//! The runtime consumes the resolved smart primitive data of
//! [`crate::documents`]: interaction boxes, sequences, and pivots. Clip sampling needs
//! loaded clips, which the caller supplies through a loader closure (the rig module's
//! clip registry in the bins).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Result, anyhow, bail};
use glam::Vec3;
use motionbricks_rs::runtime as rt;
use rt::animation::MotionBuffer;
use rt::math::{self as rtm, Transform};

use crate::documents::smart_primitive::{Axis, ConeSide, ConstraintType, OperationOrder, PlaybackMode, PivotComponent, SmartPrimitiveDocument};
use crate::documents::{self};

use super::interaction::InteractionBox;
use super::motion::{POSES_PER_TOKEN, PoseElem, to_glam, to_rt, wrap_angle};

/// Character values in world space, for Smart Primitive behaviors (upstream
/// `CharacterProperties`).
#[derive(Clone, Copy, Debug)]
pub struct CharacterProperties {
    /// Character position on the ground, in meters.
    pub position: Vec3,
    /// Actual velocity of the character, in m/s.
    pub velocity: Vec3,
    /// Direction the character faces on the ground, for custom behaviors.
    pub facing: Vec3,
    /// Velocity the character aims for, in m/s.
    pub desired_velocity: Vec3,
    /// Near and far ends of the character's probe box on the ground.
    pub probe_segment: (Vec3, Vec3),
}

fn axis_vector(axis: Axis) -> Option<Vec3> {
    match axis {
        Axis::None => None,
        Axis::X => Some(Vec3::X),
        Axis::Y => Some(Vec3::Y),
        Axis::Z => Some(Vec3::Z),
    }
}

/// Rules that move and turn a pivot node before its sequence plays (upstream `Pivot`).
///
/// The rules measure from the pivot's authored world transform. Moving the pivot does
/// not move its limits, so repeated alignments never drift. Axes are local to the
/// authored transform, so tilting the node tilts its rules.
#[derive(Clone, Debug)]
pub struct Pivot {
    /// Local direction that lines up with the approach, stored at unit length.
    pub motion_direction: Vec3,
    pub operation_order: OperationOrder,
    pub translation_axis: Axis,
    pub translation_constraint: ConstraintType,
    pub line_min_offset: f32,
    pub line_max_offset: f32,
    pub rotation_axis: Axis,
    pub rotation_constraint: ConstraintType,
    pub cone_half_angle_degrees: f32,
    pub cone_side: ConeSide,
}

impl Pivot {
    /// The runtime rules of a document's [`PivotComponent`], checked as upstream's
    /// `Pivot.validate` and `Pivot.set_motion_direction` (the motion direction is
    /// normalized here, as upstream's runtime does).
    pub fn from_component(component: &PivotComponent) -> Result<Self> {
        let direction = component.motion_direction;
        let length = direction.length();
        if !length.is_finite() || length == 0.0 {
            bail!("Motion Direction must be finite and non-zero.");
        }
        if component.translation_constraint == ConstraintType::Cone {
            bail!("Translation only supports Unconstrained or Line Projection.");
        }
        if component.rotation_constraint == ConstraintType::LineProjection {
            bail!("Rotation only supports Unconstrained or Cone.");
        }
        if !component.line_min_offset.is_finite() || !component.line_max_offset.is_finite() || component.line_min_offset > component.line_max_offset {
            bail!("Line offsets must be finite, with Min Offset <= Max Offset.");
        }
        if !component.cone_half_angle_degrees.is_finite() || !(0.0..=180.0).contains(&component.cone_half_angle_degrees) {
            bail!("Cone Half Angle must be between 0 and 180 degrees.");
        }
        Ok(Pivot {
            motion_direction: direction / length,
            operation_order: component.operation_order,
            translation_axis: component.translation_axis,
            translation_constraint: component.translation_constraint,
            line_min_offset: component.line_min_offset,
            line_max_offset: component.line_max_offset,
            rotation_axis: component.rotation_axis,
            rotation_constraint: component.rotation_constraint,
            cone_half_angle_degrees: component.cone_half_angle_degrees,
            cone_side: component.cone_side,
        })
    }

    /// Both ends of the line the pivot slides along, in world space. The pivot must
    /// have a translation axis.
    pub fn line_endpoints(&self, limits_transform: Transform) -> (Vec3, Vec3) {
        let axis = axis_vector(self.translation_axis).expect("line_endpoints needs a translation axis");
        let direction = to_glam(limits_transform.vector(to_rt(axis)));
        (
            to_glam(limits_transform.p) + direction * self.line_min_offset,
            to_glam(limits_transform.p) + direction * self.line_max_offset,
        )
    }

    /// Return the pivot's world transform aligned to the character's approach.
    ///
    /// An axis set to `None` keeps the current position or rotation. `limits_transform`
    /// is the pivot's authored world transform, which the limits measure from;
    /// `current_world_transform` is its world transform before this alignment.
    pub fn aligned_transform(&self, authored_world_transform: Transform, current_world_transform: Transform, character: &CharacterProperties) -> Transform {
        let mut limits_transform = authored_world_transform;
        let mut position = to_glam(current_world_transform.p);
        let mut rotation = glam_quat(current_world_transform.q);

        if self.operation_order == OperationOrder::RotationThenTranslation {
            rotation = self.aligned_rotation(limits_transform, rotation, character);
            if self.rotation_axis != Axis::None {
                limits_transform = Transform::new(limits_transform.p, rt_quat(rotation));
            }
            position = self.aligned_position(limits_transform, position, character);
        } else {
            position = self.aligned_position(limits_transform, position, character);
            rotation = self.aligned_rotation(limits_transform, rotation, character);
        }
        Transform::new(to_rt(position), rt_quat(rotation))
    }

    fn aligned_position(&self, limits_transform: Transform, current_position: Vec3, character: &CharacterProperties) -> Vec3 {
        let Some(axis) = axis_vector(self.translation_axis) else {
            return current_position;
        };
        if self.translation_constraint == ConstraintType::LineProjection {
            let (line_start, line_end) = self.line_endpoints(limits_transform);
            return closest_point_to_segment(line_start, line_end, character.probe_segment.0, character.probe_segment.1);
        }

        // Without line limits, the pivot can slide any distance along its axis.
        let character_position = character.position;
        let desired = character.desired_velocity;
        let origin = to_glam(limits_transform.p);
        let axis = to_glam(limits_transform.vector(to_rt(axis))).normalize();
        let relative = character_position - origin;
        let nearest_offset = relative.dot(axis);
        let mut offset = nearest_offset;
        let speed = desired.length();
        if speed.is_finite() && speed > 1.0e-6 {
            let ray = desired / speed;
            let cosine = axis.dot(ray);
            let denominator = 1.0 - cosine * cosine;
            if denominator > 1.0e-6 {
                offset = (nearest_offset - cosine * relative.dot(ray)) / denominator;
            }
            if (origin + axis * offset - character_position).dot(ray) < 0.0 {
                offset = nearest_offset;
            }
        }
        origin + axis * offset
    }

    fn aligned_rotation(&self, limits_transform: Transform, current_rotation: glam::Quat, character: &CharacterProperties) -> glam::Quat {
        let Some(rotation_axis) = axis_vector(self.rotation_axis) else {
            return current_rotation;
        };
        let limits_rotation = glam_quat(limits_transform.q);
        let axis = (limits_rotation * rotation_axis).normalize();
        let mut motion = limits_rotation * self.motion_direction;
        // Movement along the rotation axis cannot tell us which way to turn.
        motion -= axis * motion.dot(axis);
        let mut desired = character.desired_velocity;
        desired -= axis * desired.dot(axis);
        for length in [motion.length(), desired.length()] {
            if !length.is_finite() || length <= 1.0e-6 {
                return current_rotation;
            }
        }

        let mut angle = axis.dot(motion.cross(desired)).atan2(motion.dot(desired));
        if self.rotation_constraint == ConstraintType::Cone {
            let half_angle = self.cone_half_angle_degrees.to_radians();
            // Find the nearest allowed heading, checking both cones when enabled.
            let centers: &[f32] = match self.cone_side {
                ConeSide::Both => &[0.0, std::f32::consts::PI],
                ConeSide::Mirrored => &[std::f32::consts::PI],
                ConeSide::Forward => &[0.0],
            };
            let mut best = (f32::INFINITY, 0.0);
            for &center in centers {
                let delta = wrap_angle(angle - center);
                let clamped = delta.clamp(-half_angle, half_angle);
                let cost = (delta - clamped).abs();
                if cost < best.0 {
                    best = (cost, center + clamped);
                }
            }
            angle = best.1;
        }
        glam::Quat::from_axis_angle(axis, angle) * limits_rotation
    }
}

fn closest_point_on_segment(start: Vec3, end: Vec3, point: Vec3) -> Vec3 {
    let segment = end - start;
    let t = (point - start).dot(segment) / (segment.length_squared() + 1.0e-6);
    start + segment * t.clamp(0.0, 1.0)
}

fn half_length_and_direction(start: Vec3, end: Vec3) -> (f32, Vec3) {
    let vector = end - start;
    let length = vector.length();
    (0.5 * length, if length > 0.0 { vector / length } else { vector })
}

/// Return the point between `start` and `end` that is closest to the other segment.
fn closest_point_to_segment(start: Vec3, end: Vec3, other_start: Vec3, other_end: Vec3) -> Vec3 {
    let (half_length, direction) = half_length_and_direction(start, end);
    let (other_half_length, other_direction) = half_length_and_direction(other_start, other_end);
    let center = start + direction * half_length;
    let other_center = other_start + other_direction * other_half_length;
    let offset = center - other_center;
    let cosine = direction.dot(other_direction);
    let along = direction.dot(offset);
    let other_along = other_direction.dot(offset);
    // The bias keeps parallel segments finite. Any point where they overlap is closest.
    let t = (cosine * other_along - along) / (1.0 - cosine * cosine + 1.0e-6);
    let other_t = other_along + cosine * t;
    let point = center + direction * t.clamp(-half_length, half_length);
    let other_point = other_center + other_direction * other_t.clamp(-other_half_length, other_half_length);
    // Clamping can leave one point off the closest pair. Project each point onto the
    // other segment, and keep the pair that ends up closer.
    let projected = closest_point_on_segment(start, end, other_point);
    let other_projected = closest_point_on_segment(other_start, other_end, point);
    if (other_point - projected).length_squared() < (point - other_projected).length_squared() {
        return projected;
    }
    point
}

fn glam_quat(q: rtm::Quat) -> glam::Quat {
    glam::Quat::from_xyzw(q[0], q[1], q[2], q[3])
}

fn rt_quat(q: glam::Quat) -> rtm::Quat {
    [q.x, q.y, q.z, q.w]
}

fn document_transform(transform: documents::Transform) -> Transform {
    Transform::new(to_rt(transform.translation), rt_quat(transform.rotation_xyzw))
}

/// Target nodes in playback order, with their approach settings (upstream
/// `SmartPrimitiveSequence`).
#[derive(Clone, Debug)]
pub struct SmartPrimitiveSequence {
    pub name: String,
    /// Target nodes in playback order.
    pub node_indices: Vec<usize>,
    /// Authored speed of the first target pose, in m/s.
    pub start_speed: f32,
    /// Node whose pivot aligns the sequence, or -1 without one.
    pub pivot_node_index: i64,
}

/// Sampled poses and metadata used to build a runtime asset (upstream
/// `SmartPrimitiveData`). Node zero is the root node; each parent comes before its
/// children.
pub struct SmartPrimitiveData<E> {
    pub rig: rt::ModelRig,
    pub node_local_transforms: Vec<Transform>,
    pub node_parent_indices: Vec<i32>,
    pub node_names: Vec<String>,
    pub target_node_indices: Vec<usize>,
    /// Four poses per target, in target order.
    pub target_poses: Vec<Vec<E>>,
    pub target_inference_params: Vec<rt::TargetPoseInferenceParams>,
    pub box_node_indices: Vec<usize>,
    pub box_half_extents: Vec<Vec3>,
    pub pivot_node_indices: Vec<usize>,
    pub pivots: Vec<Pivot>,
    pub sequences: Vec<SmartPrimitiveSequence>,
}

/// Immutable Smart Primitive data shared by runtime instances (upstream
/// `SmartPrimitiveAsset`). The poses' element type is a joint [`Transform`] (SOMA) or a
/// joint coordinate (G1).
pub struct SmartPrimitiveAsset<E> {
    rig: rt::ModelRig,
    node_names: Vec<String>,
    node_indices_by_name: HashMap<String, usize>,
    node_local_transforms: Vec<Transform>,
    node_parent_indices: Vec<i32>,
    target_poses: Vec<Vec<E>>,
    target_node_indices: Vec<usize>,
    target_inference_params: Vec<rt::TargetPoseInferenceParams>,
    target_index_by_node: HashMap<usize, usize>,
    pivot_node_indices: Vec<usize>,
    pivot_index_by_node: HashMap<usize, usize>,
    pivots: Vec<Pivot>,
    sequences: Vec<SmartPrimitiveSequence>,
    sequence_indices_by_name: HashMap<String, usize>,
    max_sequence_length: usize,
    box_node_indices: Vec<usize>,
    box_index_by_node: HashMap<usize, usize>,
    box_half_extents: Vec<Vec3>,
}

impl<E: PoseElem> SmartPrimitiveAsset<E> {
    pub fn new(data: SmartPrimitiveData<E>) -> Self {
        let node_indices_by_name = data.node_names.iter().enumerate().map(|(index, name)| (name.clone(), index)).collect();
        let target_index_by_node = data.target_node_indices.iter().enumerate().map(|(index, &node)| (node, index)).collect();
        let pivot_index_by_node = data.pivot_node_indices.iter().enumerate().map(|(index, &node)| (node, index)).collect();
        let sequence_indices_by_name = data.sequences.iter().enumerate().map(|(index, sequence)| (sequence.name.clone(), index)).collect();
        let max_sequence_length = data.sequences.iter().map(|sequence| sequence.node_indices.len()).max().unwrap_or(0);
        let box_index_by_node = data.box_node_indices.iter().enumerate().map(|(index, &node)| (node, index)).collect();
        SmartPrimitiveAsset {
            rig: data.rig,
            node_names: data.node_names,
            node_indices_by_name,
            node_local_transforms: data.node_local_transforms,
            node_parent_indices: data.node_parent_indices,
            target_poses: data.target_poses,
            target_node_indices: data.target_node_indices,
            target_inference_params: data.target_inference_params,
            target_index_by_node,
            pivot_node_indices: data.pivot_node_indices,
            pivot_index_by_node,
            pivots: data.pivots,
            sequences: data.sequences,
            sequence_indices_by_name,
            max_sequence_length,
            box_node_indices: data.box_node_indices,
            box_index_by_node,
            box_half_extents: data.box_half_extents,
        }
    }

    /// Prepare the hierarchy, sampled poses, and interaction settings for playback
    /// (upstream `build_runtime_smart_primitive_data`).
    ///
    /// `document_path` resolves the target poses' clip references; `load_clip` loads a
    /// clip for this asset's rig (the rig module's clip registry in the bins). Every
    /// target pose samples [`POSES_PER_TOKEN`] poses at the model's sample rate from
    /// its clip, starting at its sample time; only the roots are moved into runtime
    /// space by `model_to_runtime`.
    pub fn build_from_document(
        document: &SmartPrimitiveDocument,
        document_path: &Path,
        model_to_runtime: Transform,
        mut load_clip: impl FnMut(&Path) -> Result<Rc<dyn MotionBuffer<Elem = E>>>,
    ) -> Result<Self> {
        let rig = match document.rig {
            documents::ModelRig::Soma => rt::ModelRig::Soma,
            documents::ModelRig::UnitreeG1 => rt::ModelRig::UnitreeG1,
        };
        // Model-space up axis: Y for SOMA, Z for G1 (the root height a constrained
        // hips target keeps).
        let up_index = match rig {
            rt::ModelRig::Soma => 1,
            rt::ModelRig::UnitreeG1 => 2,
        };

        // Runtime transforms require parents before children; `iter_nodes` guarantees it.
        let nodes = document.iter_nodes();
        let node_indices: HashMap<u32, usize> = nodes.iter().enumerate().map(|(index, node)| (node.id, index)).collect();
        let node_names: Vec<String> = nodes.iter().map(|node| node.name.clone()).collect();
        if node_names.iter().collect::<std::collections::HashSet<_>>().len() != node_names.len() {
            bail!("Runtime node names must be unique.");
        }
        let node_local_transforms: Vec<Transform> = nodes.iter().map(|node| document_transform(node.local_transform)).collect();
        let node_parent_indices: Vec<i32> = nodes
            .iter()
            .map(|node| document.find_parent(node.id).map_or(-1, |parent| node_indices[&parent.id] as i32))
            .collect();

        // Targets: four sampled poses per target, roots moved to runtime space.
        let targets = document.target_poses();
        let mut target_node_indices = Vec::with_capacity(targets.len());
        let mut target_poses: Vec<Vec<E>> = Vec::with_capacity(targets.len() * POSES_PER_TOKEN);
        let mut target_inference_params = Vec::with_capacity(targets.len());
        let mut target_velocities: HashMap<u32, f32> = HashMap::new();
        for (node_id, component) in &targets {
            let node_name = &document.find_node(*node_id).expect("target_poses returns existing nodes").name;
            let Some(clip_path) = component.clip_path(document_path) else {
                bail!("Target Pose on node {node_name:?} has no clip.");
            };
            let animation = load_clip(&clip_path)?;
            let mut sample_time = match component.playback_mode {
                PlaybackMode::Frames => component.frame_index as f32 / animation.sample_rate(),
                PlaybackMode::Time => component.sample_time_seconds,
            };
            let duration = (animation.num_frames().saturating_sub(1)).max(0) as f32 / animation.sample_rate();
            sample_time = sample_time.clamp(0.0, duration);

            let mut root_offset = Transform::IDENTITY;
            if component.hips_constrained {
                // Remove the root's offset along the ground, keeping its height.
                let first = animation.sample_at(sample_time as f64, Transform::IDENTITY)?;
                let mut offset = -E::root_of(&first).p;
                offset.0[up_index] = 0.0;
                root_offset = Transform::new(offset, rtm::QUAT_IDENTITY);
            }
            let mut sampled: Vec<Vec<E>> = Vec::with_capacity(POSES_PER_TOKEN);
            for frame in 0..POSES_PER_TOKEN {
                sampled.push(animation.sample_at(sample_time as f64 + frame as f64 / rt::MODEL_SAMPLE_RATE as f64, root_offset)?);
            }
            for pose in &sampled {
                if !E::values_are_finite(pose) {
                    bail!("Target Pose on node {node_name:?} contains non-finite values.");
                }
            }
            // Root velocity between the first two sampled poses, in m/s. It is zero
            // when only one pose is used.
            let velocity = if component.num_poses_to_use == 1 {
                0.0
            } else {
                ((E::root_of(&sampled[1]).p - E::root_of(&sampled[0]).p) * rt::MODEL_SAMPLE_RATE).length()
            };
            target_velocities.insert(*node_id, velocity);

            for pose in &mut sampled {
                let root = model_to_runtime * E::root_of(pose);
                E::write_root_of(pose, root);
            }
            target_poses.extend(sampled);
            target_node_indices.push(node_indices[node_id]);
            target_inference_params.push(rt::TargetPoseInferenceParams {
                transition_early_by_frames: component.transition_early_by_frames as usize,
                num_poses_to_use: component.num_poses_to_use as usize,
                ik_pin_mask: component.ik_pin_mask,
                allowed_token_counts: Some((component.min_tokens as i32..=component.max_tokens as i32).collect()),
            });
        }

        let boxes = document.interaction_boxes();
        let box_node_indices: Vec<usize> = boxes.iter().map(|(node_id, _)| node_indices[node_id]).collect();
        let box_half_extents: Vec<Vec3> = boxes.iter().map(|(_, component)| 0.5 * Vec3::new(component.width, component.depth, component.height)).collect();

        let pivots = document.pivots();
        let pivot_node_indices: Vec<usize> = pivots.iter().map(|(node_id, _)| node_indices[node_id]).collect();
        let pivot_rules: Vec<Pivot> = pivots.iter().map(|(_, component)| Pivot::from_component(component)).collect::<Result<_>>()?;

        let mut sequences = Vec::with_capacity(document.sequences.len());
        for sequence in &document.sequences {
            if sequence.node_ids.is_empty() {
                bail!("Sequence {:?} must contain at least one Target Pose.", sequence.name);
            }
            let start_speed = target_velocities
                .get(&sequence.node_ids[0])
                .copied()
                .filter(|speed| speed.is_finite())
                .ok_or_else(|| anyhow!("Sequence {:?} has a non-finite start speed.", sequence.name))?;
            sequences.push(SmartPrimitiveSequence {
                name: sequence.name.clone(),
                node_indices: sequence.node_ids.iter().map(|node_id| node_indices[node_id]).collect(),
                start_speed,
                pivot_node_index: sequence.pivot_node_id.map_or(-1, |node_id| node_indices[&node_id] as i64),
            });
        }

        Ok(Self::new(SmartPrimitiveData {
            rig,
            node_local_transforms,
            node_parent_indices,
            node_names,
            target_node_indices,
            target_poses,
            target_inference_params,
            box_node_indices,
            box_half_extents,
            pivot_node_indices,
            pivots: pivot_rules,
            sequences,
        }))
    }

    /// Rig of the asset's poses.
    pub fn rig(&self) -> rt::ModelRig {
        self.rig
    }
}

/// A runtime instance with its own hierarchy, backed by shared asset data (upstream
/// `SmartPrimitive`).
pub struct SmartPrimitive<E> {
    asset: Rc<SmartPrimitiveAsset<E>>,
    node_local_transforms: Vec<Transform>,
    node_world_transforms: Vec<Transform>,
    authored_pivot_transforms: Option<HashMap<usize, Transform>>,
    world_transforms_dirty: bool,
}

impl<E: PoseElem> SmartPrimitive<E> {
    /// Create an instance with its own copy of the node transforms.
    pub fn new(asset: Rc<SmartPrimitiveAsset<E>>) -> Self {
        let node_local_transforms = asset.node_local_transforms.clone();
        SmartPrimitive {
            node_world_transforms: node_local_transforms.clone(),
            node_local_transforms,
            asset,
            authored_pivot_transforms: None,
            world_transforms_dirty: true,
        }
    }

    /// Data shared with the other instances of the same document.
    pub fn asset(&self) -> &Rc<SmartPrimitiveAsset<E>> {
        &self.asset
    }

    /// Rig of the asset's poses.
    pub fn rig(&self) -> rt::ModelRig {
        self.asset.rig
    }

    /// Number of nodes, including the root node.
    pub fn node_count(&self) -> usize {
        self.asset.node_names.len()
    }

    /// Number of nodes with a target pose.
    pub fn target_pose_count(&self) -> usize {
        self.asset.target_node_indices.len()
    }

    /// Number of authored sequences.
    pub fn sequence_count(&self) -> usize {
        self.asset.sequences.len()
    }

    /// Most steps in any sequence, or 0 without sequences.
    pub fn max_sequence_length(&self) -> usize {
        self.asset.max_sequence_length
    }

    /// Number of interaction boxes.
    pub fn interaction_box_count(&self) -> usize {
        self.asset.box_node_indices.len()
    }

    /// Nodes with a pivot, independent of their display names.
    pub fn pivot_node_indices(&self) -> &[usize] {
        &self.asset.pivot_node_indices
    }

    /// A copy of the node's pivot rules.
    pub fn pivot(&self, node_index: usize) -> Pivot {
        self.asset.pivots[self.asset.pivot_index_by_node[&node_index]].clone()
    }

    /// Values in one target pose: joints for SOMA, or joint coordinates for G1.
    pub fn pose_size(&self) -> usize {
        self.asset.target_poses.first().map_or(0, |pose| pose.len())
    }

    /// Current local transform of each node. Change them with the setter methods.
    pub fn node_local_transforms(&self) -> &[Transform] {
        &self.node_local_transforms
    }

    /// Each node's parent index, or -1 for the root node.
    pub fn node_parent_indices(&self) -> &[i32] {
        &self.asset.node_parent_indices
    }

    /// The node's display name.
    pub fn node_name(&self, node_index: usize) -> &str {
        &self.asset.node_names[node_index]
    }

    /// The node index for `name`, or -1 when it is absent.
    pub fn node_index(&self, name: &str) -> i64 {
        self.asset.node_indices_by_name.get(name).map_or(-1, |&index| index as i64)
    }

    /// The sequence index for `name`, or -1 when it is absent.
    pub fn sequence_index(&self, name: &str) -> i64 {
        self.asset.sequence_indices_by_name.get(name).map_or(-1, |&index| index as i64)
    }

    /// The sequence's name.
    pub fn sequence_name(&self, sequence_index: usize) -> &str {
        &self.asset.sequences[sequence_index].name
    }

    /// The number of steps in the sequence.
    pub fn sequence_length(&self, sequence_index: usize) -> usize {
        self.asset.sequences[sequence_index].node_indices.len()
    }

    /// The authored speed of the sequence's first target pose, in m/s.
    pub fn sequence_start_speed(&self, sequence_index: usize) -> f32 {
        self.asset.sequences[sequence_index].start_speed
    }

    /// The sequence's pivot node, or -1 when it has none.
    pub fn sequence_pivot_node_index(&self, sequence_index: usize) -> i64 {
        self.asset.sequences[sequence_index].pivot_node_index
    }

    /// The display names of the sequence's target nodes, in playback order.
    pub fn target_names_for_sequence(&self, sequence_index: usize) -> Vec<&str> {
        self.asset.sequences[sequence_index].node_indices.iter().map(|&node| self.asset.node_names[node].as_str()).collect()
    }

    /// Whether the node has a target pose.
    pub fn has_target_pose(&self, node_index: usize) -> bool {
        self.asset.target_index_by_node.contains_key(&node_index)
    }

    /// Whether the node has an interaction box.
    pub fn has_interaction_box(&self, node_index: usize) -> bool {
        self.asset.box_index_by_node.contains_key(&node_index)
    }

    /// Set the node's transform relative to its parent. Its children move with it.
    pub fn set_node_local_transform(&mut self, node_index: usize, transform: Transform) {
        assert!(node_index < self.node_count(), "node index out of range");
        self.node_local_transforms[node_index] = transform;
        self.world_transforms_dirty = true;
    }

    /// Place the node in world space. Its children move with it.
    pub fn set_node_world_transform(&mut self, node_index: usize, transform: Transform) {
        assert!(node_index < self.node_count(), "node index out of range");
        self.ensure_world_transforms();
        let parent = self.asset.node_parent_indices[node_index];
        self.node_local_transforms[node_index] = if parent < 0 { transform } else { self.node_world_transforms[parent as usize].inverse() * transform };
        self.world_transforms_dirty = true;
    }

    fn ensure_world_transforms(&mut self) {
        if self.world_transforms_dirty {
            rtm::compute_world_poses_into(&self.node_local_transforms, &self.asset.node_parent_indices, Transform::IDENTITY, &mut self.node_world_transforms);
            self.world_transforms_dirty = false;
        }
    }

    /// Every node's world transform, parents before children.
    pub fn compute_world_transforms(&mut self) -> &[Transform] {
        self.ensure_world_transforms();
        &self.node_world_transforms
    }

    fn get_target_poses(&self, node_index: usize, node_transforms: &[Transform]) -> Result<(Vec<Vec<E>>, rt::TargetPoseInferenceParams)> {
        let Some(&target_index) = self.asset.target_index_by_node.get(&node_index) else {
            bail!("node {node_index} has no target pose");
        };
        let offset = target_index * POSES_PER_TOKEN;
        let node_transform = node_transforms[node_index];
        let mut poses = self.asset.target_poses[offset..offset + POSES_PER_TOKEN].to_vec();
        // Joint 0 is the root, which the node places. Joint values are unchanged.
        for pose in &mut poses {
            let root = node_transform * E::root_of(pose);
            E::write_root_of(pose, root);
        }
        Ok((poses, self.asset.target_inference_params[target_index].clone()))
    }

    /// The node's four poses in its parent's space, and their settings.
    pub fn get_local_target_poses(&self, node_index: usize) -> Result<(Vec<Vec<E>>, rt::TargetPoseInferenceParams)> {
        self.get_target_poses(node_index, &self.node_local_transforms)
    }

    /// The node's four poses in world space, and their settings.
    pub fn get_world_target_poses(&mut self, node_index: usize) -> Result<(Vec<Vec<E>>, rt::TargetPoseInferenceParams)> {
        self.ensure_world_transforms();
        let world = self.node_world_transforms.clone();
        self.get_target_poses(node_index, &world)
    }

    /// Move and turn the sequence's pivot node to match the character's approach.
    ///
    /// Call it before reading the sequence's targets. A sequence without a pivot keeps
    /// its placement. The first call records the world transform of every pivot, and
    /// each alignment measures its limits from that record. Repeated alignments never
    /// drift, even when one pivot is a child of another.
    pub fn align_sequence_pivot(&mut self, sequence_index: usize, character: &CharacterProperties) -> Result<()> {
        let node_index = self.sequence_pivot_node_index(sequence_index);
        if node_index < 0 {
            return Ok(());
        }
        let node_index = node_index as usize;
        self.ensure_world_transforms();
        if self.authored_pivot_transforms.is_none() {
            self.authored_pivot_transforms =
                Some(self.asset.pivot_node_indices.iter().map(|&node| (node, self.node_world_transforms[node])).collect());
        }
        let authored = self.authored_pivot_transforms.as_ref().expect("recorded above")[&node_index];
        let current = self.node_world_transforms[node_index];
        let rules = &self.asset.pivots[self.asset.pivot_index_by_node[&node_index]];
        let aligned = rules.aligned_transform(authored, current, character);
        self.set_node_world_transform(node_index, aligned);
        Ok(())
    }

    /// Write one sequence in world space: four poses per step, in step order, with the
    /// settings of each step.
    pub fn get_world_target_poses_for_sequence(&mut self, sequence_index: usize) -> Result<(Vec<Vec<E>>, Vec<rt::TargetPoseInferenceParams>)> {
        if sequence_index >= self.sequence_count() {
            bail!("Sequence index {sequence_index} is out of range.");
        }
        let node_indices = self.asset.sequences[sequence_index].node_indices.clone();
        self.ensure_world_transforms();
        let world = self.node_world_transforms.clone();
        let mut poses = Vec::with_capacity(node_indices.len() * POSES_PER_TOKEN);
        let mut params = Vec::with_capacity(node_indices.len());
        for node_index in node_indices {
            let (target_poses, target_params) = self.get_target_poses(node_index, &world)?;
            poses.extend(target_poses);
            params.push(target_params);
        }
        Ok((poses, params))
    }

    /// Every interaction box in world space, ready for overlap tests, in asset order.
    pub fn get_world_interaction_boxes(&mut self) -> Vec<InteractionBox> {
        self.ensure_world_transforms();
        (0..self.interaction_box_count())
            .map(|box_index| InteractionBox {
                transform: self.node_world_transforms[self.asset.box_node_indices[box_index]],
                half_extents: self.asset.box_half_extents[box_index],
            })
            .collect()
    }

    /// The node's interaction box in world space.
    pub fn get_world_interaction_box(&mut self, node_index: usize) -> Result<InteractionBox> {
        let Some(&box_index) = self.asset.box_index_by_node.get(&node_index) else {
            bail!("node {node_index} has no interaction box");
        };
        self.ensure_world_transforms();
        Ok(InteractionBox { transform: self.node_world_transforms[node_index], half_extents: self.asset.box_half_extents[box_index] })
    }
}

/// What a behavior may ask the interaction state to do, upstream's
/// `SmartPrimitiveBehavior.interaction_state`.
///
/// The requests are queued and applied by the interaction state once the behavior's
/// call returns, so a behavior never borrows the state that is running it.
pub trait InteractionCtl<E> {
    /// Generate motion through a target pose sequence, then run `on_complete`.
    ///
    /// Without a name, use the targets returned by the behavior's `get_targets`.
    /// Without a callback, return to locomotion or idle when the sequence ends.
    fn play_target_pose_sequence(&mut self, name: Option<&str>, on_complete: Option<Completion<E>>);
    /// Play a clip once, or keep repeating it when `loop_` is set. When a clip played
    /// once ends, `on_complete` runs; without a callback, return to locomotion or idle.
    fn play_clip(&mut self, path: &Path, loop_: bool, align_to_current_pose: bool, on_complete: Option<Completion<E>>);
    /// End the interaction and return to locomotion or idle.
    fn finish(&mut self);
    /// Whether a looping clip is playing.
    fn is_looping(&self) -> bool;
}

/// Runs when a sequence or clip completes: the behavior and its control handle.
pub type Completion<E> = Box<dyn FnOnce(&mut dyn SmartPrimitiveBehavior<E>, &mut dyn InteractionCtl<E>) -> Result<()>>;

/// Define how a character interacts with a Smart Primitive (upstream
/// `SmartPrimitiveBehavior`).
///
/// Scenes choose an implementation by its upstream class name through
/// [`BehaviorRegistry`]. Override [`SmartPrimitiveBehavior::enter`] to start a target
/// pose sequence or a clip, and [`SmartPrimitiveBehavior::interact`] to respond when
/// the user interacts again. By default, the character follows the targets from
/// `get_targets`, then returns to locomotion or idle.
pub trait SmartPrimitiveBehavior<E> {
    /// The runtime instance this behavior drives.
    fn primitive(&self) -> &SmartPrimitive<E>;
    fn primitive_mut(&mut self) -> &mut SmartPrimitive<E>;
    /// Sequences the document must contain.
    fn required_sequences(&self) -> &'static [&'static str] {
        &[]
    }
    /// Clips the interaction state loads before an interaction needs them.
    fn clip_paths(&self) -> Vec<PathBuf> {
        Vec::new()
    }
    /// Start the interaction. The default plays the sequence that `get_targets` chooses.
    fn enter(&mut self, ctl: &mut dyn InteractionCtl<E>) -> Result<()> {
        ctl.play_target_pose_sequence(None, None);
        Ok(())
    }
    /// Respond when the user interacts again before this behavior has finished.
    fn interact(&mut self, _ctl: &mut dyn InteractionCtl<E>) {}
    /// Write an interaction's targets and return their generation settings, in the
    /// same order. The base trait returns no targets.
    fn get_targets(&mut self, _properties: &CharacterProperties) -> Result<(Vec<Vec<E>>, Vec<rt::TargetPoseInferenceParams>)> {
        Ok((Vec::new(), Vec::new()))
    }
}

/// Check the sequences a behavior requires, upstream's `SmartPrimitiveBehavior.__init__`.
fn check_required_sequences<E: PoseElem>(primitive: &SmartPrimitive<E>, required: &[&str], behavior_name: &str) -> Result<()> {
    for name in required {
        if primitive.sequence_index(name) < 0 {
            bail!("{behavior_name} requires sequence {name:?}.");
        }
    }
    Ok(())
}

/// Pick the sequence closest to the character's speed, then align its pivot (upstream
/// `VelocityMatchingSequenceBehavior`).
pub struct VelocityMatchingSequenceBehavior<E> {
    primitive: SmartPrimitive<E>,
}

impl<E: PoseElem> VelocityMatchingSequenceBehavior<E> {
    /// The document needs at least one sequence.
    pub fn new(asset: Rc<SmartPrimitiveAsset<E>>) -> Result<Self> {
        let primitive = SmartPrimitive::new(asset);
        if primitive.sequence_count() == 0 {
            bail!("Velocity Matching Sequence behavior requires at least one Target Pose sequence.");
        }
        check_required_sequences(&primitive, Self::REQUIRED, "VelocityMatchingSequenceBehavior")?;
        Ok(VelocityMatchingSequenceBehavior { primitive })
    }

    const REQUIRED: &'static [&'static str] = &[];
}

impl<E: PoseElem> SmartPrimitiveBehavior<E> for VelocityMatchingSequenceBehavior<E> {
    fn primitive(&self) -> &SmartPrimitive<E> {
        &self.primitive
    }
    fn primitive_mut(&mut self) -> &mut SmartPrimitive<E> {
        &mut self.primitive
    }

    fn get_targets(&mut self, properties: &CharacterProperties) -> Result<(Vec<Vec<E>>, Vec<rt::TargetPoseInferenceParams>)> {
        let character_speed = properties.velocity.length();
        // A tie keeps the first sequence.
        let mut sequence_index = 0;
        let mut best = f32::INFINITY;
        for index in 0..self.primitive.sequence_count() {
            let delta = (self.primitive.sequence_start_speed(index) - character_speed).abs();
            if delta < best {
                best = delta;
                sequence_index = index;
            }
        }
        self.primitive.align_sequence_pivot(sequence_index, properties)?;
        self.primitive.get_world_target_poses_for_sequence(sequence_index)
    }
}

/// Sit down, stay seated, and stand up when the user interacts again (upstream
/// `SittingBehavior`).
pub struct SittingBehavior<E> {
    primitive: SmartPrimitive<E>,
    /// The rig's looping sitting clip.
    sitting_clip: PathBuf,
}

impl<E: PoseElem> SittingBehavior<E> {
    pub fn new(asset: Rc<SmartPrimitiveAsset<E>>, sitting_clip: impl Into<PathBuf>) -> Result<Self> {
        let primitive = SmartPrimitive::new(asset);
        check_required_sequences(&primitive, Self::REQUIRED, "SittingBehavior")?;
        Ok(SittingBehavior { primitive, sitting_clip: sitting_clip.into() })
    }

    const REQUIRED: &'static [&'static str] = &["Sit", "Stand"];
}

impl<E: PoseElem> SmartPrimitiveBehavior<E> for SittingBehavior<E> {
    fn primitive(&self) -> &SmartPrimitive<E> {
        &self.primitive
    }
    fn primitive_mut(&mut self) -> &mut SmartPrimitive<E> {
        &mut self.primitive
    }

    fn required_sequences(&self) -> &'static [&'static str] {
        Self::REQUIRED
    }

    fn clip_paths(&self) -> Vec<PathBuf> {
        vec![self.sitting_clip.clone()]
    }

    fn enter(&mut self, ctl: &mut dyn InteractionCtl<E>) -> Result<()> {
        let sitting_clip = self.sitting_clip.clone();
        ctl.play_target_pose_sequence(
            Some("Sit"),
            Some(Box::new(move |_behavior, ctl| {
                ctl.play_clip(&sitting_clip, true, true, None);
                Ok(())
            })),
        );
        Ok(())
    }

    fn interact(&mut self, ctl: &mut dyn InteractionCtl<E>) {
        if ctl.is_looping() {
            ctl.play_target_pose_sequence(Some("Stand"), None);
        }
    }
}

/// Behaviors scenes choose by upstream class name, upstream's `behavior_types()`.
///
/// The registry knows the two upstream behaviors ("VelocityMatchingSequenceBehavior",
/// "SittingBehavior"); apps can register their own under the same string keys the
/// scene documents use. `sitting_clip` is the rig's looping sitting clip, from the
/// rig's presets.
pub struct BehaviorRegistry<E> {
    factories: HashMap<String, Box<dyn Fn(Rc<SmartPrimitiveAsset<E>>) -> Result<Box<dyn SmartPrimitiveBehavior<E>>>>>,
}

impl<E: PoseElem + 'static> BehaviorRegistry<E> {
    /// A registry with the upstream behaviors.
    pub fn new(sitting_clip: impl Into<PathBuf>) -> Self {
        let mut registry = BehaviorRegistry { factories: HashMap::new() };
        registry.register("VelocityMatchingSequenceBehavior", |asset| Ok(Box::new(VelocityMatchingSequenceBehavior::new(asset)?)));
        let sitting_clip = sitting_clip.into();
        registry.register("SittingBehavior", move |asset| Ok(Box::new(SittingBehavior::new(asset, sitting_clip.clone())?)));
        registry
    }

    /// Register a behavior under the class name scene documents reference.
    pub fn register(&mut self, name: &str, factory: impl Fn(Rc<SmartPrimitiveAsset<E>>) -> Result<Box<dyn SmartPrimitiveBehavior<E>>> + 'static) {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// The registered behavior names.
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(String::as_str).collect()
    }

    /// Instantiate the behavior `name` for a Smart Primitive asset.
    pub fn create(&self, name: &str, asset: Rc<SmartPrimitiveAsset<E>>) -> Result<Box<dyn SmartPrimitiveBehavior<E>>> {
        let Some(factory) = self.factories.get(name) else {
            bail!("uses unknown behavior '{name}'");
        };
        factory(asset)
    }
}
