// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Smart Primitive documents (`.smart_primitive.json`), ported from upstream
//! `app/runtime/common/io/smart_primitive_document.py`.
//!
//! This is the authoring document layer only: the editable node hierarchy,
//! components, and target pose sequences. Clip loading, pose sampling, and the
//! runtime asset preparation are engine concerns and are out of scope here.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::transform::Transform;
use super::{
    CameraState, DocumentError, ModelRig, Result, atomic_write, check_document_header,
    document_name_from_path, normalize_document_path, parse_json, read_document_text,
    resolve_reference_path, serialize_pretty, validate_color,
};

pub const SMART_PRIMITIVE_FORMAT: &str = "motionbricks.smart_primitive.authoring";
pub const SMART_PRIMITIVE_SCHEMA_VERSION: i64 = 1;
pub const SMART_PRIMITIVE_SUFFIX: &str = ".smart_primitive.json";

// Model constants from upstream `motionbricks/model/contract.py`.
pub const MODEL_POSES_PER_TOKEN: u32 = 4;
pub const MODEL_MIN_TOKEN_COUNT: u32 = 6;
pub const MODEL_MAX_TOKEN_COUNT: u32 = 16;

/// Upstream `MAX_TRANSITION_EARLY_BY_FRAMES` in `motionbricks/runtime/target_pose.py`.
pub const MAX_TRANSITION_EARLY_BY_FRAMES: u32 = 20;

/// Upstream `MIN_TRAJECTORY_SCALE` / `MAX_TRAJECTORY_SCALE`.
pub const MIN_TRAJECTORY_SCALE: f32 = 1.0;
pub const MAX_TRAJECTORY_SCALE: f32 = 4.0;

/// All bits of upstream's `IKPin.ALL`: both hands and both feet.
pub const IK_PIN_ALL_MASK: u32 = 0b1111;

/// How the editor shows a target pose's sample position: as a time or as a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackMode {
    Time,
    Frames,
}

/// Local axis of a pivot rule. `None` turns the rule off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    None,
    X,
    Y,
    Z,
}

/// Limit of a pivot rule. Translation allows `LineProjection`, rotation allows `Cone`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintType {
    Unconstrained,
    Cone,
    LineProjection,
}

/// Where the rotation cone points, relative to the motion direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConeSide {
    Forward,
    Mirrored,
    Both,
}

/// Order in which a pivot applies its translation and rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationOrder {
    TranslationThenRotation,
    RotationThenTranslation,
}

/// A target pose's settings and the clip sample used to build it.
///
/// `clip` is stored exactly as written in the file (relative to the document);
/// use [`TargetPoseComponent::clip_path`] to resolve it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetPoseComponent {
    pub id: u32,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub clip: Option<String>,
    pub sample_time_seconds: f32,
    pub frame_index: u32,
    pub playback_mode: PlaybackMode,
    pub hips_constrained: bool,
    pub num_poses_to_use: u32,
    pub transition_early_by_frames: u32,
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub ik_pin_mask: u32,
    pub show_mesh: bool,
    pub show_skeletons: bool,
    pub show_velocity: bool,
    pub show_trajectory: bool,
    pub trajectory_scale: f32,
    pub color: [f32; 3],
}

impl TargetPoseComponent {
    /// Resolve the clip path relative to the document file, upstream's clip
    /// reference rule (`resolve_reference()` with `kind="clip"`). `None` when
    /// the target pose has no clip.
    pub fn clip_path(&self, document_path: impl AsRef<Path>) -> Option<PathBuf> {
        self.clip
            .as_deref()
            .map(|clip| resolve_reference_path(clip, document_path.as_ref()))
    }
}

/// An interaction box's size and color. Its node sets its placement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionBoxComponent {
    pub id: u32,
    pub width: f32,
    pub depth: f32,
    pub height: f32,
    pub color: [f32; 3],
}

/// Pivot rules and preview visibility for a hierarchy node, upstream's
/// `PivotComponent` wrapping `motionbricks/runtime/pivot.py`'s `Pivot`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PivotComponent {
    pub id: u32,
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
    pub show_motion_direction: bool,
    pub show_translation_constraint: bool,
    pub show_rotation_constraint: bool,
}

/// A component attached to a [`SceneNode`]. A node holds at most one component
/// of each kind; validation enforces it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeComponent {
    #[serde(rename = "target_pose")]
    TargetPose(TargetPoseComponent),
    #[serde(rename = "interaction_box")]
    InteractionBox(InteractionBoxComponent),
    #[serde(rename = "pivot")]
    Pivot(PivotComponent),
}

impl NodeComponent {
    pub fn id(&self) -> u32 {
        match self {
            Self::TargetPose(component) => component.id,
            Self::InteractionBox(component) => component.id,
            Self::Pivot(component) => component.id,
        }
    }
}

/// A node's local transform, components, and children.
///
/// `euler_degrees` is required in the JSON but may be `null` (a node that was
/// never shown in the transform editor); `rotation_xyzw` is authoritative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: u32,
    pub name: String,
    pub local_transform: Transform,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub euler_degrees: Option<Vec3>,
    pub visible: bool,
    pub translation_locks: [bool; 3],
    pub rotation_locks: [bool; 3],
    pub components: Vec<NodeComponent>,
    pub children: Vec<SceneNode>,
}

impl SceneNode {
    /// The node's Target Pose component, when it has one.
    pub fn target_pose(&self) -> Option<&TargetPoseComponent> {
        self.components.iter().find_map(|component| match component {
            NodeComponent::TargetPose(component) => Some(component),
            _ => None,
        })
    }

    /// The node's Interaction Box component, when it has one.
    pub fn interaction_box(&self) -> Option<&InteractionBoxComponent> {
        self.components.iter().find_map(|component| match component {
            NodeComponent::InteractionBox(component) => Some(component),
            _ => None,
        })
    }

    /// The node's Pivot component, when it has one.
    pub fn pivot(&self) -> Option<&PivotComponent> {
        self.components.iter().find_map(|component| match component {
            NodeComponent::Pivot(component) => Some(component),
            _ => None,
        })
    }
}

/// Targets in generation order, with an optional pivot to align them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetPoseSequence {
    pub id: u32,
    pub name: String,
    pub node_ids: Vec<u32>,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub pivot_node_id: Option<u32>,
}

/// A sequence with its node IDs resolved, the document layer's half of
/// upstream's `_prepare_runtime_sequences()`: the ordered Target Pose components
/// and the optional Pivot component.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSequence<'a> {
    pub id: u32,
    pub name: &'a str,
    pub target_poses: Vec<(u32, &'a TargetPoseComponent)>,
    pub pivot: Option<(u32, &'a PivotComponent)>,
}

/// The editable hierarchy, components, and target pose sequences, plus the
/// editor selection and camera stored in the document header.
#[derive(Debug, Clone, PartialEq)]
pub struct SmartPrimitiveDocument {
    pub name: String,
    pub rig: ModelRig,
    pub selected_node_id: u32,
    pub camera: CameraState,
    pub sequences: Vec<TargetPoseSequence>,
    pub root: SceneNode,
}

#[derive(Deserialize)]
struct SmartPrimitiveFile {
    format: String,
    schema_version: i64,
    name: String,
    rig: String,
    editor: SmartPrimitiveEditorIn,
    sequences: Vec<TargetPoseSequence>,
    root: SceneNode,
}

#[derive(Deserialize)]
struct SmartPrimitiveEditorIn {
    selected_node_id: i64,
    camera: CameraState,
}

/// Only the header, for [`SmartPrimitiveDocument::load_rig`].
#[derive(Deserialize)]
struct RigHeader {
    format: String,
    schema_version: i64,
    rig: String,
}

#[derive(Serialize)]
struct SmartPrimitiveFileOut<'a> {
    format: &'static str,
    schema_version: i64,
    name: &'a str,
    rig: &'static str,
    editor: SmartPrimitiveEditorOut<'a>,
    sequences: &'a [TargetPoseSequence],
    root: &'a SceneNode,
}

#[derive(Serialize)]
struct SmartPrimitiveEditorOut<'a> {
    selected_node_id: u32,
    camera: &'a CameraState,
}

impl SmartPrimitiveDocument {
    /// Load a document, upstream's `load_smart_primitive_document()` without the
    /// clip registry (clips are kept as paths, not loaded).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(path.as_ref(), None)
    }

    /// Load a document that must have been saved for `expected_rig`, the rig
    /// check upstream runs before loading any clips.
    pub fn load_for_rig(path: impl AsRef<Path>, expected_rig: ModelRig) -> Result<Self> {
        Self::load_impl(path.as_ref(), Some(expected_rig))
    }

    fn load_impl(path: &Path, expected_rig: Option<ModelRig>) -> Result<Self> {
        document_name_from_path(path, SMART_PRIMITIVE_SUFFIX, "Smart Primitive")?;
        let file: SmartPrimitiveFile = parse_json(&read_document_text(path)?, path)?;
        let file_name = path.file_name().unwrap_or_default().to_string_lossy();
        let rig = check_document_header(
            "Smart Primitive",
            SMART_PRIMITIVE_FORMAT,
            SMART_PRIMITIVE_SCHEMA_VERSION,
            &file_name,
            &file.format,
            file.schema_version,
            &file.rig,
        )?;
        if let Some(expected_rig) = expected_rig {
            if rig != expected_rig {
                return Err(DocumentError::new(format!(
                    "{file_name} was saved for the '{rig}' rig, but this app uses '{expected_rig}'."
                )));
            }
        }
        if file.editor.selected_node_id < 1 {
            return Err(DocumentError::new(
                "document.editor.selected_node_id must be at least 1.".to_string(),
            ));
        }
        file.editor.camera.validate("document.editor.camera")?;
        let mut document = Self {
            name: file.name,
            rig,
            selected_node_id: file.editor.selected_node_id as u32,
            camera: file.editor.camera,
            sequences: file.sequences,
            root: file.root,
        };
        document.validate()?;
        if document.find_node(document.selected_node_id).is_none() {
            document.selected_node_id = document.root.id;
        }
        Ok(document)
    }

    /// Return the rig a document was saved for, reading only its header.
    /// Upstream's `load_smart_primitive_rig()`.
    pub fn load_rig(path: impl AsRef<Path>) -> Result<ModelRig> {
        let path = path.as_ref();
        document_name_from_path(path, SMART_PRIMITIVE_SUFFIX, "Smart Primitive")?;
        let header: RigHeader = parse_json(&read_document_text(path)?, path)?;
        check_document_header(
            "Smart Primitive",
            SMART_PRIMITIVE_FORMAT,
            SMART_PRIMITIVE_SCHEMA_VERSION,
            &path.file_name().unwrap_or_default().to_string_lossy(),
            &header.format,
            header.schema_version,
            &header.rig,
        )
    }

    /// Save the document and editor settings, then return the file path,
    /// upstream's `save_smart_primitive_document()`.
    ///
    /// The path gets the `.smart_primitive.json` suffix when it lacks it, the
    /// document name is taken from the file name, and the previous file is
    /// replaced only after the write succeeds. `rotation_xyzw` is authoritative:
    /// every node's `euler_degrees` is recomputed from it before writing
    /// (nodes whose `euler_degrees` is `None` keep `None`), upstream's sync rule.
    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<PathBuf> {
        self.sync_euler_degrees();
        self.validate()?;
        self.camera.validate("document.editor.camera")?;
        let document_path = normalize_document_path(path.as_ref(), SMART_PRIMITIVE_SUFFIX);
        let name = document_name_from_path(&document_path, SMART_PRIMITIVE_SUFFIX, "Smart Primitive")?;
        let document = SmartPrimitiveFileOut {
            format: SMART_PRIMITIVE_FORMAT,
            schema_version: SMART_PRIMITIVE_SCHEMA_VERSION,
            name: &name,
            rig: self.rig.as_str(),
            editor: SmartPrimitiveEditorOut {
                selected_node_id: self.selected_node_id,
                camera: &self.camera,
            },
            sequences: &self.sequences,
            root: &self.root,
        };
        let json = serialize_pretty(&document, &document_path)?;
        atomic_write(&document_path, json.as_bytes())?;
        Ok(document_path)
    }

    /// Recompute every node's `euler_degrees` from its `rotation_xyzw`;
    /// `None` stays `None`.
    pub fn sync_euler_degrees(&mut self) {
        fn sync(node: &mut SceneNode) {
            if node.euler_degrees.is_some() {
                node.euler_degrees = Some(node.local_transform.euler_degrees());
            }
            for child in &mut node.children {
                sync(child);
            }
        }
        sync(&mut self.root);
    }

    /// The validation of upstream's `_SmartPrimitiveDocumentLoader`: unique IDs
    /// across nodes, components, and sequences; non-empty names; at most one
    /// component of each kind per node; component value ranges; and sequences
    /// that reference existing nodes holding the right components.
    pub fn validate(&self) -> Result<()> {
        let nodes = self.iter_nodes();
        let mut used_ids: HashSet<u32> = HashSet::new();
        let mut claim = |id: u32, location: &str| -> Result<()> {
            if id < 1 {
                return Err(DocumentError::new(format!(
                    "{location} must be at least 1."
                )));
            }
            if !used_ids.insert(id) {
                return Err(DocumentError::new(format!("Duplicate ID {id} at {location}.")));
            }
            Ok(())
        };

        for node in &nodes {
            let location = format!("node '{}'", node.name);
            claim(node.id, &format!("{location}.id"))?;
            if node.name.trim().is_empty() {
                return Err(DocumentError::new(format!(
                    "{location}.name must be a non-empty string."
                )));
            }
            node.local_transform
                .validate(&format!("Node '{}'.local_transform", node.name))?;
            if let Some(euler_degrees) = node.euler_degrees {
                if !euler_degrees.is_finite() {
                    return Err(DocumentError::new(format!(
                        "{location}.euler_degrees must be an array of 3 numbers."
                    )));
                }
            }
            let (mut target_poses, mut interaction_boxes, mut pivots) = (0, 0, 0);
            for component in &node.components {
                claim(component.id(), &format!("{location} component ID"))?;
                match component {
                    NodeComponent::TargetPose(component) => {
                        target_poses += 1;
                        if target_poses > 1 {
                            return Err(DocumentError::new(format!(
                                "{location} contains more than one Target Pose."
                            )));
                        }
                        validate_target_pose(component, &location)?;
                    }
                    NodeComponent::InteractionBox(component) => {
                        interaction_boxes += 1;
                        if interaction_boxes > 1 {
                            return Err(DocumentError::new(format!(
                                "{location} contains more than one Interaction Box."
                            )));
                        }
                        validate_interaction_box(component, &location)?;
                    }
                    NodeComponent::Pivot(component) => {
                        pivots += 1;
                        if pivots > 1 {
                            return Err(DocumentError::new(format!(
                                "{location} contains more than one Pivot."
                            )));
                        }
                        validate_pivot(component, &location)?;
                    }
                }
            }
        }

        let mut sequence_names = HashSet::new();
        for sequence in &self.sequences {
            claim(sequence.id, &format!("sequence '{}'.id", sequence.name))?;
            if sequence.name.trim().is_empty() {
                return Err(DocumentError::new(format!(
                    "sequence '{}'.name must be a non-empty string.",
                    sequence.name
                )));
            }
            if !sequence_names.insert(&sequence.name) {
                return Err(DocumentError::new(
                    "Sequence names must be unique.".to_string(),
                ));
            }
            for (index, node_id) in sequence.node_ids.iter().enumerate() {
                let location = format!("sequence '{}'.node_ids[{index}]", sequence.name);
                let node = self.find_node(*node_id).ok_or_else(|| {
                    DocumentError::new(format!("{location} does not identify a hierarchy node."))
                })?;
                if node.target_pose().is_none() {
                    return Err(DocumentError::new(format!(
                        "{location} identifies node '{}', which has no Target Pose.",
                        node.name
                    )));
                }
            }
            if let Some(pivot_node_id) = sequence.pivot_node_id {
                let has_pivot = self
                    .find_node(pivot_node_id)
                    .is_some_and(|node| node.pivot().is_some());
                if !has_pivot {
                    return Err(DocumentError::new(format!(
                        "sequence '{}'.pivot_node_id must identify a node with a Pivot component.",
                        sequence.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Return the node with `node_id`, or `None`. Upstream's `find_node()`.
    pub fn find_node(&self, node_id: u32) -> Option<&SceneNode> {
        self.iter_nodes().into_iter().find(|node| node.id == node_id)
    }

    /// Return the parent of the node with `node_id`, or `None` for the root and
    /// unknown IDs.
    pub fn find_parent(&self, node_id: u32) -> Option<&SceneNode> {
        fn find_in(node: &SceneNode, node_id: u32) -> Option<&SceneNode> {
            for child in &node.children {
                if child.id == node_id {
                    return Some(node);
                }
                if let found @ Some(_) = find_in(child, node_id) {
                    return found;
                }
            }
            None
        }
        find_in(&self.root, node_id)
    }

    /// Every node, parents before children, preserving the authored sibling
    /// order. Upstream's `iter_nodes()`.
    pub fn iter_nodes(&self) -> Vec<&SceneNode> {
        fn walk<'a>(node: &'a SceneNode, nodes: &mut Vec<&'a SceneNode>) {
            nodes.push(node);
            for child in &node.children {
                walk(child, nodes);
            }
        }
        let mut nodes = Vec::new();
        walk(&self.root, &mut nodes);
        nodes
    }

    /// The node's transform composed through all of its parents, upstream's
    /// `world_xform()`.
    pub fn world_transform(&self, node_id: u32) -> Option<Transform> {
        let node = self.find_node(node_id)?;
        let mut transform = node.local_transform;
        let mut current = node;
        while let Some(parent) = self.find_parent(current.id) {
            transform = parent.local_transform.mul(&transform);
            current = parent;
        }
        Some(transform)
    }

    /// Whether the node and all of its ancestors are visible, upstream's
    /// `is_visible()`.
    pub fn is_visible(&self, node_id: u32) -> Option<bool> {
        let node = self.find_node(node_id)?;
        let mut visible = node.visible;
        let mut current = node;
        while let Some(parent) = self.find_parent(current.id) {
            visible &= parent.visible;
            current = parent;
        }
        Some(visible)
    }

    /// The next free ID: one above the highest node, component, or sequence ID
    /// in use. Upstream tracks this as `_next_id = maximum_id + 1` at load.
    pub fn next_free_id(&self) -> u32 {
        self.iter_nodes()
            .iter()
            .flat_map(|node| {
                node.components
                    .iter()
                    .map(NodeComponent::id)
                    .chain([node.id])
                    .collect::<Vec<_>>()
            })
            .chain(self.sequences.iter().map(|sequence| sequence.id))
            .max()
            .unwrap_or(0)
            + 1
    }

    /// Each node ID with its Target Pose component, parents before children.
    /// Upstream's `iter_target_poses()`.
    pub fn target_poses(&self) -> Vec<(u32, &TargetPoseComponent)> {
        self.iter_nodes()
            .into_iter()
            .filter_map(|node| node.target_pose().map(|component| (node.id, component)))
            .collect()
    }

    /// Each node ID with its Interaction Box component, parents before children.
    /// Upstream's `iter_interaction_boxes()`.
    pub fn interaction_boxes(&self) -> Vec<(u32, &InteractionBoxComponent)> {
        self.iter_nodes()
            .into_iter()
            .filter_map(|node| node.interaction_box().map(|component| (node.id, component)))
            .collect()
    }

    /// Each node ID with its Pivot component, parents before children.
    /// Upstream's `iter_pivots()`.
    pub fn pivots(&self) -> Vec<(u32, &PivotComponent)> {
        self.iter_nodes()
            .into_iter()
            .filter_map(|node| node.pivot().map(|component| (node.id, component)))
            .collect()
    }

    /// Resolve one sequence: its Target Pose components in order plus the
    /// optional Pivot component. `None` for an unknown sequence ID.
    pub fn resolve_sequence(&self, sequence_id: u32) -> Option<ResolvedSequence<'_>> {
        let sequence = self.sequences.iter().find(|sequence| sequence.id == sequence_id)?;
        Some(ResolvedSequence {
            id: sequence.id,
            name: &sequence.name,
            target_poses: sequence
                .node_ids
                .iter()
                .map(|node_id| {
                    let node = self.find_node(*node_id)?;
                    node.target_pose().map(|component| (node.id, component))
                })
                .collect::<Option<Vec<_>>>()?,
            pivot: sequence.pivot_node_id.and_then(|node_id| {
                let node = self.find_node(node_id)?;
                node.pivot().map(|component| (node.id, component))
            }),
        })
    }

    /// Every sequence, resolved. See [`Self::resolve_sequence`].
    pub fn resolve_sequences(&self) -> Vec<ResolvedSequence<'_>> {
        self.sequences
            .iter()
            .filter_map(|sequence| self.resolve_sequence(sequence.id))
            .collect()
    }
}

/// Upstream's `_load_target_pose_component()` and `_load_target_pose_properties()`.
fn validate_target_pose(component: &TargetPoseComponent, location: &str) -> Result<()> {
    if !component.sample_time_seconds.is_finite() {
        return Err(DocumentError::new(format!(
            "{location}.sample_time_seconds must be a finite number."
        )));
    }
    if component.sample_time_seconds < 0.0 {
        return Err(DocumentError::new(format!(
            "{location}.sample_time_seconds cannot be negative."
        )));
    }
    if component.clip.is_none() && (component.sample_time_seconds != 0.0 || component.frame_index != 0)
    {
        return Err(DocumentError::new(format!(
            "{location} cannot select a frame without a clip."
        )));
    }
    if !(1..=MODEL_POSES_PER_TOKEN).contains(&component.num_poses_to_use) {
        return Err(DocumentError::new(format!(
            "{location}.num_poses_to_use must be 1 to {MODEL_POSES_PER_TOKEN}."
        )));
    }
    if component.transition_early_by_frames > MAX_TRANSITION_EARLY_BY_FRAMES {
        return Err(DocumentError::new(format!(
            "{location}.transition_early_by_frames must be at most {MAX_TRANSITION_EARLY_BY_FRAMES}."
        )));
    }
    if !(MODEL_MIN_TOKEN_COUNT..=MODEL_MAX_TOKEN_COUNT).contains(&component.min_tokens) {
        return Err(DocumentError::new(format!(
            "{location}.min_tokens must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}."
        )));
    }
    if component.max_tokens < component.min_tokens || component.max_tokens > MODEL_MAX_TOKEN_COUNT {
        return Err(DocumentError::new(format!(
            "{location}.max_tokens must be {} to {MODEL_MAX_TOKEN_COUNT}.",
            component.min_tokens
        )));
    }
    if component.ik_pin_mask & !IK_PIN_ALL_MASK != 0 {
        return Err(DocumentError::new(format!(
            "{location}.ik_pin_mask contains unsupported bits."
        )));
    }
    if !component.trajectory_scale.is_finite()
        || component.trajectory_scale < MIN_TRAJECTORY_SCALE
        || component.trajectory_scale > MAX_TRAJECTORY_SCALE
    {
        return Err(DocumentError::new(format!(
            "{location}.trajectory_scale must be {MIN_TRAJECTORY_SCALE:.1} to {MAX_TRAJECTORY_SCALE:.1}."
        )));
    }
    validate_color(&component.color, &format!("Target Pose {}.color", component.id))
}

/// Upstream's `_load_interaction_box_component()`.
fn validate_interaction_box(component: &InteractionBoxComponent, location: &str) -> Result<()> {
    for (field, value) in [
        ("width", component.width),
        ("depth", component.depth),
        ("height", component.height),
    ] {
        if !value.is_finite() {
            return Err(DocumentError::new(format!(
                "{location}.{field} must be a finite number."
            )));
        }
        if value < 0.0 {
            return Err(DocumentError::new(format!(
                "{location}.{field} cannot be negative."
            )));
        }
    }
    validate_color(&component.color, &format!("Interaction Box {}.color", component.id))
}

/// Upstream's `_load_pivot_component()` plus `Pivot.validate()`. Unlike upstream
/// (`Pivot.set_motion_direction()`), the motion direction is NOT normalized here:
/// documents are round-tripped byte-faithfully, and the stored directions are
/// already unit length.
fn validate_pivot(component: &PivotComponent, location: &str) -> Result<()> {
    let invalid = |message: &str| {
        DocumentError::new(format!("Invalid Pivot at {location}: {message}"))
    };
    if !component.motion_direction.is_finite() || component.motion_direction.length() == 0.0 {
        return Err(invalid("Motion Direction must be finite and non-zero."));
    }
    if !component.line_min_offset.is_finite()
        || !component.line_max_offset.is_finite()
        || component.line_min_offset > component.line_max_offset
    {
        return Err(invalid(
            "Line offsets must be finite, with Min Offset <= Max Offset.",
        ));
    }
    if !component.cone_half_angle_degrees.is_finite()
        || !(0.0..=180.0).contains(&component.cone_half_angle_degrees)
    {
        return Err(invalid("Cone Half Angle must be between 0 and 180 degrees."));
    }
    Ok(())
}
