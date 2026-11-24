// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Smart Primitive Editor: build Smart Primitives by placing clip poses,
//! pivots, and interaction boxes. Port of upstream `app/smart_primitive_editor.py`
//! (`python -m app.smart_primitive_editor`), on the shared framework in
//! `src/editor/`.
//!
//! Run with: `cargo run --release --bin smart-primitive-editor -- assets/smart_primitives/soma/sit.smart_primitive.json`
//!
//! Deviations from upstream (all deliberate):
//! - The tkinter file dialogs (document open/save, clip chooser) are in-imgui
//!   path-input modals ([`editor::EditorShell::open_path_prompt`]).
//! - The transform gizmo translates along the viewer's world axes only (the
//!   viewer has no rotation rings and no parent-axis mode); rotation is edited
//!   in the Properties panel. Axis locks from the panel apply to the gizmo.
//! - The preview math upstream runs as Warp kernels (pivot cones, trajectory
//!   frames) runs on the CPU here; the IK pin markers are small solid boxes
//!   (this viewer has no sphere primitive).
//! - Clip poses are resampled every frame instead of cached in GPU buffers;
//!   the clip files themselves are loaded once and shared.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::exit;

use anyhow::{Context as _, Result, bail};
use glam::{Mat4, Quat, Vec2, Vec3};
use lexopt::prelude::*;

use motionbricks_app::documents::smart_primitive::{
    Axis, ConeSide, ConstraintType, InteractionBoxComponent, MAX_TRAJECTORY_SCALE, MIN_TRAJECTORY_SCALE,
    MODEL_POSES_PER_TOKEN, NodeComponent, OperationOrder, PlaybackMode, PivotComponent, SMART_PRIMITIVE_SUFFIX,
    SceneNode, SmartPrimitiveDocument, TargetPoseComponent, TargetPoseSequence,
};
use motionbricks_app::documents::{CameraState, ModelRig, Transform};
use motionbricks_app::editor::widgets::{
    FileAction, InlineRename, PANEL_ALPHA, POSES_TO_USE_TOOLTIP, PROPERTIES_WINDOW_HEIGHT, PROPERTIES_WINDOW_WIDTH,
    WINDOW_MARGIN, accept_drop, apply_transform_locks, clamp_token_range, draw_add_remove_buttons, draw_box_dimensions,
    draw_character_selector, draw_clip_selector, draw_int_slider_property, draw_label, draw_token_range_property,
    draw_transform_editor, draw_tree_row, drag_source, sample_frame_at, transforms_equal,
};
use motionbricks_app::editor::{
    self, DocumentType, EditorShell, ExitChoice, PathPrompt, initial_panel_geometry, ray_mesh,
    ray_obb, viewer_ray_to_world, world_point_to_viewer, world_to_viewer, world_to_viewer_matrix,
};
use motionbricks_app::rig::clips::{Clip, ClipRegistry};
use motionbricks_app::rig::g1::G1Rig;
use motionbricks_app::rig::soma::{SomaRig, transform_to_mat4};
use motionbricks_app::runtime::behaviors::Pivot;
use motionbricks_app::viewer::{Frame, Mesh, Ray, Skin, Viewer, ViewerConfig};
use motionbricks_rs::runtime as rt;
use motionbricks_rs::runtime::MODEL_SAMPLE_RATE;
use motionbricks_rs::runtime::target_pose::{create_soma_target_pose_ik_rig, ik_pin};

// Upstream constants (app/smart_primitive_editor.py and
// app/runtime/common/io/smart_primitive_document.py).
const TRAJECTORY_HISTORY_FRAMES: usize = 10;
const TRAJECTORY_FUTURE_FRAMES: usize = 10;
const TRAJECTORY_POINT_COUNT: usize = TRAJECTORY_HISTORY_FRAMES + 1 + TRAJECTORY_FUTURE_FRAMES;
const TRAJECTORY_COLOR: [f32; 4] = [1.0, 0.8, 0.0, 1.0];
const IK_PIN_COLOR: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const IK_PIN_MARKER_HALF_EXTENT: f32 = 0.05;
const TRAJECTORY_FRAME_SCALE: f32 = 0.075;
const MIN_INTERACTION_BOX_SIZE: f32 = 0.0;
const PIVOT_CONE_SEGMENTS: usize = 64;
const PIVOT_CONE_RADIUS: f32 = 1.5;
const PIVOT_ARROW_LIFT: f32 = 0.03;
const PIVOT_ARROW_LENGTH: f32 = 0.75;
const VELOCITY_EPSILON: f32 = 1.0e-6;
const DEFAULT_MIN_TOKENS: u32 = 6;
const DEFAULT_MAX_TOKENS: u32 = 14;
const DEFAULT_TRAJECTORY_SCALE: f32 = 2.0;

const HIERARCHY_WINDOW_WIDTH: f32 = 320.0;
const HIERARCHY_WINDOW_HEIGHT: f32 = 380.0;
const SEQUENCES_WINDOW_WIDTH: f32 = 360.0;
const SEQUENCES_WINDOW_HEIGHT: f32 = 300.0;
const TREE_PAYLOAD: &str = "SMART_PRIMITIVE_NODE";
const SEQUENCE_STEP_PAYLOAD: &str = "SMART_PRIMITIVE_SEQUENCE_STEP";

const OUTLINER_WINDOW_TITLE: &str = "Hierarchy";
const EDITOR_TITLE: &str = "MotionBricks - Smart Primitive Editor";
const DOCUMENT_TYPE: DocumentType = DocumentType { kind: "Smart Primitive", suffix: SMART_PRIMITIVE_SUFFIX };

/// The IK pins in upstream's `IK_PINS` order, with `_ik_pin_label`'s labels.
const IK_PINS: [(u32, &str); 4] = [
    (ik_pin::LEFT_HAND, "LeftHand"),
    (ik_pin::RIGHT_HAND, "RightHand"),
    (ik_pin::LEFT_FOOT, "LeftFoot"),
    (ik_pin::RIGHT_FOOT, "RightFoot"),
];

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Args {
    document: Option<PathBuf>,
    clips: PathBuf,
    assets: PathBuf,
}

fn usage() -> ! {
    eprintln!(
        "MotionBricks Smart Primitive Editor\n\
         \n\
         Usage: smart-primitive-editor [OPTIONS] [DOCUMENT]\n\
         \n\
         Options:\n\
         \x20 DOCUMENT       .smart_primitive.json document to open at startup\n\
         \x20 --clips DIR     clip library root (default: checkpoints/motions)\n\
         \x20 --assets DIR    upstream checkout root, for the G1 meshes\n\
         \x20                (default: sandbox/upstream)\n\
         \x20 --help          show this help"
    );
    exit(if std::env::args().any(|arg| arg == "--help") { 0 } else { 2 });
}

fn first_existing(candidates: &[&str], fallback: &str) -> PathBuf {
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from(fallback)
}

fn parse_args() -> Result<Args> {
    let mut document = None;
    let mut clips = first_existing(&["checkpoints/motions", "../checkpoints/motions"], "checkpoints/motions");
    let mut assets = first_existing(&["sandbox/upstream", "../sandbox/upstream"], "sandbox/upstream");
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(value) => document = Some(PathBuf::from(value.string()?)),
            Long("document") => document = Some(PathBuf::from(parser.value()?)),
            Long("clips") => clips = PathBuf::from(parser.value()?),
            Long("assets") => assets = PathBuf::from(parser.value()?),
            Long("help") => usage(),
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(Args { document, clips, assets })
}

fn absolutize(path: &Path) -> PathBuf {
    let expanded: PathBuf = shellexpand::tilde(&path.to_string_lossy()).into_owned().into();
    std::path::absolute(&expanded).unwrap_or(expanded)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Upstream `_random_color`, deterministic per component ID: an HSV color with
/// the same saturation and value.
fn random_color(id: u32) -> [f32; 3] {
    // A golden-ratio hue walk keeps neighboring IDs visually distinct.
    let hue = (id as f32 * 0.618_034) % 1.0;
    hsv_to_rgb(hue, 0.65, 0.95)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let h = (hue * 6.0).rem_euclid(6.0);
    let c = value * saturation;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = value - c;
    [r + m, g + m, b + m]
}

/// The local unit vector of a pivot axis (`newton.Axis.from_string(...).to_vec3()`).
fn axis_vector(axis: Axis) -> Option<Vec3> {
    match axis {
        Axis::None => None,
        Axis::X => Some(Vec3::X),
        Axis::Y => Some(Vec3::Y),
        Axis::Z => Some(Vec3::Z),
    }
}

/// The clip axis `model_to_runtime` turns into world Z (upstream
/// `SmartPrimitiveDocument.set_rig`): 1 (Y) for SOMA, 2 (Z) for the G1.
fn up_index(model_to_runtime: rt::Transform) -> usize {
    let rotation = Quat::from_array(model_to_runtime.q);
    let up = rotation.inverse() * Vec3::Z;
    let abs = up.abs();
    if abs.y >= abs.x && abs.y >= abs.z {
        1
    } else if abs.z >= abs.x {
        2
    } else {
        0
    }
}

fn rt_transform(transform: &Transform) -> rt::Transform {
    rt::Transform::new(rt::Vec3(transform.translation.to_array()), transform.rotation_xyzw.to_array())
}

fn glam_transform(transform: rt::Transform) -> Transform {
    Transform::new(Vec3::from_array(transform.p.0), Quat::from_array(transform.q))
}

/// A motion direction at unit length, or the validation error upstream's
/// `Pivot.set_motion_direction` reports.
fn normalize_motion_direction(direction: [f32; 3]) -> Result<Vec3, &'static str> {
    let direction = Vec3::from_array(direction);
    let length = direction.length();
    if !length.is_finite() || length == 0.0 {
        return Err("Motion Direction must be finite and non-zero.");
    }
    Ok(direction / length)
}

/// Move a sequence step from one index to another; out-of-range indices are a
/// no-op (upstream `move_sequence_node`, which raises IndexError instead).
fn move_sequence_step(sequence: &mut TargetPoseSequence, source_index: usize, destination_index: usize) {
    if source_index >= sequence.node_ids.len() || destination_index >= sequence.node_ids.len() {
        return;
    }
    let node_id = sequence.node_ids.remove(source_index);
    sequence.node_ids.insert(destination_index, node_id);
}

// ---------------------------------------------------------------------------
// Document operations (upstream `SmartPrimitiveDocument`'s authoring half; the
// document layer in `documents/` holds the data and file IO)
// ---------------------------------------------------------------------------

fn find_node_mut(node: &mut SceneNode, node_id: u32) -> Option<&mut SceneNode> {
    if node.id == node_id {
        return Some(node);
    }
    for child in &mut node.children {
        if let Some(found) = find_node_mut(child, node_id) {
            return Some(found);
        }
    }
    None
}

fn find_parent_mut(node: &mut SceneNode, node_id: u32) -> Option<&mut SceneNode> {
    if node.children.iter().any(|child| child.id == node_id) {
        return Some(node);
    }
    for child in &mut node.children {
        if subtree_contains(child, node_id) {
            return find_parent_mut(child, node_id);
        }
    }
    None
}

fn subtree_contains(node: &SceneNode, node_id: u32) -> bool {
    node.id == node_id || node.children.iter().any(|child| subtree_contains(child, node_id))
}

fn subtree_ids(node: &SceneNode) -> Vec<u32> {
    let mut ids = vec![node.id];
    for child in &node.children {
        ids.extend(subtree_ids(child));
    }
    ids
}

/// Add a node under `parent_id` (the root when `None`) and return its ID
/// (upstream `add_node`).
fn add_node(document: &mut SmartPrimitiveDocument, parent_id: Option<u32>) -> u32 {
    let node_id = document.next_free_id();
    let parent = match parent_id {
        Some(parent_id) => find_node_mut(&mut document.root, parent_id).expect("parent is a document node"),
        None => &mut document.root,
    };
    parent.children.push(SceneNode {
        id: node_id,
        name: format!("Node {node_id}"),
        local_transform: Transform::IDENTITY,
        euler_degrees: None,
        visible: true,
        translation_locks: [false; 3],
        rotation_locks: [false; 3],
        components: Vec::new(),
        children: Vec::new(),
    });
    node_id
}

/// Delete a node with its descendants and drop them from every sequence
/// (upstream `remove_node`). The root node stays. Returns the removed IDs.
fn remove_node(document: &mut SmartPrimitiveDocument, node_id: u32) -> Vec<u32> {
    if node_id == document.root.id {
        return Vec::new();
    }
    let Some(node) = document.find_node(node_id) else { return Vec::new() };
    let removed: HashSet<u32> = subtree_ids(node).into_iter().collect();
    find_parent_mut(&mut document.root, node_id).expect("a non-root node has a parent").children.retain(|child| child.id != node_id);
    for sequence in &mut document.sequences {
        if sequence.pivot_node_id.is_some_and(|pivot_id| removed.contains(&pivot_id)) {
            sequence.pivot_node_id = None;
        }
        sequence.node_ids.retain(|step_id| !removed.contains(step_id));
    }
    removed.into_iter().collect()
}

/// Move a node under `new_parent_id`, keeping its world placement; return
/// whether it moved (upstream `can_reparent` + `reparent`).
fn reparent(document: &mut SmartPrimitiveDocument, node_id: u32, new_parent_id: u32) -> bool {
    if node_id == document.root.id || node_id == new_parent_id {
        return false;
    }
    // No cycles: the node may not move under itself or a descendant.
    let mut ancestor = Some(new_parent_id);
    while let Some(current) = ancestor {
        if current == node_id {
            return false;
        }
        ancestor = document.find_parent(current).map(|parent| parent.id);
    }
    let Some(parent) = document.find_parent(node_id) else { return false };
    if parent.id == new_parent_id {
        return false;
    }
    let world = document.world_transform(node_id);
    let parent_world = document.world_transform(new_parent_id);
    let (Some(world), Some(parent_world)) = (world, parent_world) else { return false };
    let local = parent_world.inverse().mul(&world);
    let parent = find_parent_mut(&mut document.root, node_id).expect("a non-root node has a parent");
    let index = parent.children.iter().position(|child| child.id == node_id).expect("the node is its parent's child");
    let mut node = parent.children.remove(index);
    node.local_transform = local;
    let new_parent = find_node_mut(&mut document.root, new_parent_id).expect("new parent is a document node");
    new_parent.children.push(node);
    true
}

fn default_target_pose(id: u32) -> TargetPoseComponent {
    TargetPoseComponent {
        id,
        clip: None,
        sample_time_seconds: 0.0,
        frame_index: 0,
        playback_mode: PlaybackMode::Time,
        hips_constrained: false,
        num_poses_to_use: MODEL_POSES_PER_TOKEN,
        transition_early_by_frames: 0,
        min_tokens: DEFAULT_MIN_TOKENS,
        max_tokens: DEFAULT_MAX_TOKENS,
        ik_pin_mask: ik_pin::NONE,
        show_mesh: true,
        show_skeletons: false,
        show_velocity: false,
        show_trajectory: false,
        trajectory_scale: DEFAULT_TRAJECTORY_SCALE,
        color: random_color(id),
    }
}

fn default_interaction_box(id: u32) -> InteractionBoxComponent {
    InteractionBoxComponent { id, width: 1.0, depth: 1.0, height: 1.0, color: random_color(id) }
}

fn default_pivot(id: u32) -> PivotComponent {
    PivotComponent {
        id,
        motion_direction: Vec3::Y,
        operation_order: OperationOrder::TranslationThenRotation,
        translation_axis: Axis::None,
        translation_constraint: ConstraintType::Unconstrained,
        line_min_offset: -0.5,
        line_max_offset: 0.5,
        rotation_axis: Axis::None,
        rotation_constraint: ConstraintType::Unconstrained,
        cone_half_angle_degrees: 45.0,
        cone_side: ConeSide::Forward,
        show_motion_direction: true,
        show_translation_constraint: true,
        show_rotation_constraint: true,
    }
}

fn has_component(node: &SceneNode, kind: &str) -> bool {
    node.components.iter().any(|component| {
        matches!(
            (component, kind),
            (NodeComponent::TargetPose(_), "target_pose")
                | (NodeComponent::InteractionBox(_), "interaction_box")
                | (NodeComponent::Pivot(_), "pivot")
        )
    })
}

/// Add a component with upstream's defaults; return its ID (upstream
/// `add_target_pose` / `add_interaction_box` / `add_pivot`).
fn add_component(document: &mut SmartPrimitiveDocument, node_id: u32, kind: &str) -> Option<u32> {
    let component_id = document.next_free_id();
    let node = find_node_mut(&mut document.root, node_id)?;
    if has_component(node, kind) {
        return None;
    }
    let component = match kind {
        "target_pose" => NodeComponent::TargetPose(default_target_pose(component_id)),
        "interaction_box" => NodeComponent::InteractionBox(default_interaction_box(component_id)),
        "pivot" => NodeComponent::Pivot(default_pivot(component_id)),
        _ => return None,
    };
    node.components.push(component);
    Some(component_id)
}

/// Remove a component from a node and drop every sequence reference to it
/// (upstream `remove_component`).
fn remove_component(document: &mut SmartPrimitiveDocument, node_id: u32, component_id: u32) {
    let Some(node) = find_node_mut(&mut document.root, node_id) else { return };
    let was_target_pose = node.components.iter().any(|c| matches!(c, NodeComponent::TargetPose(t) if t.id == component_id));
    let was_pivot = node.components.iter().any(|c| matches!(c, NodeComponent::Pivot(p) if p.id == component_id));
    node.components.retain(|component| component.id() != component_id);
    if was_target_pose {
        for sequence in &mut document.sequences {
            sequence.node_ids.retain(|step_id| *step_id != node_id);
        }
    }
    if was_pivot {
        for sequence in &mut document.sequences {
            if sequence.pivot_node_id == Some(node_id) {
                sequence.pivot_node_id = None;
            }
        }
    }
}

/// Add an empty sequence with the next free name (upstream `add_sequence`).
fn add_sequence(document: &mut SmartPrimitiveDocument) -> u32 {
    let existing: HashSet<&str> = document.sequences.iter().map(|sequence| sequence.name.as_str()).collect();
    let mut number = 1;
    while existing.contains(format!("Sequence {number}").as_str()) {
        number += 1;
    }
    let id = document.next_free_id();
    document.sequences.push(TargetPoseSequence { id, name: format!("Sequence {number}"), node_ids: Vec::new(), pivot_node_id: None });
    id
}

/// Each sequence that uses the node as a step or as its pivot (upstream
/// `sequences_referencing`).
fn sequences_referencing(document: &SmartPrimitiveDocument, node_id: u32) -> Vec<u32> {
    document
        .sequences
        .iter()
        .filter(|sequence| sequence.node_ids.contains(&node_id) || sequence.pivot_node_id == Some(node_id))
        .map(|sequence| sequence.id)
        .collect()
}

/// A document holding only the root node (upstream `SmartPrimitiveDocument(rig, ...)`).
fn new_smart_primitive_document(rig: ModelRig) -> SmartPrimitiveDocument {
    SmartPrimitiveDocument {
        name: String::new(),
        rig,
        selected_node_id: 1,
        camera: CameraState {
            pos: Vec3::new(-4.0, -4.0, 3.0),
            pivot: Vec3::new(0.0, 0.0, 1.0),
            pitch: 0.0,
            yaw: 0.0,
            fov: motionbricks_app::viewer::OrbitCamera::FOV_Y.to_degrees(),
            near: motionbricks_app::viewer::OrbitCamera::NEAR,
            far: motionbricks_app::viewer::OrbitCamera::FAR,
        },
        sequences: Vec::new(),
        root: SceneNode {
            id: 1,
            name: "Root".to_string(),
            local_transform: Transform::IDENTITY,
            euler_degrees: None,
            visible: true,
            translation_locks: [false; 3],
            rotation_locks: [false; 3],
            components: Vec::new(),
            children: Vec::new(),
        },
    }
}

// ---------------------------------------------------------------------------
// Rig assets and clip sampling (upstream `_Rig` + `TargetPoseComponent._sample`)
// ---------------------------------------------------------------------------

/// One rig's visual assets, clip registry, and preview constants.
struct RigData {
    soma: Option<Box<SomaRig>>,
    soma_skin: Option<Skin>,
    g1: Option<Box<G1Rig>>,
    g1_link_meshes: Vec<Mesh>,
    registry: ClipRegistry,
    model_to_runtime: rt::Transform,
    up_index: usize,
    /// (pin mask bit, tip joint index) of each SOMA target-pose IK chain.
    ik_tips: Vec<(u32, usize)>,
}

impl RigData {
    fn load(rig: ModelRig, clips_dir: &Path, assets_dir: &Path) -> Result<RigData> {
        match rig {
            ModelRig::Soma => {
                let soma = SomaRig::load(clips_dir).context("could not load the SOMA rig")?;
                let registry = ClipRegistry::soma(soma.skeleton.clone());
                let skin = soma.viewer_skin();
                let ik_tips = create_soma_target_pose_ik_rig(&soma.skeleton)
                    .context("could not build the SOMA target pose IK rig")?
                    .into_iter()
                    .map(|chain| (chain.pin, chain.tip_index))
                    .collect();
                Ok(RigData {
                    model_to_runtime: soma.model_to_runtime,
                    up_index: up_index(soma.model_to_runtime),
                    soma: Some(Box::new(soma)),
                    soma_skin: Some(skin),
                    g1: None,
                    g1_link_meshes: Vec::new(),
                    registry,
                    ik_tips,
                })
            }
            ModelRig::UnitreeG1 => {
                let g1 = G1Rig::load(assets_dir).with_context(|| {
                    format!("could not load the Unitree G1 rig from {}", assets_dir.display())
                })?;
                let mut link_meshes = Vec::with_capacity(g1.links.len());
                for index in 0..g1.links.len() {
                    link_meshes.push(g1.link_mesh(index)?);
                }
                Ok(RigData {
                    model_to_runtime: g1.model_to_runtime,
                    up_index: up_index(g1.model_to_runtime),
                    soma: None,
                    soma_skin: None,
                    g1: Some(Box::new(g1)),
                    g1_link_meshes: link_meshes,
                    registry: ClipRegistry::unitree_g1_contract(),
                    ik_tips: Vec::new(),
                })
            }
        }
    }

    fn rig(&self) -> ModelRig {
        if self.soma.is_some() { ModelRig::Soma } else { ModelRig::UnitreeG1 }
    }
}

/// One sampled pose: SOMA local joint transforms, or G1 joint coordinates.
#[derive(Clone)]
enum PoseSample {
    Soma(Vec<rt::Transform>),
    G1(Vec<f32>),
}

impl PoseSample {
    /// The pose's root transform in model space.
    fn root(&self) -> rt::Transform {
        match self {
            PoseSample::Soma(local) => local.first().copied().unwrap_or(rt::Transform::IDENTITY),
            PoseSample::G1(q) => rt::read_root(q),
        }
    }
}

/// The preview data of a target pose at its sample time: the four sampled
/// poses (the reference pose without a clip), plus the root position and
/// velocity of upstream `_sample`.
struct TargetSample {
    poses: Vec<PoseSample>,
    /// Root position of the first sampled pose, in model space.
    hips_position: Vec3,
    /// Root velocity between the first two sampled poses, in model space.
    velocity: Vec3,
    /// The root offset the hips constraint removes (identity when unconstrained).
    root_offset: rt::Transform,
    has_clip: bool,
}

/// The root offset the hips constraint removes: the ground-plane negation of
/// the root position at the sample time, keeping the height (upstream
/// `_sample`'s `_root_offset`).
fn hips_constraint_offset(rig: &RigData, clip: &Clip, sample_time: f32) -> Result<rt::Transform> {
    let root = match clip {
        Clip::Soma(buffer) => buffer.sample(sample_time, rt::Transform::IDENTITY)?.first().copied().unwrap_or(rt::Transform::IDENTITY),
        Clip::G1(buffer) => rt::read_root(&buffer.sample(sample_time, rt::Transform::IDENTITY)?),
    };
    let mut offset = [-root.p.0[0], -root.p.0[1], -root.p.0[2]];
    offset[rig.up_index] = 0.0;
    Ok(rt::Transform::new(rt::Vec3(offset), rt::math::QUAT_IDENTITY))
}

/// Sample the target pose (upstream `TargetPoseComponent._sample`): four
/// 30 Hz poses from the sample time, or the reference pose without a clip.
fn sample_target(rig: &RigData, component: &TargetPoseComponent, clip: Option<&Clip>) -> Result<TargetSample> {
    let Some(clip) = clip else {
        let reference = match rig.rig() {
            ModelRig::Soma => PoseSample::Soma(
                rig.soma.as_ref().expect("SOMA rig data").skeleton.reference_local_transforms().to_vec(),
            ),
            ModelRig::UnitreeG1 => PoseSample::G1(rig.g1.as_ref().expect("G1 rig data").reference_q()),
        };
        let hips_position = reference.root().p.0.into();
        return Ok(TargetSample {
            poses: vec![reference; MODEL_POSES_PER_TOKEN as usize],
            hips_position,
            velocity: Vec3::ZERO,
            root_offset: rt::Transform::IDENTITY,
            has_clip: false,
        });
    };
    let root_offset = if component.hips_constrained {
        hips_constraint_offset(rig, clip, component.sample_time_seconds)?
    } else {
        rt::Transform::IDENTITY
    };
    let mut poses = Vec::with_capacity(MODEL_POSES_PER_TOKEN as usize);
    for frame in 0..MODEL_POSES_PER_TOKEN {
        let time = component.sample_time_seconds + frame as f32 / MODEL_SAMPLE_RATE;
        poses.push(match clip {
            Clip::Soma(buffer) => PoseSample::Soma(buffer.sample(time, root_offset)?),
            Clip::G1(buffer) => PoseSample::G1(buffer.sample(time, root_offset)?),
        });
    }
    let first = poses[0].root().p.0;
    let second = poses[1].root().p.0;
    Ok(TargetSample {
        hips_position: first.into(),
        velocity: Vec3::new(second[0] - first[0], second[1] - first[1], second[2] - first[2]) * MODEL_SAMPLE_RATE,
        root_offset,
        poses,
        has_clip: true,
    })
}

/// The velocity shown by the preview: zero when only one pose is used
/// (upstream `TargetPoseComponent.velocity`).
fn preview_velocity(sample: &TargetSample, num_poses_to_use: u32) -> Vec3 {
    if num_poses_to_use == 1 { Vec3::ZERO } else { sample.velocity }
}

/// The 21 root transforms of the trajectory preview, spaced
/// `trajectory_scale` frames around the sample time (upstream
/// `_update_trajectory_samples`).
fn sample_trajectory(rig: &RigData, component: &TargetPoseComponent, clip: &Clip, root_offset: rt::Transform) -> Result<Vec<rt::Transform>> {
    let _ = rig;
    let interval = component.trajectory_scale / MODEL_SAMPLE_RATE;
    let mut transforms = Vec::with_capacity(TRAJECTORY_POINT_COUNT);
    for point in 0..TRAJECTORY_POINT_COUNT {
        let time = component.sample_time_seconds + (point as f32 - TRAJECTORY_HISTORY_FRAMES as f32) * interval;
        transforms.push(match clip {
            Clip::Soma(buffer) => buffer.sample(time, root_offset)?.first().copied().unwrap_or(rt::Transform::IDENTITY),
            Clip::G1(buffer) => rt::read_root(&buffer.sample(time, root_offset)?),
        });
    }
    Ok(transforms)
}

// ---------------------------------------------------------------------------
// The editor application (upstream `SmartPrimitiveEditor`)
// ---------------------------------------------------------------------------

struct SmartPrimitiveEditor {
    shell: EditorShell,
    document: SmartPrimitiveDocument,
    rig_data: Option<RigData>,
    clips_dir: PathBuf,
    assets_dir: PathBuf,
    /// Resolved clips by path (the registry owns the files; clones share the buffers).
    clips: HashMap<PathBuf, Clip>,
    selected_id: u32,
    selected_sequence_id: Option<u32>,
    reveal: HashSet<u32>,
    node_rename: InlineRename,
    sequence_rename: InlineRename,
    dragged_sequence_step: Option<(u32, usize)>,
    pivot_direction_edit: Option<(u32, [f32; 3])>,
    click_press: Option<Vec2>,
    left_was_held: bool,
    /// Per-component preview cache: a target pose is resampled only when its
    /// sampling inputs change, not every frame.
    previews: HashMap<u32, CachedPreview>,
}

/// A cached [`TargetSample`] and its trajectory, keyed by the sampling inputs.
struct CachedPreview {
    /// Inputs of [`sample_target`]: clip reference, sample time, hips constraint.
    signature: (Option<String>, f32, bool),
    /// The clip the sample and trajectory were built from.
    clip: Option<Clip>,
    sample: std::rc::Rc<TargetSample>,
    /// The trajectory and its own input (the spacing scale), sampled lazily.
    trajectory: Option<(f32, std::rc::Rc<Vec<rt::Transform>>)>,
}

impl SmartPrimitiveEditor {
    fn new(args: &Args) -> SmartPrimitiveEditor {
        SmartPrimitiveEditor {
            shell: EditorShell::new(EDITOR_TITLE, DOCUMENT_TYPE),
            document: new_smart_primitive_document(ModelRig::Soma),
            rig_data: None,
            clips_dir: args.clips.clone(),
            assets_dir: args.assets.clone(),
            clips: HashMap::new(),
            selected_id: 1,
            selected_sequence_id: None,
            reveal: HashSet::new(),
            node_rename: InlineRename::default(),
            sequence_rename: InlineRename::default(),
            dragged_sequence_step: None,
            pivot_direction_edit: None,
            click_press: None,
            left_was_held: false,
            previews: HashMap::new(),
        }
    }

    /// The selected node's ID, or the root's when nothing is selected
    /// (upstream `selected`).
    fn selected_id(&self) -> u32 {
        if self.document.find_node(self.selected_id).is_some() { self.selected_id } else { self.document.root.id }
    }

    /// The rig data for `rig`, loading them on first use; another rig replaces
    /// them (upstream `_rig_for`).
    fn rig_for(&mut self, rig: ModelRig) -> Result<&mut RigData> {
        let reload = match &self.rig_data {
            Some(rig_data) => rig_data.rig() != rig,
            None => true,
        };
        if reload {
            self.rig_data = Some(RigData::load(rig, &self.clips_dir, &self.assets_dir)?);
        }
        Ok(self.rig_data.as_mut().expect("rig data just loaded"))
    }

    /// Switch the document to another rig (upstream `_use_rig`): only the
    /// character selector calls this, and only when no target pose has a clip.
    fn use_rig(&mut self, rig: ModelRig) {
        match self.rig_for(rig) {
            Ok(_) => {
                self.document.rig = rig;
                self.shell.clear_status();
            }
            Err(error) => self.shell.set_status(format!("Character failed: {error:#}")),
        }
    }

    // -- Documents ----------------------------------------------------------

    fn new_document(&mut self) {
        let rig = self.document.rig;
        self.document = new_smart_primitive_document(rig);
        self.selected_id = self.document.root.id;
        self.selected_sequence_id = None;
        self.reveal.clear();
        self.node_rename = InlineRename::default();
        self.sequence_rename = InlineRename::default();
        self.dragged_sequence_step = None;
        self.pivot_direction_edit = None;
        self.previews.clear();
        self.shell.set_document_path(None);
    }

    fn open_document(&mut self, path: &Path, frame: &mut Frame) {
        let path = absolutize(path);
        let document = match SmartPrimitiveDocument::load(&path) {
            Ok(document) => document,
            Err(error) => {
                self.shell.set_status(format!("Open failed: {error:#}"));
                return;
            }
        };
        // Load the rig before replacing the current document (upstream
        // `_open_document`: a failed open keeps the old document).
        if let Err(error) = self.rig_for(document.rig) {
            self.shell.set_status(format!("Open failed: {error:#}"));
            return;
        }
        self.selected_id = document.selected_node_id;
        self.selected_sequence_id = document.sequences.first().map(|sequence| sequence.id);
        editor::apply_camera_state(frame.camera(), &document.camera);
        self.document = document;
        self.reveal.clear();
        self.node_rename = InlineRename::default();
        self.sequence_rename = InlineRename::default();
        self.dragged_sequence_step = None;
        self.pivot_direction_edit = None;
        self.previews.clear();
        self.shell.set_document_path(Some(path.clone()));
        eprintln!(
            "smart-primitive-editor: opened {} ({} target poses, {} sequences, rig {})",
            path.display(),
            self.document.target_poses().len(),
            self.document.sequences.len(),
            self.document.rig,
        );
    }

    fn save(&mut self, frame: &mut Frame) -> bool {
        let Some(path) = self.shell.document_path().map(Path::to_path_buf) else {
            return self.save_as();
        };
        self.save_to(&path, frame)
    }

    fn save_as(&mut self) -> bool {
        let initial = self
            .shell
            .document_path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| format!("untitled{SMART_PRIMITIVE_SUFFIX}"));
        self.shell.open_path_prompt("save_as", "Save Smart Primitive", true, initial);
        false
    }

    fn save_to(&mut self, path: &Path, frame: &mut Frame) -> bool {
        self.document.selected_node_id = self.selected_id();
        self.document.camera = editor::capture_camera_state(frame.camera());
        match self.document.save(path) {
            Ok(saved) => {
                self.shell.set_document_path(Some(saved));
                true
            }
            Err(error) => {
                self.shell.set_status(format!("Save failed: {error}"));
                false
            }
        }
    }

    fn handle_file_action(&mut self, action: FileAction, frame: &mut Frame) {
        match action {
            FileAction::New => self.new_document(),
            FileAction::Open => {
                let initial = self
                    .shell
                    .document_path()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| format!("smart_primitive{SMART_PRIMITIVE_SUFFIX}"));
                self.shell.open_path_prompt("open", "Open Smart Primitive", false, initial);
            }
            FileAction::Save => {
                self.save(frame);
            }
            FileAction::SaveAs => {
                self.save_as();
            }
            FileAction::Exit => self.shell.request_exit(),
        }
    }

    fn handle_path_prompt(&mut self, prompt: PathPrompt, frame: &mut Frame) {
        match prompt.id {
            "open" => self.open_document(Path::new(&prompt.text), frame),
            "save_as" => {
                self.save_to(Path::new(&prompt.text), frame);
            }
            "clip" => self.assign_clip(Path::new(&prompt.text)),
            _ => {}
        }
    }

    // -- Clips ---------------------------------------------------------------

    /// Resolve a clip path against this session's clip library roots
    /// (the `--clips` directory, then the upstream checkout's motions).
    fn resolve_clip(&self, resolved: &Path) -> PathBuf {
        let roots = [self.clips_dir.clone(), self.assets_dir.join("assets/motions")];
        resolve_clip_under(&roots, resolved)
    }

    /// The clip of a target pose, loading it on first use.
    fn clip_for(&mut self, clip_path: &Path) -> Result<&Clip> {
        let key = std::fs::canonicalize(clip_path).unwrap_or_else(|_| clip_path.to_path_buf());
        if !self.clips.contains_key(&key) {
            let rig = self.document.rig;
            let registry = &mut self.rig_for(rig)?.registry;
            let clip = match &registry.load(&key)?.clip {
                Clip::Soma(buffer) => Clip::Soma(buffer.clone()),
                Clip::G1(buffer) => Clip::G1(buffer.clone()),
            };
            self.clips.insert(key.clone(), clip);
        }
        Ok(self.clips.get(&key).expect("clip just loaded"))
    }

    /// The resolved, loaded clip of a target pose component.
    fn component_clip(&mut self, component: &TargetPoseComponent) -> Option<Result<Clip>> {
        let document_path = self.shell.document_path().map(Path::to_path_buf);
        let path = match &document_path {
            Some(document_path) => component.clip_path(document_path)?,
            None => PathBuf::from(component.clip.as_deref()?),
        };
        let resolved = self.resolve_clip(&path);
        Some(self.clip_for(&resolved).cloned())
    }

    /// The target pose's preview sample, resampling only when its inputs
    /// (clip, sample time, hips constraint) changed since the last call.
    fn preview(&mut self, node_id: u32, component: &TargetPoseComponent) -> Result<std::rc::Rc<TargetSample>> {
        let signature = (component.clip.clone(), component.sample_time_seconds, component.hips_constrained);
        if let Some(cached) = self.previews.get(&node_id) {
            if cached.signature == signature {
                return Ok(cached.sample.clone());
            }
        }
        let clip = match self.component_clip(component) {
            Some(result) => Some(result?),
            None => None,
        };
        let rig = self.rig_data.as_ref().context("rig data not loaded")?;
        let sample = std::rc::Rc::new(sample_target(rig, component, clip.as_ref())?);
        self.previews.insert(node_id, CachedPreview { signature, clip, sample: sample.clone(), trajectory: None });
        Ok(sample)
    }

    /// The trajectory preview, sampled lazily and cached by its scale.
    fn preview_trajectory(
        &mut self,
        node_id: u32,
        component: &TargetPoseComponent,
        sample: &TargetSample,
    ) -> Result<Option<std::rc::Rc<Vec<rt::Transform>>>> {
        if !sample.has_clip {
            return Ok(None);
        }
        if let Some(cached) = self.previews.get(&node_id) {
            if let Some((scale, trajectory)) = &cached.trajectory {
                if *scale == component.trajectory_scale {
                    return Ok(Some(trajectory.clone()));
                }
            }
        }
        let clip = self.previews.get(&node_id).and_then(|cached| cached.clip.clone()).context("a trajectory needs a clip")?;
        let rig = self.rig_data.as_ref().context("rig data not loaded")?;
        let transforms = std::rc::Rc::new(sample_trajectory(rig, component, &clip, sample.root_offset)?);
        if let Some(cached) = self.previews.get_mut(&node_id) {
            cached.trajectory = Some((component.trajectory_scale, transforms.clone()));
        }
        Ok(Some(transforms))
    }

    /// Assign a clip to the selected node's target pose (upstream
    /// `_draw_clip_selector`'s `component.set_clip`).
    fn assign_clip(&mut self, path: &Path) {
        let resolved = absolutize(path);
        let rig = self.document.rig;
        let suffix = match self.rig_for(rig) {
            Ok(rig_data) => rig_data.registry.suffix(),
            Err(error) => {
                self.shell.set_status(format!("Open failed: {error:#}"));
                return;
            }
        };
        if !resolved.to_string_lossy().to_lowercase().ends_with(suffix) {
            self.shell.set_status(format!("Open failed: choose a {suffix} clip."));
            return;
        }
        if let Err(error) = self.clip_for(&resolved) {
            self.shell.set_status(format!("Open failed: {error:#}"));
            return;
        }
        let selected_id = self.selected_id();
        let Some(node) = find_node_mut(&mut self.document.root, selected_id) else { return };
        let Some(component) = node.components.iter_mut().find_map(|component| match component {
            NodeComponent::TargetPose(component) => Some(component),
            _ => None,
        }) else {
            return;
        };
        // Store the path relative to the document when it has one (upstream's
        // clip references resolve against the document file).
        let stored = match self.shell.document_path() {
            Some(document_path) => relative_path(&resolved, document_path.parent().unwrap_or(Path::new("."))),
            None => resolved.to_string_lossy().into_owned(),
        };
        component.clip = Some(stored);
        component.sample_time_seconds = 0.0;
        component.frame_index = 0;
        self.shell.clear_status();
        self.shell.mark_dirty();
    }
}

/// `path` relative to `directory`, lexically (upstream writes clip paths
/// relative to the document file).
fn relative_path(path: &Path, directory: &Path) -> String {
    // Upstream writes clip paths relative to the document file.
    pathdiff::diff_paths(path, directory).unwrap_or_else(|| path.to_path_buf()).to_string_lossy().into_owned()
}

/// Resolve a clip path: as written (relative to the document), or — when that
/// file does not exist — under each clip library root, trying successively
/// shorter path tails (same rule as the Scene Editor's).
fn resolve_clip_under(roots: &[PathBuf], resolved: &Path) -> PathBuf {
    if resolved.is_file() {
        return resolved.to_path_buf();
    }
    let parts: Vec<&std::ffi::OsStr> = resolved
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part),
            _ => None,
        })
        .collect();
    for root in roots {
        for skip in 0..parts.len() {
            let candidate: PathBuf = root.join(parts[skip..].iter().collect::<PathBuf>());
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    resolved.to_path_buf()
}

// ---------------------------------------------------------------------------
// Frame loop: drawing, gizmo, picking
// ---------------------------------------------------------------------------

impl SmartPrimitiveEditor {
    fn frame(&mut self, frame: &mut Frame, ui: &mut imgui::Ui) {
        frame.log_ground();
        self.draw_scene(frame);
        let gizmo_dragging = self.draw_gizmo(frame);
        self.handle_picking(frame, ui, gizmo_dragging);

        if let Some(action) = self.shell.draw_main_menu_bar(ui) {
            self.handle_file_action(action, frame);
        }
        self.shell.draw_dockspace(ui, OUTLINER_WINDOW_TITLE);
        self.draw_outliner_window(ui);
        self.draw_sequences_window(ui);
        self.draw_properties_window(ui);
        if let Some(choice) = self.shell.draw_exit_confirmation(ui) {
            match choice {
                ExitChoice::SaveAndExit => {
                    if self.save(frame) {
                        self.shell.confirm_exit();
                    }
                }
            }
        }
        if let Some(prompt) = self.shell.draw_path_prompt(ui) {
            self.handle_path_prompt(prompt, frame);
        }
    }

    /// Draw the pivots, interaction boxes, and target poses (upstream
    /// `_draw_editor_scene`).
    fn draw_scene(&mut self, frame: &mut Frame) {
        let pivots = self.document.pivots();
        for (node_id, component) in pivots {
            let (Some(world), Some(visible)) = (self.document.world_transform(node_id), self.document.is_visible(node_id)) else {
                continue;
            };
            self.draw_pivot(frame, component, &world, visible);
        }
        let boxes = self.document.interaction_boxes();
        for (node_id, component) in boxes {
            let (Some(world), Some(true)) = (self.document.world_transform(node_id), self.document.is_visible(node_id)) else {
                continue;
            };
            let segments = editor::obb_wire_segments(
                &world_to_viewer(world.to_mat4()),
                Vec3::new(component.width, component.depth, component.height) * 0.5,
            );
            let [r, g, b] = component.color;
            frame.log_lines(&segments, [r, g, b, 1.0]);
        }
        let targets: Vec<(u32, TargetPoseComponent)> =
            self.document.target_poses().into_iter().map(|(node_id, component)| (node_id, component.clone())).collect();
        for (node_id, component) in targets {
            let (Some(world), Some(visible)) = (self.document.world_transform(node_id), self.document.is_visible(node_id)) else {
                continue;
            };
            self.draw_target_pose(frame, node_id, &component, &world, visible);
        }
    }

    /// Draw a pivot's motion direction arrow, translation line, and rotation
    /// cone (upstream `_PivotRenderer`).
    fn draw_pivot(&self, frame: &mut Frame, component: &PivotComponent, world: &Transform, visible: bool) {
        if visible && component.show_motion_direction {
            if let Some(direction) = component.motion_direction.try_normalize() {
                let origin = world.translation + Vec3::Z * PIVOT_ARROW_LIFT;
                let direction = world.rotation_xyzw * direction;
                frame.log_arrow(
                    world_point_to_viewer(origin),
                    world_point_to_viewer(origin + direction * PIVOT_ARROW_LENGTH),
                    [1.0, 0.8, 0.0, 1.0],
                );
            }
        }
        if visible
            && component.show_translation_constraint
            && component.translation_axis != Axis::None
            && component.translation_constraint == ConstraintType::LineProjection
        {
            if let Ok(pivot) = Pivot::from_component(component) {
                let (start, end) = pivot.line_endpoints(rt_transform(world));
                frame.log_lines(&[[world_point_to_viewer(start), world_point_to_viewer(end)]], [1.0, 0.0, 0.0, 1.0]);
            }
        }
        if visible
            && component.show_rotation_constraint
            && component.rotation_axis != Axis::None
            && component.rotation_constraint == ConstraintType::Cone
        {
            self.draw_pivot_cone(frame, component, world);
        }
    }

    /// The pivot's rotation constraint arcs (upstream `_update_pivot_cone_kernel`).
    fn draw_pivot_cone(&self, frame: &mut Frame, component: &PivotComponent, world: &Transform) {
        let Some(axis) = axis_vector(component.rotation_axis) else { return };
        let Some(direction) = component.motion_direction.try_normalize() else { return };
        let origin = world.translation;
        let direction = world.rotation_xyzw * direction * PIVOT_CONE_RADIUS;
        let axis = world.rotation_xyzw * axis;
        let half_angle = component.cone_half_angle_degrees.to_radians();
        let cone_count = if component.cone_side == ConeSide::Both { 2 } else { 1 };
        let first_center = if component.cone_side == ConeSide::Mirrored { std::f32::consts::PI } else { 0.0 };
        let mut segments = Vec::with_capacity(cone_count * (PIVOT_CONE_SEGMENTS + 2));
        for cone in 0..cone_count {
            let center = first_center + cone as f32 * std::f32::consts::PI;
            for segment in 0..PIVOT_CONE_SEGMENTS {
                let angle = center - half_angle + 2.0 * half_angle * segment as f32 / PIVOT_CONE_SEGMENTS as f32;
                let next_angle = angle + 2.0 * half_angle / PIVOT_CONE_SEGMENTS as f32;
                segments.push([
                    origin + Quat::from_axis_angle(axis, angle) * direction,
                    origin + Quat::from_axis_angle(axis, next_angle) * direction,
                ]);
            }
            // Join the arc's ends to the pivot. A full circle needs no closing lines.
            if half_angle < std::f32::consts::PI {
                for angle in [center - half_angle, center + half_angle] {
                    segments.push([origin, origin + Quat::from_axis_angle(axis, angle) * direction]);
                }
            }
        }
        let segments: Vec<[Vec3; 2]> = segments
            .iter()
            .map(|&[a, b]| [world_point_to_viewer(a), world_point_to_viewer(b)])
            .collect();
        frame.log_lines(&segments, [1.0, 0.0, 0.0, 1.0]);
    }

    /// Draw one target pose's previews (upstream `_TargetPoseRenderers.draw`).
    fn draw_target_pose(&mut self, frame: &mut Frame, node_id: u32, component: &TargetPoseComponent, world: &Transform, visible: bool) {
        if self.rig_data.is_none() {
            return;
        }
        let sample = match self.preview(node_id, component) {
            Ok(sample) => sample,
            Err(error) => {
                self.shell.set_status(format!("Clip failed: {error:#}"));
                return;
            }
        };
        let (model_to_runtime, is_soma) = {
            let rig = self.rig_data.as_ref().expect("checked above");
            (rig.model_to_runtime, rig.rig() == ModelRig::Soma)
        };
        let root = rt_transform(world) * model_to_runtime;
        let [r, g, b] = component.color;
        let color = [r, g, b, 1.0];

        if visible && component.show_mesh {
            self.draw_pose_mesh(frame, &sample.poses[0], root, color);
        }
        if visible && component.show_skeletons {
            let count = if sample.has_clip { component.num_poses_to_use as usize } else { 1 };
            for pose in sample.poses.iter().take(count.min(sample.poses.len())) {
                self.draw_pose_skeleton(frame, pose, root, color);
            }
        }
        if visible && component.show_velocity {
            let velocity = preview_velocity(&sample, component.num_poses_to_use);
            if velocity.length() > VELOCITY_EPSILON {
                let world_hips = root.point(rt::Vec3(sample.hips_position.to_array())).0;
                let world_velocity = root.vector(rt::Vec3(velocity.to_array())).0;
                let (start, end) = (Vec3::from_array(world_hips), Vec3::from_array(world_hips) + Vec3::from_array(world_velocity));
                frame.log_arrow(world_point_to_viewer(start), world_point_to_viewer(end), [1.0, 0.0, 0.0, 1.0]);
            }
        }
        if visible && component.show_trajectory && sample.has_clip {
            match self.preview_trajectory(node_id, component, &sample) {
                Ok(Some(transforms)) => self.draw_trajectory(frame, root, &transforms),
                Ok(None) => {}
                Err(error) => self.shell.set_status(format!("Clip failed: {error:#}")),
            }
        }
        if visible && component.ik_pin_mask != ik_pin::NONE && is_soma {
            self.draw_ik_pins(frame, component.ik_pin_mask, &sample.poses[0], root);
        }
    }

    /// The skinned SOMA mesh, or the G1 link meshes, of one sampled pose
    /// (upstream `_TargetPoseRenderers._draw_mesh`).
    fn draw_pose_mesh(&self, frame: &mut Frame, pose: &PoseSample, root: rt::Transform, color: [f32; 4]) {
        let rig = self.rig_data.as_ref().expect("rig data loaded");
        match (pose, &rig.soma, &rig.g1) {
            (PoseSample::Soma(local), Some(soma), _) => {
                let Ok(bones) = soma.evaluate(local, root) else { return };
                let skin = rig.soma_skin.as_ref().expect("SOMA rig data holds the skin");
                frame.log_skinned_mesh(&soma.mesh.mesh, skin, &bones, world_to_viewer_matrix(), color);
            }
            (PoseSample::G1(q), _, Some(g1)) => {
                let mut q = q.clone();
                rt::write_root(root * rt::read_root(&q), &mut q);
                let Ok(link_transforms) = g1.link_transforms(&q) else { return };
                for (index, link_transform) in link_transforms.iter().enumerate() {
                    let Some(mesh) = rig.g1_link_meshes.get(index) else { continue };
                    frame.log_mesh(mesh, world_to_viewer(transform_to_mat4(*link_transform)), color);
                }
            }
            _ => {}
        }
    }

    /// One sampled pose's skeleton as line segments between parent and child
    /// joints (upstream the skeleton renderer's lines).
    fn draw_pose_skeleton(&self, frame: &mut Frame, pose: &PoseSample, root: rt::Transform, color: [f32; 4]) {
        let rig = self.rig_data.as_ref().expect("rig data loaded");
        let mut segments = Vec::new();
        match (pose, &rig.soma, &rig.g1) {
            (PoseSample::Soma(local), Some(soma), _) => {
                let Ok(world) = soma.fk(local, root) else { return };
                let parents = soma.skeleton.parent_indices();
                for (joint, transform) in world.iter().enumerate().skip(1) {
                    let parent = parents[joint];
                    if parent < 0 {
                        continue;
                    }
                    segments.push([
                        world_point_to_viewer(Vec3::from_array(world[parent as usize].p.0)),
                        world_point_to_viewer(Vec3::from_array(transform.p.0)),
                    ]);
                }
            }
            (PoseSample::G1(q), _, Some(g1)) => {
                let mut q = q.clone();
                rt::write_root(root * rt::read_root(&q), &mut q);
                let Ok(link_transforms) = g1.link_transforms(&q) else { return };
                for (body, link_transform) in g1.bodies.iter().zip(link_transforms.iter()).skip(1) {
                    if body.parent < 0 {
                        continue;
                    }
                    let Some(parent_transform) = link_transforms.get(body.parent as usize) else { continue };
                    segments.push([
                        world_point_to_viewer(Vec3::from_array(parent_transform.p.0)),
                        world_point_to_viewer(Vec3::from_array(link_transform.p.0)),
                    ]);
                }
            }
            _ => {}
        }
        frame.log_lines(&segments, color);
    }

    /// The root path and small axis markers along it (upstream
    /// `_HipsTrajectoryRenderer`).
    fn draw_trajectory(&self, frame: &mut Frame, root: rt::Transform, transforms: &[rt::Transform]) {
        let world: Vec<Transform> = transforms.iter().map(|transform| glam_transform(root * *transform)).collect();
        let mut path = Vec::with_capacity(transforms.len().saturating_sub(1));
        for point in 0..transforms.len().saturating_sub(1) {
            path.push([
                world_point_to_viewer(world[point].translation),
                world_point_to_viewer(world[point + 1].translation),
            ]);
        }
        frame.log_lines(&path, TRAJECTORY_COLOR);
        let mut frames = Vec::with_capacity(transforms.len() * 3);
        let mut frame_colors = Vec::with_capacity(3);
        for (axis, color) in [(Vec3::X, [1.0, 0.0, 0.0, 1.0]), (Vec3::Y, [0.0, 1.0, 0.0, 1.0]), (Vec3::Z, [0.0, 0.0, 1.0, 1.0])] {
            for transform in &world {
                frames.push((
                    [
                        world_point_to_viewer(transform.translation),
                        world_point_to_viewer(transform.translation + transform.rotation_xyzw * axis * TRAJECTORY_FRAME_SCALE),
                    ],
                    color,
                ));
            }
            frame_colors.push(color);
        }
        for ([a, b], color) in frames {
            frame.log_lines(&[[a, b]], color);
        }
    }

    /// Mark pinned hands and feet on the first sampled pose (upstream
    /// `_IKPinRenderer`; small solid boxes stand in for its spheres).
    fn draw_ik_pins(&self, frame: &mut Frame, mask: u32, pose: &PoseSample, root: rt::Transform) {
        let rig = self.rig_data.as_ref().expect("rig data loaded");
        let (PoseSample::Soma(local), Some(soma)) = (pose, &rig.soma) else { return };
        let Ok(world) = soma.fk(local, root) else { return };
        for (pin, tip_index) in &rig.ik_tips {
            if mask & pin == 0 {
                continue;
            }
            let Some(tip) = world.get(*tip_index) else { continue };
            frame.log_solid_box(
                world_point_to_viewer(Vec3::from_array(tip.p.0)),
                Vec3::splat(IK_PIN_MARKER_HALF_EXTENT),
                IK_PIN_COLOR,
            );
        }
    }

    /// The transform gizmo on the selection (upstream `_draw_selection_gizmo`
    /// + `_apply_selection_gizmo`, fused: this viewer applies the gizmo inside
    ///   the frame). Returns true while the gizmo is dragged.
    fn draw_gizmo(&mut self, frame: &mut Frame) -> bool {
        let node_id = self.selected_id();
        let Some(world) = self.document.world_transform(node_id) else {
            return false;
        };
        if self.document.is_visible(node_id) != Some(true) {
            return false;
        }
        let mut gizmo_transform = world_to_viewer(world.to_mat4());
        let dragging = frame.gizmo_translate("selection", &mut gizmo_transform);
        let Ok(new_world) = Transform::from_mat4(&editor::viewer_to_world(gizmo_transform)) else {
            return dragging;
        };
        if transforms_equal(&world, &new_world) {
            return dragging;
        }
        let parent_world = self.document.find_parent(node_id).and_then(|parent| self.document.world_transform(parent.id));
        let edited_local = match parent_world {
            Some(parent_world) => parent_world.inverse().mul(&new_world),
            None => new_world,
        };
        let Some(node) = find_node_mut(&mut self.document.root, node_id) else {
            return dragging;
        };
        let (new_local, new_euler) = apply_transform_locks(
            &node.local_transform,
            &edited_local,
            node.euler_degrees,
            node.translation_locks,
            node.rotation_locks,
        );
        node.local_transform = new_local;
        node.euler_degrees = Some(new_euler);
        self.shell.mark_dirty();
        dragging
    }

    /// Click-picking (same press-and-release rule as the Scene Editor): a
    /// click outside the panels and the gizmo raycasts the target pose meshes
    /// and interaction boxes; the nearest hit is selected and revealed.
    fn handle_picking(&mut self, frame: &Frame, ui: &imgui::Ui, gizmo_dragging: bool) {
        let input = frame.input();
        let left_held = input.mouse_held(winit::event::MouseButton::Left);
        if input.mouse_pressed(winit::event::MouseButton::Left) && !ui.io().want_capture_mouse && !gizmo_dragging {
            self.click_press = input.cursor;
        }
        if self.left_was_held && !left_held {
            if let (Some(press), Some(cursor)) = (self.click_press.take(), input.cursor) {
                if cursor.distance(press) < 4.0 && !gizmo_dragging {
                    let ray = frame.screen_ray(cursor);
                    self.pick(&viewer_ray_to_world(&ray));
                }
            }
        }
        self.left_was_held = left_held;
    }

    /// Raycast every pickable node and select the nearest hit (upstream
    /// `_select_scene_item`).
    fn pick(&mut self, ray: &Ray) {
        let mut nearest = f32::INFINITY;
        let mut hit: Option<u32> = None;
        let targets: Vec<(u32, TargetPoseComponent)> =
            self.document.target_poses().into_iter().map(|(node_id, component)| (node_id, component.clone())).collect();
        for (node_id, component) in &targets {
            if self.document.is_visible(*node_id) != Some(true) || !component.show_mesh {
                continue;
            }
            let Some(world) = self.document.world_transform(*node_id) else { continue };
            if let Some(distance) = self.raycast_target(*node_id, component, &world, ray) {
                if distance < nearest {
                    nearest = distance;
                    hit = Some(*node_id);
                }
            }
        }
        for (node_id, component) in self.document.interaction_boxes() {
            if self.document.is_visible(node_id) != Some(true) {
                continue;
            }
            let Some(world) = self.document.world_transform(node_id) else { continue };
            let half = Vec3::new(component.width, component.depth, component.height) * 0.5;
            if let Some(distance) = ray_obb(ray, &world.to_mat4(), half) {
                if distance < nearest {
                    nearest = distance;
                    hit = Some(node_id);
                }
            }
        }
        if let Some(node_id) = hit {
            self.selected_id = node_id;
            self.reveal_selected();
        }
    }

    /// Raycast one target pose's preview mesh in world space.
    fn raycast_target(&mut self, node_id: u32, component: &TargetPoseComponent, world: &Transform, ray: &Ray) -> Option<f32> {
        let rig = self.rig_data.as_ref()?;
        let model_to_runtime = rig.model_to_runtime;
        let sample = self.preview(node_id, component).ok()?;
        let root = rt_transform(world) * model_to_runtime;
        let rig = self.rig_data.as_ref()?;
        match (&sample.poses[0], &rig.soma, &rig.g1) {
            (PoseSample::Soma(local), Some(soma), _) => {
                let bones = soma.evaluate(local, root).ok()?;
                let skin = rig.soma_skin.as_ref()?;
                let positions = skin_positions(&soma.mesh.mesh.positions, skin, &bones);
                ray_mesh(ray, &positions, &soma.mesh.mesh.indices)
            }
            (PoseSample::G1(q), _, Some(g1)) => {
                let mut q = q.clone();
                rt::write_root(root * rt::read_root(&q), &mut q);
                let link_transforms = g1.link_transforms(&q).ok()?;
                let mut nearest = None;
                for (index, link_transform) in link_transforms.iter().enumerate() {
                    let mesh = rig.g1_link_meshes.get(index)?;
                    let matrix = transform_to_mat4(*link_transform);
                    let positions: Vec<Vec3> = mesh.positions.iter().map(|&position| matrix.transform_point3(position)).collect();
                    if let Some(distance) = ray_mesh(ray, &positions, &mesh.indices) {
                        if nearest.is_none_or(|best| distance < best) {
                            nearest = Some(distance);
                        }
                    }
                }
                nearest
            }
            _ => None,
        }
    }

    /// Open the tree down to the selected node and scroll to it (upstream
    /// `_select_and_reveal_node`'s reveal half).
    fn reveal_selected(&mut self) {
        self.reveal.clear();
        let mut node = Some(self.selected_id());
        while let Some(node_id) = node {
            self.reveal.insert(node_id);
            node = self.document.find_parent(node_id).map(|parent| parent.id);
        }
    }
}

/// CPU linear-blend skinning of mesh positions (the picking half of
/// `viewer::Frame::log_skinned_mesh`, which does not keep its skinned soup).
fn skin_positions(positions: &[Vec3], skin: &Skin, bone_transforms: &[Mat4]) -> Vec<Vec3> {
    positions
        .iter()
        .enumerate()
        .map(|(vertex, &position)| {
            let mut matrix = Mat4::ZERO;
            for influence in 0..4 {
                let weight = skin.weight_values[vertex][influence];
                if weight == 0.0 {
                    continue;
                }
                let joint = skin.weights[vertex][influence] as usize;
                let bone = bone_transforms.get(joint).copied().unwrap_or(Mat4::IDENTITY);
                let inverse_bind = skin.inverse_bind.get(joint).copied().unwrap_or(Mat4::IDENTITY);
                matrix += bone * inverse_bind * weight;
            }
            matrix.transform_point3(position)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// UI: the Hierarchy, Sequences, and Properties panels
// ---------------------------------------------------------------------------

/// Node row data, cloned so the tree draws without holding a document borrow.
struct NodeRow {
    id: u32,
    name: String,
    visible: bool,
    children: Vec<u32>,
}

fn node_row(node: &SceneNode) -> NodeRow {
    NodeRow {
        id: node.id,
        name: node.name.clone(),
        visible: node.visible,
        children: node.children.iter().map(|child| child.id).collect(),
    }
}

impl SmartPrimitiveEditor {
    /// The Hierarchy window (upstream `_draw_outliner_window`).
    fn draw_outliner_window(&mut self, ui: &imgui::Ui) {
        let (pos, size) = initial_panel_geometry(ui, HIERARCHY_WINDOW_WIDTH, HIERARCHY_WINDOW_HEIGHT, WINDOW_MARGIN, 1.0);
        let mut delete_id: Option<u32> = None;
        ui.window(OUTLINER_WINDOW_TITLE)
            .position(pos, imgui::Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size(size, imgui::Condition::FirstUseEver)
            .bg_alpha(PANEL_ALPHA)
            .build(|| {
                self.draw_character_selector(ui);
                let selected = self.selected_id();
                let can_remove = selected != self.document.root.id;
                let (add, remove_selected) = draw_add_remove_buttons(ui, can_remove);
                if add {
                    self.selected_id = add_node(&mut self.document, Some(selected));
                    self.shell.mark_dirty();
                }
                if remove_selected {
                    delete_id = Some(selected);
                }
                self.shell.draw_status(ui);
                ui.separator();
                let root_id = self.document.root.id;
                delete_id = delete_id.or(self.draw_tree_node(ui, root_id));
            });
        if let Some(node_id) = delete_id {
            let removed = remove_node(&mut self.document, node_id);
            if !removed.is_empty() {
                self.selected_id = self.document.root.id;
                self.node_rename.item_id = None;
                self.shell.mark_dirty();
            }
        }
    }

    /// The rig selector, locked while a target pose has a clip (upstream
    /// `_draw_character_selector`).
    fn draw_character_selector(&mut self, ui: &imgui::Ui) {
        let has_clips = self.document.target_poses().into_iter().any(|(_, component)| component.clip.is_some());
        let rig = draw_character_selector(
            ui,
            self.document.rig,
            if has_clips {
                Some(
                    "Target Poses with clips lock the Character selector. Remove those Target Poses or start a new \
                     document to change it.",
                )
            } else {
                None
            },
        );
        if rig != self.document.rig {
            self.use_rig(rig);
        }
    }

    /// One hierarchy tree row and its children (upstream `_draw_tree_node`).
    /// Returns the ID to delete, if the context menu asked for it.
    fn draw_tree_node(&mut self, ui: &imgui::Ui, node_id: u32) -> Option<u32> {
        let row = node_row(self.document.find_node(node_id)?);
        let _id = ui.push_id_usize(row.id as usize);
        let mut delete_id = None;
        {
            let node = find_node_mut(&mut self.document.root, row.id)?;
            ui.checkbox("##Visible", &mut node.visible);
        }
        ui.same_line();
        let is_root = row.id == self.document.root.id;
        let renaming = self.node_rename.item_id == Some(row.id);
        let selected_id = self.selected_id();
        let tree_row = draw_tree_row(
            ui,
            &row.name,
            row.id,
            &mut self.node_rename,
            &mut self.reveal,
            row.id == selected_id,
            row.children.is_empty(),
            true,
            !row.visible,
        );
        if tree_row.clicked {
            self.selected_id = row.id;
        }
        if let Some(name) = tree_row.new_name {
            if let Some(node) = find_node_mut(&mut self.document.root, row.id) {
                node.name = name;
                self.shell.mark_dirty();
            }
        }
        if !renaming {
            if !is_root {
                let _ = drag_source(ui, TREE_PAYLOAD, row.id, &row.name);
            }
            if let Some(dragged_id) = accept_drop(ui, TREE_PAYLOAD) {
                if reparent(&mut self.document, dragged_id, row.id) {
                    self.selected_id = dragged_id;
                    self.shell.mark_dirty();
                }
            }
            if let Some(_popup) = ui.begin_popup_context_with_label("##NodeContext") {
                self.selected_id = row.id;
                if ui.menu_item_config("Delete").enabled(!is_root).build() {
                    delete_id = Some(row.id);
                }
                if ui.menu_item("Add Child") {
                    self.selected_id = add_node(&mut self.document, Some(row.id));
                    self.shell.mark_dirty();
                }
            }
        }
        if let Some(token) = tree_row.token {
            for child_id in &row.children {
                delete_id = delete_id.or(self.draw_tree_node(ui, *child_id));
            }
            token.pop();
        }
        delete_id
    }

    /// The Sequences window (upstream `_draw_sequences_window`).
    fn draw_sequences_window(&mut self, ui: &imgui::Ui) {
        let (pos, size) = initial_panel_geometry(
            ui,
            SEQUENCES_WINDOW_WIDTH,
            SEQUENCES_WINDOW_HEIGHT,
            HIERARCHY_WINDOW_HEIGHT + 2.0 * WINDOW_MARGIN,
            1.0,
        );
        let mut sequence_to_remove: Option<u32> = None;
        ui.window("Sequences")
            .position(pos, imgui::Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size(size, imgui::Condition::FirstUseEver)
            .bg_alpha(PANEL_ALPHA)
            .build(|| {
                let selected_sequence = self
                    .selected_sequence_id
                    .and_then(|id| self.document.sequences.iter().find(|sequence| sequence.id == id))
                    .map(|sequence| sequence.id);
                let (add, remove_selected) = draw_add_remove_buttons(ui, selected_sequence.is_some());
                if add {
                    let id = add_sequence(&mut self.document);
                    self.selected_sequence_id = Some(id);
                    let name = self.document.sequences.iter().find(|s| s.id == id).expect("just added").name.clone();
                    self.sequence_rename.begin(id, &name);
                    self.shell.mark_dirty();
                }
                if remove_selected {
                    sequence_to_remove = selected_sequence;
                }
                ui.separator();
                if self.document.sequences.is_empty() {
                    ui.text_disabled("No target pose sequences.");
                }
                let sequence_ids: Vec<u32> = self.document.sequences.iter().map(|sequence| sequence.id).collect();
                for sequence_id in sequence_ids {
                    if self.draw_sequence(ui, sequence_id) {
                        sequence_to_remove = Some(sequence_id);
                    }
                }
            });
        if let Some(sequence_id) = sequence_to_remove {
            self.document.sequences.retain(|sequence| sequence.id != sequence_id);
            self.selected_sequence_id = None;
            self.sequence_rename.item_id = None;
            self.shell.mark_dirty();
        }
    }

    /// One sequence: header, pivot row, steps, and the add popup (upstream
    /// `_draw_sequence`). Returns true when the sequence should be deleted.
    fn draw_sequence(&mut self, ui: &imgui::Ui, sequence_id: u32) -> bool {
        let _id = ui.push_id_usize(sequence_id as usize);
        let Some(index) = self.document.sequences.iter().position(|sequence| sequence.id == sequence_id) else {
            return false;
        };
        let mut flags = imgui::TreeNodeFlags::DEFAULT_OPEN;
        if self.selected_sequence_id == Some(sequence_id) {
            flags |= imgui::TreeNodeFlags::SELECTED;
        }
        let name = self.document.sequences[index].name.clone();
        let sequence_open = ui.collapsing_header(format!("{name}###Sequence"), flags);
        if ui.is_item_clicked() {
            self.selected_sequence_id = Some(sequence_id);
        }
        // Drop a hierarchy node with a Target Pose on the header to append it.
        if let Some(node_id) = accept_drop(ui, TREE_PAYLOAD) {
            if self.document.find_node(node_id).is_some_and(|node| node.target_pose().is_some()) {
                self.document.sequences[index].node_ids.push(node_id);
                self.selected_id = node_id;
                self.reveal_selected();
                self.selected_sequence_id = Some(sequence_id);
                self.shell.mark_dirty();
            }
        }
        let mut delete_sequence = false;
        if let Some(_popup) = ui.begin_popup_context_with_label("##SequenceContext") {
            self.selected_sequence_id = Some(sequence_id);
            if ui.menu_item("Rename") {
                self.sequence_rename.begin(sequence_id, &name);
            }
            if ui.menu_item("Delete") {
                delete_sequence = true;
            }
        }
        if self.sequence_rename.item_id == Some(sequence_id) {
            if let Some(new_name) = self.sequence_rename.draw(ui) {
                if self.document.sequences.iter().any(|sequence| sequence.id != sequence_id && sequence.name == new_name) {
                    self.shell.set_status(format!("Cannot rename the sequence. {new_name:?} is already in use."));
                } else {
                    self.document.sequences[index].name = new_name;
                    self.shell.clear_status();
                    self.shell.mark_dirty();
                }
            }
        }
        if sequence_open {
            self.draw_sequence_pivot(ui, sequence_id);
            let steps = self.document.sequences[index].node_ids.clone();
            let mut remove_index = None;
            for (step_index, node_id) in steps.iter().enumerate() {
                if self.draw_sequence_step(ui, sequence_id, step_index, *node_id) {
                    remove_index = Some(step_index);
                }
            }
            if let Some(step_index) = remove_index {
                self.document.sequences[index].node_ids.remove(step_index);
                self.shell.mark_dirty();
            }
            if ui.button("+ Add Target Pose") {
                ui.open_popup("##AddTargetPose");
            }
            self.draw_add_target_pose_popup(ui, sequence_id);
        }
        delete_sequence
    }

    /// The sequence's pivot combo (upstream `_draw_sequence_pivot`).
    fn draw_sequence_pivot(&mut self, ui: &imgui::Ui, sequence_id: u32) {
        let index = self.document.sequences.iter().position(|sequence| sequence.id == sequence_id).expect("drawn sequence");
        let pivot_node_id = self.document.sequences[index].pivot_node_id;
        let pivot_name = pivot_node_id.and_then(|id| self.document.find_node(id)).map(|node| node.name.clone());
        draw_label(ui, "Pivot", None, 0.0, -1.0);
        let preview = pivot_name.clone().unwrap_or_else(|| "None".to_string());
        if let Some(_combo) = ui.begin_combo("##Pivot", &preview) {
            let pivot_nodes: Vec<(u32, String)> = self
                .document
                .pivots()
                .into_iter()
                .filter_map(|(node_id, _)| self.document.find_node(node_id).map(|node| (node_id, node.name.clone())))
                .collect();
            let mut choices: Vec<(Option<u32>, String)> = vec![(None, "None".to_string())];
            choices.extend(pivot_nodes.into_iter().map(|(id, name)| (Some(id), name)));
            for (choice_id, label) in choices {
                if ui.selectable_config(format!("{label}##{choice_id:?}")).selected(choice_id == pivot_node_id).build() {
                    self.document.sequences[index].pivot_node_id = choice_id;
                    self.selected_sequence_id = Some(sequence_id);
                    if let Some(node_id) = choice_id {
                        self.selected_id = node_id;
                        self.reveal_selected();
                    }
                    self.shell.mark_dirty();
                }
            }
        } else {
            // A hierarchy node dropped on the closed combo becomes the pivot.
            if let Some(node_id) = accept_drop(ui, TREE_PAYLOAD) {
                if self.document.find_node(node_id).is_some_and(|node| node.pivot().is_some()) {
                    self.document.sequences[index].pivot_node_id = Some(node_id);
                    self.selected_id = node_id;
                    self.reveal_selected();
                    self.selected_sequence_id = Some(sequence_id);
                    self.shell.mark_dirty();
                }
            }
        }
        if ui.is_item_clicked() {
            if let Some(node_id) = pivot_node_id {
                self.selected_id = node_id;
                self.reveal_selected();
                self.selected_sequence_id = Some(sequence_id);
            }
        }
    }

    /// One sequence step row (upstream `_draw_sequence_step`). Returns true
    /// when its remove button was pressed.
    fn draw_sequence_step(&mut self, ui: &imgui::Ui, sequence_id: u32, step_index: usize, node_id: u32) -> bool {
        let Some(node) = self.document.find_node(node_id) else { return false };
        let name = node.name.clone();
        let _id = ui.push_id_usize(step_index);
        let available_width = ui.content_region_avail()[0];
        let remove_width = ui.calc_text_size("x")[0] + 20.0;
        let clicked = ui
            .selectable_config(format!("{} - {name}##TargetPose", step_index + 1))
            .selected(node_id == self.selected_id())
            .size([(available_width - remove_width).max(1.0), 0.0])
            .build();
        if clicked {
            self.selected_id = node_id;
            self.reveal_selected();
            self.selected_sequence_id = Some(sequence_id);
        }
        if drag_source(ui, SEQUENCE_STEP_PAYLOAD, node_id, &format!("{} - {name}", step_index + 1)) {
            self.dragged_sequence_step = Some((sequence_id, step_index));
        }
        if accept_drop(ui, SEQUENCE_STEP_PAYLOAD).is_some() {
            if let Some((dragged_sequence, source_index)) = self.dragged_sequence_step.take() {
                if dragged_sequence == sequence_id && source_index != step_index {
                    let index = self.document.sequences.iter().position(|s| s.id == sequence_id).expect("drawn sequence");
                    move_sequence_step(&mut self.document.sequences[index], source_index, step_index);
                    self.shell.mark_dirty();
                }
            }
        }
        ui.same_line();
        ui.small_button("x")
    }

    /// The "+ Add Target Pose" popup (upstream `_draw_add_target_pose_popup`).
    fn draw_add_target_pose_popup(&mut self, ui: &imgui::Ui, sequence_id: u32) {
        let Some(_popup) = ui.begin_popup("##AddTargetPose") else { return };
        let targets: Vec<(u32, String)> = self
            .document
            .target_poses()
            .into_iter()
            .filter_map(|(node_id, _)| self.document.find_node(node_id).map(|node| (node_id, node.name.clone())))
            .collect();
        if targets.is_empty() {
            ui.text_disabled("No hierarchy nodes own a Target Pose.");
        }
        for (node_id, name) in targets {
            if ui.selectable(format!("{name}###{node_id}")) {
                let index = self.document.sequences.iter().position(|s| s.id == sequence_id).expect("drawn sequence");
                self.document.sequences[index].node_ids.push(node_id);
                self.selected_id = node_id;
                self.reveal_selected();
                self.selected_sequence_id = Some(sequence_id);
                self.shell.mark_dirty();
                ui.close_current_popup();
            }
        }
    }

    /// The Properties window (upstream `_draw_properties_window`).
    fn draw_properties_window(&mut self, ui: &imgui::Ui) {
        let (pos, size) = initial_panel_geometry(
            ui,
            PROPERTIES_WINDOW_WIDTH,
            PROPERTIES_WINDOW_HEIGHT,
            HIERARCHY_WINDOW_HEIGHT + SEQUENCES_WINDOW_HEIGHT + 3.0 * WINDOW_MARGIN,
            1.0,
        );
        let node_id = self.selected_id();
        let mut remove_component_id: Option<u32> = None;
        ui.window("Properties")
            .position(pos, imgui::Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size(size, imgui::Condition::FirstUseEver)
            .bg_alpha(PANEL_ALPHA)
            .build(|| {
                let _id = ui.push_id_usize(node_id as usize);
                let (name, has_target, has_box, has_pivot, components_empty) = match self.document.find_node(node_id) {
                    Some(node) => (
                        node.name.clone(),
                        node.target_pose().is_some(),
                        node.interaction_box().is_some(),
                        node.pivot().is_some(),
                        node.components.is_empty(),
                    ),
                    None => return,
                };
                ui.text_colored(editor::widgets::NAME_TEXT_COLOR, &name);
                if ui.collapsing_header("Transform", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    self.draw_transform_details(ui, node_id);
                }
                if has_target && self.draw_target_pose_component(ui, node_id) {
                    remove_component_id = self.document.find_node(node_id).and_then(|node| node.target_pose()).map(|c| c.id);
                }
                if has_box && self.draw_interaction_box_component(ui, node_id) {
                    remove_component_id = self.document.find_node(node_id).and_then(|node| node.interaction_box()).map(|c| c.id);
                }
                if has_pivot && self.draw_pivot_component(ui, node_id) {
                    remove_component_id = self.document.find_node(node_id).and_then(|node| node.pivot()).map(|c| c.id);
                }
                if components_empty {
                    ui.text_disabled("Right click empty space to add a component.");
                }
                self.draw_sequence_references(ui, node_id);
                self.draw_add_component_popup(ui, node_id);
            });
        if let Some(component_id) = remove_component_id {
            remove_component(&mut self.document, node_id, component_id);
            self.shell.mark_dirty();
        }
    }

    /// The node's transform editor (upstream `_draw_transform_details`).
    fn draw_transform_details(&mut self, ui: &imgui::Ui, node_id: u32) {
        let Some(node) = find_node_mut(&mut self.document.root, node_id) else { return };
        let edit = draw_transform_editor(
            ui,
            &node.local_transform,
            node.euler_degrees,
            1.0,
            &mut node.translation_locks,
            &mut node.rotation_locks,
        );
        if edit.changed {
            node.local_transform = edit.transform;
            node.euler_degrees = Some(edit.euler_degrees);
            self.shell.mark_dirty();
        }
    }

    /// The "Referenced by" section (upstream `_draw_sequence_references`).
    fn draw_sequence_references(&mut self, ui: &imgui::Ui, node_id: u32) {
        let referencing = sequences_referencing(&self.document, node_id);
        if referencing.is_empty() {
            return;
        }
        if !ui.collapsing_header("Referenced by", imgui::TreeNodeFlags::DEFAULT_OPEN) {
            return;
        }
        for sequence_id in referencing {
            let Some(sequence) = self.document.sequences.iter().find(|sequence| sequence.id == sequence_id) else {
                continue;
            };
            let occurrences = sequence.node_ids.iter().filter(|&&id| id == node_id).count();
            let mut suffix = String::new();
            if occurrences > 1 {
                suffix = format!(" ({occurrences} steps)");
            }
            if sequence.pivot_node_id == Some(node_id) {
                suffix = " (Pivot)".to_string();
            }
            let name = sequence.name.clone();
            if ui
                .selectable_config(format!("{name}{suffix}###{sequence_id}"))
                .selected(self.selected_sequence_id == Some(sequence_id))
                .build()
            {
                self.selected_sequence_id = Some(sequence_id);
            }
        }
    }

    /// The right-click "Add / Components" popup (upstream
    /// `_draw_add_component_popup`).
    fn draw_add_component_popup(&mut self, ui: &imgui::Ui, node_id: u32) {
        let Some(_popup) = ui.begin_popup_context_window_with_label("##PropertiesContext") else { return };
        if let Some(_menu) = ui.begin_menu("Add") {
            if let Some(_submenu) = ui.begin_menu("Components") {
                let node = self.document.find_node(node_id).expect("selected node");
                let (has_target, has_box, has_pivot) =
                    (node.target_pose().is_some(), node.interaction_box().is_some(), node.pivot().is_some());
                if ui.menu_item_config("Target Pose").enabled(!has_target).build() {
                    add_component(&mut self.document, node_id, "target_pose");
                    self.shell.mark_dirty();
                }
                if ui.menu_item_config("Interaction Box").enabled(!has_box).build() {
                    add_component(&mut self.document, node_id, "interaction_box");
                    self.shell.mark_dirty();
                }
                if ui.menu_item_config("Pivot").enabled(!has_pivot).build() {
                    add_component(&mut self.document, node_id, "pivot");
                    self.shell.mark_dirty();
                }
            }
        }
    }

    /// A component header with its remove context menu (upstream
    /// `_draw_component_header`): (open, remove).
    fn draw_component_header(ui: &imgui::Ui, label: &str, component_id: u32) -> (bool, bool) {
        let open = ui.collapsing_header(format!("{label}###{component_id}"), imgui::TreeNodeFlags::DEFAULT_OPEN);
        let mut remove = false;
        if let Some(_popup) = ui.begin_popup_context_with_label(format!("##{label}Context{component_id}")) {
            if ui.menu_item("Remove Component") {
                remove = true;
            }
        }
        (open, remove)
    }

    /// The Target Pose section (upstream `_draw_target_pose_component`).
    /// Returns true when the component should be removed.
    fn draw_target_pose_component(&mut self, ui: &imgui::Ui, node_id: u32) -> bool {
        let component_id = self.document.find_node(node_id).and_then(|node| node.target_pose()).map(|c| c.id).expect("drawn component");
        let (open, remove) = Self::draw_component_header(ui, "Target Pose", component_id);
        if !open || remove {
            return remove;
        }
        self.draw_clip_row(ui, node_id, component_id);
        self.draw_clip_scrubber(ui, node_id);
        {
            let Some(node) = find_node_mut(&mut self.document.root, node_id) else { return false };
            let Some(component) = node.components.iter_mut().find_map(|c| match c {
                NodeComponent::TargetPose(c) => Some(c),
                _ => None,
            }) else { return false };
            if ui.checkbox("Constrain Root", &mut component.hips_constrained) {
                self.shell.mark_dirty();
            }
        }
        if self.document.rig == ModelRig::Soma && self.has_clip(node_id) {
            self.draw_ik_pin_selector(ui, node_id, component_id);
        }
        ui.text(format!("Speed: {:.3} m/s", self.preview_speed(node_id)));
        ui.separator();
        self.draw_sampling_options(ui, node_id, component_id);
        ui.separator();
        self.draw_visualization_options(ui, node_id, component_id);
        false
    }

    /// The clip row: name, tooltip path, and the load button (upstream
    /// `_draw_clip_selector`).
    fn draw_clip_row(&mut self, ui: &imgui::Ui, node_id: u32, component_id: u32) {
        let component = self.document.find_node(node_id).and_then(|node| node.target_pose()).expect("drawn component").clone();
        let (name, path) = match &component.clip {
            Some(clip) => {
                let document_path = self.shell.document_path().map(Path::to_path_buf);
                let resolved = document_path.and_then(|path| component.clip_path(path));
                let display = resolved.clone().unwrap_or_else(|| PathBuf::from(clip));
                let name = display.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                (name, Some(display))
            }
            None => ("None".to_string(), None),
        };
        if draw_clip_selector(ui, &name, path.as_deref(), component_id, "Clip") {
            let initial = path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| self.clips_dir.display().to_string());
            self.selected_id = node_id;
            self.shell.open_path_prompt("clip", "Open Clip", false, initial);
        }
    }

    /// Whether the node's target pose has a clip that loads.
    fn has_clip(&mut self, node_id: u32) -> bool {
        let component = match self.document.find_node(node_id).and_then(|node| node.target_pose()) {
            Some(component) => component.clone(),
            None => return false,
        };
        matches!(self.component_clip(&component), Some(Ok(_)))
    }

    /// The preview velocity's magnitude, for the Speed line.
    fn preview_speed(&mut self, node_id: u32) -> f32 {
        let Some(component) = self.document.find_node(node_id).and_then(|node| node.target_pose()).cloned() else {
            return 0.0;
        };
        if self.rig_data.is_none() {
            return 0.0;
        }
        let Ok(sample) = self.preview(node_id, &component) else { return 0.0 };
        preview_velocity(&sample, component.num_poses_to_use).length()
    }

    /// The sample time/frame scrubber (upstream `draw_clip_scrubber` in
    /// `authoring/common.py`, inlined: it is this editor's only user).
    fn draw_clip_scrubber(&mut self, ui: &imgui::Ui, node_id: u32) {
        let component = self.document.find_node(node_id).and_then(|node| node.target_pose()).expect("drawn component").clone();
        let clip = self.component_clip(&component).and_then(|result| result.ok());
        let num_frames = clip.as_ref().map(|clip| clip.num_frames()).unwrap_or(0);
        let sample_rate = clip.as_ref().map(|clip| clip.sample_rate()).unwrap_or(1.0);
        let last_frame = num_frames.saturating_sub(1) as i32;
        let duration = last_frame as f32 / sample_rate;
        let mut sample_time = component.sample_time_seconds;
        let mut show_frames = component.playback_mode == PlaybackMode::Frames;
        let original_time = sample_time;
        let mut frame = sample_frame_at(sample_time, sample_rate, last_frame.max(0) as u32) as i32;

        let mut changed = false;
        {
            let _disabled = ui.begin_disabled(clip.is_none());
            if ui.button_with_size(if show_frames { "Frames" } else { "Seconds" }, [65.0, 0.0]) {
                show_frames = !show_frames;
                if show_frames {
                    sample_time = frame as f32 / sample_rate;
                }
                changed = true;
            }
            ui.same_line();
            ui.set_next_item_width(250.0);
            if show_frames {
                if ui
                    .slider_config("##SamplePosition", 0, last_frame)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .build(&mut frame)
                {
                    sample_time = frame as f32 / sample_rate;
                    changed = true;
                }
                ui.same_line();
                ui.text(format!("{num_frames}"));
            } else {
                if ui
                    .slider_config("##SamplePosition", 0.0, duration)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .display_format("%.2f")
                    .build(&mut sample_time)
                {
                    changed = true;
                }
                ui.same_line();
                ui.text(format!("{duration:.2} s"));
            }
        }
        if changed || sample_time != original_time {
            let Some(node) = find_node_mut(&mut self.document.root, node_id) else { return };
            let Some(component) = node.components.iter_mut().find_map(|c| match c {
                NodeComponent::TargetPose(c) => Some(c),
                _ => None,
            }) else { return };
            component.playback_mode = if show_frames { PlaybackMode::Frames } else { PlaybackMode::Time };
            component.sample_time_seconds = sample_time.clamp(0.0, duration);
            if show_frames {
                component.frame_index = frame as u32;
            }
            self.shell.mark_dirty();
        }
    }

    /// The IK pin multi-select combo (upstream `_draw_target_pose_ik_pins`).
    fn draw_ik_pin_selector(&mut self, ui: &imgui::Ui, node_id: u32, component_id: u32) {
        let mask = self.document.find_node(node_id).and_then(|node| node.target_pose()).map(|c| c.ik_pin_mask).unwrap_or(0);
        let selected: Vec<&str> = IK_PINS.iter().filter(|(pin, _)| mask & pin != 0).map(|(_, label)| *label).collect();
        let preview = if selected.is_empty() {
            "None".to_string()
        } else if selected.len() == IK_PINS.len() {
            "All".to_string()
        } else {
            selected.join(", ")
        };
        draw_label(ui, "IK Pins", None, 80.0, -1.0);
        let Some(_combo) = ui.begin_combo(format!("##IKPins{component_id}"), &preview) else { return };
        let mut new_mask = mask;
        for (pin, label) in IK_PINS {
            let mut enabled = new_mask & pin != 0;
            if ui.checkbox(format!("{label}##IKPin{pin}_{component_id}"), &mut enabled) {
                if enabled {
                    new_mask |= pin;
                } else {
                    new_mask &= !pin;
                }
            }
        }
        if new_mask != mask {
            if let Some(component) = find_node_mut(&mut self.document.root, node_id).and_then(|node| {
                node.components.iter_mut().find_map(|c| match c {
                    NodeComponent::TargetPose(c) => Some(c),
                    _ => None,
                })
            }) {
                component.ik_pin_mask = new_mask;
                self.shell.mark_dirty();
            }
        }
    }

    /// Poses to Use, Transition Early By, and Tokens to Generate (upstream
    /// `_draw_target_pose_sampling_options`).
    fn draw_sampling_options(&mut self, ui: &imgui::Ui, node_id: u32, component_id: u32) {
        let component = self.document.find_node(node_id).and_then(|node| node.target_pose()).expect("drawn component");
        let num_poses = draw_int_slider_property(
            ui,
            "Poses to Use",
            component.num_poses_to_use as i32,
            1,
            MODEL_POSES_PER_TOKEN as i32,
            component_id,
            Some(POSES_TO_USE_TOOLTIP),
        );
        let early = draw_int_slider_property(
            ui,
            "Transition Early By (frames)",
            component.transition_early_by_frames as i32,
            0,
            20,
            component_id,
            Some("Move on to the next step this many frames before the generated motion ends."),
        );
        let (min_tokens, max_tokens) =
            draw_token_range_property(ui, component.min_tokens, component.max_tokens, component_id);
        let changed = num_poses as u32 != component.num_poses_to_use
            || early as u32 != component.transition_early_by_frames
            || min_tokens != component.min_tokens
            || max_tokens != component.max_tokens;
        if changed {
            let Some(component) = find_node_mut(&mut self.document.root, node_id).and_then(|node| {
                node.components.iter_mut().find_map(|c| match c {
                    NodeComponent::TargetPose(c) => Some(c),
                    _ => None,
                })
            }) else { return };
            component.num_poses_to_use = (num_poses as u32).clamp(1, MODEL_POSES_PER_TOKEN);
            component.transition_early_by_frames = (early as u32).min(20);
            (component.min_tokens, component.max_tokens) = clamp_token_range(min_tokens, max_tokens);
            self.shell.mark_dirty();
        }
    }

    /// Color and visibility toggles (upstream
    /// `_draw_target_pose_visualization_options`).
    fn draw_visualization_options(&mut self, ui: &imgui::Ui, node_id: u32, component_id: u32) {
        let Some(component) = find_node_mut(&mut self.document.root, node_id).and_then(|node| {
            node.components.iter_mut().find_map(|c| match c {
                NodeComponent::TargetPose(c) => Some(c),
                _ => None,
            })
        }) else { return };
        let mut color = component.color;
        if ui.color_edit3_config("Color", &mut color).flags(imgui::ColorEditFlags::NO_INPUTS).build() {
            component.color = color;
            self.shell.mark_dirty();
        }
        let mut changed = false;
        changed |= ui.checkbox("Show Mesh", &mut component.show_mesh);
        changed |= ui.checkbox("Show Skeleton(s)", &mut component.show_skeletons);
        changed |= ui.checkbox("Show Velocity", &mut component.show_velocity);
        changed |= ui.checkbox("Show Trajectory", &mut component.show_trajectory);
        ui.same_line();
        ui.set_next_item_width(-1.0);
        changed |= ui
            .slider_config(format!("##TrajectoryScale{component_id}"), MIN_TRAJECTORY_SCALE, MAX_TRAJECTORY_SCALE)
            .flags(imgui::SliderFlags::ALWAYS_CLAMP)
            .display_format("x %.1f")
            .build(&mut component.trajectory_scale);
        if changed {
            self.shell.mark_dirty();
        }
    }

    /// The Interaction Box section (upstream `_draw_interaction_box_component`).
    fn draw_interaction_box_component(&mut self, ui: &imgui::Ui, node_id: u32) -> bool {
        let component_id =
            self.document.find_node(node_id).and_then(|node| node.interaction_box()).map(|c| c.id).expect("drawn component");
        let (open, remove) = Self::draw_component_header(ui, "Interaction Box", component_id);
        if !open || remove {
            return remove;
        }
        let Some(component) = find_node_mut(&mut self.document.root, node_id).and_then(|node| {
            node.components.iter_mut().find_map(|c| match c {
                NodeComponent::InteractionBox(c) => Some(c),
                _ => None,
            })
        }) else { return false };
        let (changed, dimensions) = draw_box_dimensions(ui, [component.width, component.depth, component.height], MIN_INTERACTION_BOX_SIZE, 1.0);
        if changed {
            // Upstream `set_dimensions`: finite and non-negative.
            let [width, depth, height] = dimensions;
            component.width = width.max(MIN_INTERACTION_BOX_SIZE);
            component.depth = depth.max(MIN_INTERACTION_BOX_SIZE);
            component.height = height.max(MIN_INTERACTION_BOX_SIZE);
            self.shell.mark_dirty();
        }
        let mut color = component.color;
        if ui.color_edit3_config(format!("Color##InteractionBox{component_id}"), &mut color).flags(imgui::ColorEditFlags::NO_INPUTS).build() {
            component.color = color;
            self.shell.mark_dirty();
        }
        false
    }

    /// The Pivot section (upstream `_draw_pivot_component`).
    fn draw_pivot_component(&mut self, ui: &imgui::Ui, node_id: u32) -> bool {
        let component_id = self.document.find_node(node_id).and_then(|node| node.pivot()).map(|c| c.id).expect("drawn component");
        let (open, remove) = Self::draw_component_header(ui, "Pivot", component_id);
        if !open || remove {
            return remove;
        }
        let component = self.document.find_node(node_id).and_then(|node| node.pivot()).expect("drawn component").clone();

        // Motion Direction: keep the typed numbers in the fields, but use a
        // unit direction for previews and saving (upstream's edit cache).
        let mut direction = match &self.pivot_direction_edit {
            Some((id, direction)) if *id == component_id => *direction,
            _ => component.motion_direction.to_array(),
        };
        draw_label(ui, "Motion Direction", None, 0.0, -1.0);
        if ui.input_float3(format!("##MotionDirection{component_id}"), &mut direction).build() {
            self.pivot_direction_edit = Some((component_id, direction));
            match normalize_motion_direction(direction) {
                Ok(unit) => {
                    self.set_pivot(node_id, component_id, |pivot| pivot.motion_direction = unit);
                    self.shell.clear_status();
                }
                Err(message) => self.shell.set_status(message),
            }
        }
        if ui.is_item_deactivated_after_edit() {
            self.pivot_direction_edit = None;
        }

        draw_label(ui, "Operation Order", Some("Rotating first also turns the axis the pivot slides along."), 0.0, -1.0);
        let orders = [OperationOrder::TranslationThenRotation, OperationOrder::RotationThenTranslation];
        let mut order_index = orders.iter().position(|order| *order == component.operation_order).unwrap_or(0);
        if ui.combo_simple_string(
            format!("##OperationOrder{component_id}"),
            &mut order_index,
            &["Translation -> Rotation", "Rotation -> Translation"],
        ) {
            let order = orders[order_index];
            self.set_pivot(node_id, component_id, |pivot| pivot.operation_order = order);
        }

        // The Translation and Rotation rule rows (upstream `draw_rule_row`).
        self.draw_pivot_rule_row(ui, node_id, component_id, true);
        let component = self.document.find_node(node_id).and_then(|node| node.pivot()).expect("drawn component").clone();
        if component.translation_axis != Axis::None && component.translation_constraint == ConstraintType::LineProjection {
            ui.indent_by(16.0);
            draw_label(ui, "Offsets (m)", Some("Signed distances from the authored pivot along its selected local axis."), 0.0, -1.0);
            let mut min_offset = component.line_min_offset;
            let mut max_offset = component.line_max_offset;
            let mut changed = false;
            ui.set_next_item_width(110.0);
            changed |= imgui::Drag::new(format!("##LineMinOffset{component_id}"))
                .speed(0.01)
                .range(-100.0, max_offset)
                .display_format("Min: %.2f")
                .build(ui, &mut min_offset);
            ui.same_line();
            ui.set_next_item_width(110.0);
            changed |= imgui::Drag::new(format!("##LineMaxOffset{component_id}"))
                .speed(0.01)
                .range(min_offset, 100.0)
                .display_format("Max: %.2f")
                .build(ui, &mut max_offset);
            if changed {
                self.set_pivot(node_id, component_id, |pivot| {
                    pivot.line_min_offset = min_offset;
                    pivot.line_max_offset = max_offset.max(min_offset);
                });
            }
            ui.unindent_by(16.0);
        }
        self.draw_pivot_rule_row(ui, node_id, component_id, false);

        let component = self.document.find_node(node_id).and_then(|node| node.pivot()).expect("drawn component").clone();
        if component.rotation_axis != Axis::None && component.rotation_constraint == ConstraintType::Cone {
            ui.indent_by(16.0);
            draw_label(ui, "Half Angle (°)", Some("Maximum turn on each side of the center. 45 degrees makes a 90-degree cone."), 0.0, -1.0);
            let mut half_angle = component.cone_half_angle_degrees;
            if ui
                .slider_config(format!("##ConeHalfAngle{component_id}"), 0.0, 180.0)
                .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                .display_format("%.1f")
                .build(&mut half_angle)
            {
                self.set_pivot(node_id, component_id, |pivot| pivot.cone_half_angle_degrees = half_angle);
            }
            draw_label(ui, "Cone Side", Some("Mirrored centers the cone 180 degrees from Motion Direction around the selected local axis."), 0.0, -1.0);
            let sides = [ConeSide::Forward, ConeSide::Mirrored, ConeSide::Both];
            let mut side_index = sides.iter().position(|side| *side == component.cone_side).unwrap_or(0);
            if ui.combo_simple_string(format!("##ConeSide{component_id}"), &mut side_index, &["Motion Direction", "Mirrored", "Both"]) {
                let side = sides[side_index];
                self.set_pivot(node_id, component_id, |pivot| pivot.cone_side = side);
            }
            ui.unindent_by(16.0);
        }
        ui.separator();
        let mut show_direction = component.show_motion_direction;
        let mut show_translation = component.show_translation_constraint;
        let mut show_rotation = component.show_rotation_constraint;
        let mut changed = false;
        changed |= ui.checkbox(format!("Show Motion Direction##{component_id}"), &mut show_direction);
        changed |= ui.checkbox(format!("Show Translation Constraint##{component_id}"), &mut show_translation);
        changed |= ui.checkbox(format!("Show Rotation Constraint##{component_id}"), &mut show_rotation);
        if changed {
            self.set_pivot(node_id, component_id, |pivot| {
                pivot.show_motion_direction = show_direction;
                pivot.show_translation_constraint = show_translation;
                pivot.show_rotation_constraint = show_rotation;
            });
        }
        false
    }

    /// One pivot rule row: the axis combo and, when an axis is set, the
    /// constraint combo (upstream's nested `draw_rule_row`).
    fn draw_pivot_rule_row(&mut self, ui: &imgui::Ui, node_id: u32, component_id: u32, translation: bool) {
        let component = self.document.find_node(node_id).and_then(|node| node.pivot()).expect("drawn component").clone();
        let (axis, constraint, label, axis_labels, constraints, constraint_labels, tooltip) = if translation {
            (
                component.translation_axis,
                component.translation_constraint,
                "Translation",
                ["None", "TranslateX", "TranslateY", "TranslateZ"],
                [ConstraintType::Unconstrained, ConstraintType::LineProjection],
                ["Unconstrained", "Line Projection"],
                "Axes are local to the pivot.",
            )
        } else {
            (
                component.rotation_axis,
                component.rotation_constraint,
                "Rotation",
                ["None", "RotateX", "RotateY", "RotateZ"],
                [ConstraintType::Unconstrained, ConstraintType::Cone],
                ["Unconstrained", "Cone"],
                "Axes are local to the pivot.",
            )
        };
        draw_label(ui, label, Some(tooltip), 100.0, 115.0);
        let axes = [Axis::None, Axis::X, Axis::Y, Axis::Z];
        let mut axis_index = axes.iter().position(|candidate| *candidate == axis).unwrap_or(0);
        if ui.combo_simple_string(format!("##{}_axis{component_id}", label.to_lowercase()), &mut axis_index, &axis_labels) {
            let axis = axes[axis_index];
            self.set_pivot(node_id, component_id, |pivot| {
                if translation {
                    pivot.translation_axis = axis;
                } else {
                    pivot.rotation_axis = axis;
                }
            });
        }
        ui.same_line();
        ui.set_next_item_width(-1.0);
        let mut constraint_index = constraints.iter().position(|candidate| *candidate == constraint).unwrap_or(0);
        {
            let _disabled = ui.begin_disabled(axes[axis_index] == Axis::None);
            if ui.combo_simple_string(format!("##{}_constraint{component_id}", label.to_lowercase()), &mut constraint_index, &constraint_labels) {
                let constraint = constraints[constraint_index];
                self.set_pivot(node_id, component_id, |pivot| {
                    if translation {
                        pivot.translation_constraint = constraint;
                    } else {
                        pivot.rotation_constraint = constraint;
                    }
                });
            }
        }
    }

    /// Edit one pivot component and mark the document dirty.
    fn set_pivot(&mut self, node_id: u32, component_id: u32, edit: impl FnOnce(&mut PivotComponent)) {
        let Some(component) = find_node_mut(&mut self.document.root, node_id).and_then(|node| {
            node.components.iter_mut().find_map(|c| match c {
                NodeComponent::Pivot(c) if c.id == component_id => Some(c),
                _ => None,
            })
        }) else { return };
        edit(component);
        self.shell.mark_dirty();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            usage();
        }
    };
    let mut app = SmartPrimitiveEditor::new(&args);
    // The rig must be present before the first frame draws (upstream loads
    // SOMA in `__init__`).
    if let Err(error) = app.rig_for(app.document.rig) {
        bail!("could not load the {:?} rig: {error:#}", app.document.rig);
    }
    let mut initial_document = args.document.clone();

    let viewer = Viewer::new(ViewerConfig { title: EDITOR_TITLE.into(), width: 1440, height: 900 })?;
    viewer.run(move |frame, ui| {
        if let Some(path) = initial_document.take() {
            app.open_document(&path, frame);
        }
        app.frame(frame, ui);
        if app.shell.exit_confirmed() {
            exit(0);
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn document_with_child() -> SmartPrimitiveDocument {
        let mut document = new_smart_primitive_document(ModelRig::Soma);
        let root_id = document.root.id;
        add_node(&mut document, Some(root_id));
        document
    }

    #[test]
    fn add_node_names_like_upstream() {
        let document = document_with_child();
        let child = document.root.children.first().expect("a child was added");
        assert_eq!(child.name, format!("Node {}", child.id));
        assert!(child.visible);
        assert_eq!(child.local_transform, Transform::IDENTITY);
    }

    #[test]
    fn remove_node_drops_sequence_steps_and_pivots() {
        let mut document = document_with_child();
        let child_id = document.root.children[0].id;
        add_component(&mut document, child_id, "target_pose");
        add_component(&mut document, child_id, "pivot");
        let sequence_id = add_sequence(&mut document);
        document.sequences[0].node_ids.push(child_id);
        document.sequences[0].pivot_node_id = Some(child_id);
        let removed = remove_node(&mut document, child_id);
        assert_eq!(removed, vec![child_id]);
        assert!(document.find_node(child_id).is_none());
        assert!(document.sequences[0].node_ids.is_empty());
        assert_eq!(document.sequences[0].pivot_node_id, None);
        assert!(document.find_node(sequence_id).is_none(), "sequences are not nodes");
    }

    #[test]
    fn remove_component_cleans_sequence_references() {
        let mut document = document_with_child();
        let child_id = document.root.children[0].id;
        let target_id = add_component(&mut document, child_id, "target_pose").expect("added");
        let pivot_id = add_component(&mut document, child_id, "pivot").expect("added");
        add_sequence(&mut document);
        document.sequences[0].node_ids.push(child_id);
        document.sequences[0].pivot_node_id = Some(child_id);
        remove_component(&mut document, child_id, target_id);
        assert!(document.sequences[0].node_ids.is_empty(), "steps of a removed target pose go");
        assert_eq!(document.sequences[0].pivot_node_id, Some(child_id), "the pivot is untouched");
        remove_component(&mut document, child_id, pivot_id);
        assert_eq!(document.sequences[0].pivot_node_id, None);
    }

    #[test]
    fn reparent_keeps_world_placement() {
        let mut document = document_with_child();
        let parent_a = document.root.children[0].id;
        let root_id = document.root.id;
        let parent_b = add_node(&mut document, Some(root_id));
        let child = add_node(&mut document, Some(parent_a));
        // Place the parents so the composition is nontrivial.
        find_node_mut(&mut document.root, parent_b).unwrap().local_transform =
            Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        let world_before = document.world_transform(child).unwrap();
        assert!(reparent(&mut document, child, parent_b));
        let world_after = document.world_transform(child).unwrap();
        assert!(transforms_equal(&world_before, &world_after), "world placement stays: {world_before:?} vs {world_after:?}");
        assert_eq!(document.find_parent(child).unwrap().id, parent_b);
        // A node cannot move under itself (a cycle).
        assert!(!reparent(&mut document, parent_b, child));
        // The root never moves.
        assert!(!reparent(&mut document, root_id, parent_a));
    }

    #[test]
    fn add_sequence_picks_the_first_free_name() {
        let mut document = new_smart_primitive_document(ModelRig::Soma);
        add_sequence(&mut document);
        add_sequence(&mut document);
        assert_eq!(document.sequences[0].name, "Sequence 1");
        assert_eq!(document.sequences[1].name, "Sequence 2");
        document.sequences.remove(0);
        add_sequence(&mut document);
        assert_eq!(document.sequences[1].name, "Sequence 1");
    }

    #[test]
    fn component_defaults_match_upstream() {
        let target = default_target_pose(7);
        assert_eq!(target.num_poses_to_use, MODEL_POSES_PER_TOKEN);
        assert_eq!((target.min_tokens, target.max_tokens), (6, 14));
        assert_eq!(target.trajectory_scale, 2.0);
        assert!(target.show_mesh && !target.show_skeletons && !target.show_velocity && !target.show_trajectory);
        let pivot = default_pivot(9);
        assert_eq!(pivot.motion_direction, Vec3::Y);
        assert_eq!((pivot.line_min_offset, pivot.line_max_offset), (-0.5, 0.5));
        assert_eq!(pivot.cone_half_angle_degrees, 45.0);
        assert!(pivot.show_motion_direction && pivot.show_translation_constraint && pivot.show_rotation_constraint);
        let interaction_box = default_interaction_box(11);
        assert_eq!((interaction_box.width, interaction_box.depth, interaction_box.height), (1.0, 1.0, 1.0));
    }

    #[test]
    fn random_color_is_deterministic_and_in_range() {
        for id in [1, 2, 3, 42, 1000] {
            let color = random_color(id);
            assert_eq!(color, random_color(id));
            assert!(color.iter().all(|value| (0.0..=1.0).contains(value)), "{color:?}");
        }
    }

    #[test]
    fn up_index_finds_the_clip_height_axis() {
        // SOMA's model_to_runtime turns Y-up model space into Z-up world.
        let y_up_to_z_up = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        assert_eq!(up_index(rt::Transform::new(rt::Vec3([0.0; 3]), y_up_to_z_up.to_array())), 1);
        assert_eq!(up_index(rt::Transform::IDENTITY), 2);
    }

    #[test]
    fn relative_path_walks_up_from_the_directory() {
        assert_eq!(
            relative_path(Path::new("/a/b/motions/clip.bvh"), Path::new("/a/b/docs")),
            "../motions/clip.bvh"
        );
        assert_eq!(relative_path(Path::new("/a/b/docs/clip.bvh"), Path::new("/a/b/docs")), "clip.bvh");
    }

    #[test]
    fn add_component_adds_each_kind_once() {
        let mut document = document_with_child();
        let child_id = document.root.children[0].id;
        assert!(add_component(&mut document, child_id, "target_pose").is_some());
        assert!(add_component(&mut document, child_id, "target_pose").is_none(), "one slot per kind");
        assert!(add_component(&mut document, child_id, "interaction_box").is_some());
        assert!(add_component(&mut document, child_id, "pivot").is_some());
        assert!(add_component(&mut document, child_id, "pivot").is_none());
        assert!(add_component(&mut document, child_id, "unknown").is_none());
        let node = document.find_node(child_id).unwrap();
        assert_eq!(node.components.len(), 3, "Target Pose, Interaction Box, Pivot");
    }

    #[test]
    fn sequences_referencing_finds_steps_and_pivots() {
        let mut document = document_with_child();
        let child_id = document.root.children[0].id;
        add_component(&mut document, child_id, "target_pose");
        add_component(&mut document, child_id, "pivot");
        let first = add_sequence(&mut document);
        let second = add_sequence(&mut document);
        document.sequences[0].node_ids.push(child_id);
        document.sequences[1].pivot_node_id = Some(child_id);
        assert_eq!(sequences_referencing(&document, child_id), vec![first, second]);
        assert!(sequences_referencing(&document, document.root.id).is_empty());
    }

    #[test]
    fn move_sequence_step_reorders_and_ignores_bad_indices() {
        let mut sequence = TargetPoseSequence { id: 1, name: "s".to_string(), node_ids: vec![10, 20, 30], pivot_node_id: None };
        move_sequence_step(&mut sequence, 0, 2);
        assert_eq!(sequence.node_ids, vec![20, 30, 10]);
        move_sequence_step(&mut sequence, 2, 0);
        assert_eq!(sequence.node_ids, vec![10, 20, 30]);
        move_sequence_step(&mut sequence, 5, 0);
        move_sequence_step(&mut sequence, 0, 5);
        assert_eq!(sequence.node_ids, vec![10, 20, 30], "out-of-range is a no-op");
    }

    #[test]
    fn normalize_motion_direction_validates_like_upstream() {
        let unit = normalize_motion_direction([0.0, 3.0, 0.0]).expect("a non-zero direction");
        assert!(unit.abs_diff_eq(Vec3::Y, 1e-6));
        assert!(normalize_motion_direction([0.0, 0.0, 0.0]).is_err());
        assert!(normalize_motion_direction([f32::NAN, 0.0, 0.0]).is_err());
        assert!(normalize_motion_direction([f32::INFINITY, 0.0, 0.0]).is_err());
        let unit = normalize_motion_direction([1.0, 1.0, 0.0]).expect("a diagonal direction");
        assert!((unit.length() - 1.0).abs() < 1e-6);
    }

    // -- End-to-end preview test: the committed document, real rig, real clips --

    fn app_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn clips_dir() -> PathBuf {
        app_dir().join("../checkpoints/motions")
    }

    fn assets_dir() -> PathBuf {
        app_dir().join("../sandbox/upstream")
    }

    #[test]
    fn sit_document_previews_sample_every_target_pose() {
        let path = app_dir().join("assets/smart_primitives/soma/sit.smart_primitive.json");
        let document = SmartPrimitiveDocument::load(&path).expect("the committed document loads");
        assert_eq!(document.rig, ModelRig::Soma);
        let mut rig = RigData::load(ModelRig::Soma, &clips_dir(), &assets_dir()).expect("the SOMA rig loads");
        assert_eq!(rig.up_index, 1, "SOMA clips are Y-up");
        let targets = document.target_poses();
        assert!(!targets.is_empty(), "the sit document has target poses");
        let clip_roots = [clips_dir(), assets_dir().join("assets/motions")];
                for (node_id, component) in &targets {
            let world = document.world_transform(*node_id).expect("every node has a world transform");
            let clip_path = component.clip_path(&path).expect("the sit targets reference clips");
            let resolved = resolve_clip_under(&clip_roots, &clip_path);
            assert!(resolved.is_file(), "clip resolves under the clip roots: {}", clip_path.display());
            let clip = match &rig.registry.load(&resolved).unwrap_or_else(|e| panic!("{} loads: {e:#}", resolved.display())).clip {
                Clip::Soma(buffer) => Clip::Soma(buffer.clone()),
                Clip::G1(buffer) => Clip::G1(buffer.clone()),
            };
            let sample = sample_target(&rig, component, Some(&clip)).expect("the target samples");
            assert_eq!(sample.poses.len(), MODEL_POSES_PER_TOKEN as usize);
            let PoseSample::Soma(local) = &sample.poses[0] else { panic!("SOMA poses") };
            assert_eq!(local.len(), rig.soma.as_ref().unwrap().skeleton.num_joints());
            assert!(local.iter().all(|t| t.p.0.iter().all(|v| v.is_finite())), "finite pose");
            // The mesh bones and IK pin tips evaluate for drawing.
            let root = rt_transform(&world) * rig.model_to_runtime;
            let bones = rig.soma.as_ref().unwrap().evaluate(local, root).expect("bones evaluate");
            assert!(bones.iter().all(|m| m.w_axis.is_finite()), "finite bones");
            for (_, tip) in &rig.ik_tips {
                assert!(*tip < local.len(), "IK tip {tip} is a joint");
            }
            if component.show_trajectory {
                let trajectory =
                    sample_trajectory(&rig, component, &clip, sample.root_offset).expect("the trajectory samples");
                assert_eq!(trajectory.len(), TRAJECTORY_POINT_COUNT);
            }
        }
    }

    #[test]
    fn g1_document_previews_sample_every_target_pose() {
        let path = app_dir().join("assets/smart_primitives/unitree_g1/avoidance.smart_primitive.json");
        let document = SmartPrimitiveDocument::load(&path).expect("the committed document loads");
        assert_eq!(document.rig, ModelRig::UnitreeG1);
        let mut rig = RigData::load(ModelRig::UnitreeG1, &clips_dir(), &assets_dir()).expect("the G1 rig loads");
        assert_eq!(rig.up_index, 2, "G1 clips are Z-up");
        assert_eq!(rig.g1_link_meshes.len(), rig.g1.as_ref().unwrap().links.len());
        let clip_roots = [clips_dir(), assets_dir().join("assets/motions")];
                for (node_id, component) in document.target_poses() {
            let world = document.world_transform(node_id).expect("every node has a world transform");
            let clip_path = component.clip_path(&path).expect("the avoidance targets reference clips");
            let resolved = resolve_clip_under(&clip_roots, &clip_path);
            let clip = match &rig.registry.load(&resolved).unwrap_or_else(|e| panic!("{} loads: {e:#}", resolved.display())).clip {
                Clip::Soma(buffer) => Clip::Soma(buffer.clone()),
                Clip::G1(buffer) => Clip::G1(buffer.clone()),
            };
            let sample = sample_target(&rig, component, Some(&clip)).expect("the target samples");
            let PoseSample::G1(q) = &sample.poses[0] else { panic!("G1 poses") };
            assert!(q.iter().all(|v| v.is_finite()), "finite joint coordinates");
            let root = rt_transform(&world) * rig.model_to_runtime;
            let mut q = q.clone();
            rt::write_root(root * rt::read_root(&q), &mut q);
            let links = rig.g1.as_ref().unwrap().link_transforms(&q).expect("link transforms evaluate");
            assert!(links.iter().all(|t| t.p.0.iter().all(|v| v.is_finite())), "finite links");
        }
    }

    #[test]
    fn preview_cache_resamples_only_when_inputs_change() {
        let args = Args { document: None, clips: clips_dir(), assets: assets_dir() };
        let mut app = SmartPrimitiveEditor::new(&args);
        app.rig_data = Some(RigData::load(ModelRig::Soma, &clips_dir(), &assets_dir()).expect("the SOMA rig loads"));
        let node_id = 2;
        let mut component = default_target_pose(5);
        let first = app.preview(node_id, &component).expect("samples");
        let second = app.preview(node_id, &component).expect("samples");
        assert!(std::rc::Rc::ptr_eq(&first, &second), "unchanged inputs reuse the cached sample");
        component.sample_time_seconds = 0.5;
        let third = app.preview(node_id, &component).expect("samples");
        assert!(!std::rc::Rc::ptr_eq(&first, &third), "a changed sample time resamples");
        component.hips_constrained = true;
        let fourth = app.preview(node_id, &component).expect("samples");
        assert!(!std::rc::Rc::ptr_eq(&third, &fourth), "a changed hips constraint resamples");
    }
}
