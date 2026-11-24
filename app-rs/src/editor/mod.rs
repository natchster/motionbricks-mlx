// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The editor framework shared by the Scene Editor and the Smart Primitive
//! Editor, ported from upstream `app/authoring/common.py` (the `EditorViewer`
//! half) and `app/scene_editor.py`'s scene-model operations.
//!
//! - [`EditorShell`]: main menu bar with the File menu, unsaved-changes
//!   tracking with an exit confirmation, the dockspace layout, the status
//!   line, and the in-imgui path prompt that replaces upstream's tkinter
//!   file dialogs.
//! - Camera capture/restore between the viewer's [`OrbitCamera`] and the
//!   documents' [`CameraState`].
//! - The Z-up world to Y-up viewer conversion the apps apply at the logging
//!   boundary (upstream's world is Z-up; `viewer` is Y-up).
//! - CPU raycasts used for click-picking (meshes, oriented boxes, line
//!   capsules), ported from `motionbricks/renderers/picking.py`.
//! - Scene item operations ([`scene_set_parent`], [`scene_remove_item`], ...)
//!   with upstream's `SceneDefinition` semantics.
//! - [`widgets`]: the imgui widget helpers (tree rows, inline rename,
//!   drag-drop, transform editor with axis locks, ...).

pub mod widgets;

pub use widgets::*;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use imgui::{StyleColor, Ui};

use crate::documents::scene::{
    CharacterStart, CubeDimensions, SceneDocument, SmartPrimitiveInstance, SolidCube, Visibility,
};
use crate::documents::{CameraState, ModelRig, Transform};
use crate::viewer::{OrbitCamera, Ray};

// ---------------------------------------------------------------------------
// Z-up world -> Y-up viewer conversion
// ---------------------------------------------------------------------------

/// Rotation about X by -90 degrees: maps world +Z (up) to viewer +Y, world +Y
/// to viewer -Z. The MotionBricks world is Z-up; the viewer's camera, ground
/// grid, and gizmo are Y-up, so everything is converted at the logging
/// boundary and picking rays are converted back.
pub fn world_to_viewer_matrix() -> Mat4 {
    Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2)
}

/// Convert a world-space (Z-up) transform matrix to viewer space (Y-up).
pub fn world_to_viewer(matrix: Mat4) -> Mat4 {
    world_to_viewer_matrix() * matrix
}

/// Convert a viewer-space (Y-up) transform matrix back to world space (Z-up).
pub fn viewer_to_world(matrix: Mat4) -> Mat4 {
    world_to_viewer_matrix().inverse() * matrix
}

/// A world-space point in viewer space.
pub fn world_point_to_viewer(point: Vec3) -> Vec3 {
    Vec3::new(point.x, point.z, -point.y)
}

/// A viewer-space point in world space.
pub fn viewer_point_to_world(point: Vec3) -> Vec3 {
    Vec3::new(point.x, -point.z, point.y)
}

/// A viewer-space picking ray in world space.
pub fn viewer_ray_to_world(ray: &Ray) -> Ray {
    Ray { origin: viewer_point_to_world(ray.origin), dir: viewer_point_to_world(ray.dir) }
}

// ---------------------------------------------------------------------------
// Camera capture / restore
// ---------------------------------------------------------------------------

/// Capture the orbit camera as a document [`CameraState`] (upstream
/// `EditorViewer._capture_camera_state`).
///
/// `pos`/`pivot` are stored in the Z-up world and fully determine the view.
/// `pitch`/`yaw` are derived in world space (degrees): the elevation of the
/// view direction, and its azimuth about +Z. The viewer has a fixed field of
/// view and clip planes, so the constants are written back.
pub fn capture_camera_state(camera: &OrbitCamera) -> CameraState {
    let pos = viewer_point_to_world(camera.eye());
    let pivot = viewer_point_to_world(camera.target);
    let direction = (pivot - pos).normalize_or(Vec3::Z);
    CameraState {
        pos,
        pivot,
        pitch: direction.z.asin().to_degrees(),
        yaw: direction.y.atan2(direction.x).to_degrees(),
        fov: OrbitCamera::FOV_Y.to_degrees(),
        near: OrbitCamera::NEAR,
        far: OrbitCamera::FAR,
    }
}

/// Restore a document camera into the orbit camera (upstream
/// `EditorViewer._apply_camera_state`). `pos` and `pivot` are authoritative;
/// the viewer's fixed fov/near/far are left untouched.
pub fn apply_camera_state(camera: &mut OrbitCamera, state: &CameraState) {
    let eye = world_point_to_viewer(state.pos);
    let pivot = world_point_to_viewer(state.pivot);
    let offset = eye - pivot;
    let distance = offset.length();
    camera.target = pivot;
    camera.distance = distance.clamp(0.05, 500.0);
    if distance > 1e-6 {
        let direction = offset / distance;
        camera.pitch = direction.y.asin().clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        camera.yaw = direction.x.atan2(direction.z);
    }
}

// ---------------------------------------------------------------------------
// Picking (ports of motionbricks/renderers/picking.py)
// ---------------------------------------------------------------------------

/// Upstream `_LINE_PICK_RADIUS`: line segments pick as capsules of this radius.
pub const LINE_PICK_RADIUS: f32 = 0.025;

/// Möller–Trumbore ray-triangle test. Returns the ray parameter of the hit.
pub fn ray_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-8;
    let e1 = b - a;
    let e2 = c - a;
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let s = ray.origin - a;
    let u = s.dot(p) * inv;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > EPS).then_some(t)
}

/// Nearest hit of `ray` against a world-space triangle soup. When `indices` is
/// empty the positions are treated as a flat soup (3 per triangle).
pub fn ray_mesh(ray: &Ray, positions: &[Vec3], indices: &[u32]) -> Option<f32> {
    let mut best: Option<f32> = None;
    let mut test = |a: Vec3, b: Vec3, c: Vec3| {
        if let Some(t) = ray_triangle(ray, a, b, c) {
            if best.is_none_or(|d| t < d) {
                best = Some(t);
            }
        }
    };
    if indices.is_empty() {
        for triangle in positions.chunks_exact(3) {
            test(triangle[0], triangle[1], triangle[2]);
        }
    } else {
        for triangle in indices.chunks_exact(3) {
            let (a, b, c) = (triangle[0] as usize, triangle[1] as usize, triangle[2] as usize);
            if let (Some(&a), Some(&b), Some(&c)) = (positions.get(a), positions.get(b), positions.get(c)) {
                test(a, b, c);
            }
        }
    }
    best
}

/// Nearest hit of `ray` against an oriented box placed by `box_to_world` (a
/// rigid transform) with the given half extents. Upstream raycasts the boxes
/// the wire/solid box renderers draw.
pub fn ray_obb(ray: &Ray, box_to_world: &Mat4, half_extents: Vec3) -> Option<f32> {
    // The box transform is rigid, so the local-space ray parameter equals the
    // world-space one.
    let world_to_box = box_to_world.inverse();
    let origin = world_to_box.transform_point3(ray.origin);
    let direction = world_to_box.transform_vector3(ray.dir);
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    for axis in 0..3 {
        let (o, d, h) = (origin[axis], direction[axis], half_extents[axis]);
        if d.abs() < 1e-12 {
            if o.abs() > h {
                return None;
            }
            continue;
        }
        let (mut t0, mut t1) = ((-h - o) / d, (h - o) / d);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_near = t_near.max(t0);
        t_far = t_far.min(t1);
        if t_near > t_far {
            return None;
        }
    }
    if t_far < 0.0 {
        return None; // The box is behind the ray.
    }
    Some(t_near.max(0.0))
}

/// Nearest hit of `ray` against line segments, each treated as a capsule of
/// `radius` (upstream `raycast_lines`).
pub fn ray_segments(ray: &Ray, segments: &[[Vec3; 2]], radius: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for &[start, end] in segments {
        if let Some(t) = ray_segment(ray, start, end, radius) {
            if best.is_none_or(|d| t < d) {
                best = Some(t);
            }
        }
    }
    best
}

/// Hit of `ray` against one segment treated as a capsule: the ray parameter of
/// the closest approach when it comes within `radius`.
fn ray_segment(ray: &Ray, start: Vec3, end: Vec3, radius: f32) -> Option<f32> {
    // Closest points between the ray (origin + s*dir, s >= 0) and the segment
    // (start + t*(end-start), 0 <= t <= 1); see Ericson 5.1.9.
    let d1 = ray.dir;
    let d2 = end - start;
    let r = ray.origin - start;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let c = d1.dot(r);
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom > 1e-12 { (b * f - c * e) / denom } else { 0.0 };
    let mut t = if e > 1e-12 { ((b * s + f) / e).clamp(0.0, 1.0) } else { 0.0 };
    s = (b * t - c) / a;
    if s < 0.0 {
        s = 0.0;
        t = if e > 1e-12 { (f / e).clamp(0.0, 1.0) } else { 0.0 };
    }
    let on_ray = ray.origin + d1 * s;
    let on_segment = start + d2 * t;
    ((on_ray - on_segment).length() <= radius).then_some(s)
}

// ---------------------------------------------------------------------------
// Oriented box drawing helpers
// ---------------------------------------------------------------------------

/// The 8 corners of an oriented box placed by `box_to_world`.
pub fn obb_corners(box_to_world: &Mat4, half_extents: Vec3) -> [Vec3; 8] {
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let local = Vec3::new(
            if index & 1 == 0 { -half_extents.x } else { half_extents.x },
            if index & 2 == 0 { -half_extents.y } else { half_extents.y },
            if index & 4 == 0 { -half_extents.z } else { half_extents.z },
        );
        *corner = box_to_world.transform_point3(local);
    }
    corners
}

/// The 12 edges of an oriented box, for [`crate::viewer::Frame::log_lines`].
/// (`Frame::log_wire_box` only draws axis-aligned boxes.)
pub fn obb_wire_segments(box_to_world: &Mat4, half_extents: Vec3) -> [[Vec3; 2]; 12] {
    let c = obb_corners(box_to_world, half_extents);
    [
        [c[0], c[1]],
        [c[1], c[3]],
        [c[3], c[2]],
        [c[2], c[0]],
        [c[4], c[5]],
        [c[5], c[7]],
        [c[7], c[6]],
        [c[6], c[4]],
        [c[0], c[4]],
        [c[1], c[5]],
        [c[2], c[6]],
        [c[3], c[7]],
    ]
}

// ---------------------------------------------------------------------------
// Document names
// ---------------------------------------------------------------------------

/// The document name from a file path (upstream `DocumentType.name_from_path()`):
/// the file name without the document suffix, case-insensitively. `None` when
/// the suffix is wrong or the name would be empty.
pub fn document_name_from_path(path: &Path, suffix: &str) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    if !file_name.to_lowercase().ends_with(suffix) {
        return None;
    }
    let name = &file_name[..file_name.len() - suffix.len()];
    (!name.is_empty()).then(|| name.to_string())
}

// ---------------------------------------------------------------------------
// Scene item operations (upstream SceneDefinition)
// ---------------------------------------------------------------------------

/// Every item ID in the branch rooted at `item_id`, starting with the item
/// itself (upstream `SceneDefinition.subtree()`).
pub fn scene_subtree_ids(scene: &SceneDocument, item_id: u32) -> Vec<u32> {
    let mut out = vec![item_id];
    let mut stack = vec![item_id];
    while let Some(id) = stack.pop() {
        for child in scene.children(Some(id)) {
            out.push(child.id());
            stack.push(child.id());
        }
    }
    out
}

/// Delete an item and every item under it, and return their IDs (upstream
/// `SceneDefinition.remove_item()`). The CharacterStart cannot be removed.
pub fn scene_remove_item(scene: &mut SceneDocument, item_id: u32) -> Vec<u32> {
    if item_id == scene.character_start.id || scene.find_item(item_id).is_none() {
        return Vec::new();
    }
    let removed = scene_subtree_ids(scene, item_id);
    scene.smart_primitives.retain(|item| !removed.contains(&item.id));
    scene.solid_cubes.retain(|item| !removed.contains(&item.id));
    removed
}

/// Change an item's parent while keeping its world placement (upstream
/// `SceneDefinition.set_parent()`). `None` moves it to the top level. Returns
/// false when the parent is unchanged or invalid: the CharacterStart can
/// neither move nor parent anything, and an item cannot sit under itself.
pub fn scene_set_parent(scene: &mut SceneDocument, item_id: u32, parent_id: Option<u32>) -> bool {
    if item_id == scene.character_start.id || parent_id == Some(scene.character_start.id) {
        return false;
    }
    let Some(item) = scene.find_item(item_id) else {
        return false;
    };
    if item.parent_id() == parent_id {
        return false;
    }
    if let Some(parent_id) = parent_id {
        if scene.find_item(parent_id).is_none() || scene_subtree_ids(scene, item_id).contains(&parent_id) {
            return false;
        }
    }
    let Some(world) = scene.world_transform(item_id) else {
        return false;
    };
    let new_local = match parent_id {
        None => world,
        Some(parent_id) => match scene.world_transform(parent_id) {
            Some(parent_world) => parent_world.inverse().mul(&world),
            None => return false,
        },
    };
    assign_parent_and_transform(scene, item_id, parent_id, new_local)
}

/// Set an item's local transform. Returns false for an unknown ID.
pub fn scene_set_item_transform(scene: &mut SceneDocument, item_id: u32, transform: Transform) -> bool {
    if item_id == scene.character_start.id {
        scene.character_start.transform = transform;
        return true;
    }
    if let Some(item) = scene.smart_primitives.iter_mut().find(|item| item.id == item_id) {
        item.transform = transform;
        return true;
    }
    if let Some(item) = scene.solid_cubes.iter_mut().find(|item| item.id == item_id) {
        item.transform = transform;
        return true;
    }
    false
}

/// An item's stored `euler_degrees` (the transform editor's display state).
pub fn scene_item_euler(scene: &SceneDocument, item_id: u32) -> Option<Vec3> {
    if item_id == scene.character_start.id {
        return Some(scene.character_start.euler_degrees);
    }
    if let Some(item) = scene.smart_primitives.iter().find(|item| item.id == item_id) {
        return Some(item.euler_degrees);
    }
    scene.solid_cubes.iter().find(|item| item.id == item_id).map(|item| item.euler_degrees)
}

/// Update an item's `euler_degrees` display state.
pub fn scene_set_item_euler(scene: &mut SceneDocument, item_id: u32, euler_degrees: Vec3) {
    if item_id == scene.character_start.id {
        scene.character_start.euler_degrees = euler_degrees;
    } else if let Some(item) = scene.smart_primitives.iter_mut().find(|item| item.id == item_id) {
        item.euler_degrees = euler_degrees;
    } else if let Some(item) = scene.solid_cubes.iter_mut().find(|item| item.id == item_id) {
        item.euler_degrees = euler_degrees;
    }
}

/// Rename an item. The CharacterStart keeps its name; empty names are rejected.
pub fn scene_set_item_name(scene: &mut SceneDocument, item_id: u32, name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() || item_id == scene.character_start.id {
        return false;
    }
    if let Some(item) = scene.smart_primitives.iter_mut().find(|item| item.id == item_id) {
        item.name = name.to_string();
        return true;
    }
    if let Some(item) = scene.solid_cubes.iter_mut().find(|item| item.id == item_id) {
        item.name = name.to_string();
        return true;
    }
    false
}

/// Add a Smart Primitive item with a new ID and name (upstream
/// `SceneDefinition.add_smart_primitive()`), and return its ID.
pub fn scene_add_smart_primitive(scene: &mut SceneDocument) -> u32 {
    let id = scene.next_free_id();
    scene.smart_primitives.push(SmartPrimitiveInstance {
        id,
        name: scene_next_item_name(scene, "SmartPrimitive"),
        parent_id: None,
        transform: Transform::IDENTITY,
        euler_degrees: Vec3::ZERO,
        reference: None,
        visibility: Visibility { target_poses: true, interaction_boxes: true },
        target_pose_color: [1.0, 0.30, 0.20],
        behavior: String::new(),
    });
    id
}

/// Add a Solid Cube with a new ID and name (upstream
/// `SceneDefinition.add_solid_cube()`), and return its ID.
pub fn scene_add_solid_cube(scene: &mut SceneDocument) -> u32 {
    let id = scene.next_free_id();
    scene.solid_cubes.push(SolidCube {
        id,
        name: scene_next_item_name(scene, "SolidCube"),
        parent_id: None,
        transform: Transform::new(Vec3::new(0.0, 0.0, 0.5), glam::Quat::IDENTITY),
        euler_degrees: Vec3::ZERO,
        dimensions: CubeDimensions { width: 1.0, depth: 1.0, height: 1.0 },
        color: [0.45, 0.55, 0.70],
    });
    id
}

/// A new empty scene for `rig` with the default CharacterStart selected
/// (upstream `SceneDefinition()`). The scene has no name until it is saved;
/// the save takes the name from the file.
pub fn new_scene_document(rig: ModelRig) -> SceneDocument {
    SceneDocument {
        name: String::new(),
        rig,
        selected_item_id: CharacterStart::ID,
        camera: CameraState {
            pos: Vec3::new(-4.0, -4.0, 3.0),
            pivot: Vec3::new(0.0, 0.0, 1.0),
            pitch: 0.0,
            yaw: 0.0,
            fov: OrbitCamera::FOV_Y.to_degrees(),
            near: OrbitCamera::NEAR,
            far: OrbitCamera::FAR,
        },
        character_start: CharacterStart { id: CharacterStart::ID, transform: Transform::IDENTITY, euler_degrees: Vec3::ZERO },
        smart_primitives: Vec::new(),
        solid_cubes: Vec::new(),
    }
}

/// Whether a target pose preview draws (upstream
/// `_SceneSmartPrimitiveView._target_is_visible`): the instance's
/// "Show Target Poses" toggle, and — when a single sequence is previewed —
/// membership in that sequence's node IDs.
pub fn target_pose_visible(
    show_target_poses: bool,
    sequence_node_ids: Option<&HashSet<u32>>,
    node_id: u32,
) -> bool {
    show_target_poses && sequence_node_ids.is_none_or(|ids| ids.contains(&node_id))
}

/// Upstream `_next_item_name()`: the base name, or with a numeric suffix when
/// taken ("SmartPrimitive", "SmartPrimitive2", ...).
fn scene_next_item_name(scene: &SceneDocument, base_name: &str) -> String {
    let taken = |name: &str| {
        scene.smart_primitives.iter().any(|item| item.name == name)
            || scene.solid_cubes.iter().any(|item| item.name == name)
    };
    if !taken(base_name) {
        return base_name.to_string();
    }
    let mut suffix = 2;
    while taken(&format!("{base_name}{suffix}")) {
        suffix += 1;
    }
    format!("{base_name}{suffix}")
}

fn assign_parent_and_transform(
    scene: &mut SceneDocument,
    item_id: u32,
    parent_id: Option<u32>,
    transform: Transform,
) -> bool {
    if let Some(item) = scene.smart_primitives.iter_mut().find(|item| item.id == item_id) {
        item.parent_id = parent_id;
        item.transform = transform;
        return true;
    }
    if let Some(item) = scene.solid_cubes.iter_mut().find(|item| item.id == item_id) {
        item.parent_id = parent_id;
        item.transform = transform;
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Editor shell (upstream EditorViewer)
// ---------------------------------------------------------------------------

/// The kind of document an editor edits (upstream `DocumentType`): its name in
/// dialogs and its file suffix.
#[derive(Debug, Clone, Copy)]
pub struct DocumentType {
    pub kind: &'static str,
    pub suffix: &'static str,
}

/// What the user chose in the Exit confirmation modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitChoice {
    /// Save the document; exit only when the save succeeds.
    SaveAndExit,
}

/// An in-imgui file path prompt, replacing upstream's tkinter file dialogs.
///
/// The app opens one with [`EditorShell::open_path_prompt`] and receives the
/// confirmed prompt back from [`EditorShell::draw_path_prompt`], matching on
/// [`PathPrompt::id`] to tell prompts apart.
#[derive(Debug, Clone)]
pub struct PathPrompt {
    /// What the chosen path is for (e.g. "open", "save_as", "reference").
    pub id: &'static str,
    /// The modal's title (e.g. "Open Scene").
    pub title: String,
    /// True for a save prompt, false for a load prompt.
    pub save: bool,
    /// The path being edited.
    pub text: String,
    focus: bool,
    opened: bool,
}

/// Shared editor frame state: the document path, the unsaved-changes flag, the
/// status line, exit handling, the dockspace layout, and the path prompt.
///
/// Unlike upstream's `EditorViewer` base class this holds no viewer; the app
/// drives it from its frame handler and implements the document operations
/// itself.
pub struct EditorShell {
    editor_title: &'static str,
    document_type: DocumentType,
    document_path: Option<PathBuf>,
    status: Option<String>,
    dirty: bool,
    exit_confirmed: bool,
    exit_popup_pending: bool,
    dock_layout_pending: bool,
    path_prompt: Option<PathPrompt>,
}

impl EditorShell {
    pub fn new(editor_title: &'static str, document_type: DocumentType) -> Self {
        Self {
            editor_title,
            document_type,
            document_path: None,
            status: None,
            dirty: false,
            exit_confirmed: false,
            exit_popup_pending: false,
            dock_layout_pending: true,
            path_prompt: None,
        }
    }

    pub fn editor_title(&self) -> &'static str {
        self.editor_title
    }

    pub fn document_type(&self) -> DocumentType {
        self.document_type
    }

    pub fn document_path(&self) -> Option<&Path> {
        self.document_path.as_deref()
    }

    /// The current document's display name (from its file name), if it has one.
    pub fn document_name(&self) -> Option<String> {
        self.document_path
            .as_deref()
            .and_then(|path| document_name_from_path(path, self.document_type.suffix))
    }

    /// Make `document_path` the current file (upstream `_set_document_path`):
    /// clears the status line and the unsaved-changes flag.
    pub fn set_document_path(&mut self, document_path: Option<PathBuf>) {
        self.document_path = document_path;
        self.status = None;
        self.dirty = false;
    }

    /// Mark the document as edited since it was last saved.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = Some(status.into());
    }

    pub fn clear_status(&mut self) {
        self.status = None;
    }

    /// Whether the user confirmed exiting (through the Exit menu or the
    /// confirmation modal). The app should then leave its frame loop.
    pub fn exit_confirmed(&self) -> bool {
        self.exit_confirmed
    }

    /// Mark the exit as confirmed (after a successful save, or "Don't Save").
    pub fn confirm_exit(&mut self) {
        self.exit_confirmed = true;
    }

    /// The Exit menu command: with unsaved changes ask for confirmation,
    /// otherwise exit directly. (Upstream asks unconditionally.)
    pub fn request_exit(&mut self) {
        if self.dirty {
            self.exit_popup_pending = true;
        } else {
            self.exit_confirmed = true;
        }
    }

    /// Open the path prompt modal. `initial` pre-fills the path field.
    pub fn open_path_prompt(&mut self, id: &'static str, title: &str, save: bool, initial: String) {
        self.path_prompt = Some(PathPrompt { id, title: title.to_string(), save, text: initial, focus: true, opened: false });
    }

    pub fn path_prompt(&self) -> Option<&PathPrompt> {
        self.path_prompt.as_ref()
    }

    /// Draw the main menu bar and return the chosen File action, if any
    /// (upstream `_draw_main_menu_bar` + `_draw_file_menu`; the Tools menu is
    /// omitted — this viewer has no options/stats panels).
    pub fn draw_main_menu_bar(&mut self, ui: &Ui) -> Option<FileAction> {
        let mut action = None;
        {
            // Keep the menu bar's hue, only force upstream's alpha.
            let _color = ui.push_style_color(StyleColor::MenuBarBg, [0.14, 0.14, 0.16, MENU_BAR_ALPHA]);
            ui.main_menu_bar(|| {
                ui.menu("File", || {
                    if ui.menu_item("New") {
                        action = Some(FileAction::New);
                    }
                    ui.separator();
                    if ui.menu_item("Open...") {
                        action = Some(FileAction::Open);
                    }
                    if ui.menu_item("Save") {
                        action = Some(FileAction::Save);
                    }
                    if ui.menu_item("Save As...") {
                        action = Some(FileAction::SaveAs);
                    }
                    ui.separator();
                    if ui.menu_item("Exit") {
                        action = Some(FileAction::Exit);
                    }
                });
            });
        }
        action
    }

    /// Draw the Exit confirmation modal (upstream `_draw_exit_confirmation`).
    ///
    /// Returns [`ExitChoice::SaveAndExit`] when the user picks Save: the app
    /// then runs its Save and calls [`EditorShell::confirm_exit`] only when it
    /// succeeded. "Don't Save" confirms the exit immediately.
    pub fn draw_exit_confirmation(&mut self, ui: &Ui) -> Option<ExitChoice> {
        if self.exit_popup_pending {
            ui.open_popup("Exit");
            self.exit_popup_pending = false;
        }
        let mut choice = None;
        ui.modal_popup_config("Exit").always_auto_resize(true).build(|| {
            ui.text("Save the current document before exiting?");
            if ui.button("Save") {
                choice = Some(ExitChoice::SaveAndExit);
            }
            ui.same_line();
            if ui.button("Don't Save") {
                self.confirm_exit();
                ui.close_current_popup();
            }
            ui.same_line();
            if ui.button("Cancel") {
                ui.close_current_popup();
            }
            self.draw_status(ui);
        });
        if choice.is_some() {
            ui.close_current_popup();
        }
        choice
    }

    /// Draw the open path prompt modal, returning it once confirmed (OK button
    /// or Enter). The caller then acts on [`PathPrompt::id`] and
    /// [`PathPrompt::text`].
    pub fn draw_path_prompt(&mut self, ui: &Ui) -> Option<PathPrompt> {
        let (title, first_open) = match self.path_prompt.as_mut() {
            Some(prompt) => {
                let first_open = !prompt.opened;
                prompt.opened = true;
                (prompt.title.clone(), first_open)
            }
            None => return None,
        };
        if first_open {
            ui.open_popup(&title);
        }
        let mut confirm = false;
        let mut cancel = false;
        ui.modal_popup_config(&title).always_auto_resize(true).build(|| {
            if let Some(prompt) = self.path_prompt.as_mut() {
                if prompt.focus {
                    ui.set_keyboard_focus_here();
                    prompt.focus = false;
                }
                ui.set_next_item_width(420.0);
                let submitted =
                    ui.input_text("##path", &mut prompt.text).enter_returns_true(true).build();
                confirm = ui.button(if prompt.save { "Save" } else { "Open" }) || submitted;
                ui.same_line();
                cancel = ui.button("Cancel");
            }
        });
        if cancel {
            self.path_prompt = None;
            ui.close_current_popup();
            return None;
        }
        if confirm {
            let non_empty = self.path_prompt.as_ref().is_some_and(|prompt| !prompt.text.trim().is_empty());
            if non_empty {
                return self.path_prompt.take().map(|mut prompt| {
                    prompt.text = prompt.text.trim().to_string();
                    prompt
                });
            }
        }
        None
    }

    /// Draw the dockspace over the whole viewport and, on first use, dock the
    /// outliner and Properties panels into the right column (upstream
    /// `_draw_dockspace` + `_build_initial_dock_layout`).
    ///
    /// imgui-rs 0.12 does not wrap the dock builder API, so this calls the
    /// `imgui-sys` functions directly.
    pub fn draw_dockspace(&mut self, ui: &Ui, outliner_title: &str) {
        let _ = ui;
        use imgui::sys;
        let flags = (sys::ImGuiDockNodeFlags_PassthruCentralNode | sys::ImGuiDockNodeFlags_NoDockingInCentralNode)
            as sys::ImGuiDockNodeFlags;
        // SAFETY: called inside an imgui frame; all pointers are either to
        // live imgui objects or to out-parameters that outlive the call.
        let dockspace_id = unsafe {
            sys::igDockSpaceOverViewport(sys::igGetMainViewport(), flags, std::ptr::null())
        };
        if !self.dock_layout_pending {
            return;
        }
        unsafe {
            let root = sys::igDockBuilderGetNode(dockspace_id);
            if root.is_null() {
                return; // Node not created yet; retry next frame.
            }
            let node = &*root;
            let is_split = !node.ChildNodes[0].is_null();
            let has_windows = node.Windows.Size > 0;
            if is_split || has_windows {
                self.dock_layout_pending = false;
                return;
            }
            let mut right_dock = 0;
            let mut center_dock = 0;
            sys::igDockBuilderSplitNode(
                dockspace_id,
                sys::ImGuiDir_Right,
                RIGHT_DOCK_RATIO,
                &mut right_dock,
                &mut center_dock,
            );
            let mut properties_dock = 0;
            let mut outliner_dock = 0;
            sys::igDockBuilderSplitNode(
                right_dock,
                sys::ImGuiDir_Down,
                PROPERTIES_DOCK_RATIO,
                &mut properties_dock,
                &mut outliner_dock,
            );
            let outliner = std::ffi::CString::new(outliner_title).unwrap_or_default();
            sys::igDockBuilderDockWindow(outliner.as_ptr(), outliner_dock);
            sys::igDockBuilderDockWindow(c"Properties".as_ptr(), properties_dock);
            sys::igDockBuilderFinish(dockspace_id);
        }
        self.dock_layout_pending = false;
    }

    /// Draw the status line in red, when set (upstream `_draw_status`).
    pub fn draw_status(&self, ui: &Ui) {
        if let Some(status) = &self.status {
            let _color = ui.push_style_color(StyleColor::Text, [1.0, 0.35, 0.35, 1.0]);
            ui.text_wrapped(status);
        }
    }
}

/// Position and size for a floating panel before ImGui has a saved layout for
/// it (upstream `_set_initial_panel_geometry`): pinned to the right edge of
/// the viewport, below `top_offset`. Use with
/// `ui.window(title).position(pos, Condition::FirstUseEver).position_pivot([1.0, 0.0]).size(size, Condition::FirstUseEver).bg_alpha(PANEL_ALPHA)`.
pub fn initial_panel_geometry(ui: &Ui, width: f32, height: f32, top_offset: f32, dpi_scale: f32) -> ([f32; 2], [f32; 2]) {
    let margin = WINDOW_MARGIN * dpi_scale;
    let display = ui.io().display_size;
    // ~19 px for the main menu bar; docking refines this on the first frame anyway.
    let menu_bar_height = 19.0 * dpi_scale;
    let pos = [display[0] - margin, menu_bar_height + top_offset * dpi_scale];
    let size = [width * dpi_scale, height * dpi_scale];
    (pos, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_2;

    // -- Z-up <-> Y-up conversion ------------------------------------------

    #[test]
    fn world_viewer_point_conversion_round_trips() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = world_point_to_viewer(p);
        // World +Z (up) becomes viewer +Y; world +Y becomes viewer -Z.
        assert!(v.abs_diff_eq(Vec3::new(1.0, 3.0, -2.0), 1e-6));
        assert!(viewer_point_to_world(v).abs_diff_eq(p, 1e-6));
    }

    #[test]
    fn world_viewer_transform_conversion_round_trips() {
        let m = Mat4::from_rotation_translation(Quat::from_rotation_z(0.7), Vec3::new(1.0, 2.0, 3.0));
        let round_tripped = viewer_to_world(world_to_viewer(m));
        assert!(round_tripped.abs_diff_eq(m, 1e-5));
        // The conversion is rigid (unit basis vectors).
        let converted = world_to_viewer(m);
        assert!((converted.x_axis.length() - 1.0).abs() < 1e-6);
        assert!((converted.y_axis.length() - 1.0).abs() < 1e-6);
        assert!((converted.z_axis.length() - 1.0).abs() < 1e-6);
    }

    // -- Camera capture / restore -------------------------------------------

    #[test]
    fn camera_state_round_trips() {
        let camera = OrbitCamera { target: Vec3::new(1.0, 0.5, -2.0), yaw: 0.9, pitch: -0.3, distance: 12.5 };
        let state = capture_camera_state(&camera);
        state.validate("test.camera").expect("captured camera validates");
        let mut restored = OrbitCamera::default();
        apply_camera_state(&mut restored, &state);
        assert!(restored.target.abs_diff_eq(camera.target, 1e-5), "target: {:?}", restored.target);
        assert!((restored.distance - camera.distance).abs() < 1e-4, "distance: {}", restored.distance);
        assert!(restored.eye().abs_diff_eq(camera.eye(), 1e-4), "eye: {:?}", restored.eye());
    }

    #[test]
    fn camera_state_from_an_upstream_scene_gives_a_sane_view() {
        // soma_gym.scene.json's stored camera (Z-up world).
        let state = CameraState {
            pos: Vec3::new(-28.627, 4.970, 23.381),
            pivot: Vec3::new(1.604, -0.790, -2.428),
            pitch: -39.98,
            yaw: -10.79,
            fov: 45.0,
            near: 0.01,
            far: 1000.0,
        };
        let mut camera = OrbitCamera::default();
        apply_camera_state(&mut camera, &state);
        assert!((camera.distance - 40.1).abs() < 0.5, "distance: {}", camera.distance);
        assert!(camera.pitch.is_finite() && camera.yaw.is_finite());
        // Round-trips through capture.
        let captured = capture_camera_state(&camera);
        assert!(captured.pos.abs_diff_eq(state.pos, 1e-3));
        assert!(captured.pivot.abs_diff_eq(state.pivot, 1e-3));
    }

    // -- Picking -------------------------------------------------------------

    #[test]
    fn ray_mesh_hits_and_misses() {
        let positions = vec![Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let hit = Ray { origin: Vec3::new(0.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        let t = ray_mesh(&hit, &positions, &[]).expect("ray through the triangle");
        assert!((t - 5.0).abs() < 1e-5);
        let miss = Ray { origin: Vec3::new(10.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert!(ray_mesh(&miss, &positions, &[]).is_none());
        // Indexed access gives the same hit.
        assert_eq!(ray_mesh(&hit, &positions, &[0, 1, 2]), Some(t));
    }

    #[test]
    fn ray_obb_hits_rotated_box() {
        let transform = Mat4::from_rotation_translation(Quat::from_rotation_z(FRAC_PI_2), Vec3::new(5.0, 0.0, 0.0));
        let half = Vec3::new(0.5, 2.0, 0.5);
        // Rotated 90 degrees about Z, the long local-y side (half 2.0) lies along
        // world x: the box spans x in [3, 7], so the ray hits at x = 3.
        let ray = Ray { origin: Vec3::ZERO, dir: Vec3::X };
        let t = ray_obb(&ray, &transform, half).expect("ray hits the box");
        assert!((t - 3.0).abs() < 1e-4, "t = {t}");
        // A ray above the box misses (after rotation the y extent is only ±0.5).
        let miss = Ray { origin: Vec3::new(0.0, 1.0, 0.0), dir: Vec3::X };
        assert!(ray_obb(&miss, &transform, half).is_none());
        // A ray pointing away misses.
        let away = Ray { origin: Vec3::ZERO, dir: -Vec3::X };
        assert!(ray_obb(&away, &transform, half).is_none());
    }

    #[test]
    fn ray_segments_pick_as_capsules() {
        let segments = [[Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)]];
        // Straight at the segment's middle.
        let ray = Ray { origin: Vec3::new(1.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert_eq!(ray_segments(&ray, &segments, LINE_PICK_RADIUS), Some(5.0));
        // Within the pick radius (1 cm above the segment).
        let near = Ray { origin: Vec3::new(1.0, 0.01, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert!(ray_segments(&near, &segments, LINE_PICK_RADIUS).is_some());
        // Outside the pick radius.
        let far = Ray { origin: Vec3::new(1.0, 0.1, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert!(ray_segments(&far, &segments, LINE_PICK_RADIUS).is_none());
        // Past the segment's end.
        let beyond = Ray { origin: Vec3::new(5.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert!(ray_segments(&beyond, &segments, LINE_PICK_RADIUS).is_none());
    }

    #[test]
    fn obb_wire_segments_draw_the_box_edges() {
        let segments = obb_wire_segments(&Mat4::IDENTITY, Vec3::ONE);
        assert_eq!(segments.len(), 12);
        for [a, b] in &segments {
            // Each edge connects two corners sharing two coordinates.
            let shared = (0..3).filter(|&axis| a[axis] == b[axis]).count();
            assert_eq!(shared, 2, "edge {a:?} -> {b:?}");
        }
    }

    // -- Document names ------------------------------------------------------

    #[test]
    fn document_name_from_path_rules() {
        assert_eq!(document_name_from_path(Path::new("/tmp/gym.scene.json"), ".scene.json"), Some("gym".to_string()));
        // Case-insensitive suffix, like upstream.
        assert_eq!(document_name_from_path(Path::new("GYM.SCENE.JSON"), ".scene.json"), Some("GYM".to_string()));
        assert_eq!(document_name_from_path(Path::new("gym.json"), ".scene.json"), None);
        // An empty name is not a document.
        assert_eq!(document_name_from_path(Path::new(".scene.json"), ".scene.json"), None);
    }

    // -- Scene item operations -------------------------------------------------

    fn test_scene() -> SceneDocument {
        let mut scene = new_scene_document(ModelRig::Soma);
        let sp = scene_add_smart_primitive(&mut scene);
        let cube = scene_add_solid_cube(&mut scene);
        // Place both at known world transforms.
        let sp_transform = Transform::new(Vec3::new(4.0, 0.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
        scene_set_item_transform(&mut scene, sp, sp_transform);
        scene_set_item_transform(&mut scene, cube, Transform::new(Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY));
        scene
    }

    #[test]
    fn add_items_uses_upstream_naming() {
        let mut scene = new_scene_document(ModelRig::Soma);
        let a = scene_add_smart_primitive(&mut scene);
        let b = scene_add_smart_primitive(&mut scene);
        let c = scene_add_solid_cube(&mut scene);
        assert_eq!(scene.find_item(a).unwrap().id(), 2);
        assert_eq!(scene.find_item(b).unwrap().id(), 3);
        assert_eq!(scene.find_item(c).unwrap().id(), 4);
        let names: Vec<_> = scene.smart_primitives.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["SmartPrimitive", "SmartPrimitive2"]);
        assert_eq!(scene.solid_cubes[0].name, "SolidCube");
        // Default visibility: both previews on (upstream defaults).
        assert!(scene.smart_primitives[0].visibility.target_poses);
        assert!(scene.smart_primitives[0].visibility.interaction_boxes);
    }

    #[test]
    fn set_parent_keeps_the_world_transform() {
        let mut scene = test_scene();
        let cube = scene.solid_cubes[0].id;
        let sp = scene.smart_primitives[0].id;
        let world_before = scene.world_transform(cube).unwrap();
        assert!(scene_set_parent(&mut scene, cube, Some(sp)));
        assert_eq!(scene.solid_cubes[0].parent_id, Some(sp));
        let world_after = scene.world_transform(cube).unwrap();
        assert!(transforms_equal(&world_before, &world_after), "world moved: {world_before:?} -> {world_after:?}");
        // The local transform is now relative to the Smart Primitive.
        let sp_world = scene.world_transform(sp).unwrap();
        let expected_local = sp_world.inverse().mul(&world_before);
        assert!(transforms_equal(&scene.solid_cubes[0].transform, &expected_local));
    }

    #[test]
    fn set_parent_rejects_cycles_and_the_character_start() {
        let mut scene = test_scene();
        let cube = scene.solid_cubes[0].id;
        let sp = scene.smart_primitives[0].id;
        assert!(scene_set_parent(&mut scene, cube, Some(sp)));
        // The Smart Primitive cannot move under its own child.
        assert!(!scene_set_parent(&mut scene, sp, Some(cube)));
        // The CharacterStart can neither move nor parent anything.
        assert!(!scene_set_parent(&mut scene, CharacterStart::ID, Some(sp)));
        assert!(!scene_set_parent(&mut scene, sp, Some(CharacterStart::ID)));
        // Unchanged parent and unknown IDs are rejected.
        assert!(!scene_set_parent(&mut scene, cube, Some(sp)));
        assert!(!scene_set_parent(&mut scene, cube, Some(999)));
        // Move to the top level keeps the world placement too.
        let world_before = scene.world_transform(cube).unwrap();
        assert!(scene_set_parent(&mut scene, cube, None));
        assert_eq!(scene.solid_cubes[0].parent_id, None);
        assert!(transforms_equal(&scene.world_transform(cube).unwrap(), &world_before));
    }

    #[test]
    fn remove_item_deletes_the_whole_branch() {
        let mut scene = test_scene();
        let cube = scene.solid_cubes[0].id;
        let sp = scene.smart_primitives[0].id;
        assert!(scene_set_parent(&mut scene, cube, Some(sp)));
        let removed = scene_remove_item(&mut scene, sp);
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&sp) && removed.contains(&cube));
        assert!(scene.smart_primitives.is_empty() && scene.solid_cubes.is_empty());
        // The CharacterStart cannot be removed.
        assert!(scene_remove_item(&mut scene, CharacterStart::ID).is_empty());
    }

    #[test]
    fn item_transform_and_rename_helpers() {
        let mut scene = test_scene();
        let sp = scene.smart_primitives[0].id;
        assert!(scene_set_item_name(&mut scene, sp, "Jump"));
        assert_eq!(scene.smart_primitives[0].name, "Jump");
        // The CharacterStart keeps its name; empty names are rejected.
        assert!(!scene_set_item_name(&mut scene, CharacterStart::ID, "Start"));
        assert!(!scene_set_item_name(&mut scene, sp, "   "));
        let euler = scene_item_euler(&scene, sp).unwrap();
        assert!(euler.abs_diff_eq(Vec3::ZERO, 1e-6));
        scene_set_item_euler(&mut scene, sp, Vec3::new(0.0, 0.0, 90.0));
        assert!(scene_item_euler(&scene, sp).unwrap().abs_diff_eq(Vec3::new(0.0, 0.0, 90.0), 1e-6));
    }

    #[test]
    fn new_scene_is_valid_and_untitled() {
        let scene = new_scene_document(ModelRig::UnitreeG1);
        scene.validate().expect("a new scene validates");
        assert_eq!(scene.rig, ModelRig::UnitreeG1);
        assert_eq!(scene.selected_item_id, CharacterStart::ID);
        assert!(scene.name.is_empty(), "a new scene has no name until saved");
    }

    #[test]
    fn target_pose_visibility_rules() {
        let sequence: HashSet<u32> = [2, 3].into_iter().collect();
        // Off wins over every sequence filter.
        assert!(!target_pose_visible(false, None, 2));
        assert!(!target_pose_visible(false, Some(&sequence), 2));
        // "All" shows every target; a sequence shows only its own nodes.
        assert!(target_pose_visible(true, None, 9));
        assert!(target_pose_visible(true, Some(&sequence), 3));
        assert!(!target_pose_visible(true, Some(&sequence), 9));
    }

    // -- Shell state -----------------------------------------------------------

    #[test]
    fn shell_tracks_dirty_state_and_exit() {
        let mut shell = EditorShell::new("MotionBricks - Scene Editor", DocumentType { kind: "Scene", suffix: ".scene.json" });
        assert!(!shell.is_dirty());
        // Clean: exit is immediate.
        shell.request_exit();
        assert!(shell.exit_confirmed());

        let mut shell = EditorShell::new("MotionBricks - Scene Editor", DocumentType { kind: "Scene", suffix: ".scene.json" });
        shell.mark_dirty();
        shell.request_exit();
        assert!(!shell.exit_confirmed(), "dirty: a confirmation is required first");
        // Saving (setting the document path) clears the dirty flag.
        shell.set_document_path(Some(PathBuf::from("/tmp/gym.scene.json")));
        assert!(!shell.is_dirty());
        assert_eq!(shell.document_name(), Some("gym".to_string()));
    }
}
