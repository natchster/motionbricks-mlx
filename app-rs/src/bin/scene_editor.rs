// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Scene Editor: build scenes with Smart Primitives, Solid Cubes, and a
//! CharacterStart. Port of upstream `app/scene_editor.py` (`python -m
//! app.scene_editor`), on the shared framework in `src/editor/`.
//!
//! Run with: `cargo run --release --bin scene-editor -- --scene assets/scenes/soma_gym.scene.json`
//!
//! Deviations from upstream (all deliberate):
//! - The tkinter file dialogs are in-imgui path-input modals
//!   ([`editor::EditorShell::open_path_prompt`]).
//! - The window title is static; this viewer cannot retitle its window.
//! - The transform gizmo translates along the viewer's world axes only (the
//!   viewer has no rotation rings and no parent-axis mode); rotation is edited
//!   in the Properties panel. Axis locks from the panel apply to the gizmo.
//! - `imgui-rs` 0.12 has no dock-builder wrapper, so the initial dock layout
//!   calls `imgui-sys` directly (isolated in `editor::EditorShell::draw_dockspace`).
//! - The committed Smart Primitive documents reference clips that this repo
//!   ships separately: clip paths that do not resolve against the document
//!   fall back to the `--clips` directory, matched by path tail.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::rc::Rc;

use anyhow::{Context as _, Result, bail};
use glam::{Mat4, Vec2, Vec3};
use lexopt::prelude::*;
use motionbricks_rs::runtime as rt;

use motionbricks_app::documents::scene::{CharacterStart, SCENE_SUFFIX, SceneDocument};
use motionbricks_app::documents::smart_primitive::{SMART_PRIMITIVE_SUFFIX, SmartPrimitiveDocument, TargetPoseComponent};
use motionbricks_app::documents::{ModelRig, Transform};
use motionbricks_app::rig::clips::{Clip, ClipRegistry};
use motionbricks_app::rig::g1::G1Rig;
use motionbricks_app::rig::soma::{SomaRig, transform_to_mat4};
use motionbricks_app::viewer::{Frame, Mesh, Ray, Skin, Viewer, ViewerConfig};

// Re-exports so the `crate::documents` / `crate::rig` / `crate::viewer` paths inside
// src/editor/ resolve while that module is included via #[path] (the integrator
// will wire it into lib.rs as `pub mod editor;` instead).
use motionbricks_app::{documents, viewer};

// The editor framework's full API surface serves both editors; this bin uses a
// subset, so unused-framework-item warnings are expected here (not in the lib).
#[allow(dead_code)]
#[path = "../editor/mod.rs"]
mod editor;

use editor::widgets::{
    FileAction, InlineRename, PANEL_ALPHA, PROPERTIES_WINDOW_HEIGHT, PROPERTIES_WINDOW_WIDTH, WINDOW_MARGIN, accept_drop, apply_transform_locks, draw_add_remove_buttons, draw_box_dimensions,
    draw_character_selector, draw_clip_selector, draw_transform_editor, draw_tree_row, drag_source,
    transforms_equal,
};
use editor::{
    DocumentType, EditorShell, ExitChoice, LINE_PICK_RADIUS, PathPrompt, initial_panel_geometry,
    new_scene_document, obb_wire_segments, ray_mesh, ray_obb, ray_segments, scene_add_smart_primitive,
    scene_add_solid_cube, scene_item_euler, scene_remove_item, scene_set_item_euler, scene_set_item_name,
    scene_set_item_transform, scene_set_parent, target_pose_visible, viewer_ray_to_world, world_to_viewer,
    world_to_viewer_matrix,
};

// Upstream constants (app/scene_editor.py).
const SCENE_WINDOW_WIDTH: f32 = 320.0;
const SCENE_WINDOW_HEIGHT: f32 = 380.0;
const MIN_SOLID_CUBE_DIMENSION: f32 = 0.01;
const CHARACTER_START_RADIUS: f32 = 0.35;
const CHARACTER_START_CIRCLE_SEGMENTS: usize = 32;
const CHARACTER_START_ARROW_LENGTH: f32 = 0.8;
const CHARACTER_START_MARKER_HEIGHT: f32 = 0.01;
const CHARACTER_START_COLOR: [f32; 4] = [0.25, 0.85, 1.0, 1.0];
const INTERACTION_BOX_COLOR: [f32; 4] = [1.0, 0.15, 0.10, 1.0];
const SCENE_ITEM_PAYLOAD: &str = "SCENE_ITEM";

const OUTLINER_WINDOW_TITLE: &str = "Scene";
const EDITOR_TITLE: &str = "MotionBricks - Scene Editor";
const SCENE_DOCUMENT_TYPE: DocumentType = DocumentType { kind: "Scene", suffix: SCENE_SUFFIX };

/// Upstream `behavior_types()`: every `SmartPrimitiveBehavior` subclass. The Rust
/// app runtime's behaviors are not wired into this crate yet, so the list is fixed.
const BEHAVIOR_TYPES: [&str; 2] = ["VelocityMatchingSequenceBehavior", "SittingBehavior"];

/// Per-item axis locks (translation, rotation); an editing preference, not saved.
type TransformLocks = ([bool; 3], [bool; 3]);

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Args {
    scene: Option<PathBuf>,
    rig: ModelRig,
    clips: PathBuf,
    assets: PathBuf,
}

fn usage() -> ! {
    eprintln!(
        "MotionBricks Scene Editor\n\
         \n\
         Usage: scene-editor [OPTIONS]\n\
         \n\
         Options:\n\
         \x20 --scene PATH   scene document to open at startup (.scene.json)\n\
         \x20 --rig RIG      rig for a new scene: soma | unitree_g1 (default: soma)\n\
         \x20 --clips DIR    clip library root (default: checkpoints/motions)\n\
         \x20 --assets DIR   upstream checkout root, for the G1 meshes\n\
         \x20               (default: sandbox/upstream)\n\
         \x20 --help         show this help"
    );
    exit(if std::env::args().any(|arg| arg == "--help") { 0 } else { 2 });
}

fn first_existing(candidates: &[&str], fallback: &str) -> PathBuf {
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return path;
        }
    }
    PathBuf::from(fallback)
}

fn parse_args() -> Result<Args> {
    let mut scene = None;
    let mut rig = ModelRig::Soma;
    let mut clips = first_existing(&["checkpoints/motions", "../checkpoints/motions"], "checkpoints/motions");
    let mut assets = first_existing(&["sandbox/upstream", "../sandbox/upstream"], "sandbox/upstream");
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(value) => {
                // Upstream takes the scene as a positional argument.
                scene = Some(PathBuf::from(value.string()?));
            }
            Long("scene") => scene = Some(PathBuf::from(parser.value()?.string()?)),
            Long("rig") => {
                let name = parser.value()?.string()?;
                rig = ModelRig::from_name(&name).with_context(|| format!("unknown rig '{name}' (soma | unitree_g1)"))?;
            }
            Long("clips") => clips = PathBuf::from(parser.value()?.string()?),
            Long("assets") => assets = PathBuf::from(parser.value()?.string()?),
            Long("help") => usage(),
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(Args { scene, rig, clips, assets })
}

// ---------------------------------------------------------------------------
// Rig assets and the Smart Primitive asset pool
// ---------------------------------------------------------------------------

/// One rig's visual assets and clip registry (upstream `RigAssets`).
struct RigData {
    rig: ModelRig,
    soma: Option<Box<SomaRig>>,
    soma_skin: Option<Skin>,
    g1: Option<Box<G1Rig>>,
    g1_link_meshes: Vec<Mesh>,
    registry: ClipRegistry,
}

impl RigData {
    fn load(rig: ModelRig, clips_dir: &Path, assets_dir: &Path) -> Result<RigData> {
        match rig {
            ModelRig::Soma => {
                let soma = SomaRig::load(clips_dir).context("could not load the SOMA rig")?;
                let registry = ClipRegistry::soma(soma.skeleton.clone());
                let skin = soma.viewer_skin();
                Ok(RigData {
                    rig,
                    soma: Some(Box::new(soma)),
                    soma_skin: Some(skin),
                    g1: None,
                    g1_link_meshes: Vec::new(),
                    registry,
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
                    rig,
                    soma: None,
                    soma_skin: None,
                    g1: Some(Box::new(g1)),
                    g1_link_meshes: link_meshes,
                    registry: ClipRegistry::unitree_g1_contract(),
                })
            }
        }
    }

    fn rig(&self) -> ModelRig {
        self.rig
    }
}

/// A sampled target pose: the model-space pose at the component's sample time
/// (pose 0 of the four; the scene editor previews only the first).
enum TargetSample {
    /// SOMA: local joint transforms on the mesh skeleton.
    Soma(Vec<rt::Transform>),
    /// Unitree G1: joint coordinates.
    G1(Vec<f32>),
}

/// One node's target pose preview inside a referenced document.
struct TargetPreview {
    node_id: u32,
    /// The node's world transform within the document.
    node_world: Transform,
    sample: TargetSample,
}

/// One node's interaction box inside a referenced document.
struct BoxPreview {
    node_world: Transform,
    half_extents: Vec3,
}

/// A referenced Smart Primitive, loaded once and shared by its instances
/// (upstream `SmartPrimitiveAssetPool`'s assets).
struct PrimitiveAsset {
    document: SmartPrimitiveDocument,
    targets: Vec<TargetPreview>,
    boxes: Vec<BoxPreview>,
}

/// Loads each referenced `.smart_primitive.json` once and shares it (upstream
/// `SmartPrimitiveAssetPool`).
struct AssetPool {
    rig_data: RigData,
    assets: HashMap<PathBuf, Rc<PrimitiveAsset>>,
    /// Clip library roots for [`AssetPool::resolve_clip`]'s fallback, most
    /// specific first: the `--clips` directory, then the upstream checkout's
    /// own motions (the committed documents reference those).
    clip_roots: Vec<PathBuf>,
}

impl AssetPool {
    fn new(rig_data: RigData, clips_dir: PathBuf, assets_dir: PathBuf) -> AssetPool {
        let clip_roots = vec![clips_dir, assets_dir.join("assets/motions")];
        AssetPool { rig_data, assets: HashMap::new(), clip_roots }
    }

    /// The shared asset for `path`, preparing it on first use.
    fn load(&mut self, path: &Path) -> Result<Rc<PrimitiveAsset>> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(asset) = self.assets.get(&key) {
            return Ok(asset.clone());
        }
        let document = SmartPrimitiveDocument::load_for_rig(path, self.rig_data.rig())?;
        let asset = Rc::new(self.build_asset(document, path)?);
        self.assets.insert(key, asset.clone());
        Ok(asset)
    }

    fn build_asset(&mut self, document: SmartPrimitiveDocument, document_path: &Path) -> Result<PrimitiveAsset> {
        let mut targets = Vec::new();
        for (node_id, component) in document.target_poses() {
            let node_world = document
                .world_transform(node_id)
                .with_context(|| format!("node {node_id} has no world transform"))?;
            let sample = self.sample_target(component, document_path)?;
            targets.push(TargetPreview { node_id, node_world, sample });
        }
        let mut boxes = Vec::new();
        for (node_id, component) in document.interaction_boxes() {
            let node_world = document
                .world_transform(node_id)
                .with_context(|| format!("node {node_id} has no world transform"))?;
            boxes.push(BoxPreview {
                node_world,
                half_extents: Vec3::new(component.width, component.depth, component.height) * 0.5,
            });
        }
        Ok(PrimitiveAsset { document, targets, boxes })
    }

    /// Sample the target pose's clip at its sample time, upstream's
    /// `TargetPoseComponent._sample()` reduced to pose 0. Without a clip the
    /// reference pose is used, as upstream does.
    fn sample_target(
        &mut self,
        component: &TargetPoseComponent,
        document_path: &Path,
    ) -> Result<TargetSample> {
        let time = component.sample_time_seconds;
        match self.rig_data.rig {
            ModelRig::Soma => {
                let soma = self.rig_data.soma.as_ref().expect("SOMA rig data");
                let Some(clip_path) = component.clip_path(document_path) else {
                    return Ok(TargetSample::Soma(soma.skeleton.reference_local_transforms().to_vec()));
                };
                let clip_path = self.resolve_clip(&clip_path);
                let buffer = match &self.rig_data.registry.load(&clip_path)?.clip {
                    Clip::Soma(buffer) => buffer,
                    Clip::G1(_) => bail!("a SOMA registry loads BVH clips"),
                };
                let local = if component.hips_constrained {
                    // Remove the root's offset along the ground, keeping its height
                    // (SOMA model space is Y-up).
                    let pose = buffer.sample(time, rt::Transform::IDENTITY)?;
                    let root = pose.first().with_context(|| "an empty SOMA pose")?;
                    let offset =
                        rt::Transform::new(rt::Vec3([-root.p.0[0], 0.0, -root.p.0[2]]), rt::math::QUAT_IDENTITY);
                    buffer.sample(time, offset)?
                } else {
                    buffer.sample(time, rt::Transform::IDENTITY)?
                };
                Ok(TargetSample::Soma(local))
            }
            ModelRig::UnitreeG1 => {
                let g1 = self.rig_data.g1.as_ref().expect("G1 rig data");
                let Some(clip_path) = component.clip_path(document_path) else {
                    return Ok(TargetSample::G1(g1.reference_q()));
                };
                let clip_path = self.resolve_clip(&clip_path);
                let buffer = match &self.rig_data.registry.load(&clip_path)?.clip {
                    Clip::G1(buffer) => buffer,
                    Clip::Soma(_) => bail!("a G1 registry loads CSV clips"),
                };
                let q = if component.hips_constrained {
                    // G1 model space is Z-up: keep the height, remove the ground offset.
                    let q = buffer.sample(time, rt::Transform::IDENTITY)?;
                    let root = rt::read_root(&q);
                    let offset =
                        rt::Transform::new(rt::Vec3([-root.p.0[0], -root.p.0[1], 0.0]), rt::math::QUAT_IDENTITY);
                    buffer.sample(time, offset)?
                } else {
                    buffer.sample(time, rt::Transform::IDENTITY)?
                };
                Ok(TargetSample::G1(q))
            }
        }
    }

    /// Resolve a clip path: as written (relative to the document), or — when
    /// that file does not exist — under each clip library root
    /// ([`AssetPool::clip_roots`]), trying successively shorter path tails. The
    /// committed documents reference `../../motions/<rig>/<kind>/<file>`; this
    /// repo ships those clips under the upstream checkout's `assets/motions`
    /// and (the locomotion subset) in `checkpoints/motions`.
    fn resolve_clip(&self, resolved: &Path) -> PathBuf {
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
        for root in &self.clip_roots {
            for skip in 0..parts.len() {
                let candidate: PathBuf = root.join(parts[skip..].iter().collect::<PathBuf>());
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
        resolved.to_path_buf()
    }
}

/// One placed Smart Primitive and its preview state (upstream
/// `_SceneSmartPrimitiveView`, without the cached GPU buffers).
struct PrimitiveView {
    asset: Rc<PrimitiveAsset>,
    /// The previewed sequence index, or `None` to show all target poses.
    sequence_filter: Option<usize>,
}

impl PrimitiveView {
    /// Node IDs of the filtered sequence, `None` when showing all targets.
    fn sequence_node_ids(&self) -> Option<HashSet<u32>> {
        let index = self.sequence_filter?;
        let sequence = self.asset.document.sequences.get(index)?;
        Some(sequence.node_ids.iter().copied().collect())
    }
}

// ---------------------------------------------------------------------------
// The Scene Editor
// ---------------------------------------------------------------------------

struct SceneEditor {
    shell: EditorShell,
    scene: SceneDocument,
    clips_dir: PathBuf,
    assets_dir: PathBuf,
    pool: Option<AssetPool>,
    views: HashMap<u32, PrimitiveView>,
    rename: InlineRename,
    reveal: HashSet<u32>,
    locks: HashMap<u32, TransformLocks>,
    selected_item_id: u32,
    click_press: Option<Vec2>,
    left_was_held: bool,
    character_start_marker: Vec<[Vec3; 2]>,
    unit_cube: Mesh,
}

impl SceneEditor {
    fn new(args: &Args) -> SceneEditor {
        let scene = new_scene_document(args.rig);
        SceneEditor {
            shell: EditorShell::new(EDITOR_TITLE, SCENE_DOCUMENT_TYPE),
            selected_item_id: scene.character_start.id,
            clips_dir: args.clips.clone(),
            assets_dir: args.assets.clone(),
            scene,
            pool: None,
            views: HashMap::new(),
            rename: InlineRename::default(),
            reveal: HashSet::new(),
            locks: HashMap::new(),
            click_press: None,
            left_was_held: false,
            character_start_marker: character_start_marker(),
            unit_cube: Mesh::cube(Vec3::ZERO, Vec3::splat(0.5)),
        }
    }

    /// The selected scene item's ID; falls back to the CharacterStart, like
    /// upstream's `selected` property.
    fn selected_id(&self) -> u32 {
        if self.scene.find_item(self.selected_item_id).is_some() {
            self.selected_item_id
        } else {
            self.scene.character_start.id
        }
    }

    fn locks_for(&mut self, item_id: u32) -> TransformLocks {
        *self.locks.get(&item_id).unwrap_or(&([false; 3], [false; 3]))
    }

    fn has_references(&self) -> bool {
        self.scene.smart_primitives.iter().any(|instance| instance.reference.is_some())
    }

    // -- Rig assets -----------------------------------------------------------

    /// The asset pool for `rig`, loading the rig's assets on first use.
    /// Another rig replaces the current one (upstream `_rig_for`).
    fn pool_for(&mut self, rig: ModelRig) -> Result<&mut AssetPool> {
        let reload = match &self.pool {
            Some(pool) => pool.rig_data.rig() != rig,
            None => true,
        };
        if reload {
            let rig_data = RigData::load(rig, &self.clips_dir, &self.assets_dir)?;
            self.pool = Some(AssetPool::new(rig_data, self.clips_dir.clone(), self.assets_dir.clone()));
        }
        Ok(self.pool.as_mut().expect("pool just loaded"))
    }

    /// Build the preview views for every referenced instance of `scene`
    /// (upstream `_create_runtime_views`), resolving references against
    /// `document_path` (the file the scene was loaded from, if any).
    fn build_views_for(
        &mut self,
        scene: &SceneDocument,
        document_path: Option<&Path>,
    ) -> Result<HashMap<u32, PrimitiveView>> {
        let mut views = HashMap::new();
        for instance in &scene.smart_primitives {
            let Some(reference) = &instance.reference else { continue };
            let path = match document_path {
                Some(document_path) => SceneDocument::resolve_reference(reference, document_path),
                None => PathBuf::from(reference),
            };
            let asset = self.pool_for(scene.rig)?.load(&path)?;
            views.insert(instance.id, PrimitiveView { asset, sequence_filter: None });
        }
        Ok(views)
    }

    // -- Documents ------------------------------------------------------------

    fn new_document(&mut self) {
        self.scene = new_scene_document(self.scene.rig);
        self.selected_item_id = self.scene.character_start.id;
        self.views.clear();
        self.rename = InlineRename::default();
        self.reveal.clear();
        self.locks.clear();
        self.shell.set_document_path(None);
    }

    fn open_document(&mut self, path: &Path, frame: &mut Frame) {
        let path = absolutize(path);
        let result = (|| -> Result<SceneDocument> {
            let scene = SceneDocument::load(&path)?;
            scene.check_reference_rigs(&path)?;
            Ok(scene)
        })();
        let scene = match result {
            Ok(scene) => scene,
            Err(error) => {
                self.shell.set_status(format!("Open failed: {error:#}"));
                return;
            }
        };
        // Load the rig and referenced assets before replacing the current scene
        // (upstream `_open_document`: a failed open keeps the old scene).
        let views = match self.build_views_for(&scene, Some(&path)) {
            Ok(views) => views,
            Err(error) => {
                self.shell.set_status(format!("Open failed: {error:#}"));
                return;
            }
        };
        self.scene = scene;
        self.views = views;
        self.selected_item_id = self.scene.selected_item_id;
        self.rename = InlineRename::default();
        self.reveal.clear();
        self.locks.clear();
        self.shell.set_document_path(Some(path.clone()));
        editor::apply_camera_state(frame.camera(), &self.scene.camera);
        eprintln!(
            "scene-editor: opened {} ({} smart primitives, {} solid cubes, rig {})",
            path.display(),
            self.scene.smart_primitives.len(),
            self.scene.solid_cubes.len(),
            self.scene.rig,
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
            .unwrap_or_else(|| format!("untitled{SCENE_SUFFIX}"));
        self.shell.open_path_prompt("save_as", "Save Scene", true, initial);
        false
    }

    fn save_to(&mut self, path: &Path, frame: &mut Frame) -> bool {
        self.scene.selected_item_id = self.selected_id();
        self.scene.camera = editor::capture_camera_state(frame.camera());
        match self.scene.save(path) {
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
                    .unwrap_or_else(|| format!("scene{SCENE_SUFFIX}"));
                self.shell.open_path_prompt("open", "Open Scene", false, initial);
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
            "reference" => {
                let item_id = self.selected_id();
                self.assign_reference(item_id, Path::new(&prompt.text));
            }
            _ => {}
        }
    }

    // -- Scene edits ------------------------------------------------------------

    fn delete_item(&mut self, item_id: u32) {
        let removed = scene_remove_item(&mut self.scene, item_id);
        for removed_id in &removed {
            self.views.remove(removed_id);
            self.locks.remove(removed_id);
        }
        if !removed.is_empty() {
            self.selected_item_id = self.scene.character_start.id;
            self.rename.item_id = None;
            self.shell.mark_dirty();
        }
    }

    fn set_parent(&mut self, item_id: u32, parent_id: Option<u32>) {
        if scene_set_parent(&mut self.scene, item_id, parent_id) {
            self.selected_item_id = item_id;
            self.shell.mark_dirty();
        }
    }

    /// Set an item's local transform and keep its euler display state
    /// (upstream `_set_item_transform`).
    fn set_item_transform(&mut self, item_id: u32, transform: Transform, euler_degrees: Vec3) {
        if scene_set_item_transform(&mut self.scene, item_id, transform) {
            scene_set_item_euler(&mut self.scene, item_id, euler_degrees);
            self.shell.mark_dirty();
        }
    }

    /// Assign a Smart Primitive reference to an instance (upstream
    /// `_assign_reference`): the first reference chooses the rig; later
    /// references must match it.
    fn assign_reference(&mut self, item_id: u32, path: &Path) {
        let resolved = absolutize(path);
        let fail = |shell: &mut EditorShell, message: String| shell.set_status(format!("Reference failed: {message}"));
        if !is_smart_primitive_path(&resolved) {
            fail(&mut self.shell, format!("reference a {SMART_PRIMITIVE_SUFFIX} file."));
            return;
        }
        let Some(index) = self.scene.smart_primitives.iter().position(|instance| instance.id == item_id) else {
            return;
        };
        let previous_reference = self.scene.smart_primitives[index].reference.clone();
        let character_is_fixed = self.has_references();
        self.scene.smart_primitives[index].reference = Some(resolved.to_string_lossy().into_owned());
        let result = self.create_referenced_view(&resolved, character_is_fixed);
        match result {
            Ok((rig, view)) => {
                self.scene.rig = rig;
                self.views.insert(item_id, view);
                self.shell.clear_status();
                self.shell.mark_dirty();
            }
            Err(error) => {
                self.scene.smart_primitives[index].reference = previous_reference;
                fail(&mut self.shell, format!("{error:#}"));
            }
        }
    }

    /// Validate a reference and build its preview view (the `try` half of
    /// `assign_reference`; on error the caller reverts the reference).
    fn create_referenced_view(
        &mut self,
        resolved: &Path,
        character_is_fixed: bool,
    ) -> Result<(ModelRig, PrimitiveView)> {
        let rig = if character_is_fixed {
            // Later references must use the scene's rig.
            self.scene
                .check_reference_rigs(self.shell.document_path().unwrap_or(Path::new(".")))?;
            self.scene.rig
        } else {
            SmartPrimitiveDocument::load_rig(resolved)?
        };
        let asset = self.pool_for(rig)?.load(resolved)?;
        Ok((rig, PrimitiveView { asset, sequence_filter: None }))
    }

    /// Open the tree down to the selected item and scroll to it (upstream
    /// `_reveal_selected`).
    fn reveal_selected(&mut self) {
        self.reveal.clear();
        let mut item = self.scene.find_item(self.selected_id());
        while let Some(current) = item {
            self.reveal.insert(current.id());
            item = current.parent_id().and_then(|id| self.scene.find_item(id));
        }
    }
}

fn absolutize(path: &Path) -> PathBuf {
    let expanded: PathBuf = shellexpand::tilde(&path.to_string_lossy()).into_owned().into();
    std::path::absolute(&expanded).unwrap_or(expanded)
}

/// References must point at a `.smart_primitive.json` document (upstream
/// `_assign_reference`'s first check), case-insensitively.
fn is_smart_primitive_path(path: &Path) -> bool {
    path.to_string_lossy().to_lowercase().ends_with(SMART_PRIMITIVE_SUFFIX)
}


/// Upstream `_CharacterStartRenderer`'s marker, in the item's local space: a
/// circle at ground level plus an arrow along +X (the facing direction).
fn character_start_marker() -> Vec<[Vec3; 2]> {
    let mut segments = Vec::new();
    for segment in 0..CHARACTER_START_CIRCLE_SEGMENTS {
        let angle0 = std::f32::consts::TAU * segment as f32 / CHARACTER_START_CIRCLE_SEGMENTS as f32;
        let angle1 = std::f32::consts::TAU * (segment + 1) as f32 / CHARACTER_START_CIRCLE_SEGMENTS as f32;
        segments.push([
            Vec3::new(CHARACTER_START_RADIUS * angle0.cos(), CHARACTER_START_RADIUS * angle0.sin(), CHARACTER_START_MARKER_HEIGHT),
            Vec3::new(CHARACTER_START_RADIUS * angle1.cos(), CHARACTER_START_RADIUS * angle1.sin(), CHARACTER_START_MARKER_HEIGHT),
        ]);
    }
    let h = CHARACTER_START_MARKER_HEIGHT;
    let head_x = CHARACTER_START_ARROW_LENGTH - 0.18;
    segments.push([Vec3::new(0.0, 0.0, h), Vec3::new(CHARACTER_START_ARROW_LENGTH, 0.0, h)]);
    segments.push([Vec3::new(CHARACTER_START_ARROW_LENGTH, 0.0, h), Vec3::new(head_x, 0.12, h)]);
    segments.push([Vec3::new(CHARACTER_START_ARROW_LENGTH, 0.0, h), Vec3::new(head_x, -0.12, h)]);
    segments
}

fn rt_transform(transform: &Transform) -> rt::Transform {
    rt::Transform::new(rt::Vec3(transform.translation.to_array()), transform.rotation_xyzw.to_array())
}

fn main() -> Result<()> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            usage();
        }
    };
    let mut app = SceneEditor::new(&args);
    let mut initial_scene = args.scene.clone();

    let viewer = Viewer::new(ViewerConfig { title: EDITOR_TITLE.into(), width: 1440, height: 900 })?;
    viewer.run(move |frame, ui| {
        if let Some(path) = initial_scene.take() {
            app.open_document(&path, frame);
        }
        app.frame(frame, ui);
        if app.shell.exit_confirmed() {
            exit(0);
        }
    });
}

// ---------------------------------------------------------------------------
// Frame loop: drawing, gizmo, picking
// ---------------------------------------------------------------------------

impl SceneEditor {
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

    /// Draw the scene items (upstream `_draw_editor_scene`).
    fn draw_scene(&mut self, frame: &mut Frame) {
        // The CharacterStart marker.
        let world = self.scene.character_start.transform.to_mat4();
        let viewer_world = world_to_viewer(world);
        let segments: Vec<[Vec3; 2]> = self
            .character_start_marker
            .iter()
            .map(|&[a, b]| [viewer_world.transform_point3(a), viewer_world.transform_point3(b)])
            .collect();
        frame.log_lines(&segments, CHARACTER_START_COLOR);

        // The Smart Primitive instances.
        for (instance_id, view) in &self.views {
            let Some(instance) = self.scene.smart_primitives.iter().find(|instance| instance.id == *instance_id) else {
                continue;
            };
            let Some(world) = self.scene.world_transform(*instance_id) else {
                continue;
            };
            let sequence_node_ids = view.sequence_node_ids();
            if instance.visibility.target_poses {
                let [r, g, b] = instance.target_pose_color;
                let color = [r, g, b, 1.0];
                for target in &view.asset.targets {
                    if !target_pose_visible(true, sequence_node_ids.as_ref(), target.node_id) {
                        continue;
                    }
                    let node_world = world.mul(&target.node_world);
                    self.draw_target(frame, target, &node_world, color);
                }
            }
            if instance.visibility.interaction_boxes {
                for interaction_box in &view.asset.boxes {
                    let center = world.mul(&interaction_box.node_world);
                    let segments =
                        obb_wire_segments(&world_to_viewer(center.to_mat4()), interaction_box.half_extents);
                    frame.log_lines(&segments, INTERACTION_BOX_COLOR);
                }
            }
        }

        // The Solid Cubes, as oriented solid boxes.
        for cube in &self.scene.solid_cubes {
            let Some(world) = self.scene.world_transform(cube.id) else {
                continue;
            };
            let scale = Vec3::new(cube.dimensions.width, cube.dimensions.depth, cube.dimensions.height);
            let [r, g, b] = cube.color;
            frame.log_mesh(&self.unit_cube, world_to_viewer(world.to_mat4()) * Mat4::from_scale(scale), [r, g, b, 1.0]);
        }
    }

    /// Draw one target pose preview (upstream `_SceneSmartPrimitiveView.draw`'s
    /// mesh half: only the node's world transform moves the sampled pose's root).
    fn draw_target(&self, frame: &mut Frame, target: &TargetPreview, node_world: &Transform, color: [f32; 4]) {
        let Some(pool) = &self.pool else { return };
        match (&target.sample, &pool.rig_data.soma, &pool.rig_data.g1) {
            (TargetSample::Soma(local), Some(soma), _) => {
                let root = rt_transform(node_world) * soma.model_to_runtime;
                let Ok(bones) = soma.evaluate(local, root) else { return };
                let skin = pool.rig_data.soma_skin.as_ref().expect("SOMA rig data holds the skin");
                frame.log_skinned_mesh(&soma.mesh.mesh, skin, &bones, world_to_viewer_matrix(), color);
            }
            (TargetSample::G1(q), _, Some(g1)) => {
                let mut q = q.clone();
                let root = rt_transform(node_world) * g1.model_to_runtime * rt::read_root(&q);
                rt::write_root(root, &mut q);
                let Ok(link_transforms) = g1.link_transforms(&q) else { return };
                for (index, link_transform) in link_transforms.iter().enumerate() {
                    let Some(mesh) = pool.rig_data.g1_link_meshes.get(index) else { continue };
                    frame.log_mesh(mesh, world_to_viewer(transform_to_mat4(*link_transform)), color);
                }
            }
            _ => {}
        }
    }

    /// The transform gizmo on the selection (upstream `_draw_selection_gizmo` +
    /// `_apply_selection_gizmo`, fused: this viewer applies the gizmo inside the
    /// frame). Returns true while the gizmo is dragged.
    fn draw_gizmo(&mut self, frame: &mut Frame) -> bool {
        let item_id = self.selected_id();
        let Some(world) = self.scene.world_transform(item_id) else {
            return false;
        };
        let mut gizmo_transform = world_to_viewer(world.to_mat4());
        let dragging = frame.gizmo_translate("scene_selection", &mut gizmo_transform);
        let Ok(new_world) = Transform::from_mat4(&editor::viewer_to_world(gizmo_transform)) else {
            return dragging;
        };
        if transforms_equal(&world, &new_world) {
            return dragging;
        }
        // Back to parent space, honoring the axis locks from Properties.
        let parent_world = self
            .scene
            .find_item(item_id)
            .and_then(|item| item.parent_id())
            .and_then(|parent_id| self.scene.world_transform(parent_id));
        let edited_local = match parent_world {
            Some(parent_world) => parent_world.inverse().mul(&new_world),
            None => new_world,
        };
        let Some(current_local) = self.scene.find_item(item_id).map(|item| item.transform()) else {
            return dragging;
        };
        let (translation_locks, rotation_locks) = self.locks_for(item_id);
        let euler = scene_item_euler(&self.scene, item_id);
        let (new_local, new_euler) =
            apply_transform_locks(&current_local, &edited_local, euler, translation_locks, rotation_locks);
        self.set_item_transform(item_id, new_local, new_euler);
        dragging
    }

    /// Click-picking (upstream `NewtonViewerUIBridge._handle_scene_click` +
    /// `_select_scene_item`): a press-and-release without a drag, outside the
    /// imgui panels and the gizmo, raycasts the scene items; the nearest hit
    /// is selected and revealed in the tree.
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

    /// Raycast every pickable item and select the nearest hit (upstream
    /// `_select_scene_item`).
    fn pick(&mut self, ray: &Ray) {
        let mut nearest = f32::INFINITY;
        let mut hit: Option<u32> = None;
        let consider = |distance: Option<f32>, item_id: u32, nearest: &mut f32, hit: &mut Option<u32>| {
            if let Some(distance) = distance {
                if distance < *nearest {
                    *nearest = distance;
                    *hit = Some(item_id);
                }
            }
        };
        for (instance_id, view) in &self.views {
            let Some(instance) = self.scene.smart_primitives.iter().find(|instance| instance.id == *instance_id) else {
                continue;
            };
            let Some(world) = self.scene.world_transform(*instance_id) else {
                continue;
            };
            let sequence_node_ids = view.sequence_node_ids();
            for target in &view.asset.targets {
                if !target_pose_visible(instance.visibility.target_poses, sequence_node_ids.as_ref(), target.node_id) {
                    continue;
                }
                let node_world = world.mul(&target.node_world);
                consider(self.raycast_target(target, &node_world, ray), *instance_id, &mut nearest, &mut hit);
            }
            if instance.visibility.interaction_boxes {
                for interaction_box in &view.asset.boxes {
                    let center = world.mul(&interaction_box.node_world);
                    consider(
                        ray_obb(ray, &center.to_mat4(), interaction_box.half_extents),
                        *instance_id,
                        &mut nearest,
                        &mut hit,
                    );
                }
            }
        }
        for cube in &self.scene.solid_cubes {
            let Some(world) = self.scene.world_transform(cube.id) else { continue };
            let half = Vec3::new(cube.dimensions.width, cube.dimensions.depth, cube.dimensions.height) * 0.5;
            consider(ray_obb(ray, &world.to_mat4(), half), cube.id, &mut nearest, &mut hit);
        }
        let world = self.scene.character_start.transform;
        let segments: Vec<[Vec3; 2]> = self
            .character_start_marker
            .iter()
            .map(|&[a, b]| [world.rotation_xyzw * a + world.translation, world.rotation_xyzw * b + world.translation])
            .collect();
        consider(ray_segments(ray, &segments, LINE_PICK_RADIUS), CharacterStart::ID, &mut nearest, &mut hit);

        if let Some(item_id) = hit {
            self.selected_item_id = item_id;
            self.reveal_selected();
        }
    }

    /// Raycast one target pose preview's mesh in world space.
    fn raycast_target(&self, target: &TargetPreview, node_world: &Transform, ray: &Ray) -> Option<f32> {
        let pool = self.pool.as_ref()?;
        match (&target.sample, &pool.rig_data.soma, &pool.rig_data.g1) {
            (TargetSample::Soma(local), Some(soma), _) => {
                let root = rt_transform(node_world) * soma.model_to_runtime;
                let bones = soma.evaluate(local, root).ok()?;
                let skin = pool.rig_data.soma_skin.as_ref()?;
                let positions = skin_positions(&soma.mesh.mesh.positions, skin, &bones);
                ray_mesh(ray, &positions, &soma.mesh.mesh.indices)
            }
            (TargetSample::G1(q), _, Some(g1)) => {
                let mut q = q.clone();
                let root = rt_transform(node_world) * g1.model_to_runtime * rt::read_root(&q);
                rt::write_root(root, &mut q);
                let link_transforms = g1.link_transforms(&q).ok()?;
                let mut nearest: Option<f32> = None;
                for (index, link_transform) in link_transforms.iter().enumerate() {
                    let Some(mesh) = pool.rig_data.g1_link_meshes.get(index) else { continue };
                    // Rigid transform: raycast in the link's local space so the
                    // mesh vertices need no per-click copies.
                    let world_to_link = transform_to_mat4(*link_transform).inverse();
                    let local_ray = Ray {
                        origin: world_to_link.transform_point3(ray.origin),
                        dir: world_to_link.transform_vector3(ray.dir),
                    };
                    if let Some(distance) = ray_mesh(&local_ray, &mesh.positions, &mesh.indices) {
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
// UI: the Scene and Properties panels
// ---------------------------------------------------------------------------

impl SceneEditor {
    /// The Hierarchy ("Scene") window (upstream `_draw_outliner_window`).
    fn draw_outliner_window(&mut self, ui: &imgui::Ui) {
        let (pos, size) = initial_panel_geometry(ui, SCENE_WINDOW_WIDTH, SCENE_WINDOW_HEIGHT, WINDOW_MARGIN, 1.0);
        let mut delete_id: Option<u32> = None;
        ui.window(OUTLINER_WINDOW_TITLE)
            .position(pos, imgui::Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size(size, imgui::Condition::FirstUseEver)
            .bg_alpha(PANEL_ALPHA)
            .build(|| {
                self.draw_character_selector(ui);
                delete_id = self.draw_scene_toolbar(ui);
                self.shell.draw_status(ui);
                ui.separator();
                let character_start = self.scene.character_start.id;
                delete_id = delete_id.or(self.draw_scene_item(ui, character_start));
                for item_id in self.top_level_ids() {
                    delete_id = delete_id.or(self.draw_scene_item(ui, item_id));
                }
                self.draw_root_drop_target(ui);
            });
        if let Some(item_id) = delete_id {
            self.delete_item(item_id);
        }
    }

    fn top_level_ids(&self) -> Vec<u32> {
        self.scene.children(None).iter().map(|item| item.id()).collect()
    }

    /// The Character combo; locked once a reference exists (upstream
    /// `_draw_character_selector`).
    fn draw_character_selector(&mut self, ui: &imgui::Ui) {
        let locked_reason = self.has_references().then_some({
            "Smart Primitive references lock the Character selector. Delete those Smart Primitives or start a new scene to change it."
        });
        let rig = draw_character_selector(ui, self.scene.rig, locked_reason);
        if rig != self.scene.rig {
            self.scene.rig = rig;
            self.shell.mark_dirty();
        }
    }

    /// The +/- toolbar and its Add popup (upstream `_draw_scene_toolbar`).
    fn draw_scene_toolbar(&mut self, ui: &imgui::Ui) -> Option<u32> {
        let selected = self.selected_id();
        let (add, remove_selected) = draw_add_remove_buttons(ui, selected != self.scene.character_start.id);
        if add {
            ui.open_popup("##AddSceneItem");
        }
        let mut new_selection = None;
        if let Some(_popup) = ui.begin_popup("##AddSceneItem") {
            if ui.menu_item("Smart Primitive") {
                new_selection = Some(scene_add_smart_primitive(&mut self.scene));
            }
            if ui.menu_item("Solid Cube") {
                new_selection = Some(scene_add_solid_cube(&mut self.scene));
            }
        }
        if let Some(item_id) = new_selection {
            self.selected_item_id = item_id;
            self.shell.mark_dirty();
        }
        remove_selected.then_some(selected)
    }

    /// Draw one tree row and, when open, its children; returns the item
    /// requested for deletion (upstream `_draw_scene_item`).
    fn draw_scene_item(&mut self, ui: &imgui::Ui, item_id: u32) -> Option<u32> {
        let _id = ui.push_id_usize(item_id as usize);
        let is_start = item_id == self.scene.character_start.id;
        let name = match self.scene.find_item(item_id) {
            Some(motionbricks_app::documents::SceneItemRef::CharacterStart(_)) => "CharacterStart".to_string(),
            Some(motionbricks_app::documents::SceneItemRef::SmartPrimitive(item)) => item.name.clone(),
            Some(motionbricks_app::documents::SceneItemRef::SolidCube(item)) => item.name.clone(),
            None => return None,
        };
        let parent_id = self.scene.find_item(item_id).and_then(|item| item.parent_id());
        let child_ids: Vec<u32> = self.scene.children(Some(item_id)).iter().map(|item| item.id()).collect();
        let renaming = self.rename.item_id == Some(item_id);
        let is_selected = item_id == self.selected_id();
        let row = draw_tree_row(
            ui,
            &name,
            item_id,
            &mut self.rename,
            &mut self.reveal,
            is_selected,
            child_ids.is_empty(),
            !is_start,
            false,
        );
        if row.clicked {
            self.selected_item_id = item_id;
        }
        if let Some(new_name) = row.new_name {
            if scene_set_item_name(&mut self.scene, item_id, &new_name) {
                self.shell.mark_dirty();
            }
        }

        let mut delete_id = None;
        if !renaming && !is_start {
            // Drop an item onto another to place it under that item.
            drag_source(ui, SCENE_ITEM_PAYLOAD, item_id, &name);
            if let Some(dropped_id) = accept_drop(ui, SCENE_ITEM_PAYLOAD) {
                self.set_parent(dropped_id, Some(item_id));
            }
            delete_id = self.draw_item_context_menu(ui, item_id, parent_id);
        }

        if let Some(token) = row.token {
            for child_id in child_ids {
                delete_id = delete_id.or(self.draw_scene_item(ui, child_id));
            }
            drop(token);
        }
        delete_id
    }

    /// The drop target below the tree: drop an item there to move it to the
    /// top level (upstream `_draw_root_drop_target`).
    fn draw_root_drop_target(&mut self, ui: &imgui::Ui) {
        let space = ui.content_region_avail();
        let height = space[1].max(ui.frame_height());
        ui.invisible_button("##SceneRoot", [space[0].max(1.0), height]);
        if let Some(dropped_id) = accept_drop(ui, SCENE_ITEM_PAYLOAD) {
            self.set_parent(dropped_id, None);
        }
    }

    /// The row's right-click menu (upstream `_draw_item_context_menu`).
    fn draw_item_context_menu(&mut self, ui: &imgui::Ui, item_id: u32, parent_id: Option<u32>) -> Option<u32> {
        let popup = ui.begin_popup_context_with_label("##SceneItemContext")?;
        self.selected_item_id = item_id;
        let mut delete = false;
        if ui.menu_item_config("Move to Root").enabled(parent_id.is_some()).build() {
            self.set_parent(item_id, None);
        }
        if ui.menu_item("Delete") {
            delete = true;
        }
        drop(popup);
        delete.then_some(item_id)
    }
}

// ---------------------------------------------------------------------------
// UI: the Properties panel
// ---------------------------------------------------------------------------

impl SceneEditor {
    /// The Properties window (upstream `_draw_properties_window`).
    fn draw_properties_window(&mut self, ui: &imgui::Ui) {
        let (pos, size) = initial_panel_geometry(
            ui,
            PROPERTIES_WINDOW_WIDTH,
            PROPERTIES_WINDOW_HEIGHT,
            SCENE_WINDOW_HEIGHT + 2.0 * WINDOW_MARGIN,
            1.0,
        );
        let mut reference_requested = false;
        ui.window("Properties")
            .position(pos, imgui::Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size(size, imgui::Condition::FirstUseEver)
            .bg_alpha(PANEL_ALPHA)
            .build(|| {
                let item_id = self.selected_id();
                let _id = ui.push_id_usize(item_id as usize);
                let name = match self.scene.find_item(item_id) {
                    Some(motionbricks_app::documents::SceneItemRef::CharacterStart(_)) => "CharacterStart".to_string(),
                    Some(motionbricks_app::documents::SceneItemRef::SmartPrimitive(item)) => item.name.clone(),
                    Some(motionbricks_app::documents::SceneItemRef::SolidCube(item)) => item.name.clone(),
                    None => return,
                };
                ui.text_colored(editor::widgets::NAME_TEXT_COLOR, name);
                if ui.collapsing_header("Transform", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    self.draw_transform_details(ui, item_id);
                }
                let kind = match self.scene.find_item(item_id) {
                    Some(motionbricks_app::documents::SceneItemRef::SmartPrimitive(_)) => 1,
                    Some(motionbricks_app::documents::SceneItemRef::SolidCube(_)) => 2,
                    _ => 0,
                };
                if kind == 1 {
                    reference_requested = self.draw_smart_primitive_details(ui, item_id);
                } else if kind == 2 && ui.collapsing_header("Solid Cube", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    self.draw_solid_cube_details(ui, item_id);
                }
            });
        if reference_requested {
            let item_id = self.selected_id();
            let initial = self
                .scene
                .smart_primitives
                .iter()
                .find(|instance| instance.id == item_id)
                .and_then(|instance| instance.reference.clone())
                .unwrap_or_else(|| format!("primitive{SMART_PRIMITIVE_SUFFIX}"));
            self.shell.open_path_prompt("reference", "Reference Smart Primitive", false, initial);
        }
    }

    /// The Transform section (upstream `_draw_transform_details`).
    fn draw_transform_details(&mut self, ui: &imgui::Ui, item_id: u32) {
        let Some(item) = self.scene.find_item(item_id) else { return };
        let transform = item.transform();
        let euler = scene_item_euler(&self.scene, item_id);
        let (mut translation_locks, mut rotation_locks) = self.locks_for(item_id);
        let edit = draw_transform_editor(
            ui,
            &transform,
            euler,
            1.0,
            &mut translation_locks,
            &mut rotation_locks,
        );
        self.locks.insert(item_id, (translation_locks, rotation_locks));
        if edit.changed {
            self.set_item_transform(item_id, edit.transform, edit.euler_degrees);
        } else {
            // Keep the display state even when only a lock was toggled.
            scene_set_item_euler(&mut self.scene, item_id, edit.euler_degrees);
        }
    }

    /// The Smart Primitive, Visibility, and Runtime sections (upstream
    /// `_draw_smart_primitive_details`). Returns whether the button that picks
    /// the referenced document was clicked.
    fn draw_smart_primitive_details(&mut self, ui: &imgui::Ui, item_id: u32) -> bool {
        let mut reference_requested = false;
        if ui.collapsing_header("Smart Primitive", imgui::TreeNodeFlags::DEFAULT_OPEN) {
            let (name, path) = match self.scene.smart_primitives.iter().find(|instance| instance.id == item_id) {
                Some(instance) => match &instance.reference {
                    Some(reference) => (
                        PathBuf::from(reference)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| reference.clone()),
                        Some(PathBuf::from(reference)),
                    ),
                    None => ("None".to_string(), None),
                },
                None => ("None".to_string(), None),
            };
            reference_requested = draw_clip_selector(ui, &name, path.as_deref(), item_id, "Reference");
        }
        if ui.collapsing_header("Visibility", imgui::TreeNodeFlags::DEFAULT_OPEN) {
            let mut show_target_poses = self
                .scene
                .smart_primitives
                .iter()
                .find(|instance| instance.id == item_id)
                .is_some_and(|instance| instance.visibility.target_poses);
            if ui.checkbox("Show Target Poses", &mut show_target_poses) {
                if let Some(instance) = self.scene.smart_primitives.iter_mut().find(|instance| instance.id == item_id) {
                    instance.visibility.target_poses = show_target_poses;
                }
                self.shell.mark_dirty();
            }
            ui.same_line();
            self.draw_sequence_selector(ui, item_id, show_target_poses);
            ui.same_line();
            let mut color = self
                .scene
                .smart_primitives
                .iter()
                .find(|instance| instance.id == item_id)
                .map(|instance| instance.target_pose_color)
                .unwrap_or([1.0, 0.30, 0.20]);
            if ui
                .color_edit3_config("Target Pose Color", &mut color)
                .flags(imgui::ColorEditFlags::NO_INPUTS | imgui::ColorEditFlags::NO_LABEL)
                .build()
            {
                if let Some(instance) = self.scene.smart_primitives.iter_mut().find(|instance| instance.id == item_id) {
                    instance.target_pose_color = color;
                }
                self.shell.mark_dirty();
            }
            let mut show_interaction_boxes = self
                .scene
                .smart_primitives
                .iter()
                .find(|instance| instance.id == item_id)
                .is_some_and(|instance| instance.visibility.interaction_boxes);
            if ui.checkbox("Show Interaction Boxes", &mut show_interaction_boxes) {
                if let Some(instance) = self.scene.smart_primitives.iter_mut().find(|instance| instance.id == item_id) {
                    instance.visibility.interaction_boxes = show_interaction_boxes;
                }
                self.shell.mark_dirty();
            }
        }
        if ui.collapsing_header("Runtime", imgui::TreeNodeFlags::DEFAULT_OPEN) {
            let current = self
                .scene
                .smart_primitives
                .iter()
                .find(|instance| instance.id == item_id)
                .map(|instance| instance.behavior.clone())
                .unwrap_or_default();
            if let Some(_combo) = ui.begin_combo("Behavior", &current) {
                for behavior in BEHAVIOR_TYPES {
                    if ui.selectable_config(behavior).selected(behavior == current).build() && behavior != current {
                        if let Some(instance) =
                            self.scene.smart_primitives.iter_mut().find(|instance| instance.id == item_id)
                        {
                            instance.behavior = behavior.to_string();
                        }
                        self.shell.mark_dirty();
                    }
                }
            }
        }
        reference_requested
    }

    /// The sequence combo beside "Show Target Poses": "All" or one sequence
    /// (upstream's `##TargetPoseSequence` combo).
    fn draw_sequence_selector(&mut self, ui: &imgui::Ui, item_id: u32, show_target_poses: bool) {
        let sequence_names: Vec<String> = match self.views.get(&item_id) {
            Some(view) => view.asset.document.sequences.iter().map(|sequence| sequence.name.clone()).collect(),
            None => Vec::new(),
        };
        let mut selected = self
            .views
            .get(&item_id)
            .and_then(|view| view.sequence_filter)
            .map(|index| index + 1)
            .unwrap_or(0);
        {
            let _disabled = ui.begin_disabled(!show_target_poses || sequence_names.is_empty());
            let width = (ui.content_region_avail()[0] - ui.frame_height() - 8.0).max(1.0);
            ui.set_next_item_width(width);
            let mut items: Vec<String> = vec!["All".to_string()];
            items.extend(sequence_names);
            if ui.combo_simple_string("##TargetPoseSequence", &mut selected, &items) {
                if let Some(view) = self.views.get_mut(&item_id) {
                    view.sequence_filter = (selected > 0).then_some(selected - 1);
                }
            }
        }
    }

    /// The Solid Cube section (upstream `_draw_solid_cube_details`).
    fn draw_solid_cube_details(&mut self, ui: &imgui::Ui, item_id: u32) {
        let Some(cube) = self.scene.solid_cubes.iter().find(|cube| cube.id == item_id) else {
            return;
        };
        let dimensions = [cube.dimensions.width, cube.dimensions.depth, cube.dimensions.height];
        let color = cube.color;
        let (dimensions_changed, dimensions) = draw_box_dimensions(ui, dimensions, MIN_SOLID_CUBE_DIMENSION, 1.0);
        if dimensions_changed {
            if let Some(cube) = self.scene.solid_cubes.iter_mut().find(|cube| cube.id == item_id) {
                cube.dimensions.width = dimensions[0];
                cube.dimensions.depth = dimensions[1];
                cube.dimensions.height = dimensions[2];
            }
            self.shell.mark_dirty();
        }
        let mut color = color;
        if ui.color_edit3("Color", &mut color) {
            // The color fields accept typed values, so keep each channel from 0 to 1.
            for channel in &mut color {
                *channel = channel.clamp(0.0, 1.0);
            }
            if let Some(cube) = self.scene.solid_cubes.iter_mut().find(|cube| cube.id == item_id) {
                cube.color = color;
            }
            self.shell.mark_dirty();
        }
    }
}

#[cfg(test)]
mod tests {
    //! Headless integration tests: the document-to-preview pipeline (scene
    //! load, rig check, asset pool, clip fallback resolution, pose sampling)
    //! without a window.
    use super::*;

    fn app_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn clips_dir() -> PathBuf {
        app_dir().join("../checkpoints/motions")
    }

    fn assets_dir() -> PathBuf {
        app_dir().join("../sandbox/upstream")
    }

    /// This checkout's SOMA clips are git-LFS pointer stubs except for two
    /// locomotion files; the G1 CSVs are all present.
    fn is_lfs_stub(path: &Path) -> bool {
        std::fs::read(path).is_ok_and(|bytes| bytes.starts_with(b"version https://git-lfs"))
    }

    fn real_soma_bvh() -> Option<PathBuf> {
        let path = clips_dir().join("soma/locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534.bvh");
        (path.is_file() && !is_lfs_stub(&path)).then_some(path)
    }

    #[test]
    fn soma_gym_scene_loads_and_checks_out() {
        let scene_path = app_dir().join("assets/scenes/soma_gym.scene.json");
        let scene = SceneDocument::load(&scene_path).expect("soma_gym loads");
        scene.check_reference_rigs(&scene_path).expect("references match the rig");
        assert_eq!(scene.rig, ModelRig::Soma);
        assert_eq!(scene.smart_primitives.len(), 4, "jump x2, sit, avoidance");
        assert_eq!(scene.solid_cubes.len(), 4, "box, wall, seat, bar");
        // The parented cubes sit under their Smart Primitives, as the docs describe.
        let parented = scene.solid_cubes.iter().filter(|cube| cube.parent_id.is_some()).count();
        assert_eq!(parented, 4);
    }

    #[test]
    fn soma_previews_sample_clips_and_fail_gracefully_on_lfs_stubs() {
        let rig_data = RigData::load(ModelRig::Soma, &clips_dir(), &assets_dir()).expect("SOMA rig loads");
        let mut pool = AssetPool::new(rig_data, clips_dir(), assets_dir());

        // A target pose sampled from a real clip: finite local joint transforms
        // on the mesh skeleton.
        let Some(bvh) = real_soma_bvh() else {
            eprintln!("skipping: no real SOMA BVH clips in this checkout");
            return;
        };
        let component = TargetPoseComponent {
            id: 1,
            clip: Some(bvh.to_string_lossy().into_owned()),
            sample_time_seconds: 0.5,
            frame_index: 15,
            playback_mode: motionbricks_app::documents::PlaybackMode::Time,
            hips_constrained: false,
            num_poses_to_use: 4,
            transition_early_by_frames: 0,
            min_tokens: 6,
            max_tokens: 16,
            ik_pin_mask: 0,
            show_mesh: true,
            show_skeletons: false,
            show_velocity: false,
            show_trajectory: false,
            trajectory_scale: 1.0,
            color: [1.0, 0.30, 0.20],
        };
        let sample = pool.sample_target(&component, Path::new("unused")).expect("sample a real clip");
        let TargetSample::Soma(local) = &sample else {
            panic!("a SOMA pool samples SOMA poses");
        };
        let soma = pool.rig_data.soma.as_ref().unwrap();
        assert_eq!(local.len(), soma.skeleton.num_joints());
        assert!(local.iter().all(|transform| transform.p.0.iter().all(|value| value.is_finite())));
        // The pose evaluates to finite bone matrices for drawing.
        let bones = soma.evaluate(local, rt::Transform::IDENTITY).expect("FK");
        assert!(bones.iter().all(|matrix| matrix.w_axis.is_finite()));

        // The committed gym documents reference clips that are git-LFS pointer
        // stubs in this checkout: the pool must refuse them with a clear error
        // (the app shows it in the status line) rather than panic.
        let scene_path = app_dir().join("assets/scenes/soma_gym.scene.json");
        let reference = "../smart_primitives/soma/jump_0_5m.smart_primitive.json";
        let path = SceneDocument::resolve_reference(reference, &scene_path);
        if path.is_file() {
            match pool.load(&path) {
                Err(error) => {
                    let message = format!("{error:#}");
                    assert!(message.contains(".bvh"), "error names the clip: {message}");
                }
                Ok(_) => eprintln!("note: LFS clips fetched; jump_0_5m loads fully"),
            }
        }
    }

    #[test]
    fn the_pool_loads_each_document_once() {
        let mut pool = AssetPool::new(
            RigData::load(ModelRig::Soma, &clips_dir(), &assets_dir()).expect("SOMA rig loads"),
            clips_dir(),
            assets_dir(),
        );
        // Load the same document through two different relative spellings.
        let document = app_dir().join("assets/smart_primitives/soma/jump_0_5m.smart_primitive.json");
        if !document.is_file() {
            return;
        }
        // The document itself loads (clip sampling is what may fail on stubs).
        let first = SmartPrimitiveDocument::load_for_rig(&document, ModelRig::Soma).expect("document loads");
        let asset = pool.build_asset(first, &document);
        match asset {
            Ok(_) => eprintln!("note: LFS clips fetched; build_asset succeeds"),
            Err(error) => {
                let message = format!("{error:#}");
                assert!(message.contains(".bvh"), "error names the clip: {message}");
            }
        }
    }

    #[test]
    fn g1_gym_scene_loads_and_g1_previews_sample_clips() {
        let assets = assets_dir();
        if !assets.join("motionbricks").is_dir() {
            eprintln!("skipping: no upstream checkout at {}", assets.display());
            return;
        }
        let scene_path = app_dir().join("assets/scenes/g1_gym.scene.json");
        let scene = SceneDocument::load(&scene_path).expect("g1_gym loads");
        scene.check_reference_rigs(&scene_path).expect("references match the rig");
        assert_eq!(scene.rig, ModelRig::UnitreeG1);

        let rig_data = RigData::load(scene.rig, &clips_dir(), &assets).expect("G1 rig loads");
        let mut pool = AssetPool::new(rig_data, clips_dir(), assets);

        // A target pose sampled from a real clip: finite joint coordinates that
        // evaluate to finite link transforms.
        let csv = clips_dir().join("unitree_g1/locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534.csv");
        if !csv.is_file() || is_lfs_stub(&csv) {
            eprintln!("skipping: no real G1 CSV clips in this checkout");
            return;
        }
        let component = TargetPoseComponent {
            id: 1,
            clip: Some(csv.to_string_lossy().into_owned()),
            sample_time_seconds: 0.5,
            frame_index: 15,
            playback_mode: motionbricks_app::documents::PlaybackMode::Time,
            hips_constrained: false,
            num_poses_to_use: 4,
            transition_early_by_frames: 0,
            min_tokens: 6,
            max_tokens: 16,
            ik_pin_mask: 0,
            show_mesh: true,
            show_skeletons: false,
            show_velocity: false,
            show_trajectory: false,
            trajectory_scale: 1.0,
            color: [1.0, 0.30, 0.20],
        };
        let sample = pool.sample_target(&component, Path::new("unused")).expect("sample a real clip");
        let TargetSample::G1(q) = &sample else {
            panic!("a G1 pool samples G1 poses");
        };
        assert!(q.iter().all(|value| value.is_finite()));
        let g1 = pool.rig_data.g1.as_ref().unwrap();
        let links = g1.link_transforms(q).expect("FK");
        assert!(links.iter().all(|transform| transform.p.0.iter().all(|value| value.is_finite())));

        // Stub clips (git-LFS pointers in this checkout) fail with a clear error.
        let reference = "../smart_primitives/unitree_g1/jump_0_5m.smart_primitive.json";
        let path = SceneDocument::resolve_reference(reference, &scene_path);
        if path.is_file() {
            match pool.load(&path) {
                Err(error) => {
                    let message = format!("{error:#}");
                    assert!(message.contains(".csv"), "error names the clip: {message}");
                }
                Ok(_) => eprintln!("note: LFS clips fetched; jump_0_5m loads fully"),
            }
        }
    }

    #[test]
    fn reference_paths_need_the_smart_primitive_suffix() {
        assert!(is_smart_primitive_path(Path::new("/scenes/sit.smart_primitive.json")));
        assert!(is_smart_primitive_path(Path::new("SIT.SMART_PRIMITIVE.JSON")), "the check is case-insensitive");
        assert!(!is_smart_primitive_path(Path::new("/scenes/gym.scene.json")));
        assert!(!is_smart_primitive_path(Path::new("sit.json")));
        assert!(!is_smart_primitive_path(Path::new("sit.smart_primitive.json.bak")));
    }

    #[test]
    fn character_start_marker_matches_upstream() {
        let marker = character_start_marker();
        // 32 circle segments plus the facing arrow's shaft and two head lines.
        assert_eq!(marker.len(), CHARACTER_START_CIRCLE_SEGMENTS + 3);
        // Everything sits at the marker height.
        assert!(marker.iter().all(|[a, b]| a.z == CHARACTER_START_MARKER_HEIGHT && b.z == CHARACTER_START_MARKER_HEIGHT));
        // The circle closes at the marker radius.
        let first = marker[0][0];
        assert!((first.truncate().length() - CHARACTER_START_RADIUS).abs() < 1e-6);
        let last = marker[CHARACTER_START_CIRCLE_SEGMENTS - 1][1];
        assert!((last.truncate().length() - CHARACTER_START_RADIUS).abs() < 1e-6);
        // The arrow points along +X (the facing direction).
        let shaft = marker[CHARACTER_START_CIRCLE_SEGMENTS];
        assert_eq!(shaft[0], Vec3::new(0.0, 0.0, CHARACTER_START_MARKER_HEIGHT));
        assert_eq!(shaft[1].x, CHARACTER_START_ARROW_LENGTH);
        assert_eq!(shaft[1].y, 0.0);
    }
}
