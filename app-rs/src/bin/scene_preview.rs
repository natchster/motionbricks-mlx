// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Scene Preview (port of upstream `app/scene_preview.py`): play a character in a saved
//! scene and try its Smart Primitive interactions. The character runtime (`Character`,
//! `StateMachine`, locomotion modes, behaviors) lives in [`motionbricks_app::runtime`];
//! generation goes through the Rust model and Runtime API (`motionbricks_rs`) on the
//! Metal runtime engine — never Python, TensorRT, or ONNX Runtime.
//!
//! The main loop mirrors upstream's `step()`: poll input, update the camera-relative
//! locomotion command (WASD/arrows move, Space runs, O/P/[ switch modes), step the
//! character, spring-follow it with the camera, test the probe box against every
//! interaction box, then start or pass on interaction input (I).
//!
//! Deviations from upstream (Newton/pyglet/Warp):
//! - The runtime world is Z-up; the viewer is Y-up. [`ZUP_TO_YUP`] converts at draw time,
//!   as in the Inbetweener.
//! - The gamepad is not ported; keyboard and mouse only. The viewer's built-in left-drag
//!   orbits and scroll zooms (upstream orbits with the middle mouse button in the Newton
//!   viewer); the camera target is re-pinned to the character every frame, as upstream's
//!   pivot follow does.
//! - The "..." scene Open dialog and runtime scene switching are not ported; the scene is
//!   chosen on the command line. One engine is loaded, for the scene's rig.
//! - Clip references in documents resolve relative to the document (upstream's assets
//!   root holds `motions/`); when that file does not exist, the tail after the last
//!   `motions/` component is resolved under `--clips` (default: the upstream checkout's
//!   full clip library, `sandbox/upstream/assets/motions`).
//! - Behavior clips (e.g. the sitting loop) load lazily when an interaction starts, not
//!   at scene load (upstream's `InteractionState.prepare` has no counterpart here), and
//!   each interaction start creates a fresh behavior instance from the shared asset
//!   (upstream reuses one behavior object across interactions).
//! - The shown context poses are the ones of the last generation; upstream refreshes the
//!   would-be-next context from the recorder before drawing, which needs mutable player
//!   access the runtime does not expose. Likewise, a running behavior's pivot alignment
//!   is not reflected in the active Smart Primitive's debug drawings (its interaction
//!   boxes are re-tested every frame, which is what matters for prompts).
//!
//! Run with: `cargo run --release --bin scene-preview -- [--scene assets/scenes/soma_gym.scene.json]`

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context as _, Result, bail};
use glam::{Mat4, Vec3};
use imgui::{Condition, SliderFlags, StyleColor, TableFlags, TreeNodeFlags, Ui};
use lexopt::prelude::*;
use motionbricks_app::documents::{SceneDocument, SmartPrimitiveDocument, SolidCube};
use motionbricks_app::rig::clips::{Clip, ClipRegistry};
use motionbricks_app::rig::g1::G1Rig;
use motionbricks_app::rig::soma::{self, SoleOutline, SomaRig};
use motionbricks_app::runtime::behaviors::{BehaviorRegistry, SmartPrimitive, SmartPrimitiveAsset};
use motionbricks_app::runtime::blender::{G1TargetPoseBlender, SomaTargetPoseBlender};
use motionbricks_app::runtime::character::{Character, ClipLoader, PlayerAdapter};
use motionbricks_app::runtime::interaction::{InteractionBox, compute_interaction_box_overlaps};
use motionbricks_app::runtime::motion::{
    self, AnimationClip, LocomotionCommand, PoseElem, ground_direction, load_locomotion_mode, make_animation_clip,
};
use motionbricks_app::runtime::presets;
use motionbricks_app::viewer::mesh::wire_box_segments;
use motionbricks_app::viewer::{Color, Frame, Input, Mesh, OrbitCamera, Skin, Viewer, ViewerConfig};
use motionbricks_rs::config::get_spec;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::runtime as rt;
use rt::animation::MotionBuffer;
use rt::math::{self as rtm, Transform};
use rt::{
    AnimationBuffer, DofAnimationBuffer, DofToken, FootContacts, ModelRig, MotionBricksEngine, MotionBricksSomaEngine,
    MotionBricksUnitreeG1Engine, MotionToken, PoseToken, TargetPoseIkCorrector,
};
use winit::keyboard::KeyCode;

// --- Constants (upstream's module-level values) ---------------------------------------------------

const CAMERA_DISTANCE: f32 = 5.0;
const CAMERA_ELEVATION_DEGREES: f32 = 25.0;
const CAMERA_PIVOT_Z_OFFSET: f32 = 0.8;
const CAMERA_VELOCITY_HALFLIFE: f32 = 0.15;
const CAMERA_PULL_SECONDS: f32 = 0.05;
const VELOCITY_ARROW_HEIGHT: f32 = 0.025;
const CHARACTER_VELOCITY_ARROW_COLOR: [f32; 3] = [1.0, 0.25, 0.1];
const INPUT_VELOCITY_ARROW_COLOR: [f32; 3] = [0.15, 0.45, 1.0];
const DESIRED_VELOCITY_ARROW_COLOR: [f32; 3] = [0.15, 1.0, 0.35];
const CONTEXT_COLOR: [f32; 4] = [0.65, 0.3, 0.85, 1.0];
const TARGET_COLOR: [f32; 4] = [0.2, 0.75, 1.0, 1.0];
const BLENDSPACE_DEBUG_COLOR: [f32; 4] = [0.15, 1.0, 0.35, 1.0];
const INTERACTION_BOX_COLOR: [f32; 4] = [1.0, 0.25, 0.85, 1.0];
const INACTIVE_SMART_PRIMITIVE_BOX_COLOR: [f32; 4] = [1.0, 0.15, 0.1, 1.0];
const ACTIVE_SMART_PRIMITIVE_BOX_COLOR: [f32; 4] = [0.15, 1.0, 0.25, 1.0];
/// Upstream `CHARACTER_COLORS`: SOMA playback draws gray; the G1 draws its own link colors.
const PLAYBACK_SOMA_COLOR: [f32; 4] = [0.7, 0.7, 0.7, 1.0];
const CONTACT_COLOR: [f32; 4] = [0.3, 1.0, 0.4, 1.0];
const PANEL_WIDTH: f32 = 340.0;
const PANEL_HEIGHT: f32 = 390.0;
const PANEL_ALPHA: f32 = 0.8;
const WINDOW_MARGIN: f32 = 12.0;
const MAX_FRAME_SECONDS: f32 = 0.1;
const MIN_FRAME_SECONDS: f32 = 1.0e-6;
/// Upstream `FootContactRenderer.DEFAULT_THRESHOLD`.
const DEFAULT_CONTACT_THRESHOLD: f32 = 0.5;
/// Upstream `GeneratedMotionPlayer`'s default history capacity.
const HISTORY_CAPACITY: usize = 32;
const SOMA_ROOT_FORWARD: Vec3 = Vec3::Y;
const G1_ROOT_FORWARD: Vec3 = Vec3::X;

/// The runtime world is Z-up; the viewer is Y-up. This rotation (-90 degrees about X) maps
/// world space to viewer space: (x, y, z) -> (x, z, -y).
const ZUP_TO_YUP: Transform = Transform::new(rt::Vec3::ZERO, [-std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2]);

fn zup_to_yup_mat() -> Mat4 {
    soma::transform_to_mat4(ZUP_TO_YUP)
}

/// World-space (Z-up) point -> viewer-space (Y-up) point.
fn to_view(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}

/// Viewer-space (Y-up) direction -> world-space (Z-up) direction.
fn to_world_dir(d: Vec3) -> Vec3 {
    Vec3::new(d.x, -d.z, d.y)
}

fn color4(c: [f32; 3]) -> Color {
    [c[0], c[1], c[2], 1.0]
}

/// A scene document's transform as a runtime transform.
fn rt_transform(t: motionbricks_app::documents::Transform) -> Transform {
    Transform::new(
        rtm::Vec3::new(t.translation.x, t.translation.y, t.translation.z),
        [t.rotation_xyzw.x, t.rotation_xyzw.y, t.rotation_xyzw.z, t.rotation_xyzw.w],
    )
}

// --- Command line -----------------------------------------------------------------------------------

struct Cli {
    scene: PathBuf,
    /// Override of the scene's rig (must match it; upstream picks the rig from the scene).
    rig: Option<ModelRig>,
    /// A `motionbricks_rs::config::Precision` name ("float32", "float16", "bfloat16").
    precision: String,
    /// Root holding the per-rig checkpoint folders (default `checkpoints/MotionBricks`).
    checkpoints: PathBuf,
    /// Clip root (default `sandbox/upstream/assets/motions`, the full clip library).
    clips: PathBuf,
    /// Upstream checkout root, for the G1 MJCF and STL meshes (default `sandbox/upstream`).
    assets: PathBuf,
}

impl Cli {
    fn precision_label(&self) -> &'static str {
        match self.precision.as_str() {
            "float16" => "Mixed FP16",
            "bfloat16" => "BF16",
            _ => "FP32",
        }
    }
}

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map_or(manifest.clone(), PathBuf::from)
}

/// The app crate's asset root: scenes and Smart Primitive documents (upstream `ASSETS_ROOT`).
fn app_assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn usage() -> &'static str {
    "MotionBricks Scene Preview (Rust, Metal runtime)\n\
     Usage: scene-preview [OPTIONS]\n\
     \x20 --scene PATH              scene document (default app-rs/assets/scenes/soma_gym.scene.json); its rig selects the character\n\
     \x20 --rig {soma|unitree_g1}   character rig (default: from the scene; must match it)\n\
     \x20 --precision {fp32|bf16}   Metal runtime dtype (default fp32; fp16 also accepted)\n\
     \x20 --checkpoints DIR         checkpoint root holding soma/ and g1/ (default checkpoints/MotionBricks)\n\
     \x20 --clips DIR               clip root (default sandbox/upstream/assets/motions, the full library)\n\
     \x20 --assets DIR              upstream checkout root, for G1 meshes (default sandbox/upstream)\n\
     \x20 --help                    show this text"
}

fn parse_cli() -> Result<Cli> {
    let root = repo_root();
    let mut cli = Cli {
        scene: app_assets_root().join("scenes/soma_gym.scene.json"),
        rig: None,
        precision: "float32".to_string(),
        checkpoints: root.join("checkpoints/MotionBricks"),
        clips: root.join("sandbox/upstream/assets/motions"),
        assets: root.join("sandbox/upstream"),
    };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            lexopt::Arg::Long("scene") => cli.scene = PathBuf::from(parser.value()?),
            lexopt::Arg::Long("rig") => cli.rig = Some(ModelRig::parse(&parser.value()?.string()?)?),
            lexopt::Arg::Long("precision") => {
                let v = parser.value()?.string()?;
                cli.precision = match v.as_str() {
                    "fp32" | "float32" => "float32",
                    "bf16" | "bfloat16" => "bfloat16",
                    "fp16" | "float16" => "float16",
                    other => bail!("unknown precision {other:?} (expected fp32 or bf16)"),
                }
                .to_string();
            }
            lexopt::Arg::Long("checkpoints") => cli.checkpoints = PathBuf::from(parser.value()?),
            lexopt::Arg::Long("clips") => cli.clips = PathBuf::from(parser.value()?),
            lexopt::Arg::Long("assets") => cli.assets = PathBuf::from(parser.value()?),
            lexopt::Arg::Long("help") | lexopt::Arg::Short('h') => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(cli)
}

// --- Clip path resolution ----------------------------------------------------------------------------

/// Resolve a clip path: as given when the file exists, else the part after its last
/// `motions` component under `clips_dir`. Upstream's asset root holds `motions/` next to
/// `smart_primitives/`; this port keeps the clip library in a separate directory.
fn resolve_clip(path: &Path, clips_dir: &Path) -> PathBuf {
    if path.is_file() {
        return path.to_path_buf();
    }
    let components: Vec<_> = path.components().collect();
    if let Some(index) = components.iter().rposition(|c| c.as_os_str() == "motions") {
        let tail: PathBuf = components[index + 1..].iter().collect();
        if !tail.as_os_str().is_empty() {
            return clips_dir.join(tail);
        }
    }
    path.to_path_buf()
}

/// The clip registry, the clip root, and the space conversion, shared by the asset
/// building and the character's clip loader.
struct ClipSource {
    registry: RefCell<ClipRegistry>,
    clips_dir: PathBuf,
    model_to_runtime: Transform,
}

impl ClipSource {
    fn resolve(&self, path: &Path) -> PathBuf {
        resolve_clip(path, &self.clips_dir)
    }

    /// Load a clip; `loop_` returns a copy whose ends join (upstream `load_clip`).
    fn clip(&self, path: &Path, loop_: bool) -> Result<Clip> {
        let resolved = self.resolve(path);
        let mut registry = self.registry.borrow_mut();
        let clip = if loop_ { registry.load_looping(&resolved)? } else { registry.load(&resolved)?.clip.clone() };
        Ok(clip)
    }
}

// --- Input (upstream `app/utils/input.py`, keyboard only) ---------------------------------------------

/// Upstream `apply_radial_dead_zone`: ignore small inputs and scale the rest from zero
/// to one. With a zero dead zone this normalizes diagonal key movement to length 1.
fn apply_radial_dead_zone(x: f32, y: f32, dead_zone: f32) -> (f32, f32) {
    let length = x.hypot(y);
    if length <= dead_zone {
        return (0.0, 0.0);
    }
    let magnitude = (length.min(1.0) - dead_zone) / (1.0 - dead_zone);
    (x / length * magnitude, y / length * magnitude)
}

/// WASD/arrow keys as a (right, forward) movement input (upstream `InputManager.poll`'s
/// keyboard half).
fn poll_movement(input: &Input) -> (f32, f32) {
    let key = |code: KeyCode| input.key_held(code);
    let right = (key(KeyCode::KeyD) || key(KeyCode::ArrowRight)) as i32 - (key(KeyCode::KeyA) || key(KeyCode::ArrowLeft)) as i32;
    let forward = (key(KeyCode::KeyW) || key(KeyCode::ArrowUp)) as i32 - (key(KeyCode::KeyS) || key(KeyCode::ArrowDown)) as i32;
    apply_radial_dead_zone(right as f32, forward as f32, 0.0)
}

/// Upstream `camera_relative_movement`: map (right, forward) onto the ground relative to
/// the camera facing. `None` means no movement (what `MovementControl` reads as idle).
fn camera_relative_movement(camera_facing: Vec3, movement: (f32, f32)) -> Option<Vec3> {
    let (right, forward) = movement;
    let camera_right = Vec3::new(camera_facing.y, -camera_facing.x, 0.0);
    let movement = camera_facing * forward + camera_right * right;
    (movement.truncate().length() > motion::MOVEMENT_EPSILON).then_some(movement)
}

/// Upstream `LocomotionModeInput` (keyboard half): O toggles Forward/Strafing, P toggles
/// Crouching/Strafing, and [ cycles the style modes (returning to the last style used).
struct LocomotionModeInput {
    style_modes: Vec<String>,
    last_style: Option<String>,
}

impl LocomotionModeInput {
    fn new(style_modes: &[String]) -> Self {
        LocomotionModeInput { style_modes: style_modes.to_vec(), last_style: style_modes.first().cloned() }
    }

    /// The mode a newly pressed toggle asks for, or `None` (upstream `poll`).
    /// `o`, `p` and `style` are edge-triggered presses; `current_mode` is the active one.
    fn poll(&mut self, o: bool, p: bool, style: bool, current_mode: &str) -> Option<String> {
        if self.style_modes.iter().any(|mode| mode == current_mode) {
            self.last_style = Some(current_mode.to_string());
        }
        let mut requested = None;
        for (pressed, first, second) in [
            (o, motion::LINEAR_LOCOMOTION_MODE, motion::DIRECTIONAL_LOCOMOTION_MODE),
            (p, motion::CROUCH_LOCOMOTION_MODE, motion::DIRECTIONAL_LOCOMOTION_MODE),
        ] {
            if pressed {
                requested = Some(if current_mode == first { second } else { first }.to_string());
            }
        }
        if style && !self.style_modes.is_empty() {
            requested = Some(self.next_style(current_mode));
        }
        if let Some(mode) = &requested {
            if self.style_modes.iter().any(|style| style == mode) {
                self.last_style = Some(mode.clone());
            }
        }
        requested
    }

    fn next_style(&self, current_mode: &str) -> String {
        match self.style_modes.iter().position(|mode| mode == current_mode) {
            Some(index) => self.style_modes[(index + 1) % self.style_modes.len()].clone(),
            None => self.last_style.clone().expect("a last style exists when style modes do"),
        }
    }
}

/// A copy of `runtime::motion`'s spring (it is `pub(crate)` there): upstream's
/// `critical_spring_damper_exact`, used for the camera follow velocity.
fn critical_spring_damper_exact(current: f32, current_rate: f32, goal: f32, halflife: f32, delta_time: f32) -> (f32, f32) {
    let damping = (4.0 * std::f32::consts::LN_2) / (halflife + 1.0e-5);
    let decay_rate = damping / 2.0;
    let position_offset = current - goal;
    let velocity_offset = current_rate + position_offset * decay_rate;
    let decay = (-decay_rate * delta_time).exp();
    let next_current = decay * (position_offset + velocity_offset * delta_time) + goal;
    let next_rate = decay * (current_rate - velocity_offset * decay_rate * delta_time);
    if (next_current - goal).abs() <= 1.0e-4 && next_rate.abs() <= 1.0e-4 {
        return (goal, 0.0);
    }
    (next_current, next_rate)
}

// --- Pose drawing (same rig-visual helpers as the Inbetweener) -----------------------------------------

/// One pose of either rig: SOMA local joint transforms, or G1 joint coordinates.
#[derive(Clone, Debug)]
enum PoseData {
    Soma(Vec<Transform>),
    G1(Vec<f32>),
}

fn as_soma(pose: &PoseData) -> Result<&[Transform]> {
    match pose {
        PoseData::Soma(p) => Ok(p),
        PoseData::G1(_) => bail!("a SOMA pose was expected"),
    }
}

fn as_g1(pose: &PoseData) -> Result<&[f32]> {
    match pose {
        PoseData::G1(q) => Ok(q),
        PoseData::Soma(_) => bail!("a G1 pose was expected"),
    }
}

/// Convert a runtime pose into drawable form, for both pose layouts.
trait PoseDraw: PoseElem {
    fn pose_data(pose: &[Self]) -> PoseData;
}

impl PoseDraw for Transform {
    fn pose_data(pose: &[Self]) -> PoseData {
        PoseData::Soma(pose.to_vec())
    }
}

impl PoseDraw for f32 {
    fn pose_data(pose: &[Self]) -> PoseData {
        PoseData::G1(pose.to_vec())
    }
}

/// Read the pose slots of a token of either kind (the `MotionToken` trait has no poses
/// accessor).
trait TokenPoses<E> {
    fn token_poses(&self) -> &[Vec<E>];
}

impl TokenPoses<Transform> for PoseToken {
    fn token_poses(&self) -> &[Vec<Transform>] {
        self.poses()
    }
}

impl TokenPoses<f32> for DofToken {
    fn token_poses(&self) -> &[Vec<f32>] {
        self.poses()
    }
}

/// Per-rig drawing assets: what upstream's mesh/skeleton/foot-contact renderers hold.
enum RigVisual {
    Soma {
        // Boxed: SomaRig is ~450 bytes (G1Rig is small), so the enum stays compact.
        rig: Box<SomaRig>,
        skin: Skin,
        outlines: Vec<SoleOutline>,
    },
    G1 {
        rig: G1Rig,
        /// One viewer mesh per link (slices of the merged mesh, rebased).
        link_meshes: Vec<Mesh>,
        /// Body index of each foot-contact body, in contact order.
        contact_bodies: [usize; rt::FOOT_CONTACT_COUNT],
    },
}

impl RigVisual {
    /// The character's tint: gray SOMA, the G1's own link colors (upstream `CHARACTER_COLORS`).
    fn character_color(&self) -> Option<Color> {
        match self {
            RigVisual::Soma { .. } => Some(PLAYBACK_SOMA_COLOR),
            RigVisual::G1 { .. } => None,
        }
    }

    /// Log a pose's mesh: the SOMA skinned mesh or the G1's rigid links.
    fn draw_pose(&self, frame: &mut Frame, pose: &PoseData, color: Option<Color>) -> Result<()> {
        match self {
            RigVisual::Soma { rig, skin, .. } => {
                let world = rig.fk(as_soma(pose)?, ZUP_TO_YUP)?;
                let bones = rig.bone_matrices(&world);
                frame.log_skinned_mesh(&rig.mesh.mesh, skin, &bones, Mat4::IDENTITY, color.unwrap_or(PLAYBACK_SOMA_COLOR));
            }
            RigVisual::G1 { rig, link_meshes, .. } => {
                let world = rig.fk(as_g1(pose)?)?;
                let zup_to_yup = zup_to_yup_mat();
                for (link, mesh) in rig.links.iter().zip(link_meshes) {
                    let m = zup_to_yup * soma::transform_to_mat4(world[link.body] * link.offset);
                    frame.log_mesh(mesh, m, color.unwrap_or(link.rgba));
                }
            }
        }
        Ok(())
    }

    /// Log a pose as bone lines (upstream's skeleton renderers), in world space.
    fn draw_skeleton(&self, frame: &mut Frame, pose: &PoseData, color: Color) -> Result<()> {
        match self {
            RigVisual::Soma { rig, .. } => {
                // FK under ZUP_TO_YUP yields viewer-space joint transforms directly.
                let world = rig.fk(as_soma(pose)?, ZUP_TO_YUP)?;
                let mut segments = Vec::with_capacity(world.len().saturating_sub(1));
                for (j, w) in world.iter().enumerate().skip(1) {
                    let parent = world[rig.skeleton.parent_indices()[j] as usize];
                    segments.push([Vec3::from_array(w.p.0), Vec3::from_array(parent.p.0)]);
                }
                frame.log_lines(&segments, color);
            }
            RigVisual::G1 { rig, .. } => {
                let world = rig.fk(as_g1(pose)?)?;
                let mut segments = Vec::with_capacity(world.len().saturating_sub(1));
                for (b, w) in world.iter().enumerate().skip(1) {
                    let parent = world[usize::try_from(rig.bodies[b].parent)?];
                    segments.push([to_view(Vec3::from_array(w.p.0)), to_view(Vec3::from_array(parent.p.0))]);
                }
                frame.log_lines(&segments, color);
            }
        }
        Ok(())
    }

    /// Log the foot contacts of `pose` that are above `threshold` (upstream
    /// `FootContactRenderer.draw_pose`): SOMA sole outlines, G1 rings at the contact points.
    fn draw_foot_contacts(&self, frame: &mut Frame, pose: &PoseData, contacts: FootContacts, threshold: f32) -> Result<()> {
        const CONTACT_RING_SEGMENTS: usize = 12;
        const CONTACT_RING_RADIUS: f32 = 0.07;
        match self {
            RigVisual::Soma { rig, outlines, .. } => {
                let world = rig.fk(as_soma(pose)?, ZUP_TO_YUP)?;
                for outline in outlines {
                    if contacts[outline.contact] > threshold {
                        frame.log_lines(&soma::outline_segments(std::slice::from_ref(outline), &world), CONTACT_COLOR);
                    }
                }
            }
            RigVisual::G1 { rig, contact_bodies, .. } => {
                let world = rig.fk(as_g1(pose)?)?;
                for (c, &body) in contact_bodies.iter().enumerate() {
                    if contacts[c] <= threshold {
                        continue;
                    }
                    let center = to_view(Vec3::from_array(world[body].point(rt::Vec3(rt::UNITREE_G1_FOOT_CONTACT_OFFSETS[c])).0));
                    let segments: Vec<[Vec3; 2]> = (0..CONTACT_RING_SEGMENTS)
                        .map(|i| {
                            let point = |k: usize| {
                                let a = k as f32 * std::f32::consts::TAU / CONTACT_RING_SEGMENTS as f32;
                                center + Vec3::new(a.cos(), 0.0, a.sin()) * CONTACT_RING_RADIUS
                            };
                            [point(i), point(i + 1)]
                        })
                        .collect();
                    frame.log_lines(&segments, CONTACT_COLOR);
                }
            }
        }
        Ok(())
    }
}

/// Log an oriented box's 12 edges (upstream's `WireBoxRenderer`), converted to viewer space.
fn log_oriented_box(frame: &mut Frame, interaction_box: &InteractionBox, color: Color) {
    let local = wire_box_segments(Vec3::ZERO, interaction_box.half_extents);
    let point = |p: Vec3| to_view(Vec3::from_array(interaction_box.transform.point(rtm::Vec3::new(p.x, p.y, p.z)).0));
    let segments: Vec<[Vec3; 2]> = local.iter().map(|&[a, b]| [point(a), point(b)]).collect();
    frame.log_lines(&segments, color);
}

// --- The scene session (upstream `_SceneSession` + `_RuntimeSceneSmartPrimitive`) -----------------------

/// A placed Smart Primitive: its display instance (nodes, boxes, target poses) and the
/// asset its behavior is created from on each interaction start.
struct PrimitiveInstance<E> {
    id: u32,
    name: String,
    behavior_name: String,
    /// World placement of the root node (the flattened scene transform).
    world_transform: Transform,
    show_target_poses: bool,
    show_interaction_boxes: bool,
    target_pose_color: [f32; 3],
    asset: Rc<SmartPrimitiveAsset<E>>,
    runtime: SmartPrimitive<E>,
    /// Whether the character's probe box overlaps one of this instance's boxes.
    interactable: bool,
}

/// A Solid Cube as a drawable mesh (upstream's `SolidBoxRenderer` input).
struct CubeVisual {
    mesh: Mesh,
    transform: Mat4,
    color: Color,
}

fn cube_visual(cube: &SolidCube) -> CubeVisual {
    let half = Vec3::new(cube.dimensions.width, cube.dimensions.depth, cube.dimensions.height) * 0.5;
    CubeVisual {
        mesh: Mesh::cube(Vec3::ZERO, half),
        transform: zup_to_yup_mat() * soma::transform_to_mat4(rt_transform(cube.transform)),
        color: color4(cube.color),
    }
}

/// The character, the scene items, and the interaction and camera state for one scene.
struct Session<E: MotionBricksEngine> {
    character: Character<PlayerAdapter<E>>,
    scene_name: String,
    cubes: Vec<CubeVisual>,
    behavior_registry: Rc<BehaviorRegistry<E::Elem>>,
    instances: Vec<PrimitiveInstance<E::Elem>>,
    nearest_interactable: Option<usize>,
    /// Scene item id of the instance the active interaction started on.
    active_interaction: Option<u32>,
    /// Instances auto-interaction already ran on, until they leave the probe.
    auto_interaction_blocked: HashSet<u32>,
    auto_interaction: bool,
    /// The latest runtime error, shown in the window (upstream's `scene_error`).
    error: Option<String>,
    mode_input: LocomotionModeInput,
    camera_facing: Vec3,
    camera_target: Vec3,
    camera_velocity: [f32; 2],
    camera_rate: [f32; 2],
    initial_camera: OrbitCamera,
    /// Scratch for the blend-space debug poses.
    blended_poses: Vec<Vec<E::Elem>>,
    blendspace_poses: Vec<Vec<E::Elem>>,
}

impl<E: MotionBricksEngine> Session<E>
where
    E::Elem: PoseElem,
{
    fn new(
        character: Character<PlayerAdapter<E>>,
        scene: &SceneDocument,
        behavior_registry: Rc<BehaviorRegistry<E::Elem>>,
        instances: Vec<PrimitiveInstance<E::Elem>>,
    ) -> Self {
        let pose_len = character.current_pose.len();
        let camera_facing = character.movement.current_facing;
        let camera_target = character.character_ground_position;
        let initial_camera = follow_camera_at(camera_target, camera_facing);
        let style_modes = character.state_machine.style_modes().to_vec();
        Session {
            character,
            scene_name: scene.name.clone(),
            cubes: scene.solid_cubes.iter().map(cube_visual).collect(),
            behavior_registry,
            instances,
            nearest_interactable: None,
            active_interaction: None,
            auto_interaction_blocked: HashSet::new(),
            auto_interaction: true,
            error: None,
            mode_input: LocomotionModeInput::new(&style_modes),
            camera_facing,
            camera_target,
            camera_velocity: [0.0; 2],
            camera_rate: [0.0; 2],
            initial_camera,
            blended_poses: vec![vec![E::Elem::default(); pose_len]; motion::POSES_PER_TOKEN],
            blendspace_poses: vec![vec![E::Elem::default(); pose_len]; motion::POSES_PER_TOKEN],
        }
    }
}

/// The camera that looks at the character from behind its facing (upstream `_initialize_camera`).
fn follow_camera_at(ground_position: Vec3, facing: Vec3) -> OrbitCamera {
    let elevation = CAMERA_ELEVATION_DEGREES.to_radians();
    let pivot = ground_position + Vec3::new(0.0, 0.0, CAMERA_PIVOT_Z_OFFSET);
    let eye = pivot - facing * (CAMERA_DISTANCE * elevation.cos()) + Vec3::new(0.0, 0.0, CAMERA_DISTANCE * elevation.sin());
    let (target, eye) = (to_view(pivot), to_view(eye));
    let d = eye - target;
    OrbitCamera { target, distance: d.length(), pitch: (d.y / d.length()).asin(), yaw: d.x.atan2(d.z) }
}

/// Create the scene's Smart Primitive instances, upstream's `_check_scene` +
/// `_create_smart_primitive_instances`: references must exist, behaviors must be known,
/// and each behavior is created once up front so bad sequences fail at scene load.
fn build_instances<E: PoseElem + 'static>(
    scene: &SceneDocument,
    scene_path: &Path,
    load_asset: &mut dyn FnMut(&Path) -> Result<Rc<SmartPrimitiveAsset<E>>>,
    registry: &BehaviorRegistry<E>,
) -> Result<Vec<PrimitiveInstance<E>>> {
    let mut instances = Vec::with_capacity(scene.smart_primitives.len());
    for definition in &scene.smart_primitives {
        let reference = definition
            .reference
            .as_deref()
            .with_context(|| format!("Smart Primitive {:?} has no reference.", definition.name))?;
        let path = SceneDocument::resolve_reference(reference, scene_path);
        let asset = load_asset(&path)?;
        registry.create(&definition.behavior, asset.clone()).with_context(|| format!("Smart Primitive {:?}", definition.name))?;
        let world_transform = rt_transform(definition.transform);
        let mut runtime = SmartPrimitive::new(asset.clone());
        runtime.set_node_world_transform(0, world_transform);
        instances.push(PrimitiveInstance {
            id: definition.id,
            name: definition.name.clone(),
            behavior_name: definition.behavior.clone(),
            world_transform,
            show_target_poses: definition.visibility.target_poses,
            show_interaction_boxes: definition.visibility.interaction_boxes,
            target_pose_color: definition.target_pose_color,
            asset,
            runtime,
            interactable: false,
        });
    }
    Ok(instances)
}

/// Load a Smart Primitive asset per document path, shared by the locomotion modes and
/// the scene's instances (upstream's `SmartPrimitiveAssetPool`).
struct AssetPool<E> {
    model_to_runtime: Transform,
    clip_source: Rc<ClipSource>,
    assets: HashMap<PathBuf, Rc<SmartPrimitiveAsset<E>>>,
}

impl<E: PoseElem> AssetPool<E> {
    fn new(model_to_runtime: Transform, clip_source: Rc<ClipSource>) -> Self {
        AssetPool { model_to_runtime, clip_source, assets: HashMap::new() }
    }
}

impl AssetPool<Transform> {
    fn load(&mut self, path: &Path) -> Result<Rc<SmartPrimitiveAsset<Transform>>> {
        if let Some(asset) = self.assets.get(path) {
            return Ok(asset.clone());
        }
        let document = SmartPrimitiveDocument::load(path)?;
        let clip_source = self.clip_source.clone();
        let asset = Rc::new(SmartPrimitiveAsset::build_from_document(&document, path, self.model_to_runtime, move |clip_path| {
            let clip = clip_source.clip(clip_path, false)?;
            let buffer = clip.as_soma().cloned().with_context(|| format!("expected a SOMA (.bvh) clip: {}", clip_path.display()))?;
            Ok(Rc::new(buffer) as Rc<dyn MotionBuffer<Elem = Transform>>)
        })?);
        self.assets.insert(path.to_path_buf(), asset.clone());
        Ok(asset)
    }
}

impl AssetPool<f32> {
    fn load(&mut self, path: &Path) -> Result<Rc<SmartPrimitiveAsset<f32>>> {
        if let Some(asset) = self.assets.get(path) {
            return Ok(asset.clone());
        }
        let document = SmartPrimitiveDocument::load(path)?;
        let clip_source = self.clip_source.clone();
        let asset = Rc::new(SmartPrimitiveAsset::build_from_document(&document, path, self.model_to_runtime, move |clip_path| {
            let clip = clip_source.clip(clip_path, false)?;
            let buffer = clip.as_g1().cloned().with_context(|| format!("expected a Unitree G1 (.csv) clip: {}", clip_path.display()))?;
            Ok(Rc::new(buffer) as Rc<dyn MotionBuffer<Elem = f32>>)
        })?);
        self.assets.insert(path.to_path_buf(), asset.clone());
        Ok(asset)
    }
}

/// Load the engine behind the Runtime API: the Metal runtime from the rig's checkpoints.
fn build_backend(cli: &Cli, rig: ModelRig) -> Result<MetalModel> {
    let spec = get_spec(rig.name())?;
    let dir = cli.checkpoints.join(spec.hf_folder());
    eprintln!("[scene-preview] loading the {} model from {} ({})", rig.name(), dir.display(), cli.precision);
    MetalModel::from_checkpoints_with(rig.name(), &dir, 1, cli.precision.as_str())
        .with_context(|| format!("could not load the {} model from {}", rig.name(), dir.display()))
}

/// Build the SOMA character and its scene session (upstream `load_rig_assets` +
/// `SOMACharacter` + `_create_scene_session`).
fn build_soma_session(cli: &Cli, scene: &SceneDocument, scene_path: &Path) -> Result<(Session<MotionBricksSomaEngine<MetalModel>>, RigVisual)> {
    let rig_assets = SomaRig::load(&cli.clips)?;
    let skin = rig_assets.viewer_skin();
    let outlines = rig_assets.sole_outlines()?;
    let skeleton = rig_assets.skeleton.clone();
    let model_to_runtime = rig_assets.model_to_runtime;
    let clip_source = Rc::new(ClipSource {
        registry: RefCell::new(ClipRegistry::soma(skeleton.clone())),
        clips_dir: cli.clips.clone(),
        model_to_runtime,
    });

    let backend = build_backend(cli, ModelRig::Soma)?;
    let engine = MotionBricksSomaEngine::new(Some(backend), model_to_runtime)?;
    let player = PlayerAdapter::new_soma(engine, &skeleton, HISTORY_CAPACITY)?;

    // Locomotion modes: blend targets from the referenced Smart Primitive documents.
    let app_assets = app_assets_root();
    let reference_pose = skeleton.reference_local_transforms().to_vec();
    let mut pool: AssetPool<Transform> = AssetPool::new(model_to_runtime, clip_source.clone());
    let mut mode_runtimes = Vec::new();
    for mode in presets::soma::default_locomotion_modes() {
        let asset = pool.load(&app_assets.join(&mode.smart_primitive_path))?;
        mode_runtimes.push(load_locomotion_mode(&mode, &asset, &reference_pose)?);
    }

    // Idle clips, loop-corrected, as animation clips.
    let load_idle_clips = |paths: &[&str]| -> Result<Vec<AnimationClip<AnimationBuffer>>> {
        paths
            .iter()
            .map(|path| {
                let clip = clip_source.clip(Path::new(path), true)?;
                let buffer = clip.as_soma().cloned().with_context(|| format!("expected a SOMA (.bvh) clip: {path}"))?;
                make_animation_clip(buffer, model_to_runtime)
            })
            .collect()
    };
    let standing_idle_clips = load_idle_clips(&presets::soma::DEFAULT_IDLE_CLIPS)?;
    let crouching_idle_clips = load_idle_clips(&presets::soma::DEFAULT_CROUCH_IDLE_CLIPS)?;

    let behavior_registry = Rc::new(BehaviorRegistry::new(clip_source.resolve(Path::new(presets::soma::SITTING_CLIP))));
    let instances = build_instances(scene, scene_path, &mut |path| pool.load(path), &behavior_registry)?;

    // The character's clip loader (Smart Primitive behavior clips, e.g. the sitting loop).
    let loader_source = clip_source.clone();
    let clip_loader: ClipLoader<AnimationBuffer> = Box::new(move |path: &Path, loop_: bool| {
        let clip = loader_source.clip(path, loop_)?;
        let buffer = clip.as_soma().cloned().with_context(|| format!("expected a SOMA (.bvh) clip: {}", path.display()))?;
        make_animation_clip(buffer, loader_source.model_to_runtime)
    });

    let ik_corrector = TargetPoseIkCorrector::new(&skeleton, rt::create_soma_target_pose_ik_rig(&skeleton)?)?;
    let character = Character::new(
        player,
        Box::new(SomaTargetPoseBlender::new(reference_pose.clone())),
        Some(ik_corrector),
        SOMA_ROOT_FORWARD,
        model_to_runtime,
        rt_transform(scene.character_start.transform),
        mode_runtimes,
        motion::LINEAR_LOCOMOTION_MODE,
        standing_idle_clips,
        crouching_idle_clips,
        presets::soma::interaction_probe(),
        &reference_pose,
        clip_loader,
    )?;
    let session = Session::new(character, scene, behavior_registry, instances);
    Ok((session, RigVisual::Soma { rig: Box::new(rig_assets), skin, outlines }))
}

/// Build the Unitree G1 character and its scene session (upstream `load_rig_assets` +
/// `UnitreeG1Character` + `_create_scene_session`).
fn build_g1_session(cli: &Cli, scene: &SceneDocument, scene_path: &Path) -> Result<(Session<MotionBricksUnitreeG1Engine<MetalModel>>, RigVisual)> {
    let rig_assets = G1Rig::load(&cli.assets)?;
    let link_meshes = (0..rig_assets.links.len()).map(|i| rig_assets.link_mesh(i)).collect::<Result<Vec<_>>>()?;
    let mut contact_bodies = [0usize; rt::FOOT_CONTACT_COUNT];
    for (slot, name) in contact_bodies.iter_mut().zip(rt::UNITREE_G1_FOOT_CONTACT_BODIES) {
        *slot = rig_assets
            .bodies
            .iter()
            .position(|b| b.name == name)
            .with_context(|| format!("the G1 MJCF has no foot-contact body {name:?}"))?;
    }
    let reference_pose = rig_assets.reference_q();
    let model_to_runtime = rig_assets.model_to_runtime;
    let clip_source = Rc::new(ClipSource {
        registry: RefCell::new(ClipRegistry::unitree_g1_contract()),
        clips_dir: cli.clips.clone(),
        model_to_runtime,
    });

    let backend = build_backend(cli, ModelRig::UnitreeG1)?;
    let engine = MotionBricksUnitreeG1Engine::new(Some(backend), model_to_runtime)?;
    let player = PlayerAdapter::new_g1(engine, &reference_pose, HISTORY_CAPACITY)?;

    let app_assets = app_assets_root();
    let mut pool: AssetPool<f32> = AssetPool::new(model_to_runtime, clip_source.clone());
    let mut mode_runtimes = Vec::new();
    for mode in presets::unitree_g1::default_locomotion_modes() {
        let asset = pool.load(&app_assets.join(&mode.smart_primitive_path))?;
        mode_runtimes.push(load_locomotion_mode(&mode, &asset, &reference_pose)?);
    }

    let load_idle_clips = |paths: &[&str]| -> Result<Vec<AnimationClip<DofAnimationBuffer>>> {
        paths
            .iter()
            .map(|path| {
                let clip = clip_source.clip(Path::new(path), true)?;
                let buffer = clip.as_g1().cloned().with_context(|| format!("expected a Unitree G1 (.csv) clip: {path}"))?;
                make_animation_clip(buffer, model_to_runtime)
            })
            .collect()
    };
    let standing_idle_clips = load_idle_clips(&presets::unitree_g1::DEFAULT_IDLE_CLIPS)?;
    let crouching_idle_clips = load_idle_clips(&presets::unitree_g1::DEFAULT_CROUCH_IDLE_CLIPS)?;

    let behavior_registry = Rc::new(BehaviorRegistry::new(clip_source.resolve(Path::new(presets::unitree_g1::SITTING_CLIP))));
    let instances = build_instances(scene, scene_path, &mut |path| pool.load(path), &behavior_registry)?;

    let loader_source = clip_source.clone();
    let clip_loader: ClipLoader<DofAnimationBuffer> = Box::new(move |path: &Path, loop_: bool| {
        let clip = loader_source.clip(path, loop_)?;
        let buffer = clip.as_g1().cloned().with_context(|| format!("expected a Unitree G1 (.csv) clip: {}", path.display()))?;
        make_animation_clip(buffer, loader_source.model_to_runtime)
    });

    let character = Character::new(
        player,
        Box::new(G1TargetPoseBlender),
        None,
        G1_ROOT_FORWARD,
        model_to_runtime,
        rt_transform(scene.character_start.transform),
        mode_runtimes,
        motion::LINEAR_LOCOMOTION_MODE,
        standing_idle_clips,
        crouching_idle_clips,
        presets::unitree_g1::interaction_probe(),
        &reference_pose,
        clip_loader,
    )?;
    let session = Session::new(character, scene, behavior_registry, instances);
    Ok((session, RigVisual::G1 { rig: rig_assets, link_meshes, contact_bodies }))
}

// --- The per-frame logic (upstream `ScenePreview.step`) ---------------------------------------------------

/// Keep `camera_facing` on the camera's ground direction (upstream `_update_camera`'s
/// `ground_direction(camera.get_front(), ...)`).
fn update_camera_facing<E: MotionBricksEngine>(session: &mut Session<E>, camera: &OrbitCamera) {
    let forward_view = (camera.target - camera.eye()).normalize_or(Vec3::Z);
    session.camera_facing = ground_direction(to_world_dir(forward_view), session.camera_facing).unwrap_or(session.camera_facing);
}

/// Move the camera's followed point smoothly with the character (upstream
/// `_follow_character`): a damped spring smooths the velocity, the point moves at that
/// velocity and settles on the character.
fn follow_camera<E: MotionBricksEngine>(session: &mut Session<E>, camera: &mut OrbitCamera, delta_time: f32) {
    let velocity = session.character.actual_velocity;
    for axis in 0..2 {
        let (v, rate) = critical_spring_damper_exact(session.camera_velocity[axis], session.camera_rate[axis], velocity[axis], CAMERA_VELOCITY_HALFLIFE, delta_time);
        session.camera_velocity[axis] = v;
        session.camera_rate[axis] = rate;
    }
    session.camera_target.x += session.camera_velocity[0] * delta_time;
    session.camera_target.y += session.camera_velocity[1] * delta_time;
    let pull = 1.0 - (-delta_time / CAMERA_PULL_SECONDS).exp();
    session.camera_target += (session.character.character_ground_position - session.camera_target) * pull;
    camera.target = to_view(session.camera_target + Vec3::new(0.0, 0.0, CAMERA_PIVOT_Z_OFFSET));
}

/// The index of the candidate with the smallest distance, or `None` (upstream's `min`
/// with a default).
fn pick_nearest(candidates: impl Iterator<Item = (usize, f32)>) -> Option<usize> {
    candidates.min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(index, _)| index)
}

/// Test the probe box against every interaction box and keep the nearest overlapping
/// instance (upstream `_update_smart_primitive_interaction`).
fn update_interactions<E>(session: &mut Session<E>)
where
    E: MotionBricksEngine,
    E::Elem: PoseElem,
{
    let probe_box = session.character.probe_box;
    let ground = session.character.character_ground_position;
    let mut candidates = Vec::new();
    for (index, instance) in session.instances.iter_mut().enumerate() {
        let boxes = instance.runtime.get_world_interaction_boxes();
        let overlaps = compute_interaction_box_overlaps(&boxes, &probe_box);
        instance.interactable = overlaps.iter().any(|&overlap| overlap);
        if instance.interactable {
            let offset = motion::to_glam(instance.world_transform.p) - ground;
            candidates.push((index, offset.dot(offset)));
        }
    }
    session.nearest_interactable = pick_nearest(candidates.into_iter());
}

/// Handle the interaction button, upstream's `_dispatch_interaction_input`: a press
/// starts an interaction with the nearest Smart Primitive in range, or goes to the
/// running one. Auto Interact only starts interactions, once per approach.
fn dispatch_interaction<E>(session: &mut Session<E>, pressed: bool) -> Result<()>
where
    E: MotionBricksEngine,
    E::Elem: PoseElem + 'static,
{
    if session.active_interaction.is_some() && !session.character.state_machine.is_interacting() {
        let finished = session.active_interaction.take().expect("checked above");
        session.auto_interaction_blocked.insert(finished);
    }
    for instance in &session.instances {
        if !instance.interactable {
            session.auto_interaction_blocked.remove(&instance.id);
        }
    }

    if session.character.state_machine.is_interacting() {
        if pressed {
            session.character.interact()?;
        }
        return Ok(());
    }
    let Some(index) = session.nearest_interactable else {
        return Ok(());
    };
    let instance = &session.instances[index];
    if !(pressed || (session.auto_interaction && !session.auto_interaction_blocked.contains(&instance.id))) {
        return Ok(());
    }
    let (id, behavior_name, asset, world_transform) = (instance.id, instance.behavior_name.clone(), instance.asset.clone(), instance.world_transform);
    let mut behavior = session.behavior_registry.create(&behavior_name, asset)?;
    behavior.primitive_mut().set_node_world_transform(0, world_transform);
    session.character.start_interaction(behavior)?;
    session.active_interaction = Some(id);
    Ok(())
}

/// The live blend space pose, for its debug view (upstream `evaluate_current_target_poses`).
fn evaluate_blendspace<E>(session: &mut Session<E>) -> Result<()>
where
    E: MotionBricksEngine,
    E::Elem: PoseElem,
{
    let locomotion = session.character.state_machine.active_locomotion_state();
    locomotion.evaluate_current_target_poses(&session.character, &mut session.blended_poses, &mut session.blendspace_poses)
}

// --- Rendering (upstream `ScenePreview.render`) -------------------------------------------------------------

fn draw_session<E>(session: &mut Session<E>, visual: &RigVisual, frame: &mut Frame, view: &ViewFlags)
where
    E: MotionBricksEngine,
    E::Elem: PoseElem + PoseDraw,
    E::Token: TokenPoses<E::Elem>,
{
    frame.log_ground();

    // The character mesh, then its predicted foot contacts.
    let pose = E::Elem::pose_data(&session.character.current_pose);
    if let Err(error) = visual.draw_pose(frame, &pose, visual.character_color()) {
        session.error = Some(format!("{error:#}"));
    }
    if view.show_foot_contacts {
        let _ = visual.draw_foot_contacts(frame, &pose, session.character.current_foot_contacts, view.foot_contact_threshold);
    }

    // The scene's Solid Cubes.
    for cube in &session.cubes {
        frame.log_mesh(&cube.mesh, cube.transform, cube.color);
    }

    draw_instances(session, visual, frame, view);

    // Context and target skeletons of the last generation.
    if view.show_context_poses {
        let context = session.character.player().player().context();
        for pose in &context.token_poses()[..context.num_poses()] {
            let _ = visual.draw_skeleton(frame, &E::Elem::pose_data(pose), CONTEXT_COLOR);
        }
    }
    if view.show_target_poses {
        let target = session.character.player().player().target();
        for pose in &target.token_poses()[..target.num_poses()] {
            let _ = visual.draw_skeleton(frame, &E::Elem::pose_data(pose), TARGET_COLOR);
        }
    }

    // The live blend space pose, only during locomotion.
    if view.show_blendspace_targets && session.character.state_machine.is_locomoting() {
        let _ = visual.draw_skeleton(frame, &E::Elem::pose_data(&session.blendspace_poses[0]), BLENDSPACE_DEBUG_COLOR);
    }

    // The character's probe box, only during locomotion.
    if view.show_probe_box && session.character.state_machine.is_locomoting() {
        let probe_box = session.character.probe_box;
        log_oriented_box(frame, &probe_box, INTERACTION_BOX_COLOR);
    }

    // Velocity arrows.
    let origin = to_view(session.character.character_ground_position) + Vec3::new(0.0, VELOCITY_ARROW_HEIGHT, 0.0);
    let arrows = [
        (view.show_actual_velocity, session.character.actual_velocity, CHARACTER_VELOCITY_ARROW_COLOR),
        (view.show_input_velocity, session.character.movement.input_velocity(), INPUT_VELOCITY_ARROW_COLOR),
        (view.show_desired_velocity, session.character.movement.desired_velocity, DESIRED_VELOCITY_ARROW_COLOR),
    ];
    for (visible, velocity, color) in arrows {
        if visible {
            frame.log_arrow(origin, origin + to_view(velocity), color4(color));
        }
    }
}

/// Per-Smart-Primitive target-pose skeletons and wire interaction boxes, honoring each
/// instance's visibility flags (upstream `_RuntimeSceneSmartPrimitive.draw`).
fn draw_instances<E>(session: &mut Session<E>, visual: &RigVisual, frame: &mut Frame, view: &ViewFlags)
where
    E: MotionBricksEngine,
    E::Elem: PoseElem + PoseDraw,
{
    for instance in &mut session.instances {
        if view.show_scene_target_poses && instance.show_target_poses {
            for node in 0..instance.runtime.node_count() {
                if !instance.runtime.has_target_pose(node) {
                    continue;
                }
                // Upstream draws only each target's first pose, so the targets stay readable.
                if let Ok((poses, _)) = instance.runtime.get_world_target_poses(node) {
                    let _ = visual.draw_skeleton(frame, &E::Elem::pose_data(&poses[0]), color4(instance.target_pose_color));
                }
            }
        }
        if view.show_scene_interaction_boxes && instance.show_interaction_boxes {
            let color = if instance.interactable { ACTIVE_SMART_PRIMITIVE_BOX_COLOR } else { INACTIVE_SMART_PRIMITIVE_BOX_COLOR };
            for interaction_box in instance.runtime.get_world_interaction_boxes() {
                log_oriented_box(frame, &interaction_box, color);
            }
        }
    }
}

// --- The Scene Preview window (upstream `gui`) ---------------------------------------------------------------

/// The window's visibility toggles and their upstream defaults.
struct ViewFlags {
    show_target_poses: bool,
    show_context_poses: bool,
    show_blendspace_targets: bool,
    show_probe_box: bool,
    show_actual_velocity: bool,
    show_input_velocity: bool,
    show_desired_velocity: bool,
    show_foot_contacts: bool,
    foot_contact_threshold: f32,
    show_scene_target_poses: bool,
    show_scene_interaction_boxes: bool,
}

impl Default for ViewFlags {
    fn default() -> Self {
        ViewFlags {
            show_target_poses: false,
            show_context_poses: false,
            show_blendspace_targets: false,
            show_probe_box: false,
            show_actual_velocity: true,
            show_input_velocity: true,
            show_desired_velocity: true,
            show_foot_contacts: false,
            foot_contact_threshold: DEFAULT_CONTACT_THRESHOLD,
            show_scene_target_poses: false,
            show_scene_interaction_boxes: true,
        }
    }
}

fn table_flags() -> TableFlags {
    TableFlags::SIZING_FIXED_FIT | TableFlags::NO_SAVED_SETTINGS | TableFlags::NO_PAD_OUTER_X
}

/// A two-column label/value table (upstream's Status and Model Stats tables).
fn key_value_table(ui: &Ui, id: &str, values: &[(&str, String)]) {
    if let Some(_table) = ui.begin_table_with_flags(id, 2, table_flags()) {
        for (label, value) in values {
            ui.table_next_row();
            ui.table_set_column_index(0);
            ui.text(label);
            ui.table_set_column_index(1);
            ui.text(value);
        }
    }
}

fn gui<E>(session: &mut Session<E>, ui: &mut Ui, view: &mut ViewFlags, precision_label: &str)
where
    E: MotionBricksEngine,
    E::Elem: PoseElem,
{
    let viewport = ui.io().display_size;
    let scale = ui.current_font_size() / 13.0;
    let margin = WINDOW_MARGIN * scale;
    ui.window("Scene Preview")
        .position([margin, viewport[1] - margin], Condition::FirstUseEver)
        .position_pivot([0.0, 1.0])
        .size([PANEL_WIDTH * scale, PANEL_HEIGHT * scale], Condition::FirstUseEver)
        .bg_alpha(PANEL_ALPHA)
        .build(|| {
            if ui.collapsing_header("Status", TreeNodeFlags::DEFAULT_OPEN) {
                let character = &session.character;
                let mode = character.state_machine.active_locomotion_state().active_mode();
                let mode_label = if mode.label.is_empty() { mode.name.clone() } else { mode.label.clone() };
                let position = character.character_ground_position;
                key_value_table(
                    ui,
                    "##StatusValues",
                    &[
                        ("State:", character.state_machine.state_name().to_string()),
                        ("Locomotion:", mode_label),
                        ("Time To Transition:", format!("{:.2} s", character.time_to_transition())),
                        ("Position:", format!("({:.2}, {:.2}, {:.2}) m", position.x, position.y, position.z)),
                        ("Speed:", format!("{:.2} m/s", character.actual_velocity.length())),
                    ],
                );
            }

            ui.separator();
            if ui.collapsing_header("Scene", TreeNodeFlags::DEFAULT_OPEN) {
                ui.text(format!("Scene: {}", session.scene_name));
                ui.text(format!("{} Smart Primitives, {} Solid Cubes", session.instances.len(), session.cubes.len()));
                ui.checkbox("Auto Interact", &mut session.auto_interaction);
                if session.character.state_machine.is_interacting() {
                    ui.text_colored(ACTIVE_SMART_PRIMITIVE_BOX_COLOR, "Interaction active — press I to interact");
                } else if let Some(index) = session.nearest_interactable {
                    ui.text_colored(ACTIVE_SMART_PRIMITIVE_BOX_COLOR, format!("Press I to interact: {}", session.instances[index].name));
                }
                if let Some(error) = session.error.clone() {
                    {
                        let _color = ui.push_style_color(StyleColor::Text, [1.0, 0.35, 0.35, 1.0]);
                        ui.text_wrapped(format!("Error: {error}"));
                    }
                    if ui.small_button("Dismiss##error") {
                        session.error = None;
                    }
                }
            }

            ui.separator();
            if ui.collapsing_header("Visibility", TreeNodeFlags::DEFAULT_OPEN) {
                ui.text("Character");
                ui.checkbox("Show Target Poses##Character", &mut view.show_target_poses);
                ui.checkbox("Show Context Poses", &mut view.show_context_poses);
                ui.checkbox("Show Blend Space Poses", &mut view.show_blendspace_targets);
                ui.checkbox("Show Probe Box", &mut view.show_probe_box);
                ui.checkbox("Show Actual Velocity", &mut view.show_actual_velocity);
                ui.checkbox("Show Input Velocity", &mut view.show_input_velocity);
                ui.checkbox("Show Desired Velocity", &mut view.show_desired_velocity);
                ui.checkbox("Show Foot Contacts", &mut view.show_foot_contacts);
                ui.same_line();
                ui.set_next_item_width(80.0 * scale);
                ui.disabled(!view.show_foot_contacts, || {
                    ui.slider_config("##FootContactThreshold", 0.0f32, 1.0f32)
                        .display_format("%.2f")
                        .flags(SliderFlags::ALWAYS_CLAMP)
                        .build(&mut view.foot_contact_threshold);
                });

                ui.spacing();
                ui.text("Smart Primitives");
                ui.checkbox("Show Target Poses##SceneObjects", &mut view.show_scene_target_poses);
                ui.checkbox("Show Interaction Boxes", &mut view.show_scene_interaction_boxes);
            }

            ui.separator();
            if ui.collapsing_header("Model Stats", TreeNodeFlags::DEFAULT_OPEN) {
                let timer = session.character.player().player().engine().timer();
                let stages = [
                    ("Encode:", timer.encode),
                    ("Inference:", timer.inference),
                    ("Decode:", timer.decode),
                    ("Post-Process:", timer.post_process),
                ];
                let total: f64 = stages.iter().map(|(_, value)| value).sum();
                let mut values = vec![("Precision:", precision_label.to_string())];
                values.extend(stages.iter().map(|(label, value)| (*label, format!("{value:.3} ms"))));
                values.push(("Total:", format!("{total:.3} ms")));
                key_value_table(ui, "##RuntimeTimings", &values);
            }
        });
}

// --- The app and the frame loop (upstream `ScenePreview`) ----------------------------------------------------

/// The one loaded scene session, over its rig's engine and visuals.
enum RigSession {
    Soma(Session<MotionBricksSomaEngine<MetalModel>>, RigVisual),
    G1(Session<MotionBricksUnitreeG1Engine<MetalModel>>, RigVisual),
}

struct App {
    session: RigSession,
    view: ViewFlags,
    precision_label: &'static str,
    reset_camera: bool,
}

impl App {
    fn on_frame(&mut self, frame: &mut Frame, ui: &mut Ui) {
        let keyboard_free = !ui.io().want_capture_keyboard;
        match &mut self.session {
            RigSession::Soma(session, visual) => {
                if self.reset_camera {
                    *frame.camera() = session.initial_camera;
                }
                frame_impl(session, visual, frame, ui, &mut self.view, self.precision_label, keyboard_free);
            }
            RigSession::G1(session, visual) => {
                if self.reset_camera {
                    *frame.camera() = session.initial_camera;
                }
                frame_impl(session, visual, frame, ui, &mut self.view, self.precision_label, keyboard_free);
            }
        }
        self.reset_camera = false;
    }
}

/// One frame, in upstream `step()`'s order: input, camera-facing, the locomotion command,
/// the character step, the camera follow, then the interaction probe and dispatch.
fn frame_impl<E>(
    session: &mut Session<E>,
    visual: &RigVisual,
    frame: &mut Frame,
    ui: &mut Ui,
    view: &mut ViewFlags,
    precision_label: &str,
    keyboard_free: bool,
) where
    E: MotionBricksEngine,
    E::Elem: PoseElem + PoseDraw + 'static,
    E::Token: TokenPoses<E::Elem>,
{
    let delta_time = frame.dt().clamp(MIN_FRAME_SECONDS, MAX_FRAME_SECONDS);

    // 1. Input (upstream `InputManager.poll`'s keyboard half).
    let (movement_axes, use_max_speed, interact_pressed, mode_o, mode_p, mode_style) = if keyboard_free {
        let input = frame.input();
        (
            poll_movement(input),
            input.key_held(KeyCode::Space),
            input.key_pressed(KeyCode::KeyI),
            input.key_pressed(KeyCode::KeyO),
            input.key_pressed(KeyCode::KeyP),
            input.key_pressed(KeyCode::BracketLeft),
        )
    } else {
        ((0.0, 0.0), false, false, false, false, false)
    };

    // 2. Camera-facing update (the viewer's mouse orbit already ran this frame).
    update_camera_facing(session, frame.camera());

    // 3. The camera-relative locomotion command. During an interaction the command still
    // flows through `MovementControl`, as upstream does — the interaction state ignores it,
    // and held input resumes walking when the interaction ends.
    let current_mode = session.character.state_machine.active_locomotion_state().active_mode().name.clone();
    let command = LocomotionCommand {
        facing_direction: session.camera_facing,
        movement: camera_relative_movement(session.camera_facing, movement_axes),
        use_max_speed,
        requested_mode: session.mode_input.poll(mode_o, mode_p, mode_style, &current_mode),
    };

    // 4. Advance the character.
    if let Err(error) = session.character.step(delta_time, &command) {
        session.error = Some(format!("{error:#}"));
    }

    // 5. Camera follow.
    follow_camera(session, frame.camera(), delta_time);

    // 6. Probe box vs. every interaction box; 7. start or pass on interaction input.
    update_interactions(session);
    if let Err(error) = dispatch_interaction(session, interact_pressed) {
        session.error = Some(format!("{error:#}"));
    }

    // The blend-space debug poses (only while locomoting).
    if view.show_blendspace_targets && session.character.state_machine.is_locomoting() {
        if let Err(error) = evaluate_blendspace(session) {
            session.error = Some(format!("{error:#}"));
        }
    }

    draw_session(session, visual, frame, view);
    gui(session, ui, view, precision_label);
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let scene_path = std::fs::canonicalize(&cli.scene).unwrap_or_else(|_| cli.scene.clone());
    // Parenting only helps editing. Runtime items are placed by their world transforms.
    let scene = SceneDocument::load(&scene_path)?.flatten();
    scene.check_reference_rigs(&scene_path)?;
    let scene_rig = match scene.rig {
        motionbricks_app::documents::ModelRig::Soma => ModelRig::Soma,
        motionbricks_app::documents::ModelRig::UnitreeG1 => ModelRig::UnitreeG1,
    };
    let rig = cli.rig.unwrap_or(scene_rig);
    if rig != scene_rig {
        bail!("the scene {:?} is for the {} rig, not {}", scene.name, scene_rig.name(), rig.name());
    }
    eprintln!("[scene-preview] scene {} ({}, {} Smart Primitives, {} Solid Cubes)", scene_path.display(), rig.name(), scene.smart_primitives.len(), scene.solid_cubes.len());
    let session = match rig {
        ModelRig::Soma => {
            let (session, visual) = build_soma_session(&cli, &scene, &scene_path)?;
            RigSession::Soma(session, visual)
        }
        ModelRig::UnitreeG1 => {
            let (session, visual) = build_g1_session(&cli, &scene, &scene_path)?;
            RigSession::G1(session, visual)
        }
    };
    let mut app = App { session, view: ViewFlags::default(), precision_label: cli.precision_label(), reset_camera: true };
    eprintln!("[scene-preview] engine ready; opening the window");
    let viewer = Viewer::new(ViewerConfig { title: "MotionBricks - Scene Preview".to_string(), width: 1280, height: 720 })?;
    viewer.run(move |frame, ui| app.on_frame(frame, ui));
}

// --- Tests (pure logic: no GPU, no engine) --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_clip_keeps_existing_files() {
        let dir = std::env::temp_dir().join("scene_preview_test_resolve_clip");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("idle.bvh");
        std::fs::write(&file, b"test").unwrap();
        assert_eq!(resolve_clip(&file, Path::new("/clips")), file);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_clip_rebases_missing_files_under_the_clip_root() {
        // The document-relative path does not exist: the tail after "motions" is taken.
        let missing = Path::new("/repo/app-rs/assets/motions/soma/locomotion/idle/idle_loop_001__A268.bvh");
        assert_eq!(
            resolve_clip(missing, Path::new("/clips")),
            PathBuf::from("/clips/soma/locomotion/idle/idle_loop_001__A268.bvh")
        );
        // A relative preset path works the same way.
        assert_eq!(
            resolve_clip(Path::new("motions/soma/sitting/sit_on_chair_loop.bvh"), Path::new("/clips")),
            PathBuf::from("/clips/soma/sitting/sit_on_chair_loop.bvh")
        );
        // Without a "motions" component, the path is returned unchanged.
        assert_eq!(resolve_clip(Path::new("elsewhere/clip.bvh"), Path::new("/clips")), PathBuf::from("elsewhere/clip.bvh"));
    }

    #[test]
    fn dead_zone_normalizes_diagonals() {
        assert_eq!(apply_radial_dead_zone(0.0, 0.0, 0.0), (0.0, 0.0));
        assert_eq!(apply_radial_dead_zone(0.1, 0.0, 0.2), (0.0, 0.0), "inside the dead zone");
        let (x, y) = apply_radial_dead_zone(1.0, 1.0, 0.0);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((x - expected).abs() < 1e-6 && (y - expected).abs() < 1e-6, "diagonals are not faster: ({x}, {y})");
        let (x, _) = apply_radial_dead_zone(0.6, 0.0, 0.2);
        assert!((x - 0.5).abs() < 1e-6, "the remaining range scales from zero: {x}");
    }

    #[test]
    fn camera_relative_movement_maps_right_and_forward() {
        let facing = Vec3::X;
        // Forward along the facing; camera right of +X is -Y (upstream's (y, -x) rule).
        assert_eq!(camera_relative_movement(facing, (0.0, 1.0)), Some(Vec3::X));
        assert_eq!(camera_relative_movement(facing, (1.0, 0.0)), Some(Vec3::new(0.0, -1.0, 0.0)));
        assert_eq!(camera_relative_movement(facing, (0.0, 0.0)), None, "a stick at rest is no movement");
        // Facing +Y: forward is +Y and right is +X.
        let moved = camera_relative_movement(Vec3::Y, (1.0, 1.0)).unwrap();
        assert!(moved.abs_diff_eq(Vec3::new(1.0, 1.0, 0.0), 1e-6), "got {moved:?}");
    }

    #[test]
    fn mode_input_toggles_and_cycles_styles() {
        let styles = vec!["skipping_1d".to_string(), "vogue_1d".to_string()];
        let mut input = LocomotionModeInput::new(&styles);
        // O toggles Forward <-> Strafing.
        assert_eq!(input.poll(true, false, false, motion::LINEAR_LOCOMOTION_MODE).as_deref(), Some(motion::DIRECTIONAL_LOCOMOTION_MODE));
        assert_eq!(input.poll(true, false, false, motion::DIRECTIONAL_LOCOMOTION_MODE).as_deref(), Some(motion::LINEAR_LOCOMOTION_MODE));
        // P toggles Crouching <-> Strafing.
        assert_eq!(input.poll(false, true, false, motion::DIRECTIONAL_LOCOMOTION_MODE).as_deref(), Some(motion::CROUCH_LOCOMOTION_MODE));
        assert_eq!(input.poll(false, true, false, motion::CROUCH_LOCOMOTION_MODE).as_deref(), Some(motion::DIRECTIONAL_LOCOMOTION_MODE));
        // [ from another mode returns to the last style (initially the first).
        assert_eq!(input.poll(false, false, true, motion::LINEAR_LOCOMOTION_MODE).as_deref(), Some("skipping_1d"));
        // [ in a style cycles to the next one, wrapping around.
        assert_eq!(input.poll(false, false, true, "skipping_1d").as_deref(), Some("vogue_1d"));
        assert_eq!(input.poll(false, false, true, "vogue_1d").as_deref(), Some("skipping_1d"));
        // The last style is remembered across other modes.
        input.poll(false, false, true, "vogue_1d"); // now in skipping_1d
        assert_eq!(input.poll(false, false, true, motion::LINEAR_LOCOMOTION_MODE).as_deref(), Some("skipping_1d"));
        // O from a style returns to Forward.
        assert_eq!(input.poll(true, false, false, "skipping_1d").as_deref(), Some(motion::LINEAR_LOCOMOTION_MODE));
        // The toggles also cross modes: O from Crouching returns to Forward,
        // and P from Forward crouches directly.
        assert_eq!(input.poll(true, false, false, motion::CROUCH_LOCOMOTION_MODE).as_deref(), Some(motion::LINEAR_LOCOMOTION_MODE));
        assert_eq!(input.poll(false, true, false, motion::LINEAR_LOCOMOTION_MODE).as_deref(), Some(motion::CROUCH_LOCOMOTION_MODE));
        // No press, no request.
        assert_eq!(input.poll(false, false, false, motion::LINEAR_LOCOMOTION_MODE), None);
    }

    #[test]
    fn pick_nearest_selects_the_closest_candidate() {
        assert_eq!(pick_nearest([].into_iter()), None);
        assert_eq!(pick_nearest([(2, 4.0), (0, 1.0), (1, 9.0)].into_iter()), Some(0));
        assert_eq!(pick_nearest([(5, 0.25)].into_iter()), Some(5));
    }

    #[test]
    fn to_view_maps_z_up_to_y_up() {
        assert_eq!(to_view(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 3.0, -2.0));
        assert_eq!(to_view(Vec3::Z), Vec3::Y, "world up maps to viewer up");
        assert_eq!(to_world_dir(Vec3::Y), Vec3::Z, "viewer up maps to world up");
        assert_eq!(to_world_dir(to_view(Vec3::new(4.0, 5.0, 6.0))), Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn follow_camera_sits_behind_the_character() {
        let camera = follow_camera_at(Vec3::new(2.0, 3.0, 0.0), Vec3::X);
        // The pivot is the ground position plus the Z offset, in viewer space.
        assert!(camera.target.abs_diff_eq(Vec3::new(2.0, CAMERA_PIVOT_Z_OFFSET, -3.0), 1e-6));
        assert!((camera.distance - CAMERA_DISTANCE).abs() < 1e-5);
        // Facing +X: the eye sits behind (-X) and above the pivot.
        let eye = camera.eye();
        assert!(eye.x < camera.target.x && eye.y > camera.target.y, "eye: {eye:?}");
    }

    #[test]
    fn spring_converges_to_the_goal() {
        let (mut value, mut rate) = (0.0, 0.0);
        for _ in 0..600 {
            (value, rate) = critical_spring_damper_exact(value, rate, 2.0, CAMERA_VELOCITY_HALFLIFE, 1.0 / 60.0);
        }
        assert!((value - 2.0).abs() < 1e-3, "value: {value}");
        assert!(rate.abs() < 1e-3, "rate: {rate}");
    }
}
