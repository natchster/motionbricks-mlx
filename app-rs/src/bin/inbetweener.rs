// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Inbetweener (port of upstream `app/inbetweener_demo.py`): choose poses from clips,
//! place them in the view, arrange them in the pose list, and press Generate to play the
//! motion through them. Generation chains autoregressive transitions through the Rust
//! Runtime API (`motionbricks_rs`): each segment feeds the last four played poses back as
//! the next context, exactly like upstream's `reset()`, `generate_next_segment()` and
//! `step()`.
//!
//! The engine is the Metal runtime ([`MetalModel`]) behind the Runtime API engines — never
//! Python, TensorRT, or ONNX Runtime. The same flow serves both rigs; only the pose layouts
//! differ (SOMA: one local transform per skeleton joint; Unitree G1: 36 joint coordinates).
//!
//! Deviations from upstream (Newton/tkinter/Warp):
//! - The runtime world is Z-up; the viewer is Y-up. [`ZUP_TO_YUP`] converts at draw time.
//! - The viewer gizmo only translates; rotation is edited in the Transform fields.
//! - The tkinter clip dialog is an in-imgui path-input modal.
//! - Skeleton(s) and foot contacts are drawn with plain line segments instead of the
//!   instanced renderers (the sole-outline loops are the rig's own).
//!
//! Run with: `cargo run --release --bin inbetweener -- [--rig soma|unitree_g1] [--precision fp32|bf16]`

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use glam::{Mat4, Vec3};
use imgui::{Condition, Drag, DragDropFlags, DragRange, Key, MouseButton, SliderFlags, StyleColor, TreeNodeFlags, Ui};
use lexopt::prelude::*;
use motionbricks_app::editor::widgets::{clamp_token_range, sample_frame_at};
use motionbricks_app::rig::clips::{Clip, ClipRegistry};
use motionbricks_app::rig::g1::G1Rig;
use motionbricks_app::rig::soma::{self, SoleOutline, SomaRig};
use motionbricks_app::viewer::{Color, Frame, Mesh, Ray, Skin, Viewer, ViewerConfig};
use motionbricks_rs::config::get_spec;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::runtime as rt;
use rt::{
    AnimationBuffer, DofAnimationBuffer, DofToken, FootContacts, MAX_TRANSITION_EARLY_BY_FRAMES, MODEL_MAX_TOKEN_COUNT, MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ModelRig, MotionBricksEngine, MotionBricksInferenceConfig, MotionBricksSomaEngine,
    MotionBricksUnitreeG1Engine, PoseToken, Transform,
};

// --- Constants (upstream's module-level values) ---------------------------------------------------

const START_COLOR: [f32; 3] = [0.2, 0.9, 0.4];
const TARGET_COLOR: [f32; 3] = [0.25, 0.65, 1.0];
const SELECTED_COLOR: [f32; 3] = [1.0, 0.5, 0.15];
const ORDER_COLOR: [f32; 3] = [1.0, 0.8, 0.25];
/// Upstream `CHARACTER_COLORS`: SOMA playback draws gray; the G1 draws its own link colors.
const PLAYBACK_SOMA_COLOR: [f32; 4] = [0.7, 0.7, 0.7, 1.0];
const CONTACT_COLOR: [f32; 4] = [0.3, 1.0, 0.4, 1.0];
const GROUND_MARKER_HEIGHT: f32 = 0.05;
const RING_SEGMENTS: usize = 48;
const RING_RADIUS: f32 = 0.3;
/// Upstream `picking._LINE_PICK_RADIUS`: a ground ring picks like a capsule of this radius.
const LINE_PICK_RADIUS: f32 = 0.025;
/// Playback never advances more than this in one frame (upstream `_MAX_FRAME_SECONDS`).
const MAX_FRAME_SECONDS: f32 = 0.1;
/// Upstream `FootContactRenderer.DEFAULT_THRESHOLD`.
const DEFAULT_CONTACT_THRESHOLD: f32 = 0.5;
/// Upstream `DEFAULT_TOKEN_COUNTS`: each target starts allowing 8 to 14 tokens.
const DEFAULT_MIN_TOKENS: i32 = 8;
const DEFAULT_MAX_TOKENS: i32 = 14;
const DEFAULT_CLIP: [&str; 2] = [
    "soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh",
    "unitree_g1/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.csv",
];
const RIG_LABELS: [&str; 2] = ["SOMA", "Unitree G1"];
const POSE_PAYLOAD: &str = "INBETWEENER_POSE";
const POSES_TO_USE_TOOLTIP: &str =
    "How many of the target's four poses inform the model. One pose makes the motion stop at the target. More poses add velocity, acceleration, and momentum.";

/// The runtime world is Z-up; the viewer is Y-up. This rotation (-90 degrees about X) maps
/// world space to viewer space: (x, y, z) -> (x, z, -y).
const ZUP_TO_YUP: Transform = Transform::new(rt::Vec3::ZERO, [-std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2]);

fn zup_to_yup_mat() -> Mat4 {
    soma::transform_to_mat4(ZUP_TO_YUP)
}

/// World-space (Z-up) point -> viewer-space (Y-up) point.
fn to_view(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[2], -p[1])
}

fn color4(c: [f32; 3]) -> Color {
    [c[0], c[1], c[2], 1.0]
}

// --- Command line -----------------------------------------------------------------------------------

struct Cli {
    rig: ModelRig,
    /// A `motionbricks_rs::config::Precision` name ("float32", "float16", "bfloat16").
    precision: String,
    /// Root holding the per-rig checkpoint folders (default `checkpoints/MotionBricks`).
    checkpoints: PathBuf,
    /// Clip root (default `checkpoints/motions`).
    clips: PathBuf,
    /// Upstream checkout root, for the G1 MJCF and STL meshes (default `sandbox/upstream`).
    assets: PathBuf,
}

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map_or(manifest.clone(), PathBuf::from)
}

fn usage() -> &'static str {
    "MotionBricks Inbetweener (Rust, Metal runtime)\n\
     Usage: inbetweener [OPTIONS]\n\
     \x20 --rig {soma|unitree_g1}   character to start with (default soma)\n\
     \x20 --precision {fp32|bf16}  Metal runtime dtype (default fp32; fp16 also accepted)\n\
     \x20 --checkpoints DIR         checkpoint root holding soma/ and g1/ (default checkpoints/MotionBricks)\n\
     \x20 --clips DIR               clip root (default checkpoints/motions)\n\
     \x20 --assets DIR              upstream checkout root, for G1 meshes (default sandbox/upstream)\n\
     \x20 --help                    show this text"
}

fn parse_cli() -> Result<Cli> {
    let root = repo_root();
    let mut cli = Cli {
        rig: ModelRig::Soma,
        precision: "float32".to_string(),
        checkpoints: root.join("checkpoints/MotionBricks"),
        clips: root.join("checkpoints/motions"),
        assets: root.join("sandbox/upstream"),
    };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            lexopt::Arg::Long("rig") => cli.rig = ModelRig::parse(&parser.value()?.string()?)?,
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

// --- Poses ------------------------------------------------------------------------------------------

/// One pose of either rig: SOMA local joint transforms, or G1 joint coordinates.
#[derive(Clone, Debug)]
enum PoseData {
    Soma(Vec<Transform>),
    G1(Vec<f32>),
}

impl PoseData {
    /// Both layouts start with the root position (world space, Z-up).
    fn root_position(&self) -> [f32; 3] {
        match self {
            PoseData::Soma(pose) => pose[0].p.0,
            PoseData::G1(q) => [q[0], q[1], q[2]],
        }
    }
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

/// The four sample times of a pose window (upstream `sample_pose_window`): `anchor` is the
/// pose that sits at `sample_time` — 3 for Start (the window ends there), 0 for a target.
fn window_times(sample_time: f32, anchor: usize) -> [f32; MODEL_POSES_PER_TOKEN] {
    std::array::from_fn(|frame| sample_time + (frame as f32 - anchor as f32) / MODEL_SAMPLE_RATE)
}

/// Sample four poses at 30 Hz from a clip and place their root in world space (upstream
/// `sample_pose_window`). The anchor's root moves to `placement` and keeps its height; the
/// other samples move with it. Only the root needs the coordinate conversion: SOMA child
/// joints stay relative to their parents and G1 joint angles do not change.
fn sample_pose_window(clip: &Clip, sample_time: f32, placement: Transform, model_to_runtime: Transform, anchor: usize) -> Result<Vec<PoseData>> {
    let anchor_root = match clip {
        Clip::Soma(b) => b.sample(sample_time, model_to_runtime)?[0].p,
        Clip::G1(b) => rt::read_root(&b.sample(sample_time, model_to_runtime)?).p,
    };
    let ground_offset = Transform::new(rt::Vec3([-anchor_root.x(), -anchor_root.y(), 0.0]), rt::math::QUAT_IDENTITY);
    let root_offset = placement * (ground_offset * model_to_runtime);
    window_times(sample_time, anchor)
        .iter()
        .map(|&t| {
            Ok(match clip {
                Clip::Soma(b) => PoseData::Soma(b.sample(t, root_offset)?),
                Clip::G1(b) => PoseData::G1(b.sample(t, root_offset)?),
            })
        })
        .collect()
}

// --- Pose entries (upstream `PoseEntry`) ------------------------------------------------------------

/// Start or a target, sampled from a clip and placed in the view. Start comes first.
struct PoseEntry {
    id: usize,
    name: String,
    /// Clip file this pose samples (loaded through the rig's [`ClipRegistry`]).
    path: PathBuf,
    sample_time: f32,
    /// World placement of the pose's root (Z-up world space).
    placement: Transform,
    min_tokens: i32,
    max_tokens: i32,
    color: [f32; 3],
    /// Angles last shown in the Rotation fields; they keep the shown angles from jumping.
    euler_degrees: Option<[f32; 3]>,
    translation_locks: [bool; 3],
    rotation_locks: [bool; 3],
    show_frames: bool,
    num_poses_to_use: usize,
    transition_early_by_frames: usize,
    show_mesh: bool,
    show_skeletons: bool,
    show_velocity: bool,
    // Derived by `refresh_poses()` from the clip, the sample time and the placement.
    sampled_poses: Vec<PoseData>,
    /// The window pose at the sample time: 3 for Start, 0 for a target.
    anchor: usize,
    root_position: [f32; 3],
    velocity: [f32; 3],
}

impl PoseEntry {
    fn new(id: usize, name: String, path: PathBuf, sample_time: f32, placement: Transform, color: [f32; 3]) -> PoseEntry {
        PoseEntry {
            id,
            name,
            path,
            sample_time,
            placement,
            min_tokens: DEFAULT_MIN_TOKENS,
            max_tokens: DEFAULT_MAX_TOKENS,
            color,
            euler_degrees: None,
            translation_locks: [false, false, true],
            rotation_locks: [true, true, false],
            show_frames: false,
            num_poses_to_use: MODEL_POSES_PER_TOKEN,
            transition_early_by_frames: 0,
            show_mesh: true,
            show_skeletons: false,
            show_velocity: false,
            sampled_poses: Vec::new(),
            anchor: 0,
            root_position: [0.0; 3],
            velocity: [0.0; 3],
        }
    }

    /// The pose the view shows: Start's last context pose, a target's first pose.
    fn preview_pose(&self) -> &PoseData {
        &self.sampled_poses[self.anchor]
    }
}

// --- Playback (upstream's playing state, engine-free so it is unit-testable) ------------------------

/// The generation/playback flow of upstream `reset()`, `step()` and `generate_next_segment()`,
/// minus the engine calls (the app performs those on the returned decisions).
#[derive(Clone, Debug, Default)]
struct Playback {
    playing: bool,
    has_segment: bool,
    next_target_index: usize,
    playback_time: f32,
    segment_end_frame: usize,
}

/// What a [`Playback::advance`] step needs the app to do.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    /// Not playing.
    Idle,
    /// Generate the segment toward `next_target_index`, then call [`Playback::segment_ready`].
    NeedSegment,
    /// Sample the output at `time`; when `done`, the last four played poses become the next
    /// context and [`Playback::finish_segment`] runs.
    Sample { time: f32, done: bool },
}

impl Playback {
    /// Go back to Start (upstream `reset()`'s state half).
    fn reset(&mut self) {
        self.playing = false;
        self.has_segment = false;
        self.next_target_index = 1;
        self.playback_time = 0.0;
        self.segment_end_frame = 0;
    }

    fn end_time(&self) -> f32 {
        self.segment_end_frame as f32 / MODEL_SAMPLE_RATE
    }

    /// Frames that actually played when the transition point is reached.
    fn played_frames(&self) -> usize {
        self.segment_end_frame + 1
    }

    fn advance(&mut self, delta_time: f32) -> Step {
        if !self.playing {
            return Step::Idle;
        }
        if !self.has_segment {
            return Step::NeedSegment;
        }
        let end_time = self.end_time();
        self.playback_time = (self.playback_time + delta_time).min(end_time);
        Step::Sample { time: self.playback_time, done: self.playback_time >= end_time }
    }

    /// The generated segment is in the output buffer; play it up to `end_frame`.
    fn segment_ready(&mut self, end_frame: usize) {
        self.segment_end_frame = end_frame;
        self.playback_time = 0.0;
        self.has_segment = true;
    }

    /// The transition point was reached: move to the next target, if any.
    fn finish_segment(&mut self, num_entries: usize) {
        self.next_target_index += 1;
        self.has_segment = false;
        self.playing = self.next_target_index < num_entries;
    }
}

/// The segment's transition point (upstream `generate_next_segment`): Transition Early By
/// skips frames at the end of the motion; at least four frames still play, because the next
/// context is the last four played frames.
fn transition_end_frame(num_frames: usize, transition_early_by_frames: usize) -> usize {
    (num_frames - 1).saturating_sub(transition_early_by_frames).max(MODEL_POSES_PER_TOKEN - 1)
}

// --- Rig visuals (mesh drawing, skeleton lines, foot contacts) --------------------------------------

/// Per-rig drawing assets: what upstream's mesh/skeleton/foot-contact renderers hold.
enum RigVisual {
    Soma {
        rig: SomaRig,
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
    /// Log a pose's mesh: the SOMA skinned mesh or the G1's rigid links. `color` tints the
    /// whole rig; `None` uses the playback colors (gray SOMA, the G1's own link colors).
    /// Returns how many mesh instances were logged (for click-picking).
    fn draw_pose(&self, frame: &mut Frame, pose: &PoseData, color: Option<Color>) -> Result<usize> {
        match self {
            RigVisual::Soma { rig, skin, .. } => {
                let world = rig.fk(as_soma(pose)?, ZUP_TO_YUP)?;
                let bones = rig.bone_matrices(&world);
                frame.log_skinned_mesh(&rig.mesh.mesh, skin, &bones, Mat4::IDENTITY, color.unwrap_or(PLAYBACK_SOMA_COLOR));
                Ok(1)
            }
            RigVisual::G1 { rig, link_meshes, .. } => {
                let world = rig.fk(as_g1(pose)?)?;
                let zup_to_yup = zup_to_yup_mat();
                for (link, mesh) in rig.links.iter().zip(link_meshes) {
                    let m = zup_to_yup * soma::transform_to_mat4(world[link.body] * link.offset);
                    frame.log_mesh(mesh, m, color.unwrap_or(link.rgba));
                }
                Ok(rig.links.len())
            }
        }
    }

    /// Log a pose as bone lines (upstream's skeleton renderers), in world space.
    fn draw_skeleton(&self, frame: &mut Frame, pose: &PoseData, color: Color) -> Result<()> {
        match self {
            RigVisual::Soma { rig, .. } => {
                // FK under ZUP_TO_YUP yields viewer-space joint transforms directly.
                let world = rig.fk(as_soma(pose)?, ZUP_TO_YUP)?;
                let mut segments = Vec::with_capacity(world.len() - 1);
                for (j, w) in world.iter().enumerate().skip(1) {
                    let parent = world[rig.skeleton.parent_indices()[j] as usize];
                    segments.push([Vec3::from_array(w.p.0), Vec3::from_array(parent.p.0)]);
                }
                frame.log_lines(&segments, color);
            }
            RigVisual::G1 { rig, .. } => {
                let world = rig.fk(as_g1(pose)?)?;
                let mut segments = Vec::with_capacity(world.len() - 1);
                for (b, w) in world.iter().enumerate().skip(1) {
                    let parent = world[usize::try_from(rig.bodies[b].parent)?];
                    segments.push([to_view(w.p.0), to_view(parent.p.0)]);
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
                    let center = to_view(world[body].point(rt::Vec3(rt::UNITREE_G1_FOOT_CONTACT_OFFSETS[c])).0);
                    let segments: Vec<[Vec3; 2]> = (0..CONTACT_RING_SEGMENTS)
                        .map(|i| {
                            let (a0, a1) = (i as f32, (i + 1) as f32);
                            let p = |a: f32| center + Vec3::new((a * std::f32::consts::TAU / CONTACT_RING_SEGMENTS as f32).cos(), 0.0, (a * std::f32::consts::TAU / CONTACT_RING_SEGMENTS as f32).sin()) * CONTACT_RING_RADIUS;
                            [p(a0), p(a1)]
                        })
                        .collect();
                    frame.log_lines(&segments, CONTACT_COLOR);
                }
            }
        }
        Ok(())
    }
}

// --- The runtime (engine + reusable tokens and output buffer) ----------------------------------------

/// The rig's engine and its reusable context token, target token and output buffer
/// (upstream `_prepare_rig`). Every generation reuses them.
enum RigRuntime {
    Soma {
        engine: MotionBricksSomaEngine<MetalModel>,
        context: PoseToken,
        target: PoseToken,
        output: AnimationBuffer,
    },
    G1 {
        engine: MotionBricksUnitreeG1Engine<MetalModel>,
        context: DofToken,
        target: DofToken,
        output: DofAnimationBuffer,
    },
}

impl RigRuntime {
    /// Load the context token with four poses (upstream `context.update_from_poses`).
    fn set_context(&mut self, poses: &[PoseData]) -> Result<()> {
        match self {
            RigRuntime::Soma { context, .. } => {
                let poses: Vec<&[Transform]> = poses.iter().map(as_soma).collect::<Result<_>>()?;
                context.update_from_poses(&poses)
            }
            RigRuntime::G1 { context, .. } => {
                let poses: Vec<&[f32]> = poses.iter().map(as_g1).collect::<Result<_>>()?;
                context.update_from_poses(&poses)
            }
        }
    }

    /// Generate toward one to four target poses (upstream `generate_next_segment`'s engine
    /// half: `target.update_from_poses` + `engine.generate_animation`). Returns the number of
    /// valid output frames.
    fn generate(&mut self, target_poses: &[PoseData], config: &MotionBricksInferenceConfig) -> Result<usize> {
        match self {
            RigRuntime::Soma { engine, context, target, output } => {
                let poses: Vec<&[Transform]> = target_poses.iter().map(as_soma).collect::<Result<_>>()?;
                target.update_from_poses(&poses)?;
                engine.generate_animation(context, target, output, config)?;
                Ok(output.num_frames())
            }
            RigRuntime::G1 { engine, context, target, output } => {
                let poses: Vec<&[f32]> = target_poses.iter().map(as_g1).collect::<Result<_>>()?;
                target.update_from_poses(&poses)?;
                engine.generate_animation(context, target, output, config)?;
                Ok(output.num_frames())
            }
        }
    }

    /// The pose of the generated motion at a time (upstream `output.sample`).
    fn sample_output(&self, time: f32) -> Result<PoseData> {
        match self {
            RigRuntime::Soma { output, .. } => Ok(PoseData::Soma(output.sample(time, Transform::IDENTITY)?)),
            RigRuntime::G1 { output, .. } => Ok(PoseData::G1(output.sample(time, Transform::IDENTITY)?)),
        }
    }

    /// One played frame of the generated motion (for the next context).
    fn output_frame(&self, frame: usize) -> PoseData {
        match self {
            RigRuntime::Soma { output, .. } => PoseData::Soma(output.frame(frame).to_vec()),
            RigRuntime::G1 { output, .. } => PoseData::G1(output.frame(frame).to_vec()),
        }
    }

    fn output_num_frames(&self) -> usize {
        match self {
            RigRuntime::Soma { output, .. } => output.num_frames(),
            RigRuntime::G1 { output, .. } => output.num_frames(),
        }
    }

    fn output_duration(&self) -> f32 {
        match self {
            RigRuntime::Soma { output, .. } => output.duration(),
            RigRuntime::G1 { output, .. } => output.duration(),
        }
    }

    fn sample_output_foot_contacts(&self, time: f32) -> Result<FootContacts> {
        match self {
            RigRuntime::Soma { output, .. } => output.sample_foot_contacts(time),
            RigRuntime::G1 { output, .. } => output.sample_foot_contacts(time),
        }
    }
}

// --- Rig state and engine construction ---------------------------------------------------------------

/// Everything one rig needs: visuals, the runtime, the clip registry and the space conversion
/// (upstream `load_rig_assets` + `_prepare_rig`).
struct RigState {
    rig: ModelRig,
    visual: RigVisual,
    runtime: RigRuntime,
    registry: ClipRegistry,
    model_to_runtime: Transform,
}

/// Build a rig's engine (the Metal runtime behind the Rust Runtime API), its visual assets
/// and its clip registry. This is the engine-construction code path shared with the other
/// apps: `MetalModel::from_checkpoints_with(rig, checkpoints/<hf_folder>, batch 1, precision)`
/// wrapped in the rig's `motionbricks_rs` engine with `rt::model_to_runtime(rig)`.
fn build_rig_state(cli: &Cli, rig: ModelRig) -> Result<RigState> {
    let spec = get_spec(rig.name())?;
    let dir = cli.checkpoints.join(spec.hf_folder());
    eprintln!("[inbetweener] loading the {} model from {} ({})", rig.name(), dir.display(), cli.precision);
    let backend = MetalModel::from_checkpoints_with(rig.name(), &dir, 1, cli.precision.as_str())
        .with_context(|| format!("could not load the {} model from {}", rig.name(), dir.display()))?;
    let model_to_runtime = rt::model_to_runtime(rig);
    match rig {
        ModelRig::Soma => {
            let rig_assets = SomaRig::load(&cli.clips)?;
            let skin = rig_assets.viewer_skin();
            let outlines = rig_assets.sole_outlines()?;
            // Engine tokens and clips both use the mesh skeleton, exactly like upstream
            // (`ClipRegistry(rig, skeleton=skeletal_mesh.skeleton)`, `engine.create_input(skeleton)`).
            let registry = ClipRegistry::soma(rig_assets.skeleton.clone());
            let mut engine = MotionBricksSomaEngine::new(Some(backend), model_to_runtime)?;
            let context = engine.create_input(&rig_assets.skeleton)?;
            let target = engine.create_input(&rig_assets.skeleton)?;
            let output = engine.create_output(&rig_assets.skeleton)?;
            Ok(RigState {
                rig,
                visual: RigVisual::Soma { rig: rig_assets, skin, outlines },
                runtime: RigRuntime::Soma { engine, context, target, output },
                registry,
                model_to_runtime,
            })
        }
        ModelRig::UnitreeG1 => {
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
            let registry = ClipRegistry::unitree_g1_contract();
            let engine = MotionBricksUnitreeG1Engine::new(Some(backend), model_to_runtime)?;
            let context = engine.create_input();
            let target = engine.create_input();
            let output = engine.create_output();
            Ok(RigState {
                rig,
                visual: RigVisual::G1 { rig: rig_assets, link_meshes, contact_bodies },
                runtime: RigRuntime::G1 { engine, context, target, output },
                registry,
                model_to_runtime,
            })
        }
    }
}

// --- The app (upstream `InbetweenerDemo`) ------------------------------------------------------------

/// Rename state for one pose, drawn as a text field in place of its name (upstream
/// `InlineRename`). Enter or leaving the field confirms; Escape cancels.
#[derive(Default)]
struct InlineRename {
    item_id: Option<usize>,
    text: String,
    focus: bool,
}

impl InlineRename {
    fn begin(&mut self, item_id: usize, name: &str) {
        self.item_id = Some(item_id);
        self.text = name.to_string();
        self.focus = true;
    }

    /// Draw the rename field; returns the new name once it is confirmed.
    fn draw(&mut self, ui: &Ui) -> Option<String> {
        let id = self.item_id?;
        ui.set_next_item_width(-1.0);
        if self.focus {
            ui.set_keyboard_focus_here();
            self.focus = false;
        }
        let submitted = ui.input_text(format!("##rename_{id}"), &mut self.text).enter_returns_true(true).auto_select_all(true).build();
        if ui.is_key_pressed(Key::Escape) {
            self.item_id = None;
            None
        } else if submitted || ui.is_item_deactivated() {
            self.item_id = None;
            let name = self.text.trim().to_string();
            (!name.is_empty()).then_some(name)
        } else {
            None
        }
    }
}

/// A clip choice from the path-input modal, applied at the start of the next frame.
struct PendingClip {
    entry_id: usize,
    path: PathBuf,
}

struct App {
    cli: Cli,
    state: RigState,
    entries: Vec<PoseEntry>,
    selected_index: usize,
    next_id: usize,
    playback: Playback,
    current_pose: PoseData,
    rename: InlineRename,
    error: Option<String>,
    pending_rig: Option<ModelRig>,
    /// Entry whose clip path is being edited in the modal.
    clip_modal: Option<usize>,
    clip_path_text: String,
    pending_clip: Option<PendingClip>,
    show_foot_contacts: bool,
    foot_contact_threshold: f32,
    reset_camera: bool,
}

impl App {
    fn new(cli: Cli) -> Result<App> {
        let state = build_rig_state(&cli, cli.rig)?;
        let mut app = App {
            cli,
            state,
            entries: Vec::new(),
            selected_index: 0,
            next_id: 0,
            playback: Playback::default(),
            current_pose: PoseData::Soma(Vec::new()),
            rename: InlineRename::default(),
            error: None,
            pending_rig: None,
            clip_modal: None,
            clip_path_text: String::new(),
            pending_clip: None,
            show_foot_contacts: false,
            foot_contact_threshold: DEFAULT_CONTACT_THRESHOLD,
            reset_camera: true,
        };
        app.prepare_entries()?;
        Ok(app)
    }

    /// Place Start and one target, both sampled from the rig's default clip (upstream
    /// `_prepare_rig`'s pose list).
    fn prepare_entries(&mut self) -> Result<()> {
        let rig_index = match self.state.rig {
            ModelRig::Soma => 0,
            ModelRig::UnitreeG1 => 1,
        };
        let path = self.cli.clips.join(DEFAULT_CLIP[rig_index]);
        let duration = self
            .state
            .registry
            .load(&path)
            .with_context(|| format!("could not load the default clip {}", path.display()))?
            .duration();
        self.entries = vec![
            PoseEntry::new(0, "Start".to_string(), path.clone(), 1.0f32.min(duration), Transform::IDENTITY, START_COLOR),
            PoseEntry::new(
                1,
                "Target 1".to_string(),
                path,
                7.0f32.min(duration),
                Transform::new(rt::Vec3([0.0, -2.0, 0.0]), rt::math::QUAT_IDENTITY),
                TARGET_COLOR,
            ),
        ];
        self.selected_index = 1;
        self.next_id = 2;
        self.refresh_poses()
    }

    /// Switch to a rig and start over with Start, one target and the default view (upstream
    /// `use_rig`). The new engine is built before the old one is dropped, so a failed switch
    /// keeps the current rig (upstream closes the old engine first and reports the error).
    fn switch_rig(&mut self, rig: ModelRig) {
        match build_rig_state(&self.cli, rig).and_then(|state| {
            self.state = state;
            self.prepare_entries()
        }) {
            Ok(()) => self.reset_camera = true,
            Err(e) => self.error = Some(format!("Character load failed: {e:#}")),
        }
    }

    /// Resample Start and every target, update their previews, and return to Start (upstream
    /// `refresh_poses`).
    fn refresh_poses(&mut self) -> Result<()> {
        for index in 0..self.entries.len() {
            // Start shows its last context pose, and a target shows its first pose.
            let anchor = if index == 0 { MODEL_POSES_PER_TOKEN - 1 } else { 0 };
            let (sample_time, placement, num_poses_to_use, path) = {
                let e = &self.entries[index];
                (e.sample_time, e.placement, e.num_poses_to_use, e.path.clone())
            };
            let clip = self.state.registry.load(&path)?.clip.clone();
            let sampled = sample_pose_window(&clip, sample_time, placement, self.state.model_to_runtime, anchor)?;
            let roots: Vec<[f32; 3]> = sampled.iter().map(PoseData::root_position).collect();
            // Use motion leading into Start, or motion leaving a target, for its velocity.
            let (a, b) = if index == 0 { (roots[2], roots[3]) } else { (roots[0], roots[1]) };
            let mut velocity = [(b[0] - a[0]) * MODEL_SAMPLE_RATE, (b[1] - a[1]) * MODEL_SAMPLE_RATE, (b[2] - a[2]) * MODEL_SAMPLE_RATE];
            if num_poses_to_use == 1 {
                velocity = [0.0; 3];
            }
            let entry = &mut self.entries[index];
            entry.sampled_poses = sampled;
            entry.anchor = anchor;
            entry.root_position = roots[anchor];
            entry.velocity = velocity;
        }
        self.reset()
    }

    fn refresh_poses_ok(&mut self) {
        if let Err(e) = self.refresh_poses() {
            self.playback.playing = false;
            self.error = Some(format!("{e:#}"));
        }
    }

    /// Go back to Start and restore its four context poses (upstream `reset`).
    fn reset(&mut self) -> Result<()> {
        self.playback.reset();
        let start = &self.entries[0];
        self.state.runtime.set_context(&start.sampled_poses)?;
        self.current_pose = start.preview_pose().clone();
        Ok(())
    }

    /// Restart from Start and play through the targets (upstream `play`).
    fn play(&mut self) {
        if let Err(e) = self.reset() {
            self.error = Some(format!("{e:#}"));
            return;
        }
        if self.entries.len() > 1 {
            self.playback.playing = true;
        }
    }

    /// Generate from the current context toward the next placed target (upstream
    /// `generate_next_segment`).
    fn generate_next_segment(&mut self) -> Result<()> {
        let (target_poses, config, early) = {
            let entry = &self.entries[self.playback.next_target_index];
            (
                entry.sampled_poses[..entry.num_poses_to_use].to_vec(),
                MotionBricksInferenceConfig { allowed_token_counts: (entry.min_tokens..=entry.max_tokens).collect(), ..MotionBricksInferenceConfig::default() },
                entry.transition_early_by_frames,
            )
        };
        let num_frames = self.state.runtime.generate(&target_poses, &config)?;
        self.playback.segment_ready(transition_end_frame(num_frames, early));
        self.current_pose = self.state.runtime.sample_output(0.0)?;
        Ok(())
    }

    /// Advance playback (upstream `step`). At the transition point the last four played poses
    /// become the next context.
    fn step(&mut self, delta_time: f32) -> Result<()> {
        match self.playback.advance(delta_time) {
            Step::Idle => Ok(()),
            Step::NeedSegment => self.generate_next_segment(),
            Step::Sample { time, done } => {
                self.current_pose = self.state.runtime.sample_output(time)?;
                if done {
                    let played_frames = self.playback.played_frames();
                    let recent: Vec<PoseData> = (played_frames - MODEL_POSES_PER_TOKEN..played_frames).map(|f| self.state.runtime.output_frame(f)).collect();
                    self.state.runtime.set_context(&recent)?;
                    self.playback.finish_segment(self.entries.len());
                }
                Ok(())
            }
        }
    }

    /// Add a target after the last one, copying the selection's clip and sample time (upstream
    /// `add_target`).
    fn add_target(&mut self) {
        let (path, sample_time) = {
            let source = &self.entries[self.selected_index];
            (source.path.clone(), source.sample_time)
        };
        let previous = self.entries[self.entries.len() - 1].placement;
        let placement = Transform::new(rt::Vec3([previous.p.x(), previous.p.y() - 1.5, previous.p.z()]), previous.q);
        let id = self.next_id;
        self.entries.push(PoseEntry::new(id, format!("Target {id}"), path, sample_time, placement, TARGET_COLOR));
        self.next_id += 1;
        self.selected_index = self.entries.len() - 1;
        self.refresh_poses_ok();
    }

    /// Remove the selected target. Start cannot be removed (upstream `remove_target`).
    fn remove_target(&mut self) {
        if self.selected_index == 0 {
            return;
        }
        self.entries.remove(self.selected_index);
        self.rename.item_id = None;
        self.selected_index = self.selected_index.min(self.entries.len() - 1);
        self.refresh_poses_ok();
    }

    /// Move the target at `source` to `destination`. Start always stays first (upstream
    /// `move_target`).
    fn move_target(&mut self, source: usize, destination: usize) {
        if source == destination || source == 0 || destination == 0 || source >= self.entries.len() || destination >= self.entries.len() {
            return;
        }
        let entry = self.entries.remove(source);
        self.entries.insert(destination, entry);
        self.selected_index = destination;
        self.refresh_poses_ok();
    }

    /// Apply the clip path chosen in the modal (upstream `_choose_pending_clip`).
    fn apply_clip(&mut self, pending: PendingClip) {
        let result = (|| -> Result<()> {
            let clip = self.state.registry.load(&pending.path)?;
            let sample_time = initial_clip_sample_time(clip.duration());
            let Some(entry) = self.entries.iter_mut().find(|e| e.id == pending.entry_id) else {
                bail!("the pose is gone");
            };
            entry.path = pending.path;
            entry.sample_time = sample_time;
            Ok(())
        })();
        match result.and_then(|()| self.refresh_poses()) {
            Ok(()) => {}
            Err(e) => {
                self.playback.playing = false;
                self.error = Some(format!("Open failed: {e:#}"));
            }
        }
    }
}

/// The destination index of a pose-list drag: dropping on the lower half of a
/// row inserts after it (upstream `_draw_pose_list`).
fn drop_destination(source: usize, row: usize, below_mid: bool) -> usize {
    let insertion = row + usize::from(below_mid);
    insertion - usize::from(source < insertion)
}

/// A freshly assigned clip starts at the time whose four-pose window fills a
/// token (upstream `_choose_pending_clip`).
fn initial_clip_sample_time(duration: f32) -> f32 {
    ((MODEL_POSES_PER_TOKEN - 1) as f32 / MODEL_SAMPLE_RATE).min(duration)
}

// --- Picking ------------------------------------------------------------------------------------------

/// Closest distance of `ray` to the segment `a`-`b`, treated as a thin capsule of `radius`
/// (upstream `raycast_line`). Returns the ray parameter of the hit, or None.
fn ray_segment_hit(ray: &Ray, a: Vec3, b: Vec3, radius: f32, max_t: f32) -> Option<f32> {
    let d1 = ray.dir;
    let d2 = b - a;
    let r = ray.origin - a;
    let a1 = d1.dot(d1);
    let b2 = d2.dot(d2);
    if a1 < 1e-12 || b2 < 1e-12 {
        return None;
    }
    let b1 = d1.dot(d2);
    let c1 = d1.dot(r);
    let c2 = d2.dot(r);
    let denom = a1 * b2 - b1 * b1;
    // Closest ray parameter t and segment parameter s of the two infinite lines.
    let (mut t, mut s) = if denom.abs() > 1e-12 {
        ((b1 * c2 - c1 * b2) / denom, (a1 * c2 - b1 * c1) / denom)
    } else {
        (0.0, -c2 / b2)
    };
    // Clamp to the segment, then re-solve t; clamp to the ray, then re-solve s.
    let s_clamped = s.clamp(0.0, 1.0);
    if s_clamped != s {
        s = s_clamped;
        t = (s * b1 - c1) / a1;
    }
    if t < 0.0 {
        t = 0.0;
        s = ((-c2) / b2).clamp(0.0, 1.0);
    }
    let closest = (ray.origin + d1 * t) - (a + d2 * s);
    if t > 0.0 && t < max_t && closest.length() < radius { Some(t) } else { None }
}

/// Closest hit of `ray` on a pose's ground ring (a circle of [`RING_SEGMENTS`] segments).
fn ray_ring_hit(ray: &Ray, center: Vec3, max_t: f32) -> Option<f32> {
    let mut best = max_t;
    let point = |i: usize| {
        let a = i as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
        center + Vec3::new(a.cos(), 0.0, a.sin()) * RING_RADIUS
    };
    for i in 0..RING_SEGMENTS {
        if let Some(t) = ray_segment_hit(ray, point(i), point(i + 1), LINE_PICK_RADIUS, best) {
            best = t;
        }
    }
    (best < max_t).then_some(best)
}

/// Maps the viewer's per-frame mesh-instance indices to pose entries, so a clicked mesh
/// selects its pose. Preview meshes are logged first, in entry order, so the app knows each
/// entry's instance range without the viewer exposing its log.
#[derive(Default)]
struct PosePicker {
    /// (first instance, one past the last instance, entry index) per visible pose mesh.
    ranges: Vec<(usize, usize, usize)>,
    next_instance: usize,
}

impl PosePicker {
    fn register(&mut self, entry_index: usize, instances: usize) {
        if instances > 0 {
            self.ranges.push((self.next_instance, self.next_instance + instances, entry_index));
            self.next_instance += instances;
        }
    }

    /// Select the nearest pose whose mesh or ground ring is under the cursor (upstream
    /// `_select_pose`).
    fn pick(&self, frame: &Frame, ray: &Ray, entries: &[PoseEntry]) -> Option<usize> {
        let mut best = f32::MAX;
        let mut selected = None;
        if let Some(pick) = frame.pick_mesh(ray) {
            for &(start, end, index) in &self.ranges {
                if (start..end).contains(&pick.mesh_index) && pick.distance < best {
                    best = pick.distance;
                    selected = Some(index);
                }
            }
        }
        for (index, entry) in entries.iter().enumerate() {
            let center = to_view(entry.root_position) + Vec3::new(0.0, GROUND_MARKER_HEIGHT, 0.0);
            if let Some(t) = ray_ring_hit(ray, center, best) {
                best = t;
                selected = Some(index);
            }
        }
        selected
    }
}

// --- Transform fields (upstream authoring/common.py's transform editor) --------------------------------

/// Return 0 for a value too small to show (upstream `clean_display_value`).
fn clean_display_value(value: f32) -> f32 {
    if value.abs() < 5.0e-4 { 0.0 } else { value }
}

/// The equivalent angle (mod 360) closest to `reference` (upstream `_nearest_angle`).
fn nearest_angle(angle: f32, reference: f32) -> f32 {
    angle - 360.0 * ((angle - reference) / 360.0).round()
}

/// Quaternion (XYZW) -> display euler angles in degrees, kept close to the previous shown
/// angles to avoid jumps (a simplification of upstream `stable_rotation_degrees`: the
/// alternate 180-degree branch is not tried).
fn stable_rotation_degrees(q: rt::Quat, reference: Option<[f32; 3]>) -> [f32; 3] {
    let (x, y, z) = glam::Quat::from_array(q).to_euler(glam::EulerRot::XYZ);
    let primary = [x.to_degrees(), y.to_degrees(), z.to_degrees()];
    match reference {
        None => primary.map(clean_display_value),
        Some(reference) => [0, 1, 2].map(|i| clean_display_value(nearest_angle(primary[i], reference[i]))),
    }
}

fn euler_degrees_to_quat(degrees: [f32; 3]) -> rt::Quat {
    glam::Quat::from_euler(glam::EulerRot::XYZ, degrees[0].to_radians(), degrees[1].to_radians(), degrees[2].to_radians()).to_array()
}

/// One Translation/Rotation row: the label (right-click resets) and the X, Y and Z fields
/// with their axis locks (right-click an axis label to lock it, like upstream).
fn draw_transform_row(ui: &Ui, label: &str, values: &mut [f32; 3], locks: &mut [bool; 3], scale: f32, speed: f32) -> bool {
    let mut changed = false;
    let _row = ui.push_id(label);
    ui.small_button(label);
    if ui.is_item_clicked_with_button(MouseButton::Right) {
        *values = [0.0; 3];
        changed = true;
    }
    ui.same_line_with_pos(110.0 * scale);
    for (axis, name) in ["X", "Y", "Z"].iter().enumerate() {
        if axis > 0 {
            ui.same_line();
        }
        let axis_label = format!("{name}##{label}{axis}");
        let _color = if locks[axis] { Some(ui.push_style_color(StyleColor::Text, [1.0, 0.3, 0.3, 1.0])) } else { None };
        ui.small_button(axis_label);
        drop(_color);
        if ui.is_item_clicked_with_button(MouseButton::Right) {
            locks[axis] = !locks[axis];
        }
        ui.same_line();
        ui.set_next_item_width(60.0 * scale);
        changed |= Drag::new(format!("##{label}{axis}value")).speed(speed).build(ui, &mut values[axis]);
    }
    changed
}

/// Edit a pose's position and rotation with reset menus and axis locks (upstream
/// `draw_transform_editor`). Locked axes keep their value.
fn draw_transform_editor(ui: &Ui, entry: &mut PoseEntry, scale: f32) -> bool {
    let position = entry.placement.p.0.map(clean_display_value);
    let rotation = stable_rotation_degrees(entry.placement.q, entry.euler_degrees);
    entry.euler_degrees = Some(rotation);
    let (mut pos, mut rot) = (position, rotation);
    let moved = draw_transform_row(ui, "Translation", &mut pos, &mut entry.translation_locks, scale, 0.02);
    let rotated = draw_transform_row(ui, "Rotation", &mut rot, &mut entry.rotation_locks, scale, 0.5);
    if moved {
        let mut p = entry.placement.p.0;
        for axis in 0..3 {
            if !entry.translation_locks[axis] {
                p[axis] = pos[axis];
            }
        }
        entry.placement.p = rt::Vec3(p);
    }
    if rotated {
        let mut r = rotation;
        for axis in 0..3 {
            if !entry.rotation_locks[axis] {
                r[axis] = rot[axis];
            }
        }
        entry.placement.q = euler_degrees_to_quat(r);
    }
    moved || rotated
}

/// Compare two viewer-space transforms like upstream `transforms_equal` (q and -q are the
/// same rotation).
fn transforms_equal(a: Mat4, b: Mat4) -> bool {
    let (a, b) = (soma::mat4_to_transform(a), soma::mat4_to_transform(b));
    let position_equal = a.p.0.iter().zip(b.p.0).all(|(x, y)| (x - y).abs() <= 1e-6);
    let rotation_equal = a.q.iter().zip(b.q).all(|(x, y)| (x - y).abs() <= 1e-6) || a.q.iter().zip(b.q).all(|(x, y)| (x + y).abs() <= 1e-6);
    position_equal && rotation_equal
}

// --- imgui helpers (upstream authoring/common.py) -------------------------------------------------------

/// A label, then place and size the next widget on the same row (upstream `draw_label`).
fn draw_label(ui: &Ui, label: &str) {
    ui.align_text_to_frame_padding();
    ui.text(format!("{label}:"));
    ui.same_line();
    ui.set_next_item_width(-1.0);
}

fn show_tooltip(ui: &Ui, text: &str) {
    if ui.is_item_hovered() {
        ui.tooltip_text(text);
    }
}

/// Shorten `text` with "..." to fit `max_width` pixels (upstream `ellipsize_text`).
fn ellipsize_text(ui: &Ui, text: &str, max_width: f32) -> String {
    if max_width <= 0.0 || ui.calc_text_size(text)[0] <= max_width {
        return text.to_string();
    }
    const ELLIPSIS: &str = "...";
    if ui.calc_text_size(ELLIPSIS)[0] > max_width {
        return String::new();
    }
    let (mut lower, mut upper) = (0, text.len());
    while lower < upper {
        let middle = (lower + upper + 1) / 2;
        if ui.calc_text_size(format!("{}{ELLIPSIS}", &text[..middle]))[0] <= max_width {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    format!("{}{ELLIPSIS}", &text[..lower])
}

// --- The Inbetweener window (upstream `gui`, `_draw_pose_list`, `_draw_properties`) --------------------

impl App {
    fn gui(&mut self, ui: &Ui) {
        let viewport = ui.io().display_size;
        let scale = ui.current_font_size() / 13.0;
        ui.window("Inbetweener")
            .position([viewport[0] - 12.0, 12.0], Condition::FirstUseEver)
            .position_pivot([1.0, 0.0])
            .size([440.0 * scale, (740.0 * scale).min(viewport[1] - 24.0)], Condition::FirstUseEver)
            .build(|| self.gui_body(ui, scale));
    }

    fn gui_body(&mut self, ui: &Ui, scale: f32) {
        let footer_height = ui.frame_height_with_spacing() + ui.clone_style().item_spacing[1];
        ui.child_window("pose_settings").size([0.0, -footer_height]).build(|| {
            // Character selector (upstream draw_character_selector).
            draw_label(ui, "Character");
            let mut rig_index = match self.state.rig {
                ModelRig::Soma => 0,
                ModelRig::UnitreeG1 => 1,
            };
            if ui.combo("##Character", &mut rig_index, &RIG_LABELS, |label| Cow::Borrowed(*label)) {
                let rig = match rig_index {
                    0 => ModelRig::Soma,
                    _ => ModelRig::UnitreeG1,
                };
                if rig != self.state.rig {
                    self.pending_rig = Some(rig);
                }
            }
            if ui.collapsing_header("Target Poses", TreeNodeFlags::DEFAULT_OPEN) {
                self.draw_pose_list(ui);
                let add = ui.button("+");
                ui.same_line();
                ui.disabled(self.selected_index == 0, || {
                    if ui.button("-") {
                        self.remove_target();
                    }
                });
                if add {
                    self.add_target();
                }
            }
            if ui.collapsing_header("Properties", TreeNodeFlags::DEFAULT_OPEN) {
                self.draw_properties(ui, scale);
            }
        });
        ui.separator();
        ui.disabled(self.entries.len() < 2, || {
            if ui.button("Generate") {
                self.play();
            }
        });
        ui.same_line();
        ui.checkbox("Show Foot Contacts", &mut self.show_foot_contacts);
        show_tooltip(ui, "Show outlines for foot contacts above the threshold on the right.");
        ui.same_line();
        ui.set_next_item_width(80.0 * scale);
        ui.disabled(!self.show_foot_contacts, || {
            ui.slider_config("##FootContactThreshold", 0.0f32, 1.0f32)
                .display_format("%.2f")
                .flags(SliderFlags::ALWAYS_CLAMP)
                .build(&mut self.foot_contact_threshold);
        });
        self.draw_error_modal(ui);
        self.draw_clip_modal(ui);
    }

    fn draw_pose_list(&mut self, ui: &Ui) {
        let mut move_request: Option<(usize, usize)> = None;
        ui.child_window("pose_list").size([0.0, 160.0]).border(true).build(|| {
            for index in 0..self.entries.len() {
                let (id, name) = { let e = &self.entries[index]; (e.id, e.name.clone()) };
                if self.rename.item_id == Some(id) {
                    if let Some(new_name) = self.rename.draw(ui) {
                        self.entries[index].name = new_name;
                    }
                    continue;
                }
                let clicked = ui.selectable_config(format!("{name}###pose_{id}")).selected(self.selected_index == index).build();
                let (row_min, row_max) = (ui.item_rect_min(), ui.item_rect_max());
                if clicked {
                    self.selected_index = index;
                }
                if ui.is_item_hovered() && ui.is_mouse_double_clicked(MouseButton::Left) {
                    self.selected_index = index;
                    self.rename.begin(id, &self.entries[index].name.clone());
                }
                if index == 0 {
                    continue; // Start always stays first: it is no drag source or drop target.
                }
                if let Some(tooltip) = ui.drag_drop_source_config(POSE_PAYLOAD).begin_payload(id) {
                    ui.text(&name);
                    tooltip.end();
                }
                if let Some(target) = ui.drag_drop_target() {
                    if let Some(Ok(payload)) = target.accept_payload::<usize, _>(POSE_PAYLOAD, DragDropFlags::empty()) {
                        if let Some(source) = self.entries.iter().position(|e| e.id == payload.data) {
                            let below_mid = ui.io().mouse_pos[1] > (row_min[1] + row_max[1]) * 0.5;
                            move_request = Some((source, drop_destination(source, index, below_mid)));
                        }
                    }
                }
            }
        });
        if let Some((source, destination)) = move_request {
            self.move_target(source, destination);
        }
    }

    fn draw_properties(&mut self, ui: &Ui, scale: f32) {
        let index = self.selected_index;
        let (id, name, color) = {
            let e = &self.entries[index];
            (e.id, e.name.clone(), e.color)
        };
        ui.text_colored([color[0], color[1], color[2], 1.0], name);
        let mut refresh = false;
        if ui.collapsing_header("Transform", TreeNodeFlags::DEFAULT_OPEN) {
            let _id = ui.push_id_int(id as i32);
            refresh |= draw_transform_editor(ui, &mut self.entries[index], scale);
        }
        if ui.collapsing_header("Target Pose", TreeNodeFlags::DEFAULT_OPEN) {
            let _id = ui.push_id("target_pose");
            refresh |= self.draw_target_pose(ui, index, scale);
        }
        if ui.collapsing_header("Visibility", TreeNodeFlags::DEFAULT_OPEN) {
            let _id = ui.push_id("visibility");
            let entry = &mut self.entries[index];
            ui.color_edit3("Color", &mut entry.color);
            ui.checkbox("Show Mesh", &mut entry.show_mesh);
            ui.checkbox("Show Skeleton(s)", &mut entry.show_skeletons);
            ui.checkbox("Show Velocity", &mut entry.show_velocity);
        }
        if refresh {
            self.refresh_poses_ok();
        }
    }

    /// The clip selector, the time/frame scrubber and the per-target generation settings
    /// (upstream `_draw_properties`' "Target Pose" section). Returns whether poses changed.
    fn draw_target_pose(&mut self, ui: &Ui, index: usize, scale: f32) -> bool {
        let mut refresh = false;
        let (id, path, sample_time, show_frames, velocity) = {
            let e = &self.entries[index];
            (e.id, e.path.clone(), e.sample_time, e.show_frames, e.velocity)
        };
        // Clip selector (upstream draw_clip_selector; the tkinter dialog is a path modal).
        ui.align_text_to_frame_padding();
        ui.text("Clip:");
        ui.same_line();
        let button_width = ui.calc_text_size("...")[0] + 2.0 * ui.clone_style().frame_padding[0];
        let name_width = ui.content_region_avail()[0] - button_width - ui.clone_style().item_spacing[0];
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        ui.text(ellipsize_text(ui, &name, name_width));
        show_tooltip(ui, &path.display().to_string());
        ui.same_line();
        if ui.button_with_size(format!("...##LoadClip{id}"), [button_width, 0.0]) {
            self.clip_modal = Some(id);
            self.clip_path_text = path.display().to_string();
        }
        // Time/frame scrubber (upstream draw_clip_scrubber).
        let clip_stats = self.state.registry.load(&path).ok().map(|asset| (asset.last_frame(), asset.clip.sample_rate()));
        if let Some((last_frame, sample_rate)) = clip_stats {
            let duration = last_frame as f32 / sample_rate;
            let mut new_time = sample_time;
            let mut show_frames = show_frames;
            if ui.button_with_size(if show_frames { "Frames" } else { "Seconds" }, [65.0 * scale, 0.0]) {
                show_frames = !show_frames;
                if show_frames {
                    new_time = sample_frame_at(new_time, sample_rate, last_frame as u32) as f32 / sample_rate;
                }
            }
            ui.same_line();
            ui.set_next_item_width(250.0 * scale);
            if show_frames {
                let mut frame = sample_frame_at(new_time, sample_rate, last_frame as u32) as i32;
                if ui.slider_config("##SamplePosition", 0, last_frame as i32).flags(SliderFlags::ALWAYS_CLAMP).build(&mut frame) {
                    new_time = frame as f32 / sample_rate;
                }
                ui.same_line();
                ui.text(format!("{}", last_frame + 1));
            } else {
                ui.slider_config("##SamplePosition", 0.0f32, duration)
                    .display_format("%.2f")
                    .flags(SliderFlags::ALWAYS_CLAMP)
                    .build(&mut new_time);
                ui.same_line();
                ui.text(format!("{duration:.2} s"));
            }
            let entry = &mut self.entries[index];
            if new_time != sample_time {
                entry.sample_time = new_time;
                refresh = true;
            }
            entry.show_frames = show_frames;
        }
        let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
        ui.text(format!("Speed: {speed:.3} m/s"));
        ui.separator();
        // Start supplies the model's four required context poses, so these stay disabled.
        let (mut num_poses, mut early, mut min_tokens, mut max_tokens) = {
            let e = &self.entries[index];
            (e.num_poses_to_use as i32, e.transition_early_by_frames as i32, e.min_tokens, e.max_tokens)
        };
        let mut settings_changed = false;
        ui.disabled(index == 0, || {
            draw_label(ui, "Poses to Use");
            settings_changed |= ui.slider_config("##PosesToUse", 1, MODEL_POSES_PER_TOKEN as i32).flags(SliderFlags::ALWAYS_CLAMP).build(&mut num_poses);
            show_tooltip(ui, POSES_TO_USE_TOOLTIP);
            draw_label(ui, "Transition Early By (frames)");
            settings_changed |= ui.slider_config("##TransitionEarlyBy", 0, MAX_TRANSITION_EARLY_BY_FRAMES as i32).flags(SliderFlags::ALWAYS_CLAMP).build(&mut early);
            show_tooltip(ui, "Move on to the next target this many frames before the generated motion ends. At least four frames still play.");
            draw_label(ui, "Tokens to Generate");
            settings_changed |= DragRange::<i32, _>::new(format!("##tokens{id}"))
                .range(MODEL_MIN_TOKEN_COUNT, MODEL_MAX_TOKEN_COUNT)
                .display_format("Min: %d")
                .max_display_format("Max: %d")
                .flags(SliderFlags::ALWAYS_CLAMP)
                .build(ui, &mut min_tokens, &mut max_tokens);
            let (min_frames, max_frames) = (min_tokens * MODEL_POSES_PER_TOKEN as i32, max_tokens * MODEL_POSES_PER_TOKEN as i32);
            if min_frames == max_frames {
                ui.text(format!("Raw frames: {min_frames}"));
            } else {
                ui.text(format!("Raw frames: {min_frames} to {max_frames}"));
            }
        });
        if settings_changed {
            let entry = &mut self.entries[index];
            entry.num_poses_to_use = num_poses as usize;
            entry.transition_early_by_frames = early as usize;
            // Upstream `set_token_range`: the maximum never drops below the minimum.
            let (min_tokens, max_tokens) = clamp_token_range(min_tokens as u32, max_tokens as u32);
            entry.min_tokens = min_tokens as i32;
            entry.max_tokens = max_tokens as i32;
            refresh = true;
        }
        refresh
    }

    fn draw_error_modal(&mut self, ui: &Ui) {
        if self.error.is_some() {
            ui.open_popup("Error");
        }
        ui.modal_popup_config("Error").always_auto_resize(true).build(|| {
            if let Some(message) = &self.error {
                ui.text(message.clone());
            }
            if ui.button("OK") {
                self.error = None;
                ui.close_current_popup();
            }
        });
    }

    /// The clip path modal — upstream shows a tkinter Open dialog; this port takes a path in
    /// an imgui modal instead (noted deviation).
    fn draw_clip_modal(&mut self, ui: &Ui) {
        if self.clip_modal.is_some() {
            ui.open_popup("Open Clip");
        }
        ui.modal_popup_config("Open Clip").always_auto_resize(true).build(|| {
            let suffix = self.state.registry.suffix();
            ui.text(format!("Clip path ({suffix}):"));
            ui.set_next_item_width(360.0);
            ui.input_text("##clip_path", &mut self.clip_path_text).build();
            if ui.button("Open") {
                if let Some(entry_id) = self.clip_modal.take() {
                    self.pending_clip = Some(PendingClip { entry_id, path: PathBuf::from(self.clip_path_text.trim()) });
                }
                ui.close_current_popup();
            }
            ui.same_line();
            if ui.button("Cancel") {
                self.clip_modal = None;
                ui.close_current_popup();
            }
        });
    }
}

// --- The frame loop (upstream `render` + `run`) ---------------------------------------------------------

impl App {
    fn on_frame(&mut self, frame: &mut Frame, ui: &mut Ui) {
        if self.reset_camera {
            set_default_camera(frame.camera());
            self.reset_camera = false;
        }
        // A switch can build an engine, so it runs before anything else in the frame.
        if let Some(rig) = self.pending_rig.take() {
            self.switch_rig(rig);
        }
        if let Some(pending) = self.pending_clip.take() {
            self.apply_clip(pending);
        }
        let delta_time = frame.dt().clamp(0.0, MAX_FRAME_SECONDS);
        if let Err(e) = self.step(delta_time) {
            self.playback.playing = false;
            self.error = Some(format!("Generation failed: {e:#}"));
        }

        self.draw_scene(frame, ui);
        self.gui(ui);
    }

    /// Draw one frame: the pose previews, the playback pose, the rings and arrows, and the
    /// placement gizmo; then apply picking and gizmo edits (upstream `render`). The UI runs
    /// afterwards in `on_frame`, so a click or gizmo drag is applied to this frame's log.
    fn draw_scene(&mut self, frame: &mut Frame, ui: &Ui) {
        frame.log_ground();

        // Pose previews, first: the picker maps their mesh-instance ranges to entries.
        let mut picker = PosePicker::default();
        for index in 0..self.entries.len() {
            let entry = &self.entries[index];
            if entry.sampled_poses.is_empty() {
                continue;
            }
            if entry.show_mesh {
                match self.state.visual.draw_pose(frame, entry.preview_pose(), Some(color4(entry.color))) {
                    Ok(instances) => picker.register(index, instances),
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            }
            if entry.show_skeletons {
                for pose in &entry.sampled_poses[..entry.num_poses_to_use.min(entry.sampled_poses.len())] {
                    if let Err(e) = self.state.visual.draw_skeleton(frame, pose, color4(entry.color)) {
                        self.error = Some(format!("{e:#}"));
                    }
                }
            }
            if entry.show_velocity && entry.velocity.iter().any(|v| v.abs() > 1e-6) {
                let root = to_view(entry.root_position);
                let velocity = to_view(entry.velocity);
                frame.log_arrow(root, root + velocity, [1.0, 0.0, 0.0, 1.0]);
            }
        }

        self.draw_playback_pose(frame);
        self.draw_foot_contacts(frame);
        self.draw_order_markers(frame);

        // The transform gizmo places the selection (translation + the axis locks; the viewer
        // gizmo does not rotate — the Rotation fields do that).
        let gizmo_index = self.selected_index;
        let mut gizmo = zup_to_yup_mat() * soma::transform_to_mat4(self.entries[gizmo_index].placement);
        let before = gizmo;
        let gizmo_dragging = frame.gizmo_translate("pose_placement", &mut gizmo);

        // Click-picking selects a pose (blocked while imgui or the gizmo captures the mouse).
        if frame.input().mouse_pressed(winit::event::MouseButton::Left) && !ui.io().want_capture_mouse && !gizmo_dragging {
            if let Some(cursor) = frame.input().cursor {
                let ray = frame.screen_ray(cursor);
                if let Some(index) = picker.pick(frame, &ray, &self.entries) {
                    self.selected_index = index;
                }
            }
        }

        // Apply a gizmo edit to the selection, honoring the translation locks (upstream
        // `apply_transform_locks`, translation-only because the gizmo cannot rotate).
        if self.selected_index == gizmo_index && !transforms_equal(gizmo, before) {
            let entry = &mut self.entries[gizmo_index];
            let mut edited = soma::mat4_to_transform(zup_to_yup_mat().inverse() * gizmo);
            for axis in 0..3 {
                if entry.translation_locks[axis] {
                    edited.p.0[axis] = entry.placement.p.0[axis];
                }
            }
            entry.placement = edited;
            entry.euler_degrees = None;
            self.refresh_poses_ok();
        }
    }

    /// The playback mesh (upstream `_draw_generated_pose`): hidden when a visible target
    /// preview already shows the same pose.
    fn draw_playback_pose(&mut self, frame: &mut Frame) {
        let mut target_index = self.playback.next_target_index;
        if !self.playback.has_segment {
            target_index -= 1;
        }
        let endpoint_visible = self.entries[target_index].show_mesh;
        let at_endpoint = self.playback.playback_time >= self.state.runtime.output_duration();
        let previous_entry = &self.entries[self.playback.next_target_index - 1];
        let at_visible_start = self.playback.playback_time == 0.0 && previous_entry.show_mesh && previous_entry.transition_early_by_frames == 0;
        let has_playback = self.playback.has_segment || self.playback.next_target_index > 1;
        if has_playback && !at_visible_start && !(at_endpoint && endpoint_visible) {
            if let Err(e) = self.state.visual.draw_pose(frame, &self.current_pose, None) {
                self.error = Some(format!("{e:#}"));
            }
        }
    }

    /// Foot contacts of the generated motion, including its final pose (upstream
    /// `_draw_foot_contacts`).
    fn draw_foot_contacts(&mut self, frame: &mut Frame) {
        if !self.show_foot_contacts || self.state.runtime.output_num_frames() == 0 || !(self.playback.has_segment || self.playback.next_target_index > 1) {
            return;
        }
        match self
            .state
            .runtime
            .sample_output_foot_contacts(self.playback.playback_time)
            .and_then(|contacts| self.state.visual.draw_foot_contacts(frame, &self.current_pose, contacts, self.foot_contact_threshold))
        {
            Ok(()) => {}
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Ground rings (Start green, targets blue, the selection orange) and order arrows
    /// between consecutive poses (upstream `_refresh_order_markers` + the render calls).
    fn draw_order_markers(&self, frame: &mut Frame) {
        let centers: Vec<Vec3> = self
            .entries
            .iter()
            .map(|e| to_view(e.root_position) + Vec3::new(0.0, GROUND_MARKER_HEIGHT, 0.0))
            .collect();
        for (index, entry) in self.entries.iter().enumerate() {
            let color = if index == self.selected_index { SELECTED_COLOR } else { entry.color };
            let segments: Vec<[Vec3; 2]> = (0..RING_SEGMENTS)
                .map(|i| {
                    let p = |k: usize| {
                        let a = k as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
                        centers[index] + Vec3::new(a.cos(), 0.0, a.sin()) * RING_RADIUS
                    };
                    [p(i), p(i + 1)]
                })
                .collect();
            frame.log_lines(&segments, color4(color));
        }
        for pair in centers.windows(2) {
            frame.log_arrow(pair[0], pair[1], color4(ORDER_COLOR));
        }
    }
}

/// Upstream's rig-switch camera: `pos (-7, -4, 3.5)`, `look_at (0, -1, 1)` in Z-up world.
fn set_default_camera(camera: &mut motionbricks_app::viewer::OrbitCamera) {
    let eye = to_view([-7.0, -4.0, 3.5]);
    let target = to_view([0.0, -1.0, 1.0]);
    let d = eye - target;
    camera.target = target;
    camera.distance = d.length();
    camera.pitch = (d.y / camera.distance).asin();
    camera.yaw = d.x.atan2(d.z);
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let mut app = App::new(cli)?;
    eprintln!("[inbetweener] engine ready; opening the window");
    let viewer = Viewer::new(ViewerConfig { title: "MotionBricks - Inbetweener".to_string(), width: 1280, height: 720 })?;
    viewer.run(move |frame, ui| app.on_frame(frame, ui));
}

// --- Tests (pure logic: no GPU, no engine) --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_end_frame_skips_and_keeps_four_frames() {
        // max(MODEL_POSES_PER_TOKEN - 1, num_frames - 1 - early), like upstream.
        assert_eq!(transition_end_frame(30, 0), 29);
        assert_eq!(transition_end_frame(30, 20), 9);
        assert_eq!(transition_end_frame(30, 50), 3, "at least four frames always play");
        assert_eq!(transition_end_frame(18, 0), 17);
    }

    #[test]
    fn window_times_anchor_sits_at_sample_time() {
        for anchor in 0..MODEL_POSES_PER_TOKEN {
            let times = window_times(2.5, anchor);
            assert_eq!(times.len(), MODEL_POSES_PER_TOKEN);
            assert!((times[anchor] - 2.5).abs() < 1e-7);
            for pair in times.windows(2) {
                assert!((pair[1] - pair[0] - 1.0 / MODEL_SAMPLE_RATE).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn playback_walks_the_targets_in_order() {
        // Two targets; each generated segment has 30 frames, no early transition.
        let mut playback = Playback::default();
        playback.reset();
        assert_eq!(playback.advance(1.0), Step::Idle, "not playing yet");

        playback.reset();
        playback.playing = true;
        assert_eq!(playback.next_target_index, 1);
        assert_eq!(playback.advance(0.016), Step::NeedSegment);
        playback.segment_ready(transition_end_frame(30, 0));
        // Half a second in: still sampling the same segment.
        match playback.advance(0.5) {
            Step::Sample { time, done } => {
                assert!((time - 0.5).abs() < 1e-6);
                assert!(!done);
            }
            other => panic!("expected Sample, got {other:?}"),
        }
        // Past the end: clamps to the transition point and asks for the context swap.
        let end = 29.0 / MODEL_SAMPLE_RATE;
        match playback.advance(1.0) {
            Step::Sample { time, done } => {
                assert!((time - end).abs() < 1e-6);
                assert!(done);
            }
            other => panic!("expected Sample, got {other:?}"),
        }
        assert_eq!(playback.played_frames(), 30);
        playback.finish_segment(3);
        assert!(playback.playing && playback.next_target_index == 2 && !playback.has_segment);
        // The second (last) target: after it finishes, playback stops.
        playback.segment_ready(transition_end_frame(24, 10));
        assert_eq!(playback.played_frames(), 14, "14 frames play (early by 10 of 24)");
        for _ in 0..100 {
            if playback.advance(0.1) == (Step::Sample { time: playback.end_time(), done: true }) {
                break;
            }
        }
        playback.finish_segment(3);
        assert!(!playback.playing, "no targets left");
        assert_eq!(playback.advance(0.1), Step::Idle);
    }

    #[test]
    fn move_target_keeps_start_first() {
        // Mirror App::move_target on a bare vector.
        fn move_target(entries: &mut Vec<i32>, source: usize, destination: usize) {
            if source == destination || source == 0 || destination == 0 || source >= entries.len() || destination >= entries.len() {
                return;
            }
            let entry = entries.remove(source);
            entries.insert(destination, entry);
        }
        let mut entries = vec![0, 1, 2, 3];
        move_target(&mut entries, 0, 2);
        assert_eq!(entries, [0, 1, 2, 3], "Start cannot move");
        move_target(&mut entries, 2, 0);
        assert_eq!(entries, [0, 1, 2, 3], "nothing moves before Start");
        move_target(&mut entries, 1, 3);
        assert_eq!(entries, [0, 2, 3, 1]);
        move_target(&mut entries, 3, 1);
        assert_eq!(entries, [0, 1, 2, 3]);
        // The drop-insertion math from the pose list.
        let (source, insertion) = (1usize, 4usize);
        assert_eq!(insertion - usize::from(source < insertion), 3);
    }

    #[test]
    fn zup_to_yup_maps_up_to_up() {
        let m = zup_to_yup_mat();
        assert!(m.transform_point3(Vec3::new(0.0, 0.0, 1.0)).abs_diff_eq(Vec3::Y, 1e-6));
        assert!(m.transform_point3(Vec3::new(1.0, 2.0, 3.0)).abs_diff_eq(to_view([1.0, 2.0, 3.0]), 1e-6));
        // A placement composed through the conversion round-trips.
        let placement = Transform::new(rt::Vec3([1.0, -2.0, 0.5]), rt::math::QUAT_IDENTITY);
        let viewer = m * soma::transform_to_mat4(placement);
        assert!(viewer.transform_point3(Vec3::ZERO).abs_diff_eq(to_view(placement.p.0), 1e-6));
    }

    #[test]
    fn ray_segment_hit_capsule() {
        let ray = Ray { origin: Vec3::new(0.0, 1.0, -5.0), dir: Vec3::Z };
        let (a, b) = (Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(ray_segment_hit(&ray, a, b, LINE_PICK_RADIUS, f32::MAX), Some(5.0));
        assert_eq!(ray_segment_hit(&ray, a + Vec3::Y, b + Vec3::Y, LINE_PICK_RADIUS, f32::MAX), None, "a meter above the capsule");
        assert_eq!(ray_segment_hit(&ray, a, b, LINE_PICK_RADIUS, 4.0), None, "beyond max_t");
    }

    #[test]
    fn nearest_angle_unwraps() {
        assert_eq!(nearest_angle(1.0, 359.0), 361.0);
        assert_eq!(nearest_angle(-179.0, 179.0), 181.0);
        assert_eq!(nearest_angle(45.0, 45.0), 45.0);
    }

    #[test]
    fn stable_rotation_round_trip() {
        let q = euler_degrees_to_quat([10.0, -20.0, 30.0]);
        let shown = stable_rotation_degrees(q, None);
        let q2 = euler_degrees_to_quat(shown);
        // Same rotation, up to sign.
        let dot: f32 = q.iter().zip(q2).map(|(a, b)| a * b).sum();
        assert!(dot.abs() > 1.0 - 1e-5, "dot: {dot}");
    }

    #[test]
    fn drop_destination_inserts_before_and_after_rows() {
        // Dragging down: dropping on a later row's top half inserts before it,
        // on the bottom half after it (and the removal shifts the index).
        assert_eq!(drop_destination(1, 3, false), 2);
        assert_eq!(drop_destination(1, 3, true), 3);
        // Dragging up: no shift, the insertion index is the destination.
        assert_eq!(drop_destination(4, 1, false), 1);
        assert_eq!(drop_destination(4, 1, true), 2);
        // Dropping back where the row is keeps the order.
        assert_eq!(drop_destination(2, 2, false), 2);
        assert_eq!(drop_destination(2, 3, false), 2);
    }

    #[test]
    fn initial_clip_sample_time_fills_a_token_without_passing_the_clip() {
        let token_window = (MODEL_POSES_PER_TOKEN - 1) as f32 / MODEL_SAMPLE_RATE;
        assert_eq!(initial_clip_sample_time(10.0), token_window);
        assert_eq!(initial_clip_sample_time(0.05), 0.05, "a short clip starts at its end");
        assert_eq!(initial_clip_sample_time(0.0), 0.0);
    }

    #[test]
    fn token_settings_clamp_to_the_model_contract() {
        // The widget's range is the contract, and the stored range keeps min <= max.
        let (min_tokens, max_tokens) = clamp_token_range(10, 8);
        assert_eq!((min_tokens, max_tokens), (10, 10));
        let (min_tokens, max_tokens) = clamp_token_range(1, 100);
        assert_eq!(min_tokens, MODEL_MIN_TOKEN_COUNT as u32);
        assert_eq!(max_tokens, MODEL_MAX_TOKEN_COUNT as u32);
    }
}
