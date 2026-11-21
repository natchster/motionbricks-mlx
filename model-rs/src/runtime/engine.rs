// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (model/engine.py, model/generation.py,
// model/soma/engine.py, model/soma/kernels.py, model/unitree_g1/engine.py, model/unitree_g1/kernels.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Engines: poses in world space -> model inputs -> inference -> generated motion.
//!
//! Upstream's engines run a TensorRT plan on a CUDA stream; these run a [`Backend`] (the Metal
//! runtime or the mlx-rs port) and keep the same flow and math: align the context and target to a
//! heading, encode them into the ONNX-named inputs, run the model, decode the prediction back to
//! world space, drop the boundary frames, and correct the endpoints.

use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use mlx_rs::Array;

use super::animation::{AnimationBuffer, DofAnimationBuffer, FootContacts, Skeleton, read_root, write_root};
use super::contract::*;
use super::math::*;
use super::tokens::{DofToken, PoseToken};
use crate::metal::model::MetalModel;
use crate::npy::{Npy, NpyData};
use crate::ops::{R, to_vec_f32, to_vec_i32};
use crate::{MotionBricks, compiled};

/// What one generation returns from the model (batch 1).
#[derive(Clone, Debug)]
pub struct Prediction {
    /// `pred_global_motions` (SOMA, 64 x 366) or `pred_qpos` (G1, 64 x 36), flattened.
    pub motion: Vec<f32>,
    /// `pred_num_tokens`.
    pub num_tokens: i32,
    /// `pred_foot_contacts` (G1, 64 x 4); SOMA stores them in the last 4 motion columns.
    pub foot_contacts: Option<Vec<f32>>,
}

/// Inference on the ONNX-named inputs: the step upstream runs with TensorRT.
pub trait Backend {
    fn rig(&self) -> ModelRig;
    fn predict(&mut self, inputs: &HashMap<String, Npy>) -> Result<Prediction>;
}

impl Backend for MetalModel {
    fn rig(&self) -> ModelRig {
        ModelRig::parse(&self.spec.name).expect("known rig")
    }
    fn predict(&mut self, inputs: &HashMap<String, Npy>) -> Result<Prediction> {
        let (motion, tokens, contacts) = self.run(inputs)?;
        Ok(Prediction { motion, num_tokens: tokens[0], foot_contacts: contacts })
    }
}

/// The mlx-rs port as a backend, running one compiled graph per generation.
pub struct MlxBackend {
    pub model: Rc<MotionBricks>,
    forward: Box<dyn FnMut(&[Array]) -> R<Vec<Array>>>,
}

impl MlxBackend {
    pub fn new(model: MotionBricks) -> Self {
        let model = Rc::new(model);
        let forward = Box::new(compiled(model.clone()));
        MlxBackend { model, forward }
    }
}

impl Backend for MlxBackend {
    fn rig(&self) -> ModelRig {
        ModelRig::parse(&self.model.spec.name).expect("known rig")
    }
    fn predict(&mut self, inputs: &HashMap<String, Npy>) -> Result<Prediction> {
        let out = (self.forward)(&self.model.prepare(inputs)?)?;
        Ok(Prediction {
            motion: to_vec_f32(&out[0])?,
            num_tokens: to_vec_i32(&out[1])?[0],
            foot_contacts: out.get(2).map(to_vec_f32).transpose()?,
        })
    }
}

/// Wall-clock milliseconds of each stage of the last generation.
#[derive(Clone, Debug, Default)]
pub struct GenerationTimer {
    pub encode: f64,
    pub inference: f64,
    pub decode: f64,
    pub post_process: f64,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn f32s(data: Vec<f32>, shape: &[i32]) -> Npy {
    Npy { shape: shape.to_vec(), data: NpyData::F32(data) }
}

fn i32s(data: Vec<i32>, shape: &[i32]) -> Npy {
    Npy { shape: shape.to_vec(), data: NpyData::I32(data) }
}

/// Which context/target slots are given, and which token counts the model may choose. Local root
/// values stay unmarked at the last context and last target pose (unknown derivatives).
pub fn conditions(num_target_poses: usize, config: &MotionBricksInferenceConfig) -> HashMap<String, Npy> {
    let p = MODEL_POSES_PER_TOKEN;
    let given: Vec<i32> = (0..MODEL_CONTEXT_AND_TARGET_POSES).map(|i| (i < p + num_target_poses) as i32).collect();
    let local: Vec<i32> = (0..MODEL_CONTEXT_AND_TARGET_POSES).map(|i| if i == p - 1 || i == p + num_target_poses - 1 { 0 } else { given[i] }).collect();
    let bits = config.allowed_token_bits();
    let n = (MODEL_MAX_TOKEN_COUNT - MODEL_MIN_TOKEN_COUNT + 1) as usize;
    let allowed: Vec<i32> = (0..n).map(|i| ((bits >> i) & 1) as i32).collect();
    HashMap::from([
        ("has_global_root_values".to_string(), i32s(given.clone(), &[1, 8])),
        ("has_local_root_values".to_string(), i32s(local, &[1, 8])),
        ("has_local_poses".to_string(), i32s(given, &[1, 8])),
        ("allowed_pred_num_tokens".to_string(), i32s(allowed, &[1, n as i32])),
    ])
}

fn check_buffers(context_poses: usize, target_poses: usize, sample_rate: f32, capacity: usize, has_contacts: bool) -> Result<()> {
    if context_poses != MODEL_POSES_PER_TOKEN {
        bail!("context token must contain {MODEL_POSES_PER_TOKEN} poses");
    }
    if !(1..=MODEL_POSES_PER_TOKEN).contains(&target_poses) {
        bail!("target token must contain 1 to {MODEL_POSES_PER_TOKEN} poses");
    }
    if sample_rate != MODEL_SAMPLE_RATE {
        bail!("output must use a sample rate of {MODEL_SAMPLE_RATE} Hz");
    }
    if capacity < MODEL_MAX_VALID_FRAMES {
        bail!("output capacity must be at least {MODEL_MAX_VALID_FRAMES} frames");
    }
    if !has_contacts {
        bail!("output must have foot_contacts; create_output() allocates them");
    }
    Ok(())
}

fn frame_count(num_tokens: i32, frames_in_prediction: usize) -> (usize, usize) {
    let raw = ((num_tokens.max(0) as usize) * MODEL_POSES_PER_TOKEN).min(frames_in_prediction);
    (raw, raw.saturating_sub(2 * MODEL_BOUNDARY_FRAMES))
}

fn check_token_count(num_tokens: i32, config: &MotionBricksInferenceConfig) -> Result<usize> {
    if !config.allowed_token_counts.contains(&num_tokens) {
        bail!("model selected {num_tokens} tokens, outside the allowed set {:?}", config.allowed_token_counts);
    }
    valid_frame_count(num_tokens)
}

// --- SOMA --------------------------------------------------------------------------------------

/// Heading about +Y from the leg joints (training's `compute_heading_method="hips_pos"`).
fn heading_from_hip_positions(right: Vec3, left: Vec3) -> f32 {
    let across = Vec3::new(right.x() - left.x(), 0.0, right.z() - left.z());
    let length = across.length();
    if length <= 1.0e-6 {
        return 0.0;
    }
    let forward = Vec3::new(0.0, 1.0, 0.0).cross(across * (1.0 / length));
    forward.x().atan2(forward.z())
}

fn rotation_from_6d(x_raw: Vec3, y_raw: Vec3) -> Quat {
    let x = x_raw.normalize();
    let z = x.cross(y_raw).normalize();
    let y = z.cross(x);
    quat_from_cols(x, y, z)
}

/// SOMA engine: poses are one local transform per skeleton joint, root (`Hips`) first. The
/// skeleton may hold more joints than the model's 30 (fingers, ...): those keep their reference
/// transforms in generated motion.
pub struct MotionBricksSomaEngine<B: Backend> {
    pub backend: Option<B>,
    model_to_runtime: Transform,
    runtime_to_model: Transform,
    pub alignment: Transform,
    pub generated_frame_count: usize,
    pub timer: GenerationTimer,
    skeleton: Option<Rc<Skeleton>>,
    model_to_pose: Vec<usize>,
    pose_to_model: Vec<i32>,
    right_leg: usize,
    left_leg: usize,
}

impl<B: Backend> MotionBricksSomaEngine<B> {
    /// `backend`: a SOMA model, or `None` for an engine whose encode/decode stages work without
    /// inference. `model_to_runtime`: model space -> world space (see `io::model_to_runtime`).
    pub fn new(backend: Option<B>, model_to_runtime: Transform) -> Result<Self> {
        if let Some(b) = &backend
            && b.rig() != ModelRig::Soma
        {
            bail!("MotionBricksSomaEngine needs a SOMA model");
        }
        Ok(MotionBricksSomaEngine {
            backend,
            model_to_runtime,
            runtime_to_model: model_to_runtime.inverse(),
            alignment: Transform::IDENTITY,
            generated_frame_count: 0,
            timer: GenerationTimer::default(),
            skeleton: None,
            model_to_pose: Vec::new(),
            pose_to_model: Vec::new(),
            right_leg: 0,
            left_leg: 0,
        })
    }

    pub fn model_to_runtime(&self) -> Transform {
        self.model_to_runtime
    }

    /// A context or target token for poses of `skeleton`.
    pub fn create_input(&mut self, skeleton: &Rc<Skeleton>) -> Result<PoseToken> {
        self.bind(skeleton)?;
        Ok(PoseToken::new(skeleton.num_joints()))
    }

    /// An empty buffer with room for the longest generated motion (58 frames).
    pub fn create_output(&mut self, skeleton: &Rc<Skeleton>) -> Result<AnimationBuffer> {
        self.bind(skeleton)?;
        let poses = (0..MODEL_MAX_VALID_FRAMES).flat_map(|_| skeleton.reference_local_transforms.iter().copied()).collect();
        AnimationBuffer::new(skeleton.clone(), MODEL_SAMPLE_RATE, poses, Some(0), Some(vec![[0.0; FOOT_CONTACT_COUNT]; MODEL_MAX_VALID_FRAMES]))
    }

    fn bind(&mut self, skeleton: &Rc<Skeleton>) -> Result<()> {
        if self.skeleton.as_ref().is_some_and(|s| Rc::ptr_eq(s, skeleton)) {
            return Ok(());
        }
        let mapping: Vec<Option<usize>> = SOMA_JOINT_NAMES.iter().map(|n| skeleton.joint_index(n)).collect();
        let missing: Vec<&str> = SOMA_JOINT_NAMES.iter().zip(&mapping).filter(|(_, i)| i.is_none()).map(|(n, _)| *n).collect();
        if !missing.is_empty() {
            bail!("the skeleton is missing model joints: {}", missing.join(", "));
        }
        let model_to_pose: Vec<usize> = mapping.into_iter().map(Option::unwrap).collect();
        if model_to_pose[0] != 0 {
            bail!("the skeleton must store the root (Hips) at index zero");
        }
        let mut pose_to_model = vec![-1i32; skeleton.num_joints()];
        for (model_joint, &pose_joint) in model_to_pose.iter().enumerate() {
            let parent = skeleton.joint_parent(pose_joint);
            // Model joints outside the connected hierarchy keep their reference transforms.
            if parent == -1 || pose_to_model[parent as usize] >= 0 {
                pose_to_model[pose_joint] = model_joint as i32;
            }
        }
        self.right_leg = skeleton.joint_index(SOMA_HEADING_JOINTS[0]).unwrap();
        self.left_leg = skeleton.joint_index(SOMA_HEADING_JOINTS[1]).unwrap();
        self.model_to_pose = model_to_pose;
        self.pose_to_model = pose_to_model;
        self.skeleton = Some(skeleton.clone());
        Ok(())
    }

    fn prepare(&mut self, context: &PoseToken, target: &PoseToken, output: &AnimationBuffer) -> Result<()> {
        self.bind(&output.skeleton)?;
        let j = output.skeleton.num_joints();
        if context.num_joints != j || target.num_joints != j {
            bail!("context and target tokens must contain {j} joints per pose");
        }
        check_buffers(context.num_poses(), target.num_poses(), output.sample_rate, output.capacity(), output.foot_contacts.is_some())
    }

    /// Generate motion at 30 Hz into `output`. `output.num_frames` then gives the valid frames
    /// (18 to 58), `output.foot_contacts` the predicted contacts, and `timer` the stage times.
    pub fn generate_animation(&mut self, context: &PoseToken, target: &PoseToken, output: &mut AnimationBuffer, config: &MotionBricksInferenceConfig) -> Result<()> {
        config.validate()?;
        self.prepare(context, target, output)?;
        output.num_frames = 0;
        let t = Instant::now();
        let inputs = self.encode_inputs(context, target, config)?;
        self.timer.encode = ms(t);
        let t = Instant::now();
        let prediction = self.backend.as_mut().ok_or_else(|| anyhow!("this engine has no backend"))?.predict(&inputs)?;
        self.timer.inference = ms(t);
        let t = Instant::now();
        self.decode_prediction(&prediction, output)?;
        self.timer.decode = ms(t);
        let t = Instant::now();
        if config.correct_generated_diff {
            self.correct_endpoints(context, target, output);
        }
        self.timer.post_process = if config.correct_generated_diff { ms(t) } else { 0.0 };
        output.num_frames = check_token_count(prediction.num_tokens, config)?;
        Ok(())
    }

    /// Align the context and target, and return the model's ONNX-named inputs (batch 1).
    pub fn encode_inputs(&mut self, context: &PoseToken, target: &PoseToken, config: &MotionBricksInferenceConfig) -> Result<HashMap<String, Npy>> {
        let sk = self.skeleton.clone().ok_or_else(|| anyhow!("create the tokens with this engine first"))?;
        // Every pose composed into SOMA model space.
        let world: Vec<Vec<Transform>> = context.poses.iter().chain(&target.poses).map(|p| sk.compute_world_transforms(p, self.runtime_to_model)).collect();
        let last = MODEL_POSES_PER_TOKEN - 1;
        let hips = world[last][0].p;
        let current = heading_from_hip_positions(world[last][self.right_leg].p, world[last][self.left_leg].p);
        let desired = config.heading_alignment[0].atan2(config.heading_alignment[1]);
        self.alignment = Transform::new(Vec3::new(hips.x(), 0.0, hips.z()), quat_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), current - desired));
        let align = self.alignment;
        // Subtract before rotating to keep precision far from the origin.
        let aligned = |p: Vec3| quat_rotate_inv(align.q, p - align.p);
        let jm = SOMA_JOINT_NAMES.len();
        let (mut global_root, mut local_root, mut local_poses) = (Vec::new(), Vec::new(), Vec::new());
        let n = MODEL_CONTEXT_AND_TARGET_POSES;
        let heading_at = |f: usize| heading_from_hip_positions(aligned(world[f][self.right_leg].p), aligned(world[f][self.left_leg].p));
        for f in 0..n {
            let next = if f == last { f } else { (f + 1).min(n - 1) };
            let (now, after) = (aligned(world[f][0].p), aligned(world[next][0].p));
            let (h, h_next) = (heading_at(f), heading_at(next));
            let delta = (h_next - h).sin().atan2((h_next - h).cos());
            global_root.extend([now.x(), now.y(), now.z(), h.cos(), h.sin()]);
            local_root.extend([delta * MODEL_SAMPLE_RATE, (after.x() - now.x()) * MODEL_SAMPLE_RATE, (after.z() - now.z()) * MODEL_SAMPLE_RATE, now.y()]);
            let root = Vec3::new(now.x(), 0.0, now.z());
            let mut positions = Vec::with_capacity((jm - 1) * 3);
            let mut rotations = Vec::with_capacity(jm * 6);
            for (m, &pj) in self.model_to_pose.iter().enumerate() {
                let joint = world[f][pj];
                if m > 0 {
                    positions.extend((aligned(joint.p) - root).0);
                }
                let r = quat_mul(quat_inverse(align.q), joint.q);
                rotations.extend(quat_rotate(r, Vec3::new(1.0, 0.0, 0.0)).0);
                rotations.extend(quat_rotate(r, Vec3::new(0.0, 1.0, 0.0)).0);
            }
            local_poses.extend(positions);
            local_poses.extend(rotations);
        }
        let mut inputs = conditions(target.num_poses(), config);
        inputs.insert("global_root_values".into(), f32s(global_root, &[1, 8, 5]));
        inputs.insert("local_root_values".into(), f32s(local_root, &[1, 8, 4]));
        inputs.insert("local_poses".into(), f32s(local_poses, &[1, 8, ((jm - 1) * 3 + jm * 6) as i32]));
        Ok(inputs)
    }

    /// Decode the prediction into `output` (world space, boundary frames dropped); returns the frame count.
    pub fn decode_prediction(&mut self, prediction: &Prediction, output: &mut AnimationBuffer) -> Result<usize> {
        let sk = self.skeleton.clone().ok_or_else(|| anyhow!("create the tokens with this engine first"))?;
        let jm = SOMA_JOINT_NAMES.len();
        let dim = 5 + (jm - 1) * 3 + jm * 6 + jm * 3 + FOOT_CONTACT_COUNT; // 366
        let frames = prediction.motion.len() / dim;
        let (_, count) = frame_count(prediction.num_tokens, frames);
        self.generated_frame_count = count;
        let rot_offset = 5 + (jm - 1) * 3;
        if output.foot_contacts.is_none() {
            bail!("output has no foot_contacts");
        }
        for frame in 0..count {
            let row = &prediction.motion[(frame + MODEL_BOUNDARY_FRAMES) * dim..(frame + MODEL_BOUNDARY_FRAMES + 1) * dim];
            output.foot_contacts.as_mut().unwrap()[frame] = std::array::from_fn(|c| row[dim - FOOT_CONTACT_COUNT + c].clamp(0.0, 1.0));
            let rotation = |m: usize| {
                let o = rot_offset + m * 6;
                rotation_from_6d(Vec3::new(row[o], row[o + 1], row[o + 2]), Vec3::new(row[o + 3], row[o + 4], row[o + 5]))
            };
            let rotations: Vec<Quat> = (0..jm).map(rotation).collect();
            let hips = self.alignment.mul(Transform::new(Vec3::new(row[0], row[1], row[2]), rotations[0]));
            let pose = output.frame_mut(frame);
            pose[0] = self.model_to_runtime.mul(hips);
            for joint in 1..sk.num_joints() {
                let reference = sk.reference_local_transforms[joint];
                let m = self.pose_to_model[joint];
                pose[joint] = if m < 0 {
                    reference
                } else {
                    let parent_m = self.pose_to_model[sk.parent_indices[joint] as usize] as usize;
                    // The common alignment cancels in parent-to-child space.
                    Transform::new(reference.p, quat_normalize(quat_mul(quat_inverse(rotations[parent_m]), rotations[m as usize])))
                };
            }
        }
        Ok(count)
    }

    /// Match the first and last generated frames to the last context pose and the first target pose.
    pub fn correct_endpoints(&self, context: &PoseToken, target: &PoseToken, output: &mut AnimationBuffer) {
        let count = self.generated_frame_count;
        if count < 2 {
            return;
        }
        let last = count - 1;
        let (first, end) = (output.frame(0).to_vec(), output.frame(last).to_vec());
        let (ctx, tgt) = (&context.poses[MODEL_POSES_PER_TOKEN - 1], &target.poses[0]);
        let start_root = compute_root_endpoint_delta(first[0], ctx[0]);
        let end_root = compute_root_endpoint_delta(end[0], tgt[0]);
        let start: Vec<Transform> = (1..first.len()).map(|j| compute_delta_transform(first[j], ctx[j])).collect();
        let finish: Vec<Transform> = (1..end.len()).map(|j| compute_delta_transform(end[j], tgt[j])).collect();
        for frame in 0..count {
            let ew = frame as f32 / last as f32;
            let sw = 1.0 - ew;
            let pose = output.frame_mut(frame);
            pose[0] = apply_root_endpoint_correction(pose[0], start_root, end_root, ew);
            for j in 1..pose.len() {
                pose[j] = apply_delta_transform(apply_delta_transform(pose[j], start[j - 1], sw), finish[j - 1], ew);
            }
        }
    }
}

// --- Unitree G1 --------------------------------------------------------------------------------

/// G1 root position filter at 30 Hz: keeps the gait below 4 Hz, removes jitter above 8 Hz.
pub const ROOT_POSITION_FILTER: [f32; 7] = [-0.0465, 0.0639, 0.2827, 0.3998, 0.2827, 0.0639, -0.0465];
/// G1 joint-angle endpoint correction fades in and out over this time at each end.
pub const JOINT_ANGLE_FADE_SECONDS: f32 = 0.3;

/// G1 engine: poses are 36 joint coordinates (root XYZ, XYZW quaternion, 29 angles in radians).
pub struct MotionBricksUnitreeG1Engine<B: Backend> {
    pub backend: Option<B>,
    model_to_runtime: Transform,
    runtime_to_model: Transform,
    pub alignment: Transform,
    pub generated_frame_count: usize,
    pub timer: GenerationTimer,
}

impl<B: Backend> MotionBricksUnitreeG1Engine<B> {
    pub fn new(backend: Option<B>, model_to_runtime: Transform) -> Result<Self> {
        if let Some(b) = &backend
            && b.rig() != ModelRig::UnitreeG1
        {
            bail!("MotionBricksUnitreeG1Engine needs a Unitree G1 model");
        }
        Ok(MotionBricksUnitreeG1Engine {
            backend,
            model_to_runtime,
            runtime_to_model: model_to_runtime.inverse(),
            alignment: Transform::IDENTITY,
            generated_frame_count: 0,
            timer: GenerationTimer::default(),
        })
    }

    pub fn model_to_runtime(&self) -> Transform {
        self.model_to_runtime
    }

    pub fn create_input(&self) -> DofToken {
        DofToken::new(UNITREE_G1_JOINT_NAMES.len())
    }

    pub fn create_output(&self) -> DofAnimationBuffer {
        let nc = ROOT_COORDINATE_COUNT + UNITREE_G1_JOINT_NAMES.len();
        let mut poses = vec![0.0; MODEL_MAX_VALID_FRAMES * nc];
        for f in 0..MODEL_MAX_VALID_FRAMES {
            poses[f * nc + 6] = 1.0;
        }
        DofAnimationBuffer::new(UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect(), MODEL_SAMPLE_RATE, poses, Some(0),
                                Some(vec![[0.0; FOOT_CONTACT_COUNT]; MODEL_MAX_VALID_FRAMES]))
        .expect("valid buffer")
    }

    fn prepare(&self, context: &DofToken, target: &DofToken, output: &DofAnimationBuffer) -> Result<()> {
        let n = UNITREE_G1_JOINT_NAMES.len();
        if context.num_dofs != n || target.num_dofs != n || output.num_dofs() != n {
            bail!("tokens and output must contain {n} joint angles per pose");
        }
        check_buffers(context.num_poses(), target.num_poses(), output.sample_rate, output.capacity(), output.foot_contacts.is_some())
    }

    pub fn generate_animation(&mut self, context: &DofToken, target: &DofToken, output: &mut DofAnimationBuffer, config: &MotionBricksInferenceConfig) -> Result<()> {
        config.validate()?;
        self.prepare(context, target, output)?;
        output.num_frames = 0;
        let t = Instant::now();
        let inputs = self.encode_inputs(context, target, config);
        self.timer.encode = ms(t);
        let t = Instant::now();
        let prediction = self.backend.as_mut().ok_or_else(|| anyhow!("this engine has no backend"))?.predict(&inputs)?;
        self.timer.inference = ms(t);
        let t = Instant::now();
        self.decode_prediction(&prediction, output)?;
        self.timer.decode = ms(t);
        let t = Instant::now();
        if config.correct_generated_diff {
            self.correct_endpoints(context, target, output);
        }
        self.timer.post_process = if config.correct_generated_diff { ms(t) } else { 0.0 };
        output.num_frames = check_token_count(prediction.num_tokens, config)?;
        Ok(())
    }

    pub fn encode_inputs(&mut self, context: &DofToken, target: &DofToken, config: &MotionBricksInferenceConfig) -> HashMap<String, Npy> {
        let root = self.runtime_to_model.mul(read_root(&context.poses[MODEL_POSES_PER_TOKEN - 1]));
        let forward = root.vector(Vec3::new(1.0, 0.0, 0.0));
        let current = forward.y().atan2(forward.x());
        let desired = config.heading_alignment[1].atan2(config.heading_alignment[0]);
        self.alignment = Transform::new(Vec3::new(root.p.x(), root.p.y(), 0.0), quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), current - desired));
        let align = self.alignment;
        let mut inputs = conditions(target.num_poses(), config);
        for (token, name) in [(context, "context_qpos"), (target, "target_qpos")] {
            let mut qpos = Vec::with_capacity(4 * token.num_coordinates());
            for pose in &token.poses {
                let r = self.runtime_to_model.mul(read_root(pose));
                let position = quat_rotate_inv(align.q, r.p - align.p);
                let mut q = pose.clone();
                write_root(Transform::new(Vec3::new(position.x(), position.y(), r.p.z()), quat_mul(quat_inverse(align.q), r.q)), &mut q);
                qpos.extend(q);
            }
            inputs.insert(name.into(), f32s(qpos, &[1, 4, token.num_coordinates() as i32]));
        }
        inputs
    }

    pub fn decode_prediction(&mut self, prediction: &Prediction, output: &mut DofAnimationBuffer) -> Result<usize> {
        let nc = output.num_coordinates();
        let frames = prediction.motion.len() / nc;
        let (raw_count, count) = frame_count(prediction.num_tokens, frames);
        self.generated_frame_count = count;
        let contacts_in = prediction.foot_contacts.as_ref().ok_or_else(|| anyhow!("prediction has no foot contacts"))?;
        let root_at = |f: usize| read_root(&prediction.motion[f * nc..(f + 1) * nc]);
        let radius = ROOT_POSITION_FILTER.len() / 2;
        for frame in 0..count {
            let raw = frame + MODEL_BOUNDARY_FRAMES;
            let contacts: FootContacts = std::array::from_fn(|c| contacts_in[raw * FOOT_CONTACT_COUNT + c].clamp(0.0, 1.0));
            output.foot_contacts.as_mut().unwrap()[frame] = contacts;
            let mut position = Vec3::ZERO;
            for (tap, w) in ROOT_POSITION_FILTER.iter().enumerate() {
                // Boundary frames provide the neighbors at each end.
                let source = (raw as i64 + tap as i64 - radius as i64).clamp(0, raw_count as i64 - 1) as usize;
                position = position + root_at(source).p * *w;
            }
            let root = self.alignment.mul(Transform::new(position, root_at(raw).q));
            let root = self.model_to_runtime.mul(Transform::new(Vec3::new(root.p.x(), root.p.y(), position.z()), root.q));
            let out = output.frame_mut(frame);
            write_root(root, out);
            out[ROOT_COORDINATE_COUNT..].copy_from_slice(&prediction.motion[raw * nc + ROOT_COORDINATE_COUNT..(raw + 1) * nc]);
        }
        Ok(count)
    }

    /// Root corrections span the motion; joint-angle corrections fade over 0.3 s at each end.
    pub fn correct_endpoints(&self, context: &DofToken, target: &DofToken, output: &mut DofAnimationBuffer) {
        let count = self.generated_frame_count;
        if count < 2 {
            return;
        }
        let last = count - 1;
        let nc = output.num_coordinates();
        let mut deltas = [vec![0.0f32; nc], vec![0.0f32; nc]];
        for (row, frame, guide) in [(0, 0, &context.poses[MODEL_POSES_PER_TOKEN - 1]), (1, last, &target.poses[0])] {
            let generated = output.frame(frame).to_vec();
            write_root(compute_root_endpoint_delta(read_root(&generated), read_root(guide)), &mut deltas[row]);
            // Joint angle differences are scalar offsets, without wrapping at +-pi.
            for c in ROOT_COORDINATE_COUNT..nc {
                deltas[row][c] = guide[c] - generated[c];
            }
        }
        let (start_root, end_root) = (read_root(&deltas[0]), read_root(&deltas[1]));
        // Cap the fade so the start and end correction windows do not overlap.
        let fade = (JOINT_ANGLE_FADE_SECONDS * MODEL_SAMPLE_RATE).min(0.5 * last as f32);
        for frame in 0..count {
            let ew = frame as f32 / last as f32;
            let q = output.frame_mut(frame);
            write_root(apply_root_endpoint_correction(read_root(q), start_root, end_root, ew), q);
            let (sw_angle, ew_angle) = (quadratic_fade_weight(frame as f32 / fade), quadratic_fade_weight((last - frame) as f32 / fade));
            for c in ROOT_COORDINATE_COUNT..nc {
                q[c] += deltas[0][c] * sw_angle + deltas[1][c] * ew_angle;
            }
        }
    }
}
