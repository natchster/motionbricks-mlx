// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Runtime API (`motionbricks_rs::runtime`) against upstream's own runtime code: the counterpart of
//! `model/tests/test_runtime.py`, with the same three layers:
//! * weight-free unit tests of the transforms, tokens, recorder, buffers, settings and player timing;
//! * teacher-forced parity with `sandbox/golden/runtime` (recorded from upstream's Warp kernels);
//! * end-to-end generation on both backends (Metal runtime, mlx-rs) with the checkpoints.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::npy::{Npy, load_npz};
use motionbricks_rs::runtime::math::{quat_from_axis_angle, quat_mul, quat_rotate, quat_rotate_inv, quat_slerp, quadratic_fade_weight};
use motionbricks_rs::runtime::*;

const SCENARIOS: [&str; 4] = ["walk", "jog_one_pose", "no_correction", "model_to_runtime"];

fn runtime_golden(name: &str) -> Option<HashMap<String, Npy>> {
    let dir = std::env::var_os("MOTIONBRICKS_RUNTIME_GOLDEN").map(PathBuf::from).unwrap_or_else(|| golden_root().join("runtime"));
    let path = dir.join(format!("{name}.npz"));
    if !path.is_file() {
        skip(&format!("runtime fixture {} missing (made by the sandbox: sandbox/scripts/make_runtime_golden.py)", path.display()));
        return None;
    }
    Some(load_npz(&path).expect("runtime fixture loads"))
}

fn clips_root() -> PathBuf {
    if let Some(p) = std::env::var_os("MOTIONBRICKS_CLIPS") {
        return PathBuf::from(p);
    }
    let downloaded = root().join("checkpoints/motions");
    if downloaded.is_dir() { downloaded } else { root().join("sandbox/upstream/assets/motions") }
}

fn clip_path(name: &str, stem: &str) -> Option<PathBuf> {
    let ext = if name == "soma" { "bvh" } else { "csv" };
    let path = clips_root().join(name).join("locomotion/bs_1d_linear").join(format!("{stem}.{ext}"));
    if !path.is_file() || std::fs::metadata(&path).map(|m| m.len() < 1024).unwrap_or(true) {
        skip(&format!("clip {} missing (motionbricks-mlx download --only-clips, or scripts/download_weights.sh clips)", path.display()));
        return None;
    }
    Some(path)
}

fn transforms(v: &[f32]) -> Vec<Transform> {
    v.chunks(7).map(Transform::from_slice).collect()
}

fn soma_poses(n: &Npy) -> Vec<Vec<Transform>> {
    let joints = n.shape[n.shape.len() - 2] as usize;
    n.to_f32().chunks(joints * 7).map(transforms).collect()
}

fn dof_poses(n: &Npy) -> Vec<Vec<f32>> {
    let width = *n.shape.last().unwrap() as usize;
    n.to_f32().chunks(width).map(<[f32]>::to_vec).collect()
}

/// Max error between transforms and flat `[.., 7]` values, treating q and -q as the same rotation.
fn transform_error(got: &[Transform], want: &[f32]) -> f32 {
    assert_eq!(got.len() * 7, want.len(), "length mismatch");
    let mut err = 0f32;
    for (t, w) in got.iter().zip(want.chunks(7)) {
        let a = t.to_array();
        let sign = if (3..7).map(|k| a[k] * w[k]).sum::<f32>() < 0.0 { -1.0 } else { 1.0 };
        for k in 0..7 {
            err = err.max((if k >= 3 { a[k] * sign } else { a[k] } - w[k]).abs());
        }
    }
    err
}

/// The same for joint coordinates (root quaternion sign-insensitive, angles exact).
fn dof_error(got: &[f32], want: &[f32], width: usize) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut err = 0f32;
    for (a, w) in got.chunks(width).zip(want.chunks(width)) {
        err = err.max(transform_error(&[Transform::from_slice(&a[..7])], &w[..7]));
        err = err.max(a[7..].iter().zip(&w[7..]).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
    }
    err
}

fn skeleton(g: &HashMap<String, Npy>) -> Arc<Skeleton> {
    let bytes: Vec<u8> = g["skeleton.names_utf8"].to_i32().iter().map(|&c| c as u8).collect();
    let names = String::from_utf8(bytes).unwrap().split('\n').map(str::to_string).collect();
    Arc::new(Skeleton::new(names, g["skeleton.parents"].to_i32(), transforms(&g["skeleton.reference"].to_f32())).unwrap())
}

fn config(g: &HashMap<String, Npy>, label: &str) -> MotionBricksInferenceConfig {
    let h = g[&format!("gen.{label}.heading")].to_f32();
    MotionBricksInferenceConfig {
        allowed_token_counts: g[&format!("gen.{label}.allowed")].to_i32(),
        correct_generated_diff: g[&format!("gen.{label}.correct")].to_i32()[0] != 0,
        heading_alignment: [h[0], h[1]],
    }
}

fn prediction(g: &HashMap<String, Npy>, label: &str, name: &str) -> Prediction {
    let k = |s: &str| format!("gen.{label}.out.{s}");
    Prediction {
        motion: g[&k(if name == "soma" { "pred_global_motions" } else { "pred_qpos" })].to_f32(),
        num_tokens: g[&k("pred_num_tokens")].to_i32()[0],
        foot_contacts: g.get(&k("pred_foot_contacts")).map(Npy::to_f32),
    }
}

// --- weight-free -------------------------------------------------------------------------------

#[test]
fn transform_algebra() {
    let q = math::quat_normalize([0.3, -0.5, 0.2, 0.7]);
    let p = Vec3::new(0.4, -1.0, 2.0);
    let back = quat_rotate_inv(q, quat_rotate(q, p));
    assert!((back - p).length() < 1e-5);
    let half = quat_slerp(math::QUAT_IDENTITY, q, 0.5);
    let full = quat_mul(half, half);
    assert!((full.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>().abs() - 1.0).abs() < 1e-5);
    let fades: Vec<f32> = [0.0, 0.25, 0.5, 1.0].iter().map(|&v| quadratic_fade_weight(v)).collect();
    assert_eq!(fades, vec![1.0, 0.875, 0.5, 0.0]);
    let r = quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
    assert!((quat_rotate(r, Vec3::new(1.0, 0.0, 0.0)) - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
}

#[test]
fn valid_frame_count_and_config() {
    assert_eq!([6, 10, 16].map(|n| valid_frame_count(n).unwrap()), [18, 34, 58]);
    assert!(valid_frame_count(5).is_err());
    assert_eq!(MotionBricksInferenceConfig::default().allowed_token_bits(), 0b1111111 << 2);
    for bad in [
        MotionBricksInferenceConfig { allowed_token_counts: vec![], ..Default::default() },
        MotionBricksInferenceConfig { allowed_token_counts: vec![5], ..Default::default() },
        MotionBricksInferenceConfig { heading_alignment: [0.0, 0.0], ..Default::default() },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn tokens_pad_with_the_last_pose() {
    let mut token = DofToken::new(29);
    let poses: Vec<Vec<f32>> = (0..2).map(|k| vec![k as f32 + 0.5; 36]).collect();
    token.update_from_poses(&poses).unwrap();
    assert_eq!(token.num_poses(), 2);
    assert_eq!(token.poses()[2], poses[1]);
    assert_eq!(token.poses()[3], poses[1]);
    token.update_from_poses(&poses[..1]).unwrap();
    assert!(token.num_poses() == 1 && token.poses().iter().all(|p| *p == poses[0]));
    assert!(token.update_from_poses(&vec![vec![0.0; 36]; 5]).is_err());
}

#[test]
fn recorder_and_resampling() {
    let mut history = MotionRecorder::new(&[0.0f32; 36], 4).unwrap();
    for k in 1..=5 {
        history.record(1.0 / 30.0, &[k as f32; 36]).unwrap();
    }
    assert_eq!(history.poses().iter().map(|p| p[10]).collect::<Vec<_>>(), vec![2.0, 3.0, 4.0, 5.0]);
    assert_eq!(*history.delta_times().last().unwrap(), 0.0);
    let mut dts = vec![1.0 / 60.0; 7];
    dts.push(0.0);
    let samples = history_sample_indices(&dts, 4, 30.0);
    // Poses every 1/60 s; samples every 1/30 s ending at the latest pose (blend 1 toward it).
    assert_eq!(samples.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>(), vec![(1, 2), (3, 4), (5, 6), (6, 7)]);
    assert!(samples.iter().map(|s| s.2).zip([0.0, 0.0, 0.0, 1.0]).all(|(a, b)| (a - b).abs() < 1e-5));
    assert!(history.record(0.0, &[0.0; 36]).is_err());
    assert!(history.record(1.0, &[0.0; 35]).is_err());
    assert!(history_sample_indices(&[], 4, 30.0).is_empty());
}

#[test]
fn buffer_sampling_clamps_and_blends() {
    let mut poses = vec![0.0f32; 3 * 36];
    for f in 0..3 {
        poses[f * 36 + 6] = 1.0;
        poses[f * 36 + 10] = f as f32;
    }
    let contacts = vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let names = UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
    let buffer = DofAnimationBuffer::new(names, 30.0, poses, None, Some(contacts)).unwrap();
    assert!((buffer.sample(0.5 / 30.0, Transform::IDENTITY).unwrap()[10] - 0.5).abs() < 1e-6);
    assert_eq!(buffer.sample(99.0, Transform::IDENTITY).unwrap()[10], 2.0);
    assert_eq!(buffer.sample(-1.0, Transform::IDENTITY).unwrap()[10], 0.0);
    let c = buffer.sample_foot_contacts(1.5 / 30.0).unwrap();
    assert!(c.iter().zip([0.0, 0.5, 0.5, 0.0]).all(|(a, b)| (a - b).abs() < 1e-6));
    assert!((buffer.duration() - 2.0 / 30.0).abs() < 1e-7);
}

/// A G1 backend that always predicts `tokens` tokens of a still pose.
struct FixedBackend {
    tokens: i32,
}

impl Backend for FixedBackend {
    fn rig(&self) -> ModelRig {
        ModelRig::UnitreeG1
    }
    fn predict(&mut self, inputs: &HashMap<String, Npy>) -> anyhow::Result<Prediction> {
        assert!(inputs.contains_key("context_qpos") && inputs.contains_key("allowed_pred_num_tokens"));
        let mut motion = vec![0.0; 64 * 36];
        for f in 0..64 {
            motion[f * 36 + 6] = 1.0;
        }
        Ok(Prediction { motion, num_tokens: self.tokens, foot_contacts: Some(vec![0.5; 64 * 4]) })
    }
}

#[test]
fn player_timing_and_overrides() {
    let engine = MotionBricksUnitreeG1Engine::new(Some(FixedBackend { tokens: 9 }), Transform::IDENTITY).unwrap();
    let mut reference = vec![0.0f32; 36];
    reference[6] = 1.0;
    let mut player = GeneratedDofMotionPlayer::new(engine, &reference, 32).unwrap();
    player.seed_history_from_pose(&reference).unwrap();
    let params = TargetPoseInferenceParams { allowed_token_counts: Some(vec![9]), ..TargetPoseInferenceParams::new(10, 2) };
    player.generate(&vec![reference.clone(); 4], &params, None).unwrap();
    assert_eq!(player.target().num_poses(), 2);
    assert_eq!(player.output().num_frames(), 30);
    assert!((player.time_to_transition() - 19.0 / 30.0).abs() < 1e-6);
    player.sample(0.5).unwrap();
    assert!((player.time_to_transition() - (19.0 / 30.0 - 0.5)).abs() < 1e-6);
    player.sample(1.0).unwrap();
    assert_eq!(player.time_to_transition(), 0.0);
    // A token count outside the allowed set is refused, as upstream does; the playing motion stays.
    let refused = TargetPoseInferenceParams { allowed_token_counts: Some(vec![10]), ..TargetPoseInferenceParams::new(0, 1) };
    assert!(player.generate(&vec![reference.clone(); 4], &refused, None).is_err());
    assert_eq!((player.output().num_frames(), player.transition_early_by_frames()), (30, 10));
    assert!((player.playback_time() - 1.5).abs() < 1e-9);
    let mut pose = vec![0.0; 36];
    let contacts = player.sample_into(0.0, &mut pose).unwrap();
    assert_eq!((pose[6], contacts), (1.0, [0.5; 4]));
    assert!(player.sample_into(0.0, &mut [0.0; 35]).is_err());
}

/// The model's 30 SOMA joints in their hierarchy, with simple offsets (no fixtures needed).
fn synthetic_soma_skeleton() -> Arc<Skeleton> {
    let parent_names = [
        "", "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Head", "Head", "Chest", "LeftShoulder", "LeftArm", "LeftForeArm",
        "LeftHand", "LeftHand", "Chest", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand", "Hips", "LeftLeg", "LeftShin",
        "LeftFoot", "Hips", "RightLeg", "RightShin", "RightFoot",
    ];
    let names: Vec<String> = SOMA_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
    let parents: Vec<i32> = parent_names.iter().map(|p| SOMA_JOINT_NAMES.iter().position(|n| n == p).map_or(-1, |i| i as i32)).collect();
    let offsets: Vec<Transform> = names
        .iter()
        .enumerate()
        .map(|(j, n)| {
            let side = if n.starts_with("Left") { 0.1 } else if n.starts_with("Right") { -0.1 } else { 0.0 };
            let down = if n.ends_with("Leg") || n.ends_with("Shin") || n.ends_with("Foot") { -0.4 } else { 0.1 };
            let q = math::quat_normalize([0.02 * j as f32, 0.0, 0.01, 1.0]);
            Transform::new(if j == 0 { Vec3::new(0.0, 0.9, 0.0) } else { Vec3::new(side, down, 0.03) }, q)
        })
        .collect();
    Arc::new(Skeleton::new(names, parents, offsets).unwrap())
}

#[test]
fn soma_player_samples_records_and_keeps_motion_on_failure() {
    let sk = synthetic_soma_skeleton();
    let engine = MotionBricksSomaEngine::new(Some(StillBackend { rig: ModelRig::Soma, tokens: 8 }), model_to_runtime(ModelRig::Soma)).unwrap();
    let mut player = GeneratedMotionPlayer::new(engine, &sk, 16).unwrap();
    let start = sk.reference_local_transforms().to_vec();
    player.seed_history_from_pose(&start).unwrap();
    player.generate(&vec![start.clone(); 4], &TargetPoseInferenceParams::new(4, 4), None).unwrap();
    assert_eq!(player.output().num_frames(), 26);
    assert!((player.time_to_transition() - 21.0 / 30.0).abs() < 1e-9);
    let mut pose = start.clone();
    let contacts = player.sample_into(0.25, &mut pose).unwrap();
    let (allocated, same_contacts) = player.sample(0.0).unwrap();
    assert!(pose == allocated && contacts == same_contacts && contacts == [0.5; 4]);
    let mut corrector = TargetPoseIkCorrector::new(&sk, create_soma_target_pose_ik_rig(&sk).unwrap()).unwrap();
    corrector.capture_pins(&start, ik_pin::ALL).unwrap();
    corrector.apply(&mut pose, 0.7).unwrap();
    player.record_pose(0.25, &pose).unwrap();
    assert_eq!(player.motion_history().poses().last().unwrap(), &pose);
    // A refused generation (8 tokens, only 9 allowed) keeps the playing motion and its clock.
    let root = player.current_root_position();
    let refused = TargetPoseInferenceParams { allowed_token_counts: Some(vec![9]), ..TargetPoseInferenceParams::new(0, 1) };
    assert!(player.generate(&vec![start.clone(); 4], &refused, None).is_err());
    assert_eq!((player.output().num_frames(), player.transition_early_by_frames(), player.playback_time()), (26, 4, 0.25));
    assert_eq!(player.current_root_position(), root);
}

#[test]
fn shared_backend_and_engines_without_backend() {
    let shared = std::rc::Rc::new(std::cell::RefCell::new(FixedBackend { tokens: 9 }));
    let mut reference = vec![0.0f32; 36];
    reference[6] = 1.0;
    let mut players: Vec<_> = (0..2)
        .map(|_| {
            let engine = MotionBricksUnitreeG1Engine::new(Some(shared.clone()), Transform::IDENTITY).unwrap();
            GeneratedDofMotionPlayer::new(engine, &reference, 8).unwrap()
        })
        .collect();
    let params = TargetPoseInferenceParams { allowed_token_counts: Some(vec![9]), ..TargetPoseInferenceParams::new(0, 4) };
    for player in &mut players {
        player.generate(&[reference.clone()], &params, None).unwrap();
        assert_eq!(player.output().num_frames(), 30);
    }
    assert_eq!(std::rc::Rc::strong_count(&shared), 3);
    // A SOMA backend is refused by the G1 engine; an engine without a backend cannot generate.
    assert!(MotionBricksUnitreeG1Engine::new(Some(StillBackend { rig: ModelRig::Soma, tokens: 8 }), Transform::IDENTITY).is_err());
    let mut e = MotionBricksUnitreeG1Engine::without_backend(Transform::IDENTITY);
    let (mut c, mut out) = (e.create_input(), e.create_output());
    c.update_from_poses(&vec![reference.clone(); 4]).unwrap();
    let err = e.generate_animation(&c, &c, &mut out, &MotionBricksInferenceConfig::default()).unwrap_err();
    assert!(err.to_string().contains("no backend"), "{err}");
    // Encoding validates the config instead of building a wrong token mask.
    let bad = MotionBricksInferenceConfig { allowed_token_counts: vec![40], ..Default::default() };
    assert!(e.encode_inputs(&c, &c, &bad).is_err());
    assert_eq!(bad.allowed_token_bits(), 0);
}

#[test]
fn malformed_predictions_are_errors() {
    let mut e = MotionBricksUnitreeG1Engine::without_backend(Transform::IDENTITY);
    let mut out = e.create_output();
    let still = |tokens| StillBackend { rig: ModelRig::UnitreeG1, tokens }.predict(&HashMap::new()).unwrap();
    assert_eq!(e.decode_prediction(&still(9), &mut out).unwrap(), 30);
    let bad = [
        Prediction { foot_contacts: None, ..still(9) },
        Prediction { foot_contacts: Some(vec![0.5; 10]), ..still(9) },
        Prediction { motion: vec![0.0; 64 * 36 - 1], ..still(9) },
        Prediction { motion: vec![0.0; 20 * 36], foot_contacts: Some(vec![0.5; 20 * 4]), ..still(9) },
    ];
    for p in &bad {
        assert!(e.decode_prediction(p, &mut out).is_err());
        assert_eq!(e.generated_frame_count(), 0);
    }
    let names = UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
    let mut no_contacts = DofAnimationBuffer::new(names, 30.0, vec![0.0; 58 * 36], Some(0), None).unwrap();
    assert!(e.decode_prediction(&still(9), &mut no_contacts).is_err());

    let sk = synthetic_soma_skeleton();
    let mut e = MotionBricksSomaEngine::without_backend(model_to_runtime(ModelRig::Soma));
    let mut out = e.create_output(&sk).unwrap();
    let soma = StillBackend { rig: ModelRig::Soma, tokens: 9 }.predict(&HashMap::new()).unwrap();
    assert_eq!(e.decode_prediction(&soma, &mut out).unwrap(), 30);
    let short = Prediction { motion: soma.motion[..20 * 366].to_vec(), ..soma.clone() };
    assert!(e.decode_prediction(&short, &mut out).is_err());
    assert!(e.decode_prediction(&Prediction { motion: vec![0.0; 365], ..soma }, &mut out).is_err());
}

#[test]
fn ik_survives_degenerate_and_nan_poses() {
    let sk = synthetic_soma_skeleton();
    let mut corrector = TargetPoseIkCorrector::new(&sk, create_soma_target_pose_ik_rig(&sk).unwrap()).unwrap();
    let reference = sk.reference_local_transforms().to_vec();
    corrector.capture_pins(&reference, ik_pin::ALL).unwrap();
    let mut collapsed = reference.clone();
    for name in ["LeftForeArm", "LeftHand", "RightForeArm", "RightHand", "LeftShin", "LeftFoot"] {
        collapsed[sk.joint_index(name).unwrap()].p = Vec3::ZERO;
    }
    corrector.apply(&mut collapsed, 1.0).unwrap();
    let mut nan = reference.clone();
    nan[0].p = Vec3::new(f32::NAN, 0.0, 0.0);
    corrector.apply(&mut nan, 0.5).unwrap();
    // Chains are checked against the skeleton.
    let chain = TwoBoneIkChainConfig { pin: ik_pin::LEFT_HAND, root_index: 99, mid_index: 1, tip_index: 2, hint_offset_local: Vec3::ZERO };
    assert!(TargetPoseIkCorrector::new(&sk, vec![chain.clone()]).is_err());
    assert!(TargetPoseIkCorrector::new(&sk, vec![TwoBoneIkChainConfig { pin: ik_pin::ALL, root_index: 11, mid_index: 12, tip_index: 13, ..chain }]).is_err());
}

#[test]
fn skeletons_and_buffers_validate_their_layout() {
    let t = vec![Transform::IDENTITY; 3];
    let names = || ["a", "b", "c"].map(String::from).to_vec();
    assert!(Skeleton::new(names(), vec![-1, 0, 1], t.clone()).is_ok());
    for parents in [vec![-1, 2, 0], vec![0, 0, 1], vec![-1, -1, 0], vec![-1, 1, 0]] {
        let err = Skeleton::new(names(), parents.clone(), t.clone()).unwrap_err().to_string();
        assert!(err.contains("joint"), "{parents:?}: {err}");
    }
    let sk = Arc::new(Skeleton::new(names(), vec![-1, 0, 1], t.clone()).unwrap());
    let mut clip = AnimationBuffer::new(sk.clone(), 30.0, [t.clone(), t.clone()].concat(), Some(1), None).unwrap();
    assert!(clip.set_num_frames(3).is_err() && clip.num_frames() == 1);
    clip.set_num_frames(2).unwrap();
    assert!(clip.sample(f32::NAN, Transform::IDENTITY).is_err());
    assert!(clip.sample_into(0.0, Transform::IDENTITY, &mut [Transform::IDENTITY; 2]).is_err());
    for rate in [f32::NAN, f32::INFINITY, 0.0] {
        assert!(AnimationBuffer::new(sk.clone(), rate, t.clone(), None, None).is_err());
    }
    let q = Transform::new(Vec3::new(1.0, 2.0, 3.0), math::quat_normalize([0.1, 0.2, 0.3, 0.9]));
    assert!((q * q.inverse()).p.length() < 1e-5);
}

fn write_file(dir: &TempDir, name: &str, text: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, text).unwrap();
    path
}

const SMALL_BVH: &str = "HIERARCHY
ROOT Hips
{
  OFFSET 0 90 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 4 Zrotation Xrotation Yrotation Custom
    End Site
    {
      OFFSET 0 5 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
0 90 0 0 0 0 0 0 0 7
1 90 0 90 0 0 0 0 0 7
";

#[test]
fn bvh_loading_rejects_malformed_files() {
    let dir = temp_dir("runtime-bvh");
    let path = write_file(&dir, "ok.bvh", SMALL_BVH);
    let clip = load_bvh(&path, None, true).unwrap();
    assert_eq!((clip.num_frames(), clip.sample_rate(), clip.skeleton().num_joints()), (2, 30.0, 2));
    assert!((clip.frame(1)[0].p.x() - 0.01).abs() < 1e-7 && (clip.frame(1)[0].q[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    let broken = [
        ("OFFSET 0 10 0", "OFFSET 0 10"),
        ("CHANNELS 4 Zrotation Xrotation Yrotation Custom", "CHANNELS"),
        ("CHANNELS 4 Zrotation Xrotation Yrotation Custom", "CHANNELS 3 \u{c9}rotation Xrotation Yrotation"),
        ("CHANNELS 4 Zrotation", "CHANNELS 5 Zrotation"),
        ("0 0 0 0 0 7\n1", "0 0 0 0 0\n1"),
        ("0 0 0 0 0 7\n1", "0 0 0 0 x 7\n1"),
        ("Frame Time: 0.0333333", "Frame Time: 0"),
        ("Frame Time: 0.0333333", "Frame Time: nan"),
    ];
    for (k, (from, to)) in broken.iter().enumerate() {
        assert!(SMALL_BVH.contains(from), "{from}");
        let path = write_file(&dir, &format!("broken{k}.bvh"), &SMALL_BVH.replacen(from, to, 1));
        assert!(load_bvh(&path, None, true).is_err(), "{to:?} should be refused");
    }
    let no_channels: String = SMALL_BVH.lines().filter(|l| !l.contains("CHANNELS")).map(|l| l.replace("Frames: 2", "Frames: 99999999999999") + "\n").collect();
    assert!(load_bvh(write_file(&dir, "nochannels.bvh", &no_channels), None, true).is_err());
}

#[test]
fn csv_round_trip_and_malformed_files() {
    let mut names: Vec<String> = UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
    names[3] = "knee, left".into(); // quoted on save, unquoted on load
    let mut poses = vec![0.0f32; 2 * 36];
    for f in 0..2 {
        let root = Transform::new(Vec3::new(0.1 * f as f32, -0.2, 0.75), math::quat_normalize([0.1, -0.2, 0.3 + f as f32, 0.9]));
        write_root(root, &mut poses[f * 36..]);
        for c in 7..36 {
            poses[f * 36 + c] = 0.01 * c as f32 - 0.1 * f as f32;
        }
    }
    let clip = DofAnimationBuffer::new(names.clone(), 120.0, poses, None, None).unwrap();
    let dir = temp_dir("runtime-csv");
    let path = dir.path().join("round.csv");
    save_csv(&path, &clip).unwrap();
    let back = load_csv(&path, CSV_SAMPLE_RATE).unwrap();
    assert_eq!(back.joint_names(), names.as_slice());
    assert!(dof_error(back.poses(), clip.poses(), 36) < 1e-5);
    let header = format!("Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,{}", UNITREE_G1_JOINT_NAMES.join(","));
    let row = |v: &str| format!("0,{}", vec![v; 35].join(","));
    for (k, text) in [
        format!("Frame,x,y\n{}", row("1")),
        format!("{header}\n{}", row("1") + ",1"),
        format!("{header}\n{}", row("nan")),
        format!("{header}\n{}", row("one")),
        header.clone(),
    ]
    .iter()
    .enumerate()
    {
        assert!(load_csv(write_file(&dir, &format!("bad{k}.csv"), text), CSV_SAMPLE_RATE).is_err(), "case {k}");
    }
    // Accepted formatting: a UTF-8 BOM, a quoted header field, padded fields, blank and whitespace lines.
    let quoted = header.replacen("root_translateX", "\"root_translateX\"", 1);
    let text = format!("\u{feff}{quoted}\n\n{}\n   \n{}\n", row(" 1 "), row("2"));
    let loose = load_csv(write_file(&dir, "loose.csv", &text), CSV_SAMPLE_RATE).unwrap();
    assert_eq!(loose.num_frames(), 2);
    assert!(loose.joint_names().iter().eq(UNITREE_G1_JOINT_NAMES.iter()));
}

/// A backend of either rig that predicts `tokens` tokens of a still pose (no weights needed).
struct StillBackend {
    rig: ModelRig,
    tokens: i32,
}

impl Backend for StillBackend {
    fn rig(&self) -> ModelRig {
        self.rig
    }
    fn predict(&mut self, _inputs: &HashMap<String, Npy>) -> anyhow::Result<Prediction> {
        if self.rig == ModelRig::Soma {
            let (jm, dim) = (SOMA_JOINT_NAMES.len(), 366);
            let mut motion = vec![0.0; 64 * dim];
            for f in 0..64 {
                let row = &mut motion[f * dim..(f + 1) * dim];
                row[1] = 0.9;
                for m in 0..jm {
                    let o = 5 + (jm - 1) * 3 + m * 6;
                    row[o] = 1.0;
                    row[o + 4] = 1.0;
                }
                row[dim - 4..].fill(0.5);
            }
            Ok(Prediction { motion, num_tokens: self.tokens, foot_contacts: None })
        } else {
            let mut motion = vec![0.0; 64 * 36];
            for f in 0..64 {
                motion[f * 36 + 6] = 1.0;
            }
            Ok(Prediction { motion, num_tokens: self.tokens, foot_contacts: Some(vec![0.5; 64 * 4]) })
        }
    }
}

/// CPU cost of the per-frame player path (sample, IK, record; with `sample` and with `sample_into`)
/// and of a generation's encode/decode/correction with a still backend, the best of five rounds.
/// `cargo test --release --test runtime bench -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_player_paths() {
    let (Some(gs), Some(gg)) = (runtime_golden("soma"), runtime_golden("unitree_g1")) else { return };
    let (frames, dt) = (20_000, 1.0f32 / 60.0);
    let params = TargetPoseInferenceParams::new(4, 4);
    let sk = skeleton(&gs);
    let soma_target = soma_poses(&gs["gen.walk.target"]);
    let g1_target = dof_poses(&gg["gen.walk.target"]);
    let best = |a: (f64, f64), b: (f64, f64)| (a.0.min(b.0), a.1.min(b.1));
    let (mut soma_alloc, mut soma_into, mut g1_alloc, mut g1_into) = ((9e9, 9e9), (9e9, 9e9), (9e9, 9e9), (9e9, 9e9));
    for _ in 0..5 {
        for into in [false, true] {
            let engine = MotionBricksSomaEngine::new(Some(StillBackend { rig: ModelRig::Soma, tokens: 14 }), model_to_runtime(ModelRig::Soma)).unwrap();
            let mut player = GeneratedMotionPlayer::new(engine, &sk, 32).unwrap();
            player.seed_history_from_pose(&soma_target[0]).unwrap();
            let mut corrector = TargetPoseIkCorrector::new(&sk, create_soma_target_pose_ik_rig(&sk).unwrap()).unwrap();
            corrector.capture_pins(&soma_target[0], ik_pin::ALL).unwrap();
            let (mut gen_time, mut gens, mut frame_time) = (0.0f64, 0, 0.0f64);
            let mut pose = sk.reference_local_transforms().to_vec();
            for _ in 0..frames {
                if player.time_to_transition() <= dt as f64 {
                    let s = std::time::Instant::now();
                    player.generate(&soma_target, &params, None).unwrap();
                    gen_time += s.elapsed().as_secs_f64();
                    gens += 1;
                }
                let s = std::time::Instant::now();
                if into {
                    player.sample_into(dt, &mut pose).unwrap();
                } else {
                    pose = player.sample(dt).unwrap().0;
                }
                corrector.apply(&mut pose, 0.5).unwrap();
                player.record_pose(dt, &pose).unwrap();
                frame_time += s.elapsed().as_secs_f64();
            }
            let t = (frame_time / frames as f64 * 1e6, gen_time / gens as f64 * 1e6);
            if into { soma_into = best(soma_into, t) } else { soma_alloc = best(soma_alloc, t) }

            let engine = MotionBricksUnitreeG1Engine::new(Some(StillBackend { rig: ModelRig::UnitreeG1, tokens: 14 }), Transform::IDENTITY).unwrap();
            let mut player = GeneratedDofMotionPlayer::new(engine, &g1_target[0], 32).unwrap();
            player.seed_history_from_pose(&g1_target[0]).unwrap();
            let (mut gen_time, mut gens, mut frame_time) = (0.0f64, 0, 0.0f64);
            let mut pose = g1_target[0].clone();
            for _ in 0..frames {
                if player.time_to_transition() <= dt as f64 {
                    let s = std::time::Instant::now();
                    player.generate(&g1_target, &params, None).unwrap();
                    gen_time += s.elapsed().as_secs_f64();
                    gens += 1;
                }
                let s = std::time::Instant::now();
                if into {
                    player.sample_into(dt, &mut pose).unwrap();
                } else {
                    pose = player.sample(dt).unwrap().0;
                }
                player.record_pose(dt, &pose).unwrap();
                frame_time += s.elapsed().as_secs_f64();
            }
            let t = (frame_time / frames as f64 * 1e6, gen_time / gens as f64 * 1e6);
            if into { g1_into = best(g1_into, t) } else { g1_alloc = best(g1_alloc, t) }
        }
    }
    for (label, (frame, generation)) in [("soma sample", soma_alloc), ("soma sample_into", soma_into), ("g1 sample", g1_alloc), ("g1 sample_into", g1_into)] {
        println!("{label}: frame {frame:.3} us, generation (still backend) {generation:.1} us");
    }
}

// --- teacher-forced parity with upstream's runtime ----------------------------------------------

fn encode_matches_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let mut f = Failures::default();
    for label in SCENARIOS {
        let m2r = Transform::from_slice(&g[&format!("gen.{label}.model_to_runtime")].to_f32());
        let cfg = config(&g, label);
        let key = |s: &str| format!("gen.{label}.{s}");
        let (inputs, alignment) = if name == "soma" {
            let sk = skeleton(&g);
            let mut e = MotionBricksSomaEngine::without_backend(m2r);
            let (mut c, mut t) = (e.create_input(&sk).unwrap(), e.create_input(&sk).unwrap());
            c.update_from_poses(&soma_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&soma_poses(&g[&key("target")])).unwrap();
            (e.encode_inputs(&c, &t, &cfg).unwrap(), e.alignment())
        } else {
            let mut e = MotionBricksUnitreeG1Engine::without_backend(m2r);
            let (mut c, mut t) = (e.create_input(), e.create_input());
            c.update_from_poses(&dof_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&dof_poses(&g[&key("target")])).unwrap();
            (e.encode_inputs(&c, &t, &cfg).unwrap(), e.alignment())
        };
        let e = transform_error(&[alignment], &g[&key("alignment")].to_f32());
        f.check(e < 1e-5, || format!("{label}: alignment error {e}"));
        for (k, v) in &inputs {
            let want = &g[&key(&format!("in.{k}"))];
            f.check(v.shape == want.shape, || format!("{label}/{k}: shape {:?} != {:?}", v.shape, want.shape));
            let e = if name == "unitree_g1" && k.ends_with("qpos") {
                dof_error(&v.to_f32(), &want.to_f32(), 36)
            } else {
                max_abs(&v.to_f32(), &want.to_f32(), |_| true)
            };
            f.check(e < 2e-5, || format!("{label}/{k}: max error {e}"));
        }
    }
    f.finish();
}

fn decode_and_correction_match_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let mut f = Failures::default();
    for label in SCENARIOS {
        let m2r = Transform::from_slice(&g[&format!("gen.{label}.model_to_runtime")].to_f32());
        let cfg = config(&g, label);
        let key = |s: &str| format!("gen.{label}.{s}");
        let want_count = g[&key("frame_count")].to_i32()[0] as usize;
        let pred = prediction(&g, label, name);
        let (decoded, generated, contacts, count) = if name == "soma" {
            let sk = skeleton(&g);
            let mut e = MotionBricksSomaEngine::without_backend(m2r);
            let (mut c, mut t, mut out) = (e.create_input(&sk).unwrap(), e.create_input(&sk).unwrap(), e.create_output(&sk).unwrap());
            c.update_from_poses(&soma_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&soma_poses(&g[&key("target")])).unwrap();
            e.encode_inputs(&c, &t, &cfg).unwrap();
            let count = e.decode_prediction(&pred, &mut out).unwrap();
            let j = sk.num_joints();
            let decoded = out.poses()[..count * j].to_vec();
            if cfg.correct_generated_diff {
                e.correct_endpoints(&c, &t, &mut out).unwrap();
            }
            (transform_error(&decoded, &g[&key("decoded")].to_f32()), transform_error(&out.poses()[..count * j], &g[&key("generated")].to_f32()), out.foot_contacts().unwrap().to_vec(), count)
        } else {
            let mut e = MotionBricksUnitreeG1Engine::without_backend(m2r);
            let (mut c, mut t, mut out) = (e.create_input(), e.create_input(), e.create_output());
            c.update_from_poses(&dof_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&dof_poses(&g[&key("target")])).unwrap();
            e.encode_inputs(&c, &t, &cfg).unwrap();
            let count = e.decode_prediction(&pred, &mut out).unwrap();
            let decoded = out.poses()[..count * 36].to_vec();
            if cfg.correct_generated_diff {
                e.correct_endpoints(&c, &t, &mut out).unwrap();
            }
            (dof_error(&decoded, &g[&key("decoded")].to_f32(), 36), dof_error(&out.poses()[..count * 36], &g[&key("generated")].to_f32(), 36), out.foot_contacts().unwrap().to_vec(), count)
        };
        f.check(count == want_count, || format!("{label}: {count} frames, upstream {want_count}"));
        f.check(decoded < 2e-5, || format!("{label}: decoded max error {decoded}"));
        f.check(generated < 2e-5, || format!("{label}: corrected max error {generated}"));
        let flat: Vec<f32> = contacts[..count].iter().flatten().copied().collect();
        let e = max_abs(&flat, &g[&key("decoded_contacts")].to_f32(), |_| true);
        f.check(e < 1e-6, || format!("{label}: foot contacts max error {e}"));
    }
    f.finish();
}

fn clip_loading_matches_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    for (label, stem) in [("walk", "Neutral_walk_forward_002__A057"), ("jog", "neutral_jog_ff_180_R_001__A534")] {
        let Some(path) = clip_path(name, stem) else { return };
        let idx = g[&format!("clip.{label}.frame_index")].to_i32();
        let want = g[&format!("clip.{label}.frames")].to_f32();
        let (frames, rate, err) = if name == "soma" {
            let clip = load_bvh(&path, Some(skeleton(&g)), true).unwrap();
            let got: Vec<Transform> = idx.iter().flat_map(|&i| clip.frame(i as usize).to_vec()).collect();
            (clip.num_frames(), clip.sample_rate(), transform_error(&got, &want))
        } else {
            let clip = load_csv(&path, CSV_SAMPLE_RATE).unwrap();
            let got: Vec<f32> = idx.iter().flat_map(|&i| clip.frame(i as usize).to_vec()).collect();
            (clip.num_frames(), clip.sample_rate(), dof_error(&got, &want, 36))
        };
        assert_eq!(frames as i32, g[&format!("clip.{label}.num_frames")].to_i32()[0]);
        assert_eq!(rate, g[&format!("clip.{label}.sample_rate")].to_f32()[0]);
        assert!(err < 2e-5, "{label}: clip max error {err}");
    }
    if name == "soma" {
        let path = clip_path(name, "Neutral_walk_forward_002__A057").unwrap();
        let sk = load_skeleton_from_bvh(&path, true, true).unwrap();
        let want = skeleton(&g);
        assert_eq!(sk.joint_names(), want.joint_names());
        assert_eq!(sk.parent_indices(), want.parent_indices());
        assert!(transform_error(sk.reference_local_transforms(), &g["skeleton.reference"].to_f32()) < 1e-6);
    }
}

fn recorder_sampling_matches_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let dts = g["recorder.dts"].to_f32();
    let want_dt = g["recorder.delta_times"].to_f32();
    let (history_err, token_err, dt_ok) = if name == "soma" {
        let poses = soma_poses(&g["recorder.poses"]);
        let mut h = MotionRecorder::new(&poses[0], 8).unwrap();
        for (dt, p) in dts.iter().zip(&poses[1..]) {
            h.record(*dt, p).unwrap();
        }
        let mut token = PoseToken::new(poses[0].len());
        token.update_from_recorder(&h).unwrap();
        let flat_h: Vec<Transform> = h.poses().concat();
        let flat_t: Vec<Transform> = token.poses().concat();
        (transform_error(&flat_h, &g["recorder.history"].to_f32()), transform_error(&flat_t, &g["recorder.token"].to_f32()), h.delta_times() == want_dt.as_slice())
    } else {
        let poses = dof_poses(&g["recorder.poses"]);
        let mut h = MotionRecorder::new(&poses[0], 8).unwrap();
        for (dt, p) in dts.iter().zip(&poses[1..]) {
            h.record(*dt, p).unwrap();
        }
        let mut token = DofToken::new(29);
        token.update_from_recorder(&h).unwrap();
        (dof_error(&h.poses().concat(), &g["recorder.history"].to_f32(), 36), dof_error(&token.poses().concat(), &g["recorder.token"].to_f32(), 36), h.delta_times() == want_dt.as_slice())
    };
    assert!(dt_ok, "delta times differ");
    assert_eq!(history_err, 0.0, "recorded history differs");
    assert!(token_err < 2e-6, "resampled token max error {token_err}");
}

fn target_placement_and_speed_match_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let (pos, dir) = (Vec3::new(3.0, -1.5, 0.0), Vec3::new(0.3, 0.9, 0.0));
    if name == "soma" {
        let targets = soma_poses(&g["target.poses"]);
        let placed = place_target_poses(pos, dir, &targets).concat();
        assert!(transform_error(&placed, &g["target.placed"].to_f32()) < 2e-6);
        for speed in [0.0f32, 1.3] {
            let mut scaled = targets.clone();
            adjust_target_pose_speed(&mut scaled, speed).unwrap();
            assert!(transform_error(&scaled.concat(), &g[&format!("target.speed_{speed}")].to_f32()) < 2e-6, "speed {speed}");
        }
    } else {
        let targets = dof_poses(&g["target.poses"]);
        assert!(dof_error(&place_target_dof(pos, dir, &targets).concat(), &g["target.placed"].to_f32(), 36) < 2e-6);
        for speed in [0.0f32, 1.3] {
            let mut scaled = targets.clone();
            adjust_target_dof_speed(&mut scaled, speed).unwrap();
            assert!(dof_error(&scaled.concat(), &g[&format!("target.speed_{speed}")].to_f32(), 36) < 2e-6, "speed {speed}");
        }
    }
}

fn buffer_sampling_and_looping_match_upstream(name: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let Some(path) = clip_path(name, "Neutral_walk_forward_002__A057") else { return };
    let offset = Transform::new(Vec3::new(1.0, 2.0, 0.0), quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 40f32.to_radians()));
    let times = [(0.0f32, "0"), (0.37, "0.37"), (1.234, "1.234"), (99.0, "99")];
    if name == "soma" {
        let clip = load_bvh(&path, Some(skeleton(&g)), true).unwrap();
        for (t, label) in times {
            let e = transform_error(&clip.sample(t, offset).unwrap(), &g[&format!("sample.{label}")].to_f32());
            assert!(e < 2e-5, "sample at {t}: {e}");
        }
        let mut lp = AnimationBuffer::new(clip.skeleton().clone(), clip.sample_rate(), transforms(&g["loop.before"].to_f32()), Some(40), None).unwrap();
        lp.make_loopable();
        assert!(transform_error(lp.poses(), &g["loop.after"].to_f32()) < 2e-5);
    } else {
        let clip = load_csv(&path, CSV_SAMPLE_RATE).unwrap();
        for (t, label) in times {
            let e = dof_error(&clip.sample(t, offset).unwrap(), &g[&format!("sample.{label}")].to_f32(), 36);
            assert!(e < 2e-5, "sample at {t}: {e}");
        }
        let mut lp = DofAnimationBuffer::new(clip.joint_names().to_vec(), clip.sample_rate(), g["loop.before"].to_f32(), Some(40), None).unwrap();
        lp.make_loopable();
        assert!(dof_error(lp.poses(), &g["loop.after"].to_f32(), 36) < 2e-5);
    }
}

#[test]
fn ik_pins_match_upstream() {
    let Some(g) = runtime_golden("soma") else { return };
    let sk = skeleton(&g);
    let mut corrector = TargetPoseIkCorrector::new(&sk, create_soma_target_pose_ik_rig(&sk).unwrap()).unwrap();
    let (target, current) = (transforms(&g["ik.target"].to_f32()), transforms(&g["ik.current"].to_f32()));
    for label in ["hands", "all"] {
        corrector.capture_pins(&target, g[&format!("ik.{label}.mask")].to_i32()[0] as u32).unwrap();
        let mut pose = current.clone();
        corrector.apply(&mut pose, g[&format!("ik.{label}.weight")].to_f32()[0]).unwrap();
        let e = transform_error(&pose, &g[&format!("ik.{label}.result")].to_f32());
        assert!(e < 5e-5, "{label}: IK max error {e}");
    }
    corrector.capture_pins(&target, ik_pin::NONE).unwrap();
    let mut pose = current.clone();
    corrector.apply(&mut pose, 1.0).unwrap();
    assert_eq!(pose, current);
}

// --- end to end on both backends ----------------------------------------------------------------

fn generation_matches_upstream<B: Backend>(name: &str, mut backend: impl FnMut() -> B, label_backend: &str) {
    let Some(g) = runtime_golden(name) else { return };
    let mut f = Failures::default();
    for label in SCENARIOS {
        let m2r = Transform::from_slice(&g[&format!("gen.{label}.model_to_runtime")].to_f32());
        let cfg = config(&g, label);
        let key = |s: &str| format!("gen.{label}.{s}");
        let want_count = g[&key("frame_count")].to_i32()[0] as usize;
        let (count, err, timer) = if name == "soma" {
            let sk = skeleton(&g);
            let mut e = MotionBricksSomaEngine::new(Some(backend()), m2r).unwrap();
            let (mut c, mut t, mut out) = (e.create_input(&sk).unwrap(), e.create_input(&sk).unwrap(), e.create_output(&sk).unwrap());
            c.update_from_poses(&soma_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&soma_poses(&g[&key("target")])).unwrap();
            e.generate_animation(&c, &t, &mut out, &cfg).unwrap();
            let n = out.num_frames().min(want_count);
            (out.num_frames(), transform_error(&out.poses()[..n * sk.num_joints()], &g[&key("generated")].to_f32()[..n * sk.num_joints() * 7]), e.timer().clone())
        } else {
            let mut e = MotionBricksUnitreeG1Engine::new(Some(backend()), m2r).unwrap();
            let (mut c, mut t, mut out) = (e.create_input(), e.create_input(), e.create_output());
            c.update_from_poses(&dof_poses(&g[&key("context")])).unwrap();
            t.update_from_poses(&dof_poses(&g[&key("target")])).unwrap();
            e.generate_animation(&c, &t, &mut out, &cfg).unwrap();
            let n = out.num_frames().min(want_count);
            (out.num_frames(), dof_error(&out.poses()[..n * 36], &g[&key("generated")].to_f32()[..n * 36], 36), e.timer().clone())
        };
        f.check(count == want_count, || format!("{label_backend} {label}: {count} frames, upstream {want_count}"));
        // The models match PyTorch to ~1e-5; decoded rotations pass a Gram-Schmidt step.
        f.check(err < 3e-3, || format!("{label_backend} {label}: max error {err}"));
        f.check(timer.inference > 0.0, || format!("{label_backend} {label}: no inference time"));
    }
    f.finish();
}

fn generation_metal(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    generation_matches_upstream(name, || MetalModel::from_checkpoints(name, &dir, 1).unwrap(), "metal");
}

fn generation_mlx(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    generation_matches_upstream(name, || MlxBackend::new(motionbricks_rs::MotionBricks::from_checkpoints(name, &dir, mlx_rs::Dtype::Float32).unwrap()), "mlx-rs");
}

for_both_models!(
    encode_matches_upstream,
    decode_and_correction_match_upstream,
    clip_loading_matches_upstream,
    recorder_sampling_matches_upstream,
    target_placement_and_speed_match_upstream,
    buffer_sampling_and_looping_match_upstream,
    generation_metal,
    generation_mlx,
);
