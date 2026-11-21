// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Rust port vs NVIDIA's upstream PyTorch model, on the same golden fixtures and with the
//! same tolerances as the Python port's `model/tests/test_parity.py`. Every sub-network is
//! teacher-forced with the exact tensors the PyTorch run fed it, then the whole graph runs end to
//! end (eager and compiled).
//!
//! Needs `checkpoints/` and `sandbox/golden/` (created by `sandbox/setup.sh`); skips otherwise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Mutex;

use mlx_rs::{Array, Dtype};
use motionbricks_rs::npy::{Npy, load_npz};
use motionbricks_rs::ops::{i, to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled};

static MLX: Mutex<()> = Mutex::new(()); // one model on the GPU at a time
const CASES: [&str; 5] = ["trace", "random", "one_target", "restricted", "chained"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn load(name: &str) -> Option<MotionBricks> {
    let folder = if name == "soma" { "soma" } else { "g1" };
    let dir = root().join("checkpoints/MotionBricks").join(folder);
    if !dir.is_dir() || !root().join("sandbox/golden").join(name).is_dir() {
        eprintln!("skipping {name}: checkpoints or golden fixtures missing (run sandbox/setup.sh)");
        return None;
    }
    Some(MotionBricks::from_checkpoints(name, dir, Dtype::Float32).expect("model loads"))
}

fn rel_err(got: &[f32], want: &[f32], mask: Option<&[bool]>) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        if mask.is_some_and(|m| !m[k]) {
            continue;
        }
        num = num.max((g as f64 - w as f64).abs());
        den = den.max((w as f64).abs());
    }
    num / den.max(1e-12)
}

struct Golden(HashMap<String, Npy>);

impl Golden {
    fn f(&self, k: &str) -> Array {
        self.0[k].f32_array()
    }
    fn mask(&self, k: &str) -> Array {
        self.0[k].i32_array().ne(i(0)).unwrap()
    }
    fn int(&self, k: &str) -> Array {
        self.0[k].i32_array()
    }
    fn vf(&self, k: &str) -> Vec<f32> {
        self.0[k].to_f32()
    }
    fn vi(&self, k: &str) -> Vec<i32> {
        self.0[k].to_i32()
    }
    fn shape(&self, k: &str) -> Vec<i32> {
        self.0[k].shape.clone()
    }
}

fn argmax_last(values: &[f32], width: usize) -> Vec<usize> {
    values
        .chunks(width)
        .map(|row| row.iter().enumerate().fold(0, |best, (k, &v)| if v > row[best] { k } else { best }))
        .collect()
}

fn check_stages(model: &MotionBricks, g: &Golden, case: &str) {
    let core = &model.core;
    // Root network.
    let num_tokens = g.int("root_in.num_tokens").reshape(&[-1]).unwrap();
    let out = core
        .root
        .forward(
            &g.f("root_in.global_root"), &g.mask("root_in.has_global"), &g.f("root_in.local_root"),
            &g.mask("root_in.has_local"), &g.f("root_in.poses"), &g.mask("root_in.has_poses"),
            &g.mask("root_in.allowed"), Some(&num_tokens),
        )
        .unwrap();
    let logits = to_vec_f32(&out.logits).unwrap();
    let want = g.vf("root_out.num_token_logits");
    let finite: Vec<bool> = want.iter().map(|v| v.is_finite()).collect();
    assert_eq!(logits.iter().map(|v| v.is_finite()).collect::<Vec<_>>(), finite, "{case}: masked classes differ");
    let e = rel_err(&logits, &want, Some(&finite));
    assert!(e < 1e-4, "{case}: root logits rel err {e}");
    assert_eq!(to_vec_i32(&out.pred_num_tokens).unwrap(), g.vi("root_out.pred_num_tokens"), "{case}: durations");
    let e = rel_err(&to_vec_f32(&out.pred_root).unwrap(), &g.vf("root_out.pred_global_root_values"), None);
    assert!(e < 1e-4, "{case}: root path rel err {e}");
    println!("  {case:<10} root logits/path ok");

    // Pose network.
    let logits = core
        .pose
        .forward(
            &g.int("pose_in.pose_tokens"), &g.f("pose_in.root_cond"), &g.f("pose_in.pose_cond"),
            &g.mask("pose_in.has_pose_cond"), &g.int("pose_in.num_tokens").reshape(&[-1]).unwrap(),
        )
        .unwrap();
    let got = to_vec_f32(&logits).unwrap();
    let want = g.vf("pose_out.pose_logits");
    let e = rel_err(&got, &want, None);
    assert!(e < 1e-4, "{case}: pose logits rel err {e}");
    let codes = *g.shape("pose_out.pose_logits").last().unwrap() as usize;
    assert_eq!(argmax_last(&got, codes), argmax_last(&want, codes), "{case}: pose tokens differ");
    println!("  {case:<10} pose logits ok (rel {e:.1e}), tokens identical");

    // VQ-VAE decoder.
    let recon = core
        .vqvae
        .decode(
            &g.int("vq_in.tokens"), &g.f("vq_in.target_cond"), &g.mask("vq_in.has_target_cond"),
            &g.f("vq_in.external_cond"), &g.mask("vq_in.token_mask"),
        )
        .unwrap();
    let e = rel_err(&to_vec_f32(&recon).unwrap(), &g.vf("vq_out.recon_state"), None);
    assert!(e < 1e-4, "{case}: vq decoder rel err {e}");
    println!("  {case:<10} vq decoder ok (rel {e:.1e})");
}

fn max_abs(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> f32 {
    got.iter().zip(want).enumerate().filter(|(k, _)| keep(*k)).map(|(_, (a, b))| (a - b).abs()).fold(0.0, f32::max)
}

fn check_end_to_end(model: &MotionBricks, outputs: &[Array], g: &Golden, case: &str, label: &str) {
    let counts = g.vi("pred_num_tokens");
    assert_eq!(to_vec_i32(&outputs[1]).unwrap(), counts, "{case}/{label}: durations");
    let first = model.output_names()[0];
    let shape = g.shape(first);
    let (frames, dim) = (shape[1] as usize, shape[2] as usize);
    // Only frames inside each request's chosen duration are meaningful.
    let valid = |k: usize| (k / dim) % frames < 4 * counts[k / (dim * frames)] as usize;
    let got = to_vec_f32(&outputs[0]).unwrap();
    let want = g.vf(first);
    if model.spec.name == "unitree_g1" {
        let e = max_abs(&got, &want, valid);
        assert!(e < 5e-4, "{case}/{label}: qpos max abs {e}");
        let contacts = to_vec_f32(&outputs[2]).unwrap();
        let ec = max_abs(&contacts, &g.vf("pred_foot_contacts"), |k| (k / 4) % frames < 4 * counts[k / (4 * frames)] as usize);
        assert!(ec < 5e-3, "{case}/{label}: contacts max abs {ec}");
        println!("  {case:<10} end to end ({label}): durations identical, qpos max abs {e:.1e}");
    } else {
        let layout = &model.core.rep.global;
        let rot = layout.range("global_rot_data");
        let (r0, r1) = (rot[0] as usize, *rot.last().unwrap() as usize + 1);
        let root = max_abs(&got, &want, |k| valid(k) && k % dim < 5);
        let rots = max_abs(&got, &want, |k| valid(k) && (r0..r1).contains(&(k % dim)));
        let other = max_abs(&got, &want, |k| valid(k) && k % dim >= 5 && !(r0..r1).contains(&(k % dim)));
        assert!(root < 1e-4, "{case}/{label}: root max abs {root}");
        assert!(other < 1e-3, "{case}/{label}: features max abs {other}");
        assert!(rots < 3e-3, "{case}/{label}: rotations max abs {rots}");
        println!("  {case:<10} end to end ({label}): durations identical, root/other/rot max abs {root:.1e}/{other:.1e}/{rots:.1e}");
    }
}

fn run_model(name: &str) {
    let _guard = MLX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = load(name) else { return };
    let model = Rc::new(model);
    let mut forward = compiled(model.clone());
    println!("{name}:");
    for case in CASES {
        let g = Golden(load_npz(root().join("sandbox/golden").join(name).join(format!("{case}.npz"))).unwrap());
        check_stages(&model, &g, case);
        let inputs: HashMap<String, Npy> =
            g.0.iter().filter_map(|(k, v)| k.strip_prefix("in.").map(|n| (n.to_string(), v.clone()))).collect();
        let args = model.prepare(&inputs).unwrap();
        let eager = model.forward(&args).unwrap();
        check_end_to_end(&model, &eager, &g, case, "eager");
        let fused = forward(&args).unwrap();
        check_end_to_end(&model, &fused, &g, case, "compiled");
    }
}

#[test]
fn soma_matches_pytorch() {
    run_model("soma");
}

#[test]
fn unitree_g1_matches_pytorch() {
    run_model("unitree_g1");
}

#[test]
fn rig_kernels_match_ops() {
    use motionbricks_rs::npy::load_npz_bytes;
    use motionbricks_rs::rig::{G1Rig, SomaRetarget};
    let _guard = MLX.lock().unwrap_or_else(|e| e.into_inner());
    let g1 = load_npz_bytes(include_bytes!("../../model/motionbricks_mlx/assets/g1_model_rig.npz")).unwrap();
    let range = |a: i32, b: i32| (a..b).collect::<Vec<i32>>();
    let rig = G1Rig::new(&g1, &range(104, 308), &range(0, 3), &range(410, 414)).unwrap();
    let mut seed = 7u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|_| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale }).collect()
    };
    let close = |a: &Array, b: &Array, what: &str| {
        let (x, y) = (to_vec_f32(a).unwrap(), to_vec_f32(b).unwrap());
        let e = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(e < 3e-5, "{what}: kernel vs ops max abs {e}");
    };
    let ctx = Array::from_slice(&rnd(3 * 4 * 36, 0.8), &[3, 4, 36]);
    let tgt = Array::from_slice(&rnd(3 * 4 * 36, 0.8), &[3, 4, 36]);
    let (a, b) = (rig.preprocess(&ctx, &tgt).unwrap(), rig.preprocess_ops(&ctx, &tgt).unwrap());
    close(&a.0, &b.0, "g1 global root");
    close(&a.1, &b.1, "g1 local root");
    close(&a.2, &b.2, "g1 local poses");
    let feats = Array::from_slice(&rnd(2 * 64 * 414, 2.0), &[2, 64, 414]);
    let (a, b) = (rig.postprocess(&feats).unwrap(), rig.postprocess_ops(&feats).unwrap());
    close(&a.0, &b.0, "g1 qpos");
    close(&a.1, &b.1, "g1 contacts");
    let soma = SomaRetarget::new(&load_npz_bytes(include_bytes!("../../model/motionbricks_mlx/assets/soma_model_rig.npz")).unwrap(), 30).unwrap();
    let six = Array::from_slice(&rnd(2 * 64 * 180, 2.0), &[2, 64, 180]);
    let (cap, std) = soma.offsets();
    close(&soma.capture_to_standard(&six).unwrap(), &soma.apply_ops(&six, cap, std).unwrap(), "soma retarget");
}
