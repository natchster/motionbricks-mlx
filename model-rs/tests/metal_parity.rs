// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The custom Metal runtime vs NVIDIA's PyTorch model on the golden fixtures, stage by stage
//! (teacher-forced) and end to end, with the tolerances of the Python suite.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Buf, Recorder, wait};
use motionbricks_rs::npy::{Npy, load_npz};

static GPU: Mutex<()> = Mutex::new(());
const CASES: [&str; 5] = ["trace", "random", "one_target", "restricted", "chained"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn put(dst: &Buf, v: &[f32]) {
    dst.write_f32(v); // converts to the buffer's element type
}

fn put_i(dst: &Buf, v: &[i32]) {
    let f: Vec<f32> = v.iter().map(|&x| f32::from_bits(x as u32)).collect();
    put(dst, &f);
}

fn mask(n: &Npy) -> Vec<f32> {
    n.to_f32().into_iter().map(|x| (x != 0.0) as i32 as f32).collect()
}

fn rel(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        if keep(k) {
            num = num.max((g as f64 - w as f64).abs());
            den = den.max((w as f64).abs());
        }
    }
    num / den.max(1e-12)
}

fn max_abs(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> f32 {
    got.iter().zip(want).enumerate().filter(|(k, _)| keep(*k)).map(|(_, (a, b))| (a - b).abs()).fold(0.0, f32::max)
}

fn argmax_rows(v: &[f32], w: usize) -> Vec<usize> {
    v.chunks(w).map(|r| r.iter().enumerate().fold(0, |b, (k, &x)| if x > r[b] { k } else { b })).collect()
}

fn run(name: &str) {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let folder = if name == "soma" { "soma" } else { "g1" };
    let dir = root().join("checkpoints/MotionBricks").join(folder);
    if !dir.is_dir() {
        eprintln!("skipping {name}: no checkpoints");
        return;
    }
    let m = MetalModel::from_checkpoints(name, &dir, 8).expect("metal model loads");
    let b = &m.b;
    println!("{name}:");
    for case in CASES {
        let g: HashMap<String, Npy> = load_npz(root().join("sandbox/golden").join(name).join(format!("{case}.npz"))).unwrap();
        let f = |k: &str| g[k].to_f32();
        let n = g["pred_num_tokens"].shape[0] as usize;

        // Root network, teacher-forced.
        put(&b.gn, &f("root_in.global_root"));
        put(&b.ln, &f("root_in.local_root"));
        put(&b.pn, &f("root_in.poses"));
        put(&b.gn_t, &f("root_in.global_root"));
        put(&b.ln_t, &f("root_in.local_root"));
        put(&b.pn_t, &f("root_in.poses"));
        put(&b.hg, &mask(&g["root_in.has_global"]));
        put(&b.hl, &mask(&g["root_in.has_local"]));
        put(&b.hp, &mask(&g["root_in.has_poses"]));
        put(&b.allowed, &mask(&g["root_in.allowed"]));
        put_i(&b.nt_in, &g["root_in.num_tokens"].to_i32());
        let d = m.dims(n, true);
        let mut rec = Recorder::new(&m.gpu);
        m.encode_root(&mut rec, &d);
        wait(&rec.commit());
        let want = f("root_out.num_token_logits");
        let got = &b.logits.as_slice()[..n * 12];
        assert_eq!(got.iter().map(|v| v.is_finite()).collect::<Vec<_>>(), want.iter().map(|v| v.is_finite()).collect::<Vec<_>>(), "{case}: masked classes");
        let e = rel(got, &want, |k| want[k].is_finite());
        assert!(e < 1e-4, "{case}: root logits rel {e}");
        assert_eq!(m.num_tokens(n), g["root_out.pred_num_tokens"].to_i32(), "{case}: durations");
        let e2 = rel(&b.pr.to_f32(n * 64 * 5), &f("root_out.pred_global_root_values"), |_| true);
        assert!(e2 < 1e-4, "{case}: root path rel {e2}");

        // Pose network, teacher-forced (release config: every input token is the mask token).
        let toks = g["pose_in.pose_tokens"].to_i32();
        assert!(toks.iter().all(|&t| t == m.spec.vq.codes_per_head), "{case}: expected all-masked pose tokens");
        let nt = g["pose_in.num_tokens"].to_i32();
        put(&b.rcond, &f("pose_in.root_cond"));
        put(&b.pcond, &f("pose_in.pose_cond"));
        put(&b.hpc, &mask(&g["pose_in.has_pose_cond"]));
        put_i(&b.nt, &nt);
        let le: Vec<f32> = (0..n).flat_map(|i| (0..16).flat_map(move |_| (0..128).map(move |c| (i, c)))).map(|(i, c)| m.len_emb_host[(nt[i] as usize - 6) * 128 + c]).collect();
        put(&b.le, &le);
        let ppad: Vec<f32> = (0..n * 16).map(|k| ((k % 16) as i32 >= nt[k / 16]) as i32 as f32).collect();
        put(&b.ppad, &ppad);
        let mut rec = Recorder::new(&m.gpu);
        m.encode_pose(&mut rec, &m.dims(n, false));
        wait(&rec.commit());
        let want = f("pose_out.pose_logits");
        let got = &b.plog.to_f32(want.len());
        let e3 = rel(got, &want, |_| true);
        assert!(e3 < 1e-4, "{case}: pose logits rel {e3}");
        let codes = m.spec.vq.codes_per_head as usize;
        assert_eq!(argmax_rows(got, codes), argmax_rows(&want, codes), "{case}: pose tokens differ");

        // VQ-VAE decoder, teacher-forced.
        let vt = g["vq_in.tokens"].to_i32();
        let (heads, cw) = (m.spec.vq.num_heads as usize, (m.spec.vq.code_dim / m.spec.vq.num_heads) as usize);
        let lat: Vec<f32> = vt.iter().enumerate().flat_map(|(k, &t)| {
            let h = k % heads;
            m.codebook_host[(h * codes + t as usize) * cw..(h * codes + t as usize + 1) * cw].to_vec()
        }).collect();
        put(&b.lat, &lat);
        put(&b.ext, &f("vq_in.external_cond"));
        put(&b.pcond, &f("vq_in.target_cond"));
        put(&b.hpc, &mask(&g["vq_in.has_target_cond"]));
        let tm = mask(&g["vq_in.token_mask"]);
        put(&b.m16, &tm);
        put(&b.m32, &tm.iter().flat_map(|&v| [v, v]).collect::<Vec<_>>());
        put(&b.m64, &tm.iter().flat_map(|&v| [v, v, v, v]).collect::<Vec<_>>());
        let mut rec = Recorder::new(&m.gpu);
        m.encode_vq(&mut rec, &m.dims(n, false));
        wait(&rec.commit());
        let want = f("vq_out.recon_state");
        let e4 = rel(&b.recon.to_f32(want.len()), &want, |_| true);
        assert!(e4 < 1e-4, "{case}: vq decoder rel {e4}");

        // End to end.
        let inputs: HashMap<String, Npy> = g.iter().filter_map(|(k, v)| k.strip_prefix("in.").map(|s| (s.to_string(), v.clone()))).collect();
        let (out, durations, contacts) = m.run(&inputs).unwrap();
        let counts = g["pred_num_tokens"].to_i32();
        assert_eq!(durations, counts, "{case}: end-to-end durations");
        let key = if name == "soma" { "pred_global_motions" } else { "pred_qpos" };
        let shape = g[key].shape.clone();
        let (frames, dim) = (shape[1] as usize, shape[2] as usize);
        let valid = |k: usize| (k / dim) % frames < 4 * counts[k / (dim * frames)] as usize;
        let want = f(key);
        if name == "soma" {
            let rot = (92usize, 272usize);
            let root_e = max_abs(&out, &want, |k| valid(k) && k % dim < 5);
            let rot_e = max_abs(&out, &want, |k| valid(k) && (rot.0..rot.1).contains(&(k % dim)));
            let other = max_abs(&out, &want, |k| valid(k) && k % dim >= 5 && !(rot.0..rot.1).contains(&(k % dim)));
            println!("  {case:<10} root logits {e:.1e}, path {e2:.1e} | pose {e3:.1e} | vq {e4:.1e} | e2e root/other/rot {root_e:.1e}/{other:.1e}/{rot_e:.1e}");
            assert!(root_e < 1e-4 && other < 1e-3 && rot_e < 3e-3, "{case}: end-to-end {root_e} {other} {rot_e}");
        } else {
            let q = max_abs(&out, &want, valid);
            let c = max_abs(contacts.as_ref().unwrap(), &f("pred_foot_contacts"), |k| (k / 4) % frames < 4 * counts[k / (4 * frames)] as usize);
            println!("  {case:<10} root logits {e:.1e}, path {e2:.1e} | pose {e3:.1e} | vq {e4:.1e} | e2e qpos {q:.1e} contacts {c:.1e}");
            assert!(q < 5e-4 && c < 5e-3, "{case}: end-to-end qpos {q} contacts {c}");
        }
    }
}

#[test]
fn metal_soma_matches_pytorch() {
    run("soma");
}

#[test]
fn metal_unitree_g1_matches_pytorch() {
    run("unitree_g1");
}

/// FP16 / BF16 / INT8 track FP32 with the Python port's criteria (test_model.py): >= 75% of
/// durations agree with the FP32 model, outputs finite, and median |diff| < 0.05 over the rows
/// whose duration agrees. (Checked by hand against the Python port at the same precision: the
/// same durations and the same error magnitudes, e.g. SOMA BF16 4.4e-2 vs Python 4.5e-2.)
fn low_precision(name: &str) {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let folder = if name == "soma" { "soma" } else { "g1" };
    let dir = root().join("checkpoints/MotionBricks").join(folder);
    if !dir.is_dir() {
        return;
    }
    let fp32 = MetalModel::from_checkpoints(name, &dir, 8).expect("model loads");
    let requests: Vec<HashMap<String, Npy>> = CASES
        .iter()
        .map(|case| {
            let g = load_npz(root().join("sandbox/golden").join(name).join(format!("{case}.npz"))).unwrap();
            g.into_iter().filter_map(|(k, v)| k.strip_prefix("in.").map(|s| (s.to_string(), v))).collect()
        })
        .collect();
    let reference: Vec<_> = requests.iter().map(|r| fp32.run(r).unwrap()).collect();
    for precision in ["float16", "bfloat16", "int8"] {
        let m = MetalModel::from_checkpoints_with(name, &dir, 8, precision).expect("model loads");
        let (mut agree, mut total, mut diffs) = (0usize, 0usize, Vec::new());
        for (inputs, (want, want_t, _)) in requests.iter().zip(&reference) {
            let (out, durations, _) = m.run(inputs).unwrap();
            assert!(out.iter().all(|v| v.is_finite()), "{name} {precision}: non-finite output");
            let per = out.len() / durations.len();
            for (i, (&a, &b)) in durations.iter().zip(want_t).enumerate() {
                total += 1;
                if a == b {
                    agree += 1;
                    diffs.extend((0..per).map(|k| (out[i * per + k] - want[i * per + k]).abs()));
                }
            }
        }
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = diffs[diffs.len() / 2];
        println!("{name} {precision}: durations agree {agree}/{total}, median |diff vs FP32| {median:.2e}");
        assert!(agree * 4 >= total * 3, "{name} {precision}: only {agree}/{total} durations match FP32");
        assert!(median < 0.05, "{name} {precision}: median diff {median}");
    }
}

#[test]
fn metal_soma_low_precision_tracks_fp32() {
    low_precision("soma");
}

#[test]
fn metal_unitree_g1_low_precision_tracks_fp32() {
    low_precision("unitree_g1");
}
