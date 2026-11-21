// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The mlx-rs port vs NVIDIA's upstream PyTorch model, stage by stage, on the golden fixtures:
//! the Rust counterpart of `model/tests/test_parity.py`, with the same checks and tolerances.
//! Each sub-network is teacher-forced with the exact tensors the PyTorch run fed it, so a
//! mismatch localizes to one module. One test per (model, stage) covers all 5 request families;
//! the end-to-end test runs both the eager and the compiled graph.
//!
//! Needs the checkpoints and the sandbox's golden fixtures; prints SKIPPED without them.

mod common;

use std::collections::HashMap;
use std::rc::Rc;

use common::*;
use mlx_rs::{Array, Dtype};
use motionbricks_rs::npy::Npy;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled};

struct G<'a>(&'a HashMap<String, Npy>);

impl G<'_> {
    fn f(&self, k: &str) -> Array {
        self.0[k].f32_array()
    }
    fn flag(&self, k: &str) -> Array {
        self.0[k].bool_array()
    }
    fn int(&self, k: &str) -> Array {
        self.0[k].i32_array()
    }
    fn v(&self, k: &str) -> Vec<f32> {
        self.0[k].to_f32()
    }
}

/// The model and all golden cases, or `None` (skip).
fn setup(name: &str) -> Option<(MotionBricks, Cases)> {
    let dir = checkpoints(name)?;
    let cases = goldens(name)?;
    Some((MotionBricks::from_checkpoints(name, dir, Dtype::Float32).expect("model loads"), cases))
}

fn root_network(name: &str) {
    let _gpu = gpu_lock();
    let Some((model, cases)) = setup(name) else { return };
    let mut f = Failures::default();
    for (case, g) in &cases {
        let g = G(g);
        let out = model
            .core
            .root
            .forward(
                &g.f("root_in.global_root"), &g.flag("root_in.has_global"), &g.f("root_in.local_root"),
                &g.flag("root_in.has_local"), &g.f("root_in.poses"), &g.flag("root_in.has_poses"), &g.flag("root_in.allowed"),
                Some(&g.int("root_in.num_tokens").reshape(&[-1]).unwrap()),
            )
            .unwrap();
        let logits = to_vec_f32(&out.logits).unwrap();
        let want = g.v("root_out.num_token_logits");
        let finite: Vec<bool> = want.iter().map(|v| v.is_finite()).collect();
        f.check(logits.iter().map(|v| v.is_finite()).collect::<Vec<_>>() == finite, || format!("{case}: masked (-inf) classes differ"));
        let e = rel_err(&logits, &want, |k| finite[k]);
        f.check(e < 1e-4, || format!("{case}: root logits rel err {e}"));
        let tokens = to_vec_i32(&out.pred_num_tokens).unwrap();
        f.check(tokens == g.0["root_out.pred_num_tokens"].to_i32(), || format!("{case}: durations {tokens:?}"));
        let e = rel_err(&to_vec_f32(&out.pred_root).unwrap(), &g.v("root_out.pred_global_root_values"), |_| true);
        f.check(e < 1e-4, || format!("{case}: root path rel err {e}"));
    }
    f.finish();
}

fn pose_network(name: &str) {
    let _gpu = gpu_lock();
    let Some((model, cases)) = setup(name) else { return };
    let mut f = Failures::default();
    for (case, g) in &cases {
        let g = G(g);
        let logits = model
            .core
            .pose
            .forward(
                &g.int("pose_in.pose_tokens"), &g.f("pose_in.root_cond"), &g.f("pose_in.pose_cond"),
                &g.flag("pose_in.has_pose_cond"), &g.int("pose_in.num_tokens").reshape(&[-1]).unwrap(),
            )
            .unwrap();
        let got = to_vec_f32(&logits).unwrap();
        let want = g.v("pose_out.pose_logits");
        let e = rel_err(&got, &want, |_| true);
        f.check(e < 1e-4, || format!("{case}: pose logits rel err {e}"));
        let codes = *g.0["pose_out.pose_logits"].shape.last().unwrap() as usize;
        f.check(argmax_rows(&got, codes) == argmax_rows(&want, codes), || format!("{case}: argmax tokens differ"));
    }
    f.finish();
}

fn vqvae_decoder(name: &str) {
    let _gpu = gpu_lock();
    let Some((model, cases)) = setup(name) else { return };
    let mut f = Failures::default();
    for (case, g) in &cases {
        let g = G(g);
        let recon = model
            .core
            .vqvae
            .decode(
                &g.int("vq_in.tokens"), &g.f("vq_in.target_cond"), &g.flag("vq_in.has_target_cond"), &g.f("vq_in.external_cond"),
                &g.flag("vq_in.token_mask"),
            )
            .unwrap();
        let e = rel_err(&to_vec_f32(&recon).unwrap(), &g.v("vq_out.recon_state"), |_| true);
        f.check(e < 1e-4, || format!("{case}: decoder rel err {e}"));
    }
    f.finish();
}

fn end_to_end(name: &str) {
    let _gpu = gpu_lock();
    let Some((model, cases)) = setup(name) else { return };
    let model = Rc::new(model);
    let mut forward = compiled(model.clone());
    let mut f = Failures::default();
    for (case, g) in &cases {
        let args = model.prepare(&golden_inputs(g)).unwrap();
        for (label, out) in [("eager", model.forward(&args).unwrap()), ("compiled", forward(&args).unwrap())] {
            let contacts = out.get(2).map(|c| to_vec_f32(c).unwrap());
            let durations = to_vec_i32(&out[1]).unwrap();
            check_end_to_end(&mut f, name, case, label, g, &to_vec_f32(&out[0]).unwrap(), &durations, contacts.as_deref());
        }
    }
    f.finish();
}

#[test]
fn unitree_g1_root_network() {
    root_network("unitree_g1");
}
#[test]
fn soma_root_network() {
    root_network("soma");
}
#[test]
fn unitree_g1_pose_network() {
    pose_network("unitree_g1");
}
#[test]
fn soma_pose_network() {
    pose_network("soma");
}
#[test]
fn unitree_g1_vqvae_decoder() {
    vqvae_decoder("unitree_g1");
}
#[test]
fn soma_vqvae_decoder() {
    vqvae_decoder("soma");
}
#[test]
fn unitree_g1_end_to_end() {
    end_to_end("unitree_g1");
}
#[test]
fn soma_end_to_end() {
    end_to_end("soma");
}
