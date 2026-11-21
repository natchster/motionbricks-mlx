// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Metal runtime vs NVIDIA's upstream PyTorch model on the golden fixtures: the same checks
//! and tolerances as `model/tests/test_parity.py`. Each stage is teacher-forced by writing the
//! PyTorch stage inputs into the runtime's buffers and encoding only that stage.
//!
//! Needs the checkpoints and the sandbox's golden fixtures; prints SKIPPED without them.

mod common;

use std::collections::HashMap;

use common::*;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Buf, Recorder, wait};
use motionbricks_rs::npy::Npy;

fn put(dst: &Buf, v: &[f32]) {
    dst.write_f32(v); // converts to the buffer's element type
}

fn put_i(dst: &Buf, v: &[i32]) {
    put(dst, &v.iter().map(|&x| f32::from_bits(x as u32)).collect::<Vec<_>>());
}

fn mask(n: &Npy) -> Vec<f32> {
    n.to_f32().into_iter().map(|x| (x != 0.0) as i32 as f32).collect()
}

fn setup(name: &str) -> Option<(MetalModel, Vec<(&'static str, HashMap<String, Npy>)>)> {
    let dir = checkpoints(name)?;
    let cases = goldens(name)?;
    Some((MetalModel::from_checkpoints(name, dir, 8).expect("metal model loads"), cases))
}

fn root_network(name: &str) {
    let _gpu = gpu_lock();
    let Some((m, cases)) = setup(name) else { return };
    let b = &m.b;
    let mut f = Failures::default();
    for (case, g) in &cases {
        let v = |k: &str| g[k].to_f32();
        let n = g["pred_num_tokens"].shape[0] as usize;
        for (dst, dst_t, k) in [(&b.gn, &b.gn_t, "root_in.global_root"), (&b.ln, &b.ln_t, "root_in.local_root"), (&b.pn, &b.pn_t, "root_in.poses")] {
            put(dst, &v(k));
            put(dst_t, &v(k));
        }
        put(&b.hg, &mask(&g["root_in.has_global"]));
        put(&b.hl, &mask(&g["root_in.has_local"]));
        put(&b.hp, &mask(&g["root_in.has_poses"]));
        put(&b.allowed, &mask(&g["root_in.allowed"]));
        put_i(&b.nt_in, &g["root_in.num_tokens"].to_i32());
        let mut rec = Recorder::new(&m.gpu);
        m.encode_root(&mut rec, &m.dims(n, true));
        wait(&rec.commit());
        let want = v("root_out.num_token_logits");
        let got = &b.logits.as_slice()[..n * 12];
        let finite: Vec<bool> = want.iter().map(|x| x.is_finite()).collect();
        f.check(got.iter().map(|x| x.is_finite()).collect::<Vec<_>>() == finite, || format!("{case}: masked (-inf) classes differ"));
        let e = rel_err(got, &want, |k| finite[k]);
        f.check(e < 1e-4, || format!("{case}: root logits rel err {e}"));
        let tokens = m.num_tokens(n);
        f.check(tokens == g["root_out.pred_num_tokens"].to_i32(), || format!("{case}: durations {tokens:?}"));
        let e = rel_err(&b.pr.to_f32(n * 64 * 5), &v("root_out.pred_global_root_values"), |_| true);
        f.check(e < 1e-4, || format!("{case}: root path rel err {e}"));
    }
    f.finish();
}

fn pose_network(name: &str) {
    let _gpu = gpu_lock();
    let Some((m, cases)) = setup(name) else { return };
    let b = &m.b;
    let mut f = Failures::default();
    for (case, g) in &cases {
        let v = |k: &str| g[k].to_f32();
        let n = g["pred_num_tokens"].shape[0] as usize;
        // Release configuration: every input pose token is the mask token (folded at load time).
        let toks = g["pose_in.pose_tokens"].to_i32();
        assert!(toks.iter().all(|&t| t == m.spec.vq.codes_per_head), "{case}: expected all-masked pose tokens");
        let nt = g["pose_in.num_tokens"].to_i32();
        put(&b.rcond, &v("pose_in.root_cond"));
        put(&b.pcond, &v("pose_in.pose_cond"));
        put(&b.hpc, &mask(&g["pose_in.has_pose_cond"]));
        put_i(&b.nt, &nt);
        let le: Vec<f32> = (0..n * 16 * 128).map(|k| m.len_emb_host[(nt[k / (16 * 128)] as usize - 6) * 128 + k % 128]).collect();
        put(&b.le, &le);
        put(&b.ppad, &(0..n * 16).map(|k| ((k % 16) as i32 >= nt[k / 16]) as i32 as f32).collect::<Vec<_>>());
        let mut rec = Recorder::new(&m.gpu);
        m.encode_pose(&mut rec, &m.dims(n, false));
        wait(&rec.commit());
        let want = v("pose_out.pose_logits");
        let got = b.plog.to_f32(want.len());
        let e = rel_err(&got, &want, |_| true);
        f.check(e < 1e-4, || format!("{case}: pose logits rel err {e}"));
        let codes = m.spec.vq.codes_per_head as usize;
        f.check(argmax_rows(&got, codes) == argmax_rows(&want, codes), || format!("{case}: argmax tokens differ"));
    }
    f.finish();
}

fn vqvae_decoder(name: &str) {
    let _gpu = gpu_lock();
    let Some((m, cases)) = setup(name) else { return };
    let b = &m.b;
    let mut f = Failures::default();
    let (heads, codes, cw) = (m.spec.vq.num_heads as usize, m.spec.vq.codes_per_head as usize, (m.spec.vq.code_dim / m.spec.vq.num_heads) as usize);
    for (case, g) in &cases {
        let v = |k: &str| g[k].to_f32();
        let n = g["pred_num_tokens"].shape[0] as usize;
        let lat: Vec<f32> = g["vq_in.tokens"]
            .to_i32()
            .iter()
            .enumerate()
            .flat_map(|(k, &t)| {
                let row = (k % heads) * codes + t as usize;
                m.codebook_host[row * cw..(row + 1) * cw].to_vec()
            })
            .collect();
        put(&b.lat, &lat);
        put(&b.ext, &v("vq_in.external_cond"));
        put(&b.pcond, &v("vq_in.target_cond"));
        put(&b.hpc, &mask(&g["vq_in.has_target_cond"]));
        let tm = mask(&g["vq_in.token_mask"]);
        put(&b.m16, &tm);
        put(&b.m32, &tm.iter().flat_map(|&x| [x, x]).collect::<Vec<_>>());
        put(&b.m64, &tm.iter().flat_map(|&x| [x, x, x, x]).collect::<Vec<_>>());
        let mut rec = Recorder::new(&m.gpu);
        m.encode_vq(&mut rec, &m.dims(n, false));
        wait(&rec.commit());
        let want = v("vq_out.recon_state");
        let e = rel_err(&b.recon.to_f32(want.len()), &want, |_| true);
        f.check(e < 1e-4, || format!("{case}: decoder rel err {e}"));
    }
    f.finish();
}

fn end_to_end(name: &str) {
    let _gpu = gpu_lock();
    let Some((m, cases)) = setup(name) else { return };
    let mut f = Failures::default();
    for (case, g) in &cases {
        let (out, durations, contacts) = m.run(&golden_inputs(g)).unwrap();
        check_end_to_end(&mut f, name, case, "metal", g, &out, &durations, contacts.as_deref());
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
