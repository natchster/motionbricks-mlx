// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Where reference and low-latency modes diverge: stage-by-stage buffer comparison.
//! Development tool (`--features dev-tools`):
//!
//!     debug-fused [bfloat16] [batch]

mod common;

use anyhow::Result;
use common::positionals;
use motionbricks_rs::config::{Precision, checkpoint_dir};
use motionbricks_rs::metal::model::{MetalModel, Mode};
use motionbricks_rs::metal::{Recorder, wait};
use motionbricks_rs::npy::load_npz;
use std::path::PathBuf;

fn main() -> Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = positionals()?.into_iter();
    let prec = args.next().unwrap_or("bfloat16".into()).parse::<Precision>()?.name();
    let batch = match args.next() { Some(v) => common::parse(&v, "batch")?, None => 1 };
    let model = std::env::var("MODEL").unwrap_or_else(|_| "soma".into());
    let reference = MetalModel::from_checkpoints_with(&model, checkpoint_dir(&model), 64, prec)?;
    let fused = MetalModel::from_checkpoints_with_mode(&model, checkpoint_dir(&model), 64, prec, Mode::LowLatency)?;
    let req = load_npz(here.join("bench/requests").join(format!("{model}_N{batch}.npz")))?;
    let n = reference.upload_inputs(&req)?;
    if model != "soma" {
        let out = |m: &MetalModel| m.run(&req).unwrap();
        let (r, f, r32) = (out(&reference), out(&fused), out(&MetalModel::from_checkpoints_with(&model, checkpoint_dir(&model), 64, "float32")?));
        let med = |a: &[f32], b: &[f32]| {
            let mut d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
            d.sort_by(|x, y| x.partial_cmp(y).unwrap());
            (d[d.len() / 2], d.iter().copied().fold(0f32, f32::max))
        };
        let (m1, x1) = med(&f.0, &r.0);
        let (m2, x2) = med(&r.0, &r32.0);
        println!("motion: fused vs reference ({prec}): median {m1:.5} max {x1:.5}");
        println!("motion: {prec} vs fp32 (reference): median {m2:.5} max {x2:.5}   <- accepted noise class");
        if let (Some(fc), Some(rc), Some(c32)) = (&f.2, &r.2, &r32.2) {
            let (m1, x1) = med(fc, rc);
            let (m2, x2) = med(rc, c32);
            println!("contacts: fused vs reference ({prec}): median {m1:.5} max {x1:.5}");
            println!("contacts: {prec} vs fp32 (reference): median {m2:.5} max {x2:.5}   <- accepted noise class");
        }
        return Ok(());
    }
    fused.upload_inputs(&req)?;
    let d = reference.dims(n, false);

    let run_stage = |m: &MetalModel, f: &dyn Fn(&MetalModel, &mut Recorder)| {
        let mut rec = Recorder::new(&m.gpu);
        f(m, &mut rec);
        wait(&rec.commit());
    };
    let mut cmp = |label: &str, stage: &dyn Fn(&MetalModel, &mut Recorder), buf: &dyn Fn(&MetalModel) -> &motionbricks_rs::metal::Buf, len: usize| {
        run_stage(&reference, stage);
        run_stage(&fused, stage);
        let (a, b) = (buf(&reference).to_f32(len), buf(&fused).to_f32(len));
        let e = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!("{label:>12}  max abs {e:.6}  (buffer max {scale:.3})");
    };
    cmp("input", &|m, r| m.encode_input(r, &d), &|m| &m.b.pn_t, batch * 8 * 268);
    cmp("root_a", &|m, r| m.encode_root_a(r, &d), &|m| &m.b.x1, batch * 9 * 512);
    cmp("root", &|m, r| m.encode_root(r, &d), &|m| &m.b.x2, batch * 25 * 512);
    cmp("root->pose", &|m, r| m.encode_root_to_pose(r, &d), &|m| &m.b.pcond, batch * 64 * 640);
    cmp("pose", &|m, r| m.encode_pose(r, &d), &|m| &m.b.px, batch * 16 * 1024);
    cmp("vq", &|m, r| m.encode_vq(r, &d), &|m| &m.b.recon, batch * 64 * 366);
    cmp("finish", &|m, r| m.encode_finish(r, &d), &|m| &m.b.out, batch * 64 * 366);

    // End-to-end outputs: fused bf16 vs reference bf16, and reference bf16 vs fp32
    // (the accepted low-precision noise class), both as median |diff|.
    let fp32 = MetalModel::from_checkpoints_with(&model, checkpoint_dir(&model), 64, "float32")?;
    let out = |m: &MetalModel| m.run(&req).unwrap().0;
    let (r_bf16, f_bf16, r_fp32) = (out(&reference), out(&fused), out(&fp32));
    let mut med = |a: &[f32], b: &[f32]| {
        let mut d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
        d.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (d[d.len() / 2], d.iter().copied().fold(0f32, f32::max))
    };
    let (m1, x1) = med(&f_bf16, &r_bf16);
    let (m2, x2) = med(&r_bf16, &r_fp32);
    println!("fused vs reference (bf16): median {m1:.5} max {x1:.5}");
    println!("bf16 vs fp32 (reference): median {m2:.5} max {x2:.5}   <- the accepted noise class");
    Ok(())
}
