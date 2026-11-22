// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! GPU time per stage of the Metal runtime (each stage re-run on valid buffers from a full pass).
//! Development tool (`--features dev-tools`):
//!
//!     metal-stages [float32|float16|bfloat16|int8] [batches...]
//!
//! Checkpoints: `$MOTIONBRICKS_CHECKPOINTS` or `checkpoints/MotionBricks`.

mod common;

use std::path::PathBuf;

use anyhow::Result;
use common::{median, positionals};
use motionbricks_rs::config::{Precision, checkpoint_dir};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Recorder, wait};
use motionbricks_rs::npy::load_npz;

fn main() -> Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = positionals()?.into_iter();
    let prec = args.next().unwrap_or("float32".into()).parse::<Precision>()?.name();
    let batches = args.map(|v| common::parse(&v, "batch size")).collect::<Result<Vec<usize>>>()?;
    let batches = if batches.is_empty() { vec![1, 32] } else { batches };
    for name in ["soma", "unitree_g1"] {
        let m = MetalModel::from_checkpoints_with(name, checkpoint_dir(name), 64, prec)?;
        for &batch in &batches {
            let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
            m.run(&req)?;
            let d = m.dims(batch, false);
            let time = |f: &dyn Fn(&mut Recorder)| -> f64 {
                let mut v: Vec<f64> = (0..15).map(|_| { let mut r = Recorder::new(&m.gpu); f(&mut r); wait(&r.commit()) }).collect();
                median(&mut v)
            };
            let input = time(&|r| m.encode_input(r, &d));
            let root = time(&|r| m.encode_root(r, &d));
            let r2p = time(&|r| m.encode_root_to_pose(r, &d));
            let pose = time(&|r| m.encode_pose(r, &d));
            let vq = time(&|r| m.encode_vq(r, &d));
            let fin = time(&|r| m.encode_finish(r, &d));
            let all = time(&|r| m.encode(r, &d));
            let mut probe = Recorder::new(&m.gpu);
            m.encode(&mut probe, &d);
            let dispatches = probe.dispatches;
            wait(&probe.commit());
            // End to end (upload, encode, wait, read back): chunked commits vs one command buffer.
            let e2e = |chunked: bool| -> Result<f64> {
                let mut v: Vec<f64> = (0..25)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        let n = m.upload_inputs(&req)?;
                        let mut r = if chunked { m.recorder() } else { Recorder::new(&m.gpu) };
                        m.encode(&mut r, &m.dims(n, false));
                        wait(&r.commit());
                        std::hint::black_box(m.outputs(n));
                        Ok(t.elapsed().as_secs_f64() * 1e3)
                    })
                    .collect::<Result<_>>()?;
                Ok(median(&mut v))
            };
            let (single, chunked) = (e2e(false)?, e2e(true)?);
            println!("{name:<10} {prec:<8} N={batch:<2} end to end ms: one command buffer {single:.2} | chunked commits {chunked:.2}");
            println!("{name:<10} {prec:<8} N={batch:<2} GPU ms: input {input:.2} | root {root:.2} | root->pose {r2p:.2} | pose {pose:.2} | vq {vq:.2} | finish {fin:.2} | whole {all:.2} | {dispatches} dispatches");
        }
    }
    Ok(())
}
