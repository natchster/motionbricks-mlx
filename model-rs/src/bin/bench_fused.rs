// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Reference vs low-latency mode, timed in one process with interleaved rounds (load and
//! clock state shared). Development tool (`--features dev-tools`):
//!
//!     bench-fused [bfloat16] [batch] [rounds]

mod common;

use anyhow::Result;
use common::positionals;
use motionbricks_rs::config::{Precision, checkpoint_dir};
use motionbricks_rs::metal::model::{MetalModel, Mode};
use motionbricks_rs::npy::load_npz;
use std::path::PathBuf;

fn main() -> Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = positionals()?.into_iter();
    let prec = args.next().unwrap_or("bfloat16".into()).parse::<Precision>()?.name();
    let batch = match args.next() {
        Some(v) => common::parse(&v, "batch")?,
        None => 1,
    };
    let rounds = match args.next() {
        Some(v) => common::parse(&v, "rounds")?,
        None => 30,
    };
    for name in ["soma", "unitree_g1"] {
        let reference = MetalModel::from_checkpoints_with(name, checkpoint_dir(name), 64, prec)?;
        let fused = MetalModel::from_checkpoints_with_mode(name, checkpoint_dir(name), 64, prec, Mode::LowLatency)?;
        let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
        // Warm up both (pipelines, GPU clocks), then interleave.
        for _ in 0..5 {
            reference.run(&req)?;
            fused.run(&req)?;
        }
        // GPU time from command-buffer timestamps (immune to CPU-side noise).
        let gpu_ms = |m: &MetalModel| {
            let n = m.upload_inputs(&req).unwrap();
            let mut rec = m.recorder();
            m.encode(&mut rec, &m.dims(n, false));
            rec.commit().wait().unwrap()
        };
        let (mut r_times, mut f_times) = (Vec::new(), Vec::new());
        for _ in 0..rounds {
            r_times.push(gpu_ms(&reference));
            f_times.push(gpu_ms(&fused));
        }
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[0], v[v.len() / 2])
        };
        let (r_min, r_med) = stats(&mut r_times);
        let (f_min, f_med) = stats(&mut f_times);
        println!(
            "{name} {prec} N={batch} GPU ms: reference min {r_min:.3} median {r_med:.3} | low-latency min {f_min:.3} median {f_med:.3} | ratio {:.3}x",
            r_med / f_med
        );
    }
    Ok(())
}
