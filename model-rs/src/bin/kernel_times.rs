// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Per-kernel GPU cost at N=1: each kernel class dispatched in a loop within one command
//! buffer, so the commit/wait overhead amortizes. Development tool (`--features dev-tools`):
//!
//!     kernel-times [bfloat16]

mod common;

use anyhow::Result;
use common::positionals;
use motionbricks_rs::config::{Precision, checkpoint_dir};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::transformer::{GemmCall, GemmParams, Weight};
use motionbricks_rs::metal::{Recorder, wait};

fn main() -> Result<()> {
    let mut args = positionals()?.into_iter();
    let prec = args.next().unwrap_or("bfloat16".into()).parse::<Precision>()?.name();
    let m = MetalModel::from_checkpoints_with("soma", checkpoint_dir("soma"), 64, prec)?;
    let (gpu, k) = (&m.gpu, &m.k);
    // The pose transformer's shapes at N=1: rows=16, d=1024, ffn=2048.
    let (rows, d, f) = (16usize, 1024usize, 2048usize);
    let (x, qkv, attn, proj, x2, ff) = (
        gpu.buffer_t(rows * d),
        gpu.buffer_t(rows * 3 * d),
        gpu.buffer_t(rows * d),
        gpu.buffer_t(rows * d),
        gpu.buffer_t(rows * d),
        gpu.buffer_t(rows * f),
    );
    let (w_qkv, b_qkv) = (gpu.buffer_t(3 * d * d), gpu.buffer_t(3 * d));
    let (w_out, b_out) = (gpu.buffer_t(d * d), gpu.buffer_t(d));
    let (w_up, b_up) = (gpu.buffer_t(f * d), gpu.buffer_t(f));
    let (w_down, b_down) = (gpu.buffer_t(d * f), gpu.buffer_t(d));
    let (ln_w, ln_b) = (gpu.buffer_t(d), gpu.buffer_t(d));
    let (w_qkv, w_out, w_up, w_down) = (Weight::Dense(w_qkv), Weight::Dense(w_out), Weight::Dense(w_up), Weight::Dense(w_down));

    // Warm the GPU clocks first (the first timed block is otherwise several times slower).
    {
        let mut rec = Recorder::new(gpu);
        for _ in 0..3000 {
            k.run(&mut rec, &GemmCall::new(&qkv, &w_qkv, &b_qkv, &qkv, GemmParams::new(rows, 3 * d, d)));
        }
        wait(&rec.commit());
    }
    // Time one kernel body dispatched `iters` times inside one command buffer.
    // GPU time (from command-buffer timestamps), best of 5 rounds.
    let time = |label: &str, iters: usize, body: &dyn Fn(&mut Recorder)| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut rec = Recorder::new(gpu);
            for _ in 0..iters {
                body(&mut rec);
            }
            let ms = wait(&rec.commit()) / iters as f64;
            best = best.min(ms);
        }
        println!("{label:>24}: {best:.4} ms x{iters} (GPU, best of 5)");
    };

    time("gemm qkv (16x3072x1024)", 100, &|r| {
        k.run(r, &GemmCall::new(&qkv, &w_qkv, &b_qkv, &qkv, GemmParams::new(rows, 3 * d, d)));
    });
    time("attention (16 tokens)", 100, &|r| k.attention(r, &qkv, None, &attn, 1, 16, d, 16));
    time("gemm out (16x1024x1024)", 100, &|r| {
        k.run(r, &GemmCall::new(&attn, &w_out, &b_out, &proj, GemmParams::new(rows, d, d)));
    });
    time("gemm ff up (16x2048x1024)", 100, &|r| {
        k.run(r, &GemmCall::new(&x2, &w_up, &b_up, &ff, GemmParams::new(rows, f, d)));
    });
    time("gemm ff down (16x1024x2048)", 100, &|r| {
        k.run(r, &GemmCall::new(&ff, &w_down, &b_down, &proj, GemmParams::new(rows, d, f)));
    });
    time("add_layernorm", 100, &|r| k.add_layernorm(r, &x, &proj, &ln_w, &ln_b, &x2, rows, d));
    time("run_ln (fused, 16x1024x1024)", 100, &|r| {
        k.run_ln(r, &GemmCall::new(&attn, &w_out, &b_out, &x2, GemmParams::new(rows, d, d)), &x, &ln_w, &ln_b);
    });
    time("run_ln (fused, 16x1024x2048)", 100, &|r| {
        k.run_ln(r, &GemmCall::new(&ff, &w_down, &b_down, &x2, GemmParams::new(rows, d, f)), &x2, &ln_w, &ln_b);
    });
    // Pairs, as the layer issues them.
    time("pair: out+reduce+add_ln", 100, &|r| {
        k.run(r, &GemmCall::new(&attn, &w_out, &b_out, &proj, GemmParams::new(rows, d, d)));
        k.add_layernorm(r, &x, &proj, &ln_w, &ln_b, &x2, rows, d);
    });
    time("pair: fused out+LN", 100, &|r| {
        k.run_ln(r, &GemmCall::new(&attn, &w_out, &b_out, &x2, GemmParams::new(rows, d, d)), &x, &ln_w, &ln_b);
    });
    Ok(())
}
