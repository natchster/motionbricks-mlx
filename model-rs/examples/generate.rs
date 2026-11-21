// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Generate motions for a request stored as .npz (ONNX input names), print durations.
//!
//!     cargo run --release --example generate -- soma ../checkpoints/MotionBricks/soma bench/requests/soma_N8.npz

use std::rc::Rc;

use anyhow::Result;
use mlx_rs::Dtype;
use mlx_rs::transforms::eval;
use motionbricks_rs::npy::load_npz;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, name, checkpoints, request] = &args[..] else {
        anyhow::bail!("usage: generate <soma|unitree_g1> <checkpoint dir> <request.npz>");
    };
    let model = Rc::new(MotionBricks::from_checkpoints(name, checkpoints, Dtype::Float32)?);
    let mut forward = compiled(model.clone());
    let inputs = model.prepare(&load_npz(request)?)?;
    let outputs = forward(&inputs)?;
    eval(outputs.iter())?;
    let tokens = to_vec_i32(&outputs[1])?;
    let motion = to_vec_f32(&outputs[0])?;
    let shape = outputs[0].shape();
    for (i, n) in tokens.iter().enumerate() {
        println!("request {i}: {n} tokens = {} frames at 30 fps ({:.2} s)", 4 * n, 4.0 * *n as f32 / 30.0);
    }
    println!("{} {:?}: {} values", model.output_names()[0], shape, motion.len());
    Ok(())
}
