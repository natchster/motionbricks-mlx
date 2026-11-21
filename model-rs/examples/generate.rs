// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Generate motions with the mlx-rs port and print the chosen durations.
//!
//!     cargo run --release --example generate -- soma ../checkpoints/MotionBricks/soma 8 [float32|float16|bfloat16|int8]
//!     cargo run --release --example generate -- soma ../checkpoints/MotionBricks/soma bench/requests/soma_N8.npz
//!
//! The third argument is a batch size (a synthetic request from `motionbricks_rs::examples`) or a
//! request `.npz` with the ONNX input names.

use std::rc::Rc;

use anyhow::Result;
use mlx_rs::transforms::eval;
use motionbricks_rs::npy::load_npz;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, Precision, compiled, examples};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (name, checkpoints, request, precision) = match &args[..] {
        [_, n, c, r] => (n, c, r, "float32"),
        [_, n, c, r, p] => (n, c, r, p.as_str()),
        _ => anyhow::bail!("usage: generate <soma|unitree_g1> <checkpoint dir> <batch | request.npz> [float32|float16|bfloat16|int8]"),
    };
    let precision: Precision = precision.parse()?;
    let model = Rc::new(MotionBricks::load(name, checkpoints, precision)?); // INT8: float32 + quantize(8, 64)
    let request = match request.parse::<usize>() {
        Ok(batch) => examples::request(name, batch, 0)?,
        Err(_) => load_npz(request)?,
    };
    let mut forward = compiled(model.clone());
    let outputs = forward(&model.prepare(&request)?)?;
    eval(outputs.iter())?;
    let tokens = to_vec_i32(&outputs[1])?;
    let motion = to_vec_f32(&outputs[0])?;
    for (i, n) in tokens.iter().enumerate() {
        println!("request {i}: {n} tokens = {} frames at 30 fps ({:.2} s)", 4 * n, 4.0 * *n as f32 / 30.0);
    }
    println!("{precision}: {} {:?}, {} values", model.output_names()[0], outputs[0].shape(), motion.len());
    Ok(())
}
