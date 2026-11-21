// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Generate motions with the custom Metal runtime (release configuration).
//!
//!     cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 8 [bfloat16]
//!     cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 bench/requests/unitree_g1_N8.npz
//!
//! The third argument is a batch size (a synthetic request from `motionbricks_rs::examples`) or a
//! request `.npz` with the ONNX input names. The optional last argument is the precision:
//! float32 (default), float16, bfloat16 or int8.

use std::time::Instant;

use anyhow::Result;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::npy::load_npz;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (name, checkpoints, request, precision) = match &args[..] {
        [_, n, c, r] => (n, c, r, "float32"),
        [_, n, c, r, p] => (n, c, r, p.as_str()),
        _ => anyhow::bail!("usage: generate_metal <soma|unitree_g1> <checkpoint dir> <batch | request.npz> [float32|float16|bfloat16|int8]"),
    };
    let model = MetalModel::from_checkpoints_with(name, checkpoints, 64, precision)?;
    let inputs = match request.parse::<usize>() {
        Ok(batch) => motionbricks_rs::examples::request(name, batch, 0),
        Err(_) => load_npz(request)?,
    };
    model.run(&inputs)?; // warm-up
    let t = Instant::now();
    let (motion, tokens, contacts) = model.run(&inputs)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    for (i, n) in tokens.iter().enumerate() {
        println!("request {i}: {n} tokens = {} frames ({:.2} s of motion)", 4 * n, 4.0 * *n as f32 / 30.0);
    }
    println!("{precision}: {} values{} in {ms:.2} ms", motion.len(), contacts.map(|c| format!(" + {} contact values", c.len())).unwrap_or_default());
    Ok(())
}
