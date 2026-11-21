// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Generate motions with the custom Metal runtime (FP32, release configuration).
//!
//!     cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 bench/requests/unitree_g1_N8.npz

use std::time::Instant;

use anyhow::Result;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::npy::load_npz;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, name, checkpoints, request] = &args[..] else {
        anyhow::bail!("usage: generate_metal <soma|unitree_g1> <checkpoint dir> <request.npz>");
    };
    let model = MetalModel::from_checkpoints(name, checkpoints, 64)?;
    let inputs = load_npz(request)?;
    model.run(&inputs)?; // warm-up
    let t = Instant::now();
    let (motion, tokens, contacts) = model.run(&inputs)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    for (i, n) in tokens.iter().enumerate() {
        println!("request {i}: {n} tokens = {} frames ({:.2} s of motion)", 4 * n, 4.0 * *n as f32 / 30.0);
    }
    println!("{} values{} in {ms:.2} ms", motion.len(), contacts.map(|c| format!(" + {} contact values", c.len())).unwrap_or_default());
    Ok(())
}
