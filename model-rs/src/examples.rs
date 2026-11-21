// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Synthetic but well-formed requests, ported from `motionbricks_mlx/examples.py`, so Rust
//! users and tests need no Python-made request files.
//!
//! The structure and distributions match the Python generator (a standing pose moving forward
//! with a turn; context and target slots filled, last-frame velocities unknown). The random
//! numbers come from a small seeded PCG generator, so the values differ from NumPy's; for
//! byte-identical inputs across ports use the `.npz` requests in `bench/requests/`.

use std::collections::HashMap;

use crate::npy::{Npy, NpyData};

const CONTEXT_AND_TARGET: usize = 8;
const NUM_TOKEN_CHOICES: usize = 11;

/// Seeded PCG32 with uniform and normal samples.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        r.next_u32();
        r
    }
    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }
    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Normal with mean `mu` and standard deviation `sigma` (Box-Muller).
    pub fn normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let u1 = self.uniform(f32::EPSILON, 1.0) as f64;
        let u2 = self.uniform(0.0, 1.0) as f64;
        mu + sigma * ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

fn f32s(data: Vec<f32>, shape: &[usize]) -> Npy {
    Npy { shape: shape.iter().map(|&d| d as i32).collect(), data: NpyData::F32(data) }
}

fn i32s(data: Vec<i32>, shape: &[usize]) -> Npy {
    Npy { shape: shape.iter().map(|&d| d as i32).collect(), data: NpyData::I32(data) }
}

/// `has` (context + 4 targets) and `has_local` (velocities unknown at the last context pose and
/// the last target pose), each `(batch, 8)`.
fn masks(batch: usize) -> (Npy, Npy) {
    let num_targets = 4;
    let row: Vec<i32> = (0..CONTEXT_AND_TARGET).map(|k| (k < 4 + num_targets) as i32).collect();
    let mut local = row.clone();
    local[3] = 0;
    local[3 + num_targets] = 0;
    let rep = |r: &Vec<i32>| r.iter().copied().cycle().take(batch * CONTEXT_AND_TARGET).collect::<Vec<_>>();
    (i32s(rep(&row), &[batch, CONTEXT_AND_TARGET]), i32s(rep(&local), &[batch, CONTEXT_AND_TARGET]))
}

fn allowed(batch: usize) -> Npy {
    i32s(vec![1; batch * NUM_TOKEN_CHOICES], &[batch, NUM_TOKEN_CHOICES])
}

/// Unitree G1: `context_qpos`/`target_qpos` `(B, 4, 36)` (root xyz, quaternion XYZW, 29 joints).
pub fn g1_request(batch: usize, seed: u64, distance: f32, turn: f32) -> HashMap<String, Npy> {
    let mut rng = Rng::new(seed);
    let (mut context, mut target) = (vec![0f32; batch * 4 * 36], vec![0f32; batch * 4 * 36]);
    for b in 0..batch {
        let joints: Vec<f32> = (0..29).map(|_| rng.normal(0.0, 0.05)).collect();
        let x = distance + rng.uniform(-0.2, 0.2);
        let y = rng.uniform(-0.3, 0.3);
        let yaw = turn + rng.uniform(-0.3, 0.3);
        for t in 0..4 {
            let c = &mut context[(b * 4 + t) * 36..(b * 4 + t + 1) * 36];
            c[2] = 0.75; // pelvis height (Z up)
            c[6] = 1.0; // identity quaternion, XYZW
            c[7..].copy_from_slice(&joints);
            let g = &mut target[(b * 4 + t) * 36..(b * 4 + t + 1) * 36];
            g.copy_from_slice(&context[(b * 4 + t) * 36..(b * 4 + t + 1) * 36]);
            g[0] = x;
            g[1] = y;
            g[5] = (yaw / 2.0).sin();
            g[6] = (yaw / 2.0).cos();
        }
    }
    let (has, has_local) = masks(batch);
    HashMap::from([
        ("context_qpos".into(), f32s(context, &[batch, 4, 36])),
        ("target_qpos".into(), f32s(target, &[batch, 4, 36])),
        ("has_global_root_values".into(), has.clone()),
        ("has_local_root_values".into(), has_local),
        ("has_local_poses".into(), has),
        ("allowed_pred_num_tokens".into(), allowed(batch)),
    ])
}

/// SOMA: root path `(B, 8, 5)`, local root `(B, 8, 4)`, local poses `(B, 8, 267)`.
pub fn soma_request(batch: usize, seed: u64, distance: f32) -> HashMap<String, Npy> {
    let mut rng = Rng::new(seed);
    let (mut global_root, mut local_root, mut local_poses) = (vec![0f32; batch * 8 * 5], vec![0f32; batch * 8 * 4], vec![0f32; batch * 8 * 267]);
    for b in 0..batch {
        let speed = rng.uniform(0.5, 1.5);
        let heading = rng.uniform(-0.5, 0.5);
        let ric: Vec<f32> = (0..87).map(|_| rng.normal(0.0, 0.3)).collect();
        let rot: Vec<f32> = (0..180).map(|k| [1.0, 0.0, 0.0, 0.0, 1.0, 0.0][k % 6] + rng.normal(0.0, 0.02)).collect();
        for t in 0..8 {
            let tf = t as f32;
            let g = &mut global_root[(b * 8 + t) * 5..(b * 8 + t + 1) * 5];
            g[0] = if t < 4 { tf * speed / 30.0 } else { distance + (tf - 4.0) * speed / 30.0 };
            g[1] = 0.95; // hip height (Y up)
            let h = if t < 4 { 0.0 } else { heading };
            g[3] = h.cos();
            g[4] = h.sin();
            let l = &mut local_root[(b * 8 + t) * 4..(b * 8 + t + 1) * 4];
            l[1] = speed;
            l[3] = 0.95;
            let p = &mut local_poses[(b * 8 + t) * 267..(b * 8 + t + 1) * 267];
            p[..87].copy_from_slice(&ric);
            p[87..].copy_from_slice(&rot);
        }
    }
    let (has, has_local) = masks(batch);
    HashMap::from([
        ("global_root_values".into(), f32s(global_root, &[batch, 8, 5])),
        ("has_global_root_values".into(), has.clone()),
        ("local_root_values".into(), f32s(local_root, &[batch, 8, 4])),
        ("has_local_root_values".into(), has_local),
        ("local_poses".into(), f32s(local_poses, &[batch, 8, 267])),
        ("has_local_poses".into(), has),
        ("allowed_pred_num_tokens".into(), allowed(batch)),
    ])
}

/// A request for `name` ("soma" or "unitree_g1") with `batch` rows (Python: `examples.request`).
pub fn request(name: &str, batch: usize, seed: u64) -> HashMap<String, Npy> {
    if name == "unitree_g1" || name == "g1" { g1_request(batch, seed, 1.0, 0.0) } else { soma_request(batch, seed, 1.5) }
}

/// Rows `lo..hi` of every input (for batch-independence checks).
pub fn rows(request: &HashMap<String, Npy>, lo: usize, hi: usize) -> HashMap<String, Npy> {
    request
        .iter()
        .map(|(k, v)| {
            let per: usize = v.shape[1..].iter().map(|&d| d as usize).product();
            let mut shape = v.shape.clone();
            shape[0] = (hi - lo) as i32;
            let data = match &v.data {
                NpyData::F32(d) => NpyData::F32(d[lo * per..hi * per].to_vec()),
                NpyData::I32(d) => NpyData::I32(d[lo * per..hi * per].to_vec()),
                NpyData::F64(d) => NpyData::F64(d[lo * per..hi * per].to_vec()),
                NpyData::I64(d) => NpyData::I64(d[lo * per..hi * per].to_vec()),
                NpyData::Bool(d) => NpyData::Bool(d[lo * per..hi * per].to_vec()),
            };
            (k.clone(), Npy { shape, data })
        })
        .collect()
}
