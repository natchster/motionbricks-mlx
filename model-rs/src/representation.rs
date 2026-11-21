// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Motion feature layouts, normalization, and global <-> local root conversion
//! (ported from `motionbricks_mlx/representation.py`). Index tables are built once on the CPU.

use mlx_rs::{Array, ops};

use crate::ops::*;

pub const FPS: f32 = 30.0;
const STATS_EPS: f32 = 1e-5;

/// Index table of one feature layout (ordered keys -> contiguous ranges).
#[derive(Debug, Clone)]
pub struct Layout {
    pub keys: Vec<(&'static str, i32, i32)>, // (name, start, stop)
    pub dim: i32,
    pub root_dim: i32,
}

impl Layout {
    fn build(root: &[(&'static str, i32)], body: &[(&'static str, i32)]) -> Layout {
        let mut keys = Vec::new();
        let mut offset = 0;
        for &(k, n) in root.iter().chain(body) {
            keys.push((k, offset, offset + n));
            offset += n;
        }
        Layout { keys, dim: offset, root_dim: root.iter().map(|r| r.1).sum() }
    }

    pub fn range(&self, key: &str) -> Vec<i32> {
        let &(_, a, b) = self.keys.iter().find(|k| k.0 == key).unwrap_or_else(|| panic!("no key {key}"));
        (a..b).collect()
    }

    pub fn body(&self) -> Vec<i32> {
        (self.root_dim..self.dim).collect()
    }
}

type Stats = (Array, Array);

pub struct MotionRepresentation {
    pub global: Layout,
    pub local: Layout,
    pub global_root_no_height: Array, // int32 index arrays for gathers
    pub local_root_velocity: Array,
    pub local_pose_dim: i32,
    g_root: Stats,
    l_root: Stats,
    l_pose: Stats,
    l_body: Stats,
}

fn body_keys(num_joints: i32) -> Vec<(&'static str, i32)> {
    vec![
        ("ric_data", (num_joints - 1) * 3),
        ("global_rot_data", num_joints * 6),
        ("local_vel", num_joints * 3),
        ("foot_contacts", 4),
    ]
}

impl MotionRepresentation {
    pub fn new(num_joints: i32, mean: &[f32], std: &[f32]) -> anyhow::Result<Self> {
        let body = body_keys(num_joints);
        let groot = [("global_root_pos", 3), ("global_root_heading", 2)];
        let lroot = [("local_root_rot_vel", 1), ("local_root_vel", 2), ("global_root_y", 1)];
        let global = Layout::build(&groot, &body);
        let local = Layout::build(&lroot, &body);
        let dual_root: Vec<_> = groot.iter().chain(lroot.iter()).copied().collect();
        let dual = Layout::build(&dual_root, &body);
        anyhow::ensure!(mean.len() == dual.dim as usize && std.len() == dual.dim as usize, "stats must have {} entries", dual.dim);

        let d_body = dual.body();
        let global_to_dual: Vec<i32> = (0..global.root_dim).chain(d_body.iter().copied()).collect();
        let local_to_dual: Vec<i32> = (global.root_dim..dual.root_dim).chain(d_body.iter().copied()).collect();
        let scale: Vec<f32> = std.iter().map(|&s| (s * s + STATS_EPS).sqrt()).collect();
        let stats = |dual_index: &[i32]| -> Stats {
            let m: Vec<f32> = dual_index.iter().map(|&j| mean[j as usize]).collect();
            let s: Vec<f32> = dual_index.iter().map(|&j| scale[j as usize]).collect();
            (Array::from_slice(&m, &[m.len() as i32]), Array::from_slice(&s, &[s.len() as i32]))
        };
        let pick = |table: &[i32], idx: &[i32]| -> Vec<i32> { idx.iter().map(|&k| table[k as usize]).collect() };

        let local_pose: Vec<i32> = [local.range("global_root_y"), local.range("ric_data"), local.range("global_rot_data")].concat();
        let gpos = global.range("global_root_pos");
        let no_height: Vec<i32> = [vec![gpos[0], gpos[2]], global.range("global_root_heading")].concat();
        let velocity = local.range("local_root_vel");

        Ok(MotionRepresentation {
            g_root: stats(&pick(&global_to_dual, &(0..global.root_dim).collect::<Vec<_>>())),
            l_root: stats(&pick(&local_to_dual, &(0..local.root_dim).collect::<Vec<_>>())),
            l_pose: stats(&pick(&local_to_dual, &local_pose)),
            l_body: stats(&pick(&local_to_dual, &local.body())),
            global_root_no_height: Array::from_slice(&no_height, &[no_height.len() as i32]),
            local_root_velocity: Array::from_slice(&velocity, &[velocity.len() as i32]),
            local_pose_dim: local_pose.len() as i32,
            global,
            local,
        })
    }

    /// (mean, scale) vectors of the global root, local root, local pose and local body features.
    pub fn stats_f32(&self) -> R<[(Vec<f32>, Vec<f32>); 4]> {
        let v = |s: &Stats| -> R<(Vec<f32>, Vec<f32>)> { Ok((crate::ops::to_vec_f32(&s.0)?, crate::ops::to_vec_f32(&s.1)?)) };
        Ok([v(&self.g_root)?, v(&self.l_root)?, v(&self.l_pose)?, v(&self.l_body)?])
    }

    fn norm(x: &Array, s: &Stats) -> R<Array> {
        ops::divide(x - &s.0, &s.1)
    }
    fn unnorm(x: &Array, s: &Stats) -> R<Array> {
        ops::add(x * &s.1, &s.0)
    }
    pub fn normalize_global_root(&self, x: &Array) -> R<Array> {
        Self::norm(x, &self.g_root)
    }
    pub fn unnormalize_global_root(&self, x: &Array) -> R<Array> {
        Self::unnorm(x, &self.g_root)
    }
    pub fn normalize_local_root(&self, x: &Array) -> R<Array> {
        Self::norm(x, &self.l_root)
    }
    pub fn unnormalize_local_root(&self, x: &Array) -> R<Array> {
        Self::unnorm(x, &self.l_root)
    }
    pub fn normalize_local_pose(&self, x: &Array) -> R<Array> {
        Self::norm(x, &self.l_pose)
    }

    /// `global_root_to_local_onnx_compatible`: normalized global root `(B, T, 5)` -> normalized
    /// local root `(B, T, 4)`; the last frame of each request repeats the frame before it.
    pub fn global_root_to_local(&self, normalized_global_root: &Array, lengths: &Array) -> R<Array> {
        let root = self.unnormalize_global_root(normalized_global_root)?;
        let positions = slice(&root, -1, 0, 3)?;
        let heading = atan2(&index(&root, -1, 4)?, &index(&root, -1, 3)?)?;
        let half = &heading / f(2.0);
        let angles = &atan2(&sin(&half)?, &cos(&half)?)? * f(2.0);
        let (c, s) = (cos(&angles)?, sin(&angles)?);
        let (c1, c0) = (slice(&c, 1, 1, END)?, slice(&c, 1, 0, -1)?);
        let (s1, s0) = (slice(&s, 1, 1, END)?, slice(&s, 1, 0, -1)?);
        let cos_diff = ops::add(&c1 * &c0, &s1 * &s0)?;
        let sin_diff = ops::subtract(&s1 * &c0, &c1 * &s0)?;
        let angular = &atan2(&sin_diff, &cos_diff)? * f(FPS);
        let angular = cat(&[&angular, &ops::zeros_like(&slice(&angular, 1, 0, 1)?)?], 1)?;
        let velocity = &ops::subtract(slice(&positions, 1, 1, END)?, slice(&positions, 1, 0, -1)?)? * f(FPS);
        let velocity = cat(&[&velocity, &ops::zeros_like(&slice(&velocity, 1, 0, 1)?)?], 1)?;

        let last = unsq(&(&lengths.as_dtype(mlx_rs::Dtype::Int32)? - 1), 1)?; // (B, 1)
        let is_last = unsq(&arange(root.dim(1))?, 0)?.eq(&last)?;
        let prev_angular = angular.take_along_axis(&last - 1, 1)?;
        let prev_velocity = velocity.take_along_axis(unsq(&(&last - 1), -1)?, 1)?;
        let angular = wh(&is_last, &prev_angular, &angular)?;
        let velocity = wh(&unsq(&is_last, -1)?, &prev_velocity, &velocity)?;
        // COMPUTE_KWARGS["removing_heading"] is False: velocities stay in the global frame.
        let local_root = cat(
            &[&unsq(&angular, -1)?, &slice(&velocity, -1, 0, 1)?, &slice(&velocity, -1, 2, 3)?, &slice(&positions, -1, 1, 2)?],
            -1,
        )?;
        self.normalize_local_root(&local_root)
    }

    /// Decoded local features (normalized) -> global features (unnormalized).
    pub fn local_to_global(&self, normalized_local: &Array) -> R<Array> {
        let root = self.unnormalize_local_root(&slice(normalized_local, -1, 0, 4)?)?;
        let body = Self::unnorm(&slice(normalized_local, -1, 4, END)?, &self.l_body)?;
        let yaw_rate = &index(&root, -1, 0)? / f(FPS);
        let planar = &slice(&root, -1, 1, 3)? / f(FPS);
        let zeros = ops::zeros_like(&slice(&yaw_rate, 1, 0, 1)?)?;
        let yaw = cat(&[&zeros, &slice(&yaw_rate, 1, 0, -1)?], 1)?.cumsum(1, None, None)?;
        let xz = cat(&[&ops::zeros_like(&slice(&planar, 1, 0, 1)?)?, &slice(&planar, 1, 0, -1)?], 1)?.cumsum(1, None, None)?;
        let half = &yaw / f(2.0);
        let heading = &atan2(&sin(&half)?, &cos(&half)?)? * f(2.0);
        let global_root = stack(
            &[&index(&xz, -1, 0)?, &index(&root, -1, 3)?, &index(&xz, -1, 1)?, &cos(&heading)?, &sin(&heading)?],
            -1,
        )?;
        cat(&[&global_root, &body], -1)
    }
}
