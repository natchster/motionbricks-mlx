// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Motion feature layouts, normalization statistics, and global <-> local root
//! conversion, emitted as ONNX nodes. Port of upstream `models/representation.py`
//! (following the verified `model-rs/src/representation.rs` math); every shape the
//! torch tracer defers into the graph is resolved here at emit time — only the
//! last-frame velocity patch (indexed by the runtime `lengths`) stays in the graph.
//!
//! Feature layouts (`DualRootGlobalJoints` with `COMPUTE_KWARGS`):
//! - global rep: root 5 (pos xyz + heading cos/sin) then body.
//! - local rep: root 4 (rot_vel, vel x/z, hip height y) then body.
//! - body (shared): ric (J-1)*3 + rot6d J*6 + local_vel J*3 + foot_contacts 4.
//! - stats (dual layout): global root 5 ++ local root 4 ++ body (G1: 418, SOMA: 370);
//!   the global/local reps each select their own root slice plus the shared body slice.

use anyhow::{Result, ensure};

use crate::config::{FPS, ModelSpec};
use crate::graph::{DType, Dim, Graph, broadcast};
use crate::ops::{self, c_scalar, c_scalar_i64};

/// Which feature layout a tensor uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepKind {
    Global,
    Local,
}

impl RepKind {
    fn name(self) -> &'static str {
        match self {
            RepKind::Global => "global",
            RepKind::Local => "local",
        }
    }
}

/// Static index tables of the global and local feature layouts and of the dual-layout
/// statistics vectors. All ranges are `(start, end)` offsets; body sub-ranges are
/// relative to the start of the body block (identical in the global, local, and dual
/// layouts — only the block start differs).
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub num_joints: i64,
    /// Root widths: global 5 (pos 3 + heading 2), local 4 (rot_vel 1 + vel 2 + y 1).
    pub global_root_dim: i64,
    pub local_root_dim: i64,
    /// Body width: ric (J-1)*3 + rot6d J*6 + local_vel J*3 + foot_contacts 4.
    pub body_dim: i64,
    /// Full rep widths (root + body): G1 414/413, SOMA 366/365.
    pub global_dim: i64,
    pub local_dim: i64,
    /// Stats (dual) width: 5 + 4 + body: G1 418, SOMA 370.
    pub dual_dim: i64,
    /// Body sub-ranges, relative to the body block start.
    pub ric: (i64, i64),
    pub rot6d: (i64, i64),
    pub local_vel: (i64, i64),
    pub foot_contacts: (i64, i64),
    /// Ranges of the dual-layout stats vector (`indices["global_root"]`,
    /// `indices["local_root"]`, `indices["body"]` of upstream `DualRootLocalBody`).
    pub stats_global_root: (i64, i64),
    pub stats_local_root: (i64, i64),
    pub stats_body: (i64, i64),
}

impl Layout {
    pub fn for_spec(spec: &ModelSpec) -> Layout {
        let j = spec.num_joints;
        let ric = (0, (j - 1) * 3);
        let rot6d = (ric.1, ric.1 + j * 6);
        let local_vel = (rot6d.1, rot6d.1 + j * 3);
        let foot_contacts = (local_vel.1, local_vel.1 + 4);
        let body_dim = foot_contacts.1;
        let (global_root_dim, local_root_dim) = (5, 4);
        Layout {
            num_joints: j,
            global_root_dim,
            local_root_dim,
            body_dim,
            global_dim: global_root_dim + body_dim,
            local_dim: local_root_dim + body_dim,
            dual_dim: global_root_dim + local_root_dim + body_dim,
            ric,
            rot6d,
            local_vel,
            foot_contacts,
            stats_global_root: (0, global_root_dim),
            stats_local_root: (global_root_dim, global_root_dim + local_root_dim),
            stats_body: (global_root_dim + local_root_dim, global_root_dim + local_root_dim + body_dim),
        }
    }

    /// Range of the body block inside a global-rep feature vector.
    pub fn global_body(&self) -> (i64, i64) {
        (self.global_root_dim, self.global_dim)
    }

    /// Range of the body block inside a local-rep feature vector.
    pub fn local_body(&self) -> (i64, i64) {
        (self.local_root_dim, self.local_dim)
    }

    /// Body sub-range `key` ("ric" | "rot6d" | "local_vel" | "foot_contacts") inside a
    /// `rep` feature vector (block start + relative range).
    pub fn range(&self, rep: RepKind, key: &str) -> (i64, i64) {
        let base = match rep {
            RepKind::Global => self.global_root_dim,
            RepKind::Local => self.local_root_dim,
        };
        let (a, b) = match key {
            "ric" => self.ric,
            "rot6d" => self.rot6d,
            "local_vel" => self.local_vel,
            "foot_contacts" => self.foot_contacts,
            other => panic!("no body key {other}"),
        };
        (base + a, base + b)
    }
}

/// Elementwise `x * 0` (upstream's `zeros_like` padding frames).
fn zeros_like(g: &mut Graph, scope: &str, x: &str) -> String {
    let zero = c_scalar(g, scope, 0.0);
    ops::mul(g, scope, x, &zero)
}

/// Elementwise int64 subtraction (length/index arithmetic; `ops::sub` is float-only).
fn sub_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let out = broadcast(g.shape(a), g.shape(b)).expect("i64 sub shapes");
    g.node(scope, "Sub", &[a, b], &[], DType::I64, &out)
}

/// 3D cross product of the last axis (`a x b`, both `[..., 3]`).
fn cross3(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let r = g.shape(a).len() as i64 - 1;
    let a0 = ops::slice_axis(g, scope, a, r, 0, 1);
    let a1 = ops::slice_axis(g, scope, a, r, 1, 2);
    let a2 = ops::slice_axis(g, scope, a, r, 2, 3);
    let b0 = ops::slice_axis(g, scope, b, r, 0, 1);
    let b1 = ops::slice_axis(g, scope, b, r, 1, 2);
    let b2 = ops::slice_axis(g, scope, b, r, 2, 3);
    let c0 = {
        let (p, q) = (ops::mul(g, scope, &a1, &b2), ops::mul(g, scope, &a2, &b1));
        ops::sub(g, scope, &p, &q)
    };
    let c1 = {
        let (p, q) = (ops::mul(g, scope, &a2, &b0), ops::mul(g, scope, &a0, &b2));
        ops::sub(g, scope, &p, &q)
    };
    let c2 = {
        let (p, q) = (ops::mul(g, scope, &a0, &b1), ops::mul(g, scope, &a1, &b0));
        ops::sub(g, scope, &p, &q)
    };
    ops::concat(g, scope, &[&c0, &c1, &c2], r)
}

/// Conjugates of WXYZ quaternions: `[w, -x, -y, -z]` (upstream `quat_conjugate`).
pub fn quat_conjugate(g: &mut Graph, scope: &str, q: &str) -> String {
    let r = g.shape(q).len() as i64 - 1;
    let w = ops::slice_axis(g, scope, q, r, 0, 1);
    let xyz = ops::slice_axis(g, scope, q, r, 1, 4);
    let neg_xyz = ops::neg(g, scope, &xyz);
    ops::concat(g, scope, &[&w, &neg_xyz], r)
}

/// Rotate vectors by WXYZ quaternions (upstream `quat_apply`):
/// `v + 2*w*(xyz x v) + 2*(xyz x (xyz x v))`, computed as
/// `t = 2 * (xyz x v); v + w*t + (xyz x t)`.
pub fn quat_apply(g: &mut Graph, scope: &str, q: &str, v: &str) -> String {
    let r = g.shape(q).len() as i64 - 1;
    let w = ops::slice_axis(g, scope, q, r, 0, 1);
    let xyz = ops::slice_axis(g, scope, q, r, 1, 4);
    let cross = cross3(g, scope, &xyz, v);
    let two = c_scalar(g, scope, 2.0);
    let t = ops::mul(g, scope, &cross, &two);
    let wt = ops::mul(g, scope, &w, &t);
    let v_wt = ops::add(g, scope, v, &wt);
    let cross_t = cross3(g, scope, &xyz, &t);
    ops::add(g, scope, &v_wt, &cross_t)
}

/// WXYZ quaternion of a y-axis rotation by `angle` (any shape -> `shape + [4]`):
/// `[cos(a/2), 0, sin(a/2), 0]`, exactly the quaternion upstream's
/// `compute_root_pos_and_rot` builds (component 0 is w, component 2 is y).
pub fn quat_from_yaw(g: &mut Graph, scope: &str, angle: &str) -> String {
    let r = g.shape(angle).len() as i64;
    let two = c_scalar(g, scope, 2.0);
    let half = ops::div(g, scope, angle, &two);
    let cos = ops::cos(g, scope, &half);
    let sin = ops::sin(g, scope, &half);
    let zeros = zeros_like(g, scope, angle);
    let c = ops::unsqueeze(g, scope, &cos, &[r]);
    let s = ops::unsqueeze(g, scope, &sin, &[r]);
    let z = ops::unsqueeze(g, scope, &zeros, &[r]);
    ops::concat(g, scope, &[&c, &z, &s, &z], r)
}

/// Shortest-turn angular difference `a - b` in `(-pi, pi]`, per upstream `diff_angles`:
/// `atan2(sin(a)cos(b) - cos(a)sin(b), cos(a)cos(b) + sin(a)sin(b))`. No fps scaling —
/// callers multiply by `FPS` themselves (upstream folds it into the same expression).
pub fn diff_angles(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let ca = ops::cos(g, scope, a);
    let sa = ops::sin(g, scope, a);
    let cb = ops::cos(g, scope, b);
    let sb = ops::sin(g, scope, b);
    let cos_diff = {
        let (p, q) = (ops::mul(g, scope, &ca, &cb), ops::mul(g, scope, &sa, &sb));
        ops::add(g, scope, &p, &q)
    };
    let sin_diff = {
        let (p, q) = (ops::mul(g, scope, &sa, &cb), ops::mul(g, scope, &ca, &sb));
        ops::sub(g, scope, &p, &q)
    };
    ops::atan2(g, scope, &sin_diff, &cos_diff)
}

/// One model's feature layouts plus its normalization statistics.
///
/// The stats vectors loaded from the checkpoint use the dual layout; `load` selects the
/// global and local reps' slices (`indices["global_rep"]` / `indices["local_rep"]`) and
/// precomputes `scale = sqrt(std^2 + 1e-5)`, so normalize is a plain `(x - mean) / scale`.
pub struct Rep {
    pub layout: Layout,
    global_mean: Vec<f32>,
    global_scale: Vec<f32>,
    local_mean: Vec<f32>,
    local_scale: Vec<f32>,
    /// Stats of the pose features `[hip_height] ++ ric ++ rot6d` (the local root's y,
    /// then the body's first two sections), for [`Rep::normalize_local_pose`].
    pose_mean: Vec<f32>,
    pose_scale: Vec<f32>,
}

impl Rep {
    /// Build the rep tables from the raw dual-layout stats (`mean`/`std`, `dual_dim`
    /// floats each: G1 418, SOMA 370). No graph nodes are emitted here; the stats
    /// initializers are emitted lazily by `normalize`/`unnormalize` and the converters,
    /// sliced to exactly the width each call needs.
    pub fn load(_g: &mut Graph, spec: &ModelSpec, mean: &[f32], std: &[f32]) -> Result<Rep> {
        let layout = Layout::for_spec(spec);
        ensure!(
            mean.len() == layout.dual_dim as usize && std.len() == layout.dual_dim as usize,
            "stats must have {} entries, got {}/{}",
            layout.dual_dim,
            mean.len(),
            std.len()
        );
        let scale: Vec<f32> = std.iter().map(|&s| (s * s + 1e-5).sqrt()).collect();
        let (g0, g1) = layout.stats_global_root;
        let (l0, l1) = layout.stats_local_root;
        let (b0, b1) = layout.stats_body;
        // indices["global_rep"] = global_root ++ body; indices["local_rep"] = local_root ++ body.
        let select = |v: &[f32], root: (i64, i64)| -> Vec<f32> {
            v[root.0 as usize..root.1 as usize]
                .iter()
                .chain(v[b0 as usize..b1 as usize].iter())
                .copied()
                .collect()
        };
        let local_mean = select(mean, (l0, l1));
        let local_scale = select(&scale, (l0, l1));
        // The pose features are [hip_height] ++ ric ++ rot6d: the local root's y (index
        // local_root_dim - 1), then the body's ric and rot6d sections.
        let (lr, rot_end) = (layout.local_root_dim as usize, (layout.local_root_dim + layout.rot6d.1) as usize);
        let pose_index = |v: &[f32]| -> Vec<f32> {
            v[lr - 1..lr].iter().chain(v[lr..rot_end].iter()).copied().collect()
        };
        Ok(Rep {
            layout,
            global_mean: select(mean, (g0, g1)),
            global_scale: select(&scale, (g0, g1)),
            pose_mean: pose_index(&local_mean),
            pose_scale: pose_index(&local_scale),
            local_mean,
            local_scale,
        })
    }

    /// Emit the (mean, scale) initializers for features `[start, start+dim)` of `rep`.
    fn stats_inits(&self, g: &mut Graph, scope: &str, rep: RepKind, start: i64, dim: i64) -> (String, String) {
        let (mean, scale) = match rep {
            RepKind::Global => (&self.global_mean, &self.global_scale),
            RepKind::Local => (&self.local_mean, &self.local_scale),
        };
        let (s, e) = (start as usize, (start + dim) as usize);
        assert!(e <= mean.len(), "{scope}: stats slice {start}..{} out of range", start + dim);
        let base = format!("{scope}/stats_{}_{start}_{dim}", rep.name());
        let mean_name = format!("{base}_mean");
        let scale_name = format!("{base}_scale");
        if !g.dtypes.contains_key(&mean_name) {
            ops::c_f32(g, &mean_name, &[dim], &mean[s..e]);
            ops::c_f32(g, &scale_name, &[dim], &scale[s..e]);
        }
        (mean_name, scale_name)
    }

    /// `(x - mean) / scale` over a `[start, start+dim)` feature slice of `rep`.
    fn norm_slice(&self, g: &mut Graph, scope: &str, x: &str, rep: RepKind, start: i64, dim: i64) -> String {
        let (mean, scale) = self.stats_inits(g, scope, rep, start, dim);
        let shifted = ops::sub(g, scope, x, &mean);
        ops::div(g, scope, &shifted, &scale)
    }

    /// `x * scale + mean` over a `[start, start+dim)` feature slice of `rep`.
    fn unnorm_slice(&self, g: &mut Graph, scope: &str, x: &str, rep: RepKind, start: i64, dim: i64) -> String {
        let (mean, scale) = self.stats_inits(g, scope, rep, start, dim);
        let scaled = ops::mul(g, scope, x, &scale);
        ops::add(g, scope, &scaled, &mean)
    }

    /// `(x - mean) / scale` for the first `dim` features of the global or local rep
    /// (`x` is `[..., dim]`).
    pub fn normalize(&self, g: &mut Graph, scope: &str, x: &str, rep: RepKind, dim: i64) -> String {
        self.norm_slice(g, scope, x, rep, 0, dim)
    }

    /// `x * scale + mean` for the first `dim` features of the global or local rep.
    pub fn unnormalize(&self, g: &mut Graph, scope: &str, x: &str, rep: RepKind, dim: i64) -> String {
        self.unnorm_slice(g, scope, x, rep, 0, dim)
    }

    /// `(x - mean) / scale` for the pose features `[hip_height] ++ ric ++ rot6d`
    /// (`x` is `[..., 1 + ric + rot6d]`: 304 for G1, 268 for SOMA). This ordering is NOT a
    /// contiguous slice of the local rep stats, hence the dedicated method.
    pub fn normalize_local_pose(&self, g: &mut Graph, scope: &str, x: &str) -> String {
        let dim = self.pose_mean.len() as i64;
        let base = format!("{scope}/stats_pose_{dim}");
        let mean_name = format!("{base}_mean");
        let scale_name = format!("{base}_scale");
        if !g.dtypes.contains_key(&mean_name) {
            ops::c_f32(g, &mean_name, &[dim], &self.pose_mean);
            ops::c_f32(g, &scale_name, &[dim], &self.pose_scale);
        }
        let shifted = ops::sub(g, scope, x, &mean_name);
        ops::div(g, scope, &shifted, &scale_name)
    }

    /// Normalized global root `[B, T, 5]` -> normalized local root `[B, T, 4]`.
    ///
    /// `lengths` is int64 `[B]` (`pred_num_tokens * FRAMES_PER_TOKEN`). Follows upstream
    /// `global_root_to_local_onnx_compatible`: heading angle via `atan2(sin, cos)`,
    /// angular velocity by shortest-turn finite differences x FPS, linear velocity by
    /// finite differences x FPS, both zero-padded at the end; then the value at each
    /// request's last valid frame (`lengths - 1`, a runtime index) is replaced by the
    /// previous frame's value via GatherElements/ScatterElements on axis 1.
    /// `COMPUTE_KWARGS["removing_heading"]` is false for these models, so the velocities
    /// stay in the global frame (no `quat_apply` rotation).
    pub fn global_root_to_local(&self, g: &mut Graph, scope: &str, global_root: &str, lengths: &str) -> String {
        let shape = g.shape(global_root);
        assert_eq!(shape.len(), 3, "global_root must be [B, T, 5]");
        assert_eq!(shape[2], Dim::Static(5));
        let t = shape[1].get().expect("static T");
        let batch = shape[0].clone();
        assert_eq!(g.dtype(lengths), DType::I64, "lengths must be int64 [B]");
        let fps = c_scalar(g, scope, FPS as f32);

        let root = self.unnorm_slice(g, scope, global_root, RepKind::Global, 0, 5);
        let positions = ops::slice_axis(g, scope, &root, 2, 0, 3); // [B, T, 3]
        let two = c_scalar(g, scope, 2.0);
        let rot_cos = ops::slice_axis(g, scope, &root, 2, 3, 4);
        let rot_cos = ops::squeeze(g, scope, &rot_cos, &[2]);
        let rot_sin = ops::slice_axis(g, scope, &root, 2, 4, 5);
        let rot_sin = ops::squeeze(g, scope, &rot_sin, &[2]);
        // compute_root_pos_and_rot + the quaternion round-trip (atan2(q_y, q_w) * 2).
        let heading = ops::atan2(g, scope, &rot_sin, &rot_cos); // [B, T]
        let half = ops::div(g, scope, &heading, &two);
        let (hs, hc) = (ops::sin(g, scope, &half), ops::cos(g, scope, &half));
        let wrapped = ops::atan2(g, scope, &hs, &hc);
        let angles = ops::mul(g, scope, &wrapped, &two);
        // Angular velocity: shortest-turn differences, zero-padded at the end.
        let next = ops::slice_axis(g, scope, &angles, 1, 1, t);
        let prev = ops::slice_axis(g, scope, &angles, 1, 0, t - 1);
        let diffs = diff_angles(g, scope, &next, &prev);
        let angular = ops::mul(g, scope, &diffs, &fps); // [B, T-1]
        let first = ops::slice_axis(g, scope, &angular, 1, 0, 1);
        let pad = zeros_like(g, scope, &first);
        let angular = ops::concat(g, scope, &[&angular, &pad], 1); // [B, T]
        // Linear velocity: position differences, zero-padded at the end.
        let pos_next = ops::slice_axis(g, scope, &positions, 1, 1, t);
        let pos_prev = ops::slice_axis(g, scope, &positions, 1, 0, t - 1);
        let velocity = ops::sub(g, scope, &pos_next, &pos_prev);
        let velocity = ops::mul(g, scope, &velocity, &fps); // [B, T-1, 3]
        let first = ops::slice_axis(g, scope, &velocity, 1, 0, 1);
        let pad = zeros_like(g, scope, &first);
        let velocity = ops::concat(g, scope, &[&velocity, &pad], 1); // [B, T, 3]

        // Last valid frame of each request repeats the frame before it (runtime indices).
        let one = c_scalar_i64(g, scope, 1);
        let last = sub_i64(g, scope, lengths, &one); // [B] = lengths - 1
        let prev = sub_i64(g, scope, &last, &one); // [B] = lengths - 2
        let last2 = ops::reshape(g, scope, &last, &[batch.clone(), Dim::Static(1)]);
        let prev2 = ops::reshape(g, scope, &prev, &[batch.clone(), Dim::Static(1)]);
        // angular [B, T]: gather at lengths-2, scatter at lengths-1 (index shape [B, 1]).
        let gathered = ops::gather_elements(g, scope, &angular, &prev2, 1);
        let angular = ops::scatter_elements(g, scope, &angular, &last2, &gathered, 1);
        // velocity [B, T, 3]: same, with the [B, 1] indices broadcast to [B, 1, 3].
        let last3 = {
            let u = ops::unsqueeze(g, scope, &last2, &[2]);
            ops::concat(g, scope, &[&u, &u, &u], 2)
        };
        let prev3 = {
            let u = ops::unsqueeze(g, scope, &prev2, &[2]);
            ops::concat(g, scope, &[&u, &u, &u], 2)
        };
        let gathered = ops::gather_elements(g, scope, &velocity, &prev3, 1);
        let velocity = ops::scatter_elements(g, scope, &velocity, &last3, &gathered, 1);

        // local root = [angular, vel_x, vel_z, hip_height] (velocities stay global frame).
        let angular_col = ops::unsqueeze(g, scope, &angular, &[2]);
        let vel_x = ops::slice_axis(g, scope, &velocity, 2, 0, 1);
        let vel_z = ops::slice_axis(g, scope, &velocity, 2, 2, 3);
        let hip = ops::slice_axis(g, scope, &positions, 2, 1, 2);
        let local_root = ops::concat(g, scope, &[&angular_col, &vel_x, &vel_z, &hip], 2); // [B, T, 4]
        self.norm_slice(g, scope, &local_root, RepKind::Local, 0, 4)
    }

    /// Normalized local root `[B, T, 4]` + normalized local body `[B, T, body]` ->
    /// unnormalized global features `[B, T, 5 + body]`
    /// (`LocalRootLocalBody.compute_root_pos_and_rot` with `removing_heading` false).
    ///
    /// Yaw rate and horizontal velocities are divided by FPS, shifted by one frame
    /// (zero prepended) and integrated with CumSum; the velocities are NOT rotated into
    /// the heading frame (upstream's `removing_heading` is false). The global root is
    /// `[pos_x, hip_y, pos_z, cos(heading), sin(heading)]`; the body passes through
    /// unnormalized (its stats slice is identical in both reps).
    pub fn local_to_global(&self, g: &mut Graph, scope: &str, local_root: &str, body: &str) -> String {
        let shape = g.shape(local_root);
        assert_eq!(shape.len(), 3, "local_root must be [B, T, 4]");
        assert_eq!(shape[2], Dim::Static(4));
        let t = shape[1].get().expect("static T");
        assert_eq!(g.shape(body)[2], Dim::Static(self.layout.body_dim));
        let fps = c_scalar(g, scope, FPS as f32);
        let two = c_scalar(g, scope, 2.0);

        let root = self.unnorm_slice(g, scope, local_root, RepKind::Local, 0, 4);
        let body = self.unnorm_slice(g, scope, body, RepKind::Local, 4, self.layout.body_dim);
        let yaw_col = ops::slice_axis(g, scope, &root, 2, 0, 1);
        let yaw_col = ops::squeeze(g, scope, &yaw_col, &[2]);
        let yaw_rate = ops::div(g, scope, &yaw_col, &fps); // [B, T]
        let vel_xz = ops::slice_axis(g, scope, &root, 2, 1, 3);
        let planar = ops::div(g, scope, &vel_xz, &fps); // [B, T, 2]
        let hip = ops::slice_axis(g, scope, &root, 2, 3, 4); // [B, T, 1]

        // Integrate: prepend a zero frame, drop the last frame, CumSum over time.
        let first = ops::slice_axis(g, scope, &yaw_rate, 1, 0, 1);
        let zeros = zeros_like(g, scope, &first);
        let shifted = ops::slice_axis(g, scope, &yaw_rate, 1, 0, t - 1);
        let yaw_in = ops::concat(g, scope, &[&zeros, &shifted], 1);
        let yaw = ops::cumsum(g, scope, &yaw_in, 1); // [B, T]
        let first = ops::slice_axis(g, scope, &planar, 1, 0, 1);
        let zeros = zeros_like(g, scope, &first);
        let shifted = ops::slice_axis(g, scope, &planar, 1, 0, t - 1);
        let xz_in = ops::concat(g, scope, &[&zeros, &shifted], 1);
        let xz = ops::cumsum(g, scope, &xz_in, 1); // [B, T, 2]

        let half = ops::div(g, scope, &yaw, &two);
        let (hs, hc) = (ops::sin(g, scope, &half), ops::cos(g, scope, &half));
        let wrapped = ops::atan2(g, scope, &hs, &hc);
        let heading = ops::mul(g, scope, &wrapped, &two); // [B, T]
        let pos_x = ops::slice_axis(g, scope, &xz, 2, 0, 1);
        let pos_z = ops::slice_axis(g, scope, &xz, 2, 1, 2);
        let cos_h = ops::cos(g, scope, &heading);
        let cos_h = ops::unsqueeze(g, scope, &cos_h, &[2]);
        let sin_h = ops::sin(g, scope, &heading);
        let sin_h = ops::unsqueeze(g, scope, &sin_h, &[2]);
        let global_root = ops::concat(g, scope, &[&pos_x, &hip, &pos_z, &cos_h, &sin_h], 2); // [B, T, 5]
        ops::concat(g, scope, &[&global_root, &body], 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{G1, MAX_FRAMES, SOMA};

    fn layout_dims() {
        let g1 = Layout::for_spec(&G1);
        assert_eq!((g1.global_dim, g1.local_dim, g1.dual_dim, g1.body_dim), (414, 413, 418, 409));
        assert_eq!(g1.ric, (0, 99));
        assert_eq!(g1.rot6d, (99, 303));
        assert_eq!(g1.local_vel, (303, 405));
        assert_eq!(g1.foot_contacts, (405, 409));
        assert_eq!(g1.stats_global_root, (0, 5));
        assert_eq!(g1.stats_local_root, (5, 9));
        assert_eq!(g1.stats_body, (9, 418));
        let soma = Layout::for_spec(&SOMA);
        assert_eq!((soma.global_dim, soma.local_dim, soma.dual_dim), (366, 365, 370));
    }

    fn rep_fixture(spec: &ModelSpec) -> (Graph, Rep) {
        let mut g = Graph::new();
        let layout = Layout::for_spec(spec);
        let mean: Vec<f32> = (0..layout.dual_dim).map(|i| i as f32 * 0.01).collect();
        let std: Vec<f32> = (0..layout.dual_dim).map(|i| 1.0 + i as f32 * 0.001).collect();
        let rep = Rep::load(&mut g, spec, &mean, &std).unwrap();
        (g, rep)
    }

    #[test]
    fn emit_conversions() {
        layout_dims();
        let (mut g, rep) = rep_fixture(&G1);
        let t = Dim::Static(MAX_FRAMES);
        let root5 = g.input("global_root", DType::F32, &[Dim::Batch, t.clone(), Dim::Static(5)]);
        let lengths = g.input("lengths", DType::I64, &[Dim::Batch]);
        let local = rep.global_root_to_local(&mut g, "/core", &root5, &lengths);
        assert_eq!(g.shape(&local), &[Dim::Batch, t.clone(), Dim::Static(4)]);

        let body = g.input("body", DType::F32, &[Dim::Batch, t.clone(), Dim::Static(rep.layout.body_dim)]);
        let global = rep.local_to_global(&mut g, "/core", &local, &body);
        assert_eq!(g.shape(&global), &[Dim::Batch, t.clone(), Dim::Static(414)]);

        // quaternion helpers
        let yaw_in = g.input("yaw", DType::F32, &[Dim::Batch, t.clone()]);
        let q = quat_from_yaw(&mut g, "/core", &yaw_in);
        assert_eq!(g.shape(&q), &[Dim::Batch, t.clone(), Dim::Static(4)]);
        let v = g.input("v", DType::F32, &[Dim::Batch, t, Dim::Static(3)]);
        let rotated = quat_apply(&mut g, "/core", &q, &v);
        assert_eq!(g.shape(&rotated), g.shape(&v));
        let conj = quat_conjugate(&mut g, "/core", &q);
        assert_eq!(g.shape(&conj), g.shape(&q));
        let a = g.input("a", DType::F32, &[Dim::Batch]);
        let b = g.input("b", DType::F32, &[Dim::Batch]);
        let d = diff_angles(&mut g, "/core", &a, &b);
        assert_eq!(g.shape(&d), &[Dim::Batch]);

        // normalize/unnormalize over a feature prefix
        let prefix = ops::slice_axis(&mut g, "/core", &global, 2, 0, 10);
        let n = rep.normalize(&mut g, "/core", &prefix, RepKind::Global, 10);
        assert_eq!(g.shape(&n)[2], Dim::Static(10));
        let u = rep.unnormalize(&mut g, "/core", &local, RepKind::Local, 4);
        assert_eq!(g.shape(&u)[2], Dim::Static(4));
    }
}
