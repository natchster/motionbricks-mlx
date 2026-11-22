// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The SOMA capture <-> standard T-pose retarget, emitted as ONNX (upstream
//! `models/soma/model.py` `BatchedSOMA` pre/post and `models/soma/rotations.py`; the
//! compute reference is `model-rs/src/rig.rs` `SomaRetarget`).
//!
//! The rotation slice of the feature vectors (30 joints x 6D) is converted to rotation
//! matrices, moved between T poses by `R' = R @ offset_from @ offset_to^T`, converted
//! back to 6D, and spliced back into the feature vector with a Concat. The offsets come
//! from the rig asset (`capture_offsets` / `standard_offsets`, `[30, 3, 3]`); the
//! transposes are folded into the initializers at load time.
//!
//! Feature layouts (`models/representation.py`, 30 joints):
//! - `local_poses` input `[B, 8, 267]` (body only): ric `(J-1)*3 = 87`, then 6D
//!   rotations `J*6 = 180` at `[87, 267)`.
//! - `pred_global_motions` output `[B, 64, 366]`: root 5 (pos 3 + heading 2), then the
//!   body — ric 87 at `[5, 92)`, rotations 180 at `[92, 272)`, velocities 90, contacts 4.

use anyhow::{Context, Result, ensure};

use crate::config::ModelSpec;
use crate::graph::{DType, Dim, Graph};
use crate::npy::load_npz_bytes;
use crate::ops;

/// Width of the global root features (position xyz + heading cos/sin) preceding the body.
const GLOBAL_ROOT_DIM: i64 = 5;

/// The T-pose retarget constants, emitted as initializers by [`SomaRetarget::load`].
pub struct SomaRetarget {
    /// `capture_offsets` initializer, `[J, 3, 3]`.
    capture: String,
    /// `capture_offsets^T` initializer (transpose folded at load), `[J, 3, 3]`.
    capture_t: String,
    /// `standard_offsets` initializer, `[J, 3, 3]`.
    standard: String,
    /// `standard_offsets^T` initializer (transpose folded at load), `[J, 3, 3]`.
    standard_t: String,
    num_joints: i64,
}

/// `v / sqrt(sum(v^2))` along the last axis (upstream `cont6d_to_matrix`'s norm).
fn normalize3(g: &mut Graph, scope: &str, v: &str) -> String {
    let axis = g.shape(v).len() as i64 - 1;
    let sq = ops::mul(g, scope, v, v);
    let sum = ops::reduce_sum(g, scope, &sq, &[axis], true);
    let norm = ops::sqrt(g, scope, &sum);
    ops::div(g, scope, v, &norm)
}

/// Cross product of `[..., 3]` vectors along the last axis, via slices and a Concat.
fn cross3(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let axis = g.shape(a).len() as i64 - 1;
    let (ax, ay, az) = (
        ops::slice_axis(g, scope, a, axis, 0, 1),
        ops::slice_axis(g, scope, a, axis, 1, 2),
        ops::slice_axis(g, scope, a, axis, 2, 3),
    );
    let (bx, by, bz) = (
        ops::slice_axis(g, scope, b, axis, 0, 1),
        ops::slice_axis(g, scope, b, axis, 1, 2),
        ops::slice_axis(g, scope, b, axis, 2, 3),
    );
    let (ay_bz, az_by) = (ops::mul(g, scope, &ay, &bz), ops::mul(g, scope, &az, &by));
    let (az_bx, ax_bz) = (ops::mul(g, scope, &az, &bx), ops::mul(g, scope, &ax, &bz));
    let (ax_by, ay_bx) = (ops::mul(g, scope, &ax, &by), ops::mul(g, scope, &ay, &bx));
    let cx = ops::sub(g, scope, &ay_bz, &az_by);
    let cy = ops::sub(g, scope, &az_bx, &ax_bz);
    let cz = ops::sub(g, scope, &ax_by, &ay_bx);
    ops::concat(g, scope, &[&cx, &cy, &cz], axis)
}

impl SomaRetarget {
    /// Load the rig offsets from `spec.rig_asset()` and emit them (plus their transposes)
    /// as float32 initializers.
    pub fn load(g: &mut Graph, spec: &ModelSpec) -> Result<SomaRetarget> {
        let arrays = load_npz_bytes(spec.rig_asset())?;
        let j = spec.num_joints;
        let get = |k: &str| arrays.get(k).with_context(|| format!("rig asset lacks {k}"));
        let offsets = |key: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            let array = get(key)?;
            ensure!(array.shape == [j as i32, 3, 3], "{key} must be [{j}, 3, 3], got {:?}", array.shape);
            let data = array.to_f32();
            // Fold the transposes upstream emits at run time into the initializer.
            let mut transposed = vec![0f32; data.len()];
            for joint in 0..j as usize {
                for r in 0..3 {
                    for c in 0..3 {
                        transposed[joint * 9 + r * 3 + c] = data[joint * 9 + c * 3 + r];
                    }
                }
            }
            Ok((data, transposed))
        };
        let (capture, capture_t) = offsets("capture_offsets")?;
        let (standard, standard_t) = offsets("standard_offsets")?;
        let dims = &[j, 3, 3];
        Ok(SomaRetarget {
            capture: ops::c_f32(g, "capture_offsets", dims, &capture),
            capture_t: ops::c_f32(g, "capture_offsets_transposed", dims, &capture_t),
            standard: ops::c_f32(g, "standard_offsets", dims, &standard),
            standard_t: ops::c_f32(g, "standard_offsets_transposed", dims, &standard_t),
            num_joints: j,
        })
    }

    /// Model input: capture-convention local_poses `[B, 8, 267]` -> standard-convention
    /// `[B, 8, 267]` (`R @ capture @ standard^T` on the `[87, 267)` rotation slice).
    pub fn emit_to_standard(&self, g: &mut Graph, scope: &str, local_poses: &str) -> String {
        let ric = (self.num_joints - 1) * 3;
        self.emit_retarget(g, scope, local_poses, ric, ric + self.num_joints * 6, &self.capture, &self.standard_t)
    }

    /// Model output: standard-convention pred_global_motions `[B, 64, 366]` ->
    /// capture-convention (`R @ standard @ capture^T` on the `[92, 272)` rotation slice).
    pub fn emit_to_capture(&self, g: &mut Graph, scope: &str, global_motions: &str) -> String {
        let start = GLOBAL_ROOT_DIM + (self.num_joints - 1) * 3;
        self.emit_retarget(g, scope, global_motions, start, start + self.num_joints * 6, &self.standard, &self.capture_t)
    }

    /// `features[..., rot_start:rot_end]` 6D -> matrices -> `R @ from @ to_t` -> 6D,
    /// spliced back with a Concat of the untouched before/after slices.
    fn emit_retarget(
        &self,
        g: &mut Graph,
        scope: &str,
        features: &str,
        rot_start: i64,
        rot_end: i64,
        from: &str,
        to_t: &str,
    ) -> String {
        let fshape = g.shape(features).to_vec();
        assert_eq!(g.dtype(features), DType::F32, "retarget features must be f32");
        let feature_axis = fshape.len() as i64 - 1;
        let total = fshape.last().and_then(Dim::get).expect("static feature width");
        assert!(rot_end <= total, "rotation slice [{rot_start}, {rot_end}) past width {total}");
        let prefix = &fshape[..fshape.len() - 1]; // [B, T]
        let j = self.num_joints;
        let mut joint_shape = prefix.to_vec();
        joint_shape.extend([Dim::Static(j), Dim::Static(6)]);
        let mut flat_shape = prefix.to_vec();
        flat_shape.push(Dim::Static(j * 6));

        // 6D -> matrices: x = norm(six[..., 0:3]); z = norm(x cross six[..., 3:6]);
        // y = z cross x; the matrix's columns are (x, y, z).
        let rot = ops::slice_axis(g, scope, features, feature_axis, rot_start, rot_end);
        let six = ops::reshape(g, scope, &rot, &joint_shape);
        let vec_axis = joint_shape.len() as i64 - 1;
        let x_raw = ops::slice_axis(g, scope, &six, vec_axis, 0, 3);
        let y_raw = ops::slice_axis(g, scope, &six, vec_axis, 3, 6);
        let x = normalize3(g, scope, &x_raw);
        let z_raw = cross3(g, scope, &x, &y_raw);
        let z = normalize3(g, scope, &z_raw);
        let y = cross3(g, scope, &z, &x);
        let mat_axis = vec_axis + 1;
        let cols = [
            ops::unsqueeze(g, scope, &x, &[mat_axis]),
            ops::unsqueeze(g, scope, &y, &[mat_axis]),
            ops::unsqueeze(g, scope, &z, &[mat_axis]),
        ];
        let mats = ops::concat(g, scope, &[&cols[0], &cols[1], &cols[2]], mat_axis);

        // T-pose change: R' = R @ offset_from @ offset_to^T (batched, [B, T, J, 3, 3]).
        let from_applied = ops::matmul(g, scope, &mats, from);
        let moved = ops::matmul(g, scope, &from_applied, to_t);

        // Matrices -> 6D (upstream `matrix_to_cont6d`: the first two columns, plain
        // concatenated [x0, x1, x2, y0, y1, y2] — NOT row-major interleaved).
        let c0 = ops::slice_axis(g, scope, &moved, mat_axis, 0, 1);
        let c1 = ops::slice_axis(g, scope, &moved, mat_axis, 1, 2);
        let c0 = ops::squeeze(g, scope, &c0, &[mat_axis]);
        let c1 = ops::squeeze(g, scope, &c1, &[mat_axis]);
        let six2 = ops::concat(g, scope, &[&c0, &c1], vec_axis);
        let flat = ops::reshape(g, scope, &six2, &flat_shape);

        // Splice back: before / retargeted rotations / after (empty slices are skipped).
        let mut parts: Vec<String> = Vec::new();
        if rot_start > 0 {
            parts.push(ops::slice_axis(g, scope, features, feature_axis, 0, rot_start));
        }
        parts.push(flat);
        if rot_end < total {
            parts.push(ops::slice_axis(g, scope, features, feature_axis, rot_end, total));
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        ops::concat(g, scope, &refs, feature_axis)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    #[test]
    fn retarget_round_trip_shapes() {
        let mut g = Graph::new();
        let retarget = SomaRetarget::load(&mut g, &config::SOMA).unwrap();
        let local = g.input("local_poses", DType::F32, &[Dim::Batch, 8.into(), 267.into()]);
        let standard = retarget.emit_to_standard(&mut g, "/pre", &local);
        assert_eq!(g.shape(&standard), &[Dim::Batch, Dim::Static(8), Dim::Static(267)]);
        let global = g.input("pred_global_motions", DType::F32, &[Dim::Batch, 64.into(), 366.into()]);
        let capture = retarget.emit_to_capture(&mut g, "/post", &global);
        assert_eq!(g.shape(&capture), &[Dim::Batch, Dim::Static(64), Dim::Static(366)]);
        assert!(g.nodes.iter().all(|n| n.name.starts_with("/pre/") || n.name.starts_with("/post/")));
    }
}
