// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The rig kernels of `motionbricks_mlx/kernels.py` on mlx-rs: same Metal source
//! (`model/motionbricks_mlx/assets/rig_kernels.metal`), dispatched through `mlx_fast_metal_kernel`.

use mlx_rs::fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg};
use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use crate::ops::R;

const SOURCE: &str = include_str!("../../model/motionbricks_mlx/assets/rig_kernels.metal");

fn section(name: &str) -> &'static str {
    let marker = format!("// @@ {name}\n");
    let start = SOURCE.find(&marker).unwrap_or_else(|| panic!("kernel section {name}")) + marker.len();
    let end = SOURCE[start..].find("// @@ ").map_or(SOURCE.len(), |i| start + i);
    &SOURCE[start..end]
}

fn kernel(name: &str, inputs: &[&str], outputs: &[&str], body: &str) -> R<MetalKernel> {
    let options = MetalKernelOptions { header: section("HEADER").to_string(), ensure_row_contiguous: true, atomic_outputs: false };
    MetalKernel::new(name, inputs, outputs, body, options)
}

fn out(shape: &[i32]) -> MetalKernelOutput {
    MetalKernelOutput { shape: shape.to_vec(), dtype: Dtype::Float32 }
}

fn launch(n: i32, per_group: i32) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((n.max(1), 1, 1), (per_group.min(n.max(1)), 1, 1))
}

/// Row widths the kernels index with (rig_kernels.metal): a G1 MuJoCo pose, the G1 global
/// feature layout, and the G1 local-pose features the preprocess writes.
pub const G1_QPOS: i32 = crate::config::G1_QPOS_DIM;
pub const G1_FEATURES: i32 = 414;
pub const G1_LOCAL_POSES: i32 = 303;

/// The kernels read fixed row widths, so a mismatched shape would read out of bounds or leave
/// output rows unwritten: refuse it on the host (metadata only, no GPU work).
fn expect_shape(what: &str, a: &Array, want: &[i32]) -> R<()> {
    if a.shape() == want {
        Ok(())
    } else {
        Err(Exception::custom(format!("rig kernel: {what} has shape {:?}; expected {want:?}", a.shape())))
    }
}

pub struct RigKernels {
    g1_pre: MetalKernel,
    g1_post: MetalKernel,
    soma: MetalKernel,
}

impl RigKernels {
    pub fn new() -> R<Self> {
        Ok(RigKernels {
            g1_pre: kernel("mb_g1_preprocess", &["context", "target", "rf", "ri"], &["global_root", "local_root", "local_poses"], section("G1_PREPROCESS"))?,
            g1_post: kernel("mb_g1_postprocess", &["features", "rf", "ri"], &["qpos", "contacts"], section("G1_POSTPROCESS"))?,
            soma: kernel("mb_soma_retarget", &["six", "src", "dst"], &["out"], section("SOMA_RETARGET"))?,
        })
    }

    /// `ctx`, `tgt`: `(B, 4, 36)` float32 MuJoCo poses.
    pub fn g1_preprocess(&self, ctx: &Array, tgt: &Array, rf: &Array, ri: &Array) -> R<Vec<Array>> {
        let b = ctx.shape().first().copied().unwrap_or(0);
        expect_shape("context_qpos", ctx, &[b, 4, G1_QPOS])?;
        expect_shape("target_qpos", tgt, &[b, 4, G1_QPOS])?;
        let (grid, group) = launch(b, 32);
        let opts = MetalKernelApplyOptions { template_args: vec![("B".into(), TemplateArg::Int(b))], ..Default::default() };
        // SAFETY: inputs checked above; one thread per sequence writes exactly its own rows of
        // each output (shapes below).
        unsafe { self.g1_pre.apply(&[ctx, tgt, rf, ri], &[out(&[b, 8, 5]), out(&[b, 8, 4]), out(&[b, 8, G1_LOCAL_POSES])], grid, group, opts) }
    }

    /// `features`: `(B, T, 414)` float32 global features.
    pub fn g1_postprocess(&self, features: &Array, rf: &Array, ri: &Array) -> R<Vec<Array>> {
        let (b, t) = match features.shape() {
            &[b, t, _] => (b, t),
            _ => (0, 0),
        };
        expect_shape("features", features, &[b, t, G1_FEATURES])?;
        let (grid, group) = launch(b * t, 64);
        let opts = MetalKernelApplyOptions { template_args: vec![("ROWS".into(), TemplateArg::Int(b * t))], ..Default::default() };
        // SAFETY: input checked above; one thread per frame reads its feature row and writes its
        // qpos/contact rows.
        unsafe { self.g1_post.apply(&[features, rf, ri], &[out(&[b, t, G1_QPOS]), out(&[b, t, 4])], grid, group, opts) }
    }

    /// `six`: `(..., joints * 6)` float32; `src`/`dst`: `(joints, 3, 3)` offsets.
    pub fn soma_retarget(&self, six: &Array, src: &Array, dst: &Array, joints: i32) -> R<Array> {
        if six.ndim() == 0 {
            return Err(Exception::custom("rig kernel: 6D rotations need at least one axis"));
        }
        let mut want = six.shape().to_vec();
        if let Some(last) = want.last_mut() {
            *last = joints * 6;
        }
        expect_shape("6D rotations", six, &want)?;
        expect_shape("source offsets", src, &[joints, 3, 3])?;
        expect_shape("target offsets", dst, &[joints, 3, 3])?;
        let rows = (six.size() / (joints as usize * 6)) as i32;
        let (grid, group) = launch(rows * joints, 64);
        let opts = MetalKernelApplyOptions {
            template_args: vec![("ROWS".into(), TemplateArg::Int(rows)), ("J".into(), TemplateArg::Int(joints))],
            ..Default::default()
        };
        // SAFETY: shapes checked above; one thread per (row, joint) writes its own 6 values.
        Ok(unsafe { self.soma.apply(&[six, src, dst], &[out(six.shape())], grid, group, opts) }?.remove(0))
    }
}
