// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! One generation, emitted as ONNX: root path + duration -> pose tokens -> decoded motion
//! (mirrors `model-rs/src/inference.rs` `predict_with` in the release configuration:
//! one MaskGIT step with argmax sampling, so the step-0 remasking folds away — the tokens
//! provably stay all-masked, and the traced Sort/TopK machinery is constant).

use anyhow::Result;

use crate::config::{CONTEXT_AND_TARGET, FRAMES_PER_TOKEN, Kind, MAX_FRAMES, MAX_TOKENS, ModelSpec};
use crate::graph::{DType, Dim, Graph};
use crate::networks::{PoseNetwork, RootNetwork, VqDecoder, backbone_scope, decoder_scope};
use crate::onnx;
use crate::ops::*;
use crate::representation::{Rep, RepKind};
use crate::weights::Weights;

/// SOMA's root-from-height-constraint-violation threshold (metres).
const HEIGHT_ERROR_THRESHOLD: f32 = 0.05;

pub struct Core {
    pub rep: Rep,
    pub root: RootNetwork,
    pub pose: PoseNetwork,
    pub vqvae: VqDecoder,
    spec: &'static ModelSpec,
}

/// The emitted predictions: unnormalized global features `[B, 64, global_dim]` (recentring
/// undone) and `pred_num_tokens` int32 `[B]`.
pub struct Prediction {
    pub global_features: String,
    pub num_tokens: String,
}

// ---- small local helpers (int arithmetic and runtime-batch constants) ----

fn bin_i64(g: &mut Graph, scope: &str, op: &str, a: &str, b: &str) -> String {
    let out = crate::graph::broadcast(g.shape(a), g.shape(b)).expect("int op shapes");
    g.node(scope, op, &[a, b], &[], DType::I64, &out)
}

pub fn add_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    bin_i64(g, scope, "Add", a, b)
}

pub fn sub_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    bin_i64(g, scope, "Sub", a, b)
}

pub fn mul_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    bin_i64(g, scope, "Mul", a, b)
}

/// Zeros with the shape of `x` (f32), via `x * 0` (upstream's `zeros_like`).
fn zeros_like(g: &mut Graph, scope: &str, x: &str) -> String {
    let zero = c_scalar(g, scope, 0.0);
    mul(g, scope, x, &zero)
}

/// A runtime `[B] ++ tail` int64 shape vector (`Shape` -> `Gather(0)` -> `Concat`).
fn batch_shape(g: &mut Graph, scope: &str, like: &str, tail: &[i64]) -> String {
    let s = shape_of(g, scope, like);
    let b = gather_static(g, scope, &s, &[0], 0);
    let tail_t = c_vec(g, scope, "tail", tail);
    let dims = [Dim::Static(1 + tail.len() as i64)];
    g.node(scope, "Concat", &[&b, &tail_t], &[("axis", crate::graph::Attr::Int(0))], DType::I64, &dims)
}

/// An f32 constant tensor with a runtime batch dimension (ConstantOfShape).
fn const_of_shape(g: &mut Graph, scope: &str, like: &str, tail: &[i64], value: f32) -> String {
    let shape = batch_shape(g, scope, like, tail);
    let dims: Vec<Dim> = std::iter::once(Dim::Batch).chain(tail.iter().map(|&t| Dim::Static(t))).collect();
    constant_of_shape(g, scope, &shape, value, &dims)
}

/// A bool constant tensor with a runtime batch dimension.
fn const_bool(g: &mut Graph, scope: &str, like: &str, tail: &[i64], value: bool) -> String {
    let shape = batch_shape(g, scope, like, tail);
    let dims: Vec<Dim> = std::iter::once(Dim::Batch).chain(tail.iter().map(|&t| Dim::Static(t))).collect();
    let t = onnx::tensor_bool(&g.fresh(scope), &[1], &[value]);
    g.node(scope, "ConstantOfShape", &[&shape], &[("value", crate::graph::Attr::Tensor(t))], DType::Bool, &dims)
}

impl Core {
    pub fn load(g: &mut Graph, w: &mut Weights, spec: &'static ModelSpec, rep: Rep) -> Result<Core> {
        Ok(Core {
            root: RootNetwork::load(g, w, spec)?,
            pose: PoseNetwork::load(g, w, spec)?,
            vqvae: VqDecoder::load(g, w, spec)?,
            rep,
            spec,
        })
    }

    /// Context poses at frames 0..3, target poses at the last token's 4 frames
    /// (model-rs `dense_pose_conditions`). `p_n`: normalized poses `[B, 8, P]`;
    /// `has_poses` bool `[B, 8]`; `num_tokens` int64 `[B]`.
    fn dense_pose_conditions(&self, g: &mut Graph, scope: &str, p_n: &str, has_poses: &str, num_tokens: &str) -> (String, String) {
        let f = FRAMES_PER_TOKEN;
        let p = g.shape(p_n)[2].get().expect("static pose dim");
        let head = slice_axis(g, scope, p_n, 1, 0, f);
        let zeros = const_of_shape(g, scope, p_n, &[MAX_FRAMES - f, p], 0.0);
        let mut cond = concat(g, scope, &[&head, &zeros], 1);
        let has_head = slice_axis(g, scope, has_poses, 1, 0, f);
        let falses = const_bool(g, scope, has_poses, &[MAX_FRAMES - f], false);
        let mut has = concat(g, scope, &[&has_head, &falses], 1);
        let range = arange(g, scope, MAX_FRAMES);
        let frames = unsqueeze(g, scope, &range, &[0]); // [1, 64]
        let four = c_scalar_i64(g, scope, 4);
        let times_four = mul_i64(g, scope, num_tokens, &four);
        let base = sub_i64(g, scope, &times_four, &four);
        for k in 0..f {
            let k_t = c_scalar_i64(g, scope, k);
            let target = add_i64(g, scope, &base, &k_t); // [B] = num_tokens*4 - 4 + k
            let target_col = unsqueeze(g, scope, &target, &[1]);
            let at = equal(g, scope, &frames, &target_col); // [B, 64] bool
            let frame = slice_axis(g, scope, p_n, 1, f + k, f + k + 1); // [B, 1, P]
            let at_col = unsqueeze(g, scope, &at, &[2]);
            cond = where_(g, scope, &at_col, &frame, &cond);
            let has_k = slice_axis(g, scope, has_poses, 1, f + k, f + k + 1); // [B, 1]
            let hit = and_(g, scope, &at, &has_k);
            has = or_(g, scope, &has, &hit);
        }
        (cond, has)
    }

    /// The whole prediction (model-rs `Core::predict_with` with `num_inference_steps = 1`
    /// and no forced durations). Inputs are the joint-space features (already rig-converted
    /// and, for SOMA, retargeted): `global_root` `[B, 8, 5]`, `local_root` `[B, 8, 4]`,
    /// `local_poses` `[B, 8, P]` f32; the `has_*` masks bool `[B, 8]`; `allowed` `[B, 11]`.
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        g: &mut Graph,
        global_root: &str,
        has_global: &str,
        local_root: &str,
        has_local: &str,
        local_poses: &str,
        has_poses: &str,
        allowed: &str,
    ) -> Result<Prediction> {
        let scope = "/core";
        let spec = self.spec;
        let f = FRAMES_PER_TOKEN;
        let global_dim = self.rep.layout.global_dim;

        // Recentre on the first context frame's ground position; remember its heading.
        let first = slice_axis(g, scope, global_root, 1, 0, 1); // [B, 1, 5]
        let offset_x = slice_axis(g, scope, &first, 2, 0, 1);
        let offset_z = slice_axis(g, scope, &first, 2, 2, 3);
        let cos0 = slice_axis(g, scope, &first, 2, 3, 4);
        let sin0 = slice_axis(g, scope, &first, 2, 4, 5);
        let heading0 = atan2(g, scope, &sin0, &cos0); // [B, 1, 1]
        let zero = zeros_like(g, scope, &offset_x);
        let shift = concat(g, scope, &[&offset_x, &zero, &offset_z, &zero, &zero], 2);
        let recentred = sub(g, scope, global_root, &shift);

        // Normalize the three feature groups (the pose vector gets the hip height from the
        // global root's y, upstream `joint_positions_and_rotations_and_hip_height`).
        let g_n = self.rep.normalize(g, scope, &recentred, RepKind::Global, 5);
        let l_n = self.rep.normalize(g, scope, local_root, RepKind::Local, 4);
        let hip = slice_axis(g, scope, &recentred, 2, 1, 2);
        let pose_raw = concat(g, scope, &[&hip, local_poses], 2);
        let p_n = self.rep.normalize_local_pose(g, scope, &pose_raw);

        // 1) Duration + root path.
        let out = self.root.emit(g, backbone_scope(spec), &p_n, has_poses, &l_n, has_local, &g_n, has_global, allowed)?;
        let (num_tokens, num_tokens_i64) = (out.num_tokens, out.num_tokens_i64);
        let pred_root = out.global_root_values; // normalized [B, 64, 5]
        let four_i = c_scalar_i64(g, scope, 4);
        let lengths = mul_i64(g, scope, &num_tokens_i64, &four_i);
        let pred_local_root = self.rep.global_root_to_local(g, scope, &pred_root, &lengths);
        // The last valid frame mixes the input's last local root where `has_local[-1]`.
        let pred_local_root = {
            let one = c_scalar_i64(g, scope, 1);
            let last = sub_i64(g, scope, &lengths, &one); // [B]
            let last_col = reshape(g, scope, &last, &[Dim::Batch, Dim::Static(1)]);
            let last_u = unsqueeze(g, scope, &last_col, &[2]);
            let last4 = concat(g, scope, &[&last_u, &last_u, &last_u, &last_u], 2); // [B, 1, 4]
            let estimated = gather_elements(g, scope, &pred_local_root, &last4, 1); // [B, 1, 4]
            let h_row = slice_axis(g, scope, has_local, 1, 7, 8);
            let h_f = cast(g, scope, &h_row, DType::F32); // [B, 1]
            let h = unsqueeze(g, scope, &h_f, &[2]); // [B, 1, 1]
            let l_last = slice_axis(g, scope, &l_n, 1, 7, 8); // [B, 1, 4]
            let one_f = c_scalar(g, scope, 1.0);
            let inv_h = sub(g, scope, &one_f, &h);
            let keep = mul(g, scope, &h, &l_last);
            let fill = mul(g, scope, &inv_h, &estimated);
            let final_frame = add(g, scope, &keep, &fill);
            let range = arange(g, scope, MAX_FRAMES);
            let frames = unsqueeze(g, scope, &range, &[0]); // [1, 64]
            let last_b = unsqueeze(g, scope, &last, &[1]);
            let at_last = equal(g, scope, &frames, &last_b); // [B, 64]
            let at_last_col = unsqueeze(g, scope, &at_last, &[2]);
            where_(g, scope, &at_last_col, &final_frame, &pred_local_root)
        };

        // 2) Pose tokens (one MaskGIT step, argmax sampling): the all-masked start is a
        // constant, the step-0 remasking provably keeps it all-masked, so both fold away.
        let (pose_cond, has_pose_cond) = self.dense_pose_conditions(g, scope, &p_n, has_poses, &num_tokens_i64);
        let root_cond = gather_static(g, scope, &pred_root, &[0, 2, 3, 4], 2); // no hip height
        let tokens = {
            let shape = batch_shape(g, scope, &pose_cond, &[MAX_TOKENS, self.spec.vq_heads]);
            let t = onnx::tensor_i64(&g.fresh(scope), &[1], &[self.pose.mask_id]);
            g.node(
                scope,
                "ConstantOfShape",
                &[&shape],
                &[("value", crate::graph::Attr::Tensor(t))],
                DType::I64,
                &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(self.spec.vq_heads)],
            )
        };
        let logits = self.pose.emit(g, backbone_scope(spec), &tokens, &root_cond, &pose_cond, &has_pose_cond, &num_tokens_i64)?;
        let tokens = argmax(g, scope, &logits, -1, false); // int64 [B, 16, H]

        // 3) Decode tokens to local motion, integrate to global.
        let positions = arange(g, scope, MAX_TOKENS);
        let positions = unsqueeze(g, scope, &positions, &[0]); // [1, 16]
        let num_col = unsqueeze(g, scope, &num_tokens_i64, &[1]);
        let valid = less(g, scope, &positions, &num_col);
        let token_mask = cast(g, scope, &valid, DType::F32); // [B, 16]
        let external = slice_axis(g, scope, &pred_local_root, 2, 1, 3); // local root velocity
        let recon = self.vqvae.emit(g, decoder_scope(spec), &tokens, &token_mask, &pose_cond, &has_pose_cond, &external)?;
        let recon_root = slice_axis(g, scope, &recon, 2, 0, 4);
        let recon_body = slice_axis(g, scope, &recon, 2, 4, self.rep.layout.local_dim);
        let mut pred_global = self.rep.local_to_global(g, scope, &recon_root, &recon_body);

        // SOMA: take the root module's root where it beats the pose module's by height error.
        if spec.kind == Kind::Soma {
            let scope = "/core/final_root";
            let root_from_root = self.rep.unnormalize(g, scope, &pred_root, RepKind::Global, 5);
            let one = c_scalar_i64(g, scope, 1);
            let n_minus_1 = sub_i64(g, scope, &num_tokens_i64, &one);
            let base = mul_i64(g, scope, &n_minus_1, &four_i);
            let base = unsqueeze(g, scope, &base, &[1]); // [B, 1]
            let four_range = arange(g, scope, f);
            let four_range = unsqueeze(g, scope, &four_range, &[0]); // [1, 4]
            let tf = add_i64(g, scope, &base, &four_range); // [B, 4]
            let tf_col = unsqueeze(g, scope, &tf, &[2]);
            let rr_y = slice_axis(g, scope, &root_from_root, 2, 1, 2);
            let h_root = gather_elements(g, scope, &rr_y, &tf_col, 1);
            let pg_y = slice_axis(g, scope, &pred_global, 2, 1, 2);
            let h_pose = gather_elements(g, scope, &pg_y, &tf_col, 1);
            let g_un = self.rep.unnormalize(g, scope, &g_n, RepKind::Global, 5);
            // The last 4 of the 8 input slots are the target frames.
            let g_tail = slice_axis(g, scope, &g_un, 1, CONTEXT_AND_TARGET - f, CONTEXT_AND_TARGET);
            let h_target = slice_axis(g, scope, &g_tail, 2, 1, 2); // [B, 4, 1]
            let h_target = squeeze(g, scope, &h_target, &[2]);
            let w_row = slice_axis(g, scope, has_global, 1, CONTEXT_AND_TARGET - f, CONTEXT_AND_TARGET);
            let w = cast(g, scope, &w_row, DType::F32); // [B, 4]
            let w_sum = reduce_sum(g, scope, &w, &[1], false);
            let h_pose = squeeze(g, scope, &h_pose, &[2]);
            let h_root = squeeze(g, scope, &h_root, &[2]);
            let d_pose = {
                let d = sub(g, scope, &h_pose, &h_target);
                abs_(g, scope, &d)
            };
            let d_root = {
                let d = sub(g, scope, &h_root, &h_target);
                abs_(g, scope, &d)
            };
            let err_pose = {
                let m = mul(g, scope, &d_pose, &w);
                let num = reduce_sum(g, scope, &m, &[1], false);
                div(g, scope, &num, &w_sum)
            };
            let err_root = {
                let m = mul(g, scope, &d_root, &w);
                let num = reduce_sum(g, scope, &m, &[1], false);
                div(g, scope, &num, &w_sum)
            };
            let thr = c_scalar(g, scope, HEIGHT_ERROR_THRESHOLD);
            let root_better = less(g, scope, &err_root, &err_pose);
            let pose_bad = greater(g, scope, &err_pose, &thr);
            let use_root = and_(g, scope, &root_better, &pose_bad);
            let use_root_r = reshape(g, scope, &use_root, &[Dim::Batch, Dim::Static(1), Dim::Static(1)]);
            let use_root_f = cast(g, scope, &use_root_r, DType::F32);
            let one_f = c_scalar(g, scope, 1.0);
            let inv = sub(g, scope, &one_f, &use_root_f);
            let pose_root = slice_axis(g, scope, &pred_global, 2, 0, 5);
            let a = mul(g, scope, &use_root_f, &root_from_root);
            let b = mul(g, scope, &inv, &pose_root);
            let picked = add(g, scope, &a, &b);
            let body = slice_axis(g, scope, &pred_global, 2, 5, global_dim);
            pred_global = concat(g, scope, &[&picked, &body], 2);
        }

        // Undo the recentring: restore the ground offset and rotate headings back.
        let pred_global = {
            let scope = "/post/recentre";
            let cos_p = slice_axis(g, scope, &pred_global, 2, 3, 4);
            let sin_p = slice_axis(g, scope, &pred_global, 2, 4, 5);
            let a2 = atan2(g, scope, &sin_p, &cos_p);
            let angle = add(g, scope, &a2, &heading0); // [B, 64, 1]
            let px0 = slice_axis(g, scope, &pred_global, 2, 0, 1);
            let px = add(g, scope, &px0, &offset_x);
            let py = slice_axis(g, scope, &pred_global, 2, 1, 2);
            let pz0 = slice_axis(g, scope, &pred_global, 2, 2, 3);
            let pz = add(g, scope, &pz0, &offset_z);
            let cos_a = cos(g, scope, &angle);
            let sin_a = sin(g, scope, &angle);
            let root = concat(g, scope, &[&px, &py, &pz, &cos_a, &sin_a], 2);
            let body = slice_axis(g, scope, &pred_global, 2, 5, global_dim);
            concat(g, scope, &[&root, &body], 2)
        };

        Ok(Prediction { global_features: pred_global, num_tokens })
    }
}
