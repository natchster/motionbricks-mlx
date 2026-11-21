// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! One generation: root path + duration -> pose tokens -> decoded motion
//! (ported from `motionbricks_mlx/inference.py`). Static shapes (16 tokens / 64 frames).

use mlx_rs::{Array, Dtype, ops};

use crate::config::*;
use crate::networks::*;
use crate::ops::*;
use crate::representation::MotionRepresentation;
use crate::weights::Weights;

const HEIGHT_ERROR_THRESHOLD: f32 = 0.05; // metres

pub struct Core {
    pub spec: ModelSpec,
    pub rep: MotionRepresentation,
    pub root: RootNetwork,
    pub pose: PoseNetwork,
    pub vqvae: VqDecoder,
    pub dtype: Dtype,
}

impl Core {
    pub fn load(w: &mut Weights, spec: ModelSpec, rep: MotionRepresentation) -> R<Self> {
        Ok(Core {
            root: RootNetwork::load(w, &spec.root)?,
            pose: PoseNetwork::load(w, &spec.pose, &spec.vq)?,
            vqvae: VqDecoder::load(w, &spec.vq)?,
            dtype: w.dtype,
            spec,
            rep,
        })
    }

    /// MaskGIT re-masking: mask the `cos(step / steps * pi / 2)` fraction of valid tokens with
    /// the lowest confidence.
    fn remask(&self, tokens: &Array, prob: &Array, num_tokens: &Array, step: usize, steps: usize) -> R<Array> {
        let (batch, positions, heads) = (tokens.dim(0), tokens.dim(1), tokens.dim(2));
        let ratio = (step as f64 / steps as f64 * std::f64::consts::PI * 0.5).cos() as f32;
        let count = ops::maximum((&(f(ratio) * num_tokens) * f(heads as f32)).as_dtype(Dtype::Int32)?, i(1))?;
        let valid = arange(positions)?.reshape(&[1, positions, 1])?.lt(num_tokens.reshape(&[batch, 1, 1])?)?;
        let prob = wh(&valid, prob, &f(f32::INFINITY))?.reshape(&[batch, -1])?;
        let order = ops::argsort_axis(&prob, -1)?;
        let selected = unsq(&arange(positions * heads)?, 0)?.lt(unsq(&count, 1)?)?;
        let order = wh(&selected, &order, &slice(&order, 1, 0, 1)?)?;
        let hit = unsq(&order, -1)?.eq(arange(positions * heads)?.reshape(&[1, 1, positions * heads])?)?.any_axis(1, None)?;
        let flat = wh(&hit, &i(self.pose.mask_id), &tokens.reshape(&[batch, -1])?)?;
        flat.reshape(&[batch, positions, heads])
    }

    /// Context poses at frames 0..3, target poses at the last token's 4 frames.
    fn dense_pose_conditions(&self, poses: &Array, has_poses: &Array, num_tokens: &Array) -> R<(Array, Array)> {
        let f = FRAMES_PER_TOKEN;
        let (batch, dim) = (poses.dim(0), poses.dim(2));
        let mut cond = cat(&[&slice(poses, 1, 0, f)?, &ops::zeros_dtype(&[batch, MAX_FRAMES - f, dim], poses.dtype())?], 1)?;
        let mut has = cat(&[&slice(has_poses, 1, 0, f)?, &ops::zeros_dtype(&[batch, MAX_FRAMES - f], Dtype::Bool)?], 1)?;
        let frames = unsq(&arange(MAX_FRAMES)?, 0)?;
        for k in 0..f {
            let at = frames.eq(unsq(&(&(num_tokens * f) - f + k), 1)?)?;
            cond = wh(&unsq(&at, -1)?, &unsq(&index(poses, 1, f + k)?, 1)?, &cond)?;
            has = has.logical_or(at.logical_and(unsq(&index(has_poses, 1, f + k)?, 1)?)?)?;
        }
        Ok((cond, has))
    }

    /// Joint-space inputs (before the rig conversion): `global_root (B,8,5)`, `local_root (B,8,4)`,
    /// `local_poses (B,8,ric+rot)` float32; masks bool `(B,8)`; `allowed` bool `(B,11)`.
    /// Returns `(pred_global (B,64,global_dim) float32, pred_num_tokens (B,) int32)`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self, global_root: &Array, has_global: &Array, local_root: &Array, has_local: &Array, local_poses: &Array,
        has_poses: &Array, allowed: &Array, num_inference_steps: usize,
    ) -> R<(Array, Array)> {
        self.predict_with(global_root, has_global, local_root, has_local, local_poses, has_poses, allowed, num_inference_steps, None)
    }

    /// Like [`Self::predict`]; `num_tokens` `(B,)` int32 forces the durations instead of letting the
    /// root network choose them (Python: `core.predict(..., num_tokens=...)`).
    #[allow(clippy::too_many_arguments)]
    pub fn predict_with(
        &self, global_root: &Array, has_global: &Array, local_root: &Array, has_local: &Array, local_poses: &Array,
        has_poses: &Array, allowed: &Array, num_inference_steps: usize, num_tokens: Option<&Array>,
    ) -> R<(Array, Array)> {
        if num_inference_steps == 0 {
            // Zero steps would decode the mask token, which has no codebook entry.
            return Err(mlx_rs::error::Exception::custom("num_inference_steps must be at least 1"));
        }
        let (rep, dt, f32t) = (&self.rep, self.dtype, Dtype::Float32);
        let f = FRAMES_PER_TOKEN;
        let batch = global_root.dim(0);

        // Recentre on the first context frame's ground position; remember its heading.
        let first = slice(global_root, 1, 0, 1)?;
        let offset_x = slice(&first, -1, 0, 1)?;
        let offset_z = slice(&first, -1, 2, 3)?;
        let g0 = index(global_root, 1, 0)?;
        let heading0 = atan2(&index(&g0, -1, 4)?, &index(&g0, -1, 3)?)?;
        let zero = ops::zeros_like(&offset_x)?;
        let recentred = global_root - &cat(&[&offset_x, &zero, &offset_z, &zero, &zero], -1)?;
        let g_n = rep.normalize_global_root(&recentred)?;
        let l_n = rep.normalize_local_root(local_root)?;
        let p_n = rep.normalize_local_pose(&cat(&[&slice(&recentred, -1, 1, 2)?, local_poses], -1)?)?;

        // 1) Duration + root path.
        let out = self.root.forward(
            &g_n.as_dtype(dt)?, has_global, &l_n.as_dtype(dt)?, has_local, &p_n.as_dtype(dt)?, has_poses, allowed, num_tokens,
        )?;
        let num_tokens = out.pred_num_tokens;
        let pred_root = out.pred_root.as_dtype(f32t)?;
        let lengths = &num_tokens * f;
        let pred_local_root = rep.global_root_to_local(&pred_root, &lengths)?;
        let last = unsq(&(&lengths - 1), 1)?;
        let estimated = pred_local_root.take_along_axis(unsq(&last, -1)?, 1)?;
        let h = unsq(&slice(has_local, 1, -1, END)?, -1)?.as_dtype(f32t)?;
        let final_ = ops::add(&h * &slice(&l_n, 1, -1, END)?, (i(1) - &h) * &estimated)?;
        let at_last = unsq(&unsq(&arange(MAX_FRAMES)?, 0)?.eq(&last)?, -1)?;
        let pred_local_root = wh(&at_last, &final_, &pred_local_root)?;

        // 2) Pose tokens (release config: one MaskGIT step, argmax sampling).
        let (pose_cond, has_pose_cond) = self.dense_pose_conditions(&p_n, has_poses, &num_tokens)?;
        let root_cond = pred_root.take_axis(&rep.global_root_no_height, -1)?;
        let heads = self.spec.vq.num_heads;
        let mut tokens = full(&[batch, MAX_TOKENS, heads], &i(self.pose.mask_id))?;
        let mut prob = full(tokens.shape(), &crate::ops::f((1.0 / heads as f64) as f32))?;
        for step in 0..num_inference_steps {
            tokens = self.remask(&tokens, &prob, &num_tokens, step, num_inference_steps)?;
            let logits = self.pose.forward(&tokens, &root_cond.as_dtype(dt)?, &pose_cond.as_dtype(dt)?, &has_pose_cond, &num_tokens)?;
            tokens = ops::indexing::argmax_axis(&logits, -1, None)?.as_dtype(Dtype::Int32)?;
            if step + 1 < num_inference_steps {
                let p = ops::softmax_axis(logits.as_dtype(f32t)?, -1, None)?;
                prob = index(&p.take_along_axis(unsq(&tokens, -1)?, -1)?, -1, 0)?;
            }
        }

        // 3) Decode tokens to local motion, integrate to global.
        let token_mask = unsq(&arange(MAX_TOKENS)?, 0)?.lt(unsq(&num_tokens, 1)?)?;
        let external = pred_local_root.take_axis(&rep.local_root_velocity, -1)?;
        let recon = self.vqvae.decode(&tokens, &pose_cond.as_dtype(dt)?, &has_pose_cond, &external.as_dtype(dt)?, &token_mask)?;
        let mut pred_global = rep.local_to_global(&recon.as_dtype(f32t)?)?;

        if self.spec.final_root_mode == FinalRootMode::FromHeightConstraintViolation {
            let root_from_root_module = rep.unnormalize_global_root(&pred_root)?;
            let target_frames = ops::add(&(&unsq(&num_tokens, 1)? - 1) * f, unsq(&arange(f)?, 0)?)?;
            let h_root = index(&root_from_root_module, -1, 1)?.take_along_axis(&target_frames, 1)?;
            let h_pose = index(&pred_global, -1, 1)?.take_along_axis(&target_frames, 1)?;
            let h_target = index(&slice(&rep.unnormalize_global_root(&g_n)?, 1, -f, END)?, -1, 1)?;
            let w = slice(has_global, 1, -f, END)?.as_dtype(f32t)?;
            let w_sum = w.sum_axis(1, None)?;
            let err_pose = ops::divide((&ops::abs(&h_pose - &h_target)? * &w).sum_axis(1, None)?, &w_sum)?;
            let err_root = ops::divide((&ops::abs(&h_root - &h_target)? * &w).sum_axis(1, None)?, &w_sum)?;
            let use_root = err_root.lt(&err_pose)?.logical_and(err_pose.gt(crate::ops::f(HEIGHT_ERROR_THRESHOLD))?)?;
            let use_root = use_root.reshape(&[batch, 1, 1])?.as_dtype(f32t)?;
            let root = ops::add(&use_root * &root_from_root_module, (i(1) - &use_root) * slice(&pred_global, -1, 0, 5)?)?;
            pred_global = cat(&[&root, &slice(&pred_global, -1, 5, END)?], -1)?;
        }

        // Undo the recentring: restore the ground offset and rotate headings back.
        let angle = ops::add(atan2(&index(&pred_global, -1, 4)?, &index(&pred_global, -1, 3)?)?, unsq(&heading0, 1)?)?;
        let root = stack(
            &[
                &ops::add(index(&pred_global, -1, 0)?, index(&offset_x, 2, 0)?)?,
                &index(&pred_global, -1, 1)?,
                &ops::add(index(&pred_global, -1, 2)?, index(&offset_z, 2, 0)?)?,
                &cos(&angle)?,
                &sin(&angle)?,
            ],
            -1,
        )?;
        Ok((cat(&[&root, &slice(&pred_global, -1, 5, END)?], -1)?, num_tokens))
    }
}
