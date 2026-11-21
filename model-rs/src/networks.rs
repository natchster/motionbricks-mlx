// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Root predictor, masked pose-token transformer, VQ-VAE decoder
//! (ported from `motionbricks_mlx/networks.py`). Masks are bool.

use mlx_rs::{Array, Dtype, ops};

use crate::config::*;
use crate::layers::*;
use crate::ops::*;
use crate::weights::Weights;

pub const GLOBAL_ROOT_DIM: i32 = 5;

/// `value * has + missing * (1 - has)` (has is bool, cast to value's dtype).
fn blend(value: &Array, missing: &Array, has: &Array) -> R<Array> {
    let h = unsq(has, -1)?.as_dtype(value.dtype())?;
    ops::add(value * &h, missing * (i(1) - &h))
}

pub struct RootNetwork {
    shared: TransformerEncoder,
    root_token: TransformerEncoder,
    conv_output: DoubleCondDecoder,
    proj_local_pose: Linear,
    proj_local_root_value: Linear,
    proj_global_root_value: Linear,
    no_local_pose_emb: Array,
    no_local_root_emb: Array,
    no_global_root_emb: Array,
    conv_no_frame_emb: Array,
    proj_start_input: FcBlock,
    proj_end_input: FcBlock,
    input_position_emb: Array,
    proj_input_num_tokens: Embedding,
    position_emb: PositionEmbedding,
    proj_num_token_output_logit: Linear,
    middle_token_emb: Embedding,
    n_embd: i32,
}

pub struct RootOutput {
    pub logits: Array,
    pub pred_num_tokens: Array,
    pub pred_root: Array,
}

impl RootNetwork {
    pub fn load(w: &mut Weights, spec: &RootSpec) -> R<Self> {
        let p = "root";
        let g = |w: &mut Weights, n: &str| w.get(&format!("{p}.{n}"));
        Ok(RootNetwork {
            shared: TransformerEncoder::load(w, &format!("{p}.shared_transformer_model"), spec.n_layers_shared, spec.n_head)?,
            root_token: TransformerEncoder::load(w, &format!("{p}.root_token_transformer_model"), spec.n_layers_root_token, spec.n_head)?,
            conv_output: DoubleCondDecoder::load(w, &format!("{p}.conv_output"), DOWN_T, spec.width, spec.depth, spec.dilation_growth_rate)?,
            proj_local_pose: Linear::load(w, &format!("{p}.proj_local_pose"))?,
            proj_local_root_value: Linear::load(w, &format!("{p}.proj_local_root_value"))?,
            proj_global_root_value: Linear::load(w, &format!("{p}.proj_global_root_value"))?,
            no_local_pose_emb: g(w, "no_local_pose_emb")?,
            no_local_root_emb: g(w, "no_local_root_emb")?,
            no_global_root_emb: g(w, "no_global_root_emb")?,
            conv_no_frame_emb: g(w, "conv_no_frame_emb")?,
            proj_start_input: FcBlock::load(w, &format!("{p}.proj_start_input"), 2)?,
            proj_end_input: FcBlock::load(w, &format!("{p}.proj_end_input"), 2)?,
            input_position_emb: g(w, "input_position_emb.weight")?,
            proj_input_num_tokens: Embedding::load(w, &format!("{p}.proj_input_num_tokens"))?,
            position_emb: PositionEmbedding::load(w, &format!("{p}.position_emb"))?,
            proj_num_token_output_logit: Linear::load(w, &format!("{p}.proj_num_token_output_logit"))?,
            middle_token_emb: Embedding::load(w, &format!("{p}.middle_token_emb"))?,
            n_embd: spec.n_embd,
        })
    }

    /// Inputs normalized `(B, 8, D)`, masks bool `(B, 8)`, `allowed` bool `(B, 11)`;
    /// `num_tokens` int32 `(B,)` (6..16 forces the duration, `MASKED_NUM_TOKENS` lets the model choose).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, global_root: &Array, has_global: &Array, local_root: &Array, has_local: &Array, poses: &Array,
        has_poses: &Array, allowed: &Array, num_tokens: Option<&Array>,
    ) -> R<RootOutput> {
        let f = FRAMES_PER_TOKEN;
        let batch = poses.dim(0);
        let e = self.n_embd;
        let lp = blend(&self.proj_local_pose.forward(poses)?, &self.no_local_pose_emb, has_poses)?;
        let lr = blend(&self.proj_local_root_value.forward(local_root)?, &self.no_local_root_emb, has_local)?;
        let gr = blend(&self.proj_global_root_value.forward(global_root)?, &self.no_global_root_emb, has_global)?;
        let start = self.proj_start_input.forward(&cat(&[&slice(&lp, 1, 0, f)?, &slice(&lr, 1, 0, f)?, &slice(&gr, 1, 0, f)?], -1)?)?;
        let end = self.proj_end_input.forward(&cat(&[&slice(&lp, 1, -f, END)?, &slice(&lr, 1, -f, END)?, &slice(&gr, 1, -f, END)?], -1)?)?;
        let frame_emb = cat(&[&start, &end], 1)?;
        let positioned = ops::add(&frame_emb, unsq(&self.input_position_emb, 0)?)?;

        let num_tokens = match num_tokens {
            Some(n) => n.as_dtype(Dtype::Int32)?,
            None => full(&[batch], &i(MASKED_NUM_TOKENS))?,
        };
        let query = unsq(&self.proj_input_num_tokens.forward(&(&num_tokens - MIN_TOKENS))?, 1)?;
        let first = self.shared.forward(&cat(&[&query, &positioned], 1)?, None)?;
        let logits = self.proj_num_token_output_logit.forward(&index(&first, 1, 0)?)?;
        // The out-of-reach class is never allowed; neither are durations the caller excluded.
        let neg_inf = scalar(f32::NEG_INFINITY, logits.dtype());
        let logits = cat(
            &[&wh(allowed, &slice(&logits, 1, 0, NUM_TOKEN_CHOICES)?, &neg_inf)?, &full(&[batch, 1], &neg_inf)?],
            -1,
        )?;
        let chosen = wh(
            &num_tokens.eq(i(MASKED_NUM_TOKENS))?,
            &ops::indexing::argmax_axis(&logits, -1, None)?.as_dtype(Dtype::Int32)?,
            &(&num_tokens - MIN_TOKENS),
        )?;
        let pred_num_tokens = &chosen + MIN_TOKENS;

        let token_mask = unsq(&arange(MAX_TOKENS)?, 0)?.lt(unsq(&pred_num_tokens, 1)?)?;
        let position_emb = ops::broadcast_to(unsq(&self.position_emb.table(MAX_TOKENS)?, 0)?, &[batch, MAX_TOKENS, e])?;
        let second_in = cat(&[&unsq(&self.middle_token_emb.forward(&chosen)?, 1)?, &positioned, &position_emb], 1)?;
        let padding = cat(&[&ops::zeros_dtype(&[batch, 1 + 2 * f], Dtype::Bool)?, &token_mask.logical_not()?], 1)?;
        let second = slice(&self.root_token.forward(&second_in, Some(&padding))?, 1, -MAX_TOKENS, END)?;

        // Dense 64-frame conditions: context frames first, target frames at the chosen end.
        let has_frame = has_poses.logical_or(has_global)?.logical_or(has_local)?;
        let frame = wh(&unsq(&has_frame, -1)?, &frame_emb, &self.conv_no_frame_emb)?;
        let rest = MAX_FRAMES - f;
        let mut dense_frame = cat(
            &[&slice(&frame, 1, 0, f)?, &ops::broadcast_to(&self.conv_no_frame_emb, &[batch, rest, frame.dim(-1)])?],
            1,
        )?;
        let mut dense_root = cat(
            &[&slice(global_root, 1, 0, f)?, &ops::zeros_dtype(&[batch, rest, GLOBAL_ROOT_DIM], global_root.dtype())?],
            1,
        )?;
        let mut dense_has_root = cat(&[&slice(has_global, 1, 0, f)?, &ops::zeros_dtype(&[batch, rest], Dtype::Bool)?], 1)?;
        let frames = unsq(&arange(MAX_FRAMES)?, 0)?;
        for k in 0..f {
            let at = frames.eq(unsq(&(&(&pred_num_tokens * f) - f + k), 1)?)?;
            let at3 = unsq(&at, -1)?;
            dense_frame = wh(&at3, &unsq(&index(&frame, 1, f + k)?, 1)?, &dense_frame)?;
            dense_root = wh(&at3, &unsq(&index(global_root, 1, f + k)?, 1)?, &dense_root)?;
            dense_has_root = wh(&at, &unsq(&index(has_global, 1, f + k)?, 1)?, &dense_has_root)?;
        }
        let pred_root = self.conv_output.forward(&second, &dense_frame, &dense_root, &dense_has_root, &token_mask)?;
        Ok(RootOutput { logits, pred_num_tokens, pred_root })
    }
}

pub struct PoseNetwork {
    spec: PoseSpec,
    heads: i32,
    pub mask_id: i32,
    num_with_mask: i32,
    transformer: TransformerEncoder,
    pose_token_emb: Embedding,
    proj_pose_token_emb: FcBlock,
    proj_local_root_values: Linear,
    proj_local_pose: Linear,
    proj_num_valid_positions: Embedding,
    proj_input: Linear,
    position_emb: PositionEmbedding,
    proj_pose_output_logit: Linear,
}

impl PoseNetwork {
    pub fn load(w: &mut Weights, spec: &PoseSpec, vq: &VqSpec) -> R<Self> {
        let p = "pose";
        Ok(PoseNetwork {
            spec: *spec,
            heads: vq.num_heads,
            mask_id: vq.codes_per_head,
            num_with_mask: vq.codes_per_head + 1,
            transformer: TransformerEncoder::load(w, &format!("{p}.transformer_model"), spec.n_layers, spec.n_head)?,
            pose_token_emb: Embedding::load(w, &format!("{p}.pose_token_emb"))?,
            proj_pose_token_emb: FcBlock::load(w, &format!("{p}.proj_pose_token_emb"), 2)?,
            proj_local_root_values: Linear::load(w, &format!("{p}.proj_local_root_values"))?,
            proj_local_pose: Linear::load(w, &format!("{p}.proj_local_pose"))?,
            proj_num_valid_positions: Embedding::load(w, &format!("{p}.proj_num_valid_positions"))?,
            proj_input: Linear::load(w, &format!("{p}.proj_input.0"))?,
            position_emb: PositionEmbedding::load(w, &format!("{p}.position_emb"))?,
            proj_pose_output_logit: Linear::load(w, &format!("{p}.proj_pose_output_logit"))?,
        })
    }

    /// pose_tokens int32 `(B, 16, H)`; root_cond `(B, 64, 4)`; pose_cond `(B, 64, D)`;
    /// has_pose_cond bool `(B, 64)`; num_tokens int32 `(B,)`. Returns logits `(B, 16, H, codes)`.
    pub fn forward(&self, pose_tokens: &Array, root_cond: &Array, pose_cond: &Array, has_pose_cond: &Array, num_tokens: &Array) -> R<Array> {
        let (batch, positions, heads) = (pose_tokens.dim(0), pose_tokens.dim(1), pose_tokens.dim(2));
        let f = FRAMES_PER_TOKEN;
        let width = self.spec.pose_feat_width;
        let feat = width / f;
        let root_emb = self.proj_local_root_values.forward(&root_cond.reshape(&[batch, positions, -1])?)?;
        let cond_emb = self.proj_local_pose.forward(pose_cond)?;
        let offsets = &arange(heads)? * self.num_with_mask;
        let token_emb = self.pose_token_emb.forward(&(pose_tokens + &offsets))?.reshape(&[batch, positions, -1])?;
        let token_emb = self.proj_pose_token_emb.forward(&token_emb)?.reshape(&[batch, positions * f, feat])?;
        let pose_emb = blend(&cond_emb, &token_emb, has_pose_cond)?.reshape(&[batch, positions, width])?;
        let length_emb = unsq(&self.proj_num_valid_positions.forward(&(num_tokens - MIN_TOKENS))?, 1)?;
        let length_emb = ops::broadcast_to(&length_emb, &[batch, positions, length_emb.dim(-1)])?;
        let x = relu(&self.proj_input.forward(&cat(&[&pose_emb, &root_emb, &length_emb], -1)?)?)?;
        let x = ops::add(&x, self.position_emb.table(positions)?)?;
        let padding = unsq(&arange(positions)?, 0)?.ge(unsq(num_tokens, 1)?)?;
        let out = self.transformer.forward(&x, Some(&padding))?;
        let _ = self.heads;
        self.proj_pose_output_logit.forward(&out)?.reshape(&[batch, positions, heads, -1])
    }
}

/// Product-codebook VQ-VAE decoder (`quantizer.dequantize` + `DoubleCondDecoder`).
pub struct VqDecoder {
    codebook: Array, // (H * E, W), flattened once at load time
    heads: i32,
    entries: i32,
    width: i32,
    decoder: DoubleCondDecoder,
}

impl VqDecoder {
    pub fn load(w: &mut Weights, vq: &VqSpec) -> R<Self> {
        let embed = w.get("vqvae.quantizer.vq.codebook.embed")?;
        let (heads, entries, width) = (embed.dim(0), embed.dim(1), embed.dim(2));
        Ok(VqDecoder {
            codebook: embed,
            heads,
            entries,
            width,
            decoder: DoubleCondDecoder::load(w, "vqvae.decoder", DOWN_T, vq.width, vq.depth, vq.dilation_growth_rate)?,
        })
    }

    pub fn dequantize(&self, indices: &Array) -> R<Array> {
        let flat = indices + &(&arange(self.heads)? * self.entries);
        let codes = self.codebook.reshape(&[self.heads * self.entries, self.width])?.take_axis(&flat, 0)?;
        codes.reshape(&[indices.dim(0), indices.dim(1), self.heads * self.width])
    }

    pub fn decode(&self, tokens: &Array, target: &Array, has_target: &Array, external: &Array, token_mask: &Array) -> R<Array> {
        let latents = self.dequantize(tokens)?;
        self.decoder.forward(&latents, external, target, has_target, token_mask)
    }
}
