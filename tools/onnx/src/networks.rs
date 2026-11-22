// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The three networks: the root predictor (duration + root trajectory), the masked
//! pose-token transformer, and the VQ-VAE decoder. Mirrors `model-rs/src/networks.rs`
//! (ground truth: upstream `tools/onnx/models/**/networks.py`), but every shape the
//! torch tracer defers into the graph is resolved here at emit time; only the
//! data-dependent index arithmetic on `pred_num_tokens` becomes runtime nodes.
//!
//! The exported graph always lets the model choose the duration (upstream exports with
//! `num_tokens=None` -> `MASKED_NUM_TOKENS`), so the forced-duration `where` of the
//! eager model is not emitted: `chosen = argmax(masked_logits)` directly.

use anyhow::{Result, bail};

use crate::config::{CONTEXT_AND_TARGET, FRAMES_PER_TOKEN, Kind, MASKED_NUM_TOKENS, MAX_FRAMES, MAX_TOKENS, MIN_TOKENS, ModelSpec, NUM_TOKEN_CHOICES};
use crate::graph::{Attr, DType, Dim, Graph, broadcast};
use crate::layers::{DoubleCondDecoder, Embedding, FcBlock, Linear, PositionEmbedding, TransformerEncoder};
use crate::onnx;
use crate::ops::*;
use crate::weights::Weights;

// Shared architecture constants (model-rs `RootSpec`/`PoseSpec`/`VqSpec`; identical for
// both released models, so they are not carried by `config::ModelSpec`).
const N_HEAD: i64 = 16;
const POSE_N_LAYERS: usize = 16;
const DOWN_T: i64 = 2;
const DECODER_WIDTH: i64 = 512;
const DECODER_DEPTH: usize = 4;
const DILATION_GROWTH_RATE: i64 = 3;
/// Values of the normalized global root feature (`root`).
const GLOBAL_ROOT_DIM: i64 = 5;

/// The node-name scope of the root and pose networks; the mixed-FP16 pass keys on this
/// prefix (upstream `tools/onnx/precision.py`).
pub fn backbone_scope(spec: &ModelSpec) -> &'static str {
    match spec.kind {
        Kind::G1 => "/core/backbone_net",
        Kind::Soma => "/mmm_net",
    }
}

/// The node-name scope of the VQ-VAE decoder (dequantize + conv decoder).
pub fn decoder_scope(spec: &ModelSpec) -> &'static str {
    match spec.kind {
        Kind::G1 => "/core/decoder",
        Kind::Soma => "/decoder",
    }
}

/// An int64 elementwise binary op (the `ops` helpers are float-only).
fn int_bin(g: &mut Graph, scope: &str, op: &str, a: &str, b: &str) -> String {
    let out = broadcast(g.shape(a), g.shape(b)).expect("int op shapes");
    g.node(scope, op, &[a, b], &[], DType::I64, &out)
}

fn add_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    int_bin(g, scope, "Add", a, b)
}
fn sub_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    int_bin(g, scope, "Sub", a, b)
}
fn mul_i64(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    int_bin(g, scope, "Mul", a, b)
}

/// A scalar int32 initializer, deduplicated per scope+value (like `c_scalar_i64`).
fn c_scalar_i32(g: &mut Graph, scope: &str, value: i32) -> String {
    let name = format!("{scope}/const_i32_{value}");
    if g.dtypes.contains_key(&name) {
        return name;
    }
    g.initializer(onnx::tensor_i32(&name, &[], &[value]))
}

/// `ConstantOfShape` with an int64 scalar value.
fn const_of_shape_i64(g: &mut Graph, scope: &str, shape_t: &str, value: i64, dims: &[Dim]) -> String {
    let t = onnx::tensor_i64(&g.fresh("cos_i64_value"), &[1], &[value]);
    g.node(scope, "ConstantOfShape", &[shape_t], &[("value", Attr::Tensor(t))], DType::I64, dims)
}

/// `ConstantOfShape` with a bool scalar value.
fn const_of_shape_bool(g: &mut Graph, scope: &str, shape_t: &str, value: bool, dims: &[Dim]) -> String {
    let t = onnx::tensor_bool(&g.fresh("cos_bool_value"), &[1], &[value]);
    g.node(scope, "ConstantOfShape", &[shape_t], &[("value", Attr::Tensor(t))], DType::Bool, dims)
}

/// The runtime shape `[B] ++ tail` (int64) of a batched reference tensor.
fn batch_shape(g: &mut Graph, scope: &str, batch_ref: &str, tail: &[i64]) -> String {
    let sh = shape_of(g, scope, batch_ref);
    let b = gather_static(g, scope, &sh, &[0], 0); // [1]
    if tail.is_empty() {
        return b;
    }
    let tail_t = c_vec(g, scope, "tail_dims", tail);
    concat(g, scope, &[&b, &tail_t], 0)
}

/// `Expand` with a runtime target shape (for constants that need the batch dimension).
fn expand_runtime(g: &mut Graph, scope: &str, x: &str, shape_t: &str, dims: &[Dim]) -> String {
    let dtype = g.dtype(x);
    g.node(scope, "Expand", &[x, shape_t], &[], dtype, dims)
}

/// A bool view of a 0/1 mask input (bool passes through, ints cast).
fn as_bool(g: &mut Graph, scope: &str, x: &str) -> String {
    if g.dtype(x) == DType::Bool {
        x.to_string()
    } else {
        cast(g, scope, x, DType::Bool)
    }
}

/// `value * has + missing * (1 - has)` (model-rs `blend`): `value` and `missing` are
/// float, `has` is a 0/1 mask broadcast over the last axis. A bool `has` emits the
/// exactly equivalent `Where`.
fn blend(g: &mut Graph, scope: &str, value: &str, missing: &str, has: &str) -> String {
    let axis = g.shape(has).len() as i64;
    if g.dtype(has) == DType::Bool {
        let h = unsqueeze(g, scope, has, &[axis]);
        return where_(g, scope, &h, value, missing);
    }
    let mut h = unsqueeze(g, scope, has, &[axis]);
    if g.dtype(&h) != DType::F32 {
        h = cast(g, scope, &h, DType::F32);
    }
    let one = c_scalar(g, scope, 1.0);
    let inv = sub(g, scope, &one, &h);
    let present = mul(g, scope, value, &h);
    let absent = mul(g, scope, missing, &inv);
    add(g, scope, &present, &absent)
}

/// The static size of the last dimension of a value.
fn rows_dim(g: &Graph, x: &str) -> i64 {
    let s = g.shape(x);
    s[s.len() - 1].get().expect("static last dim")
}

/// Root predictor: shared transformer (duration query + 8 context/target slots), the
/// duration head, the root-token transformer, and the conv decoder of the dense root
/// trajectory. Weight prefix `root` (model-rs `RootNetwork`).
pub struct RootNetwork {
    shared: TransformerEncoder,
    root_token: TransformerEncoder,
    conv_output: DoubleCondDecoder,
    proj_local_pose: Linear,
    proj_local_root_value: Linear,
    proj_global_root_value: Linear,
    no_local_pose_emb: String,
    no_local_root_emb: String,
    no_global_root_emb: String,
    conv_no_frame_emb: String,
    proj_start_input: FcBlock,
    proj_end_input: FcBlock,
    input_position_emb: String,
    proj_input_num_tokens: Embedding,
    position_emb: PositionEmbedding,
    proj_num_token_output_logit: Linear,
    middle_token_emb: Embedding,
}

/// The graph values the root predictor hands to the caller.
pub struct RootOutput {
    /// `pred_num_tokens`, int32 `[B]` (`chosen + MIN_TOKENS`).
    pub num_tokens: String,
    /// The same value as int64 `[B]`, for downstream runtime index arithmetic.
    pub num_tokens_i64: String,
    /// The dense root trajectory `[B, 64, 5]`, still NORMALIZED (the caller unnormalizes).
    pub global_root_values: String,
}

impl RootNetwork {
    pub fn load(g: &mut Graph, w: &mut Weights, spec: &ModelSpec) -> Result<Self> {
        let p = "root";
        let raw = |g: &mut Graph, w: &mut Weights, key: &str| -> Result<String> {
            let t = w.get(&format!("{p}.{key}"))?;
            Ok(c_f32(g, &format!("{p}.{key}"), &t.shape, &t.data))
        };
        Ok(RootNetwork {
            shared: TransformerEncoder::load(g, w, &format!("{p}.shared_transformer_model"), spec.root_layers_shared, N_HEAD)?,
            root_token: TransformerEncoder::load(g, w, &format!("{p}.root_token_transformer_model"), spec.root_layers_token, N_HEAD)?,
            conv_output: DoubleCondDecoder::load(g, w, &format!("{p}.conv_output"), DOWN_T, DECODER_WIDTH, DECODER_DEPTH, DILATION_GROWTH_RATE)?,
            proj_local_pose: Linear::load(g, w, &format!("{p}.proj_local_pose"))?,
            proj_local_root_value: Linear::load(g, w, &format!("{p}.proj_local_root_value"))?,
            proj_global_root_value: Linear::load(g, w, &format!("{p}.proj_global_root_value"))?,
            no_local_pose_emb: raw(g, w, "no_local_pose_emb")?,
            no_local_root_emb: raw(g, w, "no_local_root_emb")?,
            no_global_root_emb: raw(g, w, "no_global_root_emb")?,
            conv_no_frame_emb: raw(g, w, "conv_no_frame_emb")?,
            proj_start_input: FcBlock::load(g, w, &format!("{p}.proj_start_input"))?,
            proj_end_input: FcBlock::load(g, w, &format!("{p}.proj_end_input"))?,
            input_position_emb: raw(g, w, "input_position_emb.weight")?,
            proj_input_num_tokens: Embedding::load(g, w, &format!("{p}.proj_input_num_tokens"))?,
            position_emb: PositionEmbedding::load(g, w, &format!("{p}.position_emb"))?,
            proj_num_token_output_logit: Linear::load(g, w, &format!("{p}.proj_num_token_output_logit"))?,
            middle_token_emb: Embedding::load(g, w, &format!("{p}.middle_token_emb"))?,
        })
    }

    /// Mirrors `model-rs` `RootNetwork::forward` with `num_tokens=None` (the model always
    /// chooses the duration in the exported graph). Inputs are the NORMALIZED features:
    /// `local_poses` f32 `[B, 8, pose_in]`, `local_root` f32 `[B, 8, 4]`, `global_root`
    /// f32 `[B, 8, 5]`, the `has_*` masks bool `[B, 8]`, and `allowed_pred_num_tokens`
    /// `[B, 11]` (bool, or an int mask with 1 = allowed). `scope` is the network's node
    /// scope ([`backbone_scope`]).
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        g: &mut Graph,
        scope: &str,
        local_poses: &str,
        has_local_poses: &str,
        local_root: &str,
        has_local_root: &str,
        global_root: &str,
        has_global_root: &str,
        allowed_pred_num_tokens: &str,
    ) -> Result<RootOutput> {
        let f = FRAMES_PER_TOKEN;
        let slots = CONTEXT_AND_TARGET; // 8

        // Per-slot feature projections, each blended with its "missing" embedding.
        let lp = self.proj_local_pose.emit(g, scope, local_poses);
        let lp = blend(g, scope, &lp, &self.no_local_pose_emb, has_local_poses);
        let lr = self.proj_local_root_value.emit(g, scope, local_root);
        let lr = blend(g, scope, &lr, &self.no_local_root_emb, has_local_root);
        let gr = self.proj_global_root_value.emit(g, scope, global_root);
        let gr = blend(g, scope, &gr, &self.no_global_root_emb, has_global_root);

        // Context (first 4 slots) and target (last 4 slots) frame embeddings.
        let start = {
            let a = slice_axis(g, scope, &lp, 1, 0, f);
            let b = slice_axis(g, scope, &lr, 1, 0, f);
            let c = slice_axis(g, scope, &gr, 1, 0, f);
            let cat = concat(g, scope, &[&a, &b, &c], 2);
            self.proj_start_input.emit(g, scope, &cat)
        };
        let end = {
            let a = slice_axis(g, scope, &lp, 1, f, slots);
            let b = slice_axis(g, scope, &lr, 1, f, slots);
            let c = slice_axis(g, scope, &gr, 1, f, slots);
            let cat = concat(g, scope, &[&a, &b, &c], 2);
            self.proj_end_input.emit(g, scope, &cat)
        };
        let frame_emb = concat(g, scope, &[&start, &end], 1); // [B, 8, 512]
        let positioned = add(g, scope, &frame_emb, &self.input_position_emb);

        // First stage: [duration query, 8 frame slots] through the shared transformer.
        // The exported graph always uses the MASKED_NUM_TOKENS query row.
        let query = {
            let shape_t = batch_shape(g, scope, local_poses, &[]);
            let ids = const_of_shape_i64(g, scope, &shape_t, MASKED_NUM_TOKENS - MIN_TOKENS, &[Dim::Batch]);
            let q = self.proj_input_num_tokens.emit(g, scope, &ids); // [B, 512]
            unsqueeze(g, scope, &q, &[1])
        };
        let first_in = concat(g, scope, &[&query, &positioned], 1); // [B, 9, 512]
        let first = self.shared.emit(g, &format!("{scope}/_shared_transformer_model"), &first_in, None);

        // Duration logits of the query position: 11 reachable classes + out-of-reach.
        let tok0 = {
            let t = slice_axis(g, scope, &first, 1, 0, 1);
            squeeze(g, scope, &t, &[1])
        };
        let logits = self.proj_num_token_output_logit.emit(g, scope, &tok0); // [B, 12]
        let allowed_cond = match g.dtype(allowed_pred_num_tokens) {
            DType::Bool => allowed_pred_num_tokens.to_string(),
            DType::I32 => {
                let one = c_scalar_i32(g, scope, 1);
                equal(g, scope, allowed_pred_num_tokens, &one)
            }
            DType::I64 => {
                let one = c_scalar_i64(g, scope, 1);
                equal(g, scope, allowed_pred_num_tokens, &one)
            }
            other => bail!("allowed_pred_num_tokens must be bool or int, got {other:?}"),
        };
        let neg_inf = c_scalar(g, scope, f32::NEG_INFINITY);
        let masked = {
            let reachable = slice_axis(g, scope, &logits, 1, 0, NUM_TOKEN_CHOICES);
            where_(g, scope, &allowed_cond, &reachable, &neg_inf)
        };
        let logits = {
            let shape_t = batch_shape(g, scope, local_poses, &[1]);
            let out_of_reach = constant_of_shape(g, scope, &shape_t, f32::NEG_INFINITY, &[Dim::Batch, Dim::Static(1)]);
            concat(g, scope, &[&masked, &out_of_reach], 1) // [B, 12], class 11 = -inf
        };
        let chosen = argmax(g, scope, &logits, 1, false); // int64 [B]
        let min_tokens = c_scalar_i64(g, scope, MIN_TOKENS);
        let pred = add_i64(g, scope, &chosen, &min_tokens);
        let num_tokens = cast(g, scope, &pred, DType::I32);

        // Second stage: [middle token, 8 frame slots, 16 position rows] through the
        // root-token transformer; the position rows see a key-padding mask.
        let token_mask = {
            let positions = arange(g, scope, MAX_TOKENS);
            let n = unsqueeze(g, scope, &pred, &[1]);
            less(g, scope, &positions, &n) // [B, 16] bool
        };
        let position_emb = {
            let rows = self.position_emb.emit_rows(g, scope, 0, MAX_TOKENS); // [1, 16, 512]
            let dim = rows_dim(g, &rows);
            let shape_t = batch_shape(g, scope, local_poses, &[MAX_TOKENS, dim]);
            expand_runtime(g, scope, &rows, &shape_t, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(dim)])
        };
        let middle = {
            let m = self.middle_token_emb.emit(g, scope, &chosen); // [B, 512]
            unsqueeze(g, scope, &m, &[1])
        };
        let second_in = concat(g, scope, &[&middle, &positioned, &position_emb], 1); // [B, 25, 512]
        let padding = {
            let shape_t = batch_shape(g, scope, local_poses, &[1 + 2 * f]);
            let zeros = const_of_shape_bool(g, scope, &shape_t, false, &[Dim::Batch, Dim::Static(1 + 2 * f)]);
            let padded = not_(g, scope, &token_mask);
            concat(g, scope, &[&zeros, &padded], 1) // [B, 25] bool, true = padded
        };
        let second = {
            let all = self.root_token.emit(g, &format!("{scope}/_root_token_transformer_model"), &second_in, Some(&padding));
            slice_axis(g, scope, &all, 1, 1 + 2 * f, 1 + 2 * f + MAX_TOKENS)
        };

        // Dense 64-frame decoder conditions: context frames first, the 4 target frames
        // scattered at the data-dependent positions `pred_num_tokens * 4 - 4 + k`.
        let has_global_root_b = as_bool(g, scope, has_global_root);
        let has_frame = {
            let has_poses_b = as_bool(g, scope, has_local_poses);
            let has_root_b = as_bool(g, scope, has_local_root);
            let ab = or_(g, scope, &has_poses_b, &has_global_root_b);
            or_(g, scope, &ab, &has_root_b)
        };
        let frame = {
            let h = unsqueeze(g, scope, &has_frame, &[2]);
            where_(g, scope, &h, &frame_emb, &self.conv_no_frame_emb)
        };
        let rest = MAX_FRAMES - f;
        let frame_dim = rows_dim(g, &frame);
        let mut dense_frame = {
            let first4 = slice_axis(g, scope, &frame, 1, 0, f);
            let shape_t = batch_shape(g, scope, local_poses, &[rest, frame_dim]);
            let filler = expand_runtime(g, scope, &self.conv_no_frame_emb, &shape_t, &[Dim::Batch, Dim::Static(rest), Dim::Static(frame_dim)]);
            concat(g, scope, &[&first4, &filler], 1)
        };
        let mut dense_root = {
            let first4 = slice_axis(g, scope, global_root, 1, 0, f);
            let shape_t = batch_shape(g, scope, local_poses, &[rest, GLOBAL_ROOT_DIM]);
            let zeros = constant_of_shape(g, scope, &shape_t, 0.0, &[Dim::Batch, Dim::Static(rest), Dim::Static(GLOBAL_ROOT_DIM)]);
            concat(g, scope, &[&first4, &zeros], 1)
        };
        let mut dense_has_root = {
            let first4 = slice_axis(g, scope, &has_global_root_b, 1, 0, f);
            let shape_t = batch_shape(g, scope, local_poses, &[rest]);
            let zeros = const_of_shape_bool(g, scope, &shape_t, false, &[Dim::Batch, Dim::Static(rest)]);
            concat(g, scope, &[&first4, &zeros], 1)
        };
        let base = {
            let f_t = c_scalar_i64(g, scope, f);
            let times_f = mul_i64(g, scope, &pred, &f_t);
            sub_i64(g, scope, &times_f, &f_t)
        };
        for k in 0..f {
            let k_t = c_scalar_i64(g, scope, k);
            let at = add_i64(g, scope, &base, &k_t); // int64 [B]
            let at3 = unsqueeze(g, scope, &at, &[1, 2]); // [B, 1, 1]
            let frame_idx = expand(g, scope, &at3, &[Dim::Batch, Dim::Static(1), Dim::Static(frame_dim)]);
            let root_idx = expand(g, scope, &at3, &[Dim::Batch, Dim::Static(1), Dim::Static(GLOBAL_ROOT_DIM)]);
            let has_idx = unsqueeze(g, scope, &at, &[1]); // [B, 1]
            let frame_upd = slice_axis(g, scope, &frame, 1, f + k, f + k + 1);
            let root_upd = slice_axis(g, scope, global_root, 1, f + k, f + k + 1);
            let has_upd = slice_axis(g, scope, &has_global_root_b, 1, f + k, f + k + 1);
            dense_frame = scatter_elements(g, scope, &dense_frame, &frame_idx, &frame_upd, 1);
            dense_root = scatter_elements(g, scope, &dense_root, &root_idx, &root_upd, 1);
            dense_has_root = scatter_elements(g, scope, &dense_has_root, &has_idx, &has_upd, 1);
        }

        // The conv decoder takes channels-first input (upstream transposes around the
        // call) and returns the channels-last dense trajectory.
        let token_mask_f = cast(g, scope, &token_mask, DType::F32); // [B, 16]
        let second_ct = transpose(g, scope, &second, &[0, 2, 1]); // [B, 512, 16]
        let global_root_values =
            self.conv_output.emit(g, &format!("{scope}/_conv_output"), &second_ct, &dense_frame, &dense_root, &dense_has_root, &token_mask_f);

        Ok(RootOutput { num_tokens, num_tokens_i64: pred, global_root_values })
    }
}

/// Masked-token transformer predicting the pose-code logits (model-rs `PoseNetwork`).
pub struct PoseNetwork {
    /// Token value that marks a masked pose token (`codes_per_head`).
    pub mask_id: i64,
    /// Embedding rows per head, including the mask token (`codes_per_head + 1`).
    pub num_with_mask: i64,
    transformer: TransformerEncoder,
    pose_token_emb: Embedding,
    proj_pose_token_emb: FcBlock,
    proj_local_root_values: Linear,
    proj_local_pose: Linear,
    proj_num_valid_positions: Embedding,
    proj_input: Linear,
    position_emb: PositionEmbedding,
    proj_pose_output_logit: Linear,
    heads: i64,
    codes: i64,
}

impl PoseNetwork {
    pub fn load(g: &mut Graph, w: &mut Weights, spec: &ModelSpec) -> Result<Self> {
        let p = "pose";
        Ok(PoseNetwork {
            mask_id: spec.vq_codes,
            num_with_mask: spec.vq_codes + 1,
            transformer: TransformerEncoder::load(g, w, &format!("{p}.transformer_model"), POSE_N_LAYERS, N_HEAD)?,
            pose_token_emb: Embedding::load(g, w, &format!("{p}.pose_token_emb"))?,
            proj_pose_token_emb: FcBlock::load(g, w, &format!("{p}.proj_pose_token_emb"))?,
            proj_local_root_values: Linear::load(g, w, &format!("{p}.proj_local_root_values"))?,
            proj_local_pose: Linear::load(g, w, &format!("{p}.proj_local_pose"))?,
            proj_num_valid_positions: Embedding::load(g, w, &format!("{p}.proj_num_valid_positions"))?,
            proj_input: Linear::load(g, w, &format!("{p}.proj_input.0"))?,
            position_emb: PositionEmbedding::load(g, w, &format!("{p}.position_emb"))?,
            proj_pose_output_logit: Linear::load(g, w, &format!("{p}.proj_pose_output_logit"))?,
            heads: spec.vq_heads,
            codes: spec.vq_codes,
        })
    }

    /// `pose_tokens` int64 `[B, 16, H]` (all-masked at emit time); `root_cond` f32
    /// `[B, 64, 4]`; `pose_cond` f32 `[B, 64, pose_dim]`; `has_pose_cond` `[B, 64]`
    /// (bool or 0/1 float); `num_tokens` int64 `[B]`. Returns the logits
    /// `[B, 16, H, codes]` (the head emits `H * codes` values — no mask logit).
    pub fn emit(
        &self,
        g: &mut Graph,
        scope: &str,
        pose_tokens: &str,
        root_cond: &str,
        pose_cond: &str,
        has_pose_cond: &str,
        num_tokens: &str,
    ) -> Result<String> {
        let f = FRAMES_PER_TOKEN;
        let (heads, codes) = (self.heads, self.codes);

        // Root conditioning: the 4 root values of each token, projected.
        let root_emb = {
            let flat = reshape(g, scope, root_cond, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(f * 4)]);
            self.proj_local_root_values.emit(g, scope, &flat) // [B, 16, 256]
        };
        // Pose conditioning, already dense (64 frames), projected per frame.
        let cond_emb = self.proj_local_pose.emit(g, scope, pose_cond); // [B, 64, 160]
        // Token embeddings: Gather with per-head id offsets, through the FC block.
        let token_emb = {
            let offsets = {
                let h = arange(g, scope, heads);
                let nwm = c_scalar_i64(g, scope, self.num_with_mask);
                mul_i64(g, scope, &h, &nwm) // [H]
            };
            let ids = add_i64(g, scope, pose_tokens, &offsets); // [B, 16, H]
            let emb = self.pose_token_emb.emit(g, scope, &ids); // [B, 16, H, 32]
            let emb_dim = rows_dim(g, &emb);
            let flat = reshape(g, scope, &emb, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(heads * emb_dim)]);
            self.proj_pose_token_emb.emit(g, scope, &flat) // [B, 16, pose_feat_width]
        };
        let token_emb = {
            let feat = rows_dim(g, &cond_emb);
            reshape(g, scope, &token_emb, &[Dim::Batch, Dim::Static(MAX_TOKENS * f), Dim::Static(feat)])
        };
        // Real conditioning where present, the (all-masked) token embedding elsewhere.
        let pose_emb = {
            let blended = blend(g, scope, &cond_emb, &token_emb, has_pose_cond); // [B, 64, 160]
            let width = rows_dim(g, &blended) * f;
            reshape(g, scope, &blended, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(width)])
        };
        // Duration embedding broadcast to all positions.
        let length_emb = {
            let min_tokens = c_scalar_i64(g, scope, MIN_TOKENS);
            let ids = sub_i64(g, scope, num_tokens, &min_tokens);
            let e = self.proj_num_valid_positions.emit(g, scope, &ids); // [B, 128]
            let e = unsqueeze(g, scope, &e, &[1]);
            let dim = rows_dim(g, &e);
            expand(g, scope, &e, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(dim)])
        };
        let x = {
            let cat = concat(g, scope, &[&pose_emb, &root_emb, &length_emb], 2);
            let projected = self.proj_input.emit(g, scope, &cat);
            relu(g, scope, &projected)
        };
        let x = {
            let pe = self.position_emb.emit_rows(g, scope, 0, MAX_TOKENS); // [1, 16, n_embd]
            add(g, scope, &x, &pe)
        };
        let padding = {
            let positions = arange(g, scope, MAX_TOKENS);
            let n = unsqueeze(g, scope, num_tokens, &[1]);
            let valid = less(g, scope, &positions, &n);
            not_(g, scope, &valid) // [B, 16] bool, true = padded
        };
        let out = self.transformer.emit(g, &format!("{scope}/_transformer_model"), &x, Some(&padding));
        let logits = self.proj_pose_output_logit.emit(g, scope, &out); // [B, 16, H * codes]
        Ok(reshape(g, scope, &logits, &[Dim::Batch, Dim::Static(MAX_TOKENS), Dim::Static(heads), Dim::Static(codes)]))
    }
}

/// Product-codebook VQ-VAE decoder: per-head codebook Gather + the conv decoder
/// (model-rs `VqDecoder`). Weight prefix `vqvae`.
pub struct VqDecoder {
    codebook: String, // [H, codes, W]
    decoder: DoubleCondDecoder,
    heads: i64,
}

impl VqDecoder {
    pub fn load(g: &mut Graph, w: &mut Weights, spec: &ModelSpec) -> Result<Self> {
        let t = w.get("vqvae.quantizer.vq.codebook.embed")?;
        let [heads, codes, width] = t.shape.as_slice() else {
            bail!("vqvae.quantizer.vq.codebook.embed must be [heads, codes, width], got {:?}", t.shape);
        };
        let (heads, codes, width) = (*heads, *codes, *width);
        if heads != spec.vq_heads || codes != spec.vq_codes || width * heads != spec.vq_code_dim {
            bail!(
                "codebook [{heads}, {codes}, {width}] does not match the spec ({} heads, {} codes, code dim {})",
                spec.vq_heads,
                spec.vq_codes,
                spec.vq_code_dim
            );
        }
        Ok(VqDecoder {
            codebook: c_f32(g, "vqvae.quantizer.vq.codebook.embed", &t.shape, &t.data),
            decoder: DoubleCondDecoder::load(g, w, "vqvae.decoder", DOWN_T, DECODER_WIDTH, DECODER_DEPTH, DILATION_GROWTH_RATE)?,
            heads,
        })
    }

    /// `tokens` int64 `[B, 16, H]`; `token_mask` f32 `[B, 16]` (1 = valid token);
    /// `target_cond` f32 `[B, 64, target_dim]`; `has_target` bool `[B, 64]`;
    /// `external` f32 `[B, 64, external_dim]` (the root-velocity features the caller
    /// derives from the predicted root; model-rs `VqDecoder::decode`'s `external`).
    /// Returns the decoded local poses `[B, 64, out_dim]`.
    pub fn emit(
        &self,
        g: &mut Graph,
        scope: &str,
        tokens: &str,
        token_mask: &str,
        target_cond: &str,
        has_target: &str,
        external: &str,
    ) -> Result<String> {
        // Per-head codebook lookup, then concat: [B, 16, H * W] (`dequantize`).
        let mut embs = Vec::with_capacity(self.heads as usize);
        for h in 0..self.heads {
            let cb = {
                let cb = gather_static(g, scope, &self.codebook, &[h], 0); // [1, codes, W]
                squeeze(g, scope, &cb, &[0]) // [codes, W]
            };
            let ids = {
                let col = slice_axis(g, scope, tokens, 2, h, h + 1); // [B, 16, 1]
                squeeze(g, scope, &col, &[2]) // [B, 16]
            };
            embs.push(gather(g, scope, &cb, &ids, 0)); // [B, 16, W]
        }
        let latents = concat(g, scope, &embs.iter().map(String::as_str).collect::<Vec<_>>(), 2); // [B, 16, H * W]
        let latents_ct = transpose(g, scope, &latents, &[0, 2, 1]); // [B, H * W, 16]
        Ok(self.decoder.emit(g, scope, &latents_ct, external, target_cond, has_target, token_mask)) // [B, 64, out_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    /// Emit all three networks of `spec` from the real checkpoints into one graph with
    /// dummy inputs, and check the emitted shapes and dtypes. Every `ops` helper
    /// resolves its inputs' static shapes while emitting, so a successful emit also
    /// proves the graph wiring (an unknown or mis-shaped value panics).
    fn smoke(spec: &ModelSpec) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../checkpoints/MotionBricks")
            .join(spec.kind_name);
        let Ok(mut loaded) = crate::weights::read_checkpoints(&dir, spec) else {
            eprintln!("skipping networks smoke test: no checkpoints at {}", dir.display());
            return;
        };
        let w = &mut loaded.weights;
        let in_dim = |w: &mut Weights, key: &str| w.get(key).unwrap().shape[1];
        let pose_in = in_dim(w, "root.proj_local_pose.weight");
        let pose_cond_dim = in_dim(w, "pose.proj_local_pose.weight");
        let target_dim = in_dim(w, "vqvae.decoder.target_cond_blocks.0.weight");
        // external_cond_blocks.0 fuses [hidden (512) | external (ext_dim * 4)].
        let external_dim = (in_dim(w, "vqvae.decoder.external_cond_blocks.0.weight") - 512) / 4;
        let out_dim = w.get("vqvae.decoder.model.6.weight").unwrap().shape[0]; // out_conv = model.{4 + down_t}

        let mut g = Graph::new();
        let root = RootNetwork::load(&mut g, w, spec).unwrap();
        let pose = PoseNetwork::load(&mut g, w, spec).unwrap();
        let vq = VqDecoder::load(&mut g, w, spec).unwrap();

        let b = || Dim::Batch;
        let s = Dim::Static;
        let local_poses = g.input("local_poses", DType::F32, &[b(), s(8), s(pose_in)]);
        let has_local_poses = g.input("has_local_poses", DType::Bool, &[b(), s(8)]);
        let local_root = g.input("local_root", DType::F32, &[b(), s(8), s(4)]);
        let has_local_root = g.input("has_local_root", DType::Bool, &[b(), s(8)]);
        let global_root = g.input("global_root", DType::F32, &[b(), s(8), s(5)]);
        let has_global_root = g.input("has_global_root", DType::Bool, &[b(), s(8)]);
        // The graph inputs are int32 masks; the emit must handle them.
        let allowed = g.input("allowed_pred_num_tokens", DType::I32, &[b(), s(11)]);

        let out = root
            .emit(&mut g, backbone_scope(spec), &local_poses, &has_local_poses, &local_root, &has_local_root, &global_root, &has_global_root, &allowed)
            .unwrap();
        assert_eq!((g.dtype(&out.num_tokens), g.shape(&out.num_tokens)), (DType::I32, &[b()][..]));
        assert_eq!((g.dtype(&out.num_tokens_i64), g.shape(&out.num_tokens_i64)), (DType::I64, &[b()][..]));
        assert_eq!((g.dtype(&out.global_root_values), g.shape(&out.global_root_values)), (DType::F32, &[b(), s(64), s(5)][..]));

        let tokens = g.input("pose_tokens", DType::I64, &[b(), s(16), s(spec.vq_heads)]);
        let root_cond = g.input("root_cond", DType::F32, &[b(), s(64), s(4)]);
        let pose_cond = g.input("pose_cond", DType::F32, &[b(), s(64), s(pose_cond_dim)]);
        let has_pose_cond = g.input("has_pose_cond", DType::Bool, &[b(), s(64)]);
        let logits = pose.emit(&mut g, backbone_scope(spec), &tokens, &root_cond, &pose_cond, &has_pose_cond, &out.num_tokens_i64).unwrap();
        assert_eq!(
            (g.dtype(&logits), g.shape(&logits)),
            (DType::F32, &[b(), s(16), s(spec.vq_heads), s(spec.vq_codes)][..])
        );

        let token_mask = g.input("token_mask", DType::F32, &[b(), s(16)]);
        let target_cond = g.input("target_cond", DType::F32, &[b(), s(64), s(target_dim)]);
        let has_target = g.input("has_target", DType::Bool, &[b(), s(64)]);
        let external = g.input("external", DType::F32, &[b(), s(64), s(external_dim)]);
        let recon = vq.emit(&mut g, decoder_scope(spec), &tokens, &token_mask, &target_cond, &has_target, &external).unwrap();
        assert_eq!((g.dtype(&recon), g.shape(&recon)), (DType::F32, &[b(), s(64), s(out_dim)][..]));

        // Every node input must resolve (the structural half of validate_output).
        let mut known: std::collections::HashSet<&str> =
            g.inputs.iter().map(|v| v.name.as_str()).chain(g.initializers.iter().map(|t| t.name.as_str())).collect();
        for node in &g.nodes {
            for name in &node.inputs {
                assert!(name.is_empty() || known.contains(name.as_str()), "{}: unknown input {name}", node.name);
            }
            known.extend(node.outputs.iter().map(String::as_str));
        }
        // Initializer names must be unique.
        let mut names = std::collections::HashSet::new();
        for t in &g.initializers {
            assert!(names.insert(&t.name), "duplicate initializer {}", t.name);
        }
        // The FP16 scopes must cover the transformers and the decoders.
        for prefix in spec.fp16_prefixes {
            assert!(g.nodes.iter().any(|n| n.name.starts_with(prefix)), "no nodes under {prefix}");
        }
    }

    #[test]
    fn emit_soma() {
        smoke(&config::SOMA);
    }

    #[test]
    fn emit_g1() {
        smoke(&config::G1);
    }
}
