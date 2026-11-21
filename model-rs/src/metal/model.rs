// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! MotionBricks (SOMA and Unitree G1) on the model-specific Metal runtime.
//!
//! One generation is a fixed list of dispatches recorded into a single serial command buffer:
//! rig/glue kernels from `model.metal`, the generalized `gemm_mma` (implicit dilated convs with
//! upsampling, concatenated inputs, masked/predicated/blended epilogues, INT8 weights),
//! fused attention and add+LayerNorm. Network tensors use one element type (FP32, FP16 or BF16);
//! INT8 mode quantizes the transformer projections like the Python port. All buffers are allocated
//! and every GEMM pipeline the model can use is compiled at load time.
//! The release configuration (one MaskGIT step) makes the pose network's token embedding a
//! constant, which is computed at load time.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};

use super::transformer::{ACT_LEAKY, ACT_RELU, GemmCall, GemmParams, Kernels, PARTIAL_MAX, Scratch, Transformer};
use super::weights::{Conv, Decoder, FcBlock, Lin, Loader, Rig, rig, token_const};
use super::{Arg, Buf, Gpu, Pipeline, Pod, Precision, Recorder, as_bytes};
use crate::config::{self, FinalRootMode, Kind, ModelSpec, get_spec};
use crate::npy::{Npy, NpyData};
use crate::representation::MotionRepresentation;
use crate::weights::read_checkpoints;

/// Dispatch counts after which a generation's recording is committed (see `MetalModel::recorder`).
/// They don't depend on the GPU: encoding a dispatch takes well under a microsecond of CPU time,
/// while every dispatch of the model runs for microseconds on the GPU even on the largest Apple
/// GPUs (small launches are bound by latency and memory, not by core count). So after the first
/// small chunk, the GPU never waits for the CPU, whatever its core count.
pub const FLUSH_AT: [usize; 3] = [8, 40, 120];
/// Largest `max_batch` (the kernels index buffers with 32-bit integers).
pub const MAX_BATCH: usize = 16384;

/// Root network embedding width.
const E: usize = 512;
/// Root input embedding: local pose (256) | local root (64) | global root (64).
const X0: [usize; 3] = [256, 64, 64];
const X0_W: usize = 384;
/// Pose network width, its local-pose embedding per frame and its length embedding width.
const POSE_E: usize = 1024;
const POSE_FRAME: usize = 160;
const LEN_W: usize = 128;
/// Decoder channels.
const DEC_W: usize = 512;
/// Transformer scratch: tokens per sequence, width and FFN width it holds.
const TF_LEN: usize = 25;
const TF_D: usize = 1024;
const TF_FFN: usize = 2048;

/// The output of one call: main output (`pred_global_motions` or `pred_qpos`), durations, and
/// G1 foot contacts.
pub type Outputs = (Vec<f32>, Vec<i32>, Option<Vec<f32>>);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dims {
    pub b: i32,
    pub pd: i32,
    pub ld: i32,
    pub gd: i32,
    pub e: i32,
    pub h: i32,
    pub codes: i32,
    pub cw: i32,
    pub len_w: i32,
    pub forced: i32,
}

// SAFETY: all fields are i32 (no padding, no pointers).
unsafe impl Pod for Dims {}

/// Every intermediate of one generation, sized for `max_batch`.
pub struct Bufs {
    // inputs
    pub ctx: Buf,
    pub tgt: Buf,
    pub gr_in: Buf,
    pub lr_in: Buf,
    pub lp_in: Buf,
    pub hg: Buf,
    pub hl: Buf,
    pub hp: Buf,
    pub allowed: Buf,
    /// Forced durations (int32).
    pub nt_in: Buf,
    /// SOMA: `lp_in` retargeted to the standard T-pose (the inputs stay untouched).
    lp_rt: Buf,
    // normalized inputs
    pub gn: Buf,
    pub ln: Buf,
    pub pn: Buf,
    pub gn_t: Buf,
    pub ln_t: Buf,
    pub pn_t: Buf,
    pub info: Buf,
    // root network
    x0: Buf,
    fa: Buf,
    fb: Buf,
    s: Buf,
    e: Buf,
    fe: Buf,
    pub x1: Buf,
    log1: Buf,
    pub x2: Buf,
    pub pad2: Buf,
    /// Chosen durations (int32).
    pub nt: Buf,
    pub m16: Buf,
    pub m32: Buf,
    pub m64: Buf,
    pub ppad: Buf,
    pub logits: Buf,
    dec_in: Buf,
    pub dframe: Buf,
    pub droot: Buf,
    pub dhas: Buf,
    pub pr: Buf,
    // decoder scratch
    h: Buf,
    h2: Buf,
    t1: Buf,
    // pose network
    pub ext: Buf,
    pub rcond: Buf,
    pub pcond: Buf,
    pub hpc: Buf,
    pub le: Buf,
    re: Buf,
    pe: Buf,
    px: Buf,
    pub plog: Buf,
    /// Pose tokens (int32).
    pub tokens: Buf,
    pub lat: Buf,
    // outputs
    pub recon: Buf,
    pub out: Buf,
    pub qpos: Buf,
    pub contacts: Buf,
    tf: Scratch,
}

/// The glue and rig kernels of `model.metal`.
struct Glue {
    prep: Pipeline,
    soma_retarget: Pipeline,
    g1_preprocess: Pipeline,
    root_assemble: Pipeline,
    root_second_input: Pipeline,
    root_dense_cond: Pipeline,
    copy_rows: Pipeline,
    root_to_pose: Pipeline,
    argmax_dequant: Pipeline,
    finish_root: Pipeline,
    body_out: Pipeline,
    g1_postprocess: Pipeline,
}

impl Glue {
    fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Glue {
            prep: gpu.pipeline("prep")?,
            soma_retarget: gpu.pipeline("soma_retarget")?,
            g1_preprocess: gpu.pipeline("g1_preprocess")?,
            root_assemble: gpu.pipeline("root_assemble")?,
            root_second_input: gpu.pipeline("root_second_input")?,
            root_dense_cond: gpu.pipeline("root_dense_cond")?,
            copy_rows: gpu.pipeline("copy_rows")?,
            root_to_pose: gpu.pipeline("root_to_pose")?,
            argmax_dequant: gpu.pipeline("argmax_dequant")?,
            finish_root: gpu.pipeline("finish_root")?,
            body_out: gpu.pipeline("body_out")?,
            g1_postprocess: gpu.pipeline("g1_postprocess")?,
        })
    }
}

/// Most ONNX inputs of a model (SOMA has 7).
const MAX_INPUTS: usize = 8;

/// Input buffer of an ONNX input.
#[derive(Clone, Copy)]
enum Slot {
    Ctx,
    Tgt,
    GrIn,
    LrIn,
    LpIn,
    Hg,
    Hl,
    Hp,
    Allowed,
}

/// One ONNX input: its buffer, whether it is a mask (`!= 0` -> 0/1) and its shape after the batch axis.
#[derive(Clone, Copy)]
struct Input {
    name: &'static str,
    slot: Slot,
    mask: bool,
    row: [i32; 2],
    row_dims: usize,
}

/// Options of a [`MetalModel::conv`] launch (input mask/ReLU, activation, output mask, 2x
/// upsampling, in-place residual).
#[derive(Clone, Copy, Default)]
struct ConvOpts<'a> {
    amask: Option<&'a Buf>,
    a_relu: bool,
    act: i32,
    ymask: Option<&'a Buf>,
    up: bool,
    res: Option<&'a Buf>,
}

/// Retarget launch parameters (`RetargetParams` in model.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct RetargetParams {
    rows: i32,
    ld: i32,
    col0: i32,
    joints: i32,
    copy_pos: i32,
}

// SAFETY: all fields are i32 (no padding, no pointers).
unsafe impl Pod for RetargetParams {}

pub struct MetalModel {
    pub gpu: Gpu,
    pub k: Kernels,
    pub spec: ModelSpec,
    pub max_batch: usize,
    /// Network element type, and whether the transformer projections are INT8.
    pub prec: Precision,
    pub int8: bool,
    pd: usize,
    ld: usize,
    gd: usize,
    // root network
    proj_local_pose: Lin,
    proj_local_root: Lin,
    proj_global_root: Lin,
    no_local_pose: Buf,
    no_local_root: Buf,
    no_global_root: Buf,
    no_frame: Buf,
    fc_start: FcBlock,
    fc_end: FcBlock,
    input_pos: Buf,
    query_emb: Buf,
    root_pos: Buf,
    logit_head: Lin,
    middle_emb: Buf,
    shared: Transformer,
    root_token: Transformer,
    root_dec: Decoder,
    // pose network
    p_root: Lin,
    p_local_pose: Lin,
    len_emb: Buf,
    p_input: Lin,
    pose_pos: Buf,
    pose_tf: Transformer,
    p_logits: Lin,
    token_const: Buf,
    // VQ-VAE
    codebook: Buf,
    vq_dec: Decoder,
    // rig and statistics
    rig: Rig,
    stats: Buf,
    body_stats: Buf,
    pub b: Bufs,
    glue: Glue,
    inputs: Vec<Input>,
}

impl MetalModel {
    /// FP32 model.
    pub fn from_checkpoints(name: &str, dir: impl AsRef<Path>, max_batch: usize) -> Result<Self> {
        Self::from_checkpoints_with(name, dir, max_batch, config::Precision::Float32)
    }

    /// `precision`: a [`config::Precision`] or its name: "float32", "float16", "bfloat16", or
    /// "int8" (FP32 activations with INT8 transformer projections, as
    /// `motionbricks_mlx.MotionBricks.quantize(8)`).
    pub fn from_checkpoints_with<P>(name: &str, dir: impl AsRef<Path>, max_batch: usize, precision: P) -> Result<Self>
    where
        P: TryInto<config::Precision>,
        P::Error: Into<anyhow::Error>,
    {
        let precision = precision.try_into().map_err(Into::into)?;
        let spec = get_spec(name)?;
        let loaded = read_checkpoints(dir, &spec)?;
        Self::from_loaded(spec, loaded, max_batch, precision)
    }

    /// From one converted file (`motionbricks_rs::convert` or `motionbricks_mlx.convert`).
    pub fn from_converted<P>(path: impl AsRef<Path>, max_batch: usize, precision: P) -> Result<Self>
    where
        P: TryInto<config::Precision>,
        P::Error: Into<anyhow::Error>,
    {
        let precision = precision.try_into().map_err(Into::into)?;
        let (name, loaded) = crate::weights::read_converted(path)?;
        Self::from_loaded(get_spec(&name)?, loaded, max_batch, precision)
    }

    fn from_loaded(spec: ModelSpec, loaded: crate::weights::Loaded, max_batch: usize, precision: config::Precision) -> Result<Self> {
        ensure!((1..=MAX_BATCH).contains(&max_batch), "max_batch must be in 1..={MAX_BATCH}, got {max_batch}");
        let prec = Precision::from(precision);
        let int8 = precision.is_quantized();
        let rep = MotionRepresentation::new(spec.num_joints, &loaded.mean, &loaded.std)?;
        let gpu = Gpu::with_precision(prec)?;
        // Split-K partials are sized after the warm-up, from the launches the model makes.
        let k = Kernels::with_partial_len(&gpu, 0)?;
        let l = Loader { gpu: &gpu, t: &loaded.tensors };
        let pd = 1 + spec.local_poses_dim() as usize;
        let ld = rep.local.dim as usize;
        let gd = rep.global.dim as usize;
        let vq = spec.vq;
        let (heads, codes, cw) = (vq.num_heads as usize, vq.codes_per_head as usize, (vq.code_dim / vq.num_heads) as usize);
        let token_const = token_const(&l, prec, heads, codes, cw)?;
        let rig = rig(&gpu, &spec)?;
        let [g, lr, lp, lb] = rep.stats_f32()?;
        let stats = gpu.upload(&[g.0, g.1, lr.0, lr.1, lp.0, lp.1].concat());
        let body_stats = gpu.upload(&[lb.0, lb.1].concat());
        let sp = spec.root;
        let pp = spec.pose;
        let inputs = Self::input_layout(&spec)?;
        let mut model = MetalModel {
            proj_local_pose: l.lin("root.proj_local_pose")?,
            proj_local_root: l.lin("root.proj_local_root_value")?,
            proj_global_root: l.lin("root.proj_global_root_value")?,
            no_local_pose: l.buf("root.no_local_pose_emb")?,
            no_local_root: l.buf("root.no_local_root_emb")?,
            no_global_root: l.buf("root.no_global_root_emb")?,
            no_frame: l.buf("root.conv_no_frame_emb")?,
            fc_start: l.fc("root.proj_start_input")?,
            fc_end: l.fc("root.proj_end_input")?,
            input_pos: l.buf("root.input_position_emb.weight")?,
            query_emb: l.buf("root.proj_input_num_tokens.weight")?,
            root_pos: l.pos_table("root.position_emb.embed", 16)?,
            logit_head: l.lin("root.proj_num_token_output_logit")?,
            middle_emb: l.buf("root.middle_token_emb.weight")?,
            shared: Transformer::load(&gpu, &loaded.tensors, "root.shared_transformer_model", sp.n_layers_shared, sp.n_head as usize, int8)?,
            root_token: Transformer::load(&gpu, &loaded.tensors, "root.root_token_transformer_model", sp.n_layers_root_token, sp.n_head as usize, int8)?,
            root_dec: l.decoder("root.conv_output")?,
            p_root: l.lin("pose.proj_local_root_values")?,
            p_local_pose: l.lin("pose.proj_local_pose")?,
            len_emb: l.buf("pose.proj_num_valid_positions.weight")?,
            p_input: l.lin("pose.proj_input.0")?,
            pose_pos: l.pos_table("pose.position_emb.embed", 16)?,
            pose_tf: Transformer::load(&gpu, &loaded.tensors, "pose.transformer_model", pp.n_layers, pp.n_head as usize, int8)?,
            p_logits: l.lin("pose.proj_pose_output_logit")?,
            token_const: gpu.upload_t(&token_const),
            codebook: l.buf("vqvae.quantizer.vq.codebook.embed")?,
            vq_dec: l.decoder("vqvae.decoder")?,
            rig,
            stats,
            body_stats,
            b: Self::alloc(&gpu, max_batch, pd, ld, gd, heads, codes, cw, spec.kind == Kind::Soma),
            glue: Glue::new(&gpu)?,
            inputs,
            k,
            spec,
            max_batch,
            prec,
            int8,
            pd,
            ld,
            gd,
            gpu,
        };
        model.validate().with_context(|| format!("{} checkpoint does not match the Metal runtime's layout", model.spec.name))?;
        model.warm_up()?;
        Ok(model)
    }

    /// Check every weight and table against the buffer layout and kernels (shapes the encoders
    /// and `model.metal` assume), so a mismatching checkpoint fails here instead of reading or
    /// writing out of bounds on the GPU.
    fn validate(&self) -> Result<()> {
        let lin = |name: &str, l: &Lin, n: usize, k: usize| -> Result<()> {
            ensure!(l.n == n && l.k == k, "{name}: {}x{} weight, expected {n}x{k}", l.n, l.k);
            Ok(())
        };
        let len = |name: &str, b: &Buf, want: usize| -> Result<()> {
            ensure!(b.len() == want, "{name}: {} values, expected {want}", b.len());
            Ok(())
        };
        let at_least = |name: &str, b: &Buf, want: usize| -> Result<()> {
            ensure!(b.len() >= want, "{name}: {} values, need {want}", b.len());
            Ok(())
        };
        let (pd, ld, gd) = (self.pd, self.ld, self.gd);
        let vq = self.spec.vq;
        let (heads, codes, cw) = (vq.num_heads as usize, vq.codes_per_head as usize, (vq.code_dim / vq.num_heads) as usize);
        let choices = config::NUM_TOKEN_CHOICES as usize;
        ensure!(ld > 4 && gd == ld + 1, "layouts: local {ld}, global {gd} (expected global = local + 1)");
        // Root network: inputs -> x0 (B*8, 384), FC blocks -> (B*8, 512).
        lin("root.proj_local_pose", &self.proj_local_pose, X0[0], pd)?;
        lin("root.proj_local_root_value", &self.proj_local_root, X0[1], 4)?;
        lin("root.proj_global_root_value", &self.proj_global_root, X0[2], 5)?;
        len("root.no_local_pose_emb", &self.no_local_pose, X0[0])?;
        len("root.no_local_root_emb", &self.no_local_root, X0[1])?;
        len("root.no_global_root_emb", &self.no_global_root, X0[2])?;
        len("root.conv_no_frame_emb", &self.no_frame, E)?;
        for (name, f) in [("root.proj_start_input", &self.fc_start), ("root.proj_end_input", &self.fc_end)] {
            let [a, b] = [&f.layers[0], &f.layers[1]];
            ensure!(a.k == X0_W && b.k == a.n && f.proj.k == b.n, "{name}: layer shapes do not chain from {X0_W}");
            ensure!(a.n <= E && b.n <= E && f.proj.n == E, "{name}: layers wider than {E} or projection not {E}");
        }
        at_least("root.input_position_emb.weight", &self.input_pos, 8 * E)?;
        at_least("root.proj_input_num_tokens.weight", &self.query_emb, (config::MASKED_NUM_TOKENS - config::MIN_TOKENS + 1) as usize * E)?;
        len("root.position_emb.embed", &self.root_pos, 16 * E)?;
        lin("root.proj_num_token_output_logit", &self.logit_head, 12, E)?;
        at_least("root.middle_token_emb.weight", &self.middle_emb, choices * E)?;
        // Transformers: scratch holds TF_LEN tokens of width TF_D and FFN TF_FFN; attention kernels.
        for (name, t, d, l) in [("root.shared_transformer_model", &self.shared, E, 9), ("root.root_token_transformer_model", &self.root_token, E, 25), ("pose.transformer_model", &self.pose_tf, POSE_E, 16)] {
            ensure!(t.d == d && t.ffn <= TF_FFN && l <= TF_LEN, "{name}: width {} / FFN {} (expected width {d}, FFN <= {TF_FFN})", t.d, t.ffn);
            ensure!(Kernels::attention_supported(l, t.d / t.heads), "{name}: no attention kernel for {l} tokens of head dim {}", t.d / t.heads);
        }
        self.root_dec.check("root.conv_output", E, 5, E, 5, DEC_W)?;
        // Pose network.
        lin("pose.proj_local_root_values", &self.p_root, 256, 16)?;
        lin("pose.proj_local_pose", &self.p_local_pose, POSE_FRAME, pd)?;
        len("pose token embedding", &self.token_const, 4 * POSE_FRAME)?;
        at_least("pose.proj_num_valid_positions.weight", &self.len_emb, choices * LEN_W)?;
        lin("pose.proj_input.0", &self.p_input, POSE_E, 4 * POSE_FRAME + 256 + LEN_W)?;
        len("pose.position_emb.embed", &self.pose_pos, 16 * POSE_E)?;
        lin("pose.proj_pose_output_logit", &self.p_logits, heads * codes, POSE_E)?;
        len("vqvae.quantizer.vq.codebook.embed", &self.codebook, heads * codes * cw)?;
        self.vq_dec.check("vqvae.decoder", heads * cw, pd, 2, ld, DEC_W)?;
        // Statistics (prep / root_to_pose / finish_root / body_out).
        len("statistics", &self.stats, 18 + 2 * pd)?;
        len("body statistics", &self.body_stats, 2 * (ld - 4))?;
        ensure!(self.b.tf.fits(self.max_batch * TF_LEN, TF_D, TF_FFN), "transformer scratch too small");
        Ok(())
    }

    /// Compile every GEMM pipeline variant the plans can pick for batch 1..=max_batch (by
    /// recording each batch's generation and discarding it unrun), then size the split-K partial
    /// buffer: `max_splits * max(M * N)` over those launches, capped at [`PARTIAL_MAX`]. Every
    /// plan therefore resolves exactly as with a `PARTIAL_MAX` buffer.
    fn warm_up(&mut self) -> Result<()> {
        self.k.begin_warm_up();
        *self.k.log.borrow_mut() = Some(Vec::new());
        for b in 1..=self.max_batch {
            let mut rec = Recorder::new(&self.gpu);
            self.encode(&mut rec, &self.dims(b, false));
            rec.discard();
        }
        let log = self.k.log.borrow_mut().take().unwrap_or_default();
        self.k.end_warm_up().context("compiling GEMM pipelines")?;
        let max_mn = log.iter().map(|(p, _)| p.m as usize * p.n as usize).max().unwrap_or(0);
        self.k.set_partial_len(PARTIAL_MAX.min(self.k.max_splits() * max_mn));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn alloc(gpu: &Gpu, b: usize, pd: usize, ld: usize, gd: usize, heads: usize, codes: usize, cw: usize, soma: bool) -> Bufs {
        let f = |n: usize| gpu.buffer(n); // float32 (inputs, masks, rig, outputs)
        let i = |n: usize| gpu.buffer_i32(n); // int32 (durations, tokens)
        let t = |n: usize| gpu.buffer_t(n); // network element type
        Bufs {
            ctx: f(b * 4 * 36),
            tgt: f(b * 4 * 36),
            gr_in: f(b * 8 * 5),
            lr_in: f(b * 8 * 4),
            lp_in: f(b * 8 * (pd - 1)),
            hg: f(b * 8),
            hl: f(b * 8),
            hp: f(b * 8),
            allowed: f(b * 11),
            nt_in: i(b),
            lp_rt: f(if soma { b * 8 * (pd - 1) } else { 0 }),
            gn: f(b * 8 * 5),
            ln: f(b * 8 * 4),
            pn: f(b * 8 * pd),
            gn_t: t(b * 8 * 5),
            ln_t: t(b * 8 * 4),
            pn_t: t(b * 8 * pd),
            info: f(b * 4),
            x0: t(b * 8 * X0_W),
            fa: t(b * 8 * E),
            fb: t(b * 8 * E),
            s: t(b * 8 * E),
            e: t(b * 8 * E),
            fe: t(b * 8 * E),
            x1: t(b * 9 * E),
            log1: t(b * 9 * 12),
            x2: t(b * 25 * E),
            pad2: f(b * 25),
            nt: i(b),
            m16: f(b * 16),
            m32: f(b * 32),
            m64: f(b * 64),
            ppad: f(b * 16),
            logits: f(b * 12),
            dec_in: t(b * 16 * E),
            dframe: t(b * 64 * E),
            droot: t(b * 64 * 5),
            dhas: f(b * 64),
            pr: t(b * 64 * 5),
            h: t(b * 64 * DEC_W),
            h2: t(b * 64 * DEC_W),
            t1: t(b * 64 * DEC_W),
            ext: t(b * 64 * 2),
            rcond: t(b * 64 * 4),
            pcond: t(b * 64 * pd),
            hpc: f(b * 64),
            le: t(b * 16 * LEN_W),
            re: t(b * 16 * 256),
            pe: t(b * 64 * POSE_FRAME),
            px: t(b * 16 * POSE_E),
            plog: t(b * 16 * heads * codes),
            tokens: i(b * 16 * heads),
            lat: t(b * 16 * heads * cw),
            recon: t(b * 64 * ld),
            out: f(b * 64 * gd),
            qpos: f(b * 64 * 36),
            contacts: f(b * 64 * 4),
            tf: Scratch::new(gpu, b * TF_LEN, TF_D, TF_FFN),
        }
    }

    /// The ONNX inputs of `spec` ([`ModelSpec::input_names`], masks per [`config::is_mask_input`])
    /// with their buffers and shapes.
    fn input_layout(spec: &ModelSpec) -> Result<Vec<Input>> {
        let frames = config::CONTEXT_AND_TARGET;
        ensure!(spec.input_names().len() <= MAX_INPUTS, "{} inputs", spec.input_names().len());
        spec.input_names()
            .iter()
            .map(|&name| {
                let (slot, row, row_dims) = match name {
                    "context_qpos" => (Slot::Ctx, [4, config::G1_QPOS_DIM], 2),
                    "target_qpos" => (Slot::Tgt, [4, config::G1_QPOS_DIM], 2),
                    "global_root_values" => (Slot::GrIn, [frames, 5], 2),
                    "local_root_values" => (Slot::LrIn, [frames, 4], 2),
                    "local_poses" => (Slot::LpIn, [frames, spec.local_poses_dim()], 2),
                    "has_global_root_values" => (Slot::Hg, [frames, 0], 1),
                    "has_local_root_values" => (Slot::Hl, [frames, 0], 1),
                    "has_local_poses" => (Slot::Hp, [frames, 0], 1),
                    "allowed_pred_num_tokens" => (Slot::Allowed, [config::NUM_TOKEN_CHOICES, 0], 1),
                    other => bail!("the Metal runtime has no buffer for input {other}"),
                };
                Ok(Input { name, slot, mask: config::is_mask_input(name), row, row_dims })
            })
            .collect()
    }

    /// Kernel dimensions of a generation of `b` requests (1..=max_batch; panics otherwise, since
    /// the buffers hold `max_batch` requests).
    pub fn dims(&self, b: usize, forced: bool) -> Dims {
        assert!((1..=self.max_batch).contains(&b), "batch {b} outside 1..={}", self.max_batch);
        let vq = self.spec.vq;
        Dims {
            b: b as i32,
            pd: self.pd as i32,
            ld: self.ld as i32,
            gd: self.gd as i32,
            e: E as i32,
            h: vq.num_heads,
            codes: vq.codes_per_head,
            cw: vq.code_dim / vq.num_heads,
            len_w: LEN_W as i32,
            forced: forced as i32,
        }
    }

    fn launch(&self, rec: &mut Recorder, pso: &Pipeline, args: &[Arg], threads: usize) {
        rec.dispatch(pso, args, (threads.div_ceil(256), 1, 1), (256, 1, 1));
    }

    // -- stages ----------------------------------------------------------------------------------

    /// Rig preprocessing (G1 forward kinematics / SOMA capture->standard) and normalization.
    /// Leaves the input buffers unchanged, so a generation can be encoded again.
    pub fn encode_input(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let lp = match self.spec.kind {
            Kind::UnitreeG1 => {
                let args = [Arg::Buf(&b.ctx, 0), Arg::Buf(&b.tgt, 0), Arg::Buf(&self.rig.f, 0), Arg::Buf(&self.rig.i, 0), Arg::Buf(&b.gr_in, 0), Arg::Buf(&b.lr_in, 0), Arg::Buf(&b.lp_in, 0), Arg::Bytes(as_bytes(d))];
                rec.dispatch(&self.glue.g1_preprocess, &args, (n.div_ceil(32), 1, 1), (32, 1, 1));
                &b.lp_in
            }
            Kind::Soma => {
                let j = self.spec.num_joints;
                let p = RetargetParams { rows: (n * 8) as i32, ld: (self.pd - 1) as i32, col0: (j - 1) * 3, joints: j, copy_pos: 1 };
                let args = [Arg::Buf(&b.lp_in, 0), Arg::Buf(&self.rig.soma_capture, 0), Arg::Buf(&self.rig.soma_standard, 0), Arg::Bytes(as_bytes(&p)), Arg::Buf(&b.lp_rt, 0)];
                self.launch(rec, &self.glue.soma_retarget, &args, n * 8 * j as usize);
                &b.lp_rt
            }
        };
        let args = [
            Arg::Buf(&b.gr_in, 0), Arg::Buf(&b.lr_in, 0), Arg::Buf(lp, 0), Arg::Buf(&self.stats, 0), Arg::Buf(&b.gn, 0),
            Arg::Buf(&b.ln, 0), Arg::Buf(&b.pn, 0), Arg::Buf(&b.info, 0), Arg::Bytes(as_bytes(d)), Arg::Buf(&b.gn_t, 0),
            Arg::Buf(&b.ln_t, 0), Arg::Buf(&b.pn_t, 0),
        ];
        self.launch(rec, &self.glue.prep, &args, n * 8);
    }

    fn fcblock(&self, rec: &mut Recorder, f: &FcBlock, x: &Buf, rows: usize, out: &Buf) {
        let b = &self.b;
        let mut p = GemmParams::new(rows, f.layers[0].n, f.layers[0].k);
        p.act = ACT_LEAKY;
        self.k.run(rec, &GemmCall::new(x, &f.layers[0].w, &f.layers[0].b, &b.fa, p));
        let mut p = GemmParams::new(rows, f.layers[1].n, f.layers[1].k);
        p.act = ACT_LEAKY;
        self.k.run(rec, &GemmCall::new(&b.fa, &f.layers[1].w, &f.layers[1].b, &b.fb, p));
        self.k.run(rec, &GemmCall::new(&b.fb, &f.proj.w, &f.proj.b, out, GemmParams::new(rows, f.proj.n, f.proj.k)));
    }

    /// Blend GEMM into a column block of `y` (`v * has + missing * (1 - has)`).
    #[allow(clippy::too_many_arguments)]
    fn blend_linear(&self, rec: &mut Recorder, l: &Lin, x: &Buf, rows: usize, y: &Buf, col: usize, ldy: usize, has: &Buf, missing: &Buf, modulo: usize) {
        let mut p = GemmParams::new(rows, l.n, l.k);
        p.ldy = ldy as i32;
        p.blend = 1;
        p.blend_mod = modulo as i32;
        let mut g = GemmCall::new(x, &l.w, &l.b, y, p);
        g.y = (y, col);
        g.ymask = Some(has);
        g.missing = Some(missing);
        self.k.run(rec, &g);
    }

    /// Conv1d over `n_batch` sequences of `t` frames (at the output resolution).
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, rec: &mut Recorder, c: &Conv, x: &Buf, t: usize, n_batch: usize, y: &Buf, o: ConvOpts) {
        let rows = n_batch * t;
        let mut p = GemmParams::new(rows, c.n, c.taps * c.cin);
        p.lda = c.cin as i32;
        p.act = o.act;
        p.a_relu = o.a_relu as i32;
        p.has_mask = o.amask.is_some() as i32;
        p.y_mask = o.ymask.is_some() as i32;
        if c.taps > 1 {
            p.conv_taps = c.taps as i32;
            p.conv_t = t as i32;
            p.conv_cin = c.cin as i32;
            p.conv_dil = c.dil as i32;
            p.conv_up = if o.up { 2 } else { 1 };
        }
        if o.res.is_some() {
            p.has_res = 1;
            p.ldr = c.n as i32;
        }
        let mut g = GemmCall::new(x, &c.w, &c.b, y, p);
        g.amask = o.amask;
        g.ymask = o.ymask;
        g.res = o.res.map(|r| (r, 0));
        self.k.run(rec, &g);
    }

    /// `DoubleCondDecoder` (layers.py) for `n` sequences: x (n*16, cin) -> out (n*64, cout).
    #[allow(clippy::too_many_arguments)]
    fn decoder(&self, rec: &mut Recorder, dec: &Decoder, n: usize, x: &Buf, ext: &Buf, ext_dim: usize, tgt: &Buf, has_tgt: &Buf, out: &Buf) {
        let b = &self.b;
        let masks = [&b.m16, &b.m32, &b.m64];
        self.conv(rec, &dec.conv0, x, 16, n, &b.h, ConvOpts { amask: Some(&b.m16), act: ACT_RELU, ..Default::default() });
        for (i, lvl) in dec.levels.iter().enumerate() {
            let fpp = 4 >> i;
            let p_len = 16 << i;
            // Overwrite the frames that carry a target pose (predicated rows of h viewed as frames).
            let mut p = GemmParams::new(n * 64, lvl.target.n, lvl.target.k);
            p.act = ACT_RELU;
            p.y_pred = 1;
            let mut g = GemmCall::new(tgt, &lvl.target.w, &lvl.target.b, &b.h, p);
            g.ymask = Some(has_tgt);
            self.k.run(rec, &g);
            // h = relu(Linear([h, external.reshape(n, P, -1)]))
            let mut p = GemmParams::new(n * p_len, lvl.ext.n, lvl.ext.k);
            p.act = ACT_RELU;
            p.lda = DEC_W as i32;
            p.k1 = DEC_W as i32;
            p.k2 = (fpp * ext_dim) as i32;
            p.lda2 = (fpp * ext_dim) as i32;
            let mut g = GemmCall::new(&b.h, &lvl.ext.w, &lvl.ext.b, &b.h2, p);
            g.a2 = Some(ext);
            self.k.run(rec, &g);
            let m = masks[i];
            for (c1, c2) in &lvl.res {
                self.conv(rec, c1, &b.h2, p_len, n, &b.t1, ConvOpts { amask: Some(m), a_relu: true, ..Default::default() });
                self.conv(rec, c2, &b.t1, p_len, n, &b.h2, ConvOpts { amask: Some(m), a_relu: true, ymask: Some(m), res: Some(&b.h2), ..Default::default() });
            }
            self.conv(rec, &lvl.up, &b.h2, p_len * 2, n, &b.h, ConvOpts { amask: Some(masks[i + 1]), up: true, ..Default::default() });
        }
        self.conv(rec, &dec.post, &b.h, 64, n, &b.h2, ConvOpts { amask: Some(&b.m64), act: ACT_RELU, ymask: Some(&b.m64), ..Default::default() });
        self.conv(rec, &dec.out, &b.h2, 64, n, out, ConvOpts { ymask: Some(&b.m64), ..Default::default() });
    }

    /// Root network from gn/ln/pn + masks: durations (nt, token masks, logits) and the root path pr.
    pub fn encode_root(&self, rec: &mut Recorder, d: &Dims) {
        self.encode_root_a(rec, d);
        self.encode_root_b(rec, d);
    }

    /// First half of the root network: duration head, second-transformer input, dense conditions.
    pub fn encode_root_a(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let rows = n * 8;
        self.blend_linear(rec, &self.proj_local_pose, &b.pn_t, rows, &b.x0, 0, X0_W, &b.hp, &self.no_local_pose, 1);
        self.blend_linear(rec, &self.proj_local_root, &b.ln_t, rows, &b.x0, X0[0], X0_W, &b.hl, &self.no_local_root, 1);
        self.blend_linear(rec, &self.proj_global_root, &b.gn_t, rows, &b.x0, X0[0] + X0[1], X0_W, &b.hg, &self.no_global_root, 1);
        self.fcblock(rec, &self.fc_start, &b.x0, rows, &b.s);
        self.fcblock(rec, &self.fc_end, &b.x0, rows, &b.e);
        let args = [Arg::Buf(&b.s, 0), Arg::Buf(&b.e, 0), Arg::Buf(&self.input_pos, 0), Arg::Buf(&self.query_emb, 0), Arg::Buf(&b.nt_in, 0), Arg::Buf(&b.x1, 0), Arg::Buf(&b.fe, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, &self.glue.root_assemble, &args, n * 9 * E);
        self.shared.encode(&self.k, rec, &b.x1, None, n, 9, &b.tf);
        self.k.run(rec, &GemmCall::new(&b.x1, &self.logit_head.w, &self.logit_head.b, &b.log1, GemmParams::new(n * 9, 12, E)));
        let args = [
            Arg::Buf(&b.log1, 0), Arg::Buf(&b.allowed, 0), Arg::Buf(&b.nt_in, 0), Arg::Buf(&b.fe, 0), Arg::Buf(&self.middle_emb, 0),
            Arg::Buf(&self.root_pos, 0), Arg::Buf(&b.x2, 0), Arg::Buf(&b.pad2, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&b.m16, 0),
            Arg::Buf(&b.logits, 0), Arg::Bytes(as_bytes(d)), Arg::Buf(&b.m32, 0), Arg::Buf(&b.m64, 0), Arg::Buf(&b.ppad, 0),
            Arg::Buf(&self.input_pos, 0),
        ];
        self.launch(rec, &self.glue.root_second_input, &args, n * 25 * E);
        let args = [
            Arg::Buf(&b.fe, 0), Arg::Buf(&b.gn, 0), Arg::Buf(&b.hg, 0), Arg::Buf(&b.hl, 0), Arg::Buf(&b.hp, 0),
            Arg::Buf(&self.no_frame, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&b.dframe, 0), Arg::Buf(&b.droot, 0), Arg::Buf(&b.dhas, 0),
            Arg::Bytes(as_bytes(d)),
        ];
        self.launch(rec, &self.glue.root_dense_cond, &args, n * 64 * E);
    }

    /// Second half of the root network: root-token transformer and the root path decoder.
    pub fn encode_root_b(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        self.root_token.encode(&self.k, rec, &b.x2, Some(&b.pad2), n, 25, &b.tf);
        let (p, w) = ([n as i32, 25, 16, 9], E as i32);
        let args = [Arg::Buf(&b.x2, 0), Arg::Buf(&b.dec_in, 0), Arg::Bytes(as_bytes(&p)), Arg::Bytes(as_bytes(&w))];
        self.launch(rec, &self.glue.copy_rows, &args, n * 16 * E);
        self.decoder(rec, &self.root_dec, n, &b.dec_in, &b.dframe, E, &b.droot, &b.dhas, &b.pr);
    }

    /// Root path -> pose conditions (global_root_to_local, last-frame patch, dense pose conditions).
    pub fn encode_root_to_pose(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let args = [
            Arg::Buf(&b.pr, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&self.stats, 0), Arg::Buf(&b.ln, 0), Arg::Buf(&b.hl, 0),
            Arg::Buf(&b.pn, 0), Arg::Buf(&b.hp, 0), Arg::Buf(&self.len_emb, 0), Arg::Buf(&b.ext, 0), Arg::Buf(&b.rcond, 0),
            Arg::Buf(&b.pcond, 0), Arg::Buf(&b.hpc, 0), Arg::Buf(&b.le, 0), Arg::Bytes(as_bytes(d)),
        ];
        self.launch(rec, &self.glue.root_to_pose, &args, n * 64);
    }

    /// Pose network (single MaskGIT step) from rcond/pcond/hpc/le/ppad: logits, tokens, latents.
    pub fn encode_pose(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        self.k.run(rec, &GemmCall::new(&b.rcond, &self.p_root.w, &self.p_root.b, &b.re, GemmParams::new(n * 16, 256, 16)));
        self.blend_linear(rec, &self.p_local_pose, &b.pcond, n * 64, &b.pe, 0, POSE_FRAME, &b.hpc, &self.token_const, 4);
        let mut p = GemmParams::new(n * 16, POSE_E, POSE_E);
        p.act = ACT_RELU;
        p.lda = (4 * POSE_FRAME) as i32;
        p.k1 = (4 * POSE_FRAME) as i32;
        p.k2 = 256;
        p.lda2 = 256;
        p.lda3 = LEN_W as i32;
        p.has_res = 1;
        p.ldr = POSE_E as i32;
        p.r_mod = 16;
        let mut g = GemmCall::new(&b.pe, &self.p_input.w, &self.p_input.b, &b.px, p);
        g.a2 = Some(&b.re);
        g.a3 = Some(&b.le);
        g.res = Some((&self.pose_pos, 0));
        self.k.run(rec, &g);
        self.pose_tf.encode(&self.k, rec, &b.px, Some(&b.ppad), n, 16, &b.tf);
        let nl = self.p_logits.n;
        self.k.run(rec, &GemmCall::new(&b.px, &self.p_logits.w, &self.p_logits.b, &b.plog, GemmParams::new(n * 16, nl, POSE_E)));
        let args = [Arg::Buf(&b.plog, 0), Arg::Buf(&self.codebook, 0), Arg::Buf(&b.tokens, 0), Arg::Buf(&b.lat, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, &self.glue.argmax_dequant, &args, n * 16 * self.spec.vq.num_heads as usize);
    }

    /// VQ-VAE decoder from lat/ext/pcond/hpc + token masks -> recon (normalized local features).
    pub fn encode_vq(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        self.decoder(rec, &self.vq_dec, n, &b.lat, &b.ext, 2, &b.pcond, &b.hpc, &b.recon);
    }

    /// recon -> global features (root integration, final root mode, re-anchoring) -> rig outputs.
    pub fn encode_finish(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let mode: i32 = (self.spec.final_root_mode == FinalRootMode::FromHeightConstraintViolation) as i32;
        let args = [
            Arg::Buf(&b.recon, 0), Arg::Buf(&b.pr, 0), Arg::Buf(&self.stats, 0), Arg::Buf(&b.gn, 0), Arg::Buf(&b.hg, 0),
            Arg::Buf(&b.nt, 0), Arg::Buf(&b.info, 0), Arg::Buf(&b.out, 0), Arg::Bytes(as_bytes(d)), Arg::Bytes(as_bytes(&mode)),
        ];
        rec.dispatch(&self.glue.finish_root, &args, (n.div_ceil(32), 1, 1), (32, 1, 1));
        let args = [Arg::Buf(&b.recon, 0), Arg::Buf(&self.body_stats, 0), Arg::Buf(&b.out, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, &self.glue.body_out, &args, n * 64 * (self.ld - 4));
        match self.spec.kind {
            Kind::Soma => {
                // In place on the output rows.
                let j = self.spec.num_joints;
                let p = RetargetParams { rows: (n * 64) as i32, ld: self.gd as i32, col0: 5 + (j - 1) * 3, joints: j, copy_pos: 0 };
                let args = [Arg::Buf(&b.out, 0), Arg::Buf(&self.rig.soma_standard, 0), Arg::Buf(&self.rig.soma_capture, 0), Arg::Bytes(as_bytes(&p)), Arg::Buf(&b.out, 0)];
                self.launch(rec, &self.glue.soma_retarget, &args, n * 64 * j as usize);
            }
            Kind::UnitreeG1 => {
                let args = [Arg::Buf(&b.out, 0), Arg::Buf(&self.rig.f, 0), Arg::Buf(&self.rig.i, 0), Arg::Buf(&b.qpos, 0), Arg::Buf(&b.contacts, 0), Arg::Bytes(as_bytes(d))];
                self.launch(rec, &self.glue.g1_postprocess, &args, n * 64);
            }
        }
    }

    /// The whole generation.
    pub fn encode(&self, rec: &mut Recorder, d: &Dims) {
        self.encode_input(rec, d);
        self.encode_root(rec, d);
        self.encode_root_to_pose(rec, d);
        self.encode_pose(rec, d);
        self.encode_vq(rec, d);
        self.encode_finish(rec, d);
    }

    // -- host I/O ---------------------------------------------------------------------------------

    pub fn input_names(&self) -> &'static [&'static str] {
        self.spec.input_names()
    }

    /// The pose network's length embedding as loaded (`T` values).
    pub fn len_emb(&self) -> &Buf {
        &self.len_emb
    }

    /// The VQ codebook as loaded (`T` values).
    pub fn codebook(&self) -> &Buf {
        &self.codebook
    }

    fn slot(&self, s: Slot) -> &Buf {
        let b = &self.b;
        match s {
            Slot::Ctx => &b.ctx,
            Slot::Tgt => &b.tgt,
            Slot::GrIn => &b.gr_in,
            Slot::LrIn => &b.lr_in,
            Slot::LpIn => &b.lp_in,
            Slot::Hg => &b.hg,
            Slot::Hl => &b.hl,
            Slot::Hp => &b.hp,
            Slot::Allowed => &b.allowed,
        }
    }

    /// Copy ONNX-named inputs into the input buffers; returns the batch size. Every input must
    /// have the ONNX shape with the same batch size (1..=max_batch); masks become `!= 0`.
    pub fn upload_inputs(&self, inputs: &HashMap<String, Npy>) -> Result<usize> {
        let mut found: [Option<&Npy>; MAX_INPUTS] = [None; MAX_INPUTS];
        let mut batch = None;
        for (slot, input) in found.iter_mut().zip(&self.inputs) {
            let v = inputs.get(input.name).ok_or_else(|| anyhow!("missing input {}", input.name))?;
            let rows = v.shape.first().copied().unwrap_or(-1);
            let n = *batch.get_or_insert(rows);
            let row = &input.row[..input.row_dims];
            if v.shape.len() != 1 + row.len() || rows != n || v.shape[1..] != *row {
                bail!("input {} has shape {:?}, expected [{n}, {}]", input.name, v.shape, row.iter().map(i32::to_string).collect::<Vec<_>>().join(", "));
            }
            v.check().with_context(|| format!("input {}", input.name))?;
            *slot = Some(v);
        }
        let n = batch.unwrap_or(0).max(0) as usize;
        if n == 0 || n > self.max_batch {
            bail!("batch {n} outside 1..={} (max_batch)", self.max_batch);
        }
        for (v, input) in found.iter().zip(&self.inputs) {
            write_input(self.slot(input.slot), v.expect("checked above"), input.mask);
        }
        Ok(n)
    }

    pub fn num_tokens(&self, n: usize) -> Vec<i32> {
        self.b.nt.to_i32(n)
    }

    /// Main output (`pred_global_motions` or `pred_qpos`), durations, and G1 foot contacts.
    pub fn outputs(&self, n: usize) -> Outputs {
        let b = &self.b;
        match self.spec.kind {
            Kind::Soma => (b.out.to_f32(n * 64 * self.gd), self.num_tokens(n), None),
            Kind::UnitreeG1 => (b.qpos.to_f32(n * 64 * 36), self.num_tokens(n), Some(b.contacts.to_f32(n * 64 * 4))),
        }
    }

    /// A recorder for one generation. It commits early (after a few dispatches, then in chunks)
    /// so the GPU runs the first stages while the CPU encodes the rest.
    pub fn recorder(&self) -> Recorder {
        Recorder::new(&self.gpu).with_flushes(&FLUSH_AT)
    }

    /// ONNX-Runtime-style call: upload, record (committing in chunks), wait, read back.
    pub fn run(&self, inputs: &HashMap<String, Npy>) -> Result<Outputs> {
        self.run_with_durations(inputs, None)
    }

    /// Like [`Self::run`]; `durations` (6..=16 tokens per request) replaces the durations the root
    /// network would choose (Python: `core.predict(..., num_tokens=...)`).
    pub fn run_with_durations(&self, inputs: &HashMap<String, Npy>, durations: Option<&[i32]>) -> Result<Outputs> {
        let n = self.upload_inputs(inputs)?;
        if let Some(t) = durations {
            if t.len() != n || t.iter().any(|v| !(config::MIN_TOKENS..=config::MAX_TOKENS).contains(v)) {
                bail!("durations must be {n} values in {}..={}, got {t:?}", config::MIN_TOKENS, config::MAX_TOKENS);
            }
            self.b.nt_in.write_i32(t);
        }
        let d = self.dims(n, durations.is_some());
        let mut rec = self.recorder();
        self.encode(&mut rec, &d);
        rec.commit().wait()?;
        Ok(self.outputs(n))
    }
}

/// Write an input array into a float32 buffer: its values, or for a mask `!= 0` (in the array's
/// own dtype, like [`Npy::to_bool`]) as 0/1. No intermediate vector.
fn write_input(dst: &Buf, v: &Npy, mask: bool) {
    let flag = |b: bool| b as i32 as f32;
    let n = v.data.len();
    match &v.data {
        NpyData::F32(d) if mask => dst.fill_f32(0, n, d.iter().map(|&x| flag(x != 0.0))),
        NpyData::F32(d) => dst.write_f32(d),
        NpyData::F64(d) if mask => dst.fill_f32(0, n, d.iter().map(|&x| flag(x != 0.0))),
        NpyData::F64(d) => dst.fill_f32(0, n, d.iter().map(|&x| x as f32)),
        NpyData::I32(d) if mask => dst.fill_f32(0, n, d.iter().map(|&x| flag(x != 0))),
        NpyData::I32(d) => dst.fill_f32(0, n, d.iter().map(|&x| x as f32)),
        NpyData::I64(d) if mask => dst.fill_f32(0, n, d.iter().map(|&x| flag(x != 0))),
        NpyData::I64(d) => dst.fill_f32(0, n, d.iter().map(|&x| x as f32)),
        NpyData::U8(d) if mask => dst.fill_f32(0, n, d.iter().map(|&x| flag(x != 0))),
        NpyData::U8(d) => dst.fill_f32(0, n, d.iter().map(|&x| x as f32)),
        NpyData::Bool(d) => dst.fill_f32(0, n, d.iter().map(|&x| flag(x))),
    }
}
