// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! MotionBricks (SOMA and Unitree G1) on the model-specific Metal runtime.
//!
//! One generation is a fixed list of dispatches recorded into a single serial command buffer:
//! rig/glue kernels from `model.metal`, the generalized `gemm_mma` (implicit dilated convs with
//! upsampling, concatenated inputs, masked/predicated/blended epilogues), MLX's own steel GEMM for
//! large plain matrix products, fused attention and add+LayerNorm. All buffers are allocated once.
//! The release configuration (one MaskGIT step) makes the pose network's token embedding a
//! constant, which is computed at load time.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use mlx_rs::Array;

use super::transformer::{ACT_LEAKY, ACT_RELU, GemmCall, GemmParams, Kernels, Scratch, Transformer, upload};
use super::{Arg, Buf, Gpu, Pipeline, Recorder, as_bytes, wait};
use crate::config::{FinalRootMode, Kind, ModelSpec, get_spec};
use crate::npy::{Npy, load_npz_bytes};
use crate::ops::to_vec_f32;
use crate::representation::MotionRepresentation;
use crate::weights::read_checkpoints;

const G1_RIG: &[u8] = include_bytes!("../../../model/motionbricks_mlx/assets/g1_model_rig.npz");
const SOMA_RIG: &[u8] = include_bytes!("../../../model/motionbricks_mlx/assets/soma_model_rig.npz");

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

struct Lin {
    w: Buf,
    b: Buf,
    n: usize,
    k: usize,
}

struct Conv {
    w: Buf,
    b: Buf,
    n: usize,
    cin: usize,
    taps: usize,
    dil: usize,
}

struct FcBlock {
    layers: Vec<Lin>,
    proj: Lin,
}

struct Level {
    target: Lin,
    ext: Lin,
    res: Vec<(Conv, Conv)>,
    up: Conv,
}

struct Decoder {
    conv0: Conv,
    levels: Vec<Level>,
    post: Conv,
    out: Conv,
}

struct Loader<'a> {
    gpu: &'a Gpu,
    t: &'a HashMap<String, Array>,
}

impl Loader<'_> {
    fn buf(&self, name: &str) -> Result<Buf> {
        upload(self.gpu, self.t, name)
    }
    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(to_vec_f32(self.t.get(name).ok_or_else(|| anyhow!("missing weight {name}"))?)?)
    }
    fn shape(&self, name: &str) -> Vec<usize> {
        self.t[name].shape().iter().map(|&d| d as usize).collect()
    }
    fn lin(&self, p: &str) -> Result<Lin> {
        let s = self.shape(&format!("{p}.weight"));
        Ok(Lin { w: self.buf(&format!("{p}.weight"))?, b: self.buf(&format!("{p}.bias"))?, n: s[0], k: s[1] })
    }
    fn conv(&self, p: &str, dil: usize) -> Result<Conv> {
        let s = self.shape(&format!("{p}.weight")); // (out, k, in), already channels-last
        Ok(Conv { w: self.buf(&format!("{p}.weight"))?, b: self.buf(&format!("{p}.bias"))?, n: s[0], taps: s[1], cin: s[2], dil })
    }
    fn fc(&self, p: &str) -> Result<FcBlock> {
        Ok(FcBlock { layers: vec![self.lin(&format!("{p}.fc_layers.0"))?, self.lin(&format!("{p}.fc_layers.1"))?], proj: self.lin(&format!("{p}.forward_projection"))? })
    }
    fn decoder(&self, p: &str) -> Result<Decoder> {
        let mut levels = Vec::new();
        for i in 0..2 {
            let res = (0..4)
                .map(|j| {
                    let dil = 3usize.pow(3 - j as u32);
                    Ok((self.conv(&format!("{p}.model.{}.0.model.{j}.conv1", 2 + i), dil)?, self.conv(&format!("{p}.model.{}.0.model.{j}.conv2", 2 + i), 1)?))
                })
                .collect::<Result<_>>()?;
            levels.push(Level {
                target: self.lin(&format!("{p}.target_cond_blocks.{}", 2 * i))?,
                ext: self.lin(&format!("{p}.external_cond_blocks.{}", 2 * i))?,
                res,
                up: self.conv(&format!("{p}.model.{}.2", 2 + i), 1)?,
            });
        }
        Ok(Decoder { conv0: self.conv(&format!("{p}.model.0"), 1)?, levels, post: self.conv(&format!("{p}.model.4"), 1)?, out: self.conv(&format!("{p}.model.6"), 1)? })
    }
}

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
    pub nt_in: Buf,
    // normalized inputs
    pub gn: Buf,
    pub ln: Buf,
    pub pn: Buf,
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
    pub nt: Buf,
    pub m16: Buf,
    pub m32: Buf,
    pub m64: Buf,
    pub ppad: Buf,
    pub logits: Buf,
    dec_in: Buf,
    dframe: Buf,
    droot: Buf,
    dhas: Buf,
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
    pub tokens: Buf,
    pub lat: Buf,
    // outputs
    pub recon: Buf,
    pub out: Buf,
    pub qpos: Buf,
    pub contacts: Buf,
    tf: Scratch,
}

pub struct MetalModel {
    pub gpu: Gpu,
    pub k: Kernels,
    pub spec: ModelSpec,
    pub max_batch: usize,
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
    pub len_emb_host: Vec<f32>,
    p_input: Lin,
    pose_pos: Buf,
    pose_tf: Transformer,
    p_logits: Lin,
    token_const: Buf,
    // VQ-VAE
    codebook: Buf,
    pub codebook_host: Vec<f32>,
    vq_dec: Decoder,
    // rig and statistics
    rig_f: Buf,
    rig_i: Buf,
    soma_capture: Buf,
    soma_standard: Buf,
    stats: Buf,
    body_stats: Buf,
    pub b: Bufs,
    pso: HashMap<&'static str, Pipeline>,
}

fn cpu_linear(x: &[f32], w: &[f32], b: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|j| b[j] + (0..k).map(|i| x[i] * w[j * k + i]).sum::<f32>()).collect()
}

fn leaky(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| (x * 0.01).max(x)).collect()
}

impl Bufs {
    pub fn dframe_debug(&self) -> &[f32] { self.dframe.as_slice() }
    pub fn droot_debug(&self) -> &[f32] { self.droot.as_slice() }
    pub fn dhas_debug(&self) -> &[f32] { self.dhas.as_slice() }
}

impl MetalModel {
    pub fn from_checkpoints(name: &str, dir: impl AsRef<Path>, max_batch: usize) -> Result<Self> {
        let spec = get_spec(name)?;
        let loaded = read_checkpoints(dir, &spec)?;
        let rep = MotionRepresentation::new(spec.num_joints, &loaded.mean, &loaded.std)?;
        let gpu = Gpu::new()?;
        let k = Kernels::new(&gpu)?;
        let l = Loader { gpu: &gpu, t: &loaded.tensors };
        let j = spec.num_joints as usize;
        let pd = 1 + (j - 1) * 3 + j * 6;
        let ld = rep.local.dim as usize;
        let gd = rep.global.dim as usize;
        let vq = spec.vq;
        let (heads, codes, cw) = (vq.num_heads as usize, vq.codes_per_head as usize, (vq.code_dim / vq.num_heads) as usize);

        // Constant token embedding of the release config (every token masked at the single step):
        // FCBlock(pose_token_emb[h * (codes + 1) + mask_id] for each head), split into 4 frames.
        let emb = l.vec("pose.pose_token_emb.weight")?;
        let tok: Vec<f32> = (0..heads).flat_map(|h| emb[(h * (codes + 1) + codes) * cw..(h * (codes + 1) + codes + 1) * cw].to_vec()).collect();
        let fc = |p: &str| -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
            let s = l.shape(&format!("{p}.weight"));
            Ok((l.vec(&format!("{p}.weight"))?, l.vec(&format!("{p}.bias"))?, s[0], s[1]))
        };
        let (w0, b0, n0, k0) = fc("pose.proj_pose_token_emb.fc_layers.0")?;
        let (w1, b1, n1, k1) = fc("pose.proj_pose_token_emb.fc_layers.1")?;
        let (w2, b2, n2, k2) = fc("pose.proj_pose_token_emb.forward_projection")?;
        let h0 = leaky(cpu_linear(&tok, &w0, &b0, n0, k0));
        let h1 = leaky(cpu_linear(&h0, &w1, &b1, n1, k1));
        let token_const = cpu_linear(&h1, &w2, &b2, n2, k2); // 640 = 4 frames x 160

        // Rig.
        let (rig_f, rig_i, soma_capture, soma_standard) = match spec.kind {
            Kind::UnitreeG1 => {
                let a = load_npz_bytes(G1_RIG)?;
                let mapping = a["mujoco_indices_to_motion_indices"].to_i32();
                let q2t = a["rot_offsets_q2t"].to_f32();
                let q2t_mapped: Vec<f32> = mapping.iter().flat_map(|&m| q2t[m as usize * 9..m as usize * 9 + 9].to_vec()).collect();
                let f: Vec<f32> = [
                    a["mujoco_to_motion_matrix"].to_f32(),
                    a["mujoco_joint_axis_values_mujoco_space"].to_f32(),
                    q2t_mapped,
                    a["capture_neutral_joints_mujoco"].to_f32(),
                    a["neutral_joints"].to_f32(),
                    a["rot_offsets_f2q"].to_f32(),
                    a["mujoco_joint_axis_values_motion_space"].to_f32(),
                ]
                .concat();
                let i: Vec<i32> = [
                    a["mujoco_joint_including_root_parent_list"].to_i32(),
                    a["joint_parents"].to_i32(),
                    a["motion_indices_to_mujoco_indices"].to_i32(),
                    mapping,
                ]
                .concat();
                let ib = gpu.buffer(i.len());
                ib.as_mut_slice().copy_from_slice(&i.iter().map(|&v| f32::from_bits(v as u32)).collect::<Vec<_>>());
                (gpu.upload(&f), ib, gpu.buffer(4), gpu.buffer(4))
            }
            Kind::Soma => {
                let a = load_npz_bytes(SOMA_RIG)?;
                (gpu.buffer(4), gpu.buffer(4), gpu.upload(&a["capture_offsets"].to_f32()), gpu.upload(&a["standard_offsets"].to_f32()))
            }
        };
        let [g, lr, lp, lb] = rep.stats_f32()?;
        let stats = gpu.upload(&[g.0, g.1, lr.0, lr.1, lp.0, lp.1].concat());
        let body_stats = gpu.upload(&[lb.0, lb.1].concat());

        let pos_table = |name: &str, n: usize| -> Result<Buf> {
            let v = l.vec(name)?; // (seq, 1, dim)
            Ok(gpu.upload(&v[..n * (v.len() / l.shape(name)[0])]))
        };
        let sp = spec.root;
        let pp = spec.pose;
        let model = MetalModel {
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
            root_pos: pos_table("root.position_emb.embed", 16)?,
            logit_head: l.lin("root.proj_num_token_output_logit")?,
            middle_emb: l.buf("root.middle_token_emb.weight")?,
            shared: Transformer::load(&gpu, &loaded.tensors, "root.shared_transformer_model", sp.n_layers_shared, sp.n_head as usize)?,
            root_token: Transformer::load(&gpu, &loaded.tensors, "root.root_token_transformer_model", sp.n_layers_root_token, sp.n_head as usize)?,
            root_dec: l.decoder("root.conv_output")?,
            p_root: l.lin("pose.proj_local_root_values")?,
            p_local_pose: l.lin("pose.proj_local_pose")?,
            len_emb: l.buf("pose.proj_num_valid_positions.weight")?,
            len_emb_host: l.vec("pose.proj_num_valid_positions.weight")?,
            p_input: l.lin("pose.proj_input.0")?,
            pose_pos: pos_table("pose.position_emb.embed", 16)?,
            pose_tf: Transformer::load(&gpu, &loaded.tensors, "pose.transformer_model", pp.n_layers, pp.n_head as usize)?,
            p_logits: l.lin("pose.proj_pose_output_logit")?,
            token_const: gpu.upload(&token_const),
            codebook: l.buf("vqvae.quantizer.vq.codebook.embed")?,
            codebook_host: l.vec("vqvae.quantizer.vq.codebook.embed")?,
            vq_dec: l.decoder("vqvae.decoder")?,
            rig_f,
            rig_i,
            soma_capture,
            soma_standard,
            stats,
            body_stats,
            b: Self::alloc(&gpu, max_batch, pd, ld, gd, heads, codes, cw),
            pso: [
                "prep", "soma_retarget", "g1_preprocess", "root_assemble", "root_second_input", "root_dense_cond", "copy_rows",
                "root_to_pose", "argmax_dequant", "finish_root", "body_out", "g1_postprocess",
            ]
            .into_iter()
            .map(|n| Ok((n, gpu.pipeline(n)?)))
            .collect::<Result<_>>()?,
            k,
            spec,
            max_batch,
            pd,
            ld,
            gd,
            gpu,
        };
        Ok(model)
    }

    #[allow(clippy::too_many_arguments)]
    fn alloc(gpu: &Gpu, b: usize, pd: usize, ld: usize, gd: usize, heads: usize, codes: usize, cw: usize) -> Bufs {
        let f = |n: usize| gpu.buffer(n);
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
            nt_in: f(b),
            gn: f(b * 8 * 5),
            ln: f(b * 8 * 4),
            pn: f(b * 8 * pd),
            info: f(b * 4),
            x0: f(b * 8 * 384),
            fa: f(b * 8 * 512),
            fb: f(b * 8 * 512),
            s: f(b * 8 * 512),
            e: f(b * 8 * 512),
            fe: f(b * 8 * 512),
            x1: f(b * 9 * 512),
            log1: f(b * 9 * 12),
            x2: f(b * 25 * 512),
            pad2: f(b * 25),
            nt: f(b),
            m16: f(b * 16),
            m32: f(b * 32),
            m64: f(b * 64),
            ppad: f(b * 16),
            logits: f(b * 12),
            dec_in: f(b * 16 * 512),
            dframe: f(b * 64 * 512),
            droot: f(b * 64 * 5),
            dhas: f(b * 64),
            pr: f(b * 64 * 5),
            h: f(b * 64 * 512),
            h2: f(b * 64 * 512),
            t1: f(b * 64 * 512),
            ext: f(b * 64 * 2),
            rcond: f(b * 64 * 4),
            pcond: f(b * 64 * pd),
            hpc: f(b * 64),
            le: f(b * 16 * 128),
            re: f(b * 16 * 256),
            pe: f(b * 64 * 160),
            px: f(b * 16 * 1024),
            plog: f(b * 16 * heads * codes),
            tokens: f(b * 16 * heads),
            lat: f(b * 16 * heads * cw),
            recon: f(b * 64 * ld),
            out: f(b * 64 * gd),
            qpos: f(b * 64 * 36),
            contacts: f(b * 64 * 4),
            tf: Scratch::new(gpu, b * 25, 1024, 2048),
        }
    }

    pub fn dims(&self, b: usize, forced: bool) -> Dims {
        let vq = self.spec.vq;
        Dims {
            b: b as i32,
            pd: self.pd as i32,
            ld: self.ld as i32,
            gd: self.gd as i32,
            e: 512,
            h: vq.num_heads,
            codes: vq.codes_per_head,
            cw: vq.code_dim / vq.num_heads,
            len_w: 128,
            forced: forced as i32,
        }
    }

    fn launch(&self, rec: &mut Recorder, kernel: &str, args: &[Arg], threads: usize) {
        rec.dispatch(&self.pso[kernel], args, (threads.div_ceil(256), 1, 1), (256, 1, 1));
    }

    // -- stages ----------------------------------------------------------------------------------

    /// Rig preprocessing (G1 forward kinematics / SOMA capture->standard) and normalization.
    pub fn encode_input(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        match self.spec.kind {
            Kind::UnitreeG1 => {
                let args = [Arg::Buf(&b.ctx, 0), Arg::Buf(&b.tgt, 0), Arg::Buf(&self.rig_f, 0), Arg::Buf(&self.rig_i, 0), Arg::Buf(&b.gr_in, 0), Arg::Buf(&b.lr_in, 0), Arg::Buf(&b.lp_in, 0), Arg::Bytes(as_bytes(d))];
                rec.dispatch(&self.pso["g1_preprocess"], &args, (n.div_ceil(32), 1, 1), (32, 1, 1));
            }
            Kind::Soma => {
                let j = self.spec.num_joints;
                let p = [(n * 8) as i32, (self.pd - 1) as i32, (j - 1) * 3, j];
                let args = [Arg::Buf(&b.lp_in, 0), Arg::Buf(&self.soma_capture, 0), Arg::Buf(&self.soma_standard, 0), Arg::Bytes(as_bytes(&p))];
                self.launch(rec, "soma_retarget", &args, n * 8 * j as usize);
            }
        }
        let args = [Arg::Buf(&b.gr_in, 0), Arg::Buf(&b.lr_in, 0), Arg::Buf(&b.lp_in, 0), Arg::Buf(&self.stats, 0), Arg::Buf(&b.gn, 0), Arg::Buf(&b.ln, 0), Arg::Buf(&b.pn, 0), Arg::Buf(&b.info, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, "prep", &args, n * 8);
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

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, rec: &mut Recorder, c: &Conv, x: &Buf, t: usize, n_batch: usize, y: &Buf, amask: Option<&Buf>, a_relu: bool, act: i32, ymask: Option<&Buf>, up: bool, res: Option<&Buf>) {
        let rows = n_batch * t;
        let mut p = GemmParams::new(rows, c.n, c.taps * c.cin);
        p.lda = c.cin as i32;
        p.act = act;
        p.a_relu = a_relu as i32;
        p.has_mask = amask.is_some() as i32;
        p.y_mask = ymask.is_some() as i32;
        if c.taps > 1 {
            p.conv_taps = c.taps as i32;
            p.conv_t = t as i32;
            p.conv_cin = c.cin as i32;
            p.conv_dil = c.dil as i32;
            p.conv_up = if up { 2 } else { 1 };
        }
        if res.is_some() {
            p.has_res = 1;
            p.ldr = c.n as i32;
        }
        let mut g = GemmCall::new(x, &c.w, &c.b, y, p);
        g.amask = amask;
        g.ymask = ymask;
        g.res = res.map(|r| (r, 0));
        self.k.run(rec, &g);
    }

    /// `DoubleCondDecoder` (layers.py) for `n` sequences: x (n*16, cin) -> out (n*64, cout).
    #[allow(clippy::too_many_arguments)]
    fn decoder(&self, rec: &mut Recorder, dec: &Decoder, n: usize, x: &Buf, ext: &Buf, ext_dim: usize, tgt: &Buf, has_tgt: &Buf, out: &Buf) {
        self.decoder_limited(rec, dec, n, x, ext, ext_dim, tgt, has_tgt, out, usize::MAX);
    }

    /// Debug: encode only the first `limit` decoder steps (0: conv0 only; 1: + level-0 target; 2: + ext; ...).
    #[allow(clippy::too_many_arguments)]
    fn decoder_limited(&self, rec: &mut Recorder, dec: &Decoder, n: usize, x: &Buf, ext: &Buf, ext_dim: usize, tgt: &Buf, has_tgt: &Buf, out: &Buf, limit: usize) {
        let b = &self.b;
        let masks = [&b.m16, &b.m32, &b.m64];
        let mut step = 0usize;
        #[allow(unused_assignments)]
        let _ = step;
        macro_rules! budget { () => { if step >= limit { return; } #[allow(unused_assignments)] { step += 1; } } }
        self.conv(rec, &dec.conv0, x, 16, n, &b.h, Some(&b.m16), false, ACT_RELU, None, false, None);
        for (i, lvl) in dec.levels.iter().enumerate() {
            budget!();
            let fpp = 4 >> i;
            let p_len = 16 << i;
            // Overwrite the frames that carry a target pose (predicated rows of h viewed as frames).
            let mut p = GemmParams::new(n * 64, lvl.target.n, lvl.target.k);
            p.act = ACT_RELU;
            p.y_pred = 1;
            let mut g = GemmCall::new(tgt, &lvl.target.w, &lvl.target.b, &b.h, p);
            g.ymask = Some(has_tgt);
            self.k.run(rec, &g);
            budget!();
            // h = relu(Linear([h, external.reshape(n, P, -1)]))
            let mut p = GemmParams::new(n * p_len, lvl.ext.n, lvl.ext.k);
            p.act = ACT_RELU;
            p.lda = 512;
            p.k1 = 512;
            p.k2 = (fpp * ext_dim) as i32;
            p.lda2 = (fpp * ext_dim) as i32;
            let mut g = GemmCall::new(&b.h, &lvl.ext.w, &lvl.ext.b, &b.h2, p);
            g.a2 = Some(ext);
            self.k.run(rec, &g);
            let m = masks[i];
            for (c1, c2) in &lvl.res {
                budget!();
                self.conv(rec, c1, &b.h2, p_len, n, &b.t1, Some(m), true, 0, None, false, None);
                self.conv(rec, c2, &b.t1, p_len, n, &b.h2, Some(m), true, 0, Some(m), false, Some(&b.h2));
            }
            budget!();
            self.conv(rec, &lvl.up, &b.h2, p_len * 2, n, &b.h, Some(masks[i + 1]), false, 0, None, true, None);
        }
        budget!();
        self.conv(rec, &dec.post, &b.h, 64, n, &b.h2, Some(&b.m64), false, ACT_RELU, Some(&b.m64), false, None);
        budget!();
        self.conv(rec, &dec.out, &b.h2, 64, n, out, None, false, 0, Some(&b.m64), false, None);
    }

    /// Root network from gn/ln/pn + masks: durations (nt, token masks, logits) and the root path pr.
    pub fn encode_root(&self, rec: &mut Recorder, d: &Dims) {
        self.encode_root_a(rec, d);
        self.encode_root_b(rec, d, usize::MAX, true);
    }

    /// First half of the root network: duration head, second-transformer input, dense conditions.
    pub fn encode_root_a(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let rows = n * 8;
        self.blend_linear(rec, &self.proj_local_pose, &b.pn, rows, &b.x0, 0, 384, &b.hp, &self.no_local_pose, 1);
        self.blend_linear(rec, &self.proj_local_root, &b.ln, rows, &b.x0, 256, 384, &b.hl, &self.no_local_root, 1);
        self.blend_linear(rec, &self.proj_global_root, &b.gn, rows, &b.x0, 320, 384, &b.hg, &self.no_global_root, 1);
        self.fcblock(rec, &self.fc_start, &b.x0, rows, &b.s);
        self.fcblock(rec, &self.fc_end, &b.x0, rows, &b.e);
        let args = [Arg::Buf(&b.s, 0), Arg::Buf(&b.e, 0), Arg::Buf(&self.input_pos, 0), Arg::Buf(&self.query_emb, 0), Arg::Buf(&b.nt_in, 0), Arg::Buf(&b.x1, 0), Arg::Buf(&b.fe, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, "root_assemble", &args, n * 9 * 512);
        self.shared.encode(&self.k, rec, &b.x1, None, n, 9, &b.tf);
        self.k.run(rec, &GemmCall::new(&b.x1, &self.logit_head.w, &self.logit_head.b, &b.log1, GemmParams::new(n * 9, 12, 512)));
        let args = [
            Arg::Buf(&b.log1, 0), Arg::Buf(&b.allowed, 0), Arg::Buf(&b.nt_in, 0), Arg::Buf(&b.fe, 0), Arg::Buf(&self.middle_emb, 0),
            Arg::Buf(&self.root_pos, 0), Arg::Buf(&b.x2, 0), Arg::Buf(&b.pad2, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&b.m16, 0),
            Arg::Buf(&b.logits, 0), Arg::Bytes(as_bytes(d)), Arg::Buf(&b.m32, 0), Arg::Buf(&b.m64, 0), Arg::Buf(&b.ppad, 0),
            Arg::Buf(&self.input_pos, 0),
        ];
        self.launch(rec, "root_second_input", &args, n * 25 * 512);
        let args = [
            Arg::Buf(&b.fe, 0), Arg::Buf(&b.gn, 0), Arg::Buf(&b.hg, 0), Arg::Buf(&b.hl, 0), Arg::Buf(&b.hp, 0),
            Arg::Buf(&self.no_frame, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&b.dframe, 0), Arg::Buf(&b.droot, 0), Arg::Buf(&b.dhas, 0),
            Arg::Bytes(as_bytes(d)),
        ];
        self.launch(rec, "root_dense_cond", &args, n * 64 * 512);
    }

    /// Second half of the root network (debug entry: `transformer` false skips it, `limit` caps the decoder).
    pub fn encode_root_b(&self, rec: &mut Recorder, d: &Dims, limit: usize, transformer: bool) {
        let (b, n) = (&self.b, d.b as usize);
        if transformer {
            self.root_token.encode(&self.k, rec, &b.x2, Some(&b.pad2), n, 25, &b.tf);
        }
        let (p, w) = ([n as i32, 25, 16, 9], 512i32);
        let args = [Arg::Buf(&b.x2, 0), Arg::Buf(&b.dec_in, 0), Arg::Bytes(as_bytes(&p)), Arg::Bytes(as_bytes(&w))];
        self.launch(rec, "copy_rows", &args, n * 16 * 512);
        self.decoder_limited(rec, &self.root_dec, n, &b.dec_in, &b.dframe, 512, &b.droot, &b.dhas, &b.pr, limit);
    }

    /// Debug access to decoder scratch buffers (h, h2, t1).
    pub fn scratch(&self) -> [&Buf; 3] {
        [&self.b.h, &self.b.h2, &self.b.t1]
    }

    /// Root path -> pose conditions (global_root_to_local, last-frame patch, dense pose conditions).
    pub fn encode_root_to_pose(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        let args = [
            Arg::Buf(&b.pr, 0), Arg::Buf(&b.nt, 0), Arg::Buf(&self.stats, 0), Arg::Buf(&b.ln, 0), Arg::Buf(&b.hl, 0),
            Arg::Buf(&b.pn, 0), Arg::Buf(&b.hp, 0), Arg::Buf(&self.len_emb, 0), Arg::Buf(&b.ext, 0), Arg::Buf(&b.rcond, 0),
            Arg::Buf(&b.pcond, 0), Arg::Buf(&b.hpc, 0), Arg::Buf(&b.le, 0), Arg::Bytes(as_bytes(d)),
        ];
        self.launch(rec, "root_to_pose", &args, n * 64);
    }

    /// Pose network (single MaskGIT step) from rcond/pcond/hpc/le/ppad: logits, tokens, latents.
    pub fn encode_pose(&self, rec: &mut Recorder, d: &Dims) {
        let (b, n) = (&self.b, d.b as usize);
        self.k.run(rec, &GemmCall::new(&b.rcond, &self.p_root.w, &self.p_root.b, &b.re, GemmParams::new(n * 16, 256, 16)));
        self.blend_linear(rec, &self.p_local_pose, &b.pcond, n * 64, &b.pe, 0, 160, &b.hpc, &self.token_const, 4);
        let mut p = GemmParams::new(n * 16, 1024, 1024);
        p.act = ACT_RELU;
        p.lda = 640;
        p.k1 = 640;
        p.k2 = 256;
        p.lda2 = 256;
        p.lda3 = 128;
        p.has_res = 1;
        p.ldr = 1024;
        p.r_mod = 16;
        let mut g = GemmCall::new(&b.pe, &self.p_input.w, &self.p_input.b, &b.px, p);
        g.a2 = Some(&b.re);
        g.a3 = Some(&b.le);
        g.res = Some((&self.pose_pos, 0));
        self.k.run(rec, &g);
        self.pose_tf.encode(&self.k, rec, &b.px, Some(&b.ppad), n, 16, &b.tf);
        let nl = self.p_logits.n;
        self.k.run(rec, &GemmCall::new(&b.px, &self.p_logits.w, &self.p_logits.b, &b.plog, GemmParams::new(n * 16, nl, 1024)));
        let args = [Arg::Buf(&b.plog, 0), Arg::Buf(&self.codebook, 0), Arg::Buf(&b.tokens, 0), Arg::Buf(&b.lat, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, "argmax_dequant", &args, n * 16 * self.spec.vq.num_heads as usize);
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
        rec.dispatch(&self.pso["finish_root"], &args, (n.div_ceil(32), 1, 1), (32, 1, 1));
        let args = [Arg::Buf(&b.recon, 0), Arg::Buf(&self.body_stats, 0), Arg::Buf(&b.out, 0), Arg::Bytes(as_bytes(d))];
        self.launch(rec, "body_out", &args, n * 64 * (self.ld - 4));
        match self.spec.kind {
            Kind::Soma => {
                let j = self.spec.num_joints;
                let p = [(n * 64) as i32, self.gd as i32, 5 + (j - 1) * 3, j];
                let args = [Arg::Buf(&b.out, 0), Arg::Buf(&self.soma_standard, 0), Arg::Buf(&self.soma_capture, 0), Arg::Bytes(as_bytes(&p))];
                self.launch(rec, "soma_retarget", &args, n * 64 * j as usize);
            }
            Kind::UnitreeG1 => {
                let args = [Arg::Buf(&b.out, 0), Arg::Buf(&self.rig_f, 0), Arg::Buf(&self.rig_i, 0), Arg::Buf(&b.qpos, 0), Arg::Buf(&b.contacts, 0), Arg::Bytes(as_bytes(d))];
                self.launch(rec, "g1_postprocess", &args, n * 64);
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
        match self.spec.kind {
            Kind::UnitreeG1 => &crate::model::G1_INPUTS,
            Kind::Soma => &crate::model::SOMA_INPUTS,
        }
    }

    /// Copy ONNX-named inputs into the input buffers; returns the batch size.
    pub fn upload_inputs(&self, inputs: &HashMap<String, Npy>) -> Result<usize> {
        let get = |k: &str| inputs.get(k).ok_or_else(|| anyhow!("missing input {k}"));
        let n = get("has_global_root_values")?.shape[0] as usize;
        if n > self.max_batch {
            bail!("batch {n} exceeds max_batch {}", self.max_batch);
        }
        let b = &self.b;
        let copy = |dst: &Buf, src: Vec<f32>| dst.as_mut_slice()[..src.len()].copy_from_slice(&src);
        let mask = |v: &Npy| v.to_f32().into_iter().map(|x| (x != 0.0) as i32 as f32).collect::<Vec<_>>();
        match self.spec.kind {
            Kind::UnitreeG1 => {
                copy(&b.ctx, get("context_qpos")?.to_f32());
                copy(&b.tgt, get("target_qpos")?.to_f32());
            }
            Kind::Soma => {
                copy(&b.gr_in, get("global_root_values")?.to_f32());
                copy(&b.lr_in, get("local_root_values")?.to_f32());
                copy(&b.lp_in, get("local_poses")?.to_f32());
            }
        }
        copy(&b.hg, mask(get("has_global_root_values")?));
        copy(&b.hl, mask(get("has_local_root_values")?));
        copy(&b.hp, mask(get("has_local_poses")?));
        copy(&b.allowed, mask(get("allowed_pred_num_tokens")?));
        Ok(n)
    }

    pub fn num_tokens(&self, n: usize) -> Vec<i32> {
        self.b.nt.as_slice()[..n].iter().map(|v| v.to_bits() as i32).collect()
    }

    /// Main output (`pred_global_motions` or `pred_qpos`), durations, and G1 foot contacts.
    pub fn outputs(&self, n: usize) -> (Vec<f32>, Vec<i32>, Option<Vec<f32>>) {
        let b = &self.b;
        match self.spec.kind {
            Kind::Soma => (b.out.as_slice()[..n * 64 * self.gd].to_vec(), self.num_tokens(n), None),
            Kind::UnitreeG1 => (b.qpos.as_slice()[..n * 64 * 36].to_vec(), self.num_tokens(n), Some(b.contacts.as_slice()[..n * 64 * 4].to_vec())),
        }
    }

    /// ONNX-Runtime-style call: upload, one command buffer, wait, read back.
    pub fn run(&self, inputs: &HashMap<String, Npy>) -> Result<(Vec<f32>, Vec<i32>, Option<Vec<f32>>)> {
        let n = self.upload_inputs(inputs)?;
        let d = self.dims(n, false);
        let mut rec = Recorder::new(&self.gpu);
        self.encode(&mut rec, &d);
        wait(&rec.commit());
        Ok(self.outputs(n))
    }
}
