// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Post-norm transformer encoder (PyTorch `nn.TransformerEncoderLayer`, ReLU FFN) on the custom
//! Metal runtime: 7 dispatches per layer (qkv GEMM, attention, out GEMM, add+LN, FFN GEMM+ReLU,
//! FFN GEMM, add+LN) into preallocated buffers.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use mlx_rs::Array;

use super::{Arg, Buf, Gpu, Pipeline, Recorder, as_bytes};
use crate::ops::to_vec_f32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GemmParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldy: i32,
    pub act: i32,
    pub conv_taps: i32,
    pub conv_t: i32,
    pub conv_cin: i32,
    pub conv_dil: i32,
    pub has_mask: i32,
    pub has_res: i32,
    pub ldr: i32,
    pub k_split: i32,
    pub splits: i32,
    pub a_relu: i32,
    pub conv_up: i32,
    pub k1: i32,
    pub k2: i32,
    pub lda2: i32,
    pub lda3: i32,
    pub y_mask: i32,
    pub y_pred: i32,
    pub blend: i32,
    pub blend_mod: i32,
    pub r_mod: i32,
}

impl GemmParams {
    /// Plain `Y[M, N] = A[M, K] Wᵀ + b` with contiguous rows.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        GemmParams { m: m as i32, n: n as i32, k: k as i32, lda: k as i32, ldy: n as i32, ..Default::default() }
    }
    fn steel_ok(&self) -> bool {
        self.conv_taps == 0 && self.has_mask == 0 && self.a_relu == 0 && self.k1 == 0 && self.y_mask == 0
            && self.y_pred == 0 && self.blend == 0 && self.r_mod == 0
    }
}

/// One GEMM launch: operands, parameters and optional side inputs.
pub struct GemmCall<'a> {
    pub a: (&'a Buf, usize),
    pub w: &'a Buf,
    pub bias: &'a Buf,
    pub y: (&'a Buf, usize),
    pub p: GemmParams,
    pub amask: Option<&'a Buf>,
    pub res: Option<(&'a Buf, usize)>,
    pub a2: Option<&'a Buf>,
    pub a3: Option<&'a Buf>,
    pub ymask: Option<&'a Buf>,
    pub missing: Option<&'a Buf>,
}

impl<'a> GemmCall<'a> {
    pub fn new(a: &'a Buf, w: &'a Buf, bias: &'a Buf, y: &'a Buf, p: GemmParams) -> Self {
        GemmCall { a: (a, 0), w, bias, y: (y, 0), p, amask: None, res: None, a2: None, a3: None, ymask: None, missing: None }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    b: i32,
    l: i32,
    d: i32,
    h: i32,
    hd: i32,
    has_pad: i32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LnParams {
    d: i32,
    eps: f32,
}

/// `GEMMParams` / `GEMMAddMMParams` of MLX's steel GEMM (mlx/backend/metal/kernels/steel/gemm/params.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct SteelParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: i64,
    batch_stride_b: i64,
    batch_stride_d: i64,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SteelAddmm {
    ldc: i32,
    fdc: i32,
    batch_stride_c: i64,
    alpha: f32,
    beta: f32,
}

/// The steel variant MLX selects on this GPU class (`applegpu_g16s`: bm64 bn64 bk16 wm2 wn2).
const STEEL: &str = "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2";
const STEEL_TILE: (usize, usize, usize, usize, usize) = (64, 64, 16, 2, 2);
/// Plain GEMMs with more rows than this use MLX's steel kernel; fewer use gemm_mma.
pub const STEEL_MIN_ROWS: usize = 33;
/// Non-plain GEMMs (convs, concat, masks, blends) with at least this many rows are materialized
/// to a plain matrix and run on steel; smaller ones use the fused gemm_mma.
pub const MATERIALIZE_MIN_ROWS: usize = 256;

pub const ACT_NONE: i32 = 0;
pub const ACT_RELU: i32 = 1;
pub const ACT_LEAKY: i32 = 2;

/// One GEMM tile configuration (must match an instantiation in kernels.metal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
}

pub const TILES: [Tile; 13] = [
    Tile { bm: 16, bn: 32, bk: 32, wm: 1, wn: 2 },
    Tile { bm: 16, bn: 64, bk: 32, wm: 1, wn: 4 },
    Tile { bm: 16, bn: 64, bk: 64, wm: 1, wn: 4 },
    Tile { bm: 32, bn: 32, bk: 16, wm: 2, wn: 2 },
    Tile { bm: 32, bn: 64, bk: 16, wm: 2, wn: 2 },
    Tile { bm: 32, bn: 64, bk: 32, wm: 2, wn: 2 },
    Tile { bm: 64, bn: 32, bk: 16, wm: 2, wn: 2 },
    Tile { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2 },
    Tile { bm: 64, bn: 32, bk: 32, wm: 2, wn: 2 },
    Tile { bm: 64, bn: 64, bk: 16, wm: 2, wn: 4 },
    Tile { bm: 64, bn: 64, bk: 32, wm: 2, wn: 4 },
    Tile { bm: 128, bn: 32, bk: 16, wm: 4, wn: 1 },
    Tile { bm: 64, bn: 64, bk: 16, wm: 4, wn: 2 },
];

/// A GEMM launch plan: tile and number of K splits.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub tile: Tile,
    pub splits: usize,
}

/// Kernels used by the runtime, resolved once.
pub struct Kernels {
    pub gemm: Vec<(Tile, Pipeline)>,
    pub reduce: Pipeline,
    pub epilogue: Pipeline,
    pub materialize: Pipeline,
    /// Scratch for materialized A (M x K) and steel outputs (M x N).
    pub a_scratch: Buf,
    pub y_scratch: Buf,
    pub attention: Pipeline,
    pub add_ln: Pipeline,
    pub dummy: Buf,
    pub zeros: Buf,
    /// Split-K partial sums (grown on demand at build time).
    pub partial: Buf,
    /// Override for sweeps: always use this plan (and never steel).
    pub force: Option<Plan>,
    steel_pipelines: HashMap<(bool, bool, bool), Pipeline>,
    splitk_pipelines: HashMap<String, Pipeline>,
    splitk_accum: Pipeline,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let gemm = TILES
            .iter()
            .map(|t| Ok((*t, gpu.pipeline(&format!("gemm_mma_{}_{}_{}_{}_{}", t.bm, t.bn, t.bk, t.wm, t.wn))?)))
            .collect::<Result<_>>()?;
        Ok(Kernels {
            gemm,
            reduce: gpu.pipeline("gemm_splitk_reduce")?,
            epilogue: gpu.pipeline("epilogue_apply")?,
            materialize: gpu.pipeline("materialize_a")?,
            a_scratch: gpu.buffer(16 << 20),
            y_scratch: gpu.buffer(4 << 20),
            attention: gpu.pipeline("attention")?,
            add_ln: gpu.pipeline("add_layernorm")?,
            dummy: gpu.buffer(4),
            zeros: gpu.buffer(16384),
            partial: gpu.buffer(8 << 20),
            force: None,
            steel_pipelines: {
                let mut map = HashMap::new();
                for bits in 0..8u8 {
                    let (am, an, ak) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
                    let consts = [(10, false), (100, true), (110, false), (200, am), (201, an), (202, ak)];
                    map.insert((am, an, ak), gpu.mlx_pipeline(STEEL, &consts)?);
                }
                map
            },
            splitk_pipelines: {
                let mut map = HashMap::new();
                for (bm, bn) in [(16, 16), (16, 32), (32, 16), (32, 32)] {
                    for mn in ["t", "n"] {
                        for k in ["t", "n"] {
                            let name = format!("steel_gemm_splitk_nt_float32_float32_bm{bm}_bn{bn}_bk16_wm2_wn2_MN_{mn}aligned_K_{k}aligned");
                            map.insert(name.clone(), gpu.mlx_pipeline(&name, &[])?);
                        }
                    }
                }
                map
            },
            splitk_accum: gpu.mlx_pipeline("steel_gemm_splitk_accum_float32_float32_axbpy", &[])?,
        })
    }

    /// Heuristic plan: small row tiles for small M (no wasted rows at batch 1), and split K
    /// until the grid has enough threadgroups to keep every GPU core streaming weights.
    pub fn plan(&self, m: usize, n: usize, k: usize) -> Plan {
        if let Some(p) = self.force {
            return p;
        }
        let tile = if m <= 16 {
            TILES[1] // 16x64x32, 1x4 simdgroups
        } else if m <= 32 {
            TILES[5] // 32x64x32
        } else if m <= 64 && n < 1536 {
            TILES[3] // 32x32x16
        } else {
            TILES[11] // 64x64x32, 2x4 simdgroups
        };
        let groups = n.div_ceil(tile.bn) * m.div_ceil(tile.bm);
        let mut splits = 1;
        while groups * splits < 128 && k / (splits * 2) >= 256 && splits < 8 {
            splits *= 2;
        }
        Plan { tile, splits }
    }

    /// `Y = epilogue(A Wᵀ + b)`: MLX's steel kernel for large plain GEMMs (as MLX itself would
    /// dispatch them), the model-specific `gemm_mma` otherwise (plus a reduce when split over K).
    pub fn run(&self, rec: &mut Recorder, g: &GemmCall) {
        let mut p = g.p;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        if self.force.is_none() && m >= STEEL_MIN_ROWS && p.steel_ok() && g.res.is_none() && p.act == 0 {
            self.steel(rec, g.a, g.w, g.bias, g.y, p, None);
            return;
        }
        if self.force.is_none() && (m >= MATERIALIZE_MIN_ROWS || (m >= STEEL_MIN_ROWS && p.steel_ok()))
            && m * k <= self.a_scratch.len && m * n <= self.y_scratch.len
        {
            self.materialized(rec, g);
            return;
        }
        let plan = self.plan(m, n, k);
        let t = plan.tile;
        let k_split = (k.div_ceil(plan.splits)).div_ceil(t.bk) * t.bk;
        let splits = k.div_ceil(k_split);
        p.k_split = k_split as i32;
        p.splits = splits as i32;
        let d = &self.dummy;
        let (rb, roff) = g.res.unwrap_or((d, 0));
        let pso = &self.gemm.iter().find(|(tt, _)| *tt == t).expect("tile instantiated").1;
        let out = if splits > 1 { (&self.partial, 0) } else { g.y };
        let args = [
            Arg::Buf(g.a.0, g.a.1 * 4),
            Arg::Buf(g.w, 0),
            Arg::Buf(g.bias, 0),
            Arg::Buf(out.0, out.1 * 4),
            Arg::Bytes(as_bytes(&p)),
            Arg::Buf(g.amask.unwrap_or(d), 0),
            Arg::Buf(rb, roff * 4),
            Arg::Buf(g.a2.unwrap_or(d), 0),
            Arg::Buf(g.a3.unwrap_or(d), 0),
            Arg::Buf(g.ymask.unwrap_or(d), 0),
            Arg::Buf(g.missing.unwrap_or(d), 0),
        ];
        rec.dispatch(pso, &args, (n.div_ceil(t.bn), m.div_ceil(t.bm), splits), (t.wm * t.wn * 32, 1, 1));
        if splits > 1 {
            let args = [
                Arg::Buf(&self.partial, 0),
                Arg::Buf(g.bias, 0),
                Arg::Buf(g.y.0, g.y.1 * 4),
                Arg::Bytes(as_bytes(&p)),
                Arg::Buf(rb, roff * 4),
                Arg::Buf(g.ymask.unwrap_or(d), 0),
                Arg::Buf(g.missing.unwrap_or(d), 0),
            ];
            rec.dispatch(&self.reduce, &args, ((m * n).div_ceil(256), 1, 1), (256, 1, 1));
        }
    }

    /// Compatibility wrapper used by the sweep: plain GEMM with optional residual.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self, rec: &mut Recorder, a: (&Buf, usize), w: &Buf, bias: &Buf, y: (&Buf, usize), p: GemmParams,
        mask: Option<&Buf>, res: Option<(&Buf, usize)>,
    ) {
        let mut g = GemmCall::new(a.0, w, bias, y.0, p);
        g.a = a;
        g.y = y;
        g.amask = mask;
        g.res = res;
        self.run(rec, &g);
    }

    /// Large GEMM with extras: materialize A (unless already plain), steel into a scratch output,
    /// then apply the full epilogue into Y.
    fn materialized(&self, rec: &mut Recorder, g: &GemmCall) {
        let p = g.p;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let d = &self.dummy;
        let plain = p.conv_taps == 0 && p.k1 == 0 && p.a_relu == 0 && p.has_mask == 0;
        let a = if plain {
            g.a
        } else {
            let args = [
                Arg::Buf(g.a.0, g.a.1 * 4),
                Arg::Buf(g.a2.unwrap_or(d), 0),
                Arg::Buf(g.a3.unwrap_or(d), 0),
                Arg::Buf(g.amask.unwrap_or(d), 0),
                Arg::Buf(&self.a_scratch, 0),
                Arg::Bytes(as_bytes(&p)),
            ];
            let items = if (k % 4 == 0) && (p.lda % 4 == 0) { m * k / 4 } else { m * k };
            rec.dispatch(&self.materialize, &args, (items.div_ceil(256), 1, 1), (256, 1, 1));
            (&self.a_scratch, 0)
        };
        let mut sp = GemmParams::new(m, n, k);
        sp.lda = if plain { p.lda } else { k as i32 };
        self.steel(rec, a, g.w, g.bias, (&self.y_scratch, 0), sp, None);
        let (rb, roff) = g.res.unwrap_or((d, 0));
        let args = [
            Arg::Buf(&self.y_scratch, 0),
            Arg::Buf(g.y.0, g.y.1 * 4),
            Arg::Bytes(as_bytes(&p)),
            Arg::Buf(&self.zeros, 0),
            Arg::Buf(rb, roff * 4),
            Arg::Buf(g.ymask.unwrap_or(d), 0),
            Arg::Buf(g.missing.unwrap_or(d), 0),
        ];
        rec.dispatch(&self.epilogue, &args, ((m * n).div_ceil(256), 1, 1), (256, 1, 1));
    }

    /// MLX's steel GEMM (`addmm` with a broadcast bias), then activation / residual if needed.
    #[allow(clippy::too_many_arguments)]
    fn steel(&self, rec: &mut Recorder, a: (&Buf, usize), w: &Buf, bias: &Buf, y: (&Buf, usize), p: GemmParams, res: Option<(&Buf, usize)>) {
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        // MLX's rule (steel_matmul_axpby): small M*N with large K goes to split-K.
        let (tm16, tn16, tk16) = (m.div_ceil(16), n.div_ceil(16), k / 16);
        if tm16 * tn16 <= 2048 && tk16 >= 8 && k >= m.max(n) && p.ldy as usize == n {
            self.steel_splitk(rec, a, w, bias, y, p);
        } else {
            self.steel_regular(rec, a, w, bias, y, p);
        }
        let _ = res;
    }

    /// `steel_gemm_splitk_axpby` as in MLX: partials into a scratch buffer, then the accumulate
    /// kernel adds the broadcast bias (alpha = beta = 1).
    fn steel_splitk(&self, rec: &mut Recorder, a: (&Buf, usize), w: &Buf, bias: &Buf, y: (&Buf, usize), p: GemmParams) {
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let (tm32, tn32, tk16) = (m.div_ceil(32), n.div_ceil(32), k / 16);
        let bm = if m < 40 { 16 } else { 32 };
        let bn = if n < 40 { 16 } else { 32 };
        let bk = 16;
        let partitions = (tk16 / (tm32 * tn32)).max(1).next_power_of_two().clamp(2, 32);
        let iters = (k / bk) / partitions;
        let mn_aligned = m % bm == 0 && n % bn == 0;
        let k_aligned = k % bk == 0;
        let name = format!(
            "steel_gemm_splitk_nt_float32_float32_bm{bm}_bn{bn}_bk16_wm2_wn2_MN_{}aligned_K_{}aligned",
            if mn_aligned { "t" } else { "n" },
            if k_aligned { "t" } else { "n" }
        );
        let params: [i32; 13] = [
            m as i32, n as i32, k as i32, p.lda, k as i32, n as i32, n.div_ceil(bn) as i32, m.div_ceil(bm) as i32,
            partitions as i32, (m * n) as i32, (iters * bk) as i32, 0, iters as i32,
        ];
        assert!(partitions * m * n <= self.partial.len, "split-K scratch too small");
        let args = [Arg::Buf(a.0, a.1 * 4), Arg::Buf(w, 0), Arg::Buf(&self.partial, 0), Arg::Bytes(as_bytes(&params))];
        rec.dispatch(&self.splitk_pipelines[&name], &args, (n.div_ceil(bn), m.div_ceil(bm), partitions), (32, 2, 2));
        let (parts, stride, ldd, ldc, fdc, alpha, beta) = (partitions as i32, (m * n) as i32, n as i32, 0i32, 1i32, 1f32, 1f32);
        let args = [
            Arg::Buf(&self.partial, 0),
            Arg::Buf(y.0, y.1 * 4),
            Arg::Bytes(as_bytes(&parts)),
            Arg::Bytes(as_bytes(&stride)),
            Arg::Bytes(as_bytes(&ldd)),
            Arg::Buf(bias, 0),
            Arg::Bytes(as_bytes(&ldc)),
            Arg::Bytes(as_bytes(&fdc)),
            Arg::Bytes(as_bytes(&alpha)),
            Arg::Bytes(as_bytes(&beta)),
        ];
        rec.dispatch_threads(&self.splitk_accum, &args, (n, m, 1), (32.min(n), 8, 1));
    }

    fn steel_regular(&self, rec: &mut Recorder, a: (&Buf, usize), w: &Buf, bias: &Buf, y: (&Buf, usize), p: GemmParams) {
        let (bm, bn, bk, wm, wn) = STEEL_TILE;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let pso = &self.steel_pipelines[&(m % bm == 0, n % bn == 0, k % bk == 0)];
        let sp = SteelParams {
            m: p.m,
            n: p.n,
            k: p.k,
            lda: p.lda,
            ldb: p.k,
            ldd: p.ldy,
            tiles_n: n.div_ceil(bn) as i32,
            tiles_m: m.div_ceil(bm) as i32,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: 0,
            swizzle_log: 0,
            gemm_k_iterations_aligned: (k / bk) as i32,
            batch_ndim: 1,
        };
        let ap = SteelAddmm { ldc: 0, fdc: 1, batch_stride_c: 0, alpha: 1.0, beta: 1.0 };
        let args = [
            Arg::Buf(a.0, a.1 * 4),
            Arg::Buf(w, 0),
            Arg::Buf(bias, 0),
            Arg::Buf(y.0, y.1 * 4),
            Arg::Bytes(as_bytes(&sp)),
            Arg::Bytes(as_bytes(&ap)),
        ];
        rec.dispatch(pso, &args, (n.div_ceil(bn), m.div_ceil(bm), 1), (32, wn, wm));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear(&self, rec: &mut Recorder, a: &Buf, w: &Buf, bias: &Buf, y: &Buf, m: usize, n: usize, k: usize, act: i32) {
        let mut p = GemmParams::new(m, n, k);
        p.act = act;
        self.run(rec, &GemmCall::new(a, w, bias, y, p));
    }

    pub fn add_layernorm(&self, rec: &mut Recorder, x: &Buf, r: &Buf, w: &Buf, b: &Buf, y: &Buf, rows: usize, d: usize) {
        let p = LnParams { d: d as i32, eps: 1e-5 };
        let args = [Arg::Buf(x, 0), Arg::Buf(r, 0), Arg::Buf(w, 0), Arg::Buf(b, 0), Arg::Buf(y, 0), Arg::Bytes(as_bytes(&p))];
        rec.dispatch(&self.add_ln, &args, (rows, 1, 1), (256, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, rec: &mut Recorder, qkv: &Buf, pad: Option<&Buf>, out: &Buf, b: usize, l: usize, d: usize, h: usize) {
        let hd = d / h;
        let p = AttnParams {
            b: b as i32,
            l: l as i32,
            d: d as i32,
            h: h as i32,
            hd: hd as i32,
            has_pad: pad.is_some() as i32,
            scale: (1.0 / (hd as f64).sqrt()) as f32,
        };
        let args = [Arg::Buf(qkv, 0), Arg::Buf(pad.unwrap_or(&self.dummy), 0), Arg::Buf(out, 0), Arg::Bytes(as_bytes(&p))];
        rec.threadgroup_memory(3 * l * (hd + 1) * 4, 0);
        rec.dispatch(&self.attention, &args, (b * h, 1, 1), (l * 32, 1, 1));
    }
}

pub fn upload(gpu: &Gpu, tensors: &HashMap<String, Array>, name: &str) -> Result<Buf> {
    let t = tensors.get(name).ok_or_else(|| anyhow!("missing weight {name}"))?;
    Ok(gpu.upload(&to_vec_f32(t)?))
}

struct Layer {
    in_w: Buf,
    in_b: Buf,
    out_w: Buf,
    out_b: Buf,
    l1_w: Buf,
    l1_b: Buf,
    l2_w: Buf,
    l2_b: Buf,
    n1_w: Buf,
    n1_b: Buf,
    n2_w: Buf,
    n2_b: Buf,
}

pub struct Transformer {
    layers: Vec<Layer>,
    pub d: usize,
    pub heads: usize,
    pub ffn: usize,
}

/// Scratch buffers for up to `rows` tokens.
pub struct Scratch {
    pub x2: Buf,
    pub qkv: Buf,
    pub attn: Buf,
    pub proj: Buf,
    pub ff: Buf,
}

impl Scratch {
    pub fn new(gpu: &Gpu, rows: usize, d: usize, ffn: usize) -> Self {
        Scratch { x2: gpu.buffer(rows * d), qkv: gpu.buffer(rows * 3 * d), attn: gpu.buffer(rows * d), proj: gpu.buffer(rows * d), ff: gpu.buffer(rows * ffn) }
    }
}

impl Transformer {
    pub fn load(gpu: &Gpu, tensors: &HashMap<String, Array>, prefix: &str, n_layers: usize, heads: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let p = |n: &str| format!("{prefix}.layers.{i}.{n}");
            let u = |n: &str| upload(gpu, tensors, &p(n));
            layers.push(Layer {
                in_w: u("self_attn.in_proj.weight")?,
                in_b: u("self_attn.in_proj.bias")?,
                out_w: u("self_attn.out_proj.weight")?,
                out_b: u("self_attn.out_proj.bias")?,
                l1_w: u("linear1.weight")?,
                l1_b: u("linear1.bias")?,
                l2_w: u("linear2.weight")?,
                l2_b: u("linear2.bias")?,
                n1_w: u("norm1.weight")?,
                n1_b: u("norm1.bias")?,
                n2_w: u("norm2.weight")?,
                n2_b: u("norm2.bias")?,
            });
        }
        let d = tensors[&format!("{prefix}.layers.0.norm1.weight")].dim(0) as usize;
        let ffn = tensors[&format!("{prefix}.layers.0.linear1.bias")].dim(0) as usize;
        Ok(Transformer { layers, d, heads, ffn })
    }

    /// In place on `x` (batch * len rows of d). `pad`: (batch, len) bytes, 1 = padded key.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, k: &Kernels, rec: &mut Recorder, x: &Buf, pad: Option<&Buf>, batch: usize, len: usize, s: &Scratch) {
        self.encode_parts(k, rec, x, pad, batch, len, s, 7);
    }

    /// `parts`: bitmask of kernel families to encode (1 GEMMs, 2 attention, 4 layer norms); for profiling.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_parts(&self, k: &Kernels, rec: &mut Recorder, x: &Buf, pad: Option<&Buf>, batch: usize, len: usize, s: &Scratch, parts: u32) {
        let (rows, d, f) = (batch * len, self.d, self.ffn);
        let (g, a, n) = (parts & 1 != 0, parts & 2 != 0, parts & 4 != 0);
        for l in &self.layers {
            if g { k.linear(rec, x, &l.in_w, &l.in_b, &s.qkv, rows, 3 * d, d, ACT_NONE); }
            if a { k.attention(rec, &s.qkv, pad, &s.attn, batch, len, d, self.heads); }
            if g { k.linear(rec, &s.attn, &l.out_w, &l.out_b, &s.proj, rows, d, d, ACT_NONE); }
            if n { k.add_layernorm(rec, x, &s.proj, &l.n1_w, &l.n1_b, &s.x2, rows, d); }
            if g { k.linear(rec, &s.x2, &l.l1_w, &l.l1_b, &s.ff, rows, f, d, ACT_RELU); }
            if g { k.linear(rec, &s.ff, &l.l2_w, &l.l2_b, &s.proj, rows, d, f, ACT_NONE); }
            if n { k.add_layernorm(rec, &s.x2, &s.proj, &l.n2_w, &l.n2_b, x, rows, d); }
        }
    }
}
