// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! GEMM front-end (tile planning, split-K, kernel specialization) and the post-norm transformer
//! encoder (PyTorch `nn.TransformerEncoderLayer`, ReLU FFN) on the custom Metal runtime.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Result, anyhow};
use mlx_rs::Array;
use mlx_rs::ops::{QuantizeOptions, quantize};

use super::{Arg, Buf, Gpu, Pipeline, Precision, Recorder, as_bytes};
use crate::ops::to_vec_f32;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
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

pub const ACT_NONE: i32 = 0;
pub const ACT_RELU: i32 = 1;
pub const ACT_LEAKY: i32 = 2;

/// INT8 weight in MLX's affine format (bits 8, group 64): packed u32 [N][K/4], f32 scales/biases [N][K/64].
pub struct QWeight {
    pub q: Buf,
    pub scales: Buf,
    pub biases: Buf,
}

/// A projection weight: dense `T` or INT8.
pub enum Weight {
    Dense(Buf),
    Q8(QWeight),
}

/// One GEMM tile configuration (must match an instantiation in kernels.metal).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
}

const fn tile(bm: usize, bn: usize, bk: usize, wm: usize, wn: usize) -> Tile {
    Tile { bm, bn, bk, wm, wn }
}

pub const TILES: [Tile; 11] = [
    tile(16, 32, 32, 1, 2),
    tile(16, 64, 32, 1, 4),
    tile(16, 64, 64, 1, 4),
    tile(32, 32, 16, 2, 2),
    tile(32, 64, 16, 2, 2),
    tile(32, 64, 32, 2, 2),
    tile(64, 32, 16, 2, 2),
    tile(64, 64, 16, 2, 2),
    tile(64, 32, 32, 2, 2),
    tile(64, 64, 16, 2, 4),
    tile(64, 64, 32, 2, 4),
];

/// A GEMM launch plan: tile and number of K splits.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub tile: Tile,
    pub splits: usize,
}

/// A GEMM launch class: shape plus the loader features that change the best plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub taps: i32,
    pub up: i32,
    pub concat: bool,
    pub mask: bool,
    pub arelu: bool,
    pub q8: bool,
}

impl PlanKey {
    pub fn of(p: &GemmParams, q8: bool) -> Self {
        PlanKey {
            m: p.m,
            n: p.n,
            k: p.k,
            taps: p.conv_taps,
            up: p.conv_up,
            concat: p.k1 > 0,
            mask: p.has_mask != 0,
            arelu: p.a_relu != 0,
            q8,
        }
    }

    /// The plan-table line for this key (see `plans/`).
    pub fn line(&self, prec: Precision, plan: &Plan) -> String {
        let t = plan.tile;
        format!(
            "{} {} {} {} {} {} {} {} {} {}  {} {} {} {} {} {}",
            prec.name(), self.m, self.n, self.k, self.taps, self.up, self.concat as i32, self.mask as i32, self.arelu as i32, self.q8 as i32,
            t.bm, t.bn, t.bk, t.wm, t.wn, plan.splits
        )
    }
}

/// Tuned plans for the model's GEMM launches, measured with `gemm-tune` on an Apple M4 Pro
/// (20-core GPU). Format per line: precision, key (m n k taps up concat mask arelu q8), plan
/// (bm bn bk wm wn splits).
pub const M4_PRO_PLANS: &str = include_str!("plans/apple_m4_pro.txt");

/// The plan table for this device and precision. `MB_PLAN_TABLE` overrides the built-in choice:
/// `0` disables tuned plans, any other value is the path of a table file written by `gemm-tune`.
fn plan_table(gpu: &Gpu) -> Result<HashMap<PlanKey, Plan>> {
    Ok(match std::env::var("MB_PLAN_TABLE") {
        Ok(v) if v == "0" => HashMap::new(),
        Ok(path) => parse_plans(&std::fs::read_to_string(&path).map_err(|e| anyhow!("MB_PLAN_TABLE {path}: {e}"))?, gpu.prec),
        Err(_) if gpu.device_name() == "Apple M4 Pro" => parse_plans(M4_PRO_PLANS, gpu.prec),
        Err(_) => HashMap::new(),
    })
}

/// Parse the plan-table lines of one precision.
pub fn parse_plans(text: &str, prec: Precision) -> HashMap<PlanKey, Plan> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 16 || f[0] != prec.name() {
            continue;
        }
        let v: Vec<i32> = f[1..].iter().filter_map(|x| x.parse().ok()).collect();
        if v.len() != 15 {
            continue;
        }
        let key = PlanKey { m: v[0], n: v[1], k: v[2], taps: v[3], up: v[4], concat: v[5] != 0, mask: v[6] != 0, arelu: v[7] != 0, q8: v[8] != 0 };
        let t = tile(v[9] as usize, v[10] as usize, v[11] as usize, v[12] as usize, v[13] as usize);
        if TILES.contains(&t) {
            out.entry(key).or_insert(Plan { tile: t, splits: v[14] as usize });
        }
    }
    out
}

/// One GEMM launch: operands, parameters and optional side inputs.
pub struct GemmCall<'a> {
    pub a: (&'a Buf, usize),
    pub w: &'a Weight,
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
    pub fn new(a: &'a Buf, w: &'a Weight, bias: &'a Buf, y: &'a Buf, p: GemmParams) -> Self {
        GemmCall { a: (a, 0), w, bias, y: (y, 0), p, amask: None, res: None, a2: None, a3: None, ymask: None, missing: None }
    }
}

/// Loader specialization flags (function constants 0..7 of gemm_mma).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
struct Flags {
    plain: bool,
    conv: bool,
    concat: bool,
    amask: bool,
    arelu: bool,
    up: bool,
    q8: bool,
    vec: bool,
}

/// Kernels used by the runtime; GEMM pipelines are specialized lazily per (tile, flags).
pub struct Kernels {
    gpu: Gpu,
    pub reduce: Pipeline,
    pub attention: Pipeline,
    /// Simdgroup-matrix attention, keyed by (padded length, head dim).
    attention_mma: HashMap<(usize, usize), Pipeline>,
    /// Use the scalar attention kernel (one lane per key) even where the MMA kernel applies.
    pub scalar_attention: bool,
    pub add_ln: Pipeline,
    pub dummy: Buf,
    pub dummy_t: Buf,
    /// Split-K partial sums.
    pub partial: Buf,
    /// Override for sweeps: always use this plan.
    pub force: Option<Plan>,
    /// Tuned plans for this device (empty on devices without a table; `MB_PLAN_TABLE=0` disables).
    pub table: HashMap<PlanKey, Plan>,
    /// When set, every GEMM launch appends (params with plan, INT8) here; for profiling tools.
    pub log: RefCell<Option<Vec<(GemmParams, bool)>>>,
    gemm: RefCell<HashMap<(Tile, Flags), Pipeline>>,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Kernels {
            reduce: gpu.pipeline("gemm_splitk_reduce")?,
            attention: gpu.pipeline("attention")?,
            attention_mma: [(16, 32), (16, 64), (32, 32), (32, 64)]
                .into_iter()
                .map(|(l, d)| Ok(((l, d), gpu.pipeline(&format!("attention_mma_{l}_{d}"))?)))
                .collect::<Result<_>>()?,
            scalar_attention: false,
            add_ln: gpu.pipeline("add_layernorm")?,
            dummy: gpu.buffer(4),
            dummy_t: gpu.buffer_t(4),
            partial: gpu.buffer(16 << 20),
            force: None,
            table: plan_table(gpu)?,
            log: RefCell::new(None),
            gemm: RefCell::new(HashMap::new()),
            gpu: gpu.clone(),
        })
    }

    fn pipeline(&self, t: Tile, f: Flags) -> Pipeline {
        if let Some(p) = self.gemm.borrow().get(&(t, f)) {
            return p.clone();
        }
        let name = format!("gemm_mma_{}_{}_{}_{}_{}", t.bm, t.bn, t.bk, t.wm, t.wn);
        let consts = [(0, f.plain), (1, f.conv), (2, f.concat), (3, f.amask), (4, f.arelu), (5, f.up), (6, f.q8), (7, f.vec)];
        let p = self.gpu.specialized(&name, &consts).expect("gemm pipeline");
        self.gemm.borrow_mut().insert((t, f), p.clone());
        p
    }

    /// Fallback plan (launches missing from the tuned table): the first preferred tile, among
    /// those wasting at most a fraction of rows to padding, whose grid has enough threadgroups;
    /// then K splits (powers of two, >= 64 K per split) until the grid fills the GPU. Fitted on
    /// `gemm-tune` measurements: FP32 prefers 32x64 tiles, FP16/BF16 16x32 tiles (cheaper to fill
    /// in half precision), INT8 weights 64-row tiles (each tile dequantizes its weights once).
    pub fn heuristic(&self, m: usize, n: usize, k: usize, q8: bool) -> Plan {
        let (prefs, g_tile, g_split, waste): (&[Tile], usize, usize, f64) = if q8 {
            (&[TILES[6], TILES[0], TILES[3]], 320, 240, 0.4)
        } else if self.gpu.prec == Precision::F32 {
            (&[TILES[4], TILES[0], TILES[3]], 160, 120, 0.4)
        } else {
            (&[TILES[0], TILES[1]], 160, 120, 0.15)
        };
        let groups = |t: &Tile| m.div_ceil(t.bm) * n.div_ceil(t.bn);
        let fits: Vec<Tile> = prefs.iter().copied().filter(|t| (m.div_ceil(t.bm) * t.bm) as f64 <= m as f64 * (1.0 + waste)).collect();
        let fits = if fits.is_empty() { vec![*prefs.last().unwrap()] } else { fits };
        let tile = fits.iter().copied().find(|t| groups(t) >= g_tile).unwrap_or(*fits.last().unwrap());
        let mut splits = 1;
        while groups(&tile) * splits < g_split && k / (splits * 2) >= 64 && splits < 8 {
            splits *= 2;
        }
        Plan { tile, splits }
    }

    /// The plan for a dense GEMM of this shape (forced, else the heuristic).
    pub fn plan(&self, m: usize, n: usize, k: usize) -> Plan {
        self.force.unwrap_or_else(|| self.heuristic(m, n, k, false))
    }

    /// The plan for a launch: forced, else tuned, else the heuristic.
    pub fn plan_for(&self, p: &GemmParams, q8: bool) -> Plan {
        if let Some(f) = self.force {
            return f;
        }
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        self.table.get(&PlanKey::of(p, q8)).copied().unwrap_or_else(|| self.heuristic(m, n, k, q8))
    }

    /// `Y = epilogue(A Wᵀ + b)` with the planned kernel (plus a reduce dispatch when split over K).
    pub fn run(&self, rec: &mut Recorder, g: &GemmCall) {
        let mut p = g.p;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let q8 = matches!(g.w, Weight::Q8(_));
        let mut plan = self.plan_for(&p, q8);
        while plan.splits > 1 && (plan.splits * m * n > self.partial.len || k / plan.splits < plan.tile.bk) {
            plan.splits /= 2;
        }
        let t = plan.tile;
        let k_split = (k.div_ceil(plan.splits)).div_ceil(t.bk) * t.bk;
        let splits = k.div_ceil(k_split);
        p.k_split = k_split as i32;
        p.splits = splits as i32;
        let conv = p.conv_taps > 1;
        let concat = p.k1 > 0;
        let m4 = |v: i32| v % 4 == 0;
        let vec = m4(p.k) && m4(p.lda) && (!conv || m4(p.conv_cin)) && (!concat || (m4(p.k1) && m4(p.k2) && m4(p.lda2) && m4(p.lda3)));
        let flags = Flags {
            plain: vec
                && m % t.bm == 0
                && n % t.bn == 0
                && k % t.bk == 0
                && (!conv || p.conv_cin as usize % t.bk == 0)
                && (!concat || (p.k1 as usize % t.bk == 0 && p.k2 as usize % t.bk == 0)),
            conv,
            concat,
            amask: p.has_mask != 0,
            arelu: p.a_relu != 0,
            up: p.conv_up == 2,
            q8,
            vec,
        };
        let pso = self.pipeline(t, flags);
        if let Some(log) = self.log.borrow_mut().as_mut() {
            log.push((p, q8));
        }
        let (d, dt) = (&self.dummy, &self.dummy_t);
        let (rb, roff) = g.res.unwrap_or((dt, 0));
        let (wd, wq, qs, qb) = match g.w {
            Weight::Dense(b) => (b, d, d, d),
            Weight::Q8(q) => (dt, &q.q, &q.scales, &q.biases),
        };
        let args = [
            Arg::Buf(g.a.0, g.a.1),
            Arg::Buf(wd, 0),
            Arg::Buf(g.bias, 0),
            Arg::Buf(g.y.0, g.y.1),
            Arg::Bytes(as_bytes(&p)),
            Arg::Buf(g.amask.unwrap_or(d), 0),
            Arg::Buf(rb, roff),
            Arg::Buf(g.a2.unwrap_or(dt), 0),
            Arg::Buf(g.a3.unwrap_or(dt), 0),
            Arg::Buf(g.ymask.unwrap_or(d), 0),
            Arg::Buf(g.missing.unwrap_or(dt), 0),
            Arg::Buf(wq, 0),
            Arg::Buf(qs, 0),
            Arg::Buf(qb, 0),
            Arg::Buf(&self.partial, 0),
        ];
        rec.dispatch(&pso, &args, (n.div_ceil(t.bn), m.div_ceil(t.bm), splits), (t.wm * t.wn * 32, 1, 1));
        if splits > 1 {
            let args = [
                Arg::Buf(&self.partial, 0),
                Arg::Buf(g.bias, 0),
                Arg::Buf(g.y.0, g.y.1),
                Arg::Bytes(as_bytes(&p)),
                Arg::Buf(rb, roff),
                Arg::Buf(g.ymask.unwrap_or(d), 0),
                Arg::Buf(g.missing.unwrap_or(dt), 0),
            ];
            rec.dispatch(&self.reduce, &args, ((m * n).div_ceil(256), 1, 1), (256, 1, 1));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear(&self, rec: &mut Recorder, a: &Buf, w: &Weight, bias: &Buf, y: &Buf, m: usize, n: usize, k: usize, act: i32) {
        let mut p = GemmParams::new(m, n, k);
        p.act = act;
        self.run(rec, &GemmCall::new(a, w, bias, y, p));
    }

    #[allow(clippy::too_many_arguments)]
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
        let lp = if l <= 16 { 16 } else { 32 };
        if let Some(pso) = self.attention_mma.get(&(lp, hd)).filter(|_| l <= 32 && !self.scalar_attention) {
            rec.dispatch(pso, &args, (b * h, 1, 1), (lp * 4, 1, 1));
            return;
        }
        rec.threadgroup_memory(3 * l * (hd + 1) * 4, 0);
        rec.dispatch(&self.attention, &args, (b * h, 1, 1), (l * 32, 1, 1));
    }
}

fn tensor<'a>(tensors: &'a HashMap<String, Array>, name: &str) -> Result<&'a Array> {
    tensors.get(name).ok_or_else(|| anyhow!("missing weight {name}"))
}

/// Upload a tensor converted to the network element type `T`.
pub fn upload(gpu: &Gpu, tensors: &HashMap<String, Array>, name: &str) -> Result<Buf> {
    Ok(gpu.upload_t(&to_vec_f32(tensor(tensors, name)?)?))
}

/// Upload a projection weight, dense or INT8-quantized exactly like `mlx.nn.quantize` (affine, group 64).
pub fn upload_weight(gpu: &Gpu, tensors: &HashMap<String, Array>, name: &str, int8: bool) -> Result<Weight> {
    if !int8 {
        return Ok(Weight::Dense(upload(gpu, tensors, name)?));
    }
    let w = tensor(tensors, name)?;
    let (q, scales, biases) = quantize(w, QuantizeOptions { group_size: Some(64), bits: Some(8), ..Default::default() })
        .map_err(|e| anyhow!("quantizing {name}: {e:?}"))?;
    let biases = biases.ok_or_else(|| anyhow!("affine quantization returned no biases"))?;
    q.eval()?;
    let qv: Vec<u32> = q.as_slice::<u32>().to_vec();
    Ok(Weight::Q8(QWeight { q: gpu.upload_u32(&qv), scales: gpu.upload(&to_vec_f32(&scales)?), biases: gpu.upload(&to_vec_f32(&biases)?) }))
}

struct Layer {
    in_w: Weight,
    in_b: Buf,
    out_w: Weight,
    out_b: Buf,
    l1_w: Weight,
    l1_b: Buf,
    l2_w: Weight,
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

/// Scratch buffers (network element type) for up to `rows` tokens.
pub struct Scratch {
    pub x2: Buf,
    pub qkv: Buf,
    pub attn: Buf,
    pub proj: Buf,
    pub ff: Buf,
}

impl Scratch {
    pub fn new(gpu: &Gpu, rows: usize, d: usize, ffn: usize) -> Self {
        Scratch {
            x2: gpu.buffer_t(rows * d),
            qkv: gpu.buffer_t(rows * 3 * d),
            attn: gpu.buffer_t(rows * d),
            proj: gpu.buffer_t(rows * d),
            ff: gpu.buffer_t(rows * ffn),
        }
    }
}

impl Transformer {
    /// `int8`: quantize the four projections of every layer (as the Python port's `quantize(8)`).
    pub fn load(gpu: &Gpu, tensors: &HashMap<String, Array>, prefix: &str, n_layers: usize, heads: usize, int8: bool) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let p = |n: &str| format!("{prefix}.layers.{i}.{n}");
            let u = |n: &str| upload(gpu, tensors, &p(n));
            let w = |n: &str| upload_weight(gpu, tensors, &p(n), int8);
            layers.push(Layer {
                in_w: w("self_attn.in_proj.weight")?,
                in_b: u("self_attn.in_proj.bias")?,
                out_w: w("self_attn.out_proj.weight")?,
                out_b: u("self_attn.out_proj.bias")?,
                l1_w: w("linear1.weight")?,
                l1_b: u("linear1.bias")?,
                l2_w: w("linear2.weight")?,
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

    /// In place on `x` (batch * len rows of d). `pad`: (batch, len) float, 1 = padded key.
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
            if g {
                k.linear(rec, x, &l.in_w, &l.in_b, &s.qkv, rows, 3 * d, d, ACT_NONE);
            }
            if a {
                k.attention(rec, &s.qkv, pad, &s.attn, batch, len, d, self.heads);
            }
            if g {
                k.linear(rec, &s.attn, &l.out_w, &l.out_b, &s.proj, rows, d, d, ACT_NONE);
            }
            if n {
                k.add_layernorm(rec, x, &s.proj, &l.n1_w, &l.n1_b, &s.x2, rows, d);
            }
            if g {
                k.linear(rec, &s.x2, &l.l1_w, &l.l1_b, &s.ff, rows, f, d, ACT_RELU);
                k.linear(rec, &s.ff, &l.l2_w, &l.l2_b, &s.proj, rows, d, f, ACT_NONE);
            }
            if n {
                k.add_layernorm(rec, &s.x2, &s.proj, &l.n2_w, &l.n2_b, x, rows, d);
            }
        }
    }
}
