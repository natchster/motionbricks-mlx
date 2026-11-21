// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! GEMM front-end (tile planning, split-K, kernel specialization) and the post-norm transformer
//! encoder (PyTorch `nn.TransformerEncoderLayer`, ReLU FFN) on the custom Metal runtime.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};
use mlx_rs::Array;
use mlx_rs::ops::{QuantizeOptions, quantize};
use objc2_metal::MTLComputePipelineState;

use super::{Arg, Buf, Gpu, Pipeline, Pod, Precision, REFERENCE_GPU_CORES, Recorder, THREADGROUP_MEMORY, as_bytes};
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

// SAFETY: all fields are i32 (no padding, no pointers).
unsafe impl Pod for GemmParams {}

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

// SAFETY: 4-byte fields only (no padding, no pointers).
unsafe impl Pod for AttnParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct LnParams {
    d: i32,
    eps: f32,
}

// SAFETY: 4-byte fields only (no padding, no pointers).
unsafe impl Pod for LnParams {}

pub const ACT_NONE: i32 = 0;
pub const ACT_RELU: i32 = 1;
pub const ACT_LEAKY: i32 = 2;

/// Longest sequence the attention kernels handle (one simdgroup lane per key).
pub const MAX_ATTENTION_LEN: usize = 32;
/// Head dimensions with a simdgroup-matrix attention kernel.
pub const MMA_HEAD_DIMS: [usize; 2] = [32, 64];
/// Split-K partial sums buffer of [`Kernels::new`] (float32 elements).
pub const PARTIAL_MAX: usize = 16 << 20;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// `0` disables tuned plans, any other non-empty value is the path of a table file written by
/// `gemm-tune` (an empty value counts as unset).
///
/// The built-in table applies to every "Apple M4 Pro" (16- and 20-core GPUs): the same GPU
/// generation, cache and memory bandwidth as the one it was measured on. Other Macs get the
/// fallback rule, which scales with the GPU core count, unless `MB_PLAN_TABLE` names a table
/// measured on them. Returns the plans and a description of where they came from.
fn plan_table(gpu: &Gpu) -> Result<(HashMap<PlanKey, Plan>, String)> {
    match std::env::var_os("MB_PLAN_TABLE") {
        Some(v) if v == "0" => Ok((HashMap::new(), "none (MB_PLAN_TABLE=0): fallback rule".into())),
        Some(v) if !v.is_empty() => {
            let path = PathBuf::from(v);
            let text = std::fs::read_to_string(&path).with_context(|| format!("MB_PLAN_TABLE {}", path.display()))?;
            let plans = parse_plans(&text, gpu.prec).with_context(|| format!("MB_PLAN_TABLE {}", path.display()))?;
            Ok((plans, format!("MB_PLAN_TABLE {}", path.display())))
        }
        _ if gpu.info.name == "Apple M4 Pro" => {
            let plans = parse_plans(M4_PRO_PLANS, gpu.prec).context("built-in plan table plans/apple_m4_pro.txt")?;
            Ok((plans, "built-in table plans/apple_m4_pro.txt".into()))
        }
        _ => Ok((HashMap::new(), format!("none for {}: fallback rule for {} GPU cores", gpu.info.name, gpu.info.cores))),
    }
}

/// Parse a plan table (see `plans/`) and return the plans of one precision. Every line is
/// checked, whatever its precision: 16 fields, integer values, 0/1 flags, a tile of [`TILES`] and
/// at least one split; errors name the line. `#` starts a comment. A key listed twice keeps its
/// last plan (`gemm-tune` appends to the table file).
pub fn parse_plans(text: &str, prec: Precision) -> Result<HashMap<PlanKey, Plan>> {
    let mut out = HashMap::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let parsed = (|| -> Result<Option<(PlanKey, Plan)>> {
            let f: Vec<&str> = line.split_whitespace().collect();
            ensure!(f.len() == 16, "expected 16 fields (precision, m n k taps up concat mask arelu q8, bm bn bk wm wn splits), got {}", f.len());
            ensure!(
                [Precision::F32, Precision::F16, Precision::BF16].iter().any(|p| p.name() == f[0]),
                "unknown precision {:?} (float32, float16 or bfloat16; INT8 rows are float32 with q8 = 1)",
                f[0]
            );
            let v = f[1..].iter().map(|x| x.parse::<i32>().map_err(|_| anyhow!("{x:?} is not an integer"))).collect::<Result<Vec<i32>>>()?;
            ensure!(v[0] > 0 && v[1] > 0 && v[2] > 0, "m, n and k must be positive");
            ensure!(v[5..9].iter().all(|&b| b == 0 || b == 1), "concat, mask, arelu and q8 must be 0 or 1");
            ensure!(v[9..14].iter().all(|&x| x > 0), "tile sizes must be positive");
            let t = tile(v[9] as usize, v[10] as usize, v[11] as usize, v[12] as usize, v[13] as usize);
            ensure!(TILES.contains(&t), "tile {}x{}x{} {}x{} is not one of the compiled tiles", t.bm, t.bn, t.bk, t.wm, t.wn);
            ensure!(v[14] >= 1, "splits must be at least 1, got {}", v[14]);
            if f[0] != prec.name() {
                return Ok(None);
            }
            let key = PlanKey { m: v[0], n: v[1], k: v[2], taps: v[3], up: v[4], concat: v[5] != 0, mask: v[6] != 0, arelu: v[7] != 0, q8: v[8] != 0 };
            Ok(Some((key, Plan { tile: t, splits: v[14] as usize })))
        })()
        .with_context(|| format!("plan table line {}: {raw:?}", i + 1))?;
        if let Some((key, plan)) = parsed {
            out.insert(key, plan);
        }
    }
    Ok(out)
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

/// Kernels used by the runtime; GEMM pipelines are specialized lazily per (tile, flags)
/// (`MetalModel` compiles every variant it can use at load time).
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
    /// Split-K partial sums. Its length caps splits * M * N (larger plans use fewer splits).
    pub partial: Buf,
    /// Override for sweeps: always use this plan.
    pub force: Option<Plan>,
    /// Tuned plans for this device (empty on devices without a table; `MB_PLAN_TABLE=0` disables).
    pub table: HashMap<PlanKey, Plan>,
    /// Where `table` came from (`MB_PLAN_TABLE`, the built-in table, or none).
    pub plan_source: String,
    /// GPU cores the fallback plans are sized for (from [`super::GpuInfo`]; settable for tests).
    pub cores: usize,
    /// When set, every GEMM launch appends (params with plan, INT8) here; for profiling tools.
    pub log: RefCell<Option<Vec<(GemmParams, bool)>>>,
    gemm: RefCell<HashMap<(Tile, Flags), Pipeline>>,
    /// While warming up, GEMM pipeline errors are collected here instead of panicking.
    warming: Cell<bool>,
    warm_error: RefCell<Option<anyhow::Error>>,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Self::with_partial_len(gpu, PARTIAL_MAX)
    }

    /// Like [`Kernels::new`] with a split-K partial buffer of `len` float32 elements.
    pub fn with_partial_len(gpu: &Gpu, len: usize) -> Result<Self> {
        let (table, plan_source) = plan_table(gpu)?;
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
            partial: gpu.buffer(len),
            force: None,
            table,
            plan_source,
            cores: gpu.info.cores,
            log: RefCell::new(None),
            gemm: RefCell::new(HashMap::new()),
            warming: Cell::new(false),
            warm_error: RefCell::new(None),
            gpu: gpu.clone(),
        })
    }

    /// Number of GEMM pipeline variants compiled so far.
    pub fn gemm_variants(&self) -> usize {
        self.gemm.borrow().len()
    }

    /// Replace the split-K partial buffer by one of `len` float32 elements.
    pub fn set_partial_len(&mut self, len: usize) {
        self.partial = self.gpu.buffer(len);
    }

    fn pipeline(&self, t: Tile, f: Flags) -> Pipeline {
        if let Some(p) = self.gemm.borrow().get(&(t, f)) {
            return p.clone();
        }
        let name = format!("gemm_mma_{}_{}_{}_{}_{}", t.bm, t.bn, t.bk, t.wm, t.wn);
        let consts = [(0, f.plain), (1, f.conv), (2, f.concat), (3, f.amask), (4, f.arelu), (5, f.up), (6, f.q8), (7, f.vec)];
        match self.gpu.specialized(&name, &consts) {
            Ok(p) => {
                self.gemm.borrow_mut().insert((t, f), p.clone());
                p
            }
            Err(e) if self.warming.get() => {
                // The warm-up recording is discarded, never run: any pipeline stands in.
                self.warm_error.borrow_mut().get_or_insert(e.context(format!("{f:?}")));
                self.reduce.clone()
            }
            Err(e) => panic!("{:#}", e.context(format!("{f:?}"))),
        }
    }

    /// Start collecting GEMM pipeline errors instead of panicking (recordings made until
    /// [`Kernels::end_warm_up`] must be discarded, not committed).
    pub(crate) fn begin_warm_up(&self) {
        self.warming.set(true);
    }

    /// Stop collecting; the first GEMM pipeline error since [`Kernels::begin_warm_up`], if any.
    pub(crate) fn end_warm_up(&self) -> Result<()> {
        self.warming.set(false);
        match self.warm_error.borrow_mut().take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Fallback plan (launches missing from the tuned table): the first preferred tile, among
    /// those wasting at most a fraction of rows to padding, whose grid has enough threadgroups;
    /// then K splits (powers of two, >= 64 K per split) until the grid fills the GPU. Fitted on
    /// `gemm-tune` measurements: FP32 prefers 32x64 tiles, FP16/BF16 16x32 tiles (cheaper to fill
    /// in half precision), INT8 weights 64-row tiles (each tile dequantizes its weights once).
    ///
    /// "Enough" and "fills" are threadgroups per GPU core (8 and 6, or 16 and 12 for INT8 weights,
    /// which were 160/120 and 320/240 on the 20-core GPU of the fit), and the split cap grows with
    /// the core count ([`Kernels::heuristic_split_cap`]). On a 20-core GPU the plans are the
    /// fitted ones; other core counts follow the same rule but have not been measured.
    pub fn heuristic(&self, m: usize, n: usize, k: usize, q8: bool) -> Plan {
        let (prefs, tile_per_core, split_per_core, waste): (&[Tile], usize, usize, f64) = if q8 {
            (&[TILES[6], TILES[0], TILES[3]], 16, 12, 0.4)
        } else if self.gpu.prec == Precision::F32 {
            (&[TILES[4], TILES[0], TILES[3]], 8, 6, 0.4)
        } else {
            (&[TILES[0], TILES[1]], 8, 6, 0.15)
        };
        let (g_tile, g_split) = (tile_per_core * self.cores, split_per_core * self.cores);
        let cap = Self::heuristic_split_cap(self.cores);
        let groups = |t: &Tile| m.div_ceil(t.bm) * n.div_ceil(t.bn);
        let fits: Vec<Tile> = prefs.iter().copied().filter(|t| (m.div_ceil(t.bm) * t.bm) as f64 <= m as f64 * (1.0 + waste)).collect();
        let fits = if fits.is_empty() { vec![*prefs.last().unwrap()] } else { fits };
        let tile = fits.iter().copied().find(|t| groups(t) >= g_tile).unwrap_or(*fits.last().unwrap());
        let mut splits = 1;
        while groups(&tile) * splits < g_split && k / (splits * 2) >= 64 && splits < cap {
            splits *= 2;
        }
        Plan { tile, splits }
    }

    /// Most K splits the fallback plan uses: 8 up to the 20-core reference GPU, then growing in
    /// proportion to the core count (powers of two: 16 for 40 cores, 32 for 80), so that small
    /// launches can still fill larger GPUs.
    pub fn heuristic_split_cap(cores: usize) -> usize {
        (8 * cores).div_ceil(REFERENCE_GPU_CORES).next_power_of_two().max(8)
    }

    /// The plan for a launch: forced, else tuned, else the heuristic.
    pub fn plan_for(&self, p: &GemmParams, q8: bool) -> Plan {
        if let Some(f) = self.force {
            return f;
        }
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        self.table.get(&PlanKey::of(p, q8)).copied().unwrap_or_else(|| self.heuristic(m, n, k, q8))
    }

    /// Largest number of K splits a plan of this `Kernels` can ask for.
    pub fn max_splits(&self) -> usize {
        self.table.values().map(|p| p.splits).chain(self.force.map(|p| p.splits)).fold(Self::heuristic_split_cap(self.cores), usize::max)
    }

    /// The launch of `p`: parameters with the K split filled in, tile and loader flags.
    fn resolve(&self, p: &GemmParams, q8: bool) -> (GemmParams, Tile, Flags) {
        let mut p = *p;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let mut plan = self.plan_for(&p, q8);
        while plan.splits > 1 && (plan.splits * m * n > self.partial.len() || k / plan.splits < plan.tile.bk) {
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
        // The scalar loader (neither plain nor vector) has no INT8 path.
        assert!(!q8 || vec, "INT8 weights need K, lda and the conv/concat extents to be multiples of 4: {p:?}");
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
        (p, t, flags)
    }

    /// `Y = epilogue(A Wᵀ + b)` with the planned kernel (plus a reduce dispatch when split over K).
    pub fn run(&self, rec: &mut Recorder, g: &GemmCall) {
        let q8 = matches!(g.w, Weight::Q8(_));
        let (p, t, flags) = self.resolve(&g.p, q8);
        let (m, n, splits) = (p.m as usize, p.n as usize, p.splits as usize);
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

    /// Whether [`Kernels::attention`] supports `l` tokens of head dimension `hd`.
    pub fn attention_supported(l: usize, hd: usize) -> bool {
        l <= MAX_ATTENTION_LEN && (MMA_HEAD_DIMS.contains(&hd) || 3 * l * (hd + 1) * 4 <= THREADGROUP_MEMORY)
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
        if let Some(pso) = self.attention_mma.get(&(lp, hd)).filter(|_| l <= MAX_ATTENTION_LEN && !self.scalar_attention) {
            rec.dispatch(pso, &args, (b * h, 1, 1), (lp * 4, 1, 1));
            return;
        }
        // Scalar fallback (not used by the model: its head dims have MMA kernels).
        assert!(l <= MAX_ATTENTION_LEN && h > 0 && d % h == 0, "attention: {l} tokens, d {d}, {h} heads unsupported (<= {MAX_ATTENTION_LEN} tokens)");
        let smem = 3 * l * (hd + 1) * 4;
        assert!(smem <= THREADGROUP_MEMORY, "scalar attention: {l} tokens of head dim {hd} need {smem} bytes of threadgroup memory");
        assert!(l * 32 <= self.attention.maxTotalThreadsPerThreadgroup(), "scalar attention: {l} tokens need {} threads per threadgroup", l * 32);
        rec.threadgroup_memory(smem, 0);
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

    /// Whether these buffers hold `rows` tokens of a transformer of width `d` and FFN width `ffn`.
    pub fn fits(&self, rows: usize, d: usize, ffn: usize) -> bool {
        self.x2.len() >= rows * d && self.qkv.len() >= rows * 3 * d && self.attn.len() >= rows * d && self.proj.len() >= rows * d && self.ff.len() >= rows * ffn
    }
}

impl Transformer {
    /// `int8`: quantize the four projections of every layer (as the Python port's `quantize(8)`).
    pub fn load(gpu: &Gpu, tensors: &HashMap<String, Array>, prefix: &str, n_layers: usize, heads: usize, int8: bool) -> Result<Self> {
        let d = tensor(tensors, &format!("{prefix}.layers.0.norm1.weight"))?.dim(0) as usize;
        let ffn = tensor(tensors, &format!("{prefix}.layers.0.linear1.bias"))?.dim(0) as usize;
        if heads == 0 || d % heads != 0 {
            bail!("{prefix}: width {d} is not divisible into {heads} heads");
        }
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
        Ok(Transformer { layers, d, heads, ffn })
    }

    /// In place on `x` (batch * len rows of d). `pad`: (batch, len) float, 1 = padded key.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, k: &Kernels, rec: &mut Recorder, x: &Buf, pad: Option<&Buf>, batch: usize, len: usize, s: &Scratch) {
        let (rows, d, f) = (batch * len, self.d, self.ffn);
        for l in &self.layers {
            k.linear(rec, x, &l.in_w, &l.in_b, &s.qkv, rows, 3 * d, d, ACT_NONE);
            k.attention(rec, &s.qkv, pad, &s.attn, batch, len, d, self.heads);
            k.linear(rec, &s.attn, &l.out_w, &l.out_b, &s.proj, rows, d, d, ACT_NONE);
            k.add_layernorm(rec, x, &s.proj, &l.n1_w, &l.n1_b, &s.x2, rows, d);
            k.linear(rec, &s.x2, &l.l1_w, &l.l1_b, &s.ff, rows, f, d, ACT_RELU);
            k.linear(rec, &s.ff, &l.l2_w, &l.l2_b, &s.proj, rows, d, f, ACT_NONE);
            k.add_layernorm(rec, &s.x2, &s.proj, &l.n2_w, &l.n2_b, x, rows, d);
        }
    }
}
