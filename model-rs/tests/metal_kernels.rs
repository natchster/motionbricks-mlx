// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Weight-free unit tests of the Metal runtime's kernels against CPU references. These have no
//! Python counterpart (the Python port runs MLX's kernels); they pin down every specialization
//! the plan table can pick:
//! * `gemm_mma`: every tile and split-K count, over the plain, unaligned (vector) and scalar
//!   loaders; implicit dilated convs with 2x upsampling, input masks and ReLU; concatenated
//!   inputs; INT8 weights; every epilogue (activations, residual with row modulus, output mask,
//!   predicated rows, missing-value blend).
//! * a full post-norm transformer layer (both attention kernels, add+LayerNorm, all precisions)
//!   against the float64 PyTorch-semantics reference shared with `layers.rs`.

mod common;

use std::collections::HashMap;

use common::*;
use mlx_rs::ops::{DequantizeOptions, QuantizeOptions, dequantize, quantize};
use mlx_rs::Array;
use motionbricks_rs::metal::transformer::{
    ACT_LEAKY, ACT_RELU, GemmCall, GemmParams, Kernels, M4_PRO_PLANS, Plan, PlanKey, QWeight, Scratch, TILES, Transformer, Weight,
    parse_plans,
};
use motionbricks_rs::metal::{Buf, Elem, Gpu, Precision, Recorder, wait};
use motionbricks_rs::ops::to_vec_f32;

const DQ: DequantizeOptions = DequantizeOptions { mode: mlx_rs::ops::QuantizationMode::Affine, group_size: Some(64), bits: Some(8) };

/// Inputs of one GEMM launch, in float32 (the CPU reference reads these).
struct Case {
    name: &'static str,
    p: GemmParams,
    a: Vec<f32>,
    a2: Vec<f32>,
    a3: Vec<f32>,
    w: Vec<f32>, // dense [N][K] (for INT8: the dequantized weights)
    bias: Vec<f32>,
    r: Vec<f32>,
    amask: Vec<f32>,
    ymask: Vec<f32>,
    missing: Vec<f32>,
    q8: bool,
}

fn activate(v: f32, act: i32) -> f32 {
    match act {
        1 => v.max(0.0),
        2 => (v * 0.01).max(v),
        _ => v,
    }
}

/// `A(m, k)` as `gemm_mma` reads it (kernels.metal: GemmParams semantics).
fn a_value(c: &Case, m: usize, k: usize) -> f32 {
    let p = &c.p;
    let (v, mask) = if p.conv_taps > 1 {
        let (t_out, cin) = (p.conv_t as usize, p.conv_cin as usize);
        let (b, t) = (m / t_out, m % t_out);
        let (tap, ch) = (k / cin, k % cin);
        let ts = t as i32 + (tap as i32 - p.conv_taps / 2) * p.conv_dil;
        if ts < 0 || ts >= t_out as i32 {
            return 0.0;
        }
        let ts = ts as usize;
        let src = if p.conv_up == 2 { b * (t_out / 2) + ts / 2 } else { b * t_out + ts };
        (c.a[src * p.lda as usize + ch], if p.has_mask != 0 { c.amask[b * t_out + ts] } else { 1.0 })
    } else {
        let (k1, k2) = (p.k1 as usize, p.k2 as usize);
        let v = if p.k1 > 0 && k >= k1 + k2 {
            c.a3[m * p.lda3 as usize + k - k1 - k2]
        } else if p.k1 > 0 && k >= k1 {
            c.a2[m * p.lda2 as usize + k - k1]
        } else {
            c.a[m * p.lda as usize + k]
        };
        (v, if p.has_mask != 0 { c.amask[m] } else { 1.0 })
    };
    let v = if p.a_relu != 0 { v.max(0.0) } else { v };
    v * mask
}

/// Expected Y (initialized to `y0`, which predicated rows keep).
fn reference(c: &Case, y0: &[f32]) -> Vec<f32> {
    let p = &c.p;
    let (m_, n_, k_) = (p.m as usize, p.n as usize, p.k as usize);
    let mut y = y0.to_vec();
    for m in 0..m_ {
        if p.y_pred != 0 && c.ymask[m] == 0.0 {
            continue;
        }
        let a: Vec<f64> = (0..k_).map(|k| a_value(c, m, k) as f64).collect();
        for n in 0..n_ {
            let acc: f64 = (0..k_).map(|k| a[k] * c.w[n * k_ + k] as f64).sum();
            let mut v = activate(acc as f32 + c.bias[n], p.act);
            if p.has_res != 0 {
                let row = if p.r_mod > 0 { m % p.r_mod as usize } else { m };
                v += c.r[row * p.ldr as usize + n];
            }
            if p.y_mask != 0 {
                v *= c.ymask[m];
            }
            if p.blend != 0 {
                let h = c.ymask[m];
                v = v * h + c.missing[(m % p.blend_mod as usize) * n_ + n] * (1.0 - h);
            }
            y[m * p.ldy as usize + n] = v;
        }
    }
    y
}

fn base(name: &'static str, m: usize, n: usize, k: usize, rng: &mut Rand) -> Case {
    Case {
        name,
        p: GemmParams::new(m, n, k),
        a: rng.vec(m * k + 64, 1.0),
        a2: Vec::new(),
        a3: Vec::new(),
        w: rng.vec(n * k, 0.2),
        bias: rng.vec(n, 0.5),
        r: rng.vec(m * n, 1.0),
        amask: (0..2 * m).map(|i| (i % 3 != 1) as i32 as f32).collect(),
        ymask: (0..m).map(|i| (i % 4 != 2) as i32 as f32).collect(),
        missing: rng.vec(m * n, 1.0),
        q8: false,
    }
}

fn cases(rng: &mut Rand) -> Vec<Case> {
    let mut v = Vec::new();
    v.push(base("plain aligned", 64, 64, 96, rng));
    v.push(base("plain unaligned (vector loader)", 37, 50, 100, rng));
    v.push(base("plain K=5 (scalar loader)", 20, 24, 5, rng));
    let mut c = base("relu + residual (row modulus) + output mask", 48, 64, 64, rng);
    c.p.act = ACT_RELU;
    c.p.has_res = 1;
    c.p.ldr = 64;
    c.p.r_mod = 16;
    c.p.y_mask = 1;
    v.push(c);
    let mut c = base("leaky relu + predicated rows", 32, 48, 64, rng);
    c.p.act = ACT_LEAKY;
    c.p.y_pred = 1;
    v.push(c);
    let mut c = base("missing-value blend", 32, 32, 32, rng);
    c.p.blend = 1;
    c.p.blend_mod = 8;
    v.push(c);
    let mut c = base("input mask + relu (non-conv)", 32, 64, 64, rng);
    c.p.has_mask = 1;
    c.p.a_relu = 1;
    v.push(c);
    // Implicit Conv1d: 2 sequences x 16 frames, 32 input channels, kernel 3.
    for (name, dil, up, masked) in [("conv k3 dilation 3", 3, 1, false), ("conv k3 dilation 9 + mask + relu", 9, 1, true), ("conv k3 2x upsampled", 1, 2, false)] {
        let mut c = base(name, 32, 48, 96, rng);
        c.p.conv_taps = 3;
        c.p.conv_t = 16;
        c.p.conv_cin = 32;
        c.p.conv_dil = dil;
        c.p.conv_up = up;
        c.p.lda = 32;
        c.p.has_mask = masked as i32;
        c.p.a_relu = masked as i32;
        c.a = rng.vec(32 * 32 + 64, 1.0);
        v.push(c);
    }
    // Concatenated inputs [A | A2 | A3], aligned and unaligned segment boundaries.
    for (name, k1, k2, k3) in [("concat aligned", 32usize, 32usize, 32usize), ("concat unaligned", 20, 24, 8)] {
        let m = 32;
        let mut c = base(name, m, 40, k1 + k2 + k3, rng);
        c.p.k1 = k1 as i32;
        c.p.k2 = k2 as i32;
        c.p.lda = k1 as i32;
        c.p.lda2 = k2 as i32;
        c.p.lda3 = k3 as i32;
        c.a = rng.vec(m * k1, 1.0);
        c.a2 = rng.vec(m * k2, 1.0);
        c.a3 = rng.vec(m * k3, 1.0);
        v.push(c);
    }
    let mut c = base("INT8 weights", 32, 64, 128, rng);
    c.q8 = true;
    v.push(c);
    v
}

/// Quantize like `mlx.nn.quantize` (affine, 8 bits, group 64); returns the GPU weight and the
/// dequantized float weights for the reference.
fn quantized(gpu: &Gpu, w: &[f32], n: usize, k: usize) -> (Weight, Vec<f32>) {
    let wa = Array::from_slice(w, &[n as i32, k as i32]);
    let (q, s, b) = quantize(&wa, QuantizeOptions { group_size: Some(64), bits: Some(8), ..Default::default() }).unwrap();
    let b = b.unwrap();
    let deq = to_vec_f32(&dequantize(&q, &s, &b, DQ).unwrap()).unwrap();
    q.eval().unwrap();
    let qv: Vec<u32> = q.as_slice::<u32>().to_vec();
    let weight = Weight::Q8(QWeight { q: gpu.upload_u32(&qv), scales: gpu.upload(&to_vec_f32(&s).unwrap()), biases: gpu.upload(&to_vec_f32(&b).unwrap()) });
    (weight, deq)
}

fn run_case(k: &mut Kernels, gpu: &Gpu, c: &mut Case, plan: Option<Plan>) -> (Vec<f32>, Vec<f32>) {
    let (m, n, kk) = (c.p.m as usize, c.p.n as usize, c.p.k as usize);
    let w = if c.q8 {
        let (w, deq) = quantized(gpu, &c.w.clone(), n, kk);
        c.w = deq;
        w
    } else {
        Weight::Dense(gpu.upload_t(&c.w))
    };
    let up = |v: &Vec<f32>| gpu.upload_t(if v.is_empty() { &[0.0f32; 4][..] } else { v });
    let (a, a2, a3, bias, r, missing) = (up(&c.a), up(&c.a2), up(&c.a3), up(&c.bias), up(&c.r), up(&c.missing));
    let (amask, ymask) = (gpu.upload(&c.amask), gpu.upload(&c.ymask));
    let y0 = vec![-7.0f32; m * c.p.ldy as usize];
    let y: Buf = gpu.upload_t(&y0);
    let call = GemmCall {
        a: (&a, 0),
        w: &w,
        bias: &bias,
        y: (&y, 0),
        p: c.p,
        amask: Some(&amask),
        res: Some((&r, 0)),
        a2: Some(&a2),
        a3: Some(&a3),
        ymask: Some(&ymask),
        missing: Some(&missing),
    };
    k.force = plan;
    let mut rec = Recorder::new(gpu);
    k.run(&mut rec, &call);
    wait(&rec.commit());
    k.force = None;
    (y.to_f32(y0.len()), reference(c, &y0))
}

#[test]
fn gemm_variants_match_reference_under_every_plan() {
    let _gpu = gpu_lock();
    let gpu = Gpu::new().unwrap();
    let mut k = Kernels::new(&gpu).unwrap();
    let mut rng = Rand::new(11);
    let mut f = Failures::default();
    let mut launches = 0;
    for mut c in cases(&mut rng) {
        let mut plans: Vec<Option<Plan>> = vec![None];
        for t in TILES {
            for splits in [1, 2, 4] {
                plans.push(Some(Plan { tile: t, splits }));
            }
        }
        for plan in plans {
            let (got, want) = run_case(&mut k, &gpu, &mut c, plan);
            let e = rel_err(&got, &want, |_| true);
            launches += 1;
            f.check(e < 2e-6, || format!("{} under {plan:?}: rel err {e}", c.name));
        }
    }
    println!("{launches} GEMM launches checked");
    f.finish();
}

#[test]
fn gemm_half_precision_matches_reference() {
    let _gpu = gpu_lock();
    for prec in [Precision::F16, Precision::BF16] {
        let gpu = Gpu::with_precision(prec).unwrap();
        let mut k = Kernels::new(&gpu).unwrap();
        let mut rng = Rand::new(12);
        let mut f = Failures::default();
        for mut c in cases(&mut rng).into_iter().filter(|c| !c.q8) {
            // Round the inputs to T so only accumulation and output rounding remain.
            let rt = |v: &mut Vec<f32>| *v = prec.decode(&prec.encode(v));
            for v in [&mut c.a, &mut c.a2, &mut c.a3, &mut c.w, &mut c.bias, &mut c.r, &mut c.missing] {
                rt(v);
            }
            let (got, want) = run_case(&mut k, &gpu, &mut c, None);
            let e = rel_err(&got, &want, |_| true);
            let tol = if prec == Precision::F16 { 2e-3 } else { 1e-2 };
            f.check(e < tol, || format!("{prec:?} {}: rel err {e}", c.name));
        }
        f.finish();
    }
}

/// The Metal transformer stack vs the float64 reference: both attention kernels, padded keys,
/// short (9) and long (25) sequences, every precision.
/// The fused GEMM + add + LayerNorm path (`run_ln`, low-latency mode) against the
/// unfused run + add_layernorm on the same buffers, at every precision.
#[test]
fn reduce_ln_matches_unfused() {
    let _gpu = gpu_lock();
    let mut rng = Rand::new(7);
    for precision in ["float32", "float16", "bfloat16", "int8"] {
        let gpu = Gpu::with_precision(Precision::parse(precision).unwrap()).unwrap();
        let k = Kernels::new(&gpu).unwrap();
        for (m, n, kk) in [(9usize, 512usize, 512usize), (16, 1024, 2048), (25, 512, 2048), (16, 512, 1024)] {
            let a = gpu.upload_t(&rng.vec(m * kk, 1.0));
            let w = gpu.upload_t(&rng.vec(n * kk, 0.05));
            let bias = gpu.upload_t(&rng.vec(n, 0.5));
            let x = gpu.upload_t(&rng.vec(m * n, 1.0));
            let ln_w = gpu.upload_t(&rng.vec(n, 1.0).iter().map(|v| v.abs() + 0.5).collect::<Vec<_>>());
            let ln_b = gpu.upload_t(&rng.vec(n, 0.5));
            let y_ref = gpu.upload_t(&vec![0.0f32; m * n]);
            let mid = gpu.upload_t(&vec![0.0f32; m * n]);
            let y_fused = gpu.upload_t(&vec![0.0f32; m * n]);
            let weight = Weight::Dense(w);
            let mut rec = Recorder::new(&gpu);
            k.run(&mut rec, &GemmCall::new(&a, &weight, &bias, &mid, GemmParams::new(m, n, kk)));
            k.add_layernorm(&mut rec, &x, &mid, &ln_w, &ln_b, &y_ref, m, n);
            k.run_ln(&mut rec, &GemmCall::new(&a, &weight, &bias, &y_fused, GemmParams::new(m, n, kk)), &x, &ln_w, &ln_b);
            wait(&rec.commit());
            let (got, want) = (y_fused.to_f32(m * n), y_ref.to_f32(m * n));
            let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            // The fused path keeps the GEMM sum in f32 through the LayerNorm; the unfused
            // path rounds it to the network type first, so the two differ by one rounding of
            // the inputs to LN (up to ~0.4% relative at bf16, on pre-LN values up to ~±10).
            let tol = if precision == "float32" { 1e-4 } else { 6e-2 };
            assert!(e < tol, "{precision} m={m} n={n} k={kk}: max abs {e}");
        }
    }
}

/// Timing harness (run with --ignored --nocapture): the unfused GEMM + add_layernorm pair
/// vs the fused run_ln, at the model's N=1 shapes.
#[test]
#[ignore = "timing only"]
fn time_fused_vs_unfused_pair() {
    let _gpu = gpu_lock();
    let gpu = Gpu::with_precision(Precision::BF16).unwrap();
    let k = Kernels::new(&gpu).unwrap();
    let mut rng = Rand::new(3);
    for (m, n, kk) in [(16usize, 1024usize, 1024usize), (16, 1024, 2048), (9, 512, 512), (25, 512, 2048)] {
        let a = gpu.upload_t(&rng.vec(m * kk, 1.0));
        let w = gpu.upload_t(&rng.vec(n * kk, 0.05));
        let bias = gpu.upload_t(&rng.vec(n, 0.5));
        let x = gpu.upload_t(&rng.vec(m * n, 1.0));
        let ln_w = gpu.upload_t(&rng.vec(n, 1.0));
        let ln_b = gpu.upload_t(&rng.vec(n, 0.5));
        let y = gpu.upload_t(&vec![0.0f32; m * n]);
        let mid = gpu.upload_t(&vec![0.0f32; m * n]);
        let weight = Weight::Dense(w);
        let time = |f: &mut dyn FnMut(&mut Recorder)| -> f64 {
            let mut times = Vec::new();
            for _ in 0..30 {
                let mut rec = Recorder::new(&gpu);
                f(&mut rec);
                let t0 = std::time::Instant::now();
                wait(&rec.commit());
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[times.len() / 2]
        };
        let unfused = time(&mut |rec| {
            k.run(rec, &GemmCall::new(&a, &weight, &bias, &mid, GemmParams::new(m, n, kk)));
            k.add_layernorm(rec, &x, &mid, &ln_w, &ln_b, &y, m, n);
        });
        let fused = time(&mut |rec| {
            k.run_ln(rec, &GemmCall::new(&a, &weight, &bias, &y, GemmParams::new(m, n, kk)), &x, &ln_w, &ln_b);
        });
        eprintln!("m={m} n={n} k={kk}: unfused {unfused:.3} ms, fused {fused:.3} ms");
    }
}

#[test]
fn transformer_layer_matches_reference() {
    let _gpu = gpu_lock();
    let (b, ffn) = (2usize, 256usize);
    for precision in ["float32", "float16", "bfloat16", "int8"] {
        let gpu = Gpu::with_precision(Precision::parse(precision).unwrap()).unwrap();
        let mut k = Kernels::new(&gpu).unwrap();
        // (d, heads, length): head dims 32 and 64 (MMA attention), 16 (scalar attention).
        for (d, heads, l) in [(128usize, 4usize, 9usize), (128, 2, 25), (64, 4, 7)] {
            for scalar in [false, true] {
                k.scalar_attention = scalar;
                let mut rng = Rand::new((d * 31 + l) as u64);
                let w = encoder_weights(&mut rng, 1, d, ffn);
                let tensors: HashMap<String, Array> = to_arrays(&w, d, ffn);
                let tf = Transformer::load(&gpu, &tensors, "t", 1, heads, precision == "int8").unwrap();
                let x = rng.vec(b * l * d, 1.0);
                let pad: Vec<bool> = (0..b * l).map(|i| i / l == 1 && i % l >= l - 3).collect();
                let xb = gpu.upload_t(&x);
                let padb = gpu.upload(&pad.iter().map(|&p| p as i32 as f32).collect::<Vec<_>>());
                let scratch = Scratch::new(&gpu, b * l, d, ffn);
                let mut rec = Recorder::new(&gpu);
                tf.encode(&k, &mut rec, &xb, Some(&padb), b, l, &scratch);
                wait(&rec.commit());
                let got = xb.to_f32(b * l * d);
                // Reference on the weights the runtime actually uses (INT8: dequantized).
                let mut w_ref = w.clone();
                if precision == "int8" {
                    for (name, v) in w_ref.iter_mut() {
                        if name.ends_with(".weight") && !name.contains("norm") {
                            let rows = v.len() / if name.ends_with("linear2.weight") { ffn } else { d };
                            let wa = Array::from_slice(v, &[rows as i32, (v.len() / rows) as i32]);
                            let (q, s, bb) = quantize(&wa, QuantizeOptions { group_size: Some(64), bits: Some(8), ..Default::default() }).unwrap();
                            *v = to_vec_f32(&dequantize(&q, &s, &bb.unwrap(), DQ).unwrap()).unwrap();
                        }
                    }
                }
                let want = reference_layer(&x.iter().map(|&v| v as f64).collect::<Vec<_>>(), &w_ref, 0, b, l, d, ffn, heads, &pad);
                let e = got.iter().zip(&want).map(|(a, c)| (*a as f64 - c).abs()).fold(0f64, f64::max);
                let tol = match precision {
                    "float32" | "int8" => 5e-5,
                    "float16" => 2e-2,
                    _ => 1e-1,
                };
                assert!(e < tol, "{precision} d={d} heads={heads} L={l} scalar attention {scalar}: max abs {e}");
            }
        }
    }
}

/// A plan-table line for `key` with `plan` (the format `gemm-tune` writes).
fn plan_line(prec: &str, m: i32, plan: &str) -> String {
    format!("{prec} {m} 512 512 0 0 0 0 0 0  {plan}")
}

#[test]
fn plan_tables_parse_strictly() {
    let key = |m: i32| PlanKey { m, n: 512, k: 512, taps: 0, up: 0, concat: false, mask: false, arelu: false, q8: false };
    // Comments, blank lines, other precisions; a repeated key keeps its last plan (gemm-tune appends).
    let text = [
        "# header".to_string(),
        String::new(),
        plan_line("float32", 8, "32 32 16 2 2 8  # 1.0 us"),
        plan_line("bfloat16", 8, "16 32 32 1 2 4"),
        plan_line("float32", 16, "16 64 32 1 4 4"),
        plan_line("float32", 8, "16 64 32 1 4 2"),
    ]
    .join("\n");
    let t = parse_plans(&text, Precision::F32).unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(t[&key(8)], Plan { tile: TILES[1], splits: 2 }, "last entry wins");
    assert_eq!(t[&key(16)], Plan { tile: TILES[1], splits: 4 });
    assert_eq!(parse_plans(&text, Precision::BF16).unwrap()[&key(8)], Plan { tile: TILES[0], splits: 4 });
    assert!(parse_plans(&text, Precision::F16).unwrap().is_empty());
    // Malformed lines are errors naming the line, in any precision.
    for (bad, why) in [
        (plan_line("float32", 8, "32 32 16 2 2 0"), "splits must be at least 1"),
        (plan_line("float32", 8, "32 32 16 2 2 -2"), "splits must be at least 1"),
        (plan_line("float16", 8, "48 32 16 2 2 1"), "not one of the compiled tiles"),
        (plan_line("float32", 8, "32 32 16 2 2"), "expected 16 fields"),
        (plan_line("float32", 8, "32 32 16 2 x 1"), "not an integer"),
        (plan_line("int8", 8, "32 32 16 2 2 1"), "unknown precision"),
        ("float32 8 512 512 0 0 2 0 0 0  32 32 16 2 2 1".to_string(), "must be 0 or 1"),
        ("float32 0 512 512 0 0 0 0 0 0  32 32 16 2 2 1".to_string(), "must be positive"),
    ] {
        let text = format!("# ok\n{}\n{bad}\n", plan_line("float32", 8, "32 32 16 2 2 1"));
        let e = format!("{:#}", parse_plans(&text, Precision::F32).unwrap_err());
        assert!(e.contains("line 3") && e.contains(why), "{bad:?}: {e}");
    }
}

#[test]
fn built_in_plan_table_parses_completely() {
    for prec in [Precision::F32, Precision::F16, Precision::BF16] {
        let lines = M4_PRO_PLANS.lines().filter(|l| l.split_whitespace().next() == Some(prec.name())).count();
        let table = parse_plans(M4_PRO_PLANS, prec).unwrap();
        assert!(lines > 200, "{prec:?}: {lines} lines");
        assert_eq!(table.len(), lines, "{prec:?}: every line is a distinct launch class");
    }
}

#[test]
fn attention_support_matches_the_kernels() {
    for (l, hd, ok) in [(9, 32, true), (25, 32, true), (16, 64, true), (32, 64, true), (7, 16, true), (33, 32, false), (25, 128, false)] {
        assert_eq!(Kernels::attention_supported(l, hd), ok, "L={l} head dim {hd}");
    }
}

#[test]
fn int8_weights_refuse_the_scalar_loader() {
    let _gpu = gpu_lock();
    let gpu = Gpu::new().unwrap();
    let k = Kernels::new(&gpu).unwrap();
    let (m, n, kk) = (8, 64, 64);
    let w = Weight::Q8(QWeight { q: gpu.upload_u32(&vec![0u32; n * kk / 4]), scales: gpu.upload(&vec![0.0; n]), biases: gpu.upload(&vec![0.0; n]) });
    let (a, bias, y) = (gpu.buffer_t(m * 66), gpu.buffer_t(n), gpu.buffer_t(m * n));
    let mut p = GemmParams::new(m, n, kk);
    p.lda = 66; // not a multiple of 4: only the scalar loader could read A, and it has no INT8 path
    let mut rec = Recorder::new(&gpu);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.run(&mut rec, &GemmCall::new(&a, &w, &bias, &y, p))));
    assert!(r.is_err(), "INT8 launch with an unaligned A was accepted");
    assert_eq!(rec.dispatches, 0);
    rec.discard();
}

#[test]
fn buffers_check_element_type_and_bounds() {
    let _gpu = gpu_lock();
    let gpu = Gpu::with_precision(Precision::BF16).unwrap();
    let ints = gpu.upload_i32(&[7, -3, 1 << 30]);
    assert_eq!(ints.kind(), Elem::I32);
    assert_eq!(ints.to_i32(3), vec![7, -3, 1 << 30]);
    let t = gpu.upload_t(&[1.0, 2.5, -0.375]);
    assert_eq!((t.kind(), t.len()), (Elem::T(Precision::BF16), 3));
    assert_eq!(t.to_f32(3), vec![1.0, 2.5, -0.375]);
    let f = gpu.buffer(4);
    f.write_f32_at(2, &[5.0, 6.0]);
    f.fill_f32(0, 2, [1.0, 2.0].into_iter());
    assert_eq!(f.to_f32(4), vec![1.0, 2.0, 5.0, 6.0]);
    let refused = |f: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
    assert!(refused(&|| ints.write_f32(&[1.0])), "float write into an int32 buffer");
    assert!(refused(&|| drop(t.to_i32(1))), "int read of a T buffer");
    assert!(refused(&|| f.write_f32_at(3, &[1.0, 2.0])), "write past the end");
    assert!(refused(&|| drop(f.to_f32(5))), "read past the end");
    assert!(refused(&|| f.fill_f32(0, 3, [1.0].into_iter())), "short iterator");
}

/// The fallback rule with explicit threadgroup targets and split cap. On the 20-core M4 Pro it was
/// fitted with 160/120 (INT8 weights 320/240) and at most 8 splits.
fn heuristic_reference(prec: Precision, (m, n, k): (usize, usize, usize), q8: bool, targets: (usize, usize, usize, usize), cap: usize) -> Plan {
    let (prefs, g_tile, g_split, waste): (&[_], usize, usize, f64) = if q8 {
        (&[TILES[6], TILES[0], TILES[3]], targets.2, targets.3, 0.4)
    } else if prec == Precision::F32 {
        (&[TILES[4], TILES[0], TILES[3]], targets.0, targets.1, 0.4)
    } else {
        (&[TILES[0], TILES[1]], targets.0, targets.1, 0.15)
    };
    let groups = |t: &motionbricks_rs::metal::transformer::Tile| m.div_ceil(t.bm) * n.div_ceil(t.bn);
    let fits: Vec<_> = prefs.iter().copied().filter(|t| (m.div_ceil(t.bm) * t.bm) as f64 <= m as f64 * (1.0 + waste)).collect();
    let fits = if fits.is_empty() { vec![*prefs.last().unwrap()] } else { fits };
    let tile = fits.iter().copied().find(|t| groups(t) >= g_tile).unwrap_or(*fits.last().unwrap());
    let mut splits = 1;
    while groups(&tile) * splits < g_split && k / (splits * 2) >= 64 && splits < cap {
        splits *= 2;
    }
    Plan { tile, splits }
}

#[test]
fn fallback_plans_scale_with_the_gpu_core_count() {
    let _gpu = gpu_lock();
    let shapes: Vec<(usize, usize, usize)> = [1usize, 8, 24, 64, 200, 512, 2048]
        .iter()
        .flat_map(|&m| [64usize, 256, 512, 1024, 2048].into_iter().flat_map(move |n| [64usize, 268, 512, 2048, 4096].into_iter().map(move |k| (m, n, k))))
        .collect();
    for prec in [Precision::F32, Precision::F16, Precision::BF16] {
        let mut k = Kernels::new(&Gpu::with_precision(prec).unwrap()).unwrap();
        for q8 in [false, true].into_iter().filter(|&q| !q || prec == Precision::F32) {
            for &s in &shapes {
                // 20 cores: exactly the fitted rule with its absolute targets.
                k.cores = 20;
                assert_eq!(k.heuristic(s.0, s.1, s.2, q8), heuristic_reference(prec, s, q8, (160, 120, 320, 240), 8), "{prec:?} q8={q8} {s:?}");
                // Other core counts: 8 / 6 threadgroups per core (16 / 12 for INT8 weights).
                for cores in [8, 10, 16, 40, 60, 80] {
                    k.cores = cores;
                    let targets = (8 * cores, 6 * cores, 16 * cores, 12 * cores);
                    let want = heuristic_reference(prec, s, q8, targets, Kernels::heuristic_split_cap(cores));
                    assert_eq!(k.heuristic(s.0, s.1, s.2, q8), want, "{prec:?} q8={q8} {s:?} at {cores} cores");
                }
            }
        }
        // A batch-1 projection fills a large GPU with more K splits.
        let (s20, s80) = {
            k.cores = 20;
            let a = k.heuristic(8, 512, 2048, false).splits;
            k.cores = 80;
            (a, k.heuristic(8, 512, 2048, false).splits)
        };
        assert!(s80 > s20, "{prec:?}: 80 cores use {s80} splits, 20 cores {s20}");
    }
    for (cores, cap) in [(8, 8), (10, 8), (16, 8), (20, 8), (30, 16), (40, 16), (60, 32), (80, 32)] {
        assert_eq!(Kernels::heuristic_split_cap(cores), cap, "{cores} cores");
    }
}

#[test]
fn gpu_info_describes_this_gpu() {
    let _gpu = gpu_lock();
    let gpu = Gpu::new().unwrap();
    let info = &gpu.info;
    assert!(info.apple_family >= 7, "{info:?}");
    assert!(info.architecture.starts_with("applegpu_"), "{info:?}");
    assert!(info.max_threadgroup_memory >= motionbricks_rs::metal::THREADGROUP_MEMORY, "{info:?}");
    assert!((1..=1024).contains(&info.cores), "{info:?}");
    if std::env::var_os("MB_GPU_CORES").is_none() {
        // Apple silicon publishes the core count in the IORegistry (`gpu-core-count`).
        assert_eq!(info.cores_source, "IORegistry", "{info:?}");
    }
    let k = Kernels::new(&gpu).unwrap();
    assert_eq!(k.cores, info.cores);
    assert!(k.max_splits() >= Kernels::heuristic_split_cap(info.cores));
    assert!(!k.plan_source.is_empty());
}
