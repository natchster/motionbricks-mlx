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
    ACT_LEAKY, ACT_RELU, GemmCall, GemmParams, Kernels, Plan, QWeight, Scratch, TILES, Transformer, Weight,
};
use motionbricks_rs::metal::{Buf, Gpu, Precision, Recorder, wait};
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
