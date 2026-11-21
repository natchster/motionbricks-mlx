// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Weight-free unit tests of the mlx-rs building blocks against float64 re-derivations: the
//! counterpart of `model/tests/test_layers.py`, test for test, plus the attention path for head
//! dims MLX's fused kernels don't cover. Need no checkpoints and no sandbox.

mod common;

use std::collections::HashMap;
use std::f64::consts::PI;

use common::*;
use mlx_rs::{Array, Dtype};
use motionbricks_rs::layers::{Conv1d, DoubleCondDecoder, ResConv1dBlock, TransformerEncoder, sincos_position_encoding};
use motionbricks_rs::networks::{dequantize, overall_to_heads};
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::representation::MotionRepresentation;
use motionbricks_rs::rig;
use motionbricks_rs::weights::Weights;

fn arr(v: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(v, shape)
}

fn vals(a: &Array) -> Vec<f32> {
    to_vec_f32(&a.as_dtype(Dtype::Float32).unwrap()).unwrap()
}

#[test]
fn sincos_matches_formula() {
    let pe = vals(&sincos_position_encoding(16, 8).unwrap());
    for pos in 0..16 {
        for i in 0..4 {
            let div = (-(2.0 * i as f64) * (10000f64.ln() / 8.0)).exp();
            assert!((pe[pos * 8 + 2 * i] as f64 - (pos as f64 * div).sin()).abs() < 1e-6);
            assert!((pe[pos * 8 + 2 * i + 1] as f64 - (pos as f64 * div).cos()).abs() < 1e-6);
        }
    }
}

#[test]
fn transformer_layer_matches_pytorch_semantics() {
    let _gpu = gpu_lock();
    // (d, heads): head dim 16 runs the explicit fallback, 64 MLX's fused kernel.
    for (d, heads) in [(64usize, 4usize), (128, 2)] {
        let (b, l, ffn) = (2usize, 7usize, 96usize);
        let mut rng = Rand::new(d as u64);
        let w = encoder_weights(&mut rng, 1, d, ffn);
        let tf = TransformerEncoder::load(&mut Weights::new(to_arrays(&w, d, ffn), Dtype::Float32), "t", 1, heads as i32).unwrap();
        let x = rng.vec(b * l * d, 1.0);
        let pad: Vec<bool> = (0..b * l).map(|k| k / l == 1 && k % l >= 5).collect();
        let pad_arr = Array::from_slice(&pad, &[b as i32, l as i32]);
        let got = vals(&tf.forward(&arr(&x, &[b as i32, l as i32, d as i32]), Some(&pad_arr)).unwrap());
        let want = reference_layer(&x.iter().map(|&v| v as f64).collect::<Vec<_>>(), &w, 0, b, l, d, ffn, heads, &pad);
        let e = got.iter().zip(&want).map(|(a, c)| (*a as f64 - c).abs()).fold(0f64, f64::max);
        assert!(e < 2e-5, "d={d} heads={heads}: max abs {e}");
    }
}

#[test]
fn resconv_block_masks_and_residual() {
    let _gpu = gpu_lock();
    let mut rng = Rand::new(1);
    let block = ResConv1dBlock {
        conv1: Conv1d::new(arr(&rng.vec(8 * 3 * 8, 0.17), &[8, 3, 8]), arr(&rng.vec(8, 0.17), &[8]), 3, 3),
        conv2: Conv1d::new(arr(&rng.vec(8 * 8, 0.17), &[8, 1, 8]), arr(&rng.vec(8, 0.17), &[8]), 0, 1),
    };
    let x = arr(&rng.vec(2 * 10 * 8, 1.0), &[2, 10, 8]);
    let mask: Vec<f32> = (0..20).map(|k| ((k % 10) < [10, 6][k / 10]) as i32 as f32).collect();
    let y = vals(&block.forward(&x, Some(&arr(&mask, &[2, 10, 1]))).unwrap());
    assert!(y[(10 + 6) * 8..].iter().all(|&v| v == 0.0), "masked frames must be exactly zero");
    // Without masking, a zero block (zero weights) is the identity.
    let zero = |n: usize, k: i32| Conv1d::new(arr(&vec![0.0; n * k as usize * n], &[n as i32, k, n as i32]), arr(&vec![0.0; n], &[n as i32]), if k == 3 { 1 } else { 0 }, 1);
    let identity = ResConv1dBlock { conv1: zero(8, 3), conv2: zero(8, 1) };
    let x_v = vals(&x);
    let y0 = vals(&identity.forward(&x, None).unwrap());
    assert!(y0.iter().zip(&x_v).all(|(a, b)| (a - b).abs() < 1e-6));
}

#[test]
fn decoder_shapes_and_target_override() {
    let _gpu = gpu_lock();
    // DoubleCondDecoder(input 5, code 32, down_t 2, width 16, depth 2, rate 3, target 5, external 3).
    let mut rng = Rand::new(2);
    let mut t: HashMap<String, Array> = HashMap::new();
    let mut add = |name: String, shape: &[i32]| {
        let n: i32 = shape.iter().product();
        t.insert(name, arr(&rng.vec(n as usize, 0.17), shape));
    };
    let conv = |add: &mut dyn FnMut(String, &[i32]), p: &str, out: i32, k: i32, inp: i32| {
        add(format!("{p}.weight"), &[out, k, inp]);
        add(format!("{p}.bias"), &[out]);
    };
    conv(&mut add, "d.model.0", 16, 3, 32);
    for i in 0..2 {
        for j in 0..2 {
            conv(&mut add, &format!("d.model.{}.0.model.{j}.conv1", 2 + i), 16, 3, 16);
            conv(&mut add, &format!("d.model.{}.0.model.{j}.conv2", 2 + i), 16, 1, 16);
        }
        conv(&mut add, &format!("d.model.{}.2", 2 + i), 16, 3, 16);
        let fpp = 1 << (2 - i);
        add(format!("d.external_cond_blocks.{}.weight", 2 * i), &[16, fpp * 3 + 16]);
        add(format!("d.external_cond_blocks.{}.bias", 2 * i), &[16]);
        add(format!("d.target_cond_blocks.{}.weight", 2 * i), &[16 / fpp, 5]);
        add(format!("d.target_cond_blocks.{}.bias", 2 * i), &[16 / fpp]);
    }
    conv(&mut add, "d.model.4", 16, 3, 16);
    conv(&mut add, "d.model.6", 5, 3, 16);
    let dec = DoubleCondDecoder::load(&mut Weights::new(t, Dtype::Float32), "d", 2, 16, 2, 3).unwrap();
    let x = arr(&rng.vec(2 * 16 * 32, 1.0), &[2, 16, 32]);
    let ext = arr(&rng.vec(2 * 64 * 3, 1.0), &[2, 64, 3]);
    let tgt = arr(&rng.vec(2 * 64 * 5, 1.0), &[2, 64, 5]);
    let has = Array::from_slice(&[false; 128], &[2, 64]);
    let mask: Vec<bool> = (0..32).map(|k| (k % 16) < [16, 10][k / 16]).collect();
    let y = dec.forward(&x, &ext, &tgt, &has, &Array::from_slice(&mask, &[2, 16])).unwrap();
    assert_eq!(y.shape(), &[2, 64, 5]);
    let v = vals(&y);
    assert!(v[(64 + 40) * 5..].iter().all(|&x| x == 0.0), "frames after 10 tokens x 4 must be zero");
}

#[test]
fn quantizer_lookup_and_digits() {
    let _gpu = gpu_lock();
    let mut rng = Rand::new(3);
    let embed = rng.vec(3 * 4 * 2, 1.0); // (heads 3, entries 4, width 2)
    let idx: Vec<i32> = (0..2 * 5 * 3).map(|k| (k * 7 + 3) % 4).collect();
    let got = vals(&dequantize(&arr(&embed, &[3, 4, 2]), &Array::from_slice(&idx, &[2, 5, 3])).unwrap());
    let want: Vec<f32> = (0..2 * 5).flat_map(|r| (0..3).flat_map(|h| { let e = idx[r * 3 + h] as usize; embed[(h * 4 + e) * 2..(h * 4 + e) * 2 + 2].to_vec() }).collect::<Vec<_>>()).collect();
    assert_eq!(got, want);
    let overall: Vec<i32> = (0..10).map(|r| (0..3).map(|h| idx[r * 3 + h] * 4i32.pow(h as u32)).sum()).collect();
    let heads = to_vec_i32(&overall_to_heads(&Array::from_slice(&overall, &[2, 5]), 3, 4).unwrap().as_dtype(Dtype::Int32).unwrap()).unwrap();
    assert_eq!(heads, idx);
}

#[test]
fn rotation_round_trips() {
    let _gpu = gpu_lock();
    let mut rng = Rand::new(4);
    let mut q = rng.vec(50 * 4, 1.0);
    for r in q.chunks_mut(4) {
        let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let s = if r[0] < 0.0 { -1.0 } else { 1.0 };
        r.iter_mut().for_each(|v| *v = *v / n * s);
    }
    let m = rig::quat_matrix(&arr(&q, &[50, 4])).unwrap();
    let mut back = vals(&rig::matrix_quat(&m).unwrap());
    for r in back.chunks_mut(4) {
        let s = if r[0] < 0.0 { -1.0 } else { 1.0 };
        r.iter_mut().for_each(|v| *v *= s);
    }
    assert!(max_abs(&back, &q, |_| true) < 1e-5);
    let six = rig::matrix_to_cont6d(&m).unwrap();
    assert!(max_abs(&vals(&rig::cont6d_to_matrix(&six).unwrap()), &vals(&m), |_| true) < 1e-5);
    let w = vals(&rig::wrap(&arr(&[-7.0, -PI as f32, 0.0, PI as f32, 7.0], &[5])).unwrap());
    assert!(w.iter().all(|&v| v >= -PI as f32 && v < PI as f32), "{w:?}");
}

#[test]
fn root_conversion_integrates_back() {
    let _gpu = gpu_lock();
    // global -> local (finite differences) then local -> global (integration) recovers the path.
    let dim = 9 + 87 + 180 + 90 + 4;
    let rep = MotionRepresentation::new(30, &vec![0.0; dim], &vec![1.0; dim]).unwrap();
    let t: Vec<f64> = (0..64).map(|k| 2.0 * k as f64 / 63.0).collect();
    let heading: Vec<f64> = t.iter().map(|v| 0.6 * v.sin()).collect();
    let (mut x, mut z) = (0f64, 0f64);
    let mut root = Vec::new();
    for k in 0..64 {
        x += heading[k].cos() / 30.0;
        z += heading[k].sin() / 30.0;
        root.extend([x, 0.9 + 0.01 * t[k], z, heading[k].cos(), heading[k].sin()]);
    }
    let scale = (1.0f64 + 1e-5).sqrt();
    let rootn: Vec<f32> = root.iter().map(|v| (*v as f32) / scale as f32).collect();
    let local = rep.global_root_to_local(&arr(&rootn, &[1, 64, 5]), &Array::from_slice(&[64i32], &[1])).unwrap();
    let body = mlx_rs::ops::zeros::<f32>(&[1, 64, rep.local.dim - 4]).unwrap();
    let recon = vals(&rep.local_to_global(&mlx_rs::ops::concatenate(&[&local, &body], -1).unwrap()).unwrap());
    let gdim = recon.len() / 64;
    // removing_heading=False: integration returns the path relative to frame 0 and headings
    // relative to the first heading.
    let mut worst = 0f64;
    for k in 0..64 {
        let r = &root[k * 5..k * 5 + 5];
        let want = [r[0] - root[0], r[1], r[2] - root[2], (heading[k] - heading[0]).cos(), (heading[k] - heading[0]).sin()];
        for c in 0..5 {
            worst = worst.max((recon[k * gdim + c] as f64 - want[c]).abs());
        }
    }
    assert!(worst < 1e-4, "max abs {worst}");
}

#[test]
fn low_precision_attention_is_finite() {
    let _gpu = gpu_lock();
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let mut rng = Rand::new(5);
        let w = encoder_weights(&mut rng, 2, 32, 2048);
        let tf = TransformerEncoder::load(&mut Weights::new(to_arrays(&w, 32, 2048), dtype), "t", 2, 4).unwrap();
        let pad: Vec<bool> = (0..16).map(|k| (k % 8) >= [8, 3][k / 8]).collect();
        let y = tf.forward(&arr(&rng.vec(2 * 8 * 32, 1.0), &[2, 8, 32]).as_dtype(dtype).unwrap(), Some(&Array::from_slice(&pad, &[2, 8]))).unwrap();
        assert_eq!(y.dtype(), dtype);
        assert!(vals(&y).iter().all(|v| v.is_finite()), "{dtype:?}: non-finite attention output");
    }
}

#[test]
fn rig_kernels_match_ops() {
    use motionbricks_rs::npy::load_npz_bytes;
    use motionbricks_rs::rig::{G1Rig, SomaRetarget};
    let _gpu = gpu_lock();
    let g1 = load_npz_bytes(include_bytes!("../../model/motionbricks_mlx/assets/g1_model_rig.npz")).unwrap();
    let range = |a: i32, b: i32| (a..b).collect::<Vec<i32>>();
    let rig = G1Rig::new(&g1, &range(104, 308), &range(0, 3), &range(410, 414)).unwrap();
    // Inputs as in test_layers.py: qpos ~ N(0, 0.4) with the pelvis 0.75 up, features ~ N(0, 1).
    let mut rng = motionbricks_rs::examples::Rng::new(3);
    let mut normal = |n: usize, sigma: f32| -> Vec<f32> { (0..n).map(|_| rng.normal(0.0, sigma)).collect() };
    // np.testing.assert_allclose(got, want, atol=2e-5, rtol=1e-5)
    let close = |a: &Array, b: &Array, what: &str| {
        let (x, y) = (vals(a), vals(b));
        let bad = x.iter().zip(&y).filter(|(g, w)| (*g - *w).abs() > 2e-5 + 1e-5 * w.abs()).count();
        assert_eq!(bad, 0, "{what}: {bad} kernel values differ from the ops (max abs {})", max_abs(&x, &y, |_| true));
    };
    let mut qpos = normal(3 * 8 * 36, 0.4);
    for r in qpos.chunks_mut(36) {
        r[2] += 0.75;
    }
    let ctx = arr(&qpos.chunks(36 * 8).flat_map(|b| b[..4 * 36].to_vec()).collect::<Vec<_>>(), &[3, 4, 36]);
    let tgt = arr(&qpos.chunks(36 * 8).flat_map(|b| b[4 * 36..].to_vec()).collect::<Vec<_>>(), &[3, 4, 36]);
    let (a, b) = (rig.preprocess(&ctx, &tgt).unwrap(), rig.preprocess_ops(&ctx, &tgt).unwrap());
    close(&a.0, &b.0, "g1 global root");
    close(&a.1, &b.1, "g1 local root");
    close(&a.2, &b.2, "g1 local poses");
    let feats = arr(&normal(2 * 64 * 414, 1.0), &[2, 64, 414]);
    let (a, b) = (rig.postprocess(&feats).unwrap(), rig.postprocess_ops(&feats).unwrap());
    close(&a.0, &b.0, "g1 qpos");
    close(&a.1, &b.1, "g1 contacts");
    let soma = SomaRetarget::new(&load_npz_bytes(include_bytes!("../../model/motionbricks_mlx/assets/soma_model_rig.npz")).unwrap(), 30).unwrap();
    let six = arr(&normal(2 * 64 * 180, 1.0), &[2, 64, 180]);
    let (cap, std) = soma.offsets();
    close(&soma.capture_to_standard(&six).unwrap(), &soma.apply_ops(&six, cap, std).unwrap(), "soma retarget");

    // The kernels index fixed row widths: other shapes are refused before the launch.
    let wide = arr(&normal(3 * 4 * 37, 0.4), &[3, 4, 37]);
    assert!(rig.preprocess(&wide, &tgt).is_err(), "qpos rows of 37");
    assert!(rig.preprocess(&ctx, &arr(&normal(2 * 4 * 36, 0.4), &[2, 4, 36])).is_err(), "batch mismatch");
    assert!(rig.postprocess(&arr(&normal(64 * 414, 1.0), &[64, 414])).is_err(), "features need (B, T, 414)");
    assert!(rig.postprocess(&arr(&normal(2 * 413, 1.0), &[1, 2, 413])).is_err(), "features need 414 columns");
    assert!(soma.capture_to_standard(&arr(&normal(2 * 179, 1.0), &[2, 179])).is_err(), "rotations need 30 * 6 columns");
    assert!(soma.capture_to_standard(&Array::from_f32(1.0)).is_err(), "0-d rotations");
}
