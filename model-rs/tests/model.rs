// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Behavioural tests of the full mlx-rs model (need the checkpoints, not the sandbox): the Rust
//! counterpart of `model/tests/test_model.py`, test for test, with the same requests, thresholds
//! and checks. Requests come from `motionbricks_rs::examples` (same structure as the Python
//! generator, different random values). Each test runs for both models.

mod common;

use std::collections::HashMap;
use std::rc::Rc;

use common::*;
use mlx_rs::{Array, Dtype};
use motionbricks_rs::npy::{Npy, NpyData};
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled, convert, examples, parse_dtype};

fn load(name: &str, dtype: &str) -> Option<MotionBricks> {
    let dir = checkpoints(name)?;
    Some(MotionBricks::from_checkpoints(name, dir, parse_dtype(dtype).unwrap()).expect("model loads"))
}

struct Out {
    arrays: Vec<Array>,
}

impl Out {
    fn values(&self, k: usize) -> Vec<f32> {
        to_vec_f32(&self.arrays[k]).unwrap()
    }
    fn tokens(&self) -> Vec<i32> {
        to_vec_i32(&self.arrays[1]).unwrap()
    }
    fn finite(&self) -> bool {
        self.arrays.iter().all(|a| to_vec_f32(&a.as_dtype(Dtype::Float32).unwrap()).unwrap().iter().all(|v| v.is_finite()))
    }
}

fn run(model: &Rc<MotionBricks>, request: &HashMap<String, Npy>, compiled_graph: bool) -> Out {
    let args = model.prepare(request).unwrap();
    let arrays = if compiled_graph { compiled(model.clone())(&args).unwrap() } else { model.forward(&args).unwrap() };
    mlx_rs::transforms::eval(arrays.iter()).unwrap();
    Out { arrays }
}

fn interface_matches_onnx_contract(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let model = Rc::new(model);
    let out = run(&model, &examples::request(name, 2, 0), true);
    if name == "unitree_g1" {
        assert_eq!(out.arrays[0].shape(), &[2, 64, 36]);
        assert_eq!(out.arrays[2].shape(), &[2, 64, 4]);
    } else {
        assert_eq!(out.arrays[0].shape(), &[2, 64, 366]);
    }
    assert_eq!(out.arrays[1].dtype(), Dtype::Int32);
    assert_eq!(out.arrays[1].shape(), &[2]);
    assert!(out.finite());
}

fn compiled_equals_eager(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let model = Rc::new(model);
    let request = examples::request(name, 3, 5);
    let (eager, fused) = (run(&model, &request, false), run(&model, &request, true));
    for k in 0..eager.arrays.len() {
        let e = max_abs(&eager.values(k), &fused.values(k), |_| true);
        assert!(e < 1e-5, "output {k}: max abs {e}");
    }
}

fn batch_rows_are_independent(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let model = Rc::new(model);
    let request = examples::request(name, 4, 7);
    let batched = run(&model, &request, true);
    for i in 0..4 {
        let single = run(&model, &examples::rows(&request, i, i + 1), true);
        assert_eq!(batched.tokens()[i], single.tokens()[0], "row {i}: duration");
        // Different batch sizes pick different matmul tilings (accumulation order), so allow
        // float32 noise at the level of the PyTorch parity tolerance.
        for k in 0..batched.arrays.len() {
            let all = batched.values(k);
            let per = all.len() / 4;
            let e = max_abs(&all[i * per..(i + 1) * per], &single.values(k), |_| true);
            assert!(e < 2e-3, "row {i}, output {k}: max abs {e}");
        }
    }
}

fn allowed_token_counts_are_respected(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let model = Rc::new(model);
    let mut request = examples::request(name, 6, 11);
    let mut allowed = vec![0i32; 6 * 11];
    for i in 0..6 {
        allowed[i * 11 + i] = 1;
        allowed[i * 11 + 10 - i] = 1;
    }
    request.insert("allowed_pred_num_tokens".into(), Npy { shape: vec![6, 11], data: NpyData::I32(allowed) });
    for (i, count) in run(&model, &request, true).tokens().into_iter().enumerate() {
        assert!([i as i32, 10 - i as i32].contains(&(count - 6)), "row {i}: {count} tokens not allowed");
    }
}

fn forced_duration(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let args = model.prepare(&examples::request(name, 2, 0)).unwrap();
    let out = model.forward_with(&args, Some(&Array::from_slice(&[7i32, 13], &[2]))).unwrap();
    assert_eq!(to_vec_i32(&out[1]).unwrap(), vec![7, 13]);
}

/// Shared by the FP16/BF16 and INT8 tests: >= 75% of durations match FP32; outputs finite; and
/// (`check_motion`) the median deviation stays below 0.05 where the duration matches.
fn tracks_fp32(name: &str, fp32: &Out, low: &Out, check_motion: bool) {
    let (a, b) = (fp32.tokens(), low.tokens());
    let same: Vec<bool> = a.iter().zip(&b).map(|(x, y)| x == y).collect();
    let agree = same.iter().filter(|&&s| s).count();
    assert!(agree * 4 >= same.len() * 3, "only {agree}/{} durations match FP32", same.len());
    assert!(low.finite(), "non-finite outputs");
    if check_motion {
        let (x, y) = (fp32.values(0), low.values(0));
        let per = x.len() / same.len();
        let diffs: Vec<f32> = (0..x.len()).filter(|k| same[k / per]).map(|k| (x[k] - y[k]).abs()).collect();
        let m = median(diffs);
        assert!(m < 0.05, "median |diff| {m} where durations match");
    }
    let _ = name;
}

fn low_precision_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let fp32 = Rc::new(fp32);
    let request = examples::request(name, 8, 3);
    let reference = run(&fp32, &request, true);
    for dtype in ["float16", "bfloat16"] {
        let low = Rc::new(load(name, dtype).unwrap());
        assert_eq!(low.core.dtype, parse_dtype(dtype).unwrap());
        tracks_fp32(name, &reference, &run(&low, &request, true), true);
    }
}

fn converted_weights_round_trip(name: &str) {
    let _gpu = gpu_lock();
    let Some(original) = load(name, "float32") else { return };
    let original = Rc::new(original);
    let dir = std::env::temp_dir().join(format!("motionbricks-rs-test-{}-{name}", std::process::id()));
    let path = convert(name, checkpoints(name).unwrap(), dir.join("w.safetensors"), Dtype::Float32).unwrap();
    let reloaded = Rc::new(MotionBricks::from_converted(&path, Dtype::Float32).unwrap());
    std::fs::remove_dir_all(&dir).ok();
    let request = examples::request(name, 2, 4);
    let (a, b) = (run(&original, &request, true), run(&reloaded, &request, true));
    for k in 0..a.arrays.len() {
        assert_eq!(a.values(k), b.values(k), "output {k} differs after the round trip");
    }
}

fn multi_step_decoding_runs(name: &str) {
    let _gpu = gpu_lock();
    let Some(mut model) = load(name, "float32") else { return };
    model.num_inference_steps = 3;
    let model = Rc::new(model);
    assert!(run(&model, &examples::request(name, 2, 9), true).finite());
}

fn quantized_8bit_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let fp32 = Rc::new(fp32);
    let mut q8 = load(name, "float32").unwrap();
    q8.quantize(8, 64).unwrap();
    let q8 = Rc::new(q8);
    let request = examples::request(name, 16, 21);
    tracks_fp32(name, &run(&fp32, &request, true), &run(&q8, &request, true), false);
}

macro_rules! for_both_models {
    ($($t:ident),* $(,)?) => {
        mod unitree_g1 { $( #[test] fn $t() { super::$t("unitree_g1") } )* }
        mod soma { $( #[test] fn $t() { super::$t("soma") } )* }
    };
}

for_both_models!(
    interface_matches_onnx_contract,
    compiled_equals_eager,
    batch_rows_are_independent,
    allowed_token_counts_are_respected,
    forced_duration,
    low_precision_tracks_fp32,
    converted_weights_round_trip,
    multi_step_decoding_runs,
    quantized_8bit_tracks_fp32,
);
