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
use motionbricks_rs::npy::Npy;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, Precision, compiled, convert, examples};

fn load(name: &str, precision: &str) -> Option<MotionBricks> {
    let dir = checkpoints(name)?;
    Some(MotionBricks::load(name, dir, precision).expect("model loads"))
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
    let out = run(&model, &request(name, 2, 0), true);
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
    let request = request(name, 3, 5);
    let (eager, fused) = (run(&model, &request, false), run(&model, &request, true));
    for k in 0..eager.arrays.len() {
        let e = max_abs(&eager.values(k), &fused.values(k), |_| true);
        assert!(e < 1e-5, "output {k}: max abs {e}");
        // Rust-only, stricter than test_model.py: the explicit attention fallback in layers.rs
        // exists to make compiled runs bitwise equal to eager ones (MLX 0.32.2 printed scalar
        // constants into compiled kernels one ulp off), so guard that exactly.
        let (a, b) = (eager.values(k), fused.values(k));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "output {k}: compiled differs from eager");
    }
}

fn batch_rows_are_independent(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let model = Rc::new(model);
    let request = request(name, 4, 7);
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
    check_restricted(&run(&model, &restricted_request(name), true).tokens());
}

fn forced_duration(name: &str) {
    let _gpu = gpu_lock();
    let Some(model) = load(name, "float32") else { return };
    let args = model.prepare(&request(name, 2, 0)).unwrap();
    let out = model.forward_with(&args, Some(&Array::from_slice(&[7i32, 13], &[2]))).unwrap();
    assert_eq!(to_vec_i32(&out[1]).unwrap(), vec![7, 13]);
    let checked = model.forward_with_durations(&args, &[7, 13]).unwrap();
    assert_eq!(to_vec_i32(&checked[1]).unwrap(), vec![7, 13]);
    assert!(model.forward_with_durations(&args, &[5, 13]).is_err(), "duration 5 is outside 6..=16");
    assert!(model.forward_with_durations(&args, &[7]).is_err(), "one duration per request");
}

/// Shared by the FP16/BF16 and INT8 tests (see `common::tracks_fp32`), plus finite outputs.
fn check_tracks_fp32(fp32: &Out, low: &Out, check_motion: bool) {
    assert!(low.finite(), "non-finite outputs");
    tracks_fp32(&fp32.tokens(), &low.tokens(), &fp32.values(0), &low.values(0), check_motion);
}

fn low_precision_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let fp32 = Rc::new(fp32);
    let request = request(name, 8, 3);
    let reference = run(&fp32, &request, true);
    for precision in [Precision::Float16, Precision::BFloat16] {
        let low = Rc::new(load(name, precision.name()).unwrap());
        assert_eq!(low.core.dtype, precision.compute_dtype());
        check_tracks_fp32(&reference, &run(&low, &request, true), true);
    }
}

fn converted_weights_round_trip(name: &str) {
    let _gpu = gpu_lock();
    let Some(original) = load(name, "float32") else { return };
    let original = Rc::new(original);
    let dir = TempDir::new(&format!("test-{name}"));
    let path = convert(name, checkpoints(name).unwrap(), dir.0.join("w.safetensors"), Dtype::Float32).unwrap();
    let reloaded = Rc::new(MotionBricks::from_converted(&path, Dtype::Float32).unwrap());
    drop(dir);
    let request = request(name, 2, 4);
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
    assert!(run(&model, &request(name, 2, 9), true).finite());
}

fn quantized_8bit_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let fp32 = Rc::new(fp32);
    let mut q8 = load(name, "int8").unwrap();
    assert_eq!(q8.quantization, Some((8, 64)));
    assert!(q8.quantize(8, 64).is_err(), "quantizing twice is refused");
    let q8 = Rc::new(q8);
    let request = request(name, 16, 21);
    check_tracks_fp32(&run(&fp32, &request, true), &run(&q8, &request, true), false);
}

/// Malformed requests are errors (not panics or out-of-bounds kernel reads): wrong input
/// count, shapes, batch sizes, inconsistent arrays, unknown names, zero MaskGIT steps.
fn malformed_inputs_are_errors(name: &str) {
    let _gpu = gpu_lock();
    let Some(mut model) = load(name, "float32") else { return };
    let good = request(name, 2, 0);
    let args = model.prepare(&good).unwrap();
    assert!(model.forward(&args).is_ok());
    assert!(model.forward(&args[1..]).is_err(), "missing input");
    for k in 0..args.len() {
        let mut bad = args.clone();
        let shape = bad[k].shape().to_vec();
        bad[k] = mlx_rs::ops::zeros_dtype(&[shape[0], shape[1] + 1], bad[k].dtype()).unwrap();
        assert!(model.forward(&bad).is_err(), "{}: wrong width", model.input_names()[k]);
        let mut bad = args.clone();
        let mut shape = shape.clone();
        shape[0] = 3;
        bad[k] = mlx_rs::ops::zeros_dtype(&shape, bad[k].dtype()).unwrap();
        assert!(model.forward(&bad).is_err(), "{}: other batch size", model.input_names()[k]);
    }
    assert!(model.forward_with(&args, Some(&Array::from_slice(&[7i32, 8, 9], &[3]))).is_err(), "one duration per request");
    let mut inconsistent = good.clone();
    let first = model.input_names()[0];
    inconsistent.get_mut(first).unwrap().shape[0] = 3;
    assert!(model.prepare(&inconsistent).is_err(), "shape and data disagree");
    let mut missing = good.clone();
    missing.remove(first);
    assert!(model.prepare(&missing).is_err());
    assert!(examples::request("smoa", 1, 0).is_err());
    model.num_inference_steps = 0;
    assert!(model.forward(&args).is_err(), "zero MaskGIT steps");
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
    malformed_inputs_are_errors,
);
