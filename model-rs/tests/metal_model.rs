// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Behavioural tests of the Metal runtime (need the checkpoints, not the sandbox): the
//! counterpart of `model/tests/test_model.py` with the same requests, thresholds and checks.
//! Differences by design:
//! * `mx.compile` = eager becomes "chunked commits = one command buffer, and repeated runs are
//!   bitwise identical": the runtime has no eager/compiled split, but it does split a generation
//!   over several command buffers.
//! * multi-step MaskGIT decoding is outside the runtime's release-only configuration, so it has
//!   no counterpart; the runtime refuses invalid durations and over-sized batches instead.

mod common;

use std::collections::HashMap;

use common::*;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Recorder, wait};
use motionbricks_rs::npy::{Npy, NpyData};
use motionbricks_rs::{convert, examples};

fn load(name: &str, precision: &str) -> Option<MetalModel> {
    let dir = checkpoints(name)?;
    Some(MetalModel::from_checkpoints_with(name, dir, 32, precision).expect("metal model loads"))
}

type Out = (Vec<f32>, Vec<i32>, Option<Vec<f32>>);

fn run(m: &MetalModel, request: &HashMap<String, Npy>) -> Out {
    m.run(request).unwrap()
}

fn finite(out: &Out) -> bool {
    out.0.iter().chain(out.2.iter().flatten()).all(|v| v.is_finite())
}

fn interface_matches_onnx_contract(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let (motion, tokens, contacts) = run(&m, &examples::request(name, 2, 0));
    let dim = if name == "unitree_g1" { 36 } else { 366 };
    assert_eq!(motion.len(), 2 * 64 * dim);
    assert_eq!(contacts.as_ref().map(Vec::len), (name == "unitree_g1").then_some(2 * 64 * 4));
    assert_eq!(tokens.len(), 2);
    assert!(tokens.iter().all(|t| (6..=16).contains(t)));
    assert!(finite(&(motion, tokens, contacts)));
}

fn chunked_commits_equal_one_command_buffer(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let request = examples::request(name, 3, 5);
    let chunked = run(&m, &request);
    let n = m.upload_inputs(&request).unwrap();
    let mut rec = Recorder::new(&m.gpu);
    m.encode(&mut rec, &m.dims(n, false));
    wait(&rec.commit());
    let single = m.outputs(n);
    assert_eq!(chunked, single, "chunked commits changed the result");
    assert_eq!(run(&m, &request), chunked, "repeated runs differ");
}

fn batch_rows_are_independent(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let request = examples::request(name, 4, 7);
    let batched = run(&m, &request);
    for i in 0..4 {
        let single = run(&m, &examples::rows(&request, i, i + 1));
        assert_eq!(batched.1[i], single.1[0], "row {i}: duration");
        // Different batch sizes use different GEMM plans (accumulation order): float32 noise.
        let per = batched.0.len() / 4;
        let e = max_abs(&batched.0[i * per..(i + 1) * per], &single.0, |_| true);
        assert!(e < 2e-3, "row {i}: max abs {e}");
        if let (Some(a), Some(b)) = (&batched.2, &single.2) {
            let per = a.len() / 4;
            let e = max_abs(&a[i * per..(i + 1) * per], b, |_| true);
            assert!(e < 2e-3, "row {i}: contacts max abs {e}");
        }
    }
}

fn allowed_token_counts_are_respected(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let mut request = examples::request(name, 6, 11);
    let mut allowed = vec![0i32; 6 * 11];
    for i in 0..6 {
        allowed[i * 11 + i] = 1;
        allowed[i * 11 + 10 - i] = 1;
    }
    request.insert("allowed_pred_num_tokens".into(), Npy { shape: vec![6, 11], data: NpyData::I32(allowed) });
    for (i, count) in run(&m, &request).1.into_iter().enumerate() {
        assert!([i as i32, 10 - i as i32].contains(&(count - 6)), "row {i}: {count} tokens not allowed");
    }
}

fn forced_duration(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let request = examples::request(name, 2, 0);
    let out = m.run_with_durations(&request, Some(&[7, 13])).unwrap();
    assert_eq!(out.1, vec![7, 13]);
    assert!(finite(&out));
    assert!(m.run_with_durations(&request, Some(&[5, 13])).is_err(), "duration 5 is outside 6..=16");
    assert!(m.run_with_durations(&request, Some(&[7])).is_err(), "one duration per request");
}

/// >= 75% of durations match FP32; outputs finite; with `check_motion`, the median deviation
/// stays below 0.05 where the duration matches (as in test_model.py).
fn tracks_fp32(fp32: &Out, low: &Out, check_motion: bool) {
    let same: Vec<bool> = fp32.1.iter().zip(&low.1).map(|(a, b)| a == b).collect();
    let agree = same.iter().filter(|&&s| s).count();
    assert!(agree * 4 >= same.len() * 3, "only {agree}/{} durations match FP32", same.len());
    assert!(finite(low), "non-finite outputs");
    if check_motion {
        let per = fp32.0.len() / same.len();
        let diffs: Vec<f32> = (0..fp32.0.len()).filter(|k| same[k / per]).map(|k| (fp32.0[k] - low.0[k]).abs()).collect();
        let m = median(diffs);
        assert!(m < 0.05, "median |diff| {m} where durations match");
    }
}

fn low_precision_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let request = examples::request(name, 8, 3);
    let reference = run(&fp32, &request);
    for precision in ["float16", "bfloat16"] {
        let low = load(name, precision).unwrap();
        tracks_fp32(&reference, &run(&low, &request), true);
    }
}

fn converted_weights_round_trip(name: &str) {
    let _gpu = gpu_lock();
    let Some(original) = load(name, "float32") else { return };
    let dir = std::env::temp_dir().join(format!("motionbricks-metal-test-{}-{name}", std::process::id()));
    let path = convert(name, checkpoints(name).unwrap(), dir.join("w.safetensors"), mlx_rs::Dtype::Float32).unwrap();
    let reloaded = MetalModel::from_converted(&path, 32, "float32").unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let request = examples::request(name, 2, 4);
    assert_eq!(run(&original, &request), run(&reloaded, &request), "outputs differ after the round trip");
}

fn quantized_8bit_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let q8 = load(name, "int8").unwrap();
    let request = examples::request(name, 16, 21);
    tracks_fp32(&run(&fp32, &request), &run(&q8, &request), false);
}

fn batch_larger_than_max_batch_is_refused(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    let m = MetalModel::from_checkpoints(name, dir, 2).unwrap();
    assert!(m.run(&examples::request(name, 3, 0)).is_err());
    assert!(m.run(&examples::request(name, 2, 0)).is_ok());
}

macro_rules! for_both_models {
    ($($t:ident),* $(,)?) => {
        mod unitree_g1 { $( #[test] fn $t() { super::$t("unitree_g1") } )* }
        mod soma { $( #[test] fn $t() { super::$t("soma") } )* }
    };
}

for_both_models!(
    interface_matches_onnx_contract,
    chunked_commits_equal_one_command_buffer,
    batch_rows_are_independent,
    allowed_token_counts_are_respected,
    forced_duration,
    low_precision_tracks_fp32,
    converted_weights_round_trip,
    quantized_8bit_tracks_fp32,
    batch_larger_than_max_batch_is_refused,
);
