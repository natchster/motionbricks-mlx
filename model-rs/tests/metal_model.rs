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
//!
//! Runtime-only checks: re-encoding without a new upload, malformed inputs, a batch size the
//! plan table lacks at exactly `max_batch`, and the fitted plan rule against the tuned table.

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
    let (motion, tokens, contacts) = run(&m, &request(name, 2, 0));
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
    let request = request(name, 3, 5);
    let chunked = run(&m, &request);
    let n = m.upload_inputs(&request).unwrap();
    let mut rec = Recorder::new(&m.gpu);
    m.encode(&mut rec, &m.dims(n, false));
    wait(&rec.commit());
    let single = m.outputs(n);
    assert_eq!(chunked, single, "chunked commits changed the result");
    assert_eq!(run(&m, &request), chunked, "repeated runs differ");
}

/// Every row of an `n`-request batch matches that request run alone.
fn check_rows_match_singles(m: &MetalModel, request: &HashMap<String, Npy>, n: usize) {
    let batched = run(m, request);
    for i in 0..n {
        let single = run(m, &examples::rows(request, i, i + 1));
        assert_eq!(batched.1[i], single.1[0], "row {i}: duration");
        // Different batch sizes use different GEMM plans (accumulation order): float32 noise.
        let per = batched.0.len() / n;
        let e = max_abs(&batched.0[i * per..(i + 1) * per], &single.0, |_| true);
        assert!(e < 2e-3, "row {i}: max abs {e}");
        if let (Some(a), Some(b)) = (&batched.2, &single.2) {
            let per = a.len() / n;
            let e = max_abs(&a[i * per..(i + 1) * per], b, |_| true);
            assert!(e < 2e-3, "row {i}: contacts max abs {e}");
        }
    }
}

fn batch_rows_are_independent(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    check_rows_match_singles(&m, &request(name, 4, 7), 4);
}

/// N=5 is not in the plan table (fitted rule, unaligned tiles) and fills the buffers exactly.
fn untuned_batch_at_max_batch(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    let m = MetalModel::from_checkpoints(name, dir, 5).unwrap();
    check_rows_match_singles(&m, &request(name, 5, 13), 5);
}

/// Loading compiles every GEMM variant the plans pick for batch 1..=max_batch: generations
/// compile nothing.
fn generations_compile_no_pipelines(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    let m = MetalModel::from_checkpoints(name, dir, 6).unwrap();
    let compiled = m.k.gemm_variants();
    assert!(compiled > 0);
    for n in 1..=6 {
        run(&m, &request(name, n, n as u64));
    }
    assert_eq!(m.k.gemm_variants(), compiled, "a generation compiled a GEMM pipeline");
}

/// The fitted rule (an empty table, as on other GPUs) agrees with the tuned plans.
fn fitted_rule_tracks_tuned_plans(name: &str) {
    let _gpu = gpu_lock();
    let Some(mut m) = load(name, "float32") else { return };
    let request = request(name, 4, 17);
    let tuned = run(&m, &request);
    m.k.table.clear();
    let rule = run(&m, &request);
    assert_eq!(tuned.1, rule.1, "durations");
    let e = max_abs(&tuned.0, &rule.0, |_| true);
    assert!(e < 2e-3, "max abs {e}");
}

/// Encoding leaves the input buffers as uploaded (the SOMA retarget writes elsewhere), so a
/// generation can be encoded again without a new upload.
fn encoding_leaves_inputs_unchanged(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let first = run(&m, &request(name, 3, 9));
    for _ in 0..2 {
        let mut rec = m.recorder();
        m.encode(&mut rec, &m.dims(3, false));
        wait(&rec.commit());
        assert_eq!(m.outputs(3), first, "re-encoding the uploaded inputs changed the result");
    }
}

fn malformed_inputs_are_refused(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let good = request(name, 2, 0);
    let want = run(&m, &good);
    let data = m.input_names()[0]; // context_qpos / global_root_values
    let with = |key: &str, v: Npy| {
        let mut r = good.clone();
        r.insert(key.into(), v);
        r
    };
    let refused = |r: &HashMap<String, Npy>, why: &str| assert!(m.run(r).is_err(), "{name}: accepted {why}");
    refused(&with(data, request(name, 3, 0)[data].clone()), "an input with more rows than the others");
    refused(&with("has_local_poses", request(name, 1, 0)["has_local_poses"].clone()), "a mask with fewer rows");
    let mut shape = good[data].shape.clone();
    *shape.last_mut().unwrap() -= 1;
    let len = shape.iter().product::<i32>() as usize;
    refused(&with(data, Npy::new(shape, NpyData::F32(vec![0.0; len])).unwrap()), "a narrower row");
    refused(&with("allowed_pred_num_tokens", Npy { shape: vec![2, 11], data: NpyData::I32(vec![1; 21]) }), "data shorter than the shape");
    let mut missing = good.clone();
    missing.remove("allowed_pred_num_tokens");
    refused(&missing, "a missing input");
    refused(&examples::rows(&good, 0, 0), "an empty batch");
    assert_eq!(run(&m, &good), want, "a refused request changed the model");
}

fn allowed_token_counts_are_respected(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    check_restricted(&run(&m, &restricted_request(name)).1);
}

fn forced_duration(name: &str) {
    let _gpu = gpu_lock();
    let Some(m) = load(name, "float32") else { return };
    let request = request(name, 2, 0);
    let out = m.run_with_durations(&request, Some(&[7, 13])).unwrap();
    assert_eq!(out.1, vec![7, 13]);
    assert!(finite(&out));
    assert!(m.run_with_durations(&request, Some(&[5, 13])).is_err(), "duration 5 is outside 6..=16");
    assert!(m.run_with_durations(&request, Some(&[7])).is_err(), "one duration per request");
}

/// Shared FP16/BF16/INT8 checks (`common::tracks_fp32`), plus finite outputs.
fn check_tracks_fp32(fp32: &Out, low: &Out, check_motion: bool) {
    assert!(finite(low), "non-finite outputs");
    tracks_fp32(&fp32.1, &low.1, &fp32.0, &low.0, check_motion);
}

fn low_precision_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let request = request(name, 8, 3);
    let reference = run(&fp32, &request);
    for precision in ["float16", "bfloat16"] {
        let low = load(name, precision).unwrap();
        check_tracks_fp32(&reference, &run(&low, &request), true);
    }
}

fn converted_weights_round_trip(name: &str) {
    let _gpu = gpu_lock();
    let Some(original) = load(name, "float32") else { return };
    let dir = TempDir::new(&format!("metal-test-{name}"));
    let path = convert(name, checkpoints(name).unwrap(), dir.0.join("w.safetensors"), mlx_rs::Dtype::Float32).unwrap();
    let reloaded = MetalModel::from_converted(&path, 32, "float32").unwrap();
    drop(dir);
    let request = request(name, 2, 4);
    assert_eq!(run(&original, &request), run(&reloaded, &request), "outputs differ after the round trip");
}

fn quantized_8bit_tracks_fp32(name: &str) {
    let _gpu = gpu_lock();
    let Some(fp32) = load(name, "float32") else { return };
    let q8 = load(name, "int8").unwrap();
    let request = request(name, 16, 21);
    check_tracks_fp32(&run(&fp32, &request), &run(&q8, &request), false);
}

fn batch_larger_than_max_batch_is_refused(name: &str) {
    let _gpu = gpu_lock();
    let Some(dir) = checkpoints(name) else { return };
    let m = MetalModel::from_checkpoints(name, dir, 2).unwrap();
    assert!(m.run(&request(name, 3, 0)).is_err());
    assert!(m.run(&request(name, 2, 0)).is_ok());
    // Encoding directly: `dims` refuses batches the buffers cannot hold.
    for b in [0, 3] {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.dims(b, false))).is_err(), "dims({b}) with max_batch 2");
    }
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
    untuned_batch_at_max_batch,
    generations_compile_no_pipelines,
    fitted_rule_tracks_tuned_plans,
    encoding_leaves_inputs_unchanged,
    malformed_inputs_are_refused,
);
