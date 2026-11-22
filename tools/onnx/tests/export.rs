// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! End-to-end export against the real checkpoints (skipped when they are missing; set
//! MOTIONBRICKS_REQUIRE_FIXTURES=1 to turn the skip into a failure). Numeric equivalence
//! with the original graphs is checked by `sandbox/tests/test_onnx_rs.py` (ONNX Runtime).

use std::path::PathBuf;

use motionbricks_onnx::config;
use motionbricks_onnx::export::{ExportPrecision, export_model};

fn checkpoint_root() -> Option<PathBuf> {
    let root = std::env::var_os("MOTIONBRICKS_CHECKPOINTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/MotionBricks"));
    let present = |spec: &config::ModelSpec| root.join(spec.kind_name).join(spec.checkpoints[0].path).is_file();
    (present(&config::G1) && present(&config::SOMA)).then_some(root)
}

macro_rules! require {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None if std::env::var_os("MOTIONBRICKS_REQUIRE_FIXTURES").is_some() => {
                panic!("fixtures missing and MOTIONBRICKS_REQUIRE_FIXTURES is set")
            }
            None => {
                eprintln!("SKIPPED: checkpoints not found (run sandbox/setup.sh)");
                return;
            }
        }
    };
}

#[test]
fn exports_both_models_fp32_and_fp16() {
    let root = require!(checkpoint_root());
    for spec in [&config::G1, &config::SOMA] {
        for precision in [ExportPrecision::Fp32, ExportPrecision::Fp16] {
            let dir = root.join(spec.kind_name);
            let model = export_model(&dir, &spec, precision)
                .unwrap_or_else(|e| panic!("export {} {}: {e:#}", spec.name, precision.name()));
            let graph = model.graph.as_ref().unwrap();
            // The hand-built graph is far leaner than the traced reference (8,662 / 6,993 nodes;
            // FP16 adds boundary Cast nodes on top of the FP32 graph).
            assert!(
                graph.node.len() < 3000,
                "{} {}: {} nodes is not the lean graph this exporter should emit",
                spec.name,
                precision.name(),
                graph.node.len()
            );
            let weight_bytes: usize = graph.initializer.iter().map(|t| t.raw_data.len()).sum();
            match precision {
                ExportPrecision::Fp32 => assert!(weight_bytes > 600_000_000, "{}: {weight_bytes}", spec.name),
                // FP16 halves the network bulk (the geometry stays FP32).
                ExportPrecision::Fp16 => assert!(
                    (300_000_000..600_000_000).contains(&weight_bytes),
                    "{}: {weight_bytes}",
                    spec.name
                ),
            }
        }
    }
}
