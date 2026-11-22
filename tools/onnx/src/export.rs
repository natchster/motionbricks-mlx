// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The export pipeline (upstream `export_release`): verify and load the checkpoints,
//! build the graph, optionally convert to mixed FP16, validate the interface.

use std::path::Path;

use anyhow::Result;

use crate::config::ModelSpec;
use crate::onnx::ModelProto;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportPrecision {
    Fp32,
    Fp16,
}

impl ExportPrecision {
    pub fn name(self) -> &'static str {
        match self {
            ExportPrecision::Fp32 => "fp32",
            ExportPrecision::Fp16 => "fp16",
        }
    }
}

/// Build a validated model for `spec` from the checkpoint folder `dir`.
pub fn export_model(dir: impl AsRef<Path>, spec: &'static ModelSpec, precision: ExportPrecision) -> Result<ModelProto> {
    let mut loaded = crate::weights::read_checkpoints(dir, spec)?;
    let mut graph = crate::emit::emit_graph(spec, &mut loaded)?;
    loaded.weights.finish()?;
    if precision == ExportPrecision::Fp16 {
        crate::fp16::convert_to_fp16(&mut graph, spec.fp16_prefixes)?;
    }
    let model = graph.to_model("main_graph");
    crate::validate::validate_output(&model, spec)?;
    Ok(model)
}
