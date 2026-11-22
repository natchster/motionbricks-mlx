// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The export-time graph checks, ported from upstream `tools/onnx/export.py::validate_output`:
//! the batched tensor interface the demos and TensorRT rely on (names, dtypes, shapes,
//! one shared symbolic batch dimension). Unlike the traced reference, every non-batch
//! dimension here is static.

use anyhow::{Result, bail};

use crate::config::{Kind, ModelSpec};
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{ModelProto, tensor_shape_proto};

/// (name, dtype, static shape without the batch dim) for one input or output.
fn expected(spec: &ModelSpec) -> (Vec<(&'static str, DataType, Vec<i64>)>, Vec<(&'static str, DataType, Vec<i64>)>) {
    match spec.kind {
        Kind::G1 => (
            vec![
                ("context_qpos", DataType::Float, vec![4, 36]),
                ("target_qpos", DataType::Float, vec![4, 36]),
                ("has_global_root_values", DataType::Int32, vec![8]),
                ("has_local_root_values", DataType::Int32, vec![8]),
                ("has_local_poses", DataType::Int32, vec![8]),
                ("allowed_pred_num_tokens", DataType::Int32, vec![11]),
            ],
            vec![
                ("pred_qpos", DataType::Float, vec![64, 36]),
                ("pred_num_tokens", DataType::Int32, vec![]),
                ("pred_foot_contacts", DataType::Float, vec![64, 4]),
            ],
        ),
        Kind::Soma => (
            vec![
                ("global_root_values", DataType::Float, vec![8, 5]),
                ("has_global_root_values", DataType::Int32, vec![8]),
                ("local_root_values", DataType::Float, vec![8, 4]),
                ("has_local_root_values", DataType::Int32, vec![8]),
                ("local_poses", DataType::Float, vec![8, 267]),
                ("has_local_poses", DataType::Int32, vec![8]),
                ("allowed_pred_num_tokens", DataType::Int32, vec![11]),
            ],
            vec![
                ("pred_global_motions", DataType::Float, vec![64, 366]),
                ("pred_num_tokens", DataType::Int32, vec![]),
            ],
        ),
    }
}

fn check_values(
    mode: &str,
    values: &[crate::onnx::ValueInfoProto],
    expected: &[(&str, DataType, Vec<i64>)],
    batch_dimension: &mut Option<String>,
) -> Result<()> {
    let names: std::collections::HashSet<&str> = values.iter().map(|v| v.name.as_str()).collect();
    let want: std::collections::HashSet<&str> = expected.iter().map(|e| e.0).collect();
    if names != want {
        bail!("unexpected ONNX {mode} names: got {names:?}, expected {want:?}");
    }
    for value in values {
        let (_, dtype, shape) = expected.iter().find(|e| e.0 == value.name).expect("checked above");
        let Some(tensor) = value.r#type.as_ref().and_then(|t| match &t.value {
            Some(crate::onnx::type_proto::Value::TensorType(tensor)) => Some(tensor),
            _ => None,
        }) else {
            bail!("{} is not a tensor", value.name);
        };
        let dims = &tensor.shape.as_ref().expect("tensor shape").dim;
        if tensor.elem_type != *dtype as i32 || dims.len() != shape.len() + 1 {
            bail!("unexpected ONNX dtype or rank for {}", value.name);
        }
        let Some(tensor_shape_proto::dimension::Value::DimParam(batch)) = &dims[0].value else {
            bail!("{} must have a symbolic batch dimension", value.name);
        };
        if let Some(prev) = batch_dimension.as_ref() {
            if prev != batch {
                bail!("{} must share the symbolic batch dimension", value.name);
            }
        } else {
            *batch_dimension = Some(batch.clone());
        }
        for (dimension, size) in dims[1..].iter().zip(shape) {
            match &dimension.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(v)) if v == size => {}
                _ => bail!("unexpected ONNX shape for {}", value.name),
            }
        }
    }
    Ok(())
}

/// Check the batched interface of a built model (upstream `validate_output`).
pub fn validate_output(model: &ModelProto, spec: &ModelSpec) -> Result<()> {
    let graph = model.graph.as_ref().expect("graph");
    if model.ir_version != crate::onnx::IR_VERSION
        || model.opset_import.iter().map(|o| (o.domain.as_str(), o.version)).collect::<Vec<_>>() != vec![("", crate::onnx::OPSET_VERSION)]
    {
        bail!("unexpected IR version or opset imports");
    }
    let (inputs, outputs) = expected(spec);
    let mut batch = None;
    check_values("input", &graph.input, &inputs, &mut batch)?;
    check_values("output", &graph.output, &outputs, &mut batch)?;
    // Every node input must be produced somewhere (the structural half of onnx.checker).
    let mut known: std::collections::HashSet<&str> =
        graph.input.iter().map(|v| v.name.as_str()).chain(graph.initializer.iter().map(|t| t.name.as_str())).collect();
    for node in &graph.node {
        for name in &node.input {
            if !name.is_empty() && !known.contains(name.as_str()) {
                bail!("{}: unknown input {name}", node.name);
            }
        }
        known.extend(node.output.iter().map(String::as_str));
    }
    for output in &graph.output {
        if !known.contains(output.name.as_str()) {
            bail!("graph output {} is never produced", output.name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Dim, Graph};

    fn toy_model() -> ModelProto {
        let mut g = Graph::new();
        let x = g.input("context_qpos", DType::F32, &[Dim::Batch, 4.into(), 36.into()]);
        g.output(&x);
        g.to_model("main_graph")
    }

    #[test]
    fn catches_wrong_interface() {
        let err = validate_output(&toy_model(), &crate::config::G1).unwrap_err().to_string();
        assert!(err.contains("unexpected ONNX input names"), "{err}");
    }
}
