// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Mixed-FP16 conversion, ported from upstream `tools/onnx/precision.py`: the two
//! transformers and the VQ-VAE decoder run in FP16; LayerNormalization, Softmax, Resize
//! and all motion geometry stay in FP32; boundary `Cast` nodes are inserted (deduplicated
//! per tensor). Unlike upstream, no ONNX shape inference is needed: the emitter's static
//! dtype/shape registry already knows every value.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::graph::{Attr, DType, Graph, Node};
use crate::onnx::tensor_proto::DataType;
use crate::ops;

/// Ops that always stay in FP32 (numerically sensitive reductions and resampling).
const FP32_OPS: [&str; 3] = ["LayerNormalization", "Softmax", "Resize"];

/// Convert the network subgraphs of `g` to FP16 in place. `prefixes` are the model's
/// node-name scopes (`ModelSpec::fp16_prefixes`).
pub fn convert_to_fp16(g: &mut Graph, prefixes: &[&str]) -> Result<()> {
    if g.nodes.iter().any(|n| matches!(n.op_type.as_str(), "If" | "Loop" | "Scan")) {
        bail!("mixed precision conversion requires a graph without control flow");
    }
    // Per-node precision.
    let precision: HashMap<String, DType> = g
        .nodes
        .iter()
        .map(|n| {
            let fp16 = prefixes.iter().any(|p| n.name.starts_with(p)) && !FP32_OPS.contains(&n.op_type.as_str());
            (n.name.clone(), if fp16 { DType::F16 } else { DType::F32 })
        })
        .collect();
    // Which precisions consume each value.
    let mut consumers: HashMap<&str, HashSet<DType>> = HashMap::new();
    for n in &g.nodes {
        for name in &n.inputs {
            if !name.is_empty() {
                consumers.entry(name.as_str()).or_default().insert(precision[&n.name]);
            }
        }
    }
    // Original dtypes (before conversion), then the actual dtypes after conversion.
    let original = g.dtypes.clone();
    let mut actual = original.clone();
    for tensor in &mut g.initializers {
        if tensor.data_type == DataType::Float as i32 && consumers.get(tensor.name.as_str()) == Some(&HashSet::from([DType::F16])) {
            *tensor = ops::tensor_to_f16(tensor);
            actual.insert(tensor.name.clone(), DType::F16);
        }
    }
    for n in &g.nodes {
        for name in &n.outputs {
            if original[name] == DType::F32 {
                actual.insert(name.clone(), precision[&n.name]);
            }
        }
    }
    // Graph outputs must remain FP32 (their producers are all outside the FP16 scopes).
    for output in &g.outputs {
        if original[output.name.as_str()] == DType::F32 && actual[output.name.as_str()] != DType::F32 {
            bail!("graph output {} would become FP16; its producer must stay in an FP32 scope", output.name);
        }
    }

    // Rewrite inputs, inserting boundary casts.
    let mut rewritten: Vec<Node> = Vec::with_capacity(g.nodes.len() * 2);
    let mut cast_names: HashMap<(String, DType), String> = HashMap::new();
    let mut nodes = std::mem::take(&mut g.nodes);
    for n in &mut nodes {
        let dtype = precision[&n.name];
        for input in &mut n.inputs {
            if input.is_empty() || original.get(input.as_str()) != Some(&DType::F32) {
                continue;
            }
            if actual[input.as_str()] != dtype {
                let key = (input.clone(), dtype);
                let casted = cast_names.entry(key).or_insert_with(|| {
                    let suffix = if dtype == DType::F16 { "__fp16" } else { "__fp32" };
                    let out = format!("{input}{suffix}");
                    let shape = g.shapes[input.as_str()].clone();
                    rewritten.push(Node {
                        name: format!("{out}_cast"),
                        op_type: "Cast".to_string(),
                        inputs: vec![input.clone()],
                        outputs: vec![out.clone()],
                        attrs: vec![("to".to_string(), Attr::Int(dtype.onnx() as i64))],
                    });
                    g.register_cast(&out, dtype, shape);
                    out
                });
                *input = casted.clone();
            }
        }
        if dtype == DType::F16 {
            for (_, attr) in &mut n.attrs {
                match attr {
                    Attr::Tensor(t) if t.data_type == DataType::Float as i32 => {
                        // Ordinary IEEE half conversion, which keeps the attention -inf values.
                        *t = ops::tensor_to_f16(t);
                    }
                    Attr::Int(to) if n.op_type == "Cast" && *to == DataType::Float as i64 => {
                        *to = DataType::Float16 as i64;
                    }
                    _ => {}
                }
            }
        }
        rewritten.push(n.clone());
    }
    g.nodes = rewritten;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Dim;
    use crate::onnx;

    /// A tiny graph: linear (FP16 scope) -> layernorm (FP32) -> output.
    fn toy_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", DType::F32, &[Dim::Batch, 8.into()]);
        let w = g.initializer(onnx::tensor_f32("/core/backbone_net/w_t", &[8, 8], &[0.1; 64]));
        let b = g.initializer(onnx::tensor_f32("/core/backbone_net/b", &[8], &[0.0; 8]));
        let y = ops::matmul(&mut g, "/core/backbone_net", &x, &w);
        let y = ops::add(&mut g, "/core/backbone_net", &y, &b);
        let s = g.initializer(onnx::tensor_f32("ln_scale", &[8], &[1.0; 8]));
        let z = ops::layer_norm(&mut g, "/core", &y, &s, &b, 1e-5);
        g.output(&z);
        g
    }

    #[test]
    fn converts_only_network_scopes() {
        let mut g = toy_graph();
        convert_to_fp16(&mut g, &["/core/backbone_net/"]).unwrap();
        // The weight is consumed only by FP16 nodes -> halved initializer, no input cast.
        let w = g.initializers.iter().find(|t| t.name == "/core/backbone_net/w_t").unwrap();
        assert_eq!(w.data_type, DataType::Float16 as i32);
        // The bias is shared with the FP32 LayerNormalization -> stays FP32, cast inserted.
        let b = g.initializers.iter().find(|t| t.name == "/core/backbone_net/b").unwrap();
        assert_eq!(b.data_type, DataType::Float as i32);
        let casts: Vec<&Node> = g.nodes.iter().filter(|n| n.op_type == "Cast").collect();
        assert!(casts.iter().any(|n| n.inputs[0] == "x" && n.outputs[0] == "x__fp16"));
        assert!(casts.iter().any(|n| n.inputs[0] == "/core/backbone_net/b"));
        // One fp32 cast back for the layernorm input.
        let matmul = g.nodes.iter().find(|n| n.op_type == "MatMul").unwrap();
        assert_eq!(matmul.inputs[1], "/core/backbone_net/w_t");
        let ln = g.nodes.iter().find(|n| n.op_type == "LayerNormalization").unwrap();
        assert!(ln.inputs[0].ends_with("__fp32"), "{:?}", ln.inputs);
    }
}
