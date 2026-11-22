// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The graph intermediate representation the emitters build: nodes with typed attributes,
//! inline initializers, and a static dtype/shape registry for every value. The shapes of
//! the MotionBricks graphs are fixed except for the leading batch dimension, so the
//! emitter resolves at build time everything the torch tracer defers into the graph.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};

/// A tensor dimension: the symbolic batch dimension, a static size, or another symbolic
/// dimension (used only for the data-dependent output shapes of the traced reference;
/// this emitter produces static shapes everywhere else).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Batch,
    Static(i64),
    Param(String),
}

impl Dim {
    /// The static value, if this dimension is static.
    pub fn get(&self) -> Option<i64> {
        match self {
            Dim::Static(v) => Some(*v),
            _ => None,
        }
    }

    pub fn is_batch(&self) -> bool {
        matches!(self, Dim::Batch)
    }
}

impl From<i64> for Dim {
    fn from(v: i64) -> Dim {
        Dim::Static(v)
    }
}

impl From<i32> for Dim {
    fn from(v: i32) -> Dim {
        Dim::Static(v as i64)
    }
}

/// Element type of a graph value (the subset of ONNX dtypes this model uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn onnx(self) -> DataType {
        match self {
            DType::F32 => DataType::Float,
            DType::F16 => DataType::Float16,
            DType::I32 => DataType::Int32,
            DType::I64 => DataType::Int64,
            DType::Bool => DataType::Bool,
        }
    }

    pub fn from_onnx(dtype: i32) -> Result<DType> {
        Ok(match DataType::try_from(dtype).map_err(|_| anyhow::anyhow!("bad dtype {dtype}"))? {
            DataType::Float => DType::F32,
            DataType::Float16 => DType::F16,
            DataType::Int32 => DType::I32,
            DataType::Int64 => DType::I64,
            DataType::Bool => DType::Bool,
            other => bail!("unsupported dtype {other:?}"),
        })
    }
}

/// A node attribute (the subset of ONNX attribute kinds the emitters use).
#[derive(Debug, Clone)]
pub enum Attr {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Str(String),
    Tensor(TensorProto),
}

impl Attr {
    fn proto(self, name: &str) -> onnx::AttributeProto {
        match self {
            Attr::Int(v) => onnx::attr_int(name, v),
            Attr::Ints(v) => onnx::attr_ints(name, &v),
            Attr::Float(v) => onnx::attr_float(name, v),
            Attr::Floats(v) => onnx::attr_floats(name, &v),
            Attr::Str(v) => onnx::attr_str(name, &v),
            Attr::Tensor(t) => onnx::attr_tensor(name, t),
        }
    }
}

/// One graph node (single or multiple outputs).
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, Attr)>,
}

impl Node {
    fn proto(self) -> NodeProto {
        NodeProto {
            input: self.inputs,
            output: self.outputs,
            name: self.name,
            op_type: self.op_type,
            attribute: self.attrs.into_iter().map(|(n, a)| a.proto(&n)).collect(),
            ..Default::default()
        }
    }
}

/// The graph under construction.
#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfoProto>,
    pub outputs: Vec<ValueInfoProto>,
    /// dtype of every named value (inputs, initializers, node outputs).
    pub dtypes: HashMap<String, DType>,
    /// static shape of every named value (for build-time shape checks and constants).
    pub shapes: HashMap<String, Vec<Dim>>,
    counter: usize,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Register a graph input (also registers its dtype and shape).
    pub fn input(&mut self, name: &str, dtype: DType, shape: &[Dim]) -> String {
        self.inputs.push(onnx::value_info(name, dtype.onnx(), shape));
        self.register(name, dtype, shape);
        name.to_string()
    }

    /// Register a graph output (must already be a known value).
    pub fn output(&mut self, name: &str) {
        let dtype = self.dtypes[name];
        let shape = self.shapes[name].clone();
        self.outputs.push(onnx::value_info(name, dtype.onnx(), &shape));
    }

    /// Register `out_name` as a graph output holding `value`, via an Identity node (the
    /// public ONNX names differ from the auto-generated value names).
    pub fn output_as(&mut self, out_name: &str, value: &str) {
        let dtype = self.dtypes[value];
        let shape = self.shapes[value].clone();
        self.nodes.push(Node {
            name: format!("/post/{out_name}_alias"),
            op_type: "Identity".to_string(),
            inputs: vec![value.to_string()],
            outputs: vec![out_name.to_string()],
            attrs: vec![],
        });
        self.register(out_name, dtype, &shape);
        self.output(out_name);
    }

    /// Add an initializer; returns its name.
    pub fn initializer(&mut self, tensor: TensorProto) -> String {
        let name = tensor.name.clone();
        let dtype = DType::from_onnx(tensor.data_type).expect("initializer dtype");
        let shape: Vec<Dim> = tensor.dims.iter().map(|&d| Dim::Static(d)).collect();
        self.initializers.push(tensor);
        self.register(&name, dtype, &shape);
        name
    }

    /// A unique value name derived from `hint`.
    pub fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    fn register(&mut self, name: &str, dtype: DType, shape: &[Dim]) {
        self.dtypes.insert(name.to_string(), dtype);
        self.shapes.insert(name.to_string(), shape.to_vec());
    }

    /// Register a value created outside [`Graph::node`] (the FP16 pass's cast outputs).
    pub fn register_cast(&mut self, name: &str, dtype: DType, shape: Vec<Dim>) {
        self.register(name, dtype, &shape);
    }

    /// Append a single-output node; returns the output value name.
    ///
    /// `scope` is the module path (e.g. `/core/backbone_net/_transformer_model/layers.0`);
    /// the node is named `{scope}/{op}` (suffixed when the name repeats) and its output
    /// `{node_name}_output_0`, following the naming of the traced reference graphs.
    pub fn node(
        &mut self,
        scope: &str,
        op: &str,
        inputs: &[&str],
        attrs: &[(&str, Attr)],
        out_dtype: DType,
        out_shape: &[Dim],
    ) -> String {
        self.node_multi(scope, op, inputs, attrs, &[(out_dtype, out_shape.to_vec())])[0].clone()
    }

    /// Append a node with several outputs; returns the output value names.
    pub fn node_multi(
        &mut self,
        scope: &str,
        op: &str,
        inputs: &[&str],
        attrs: &[(&str, Attr)],
        outs: &[(DType, Vec<Dim>)],
    ) -> Vec<String> {
        let base = if scope.is_empty() { op.to_string() } else { format!("{scope}/{op}") };
        let mut name = base.clone();
        let mut suffix = 1;
        while self.nodes.iter().any(|n| n.name == name) {
            suffix += 1;
            name = format!("{base}_{suffix}");
        }
        let outputs: Vec<String> = (0..outs.len()).map(|i| format!("{name}_output_{i}")).collect();
        for (i, (dtype, shape)) in outs.iter().enumerate() {
            self.register(&outputs[i], *dtype, shape);
        }
        self.nodes.push(Node {
            name,
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.clone(),
            attrs: attrs.iter().map(|(n, a)| (n.to_string(), a.clone())).collect(),
        });
        outputs
    }

    pub fn dtype(&self, name: &str) -> DType {
        *self.dtypes.get(name).unwrap_or_else(|| panic!("unknown value {name}"))
    }

    pub fn shape(&self, name: &str) -> &[Dim] {
        self.shapes.get(name).unwrap_or_else(|| panic!("unknown value {name}"))
    }

    /// The total initializer payload size in bytes.
    pub fn initializer_bytes(&self) -> usize {
        self.initializers.iter().map(|t| t.raw_data.len()).sum()
    }

    /// Assemble the final `ModelProto` (opset 17, IR version 8, like the reference exports).
    pub fn to_model(self, graph_name: &str) -> ModelProto {
        let graph = GraphProto {
            node: self.nodes.into_iter().map(Node::proto).collect(),
            name: graph_name.to_string(),
            initializer: self.initializers,
            input: self.inputs,
            output: self.outputs,
            ..Default::default()
        };
        ModelProto {
            ir_version: onnx::IR_VERSION,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: String::new(),
                version: onnx::OPSET_VERSION,
            }],
            producer_name: onnx::PRODUCER_NAME.to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            graph: Some(graph),
            ..Default::default()
        }
    }
}

/// Broadcast two shape vectors (NumPy/ONNX rules: right-aligned, 1s stretch). `Batch`
/// only matches `Batch` or a static 1. Errors on a real mismatch.
pub fn broadcast(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i < n - a.len() { &Dim::Static(1) } else { &a[i - (n - a.len())] };
        let db = if i < n - b.len() { &Dim::Static(1) } else { &b[i - (n - b.len())] };
        let d = match (da, db) {
            (Dim::Static(1), d) | (d, Dim::Static(1)) => d.clone(),
            (Dim::Batch, Dim::Batch) => Dim::Batch,
            (Dim::Static(x), Dim::Static(y)) if x == y => Dim::Static(*x),
            (Dim::Param(p), Dim::Param(q)) if p == q => Dim::Param(p.clone()),
            _ => bail!("cannot broadcast {a:?} with {b:?}"),
        };
        out.push(d);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_rules() {
        let b = Dim::Batch;
        let s = |v: i64| Dim::Static(v);
        assert_eq!(broadcast(&[b.clone(), s(64), s(36)], &[s(36)]).unwrap(), vec![b.clone(), s(64), s(36)]);
        assert_eq!(broadcast(&[b.clone(), s(1)], &[s(8)]).unwrap(), vec![b, s(8)]);
        assert!(broadcast(&[s(4)], &[s(5)]).is_err());
        assert!(broadcast(&[Dim::Batch], &[s(8)]).is_err());
    }
}
