// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Prost-generated ONNX message types (from the vendored `proto/onnx.proto3`) plus the
//! constructors every emitter uses: little-endian `raw_data` tensors and value infos with
//! a symbolic batch dimension.

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use proto::*;
pub use tensor_proto::DataType;

use crate::graph::Dim;

pub const PRODUCER_NAME: &str = "motionbricks-onnx-rs";
pub const IR_VERSION: i64 = 8;
pub const OPSET_VERSION: i64 = 17;

fn raw_tensor(name: &str, dims: Vec<i64>, data_type: DataType, raw_data: Vec<u8>) -> TensorProto {
    TensorProto {
        dims,
        data_type: data_type as i32,
        name: name.to_string(),
        raw_data,
        ..Default::default()
    }
}

/// A float32 tensor with `raw_data` payload.
pub fn tensor_f32(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    debug_assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{name}");
    raw_tensor(name, dims.to_vec(), DataType::Float, bytemuck::cast_slice(data).to_vec())
}

/// A float16 tensor (IEEE half) with `raw_data` payload.
pub fn tensor_f16(name: &str, dims: &[i64], data: &[half::f16]) -> TensorProto {
    debug_assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{name}");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    raw_tensor(name, dims.to_vec(), DataType::Float16, bytes)
}

/// An int64 tensor with `raw_data` payload (shapes, indices, aranges).
pub fn tensor_i64(name: &str, dims: &[i64], data: &[i64]) -> TensorProto {
    debug_assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{name}");
    raw_tensor(name, dims.to_vec(), DataType::Int64, bytemuck::cast_slice(data).to_vec())
}

/// An int32 tensor with `raw_data` payload.
pub fn tensor_i32(name: &str, dims: &[i64], data: &[i32]) -> TensorProto {
    debug_assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{name}");
    raw_tensor(name, dims.to_vec(), DataType::Int32, bytemuck::cast_slice(data).to_vec())
}

/// A bool tensor with `raw_data` payload.
pub fn tensor_bool(name: &str, dims: &[i64], data: &[bool]) -> TensorProto {
    debug_assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{name}");
    let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
    raw_tensor(name, dims.to_vec(), DataType::Bool, bytes)
}

/// A scalar float32 tensor (rank 0).
pub fn scalar_f32(name: &str, value: f32) -> TensorProto {
    tensor_f32(name, &[], &[value])
}

/// A scalar int64 tensor (rank 0).
pub fn scalar_i64(name: &str, value: i64) -> TensorProto {
    tensor_i64(name, &[], &[value])
}

/// A `ValueInfoProto` for a tensor input/output; [`Dim::Batch`] becomes the symbolic
/// dimension `"batch_size"`, like the upstream exports.
pub fn value_info(name: &str, dtype: DataType, shape: &[Dim]) -> ValueInfoProto {
    let dims = shape
        .iter()
        .map(|d| tensor_shape_proto::Dimension {
            value: Some(match d {
                Dim::Batch => tensor_shape_proto::dimension::Value::DimParam("batch_size".to_string()),
                Dim::Static(v) => tensor_shape_proto::dimension::Value::DimValue(*v),
                Dim::Param(p) => tensor_shape_proto::dimension::Value::DimParam(p.clone()),
            }),
            ..Default::default()
        })
        .collect();
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: dtype as i32,
                shape: Some(TensorShapeProto { dim: dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// An attribute holding a scalar or list.
pub fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

pub fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints: values.to_vec(),
        ..Default::default()
    }
}

pub fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

pub fn attr_floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Floats as i32,
        floats: values.to_vec(),
        ..Default::default()
    }
}

pub fn attr_str(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

pub fn attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Tensor as i32,
        t: Some(tensor),
        ..Default::default()
    }
}
