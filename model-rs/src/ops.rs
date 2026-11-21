// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Thin helpers over mlx-rs that keep the model code close to the Python port.
//!
//! Two details matter for parity with `motionbricks_mlx`:
//! * Python scalars are *weakly typed* in MLX (`1 - h` keeps `h`'s dtype). Rust scalars are
//!   not, so constants are created in the operand's dtype with [`scalar`].
//! * `x[:, 0]` in Python is a slice + reshape, never a gather. [`slice`]/[`index`] emit the same
//!   ops through MLX's slice primitive.

use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype, Stream, ops};

pub type R<T> = std::result::Result<T, Exception>;

/// End-of-axis marker for [`slice`] (Python's `x[a:]`).
pub const END: i32 = i32::MAX;

/// A 0-d constant in `dtype`, like a weakly typed Python scalar.
pub fn scalar(value: f32, dtype: Dtype) -> Array {
    match dtype {
        Dtype::Float16 => Array::from_slice(&[half::f16::from_f32(value)], &[]),
        Dtype::Bfloat16 => Array::from_slice(&[half::bf16::from_f32(value)], &[]),
        _ => Array::from_f32(value),
    }
}

pub fn f(value: f32) -> Array {
    Array::from_f32(value)
}

pub fn i(value: i32) -> Array {
    Array::from_int(value)
}

fn raw_slice(a: &Array, starts: &[i32], stops: &[i32], strides: &[i32]) -> R<Array> {
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice(
            &mut res,
            a.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            stops.as_ptr(),
            stops.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(res);
            return Err(Exception::custom("mlx_slice failed"));
        }
        Ok(Array::from_ptr(res))
    }
}

fn axis_of(a: &Array, axis: i32) -> usize {
    (if axis < 0 { axis + a.ndim() as i32 } else { axis }) as usize
}

/// `mx.full(shape, value)`: dtype follows `value` (mlx-rs's `full::<T>` forces `T`).
pub fn full(shape: &[i32], value: &Array) -> R<Array> {
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_full(&mut res, shape.as_ptr(), shape.len(), value.as_ptr(), value.dtype().into(), stream.as_ptr());
        if status != 0 {
            mlx_sys::mlx_array_free(res);
            return Err(Exception::custom("mlx_full failed"));
        }
        Ok(Array::from_ptr(res))
    }
}

/// `a[..., start:stop, ...]` on one axis (negative indices and [`END`] allowed).
pub fn slice(a: &Array, axis: i32, start: i32, stop: i32) -> R<Array> {
    let ax = axis_of(a, axis);
    let shape = a.shape();
    let dim = shape[ax];
    let norm = |v: i32| (if v < 0 { v + dim } else { v }).clamp(0, dim);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.to_vec();
    starts[ax] = norm(start);
    stops[ax] = norm(stop);
    raw_slice(a, &starts, &stops, &vec![1; shape.len()])
}

/// `a[..., i, ...]`: slice then drop the axis (a view, like Python's integer indexing).
pub fn index(a: &Array, axis: i32, i: i32) -> R<Array> {
    let ax = axis_of(a, axis);
    let dim = a.shape()[ax];
    let i = if i < 0 { i + dim } else { i };
    let s = slice(a, ax as i32, i, i + 1)?;
    let mut shape = s.shape().to_vec();
    shape.remove(ax);
    s.reshape(&shape)
}

/// `a[..., i, j]` on the last two axes (one slice + one reshape, like Python).
pub fn index2(a: &Array, i: i32, j: i32) -> R<Array> {
    let n = a.ndim();
    let shape = a.shape();
    let mut starts = vec![0; n];
    let mut stops = shape.to_vec();
    starts[n - 2] = i;
    stops[n - 2] = i + 1;
    starts[n - 1] = j;
    stops[n - 1] = j + 1;
    let s = raw_slice(a, &starts, &stops, &vec![1; n])?;
    s.reshape(&shape[..n - 2])
}

/// Append a unit axis (`a[..., None]`).
pub fn unsq(a: &Array, axis: i32) -> R<Array> {
    a.expand_dims(axis)
}

pub fn cat(arrays: &[&Array], axis: i32) -> R<Array> {
    ops::concatenate(arrays, axis)
}

pub fn stack(arrays: &[&Array], axis: i32) -> R<Array> {
    ops::stack(arrays, axis)
}

pub fn wh(cond: &Array, a: &Array, b: &Array) -> R<Array> {
    ops::select(cond, a, b)
}

pub fn arange(n: i32) -> R<Array> {
    Array::arange::<i32, i32>(None, n, None)
}

/// `nn.Linear`: `addmm(bias, x, weight.T)`, exactly like `mlx.nn.Linear`.
pub fn linear(x: &Array, weight: &Array, bias: &Array) -> R<Array> {
    ops::addmm(bias, x, weight.t(), None, None)
}

/// `mlx.nn.relu`: `maximum(x, 0)` with a weakly typed zero.
pub fn relu(x: &Array) -> R<Array> {
    ops::maximum(x, scalar(0.0, x.dtype()))
}

/// `mlx.nn.leaky_relu`: `maximum(slope * x, x)`.
pub fn leaky_relu(x: &Array, slope: f32) -> R<Array> {
    ops::maximum(x * scalar(slope, x.dtype()), x)
}

pub fn atan2(a: &Array, b: &Array) -> R<Array> {
    ops::atan2(a, b)
}

pub fn sin(a: &Array) -> R<Array> {
    ops::sin(a)
}

pub fn cos(a: &Array) -> R<Array> {
    ops::cos(a)
}

/// Copy an array to a row-major Vec on the CPU, following its strides (MLX may return strided
/// views; this is what `np.array(x)` does in the Python port, with no extra GPU kernel).
fn to_vec<T: Copy>(a: &Array, data: unsafe extern "C" fn(mlx_sys::mlx_array) -> *const T) -> R<Vec<T>> {
    a.eval()?;
    let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    unsafe {
        let ptr = data(a.as_ptr());
        let strides = std::slice::from_raw_parts(mlx_sys::mlx_array_strides(a.as_ptr()), shape.len());
        let mut expected = 1;
        let contiguous = shape.iter().zip(strides).rev().all(|(&d, &s)| {
            let ok = d == 1 || s == expected;
            expected *= d;
            ok
        });
        if contiguous {
            return Ok(std::slice::from_raw_parts(ptr, n).to_vec());
        }
        let mut out = Vec::with_capacity(n);
        let mut index = vec![0usize; shape.len()];
        for _ in 0..n {
            out.push(*ptr.add(index.iter().zip(strides).map(|(i, s)| i * s).sum()));
            for d in (0..shape.len()).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        Ok(out)
    }
}

pub fn to_vec_f32(a: &Array) -> R<Vec<f32>> {
    to_vec(&a.as_dtype(Dtype::Float32)?, mlx_sys::mlx_array_data_float32)
}

pub fn to_vec_i32(a: &Array) -> R<Vec<i32>> {
    to_vec(&a.as_dtype(Dtype::Int32)?, mlx_sys::mlx_array_data_int32)
}
