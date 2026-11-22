// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Typed emit helpers over [`Graph`]: one function per ONNX op pattern the MotionBricks
//! graphs need, each registering its output's dtype and static shape. All shape math the
//! torch tracer defers into the graph (Shape/Gather/Unsqueeze/Concat chains) is resolved
//! here at build time; only genuinely data-dependent index arithmetic is emitted as nodes.

use crate::graph::{Attr, DType, Dim, Graph, broadcast};
use crate::onnx::{self, TensorProto};

fn is_float(t: DType) -> bool {
    matches!(t, DType::F32 | DType::F16)
}

/// Normalize a possibly negative axis against `rank`.
fn ax(axis: i64, rank: usize) -> usize {
    let a = if axis < 0 { rank as i64 + axis } else { axis };
    assert!(0 <= a && (a as usize) < rank, "axis {axis} out of range for rank {rank}");
    a as usize
}

// ---- constants / initializers ----

/// A float32 initializer.
pub fn c_f32(g: &mut Graph, name: &str, dims: &[i64], data: &[f32]) -> String {
    g.initializer(onnx::tensor_f32(name, dims, data))
}

/// An int64 initializer.
pub fn c_i64(g: &mut Graph, name: &str, dims: &[i64], data: &[i64]) -> String {
    g.initializer(onnx::tensor_i64(name, dims, data))
}

/// An int32 initializer.
pub fn c_i32(g: &mut Graph, name: &str, dims: &[i64], data: &[i32]) -> String {
    g.initializer(onnx::tensor_i32(name, dims, data))
}

/// A scalar float32 initializer, deduplicated per scope+value.
pub fn c_scalar(g: &mut Graph, scope: &str, value: f32) -> String {
    let name = format!("{scope}/const_{value}");
    if g.dtypes.contains_key(&name) {
        return name;
    }
    g.initializer(onnx::scalar_f32(&name, value))
}

/// A scalar int64 initializer.
pub fn c_scalar_i64(g: &mut Graph, scope: &str, value: i64) -> String {
    let name = format!("{scope}/const_i64_{value}");
    if g.dtypes.contains_key(&name) {
        return name;
    }
    g.initializer(onnx::scalar_i64(&name, value))
}

/// `arange(n)` as an int64 initializer.
pub fn arange(g: &mut Graph, scope: &str, n: i64) -> String {
    let name = format!("{scope}/arange_{n}");
    if g.dtypes.contains_key(&name) {
        return name;
    }
    let data: Vec<i64> = (0..n).collect();
    c_i64(g, &name, &[n], &data)
}

/// A rank-1 int64 constant (shapes, perms, index lists).
pub fn c_vec(g: &mut Graph, scope: &str, hint: &str, values: &[i64]) -> String {
    // Always a fresh name: a helper may run many times under one scope, and initializer
    // names must be unique.
    let name = g.fresh(&format!("{scope}/{hint}"));
    c_i64(g, &name, &[values.len() as i64], values)
}

/// A tensor filled with `value` (float32).
pub fn full_f32(g: &mut Graph, scope: &str, hint: &str, dims: &[i64], value: f32) -> String {
    let n = dims.iter().product::<i64>() as usize;
    c_f32(g, &format!("{scope}/{hint}"), dims, &vec![value; n])
}

// ---- shape helpers for emitters ----

/// A static Reshape target as an initializer; [`Dim::Batch`] maps to 0 (copy the input
/// dim at that position) and [`Dim::Param`] to -1 (inferred; at most one per reshape).
pub fn shape_tensor(g: &mut Graph, scope: &str, dims: &[Dim]) -> String {
    let mut seen_infer = false;
    let values: Vec<i64> = dims
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v,
            Dim::Batch => 0,
            Dim::Param(_) => {
                assert!(!seen_infer, "one inferred dimension per reshape");
                seen_infer = true;
                -1
            }
        })
        .collect();
    c_vec(g, scope, "reshape_dims", &values)
}

// ---- core ops ----

/// `MatMul` (batched): [..., M, K] @ [..., K, N] -> [..., M, N].
pub fn matmul(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let (sa, sb) = (g.shape(a), g.shape(b));
    let (ra, rb) = (sa.len(), sb.len());
    assert!(ra >= 2 && rb >= 2, "matmul needs rank >= 2");
    let batch = broadcast(&sa[..ra - 2], &sb[..rb - 2]).expect("matmul batch dims");
    assert_eq!(sa[ra - 1], sb[rb - 2], "matmul inner dims");
    let mut out = batch;
    out.push(sa[ra - 2].clone());
    out.push(sb[rb - 1].clone());
    let dtype = g.dtype(a);
    g.node(scope, "MatMul", &[a, b], &[], dtype, &out)
}

/// `Gemm` on a rank-2 activation: [Batch, K] @ W^T + b with W in PyTorch layout [N, K].
pub fn gemm(g: &mut Graph, scope: &str, x: &str, w: &str, bias: &str) -> String {
    let (sx, sw) = (g.shape(x), g.shape(w));
    assert_eq!(sx.len(), 2, "gemm needs a rank-2 input; use `linear` for rank > 2");
    assert_eq!(sx[1], sw[1], "gemm inner dims");
    let out = [sx[0].clone(), sw[0].clone()];
    let dtype = g.dtype(x);
    g.node(
        scope,
        "Gemm",
        &[x, w, bias],
        &[("alpha", Attr::Float(1.0)), ("beta", Attr::Float(1.0)), ("transB", Attr::Int(1))],
        dtype,
        &out,
    )
}

/// `Transpose`.
pub fn transpose(g: &mut Graph, scope: &str, x: &str, perm: &[i64]) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = perm.iter().map(|&p| s[p as usize].clone()).collect();
    let dtype = g.dtype(x);
    g.node(scope, "Transpose", &[x], &[("perm", Attr::Ints(perm.to_vec()))], dtype, &out)
}

/// `Reshape` with a static target (`Dim::Batch` keeps the input batch, one `Dim::Param`
/// may be used for an inferred dimension).
pub fn reshape(g: &mut Graph, scope: &str, x: &str, dims: &[Dim]) -> String {
    let shape = shape_tensor(g, scope, dims);
    let dtype = g.dtype(x);
    g.node(scope, "Reshape", &[x, &shape], &[("allowzero", Attr::Int(0))], dtype, dims)
}

/// `Flatten` at `axis`.
pub fn flatten(g: &mut Graph, scope: &str, x: &str, axis: i64) -> String {
    let s = g.shape(x);
    let axis = ax(axis, s.len()) as i64;
    let (l, r) = (&s[..axis as usize], &s[axis as usize..]);
    let out = if r.iter().all(|d| d.get().is_some()) {
        let n: i64 = r.iter().map(|d| d.get().unwrap()).product();
        let mut o = l.to_vec();
        o.push(Dim::Static(n));
        o
    } else {
        vec![l[0].clone(), Dim::Param("flatten".into())]
    };
    let dtype = g.dtype(x);
    g.node(scope, "Flatten", &[x], &[("axis", Attr::Int(axis))], dtype, &out)
}

/// `Unsqueeze` at `axes` (sorted).
pub fn unsqueeze(g: &mut Graph, scope: &str, x: &str, axes: &[i64]) -> String {
    let s = g.shape(x);
    let mut out = s.to_vec();
    let axes_t = c_vec(g, scope, "axes", axes);
    for &ax in axes {
        out.insert(ax as usize, Dim::Static(1));
    }
    let dtype = g.dtype(x);
    g.node(scope, "Unsqueeze", &[x, &axes_t], &[], dtype, &out)
}

/// `Squeeze` at `axes` (the dims must be statically 1).
pub fn squeeze(g: &mut Graph, scope: &str, x: &str, axes: &[i64]) -> String {
    let s = g.shape(x);
    let removed: std::collections::HashSet<i64> = axes.iter().copied().collect();
    let out: Vec<Dim> = s
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(&(*i as i64)))
        .map(|(_, d)| d.clone())
        .collect();
    let axes_t = c_vec(g, scope, "axes", axes);
    let dtype = g.dtype(x);
    g.node(scope, "Squeeze", &[x, &axes_t], &[], dtype, &out)
}

/// `Concat` along `axis`.
pub fn concat(g: &mut Graph, scope: &str, xs: &[&str], axis: i64) -> String {
    assert!(!xs.is_empty());
    let ax = ax(axis, g.shape(xs[0]).len());
    let mut out = g.shape(xs[0]).to_vec();
    for &x in &xs[1..] {
        let s = g.shape(x);
        out[ax] = match (&out[ax], &s[ax]) {
            (Dim::Static(a), Dim::Static(b)) => Dim::Static(a + b),
            _ => Dim::Param("concat".into()),
        };
    }
    let dtype = g.dtype(xs[0]);
    g.node(scope, "Concat", xs, &[("axis", Attr::Int(axis))], dtype, &out)
}

/// `Slice` with static bounds on one axis.
pub fn slice_axis(g: &mut Graph, scope: &str, x: &str, axis: i64, start: i64, end: i64) -> String {
    let starts = c_vec(g, scope, "starts", &[start]);
    let ends = c_vec(g, scope, "ends", &[end]);
    let axes = c_vec(g, scope, "axes", &[axis]);
    let steps = c_vec(g, scope, "steps", &[1]);
    let mut out = g.shape(x).to_vec();
    let axis = ax(axis, out.len()) as i64;
    out[axis as usize] = Dim::Static(end - start);
    let dtype = g.dtype(x);
    g.node(scope, "Slice", &[x, &starts, &ends, &axes, &steps], &[], dtype, &out)
}

/// `Split` into static `sizes` along `axis`.
pub fn split(g: &mut Graph, scope: &str, x: &str, axis: i64, sizes: &[i64]) -> Vec<String> {
    let split_t = c_vec(g, scope, "split", sizes);
    let s = g.shape(x);
    let outs: Vec<(DType, Vec<Dim>)> = sizes
        .iter()
        .map(|&n| {
            let mut o = s.to_vec();
            let at = ax(axis, o.len());
            o[at] = Dim::Static(n);
            (g.dtype(x), o)
        })
        .collect();
    g.node_multi(scope, "Split", &[x, &split_t], &[("axis", Attr::Int(axis))], &outs)
}

/// Elementwise binary op with broadcasting.
fn binary(g: &mut Graph, scope: &str, op: &str, a: &str, b: &str, dtype: DType) -> String {
    let out = broadcast(g.shape(a), g.shape(b)).expect("binary op shapes");
    g.node(scope, op, &[a, b], &[], dtype, &out)
}

fn binary_f(g: &mut Graph, scope: &str, op: &str, a: &str, b: &str) -> String {
    let dtype = g.dtype(a);
    debug_assert!(is_float(dtype));
    binary(g, scope, op, a, b, dtype)
}

pub fn add(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary_f(g, scope, "Add", a, b)
}
pub fn sub(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary_f(g, scope, "Sub", a, b)
}
pub fn mul(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary_f(g, scope, "Mul", a, b)
}
pub fn div(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary_f(g, scope, "Div", a, b)
}
pub fn maximum(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "Max", a, b, g.dtype(a))
}
pub fn equal(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "Equal", a, b, DType::Bool)
}
pub fn less(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "Less", a, b, DType::Bool)
}
pub fn greater(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "Greater", a, b, DType::Bool)
}
pub fn and_(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "And", a, b, DType::Bool)
}
pub fn or_(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    binary(g, scope, "Or", a, b, DType::Bool)
}

/// Elementwise unary op preserving shape and (float) dtype.
fn unary(g: &mut Graph, scope: &str, op: &str, x: &str) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, op, &[x], &[], dtype, &shape)
}

pub fn sqrt(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Sqrt", x)
}
pub fn sin(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Sin", x)
}
pub fn cos(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Cos", x)
}
pub fn atan(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Atan", x)
}
pub fn abs_(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Abs", x)
}
pub fn neg(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Neg", x)
}
pub fn relu(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Relu", x)
}
pub fn floor(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Floor", x)
}
pub fn sign(g: &mut Graph, scope: &str, x: &str) -> String {
    unary(g, scope, "Sign", x)
}
pub fn not_(g: &mut Graph, scope: &str, x: &str) -> String {
    let shape = g.shape(x).to_vec();
    g.node(scope, "Not", &[x], &[], DType::Bool, &shape)
}

pub fn leaky_relu(g: &mut Graph, scope: &str, x: &str, alpha: f32) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "LeakyRelu", &[x], &[("alpha", Attr::Float(alpha))], dtype, &shape)
}

/// atan2(y, x) with the decomposition torch exports at opset 17 (no native op):
/// atan(y/x) with quadrant adjustments via Where.
pub fn atan2(g: &mut Graph, scope: &str, y: &str, x: &str) -> String {
    let pi = c_scalar(g, scope, std::f32::consts::PI);
    let half_pi = c_scalar(g, scope, std::f32::consts::FRAC_PI_2);
    let neg_half_pi = c_scalar(g, scope, -std::f32::consts::FRAC_PI_2);
    let zero = c_scalar(g, scope, 0.0);
    let ratio = div(g, scope, y, x);
    let slope = atan(g, scope, &ratio);
    // x < 0: add or subtract pi by the sign of y (sign(0) = 0, matching torch's atan2(0, -1) = pi? no:
    // torch.atan2(0., -1.) = pi; the decomposition below follows torch's symbolic exactly).
    let x_neg = less(g, scope, x, &zero);
    let y_pos = greater(g, scope, y, &zero);
    let y_neg = less(g, scope, y, &zero);
    let plus_pi = add(g, scope, &slope, &pi);
    let minus_pi = sub(g, scope, &slope, &pi);
    let adj = where_(g, scope, &y_neg, &minus_pi, &plus_pi);
    let slope_adj = where_(g, scope, &x_neg, &adj, &slope);
    // x == 0: +/- pi/2 by the sign of y (0 when y == 0).
    let x_zero = equal(g, scope, x, &zero);
    let at_zero = where_(g, scope, &y_pos, &half_pi, &neg_half_pi);
    let y_zero = equal(g, scope, y, &zero);
    let at_zero = where_(g, scope, &y_zero, &zero, &at_zero);
    where_(g, scope, &x_zero, &at_zero, &slope_adj)
}

/// `Where(cond, a, b)` with broadcasting.
pub fn where_(g: &mut Graph, scope: &str, cond: &str, a: &str, b: &str) -> String {
    let out = broadcast(g.shape(a), g.shape(b)).expect("where branches");
    let out = broadcast(&out, g.shape(cond)).expect("where cond");
    let dtype = g.dtype(a);
    g.node(scope, "Where", &[cond, a, b], &[], dtype, &out)
}

/// `Cast` to `dtype`.
pub fn cast(g: &mut Graph, scope: &str, x: &str, dtype: DType) -> String {
    let shape = g.shape(x).to_vec();
    g.node(scope, "Cast", &[x], &[("to", Attr::Int(dtype.onnx() as i64))], dtype, &shape)
}

/// `LayerNormalization` (axis -1).
pub fn layer_norm(g: &mut Graph, scope: &str, x: &str, scale: &str, bias: &str, eps: f32) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(
        scope,
        "LayerNormalization",
        &[x, scale, bias],
        &[("axis", Attr::Int(-1)), ("epsilon", Attr::Float(eps))],
        dtype,
        &shape,
    )
}

/// `Softmax` at `axis`.
pub fn softmax(g: &mut Graph, scope: &str, x: &str, axis: i64) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "Softmax", &[x], &[("axis", Attr::Int(axis))], dtype, &shape)
}

/// `ArgMax` at `axis` (int64 output).
pub fn argmax(g: &mut Graph, scope: &str, x: &str, axis: i64, keepdims: bool) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = if keepdims {
        let mut o = s.to_vec();
        let at = ax(axis, o.len());
        o[at] = Dim::Static(1);
        o
    } else {
        let keep = ax(axis, s.len()) as i64;
        s.iter().enumerate().filter(|(i, _)| *i as i64 != keep).map(|(_, d)| d.clone()).collect()
    };
    g.node(
        scope,
        "ArgMax",
        &[x],
        &[("axis", Attr::Int(axis)), ("keepdims", Attr::Int(keepdims as i64)), ("select_last_index", Attr::Int(0))],
        DType::I64,
        &out,
    )
}

/// `Gather` at `axis`: out = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:].
pub fn gather(g: &mut Graph, scope: &str, x: &str, indices: &str, axis: i64) -> String {
    let (sx, si) = (g.shape(x), g.shape(indices));
    let ax = ax(axis, sx.len());
    let mut out = sx[..ax].to_vec();
    out.extend_from_slice(si);
    out.extend_from_slice(&sx[ax + 1..]);
    let dtype = g.dtype(x);
    g.node(scope, "Gather", &[x, indices], &[("axis", Attr::Int(axis))], dtype, &out)
}

/// `Gather` with a static index vector.
pub fn gather_static(g: &mut Graph, scope: &str, x: &str, indices: &[i64], axis: i64) -> String {
    let idx = c_vec(g, scope, "gather_idx", indices);
    gather(g, scope, x, &idx, axis)
}

/// `GatherElements` along `axis` (indices have the output shape).
pub fn gather_elements(g: &mut Graph, scope: &str, x: &str, indices: &str, axis: i64) -> String {
    let shape = g.shape(indices).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "GatherElements", &[x, indices], &[("axis", Attr::Int(axis))], dtype, &shape)
}

/// `ScatterElements` along `axis` (updates/indices have the same shape as the output patch).
pub fn scatter_elements(g: &mut Graph, scope: &str, x: &str, indices: &str, updates: &str, axis: i64) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(
        scope,
        "ScatterElements",
        &[x, indices, updates],
        &[("axis", Attr::Int(axis)), ("reduction", Attr::Str("none".into()))],
        dtype,
        &shape,
    )
}

/// `ReduceSum` over `axes`.
pub fn reduce_sum(g: &mut Graph, scope: &str, x: &str, axes: &[i64], keepdims: bool) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = if keepdims {
        s.iter()
            .enumerate()
            .map(|(i, d)| if axes.contains(&(i as i64)) { Dim::Static(1) } else { d.clone() })
            .collect()
    } else {
        s.iter().enumerate().filter(|(i, _)| !axes.contains(&(*i as i64))).map(|(_, d)| d.clone()).collect()
    };
    let axes_t = c_vec(g, scope, "reduce_axes", axes);
    let dtype = g.dtype(x);
    g.node(scope, "ReduceSum", &[x, &axes_t], &[("keepdims", Attr::Int(keepdims as i64))], dtype, &out)
}

/// `ReduceMean` over `axes`.
pub fn reduce_mean(g: &mut Graph, scope: &str, x: &str, axes: &[i64], keepdims: bool) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = if keepdims {
        s.iter()
            .enumerate()
            .map(|(i, d)| if axes.contains(&(i as i64)) { Dim::Static(1) } else { d.clone() })
            .collect()
    } else {
        s.iter().enumerate().filter(|(i, _)| !axes.contains(&(*i as i64))).map(|(_, d)| d.clone()).collect()
    };
    let axes_t = c_vec(g, scope, "reduce_axes", axes);
    let dtype = g.dtype(x);
    g.node(scope, "ReduceMean", &[x, &axes_t], &[("keepdims", Attr::Int(keepdims as i64))], dtype, &out)
}

/// `CumSum` along `axis`.
pub fn cumsum(g: &mut Graph, scope: &str, x: &str, axis: i64) -> String {
    let axis_t = c_scalar_i64(g, scope, axis);
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "CumSum", &[x, &axis_t], &[("exclusive", Attr::Int(0)), ("reverse", Attr::Int(0))], dtype, &shape)
}

/// `Expand` to a target shape (broadcast-compatible with `x`). Unlike `Reshape`, `Expand`
/// has no "copy this dimension" zero, so a leading [`Dim::Batch`] is read at run time
/// (`Shape` -> `Gather(0)` -> `Concat` with the static tail).
pub fn expand(g: &mut Graph, scope: &str, x: &str, dims: &[Dim]) -> String {
    let shape = if dims.iter().any(|d| d.is_batch()) {
        assert!(dims[0].is_batch(), "the batch dimension leads");
        let s = shape_of(g, scope, x);
        let b = gather_static(g, scope, &s, &[0], 0);
        let tail: Vec<i64> = dims[1..].iter().map(|d| d.get().expect("static dims after the batch")).collect();
        let tail_t = c_vec(g, scope, "expand_tail", &tail);
        let rank = [Dim::Static(dims.len() as i64)];
        g.node(scope, "Concat", &[&b, &tail_t], &[("axis", Attr::Int(0))], DType::I64, &rank)
    } else {
        shape_tensor(g, scope, dims)
    };
    let dtype = g.dtype(x);
    g.node(scope, "Expand", &[x, &shape], &[], dtype, dims)
}

/// `Tile` with static repeats.
pub fn tile(g: &mut Graph, scope: &str, x: &str, reps: &[i64]) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = s
        .iter()
        .zip(reps)
        .map(|(d, &r)| match d {
            Dim::Static(v) => Dim::Static(v * r),
            other => other.clone(),
        })
        .collect();
    let reps_t = c_vec(g, scope, "reps", reps);
    let dtype = g.dtype(x);
    g.node(scope, "Tile", &[x, &reps_t], &[], dtype, &out)
}

/// `OneHot` (indices i64, depth scalar, values [off, on]).
pub fn one_hot(g: &mut Graph, scope: &str, indices: &str, depth: i64, axis: i64) -> String {
    let depth_t = c_scalar_i64(g, scope, depth);
    let values = c_f32(g, &format!("{scope}/one_hot_values"), &[2], &[0.0, 1.0]);
    let mut out = g.shape(indices).to_vec();
    let ax = if axis < 0 { (out.len() as i64 + 1 + axis) as usize } else { axis as usize };
    out.insert(ax, Dim::Static(depth));
    g.node(scope, "OneHot", &[indices, &depth_t, &values], &[("axis", Attr::Int(axis))], DType::F32, &out)
}

/// `Clip` with static bounds.
pub fn clip(g: &mut Graph, scope: &str, x: &str, min: f32, max: f32) -> String {
    let min_t = c_scalar(g, scope, min);
    let max_t = c_scalar(g, scope, max);
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "Clip", &[x, &min_t, &max_t], &[], dtype, &shape)
}

/// `Shape` of a value (runtime int64 vector) — for the few batch-dependent index tensors.
pub fn shape_of(g: &mut Graph, scope: &str, x: &str) -> String {
    let rank = g.shape(x).len() as i64;
    g.node(scope, "Shape", &[x], &[], DType::I64, &[Dim::Static(rank)])
}

/// `ConstantOfShape` with a float scalar value.
pub fn constant_of_shape(g: &mut Graph, scope: &str, shape_tensor: &str, value: f32, dims: &[Dim]) -> String {
    let t = onnx::tensor_f32(&format!("{scope}/cos_value"), &[1], &[value]);
    g.node(scope, "ConstantOfShape", &[shape_tensor], &[("value", Attr::Tensor(t))], DType::F32, dims)
}

/// `Range(start, limit, delta)` — runtime index vectors (e.g. batch offsets).
pub fn range(g: &mut Graph, scope: &str, start: &str, limit: &str, delta: &str, max_len: Dim) -> String {
    g.node(scope, "Range", &[start, limit, delta], &[], DType::F32, &[max_len])
}

/// A Conv1d: x [B, C_in, T], weight [C_out, C_in, K] (PyTorch layout, as stored in the
/// checkpoints), bias [C_out]; out [B, C_out, T'].
pub fn conv1d(
    g: &mut Graph,
    scope: &str,
    x: &str,
    w: &str,
    b: &str,
    padding: i64,
    dilation: i64,
    stride: i64,
) -> String {
    let (sx, sw) = (g.shape(x), g.shape(w));
    assert_eq!(sx.len(), 3, "conv1d input [B, C, T]");
    let (t, k) = (sx[2].clone(), sw[2].get().expect("static kernel"));
    let t_out = match t {
        Dim::Static(t) => Dim::Static((t + 2 * padding - dilation * (k - 1) - 1) / stride + 1),
        other => other,
    };
    let out = [sx[0].clone(), sw[0].clone(), t_out];
    let dtype = g.dtype(x);
    g.node(
        scope,
        "Conv",
        &[x, w, b],
        &[("dilations", Attr::Ints(vec![dilation])), ("group", Attr::Int(1)), ("kernel_shape", Attr::Ints(vec![k])), ("pads", Attr::Ints(vec![padding, padding])), ("strides", Attr::Ints(vec![stride]))],
        dtype,
        &out,
    )
}

/// Temporal 2x nearest-neighbour upsampling of [B, C, T] (the decoder's Resize pattern:
/// asymmetric, nearest_mode=floor).
pub fn upsample2x(g: &mut Graph, scope: &str, x: &str) -> String {
    let roi = c_f32(g, &format!("{scope}/resize_roi"), &[0], &[]);
    let scales = c_f32(g, &format!("{scope}/resize_scales"), &[3], &[1.0, 1.0, 2.0]);
    let s = g.shape(x);
    let out = [s[0].clone(), s[1].clone(), match &s[2] {
        Dim::Static(t) => Dim::Static(t * 2),
        other => other.clone(),
    }];
    let dtype = g.dtype(x);
    g.node(
        scope,
        "Resize",
        &[x, &roi, &scales],
        &[("coordinate_transformation_mode", Attr::Str("asymmetric".into())), ("mode", Attr::Str("nearest".into())), ("nearest_mode", Attr::Str("floor".into()))],
        dtype,
        &out,
    )
}

/// An `Identity` (used to give a value a second name, e.g. an output alias).
pub fn identity(g: &mut Graph, scope: &str, x: &str) -> String {
    let shape = g.shape(x).to_vec();
    let dtype = g.dtype(x);
    g.node(scope, "Identity", &[x], &[], dtype, &shape)
}

/// A 2-D weight initializer already transposed for direct `MatMul` (`linear`): the
/// checkpoints store PyTorch `[out, in]`; this stores `[in, out]` under the same name,
/// folding the per-layer Transpose the tracer emits (like upstream's constant folding).
pub fn matmul_weight(g: &mut Graph, name: &str, out: i64, in_: i64, data: &[f32]) -> String {
    assert_eq!(data.len(), (out * in_) as usize, "{name}");
    let mut t = vec![0f32; data.len()];
    for o in 0..out as usize {
        for i in 0..in_ as usize {
            t[i * out as usize + o] = data[o * in_ as usize + i];
        }
    }
    c_f32(g, name, &[in_, out], &t)
}

/// `Cast` every element of an f32 tensor to f16 (for the mixed-precision pass).
pub fn tensor_to_f16(t: &TensorProto) -> TensorProto {
    let data: Vec<f32> = bytemuck::cast_slice(&t.raw_data).to_vec();
    let half: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
    onnx::tensor_f16(&t.name, &t.dims, &half)
}

/// The inverse of [`tensor_to_f16`], for tests.
pub fn tensor_f16_to_f32(t: &TensorProto) -> Vec<f32> {
    assert_eq!(t.data_type, onnx::tensor_proto::DataType::Float16 as i32);
    bytemuck::cast_slice::<u8, u16>(&t.raw_data)
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect()
}
