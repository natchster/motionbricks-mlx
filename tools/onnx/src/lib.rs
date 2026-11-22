// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! # motionbricks-onnx
//!
//! Emits the MotionBricks batched inference graphs (`mmm_v6_{g1,soma}_{fp32,fp16}.onnx`)
//! directly from the released checkpoints, without PyTorch or tracing. The graphs are
//! semantically equivalent to upstream's exported ONNX files (same tensor interface,
//! same outputs under ONNX Runtime) with ~6x fewer nodes, because every shape the
//! tracer computes at run time is resolved here at build time.

pub mod config;
pub mod emit;
pub mod export;
pub mod fp16;
pub mod graph;
pub mod inference;
pub mod layers;
pub mod networks;
pub mod npy;
pub mod onnx;
pub mod ops;
pub mod representation;
pub mod rig_g1;
pub mod soma;
pub mod validate;
pub mod weights;
