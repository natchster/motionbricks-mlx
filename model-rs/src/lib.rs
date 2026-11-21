// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! # motionbricks-rs
//!
//! NVIDIA's MotionBricks (SOMA and Unitree G1) in Rust, without Python:
//!
//! * [`MotionBricks`]: an op-for-op port of `motionbricks_mlx` (the MLX port) on
//!   [mlx-rs](https://github.com/oxiglade/mlx-rs): the same graph and the same kernels;
//! * [`metal::model::MetalModel`]: a model-specific Metal runtime (static dispatch plan, fused
//!   kernels, tuned GEMM plans), about 2x faster at batch 1;
//! * [`runtime`]: upstream's Runtime API (engines, tokens, players, clips) on either one.
//!
//! [`config`] holds what both runtimes share: the model specs, the ONNX tensor names, the
//! checkpoint layout and [`Precision`].
//!
//! Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
//! Port: Apache-2.0 (see the repository's LICENSE and NOTICE).

pub mod config;
pub mod examples;
pub mod inference;
pub mod kernels;
pub mod layers;
pub mod metal;
pub mod model;
pub mod networks;
pub mod npy;
pub mod ops;
pub mod representation;
pub mod rig;
pub mod runtime;
pub mod weights;

pub use config::Precision;
pub use model::{MotionBricks, compiled, convert, parse_dtype};
