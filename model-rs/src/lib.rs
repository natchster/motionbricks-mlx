// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! # motionbricks-rs
//!
//! Rust port of `motionbricks_mlx`, the MLX port of NVIDIA's MotionBricks (SOMA and Unitree G1),
//! on [mlx-rs](https://github.com/oxiglade/mlx-rs). Same graph, same Metal kernels, no Python.
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

pub use model::{MotionBricks, compiled, convert, parse_dtype};
