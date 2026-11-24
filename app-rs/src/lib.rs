// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! # motionbricks-app
//!
//! The MotionBricks applications in Rust: the Inbetweener and Scene Preview demos and the
//! Smart Primitive and Scene editors. They run on a small wgpu viewer with the same
//! logging-shaped API as upstream's Newton ViewerGL, and the generation apps call the
//! Rust model and Rust Runtime API ([`motionbricks_rs`], `model-rs/`) — never Python.

pub mod documents;
pub mod editor;
pub mod rig;
pub mod runtime;
pub mod viewer;
