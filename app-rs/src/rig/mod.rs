// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Rig assets for the applications: each rig's visual mesh, skeleton, clip registry and
//! pose -> geometry evaluation (ports upstream `app/runtime/common/rig.py` and
//! `clip_registry.py`, plus the loading half of `motionbricks/io/usd.py`,
//! `motionbricks/io/utils.py` and `motionbricks/renderers/*`).
//!
//! - [`mesh`]: the `.mbmesh` reader for the embedded SOMA skinned mesh.
//! - [`soma`]: the SOMA human rig (skinned, FK + LBS bone matrices, sole outlines).
//! - [`g1`]: the Unitree G1 robot rig (MJCF body tree + STL links, rigid FK).
//! - [`clips`]: the per-rig clip registry (BVH / CSV, cached, loopable copies).

pub mod clips;
pub mod g1;
pub mod mesh;
pub mod soma;
