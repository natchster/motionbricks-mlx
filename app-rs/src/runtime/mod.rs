// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The application runtime the Scene Preview and Inbetweener demos run on, ported from
//! upstream `app/runtime/` (`common/` plus the per-rig `soma/` and `unitree_g1/` presets):
//! the character, its state machine, locomotion, Smart Primitive behaviors, and the
//! interaction probe.
//!
//! The runtime is pure behavior and math: no viewer, no wgpu, no GPU kernels. Where
//! upstream launches Warp kernels (pose blending, box overlap, target placement), this
//! port runs plain Rust loops with the same formulas. Generation goes through the
//! [`character::EngineAdapter`] trait so the state machine never sees the model; the
//! concrete [`character::PlayerAdapter`] over the `motionbricks-rs` engines lives here,
//! and the bins wire it to a Metal or MLX backend.
//!
//! - [`motion`]: locomotion modes and settings, movement smoothing, clips (`common/motion.py`).
//! - [`blender`]: blend spaces and target-pose blending (`runtime/blend_spaces.py`,
//!   `common/target_pose_blender.py`, `soma/target_pose_blender.py`, `unitree_g1/target_pose_blender.py`).
//! - [`interaction`]: the probe box and oriented-box overlap (`common/interaction.py` and
//!   the box parts of `motionbricks/runtime/smart_primitive.py`).
//! - [`behaviors`]: the Smart Primitive runtime asset/instance, pivots, and the behaviors
//!   scenes pick by class name (`motionbricks/runtime/pivot.py`,
//!   `motionbricks/runtime/smart_primitive.py`, `common/smart_primitive_behaviors.py`).
//! - [`states`]: idle, locomotion, and interaction states (`common/states.py`).
//! - [`state_machine`]: standing/crouching idle+locomotion pairs plus the shared
//!   interaction state (`common/state_machine.py`).
//! - [`character`]: three-source playback with pose recording and target-pose IK
//!   (`common/character.py`, `soma/character.py`, `unitree_g1/character.py`).
//! - [`presets`]: per-rig locomotion modes, idle clips, and probe sizes
//!   (`soma/presets.py`, `unitree_g1/presets.py`).

pub mod behaviors;
pub mod blender;
pub mod character;
pub mod interaction;
pub mod motion;
pub mod presets;
pub mod state_machine;
pub mod states;
