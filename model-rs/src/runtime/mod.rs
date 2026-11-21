// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (motionbricks/model,
// motionbricks/runtime, motionbricks/animation, motionbricks/io).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The MotionBricks Runtime API in Rust: engines, tokens, animation buffers, recorders, motion
//! players, target-pose helpers and clip loading, with upstream's behavior. Engines run on a
//! [`Backend`]: the Metal runtime ([`crate::metal::model::MetalModel`]) or the mlx-rs port
//! ([`MlxBackend`]). See `docs/runtime_api.md`.

pub mod animation;
pub mod contract;
pub mod engine;
pub mod io;
pub mod math;
pub mod player;
pub mod target_pose;
pub mod tokens;

pub use animation::{AnimationBuffer, DofAnimationBuffer, FootContacts, MotionBuffer, Skeleton, create_animation_buffer_for_skeleton, frame_index_from_time, read_root, write_root};
pub use contract::*;
pub use engine::{
    Backend, EngineCore, ForwardFn, GenerationTimer, MlxBackend, MotionBricksEngine, MotionBricksSomaEngine, MotionBricksUnitreeG1Engine, NoBackend, Prediction,
};
pub use io::{CSV_SAMPLE_RATE, FacingDirectionType, SpaceConverter, load_bvh, load_csv, load_skeleton_from_bvh, model_to_runtime, save_csv};
pub use math::{Quat, Transform, Vec3};
pub use player::{GeneratedDofMotionPlayer, GeneratedMotionPlayer, GeneratedPlayer};
pub use target_pose::{
    MAX_TRANSITION_EARLY_BY_FRAMES, TargetPoseIkCorrector, TargetPoseInferenceParams, TwoBoneIkChainConfig, adjust_target_dof_speed,
    adjust_target_pose_speed, create_soma_target_pose_ik_rig, ik_pin, place_target_dof, place_target_poses,
};
pub use tokens::{DofToken, MotionRecorder, MotionToken, PoseToken, history_sample_indices};
