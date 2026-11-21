// SPDX-License-Identifier: Apache-2.0
// Ported from NVlabs/motionbricks docs/examples/_rig.py. Original: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
//! Shared setup for the Runtime API examples (included by `generate_once.rs` and
//! `locomotion_loop.rs`): a rig's skeleton, clip and model, without a viewer. Needs the
//! checkpoints and the sample clips: `scripts/download_weights.sh all` and `... clips`.
#![allow(dead_code)]

use std::path::PathBuf;
use std::rc::Rc;

use anyhow::Result;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::runtime::*;

pub const WALK: &str = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057";
pub const JOG: &str = "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534";

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// The clip of one rig: local joint transforms (SOMA) or joint coordinates (G1).
pub enum Clip {
    Soma { clip: AnimationBuffer, skeleton: Rc<Skeleton> },
    G1(DofAnimationBuffer),
}

pub fn load_clip(model: &str, name: &str) -> Result<Clip> {
    let dir = root().join("checkpoints/motions").join(model);
    Ok(if model == "soma" {
        let path = dir.join(format!("{name}.bvh"));
        // The BVH skeleton without its identity Root joint: Hips first, as the engine needs.
        let skeleton = Rc::new(load_skeleton_from_bvh(&path, true, true)?);
        Clip::Soma { clip: load_bvh(&path, Some(skeleton.clone()), true)?, skeleton }
    } else {
        Clip::G1(load_csv(dir.join(format!("{name}.csv")), CSV_SAMPLE_RATE)?)
    })
}

/// The Metal runtime for one rig at a precision (float32, float16, bfloat16, int8), batch 1.
pub fn load_model(model: &str, precision: &str) -> Result<MetalModel> {
    let folder = if model == "soma" { "soma" } else { "g1" };
    MetalModel::from_checkpoints_with(model, root().join("checkpoints/MotionBricks").join(folder), 1, precision)
}

/// Four SOMA clip poses at 30 Hz from `start`, roots moved by `offset * model_to_runtime`.
pub fn soma_window(clip: &AnimationBuffer, start: f32, offset: Transform) -> Vec<Vec<Transform>> {
    let o = offset.mul(model_to_runtime(ModelRig::Soma));
    (0..MODEL_POSES_PER_TOKEN).map(|i| clip.sample(start + i as f32 / MODEL_SAMPLE_RATE, o).unwrap()).collect()
}

/// Four G1 clip poses at 30 Hz from `start`, roots moved by `offset` (G1 model space is world space).
pub fn dof_window(clip: &DofAnimationBuffer, start: f32, offset: Transform) -> Vec<Vec<f32>> {
    (0..MODEL_POSES_PER_TOKEN).map(|i| clip.sample(start + i as f32 / MODEL_SAMPLE_RATE, offset).unwrap()).collect()
}
