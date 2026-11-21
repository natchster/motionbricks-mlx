// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (model/contract.py, model/config.py,
// model/soma/contract.py, model/unitree_g1/contract.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Rigs, timing, joint layouts and generation settings shared by the engines.

use anyhow::{Result, bail};

pub const MODEL_SAMPLE_RATE: f32 = 30.0;
pub const MODEL_POSES_PER_TOKEN: usize = 4;
pub const MODEL_CONTEXT_AND_TARGET_POSES: usize = 2 * MODEL_POSES_PER_TOKEN;
pub const MODEL_MIN_TOKEN_COUNT: i32 = 6;
pub const MODEL_MAX_TOKEN_COUNT: i32 = 16;
pub const MODEL_BOUNDARY_FRAMES: usize = MODEL_POSES_PER_TOKEN - 1;
pub const FOOT_CONTACT_COUNT: usize = 4;
/// Joint coordinates start with the root position (XYZ) and rotation (XYZW).
pub const ROOT_COORDINATE_COUNT: usize = 7;
pub const MODEL_MAX_VALID_FRAMES: usize = MODEL_MAX_TOKEN_COUNT as usize * MODEL_POSES_PER_TOKEN - 2 * MODEL_BOUNDARY_FRAMES;

/// Frames left after removing the boundary frames: 18 to 58 for 6 to 16 tokens.
pub fn valid_frame_count(token_count: i32) -> Result<usize> {
    if !(MODEL_MIN_TOKEN_COUNT..=MODEL_MAX_TOKEN_COUNT).contains(&token_count) {
        bail!("token_count must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}, got {token_count}");
    }
    Ok(token_count as usize * MODEL_POSES_PER_TOKEN - 2 * MODEL_BOUNDARY_FRAMES)
}

/// SOMA poses are local joint transforms; Unitree G1 poses are joint coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelRig {
    Soma,
    UnitreeG1,
}

impl ModelRig {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "soma" => Ok(ModelRig::Soma),
            "unitree_g1" | "g1" => Ok(ModelRig::UnitreeG1),
            other => bail!("unknown rig {other:?}"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ModelRig::Soma => "soma",
            ModelRig::UnitreeG1 => "unitree_g1",
        }
    }
}

pub const SOMA_JOINT_NAMES: [&str; 30] = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm",
    "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
    "RightFoot", "RightToeBase",
];
pub const SOMA_FOOT_CONTACT_JOINTS: [&str; 4] = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"];
/// The right and left leg joints whose positions give a pose's heading, as in training.
pub const SOMA_HEADING_JOINTS: [&str; 2] = ["RightLeg", "LeftLeg"];

pub const UNITREE_G1_JOINT_NAMES: [&str; 29] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
];
pub const UNITREE_G1_FOOT_CONTACT_BODIES: [&str; 4] = ["left_ankle_roll_link", "left_ankle_roll_link", "right_ankle_roll_link", "right_ankle_roll_link"];
pub const UNITREE_G1_FOOT_CONTACT_OFFSETS: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [0.14, 0.0, -0.035], [0.0, 0.0, 0.0], [0.14, 0.0, -0.035]];

/// Generation settings for output length, heading alignment and endpoint correction.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionBricksInferenceConfig {
    /// Token counts the model may choose from, each 6 to 16 (default 8 to 14).
    pub allowed_token_counts: Vec<i32>,
    /// Spread small corrections over the motion so it starts at the last context pose and ends
    /// at the first target pose.
    pub correct_generated_diff: bool,
    /// Ground direction in model space that inference turns the last context pose to face:
    /// (x, z) for SOMA, (x, y) for G1. (1, 0) faces +X.
    pub heading_alignment: [f32; 2],
}

impl Default for MotionBricksInferenceConfig {
    fn default() -> Self {
        MotionBricksInferenceConfig { allowed_token_counts: (8..=14).collect(), correct_generated_diff: true, heading_alignment: [1.0, 0.0] }
    }
}

impl MotionBricksInferenceConfig {
    pub fn validate(&self) -> Result<()> {
        if self.allowed_token_counts.is_empty() {
            bail!("allowed_token_counts cannot be empty");
        }
        if !self.allowed_token_counts.iter().all(|c| (MODEL_MIN_TOKEN_COUNT..=MODEL_MAX_TOKEN_COUNT).contains(c)) {
            bail!("allowed_token_counts values must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}");
        }
        let [x, y] = self.heading_alignment;
        if !x.is_finite() || !y.is_finite() || (x * x + y * y).sqrt() <= 1.0e-6 {
            bail!("heading_alignment must be a finite, non-zero direction");
        }
        Ok(())
    }

    /// The allowed token counts as a bit mask, bit 0 for 6 tokens.
    pub fn allowed_token_bits(&self) -> u32 {
        self.allowed_token_counts.iter().fold(0, |bits, c| bits | 1 << (c - MODEL_MIN_TOKEN_COUNT))
    }
}
