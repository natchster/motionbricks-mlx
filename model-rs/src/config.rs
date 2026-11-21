// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Architecture specs of the two released MotionBricks models
//! (transcribed from `motionbricks_mlx/config.py`).

pub const MIN_TOKENS: i32 = 6;
pub const MAX_TOKENS: i32 = 16;
pub const DOWN_T: i32 = 2;
pub const FRAMES_PER_TOKEN: i32 = 4;
pub const MAX_FRAMES: i32 = MAX_TOKENS * FRAMES_PER_TOKEN; // 64
pub const NUM_TOKEN_CHOICES: i32 = MAX_TOKENS - MIN_TOKENS + 1; // 11
pub const MASKED_NUM_TOKENS: i32 = MAX_TOKENS + 2; // "let the model choose"

#[derive(Debug, Clone, Copy)]
pub struct RootSpec {
    pub n_embd: i32,
    pub n_head: i32,
    pub n_layers_shared: usize,
    pub n_layers_root_token: usize,
    pub pose_feat_dim: i32,
    pub width: i32,
    pub depth: usize,
    pub dilation_growth_rate: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PoseSpec {
    pub n_embd: i32,
    pub n_head: i32,
    pub n_layers: usize,
    pub pose_feat_width: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct VqSpec {
    pub num_heads: i32,
    pub codes_per_head: i32,
    pub code_dim: i32,
    pub width: i32,
    pub depth: usize,
    pub dilation_growth_rate: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalRootMode {
    FromPoseModule,
    FromHeightConstraintViolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    UnitreeG1,
    Soma,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub kind: Kind,
    pub name: &'static str,
    pub num_joints: i32,
    pub root: RootSpec,
    pub pose: PoseSpec,
    pub vq: VqSpec,
    pub final_root_mode: FinalRootMode,
    /// (module, path relative to the Hugging Face model folder, checkpoint prefix)
    pub checkpoint_files: [(&'static str, &'static str, &'static str); 3],
    pub stats_files: (&'static str, &'static str),
}

const fn root(n_layers_shared: usize, n_layers_root_token: usize) -> RootSpec {
    RootSpec {
        n_embd: 512,
        n_head: 16,
        n_layers_shared,
        n_layers_root_token,
        pose_feat_dim: 256,
        width: 512,
        depth: 4,
        dilation_growth_rate: 3,
    }
}

const POSE: PoseSpec = PoseSpec { n_embd: 1024, n_head: 16, n_layers: 16, pose_feat_width: 640 };

const fn vq(num_heads: i32, codes_per_head: i32, code_dim: i32) -> VqSpec {
    VqSpec { num_heads, codes_per_head, code_dim, width: 512, depth: 4, dilation_growth_rate: 3 }
}

pub fn g1() -> ModelSpec {
    ModelSpec {
        kind: Kind::UnitreeG1,
        name: "unitree_g1",
        num_joints: 34,
        root: root(3, 3),
        pose: POSE,
        vq: vq(8, 10, 256),
        final_root_mode: FinalRootMode::FromPoseModule,
        checkpoint_files: [
            ("pose", "out/motionbricks_pose/version_1/checkpoints/model-step=2000000.safetensors", "backbone_net."),
            ("root", "out/motionbricks_root/version_1/checkpoints/model-step=2000000.safetensors", "backbone_net."),
            ("vqvae", "out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.safetensors", "pose_net."),
        ],
        stats_files: (
            "out/motionbricks_pose/version_1/stats/motion/mean.npy",
            "out/motionbricks_pose/version_1/stats/motion/std.npy",
        ),
    }
}

pub fn soma() -> ModelSpec {
    ModelSpec {
        kind: Kind::Soma,
        name: "soma",
        num_joints: 30,
        root: root(6, 2),
        pose: POSE,
        vq: vq(5, 16, 160),
        final_root_mode: FinalRootMode::FromHeightConstraintViolation,
        checkpoint_files: [
            (
                "pose",
                "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
                "mmm_net.",
            ),
            (
                "root",
                "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-root-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
                "mmm_net.",
            ),
            (
                "vqvae",
                "soma-v6-nova-4n8g-ea43dd8-fp32-v1/vqvae-v6-soma-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
                "pose_net.",
            ),
        ],
        stats_files: (
            "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/mean.npy",
            "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/std.npy",
        ),
    }
}

pub fn get_spec(name: &str) -> anyhow::Result<ModelSpec> {
    match name {
        "unitree_g1" | "g1" => Ok(g1()),
        "soma" => Ok(soma()),
        other => anyhow::bail!("unknown model {other:?}; choose 'soma' or 'unitree_g1'"),
    }
}
