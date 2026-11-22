// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The released model specs (from upstream `tools/onnx/models/**` and the shared
//! `model-rs/src/config.rs`), the ONNX tensor contract, and the node-name prefixes the
//! mixed-FP16 conversion keys on.

pub const MIN_TOKENS: i64 = 6;
pub const MAX_TOKENS: i64 = 16;
pub const FRAMES_PER_TOKEN: i64 = 4;
pub const MAX_FRAMES: i64 = MAX_TOKENS * FRAMES_PER_TOKEN; // 64
pub const NUM_TOKEN_CHOICES: i64 = MAX_TOKENS - MIN_TOKENS + 1; // 11
/// "Let the model choose" sentinel duration written into the request.
pub const MASKED_NUM_TOKENS: i64 = MAX_TOKENS + 2; // 18
/// Context + target slots of a request (4 context frames, 4 target frames).
pub const CONTEXT_AND_TARGET: i64 = 8;
/// Values per G1 MuJoCo pose: root xyz, root quaternion (XYZW), 29 hinge angles.
pub const G1_QPOS_DIM: i64 = 36;
pub const FPS: i64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    G1,
    Soma,
}

/// One safetensors checkpoint file: the module it loads into and the tensor-name prefix
/// the file's entries carry (stripped on load, like upstream's strict load).
#[derive(Debug, Clone, Copy)]
pub struct CheckpointFile {
    pub module: &'static str,
    pub path: &'static str,
    pub prefix: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub kind: Kind,
    /// Upstream's short name ("g1" / "soma"), used in output file names.
    pub kind_name: &'static str,
    pub name: &'static str,
    pub num_joints: i64,
    /// Root network: shared / root-token transformer depths (d_model 512, 16 heads).
    pub root_layers_shared: usize,
    pub root_layers_token: usize,
    /// VQ: heads, codes per head, embedding width per head.
    pub vq_heads: i64,
    pub vq_codes: i64,
    pub vq_code_dim: i64,
    /// Checkpoint layout (pose, root, vqvae), paths relative to the model's HF folder.
    pub checkpoints: [CheckpointFile; 3],
    pub stats_files: (&'static str, &'static str),
    /// Node-name prefixes the mixed-FP16 conversion selects (upstream `precision.py`).
    pub fp16_prefixes: &'static [&'static str],
}

pub const G1: ModelSpec = ModelSpec {
    kind: Kind::G1,
    kind_name: "g1",
    name: "unitree_g1",
    num_joints: 34,
    root_layers_shared: 3,
    root_layers_token: 3,
    vq_heads: 8,
    vq_codes: 10,
    vq_code_dim: 256,
    checkpoints: [
        CheckpointFile { module: "pose", path: "out/motionbricks_pose/version_1/checkpoints/model-step=2000000.safetensors", prefix: "backbone_net." },
        CheckpointFile { module: "root", path: "out/motionbricks_root/version_1/checkpoints/model-step=2000000.safetensors", prefix: "backbone_net." },
        CheckpointFile { module: "vqvae", path: "out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.safetensors", prefix: "pose_net." },
    ],
    stats_files: (
        "out/motionbricks_pose/version_1/stats/motion/mean.npy",
        "out/motionbricks_pose/version_1/stats/motion/std.npy",
    ),
    fp16_prefixes: &["/core/backbone_net/", "/core/decoder/"],
};

pub const SOMA: ModelSpec = ModelSpec {
    kind: Kind::Soma,
    kind_name: "soma",
    name: "soma",
    num_joints: 30,
    root_layers_shared: 6,
    root_layers_token: 2,
    vq_heads: 5,
    vq_codes: 16,
    vq_code_dim: 160,
    checkpoints: [
        CheckpointFile {
            module: "pose",
            path: "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
            prefix: "mmm_net.",
        },
        CheckpointFile {
            module: "root",
            path: "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-root-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
            prefix: "mmm_net.",
        },
        CheckpointFile {
            module: "vqvae",
            path: "soma-v6-nova-4n8g-ea43dd8-fp32-v1/vqvae-v6-soma-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
            prefix: "pose_net.",
        },
    ],
    stats_files: (
        "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/mean.npy",
        "soma-v6-nova-4n8g-ea43dd8-fp32-v1/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/std.npy",
    ),
    fp16_prefixes: &["/mmm_net/", "/decoder/"],
};

pub fn get_spec(name: &str) -> anyhow::Result<&'static ModelSpec> {
    match name {
        "g1" | "unitree_g1" => Ok(&G1),
        "soma" => Ok(&SOMA),
        other => anyhow::bail!("unknown model {other:?}; choose 'soma' or 'unitree_g1'"),
    }
}

pub const G1_INPUTS: [&str; 6] = [
    "context_qpos", "target_qpos", "has_global_root_values", "has_local_root_values", "has_local_poses",
    "allowed_pred_num_tokens",
];
pub const G1_OUTPUTS: [&str; 3] = ["pred_qpos", "pred_num_tokens", "pred_foot_contacts"];
pub const SOMA_INPUTS: [&str; 7] = [
    "global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values", "local_poses",
    "has_local_poses", "allowed_pred_num_tokens",
];
pub const SOMA_OUTPUTS: [&str; 2] = ["pred_global_motions", "pred_num_tokens"];

// Upstream rig assets (Apache-2.0, NVIDIA), shared with the Python package and model-rs.
const G1_RIG: &[u8] = include_bytes!("../../../model/motionbricks_mlx/assets/g1_model_rig.npz");
const SOMA_RIG: &[u8] = include_bytes!("../../../model/motionbricks_mlx/assets/soma_model_rig.npz");

impl ModelSpec {
    /// The upstream rig asset (`.npz`) of this model.
    pub fn rig_asset(&self) -> &'static [u8] {
        match self.kind {
            Kind::G1 => G1_RIG,
            Kind::Soma => SOMA_RIG,
        }
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::G1 => &G1_INPUTS,
            Kind::Soma => &SOMA_INPUTS,
        }
    }

    pub fn output_names(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::G1 => &G1_OUTPUTS,
            Kind::Soma => &SOMA_OUTPUTS,
        }
    }

    /// Width of the SOMA `local_poses` input: joint positions (J-1)*3 + 6D rotations J*6.
    pub fn local_poses_dim(&self) -> i64 {
        (self.num_joints - 1) * 3 + self.num_joints * 6
    }

    /// The default output file name, like upstream's exporter.
    pub fn default_output(&self, precision: &str) -> String {
        format!("mmm_v6_{}_{}.onnx", self.kind_name, precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract() {
        assert_eq!(SOMA.local_poses_dim(), 267);
        assert_eq!(G1_INPUTS.len(), 6);
        assert_eq!(SOMA_INPUTS.len(), 7);
        assert_eq!(G1.default_output("fp32"), "mmm_v6_g1_fp32.onnx");
        assert!(get_spec("typo").is_err());
    }
}
