// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Architecture specs of the two released MotionBricks models
//! (transcribed from `motionbricks_mlx/config.py`), and the contract both runtimes share: the
//! ONNX tensor names, the rig assets, the checkpoint layout, and the precision names.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

/// The released model families (alias of [`Kind`]).
pub type ModelKind = Kind;

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

// ---- ONNX interface (shared by the mlx-rs port, the Metal runtime and the Runtime API) ----

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
/// Inputs that are masks: converted with `!= 0` (in their own dtype), like the Python port.
pub const MASK_INPUTS: [&str; 4] = ["has_global_root_values", "has_local_root_values", "has_local_poses", "allowed_pred_num_tokens"];

pub fn is_mask_input(name: &str) -> bool {
    MASK_INPUTS.contains(&name)
}

/// Context + target slots of a request (4 context frames, 4 target frames).
pub const CONTEXT_AND_TARGET: i32 = 8;
/// Values per G1 MuJoCo pose: root xyz, root quaternion (XYZW), 29 hinge angles.
pub const G1_QPOS_DIM: i32 = 36;

// Upstream rig assets (Apache-2.0, NVIDIA), shared with the Python package.
const G1_RIG: &[u8] = include_bytes!("../../model/motionbricks_mlx/assets/g1_model_rig.npz");
const SOMA_RIG: &[u8] = include_bytes!("../../model/motionbricks_mlx/assets/soma_model_rig.npz");

impl ModelSpec {
    /// ONNX input names, in positional order.
    pub fn input_names(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::UnitreeG1 => &G1_INPUTS,
            Kind::Soma => &SOMA_INPUTS,
        }
    }

    /// ONNX output names, in positional order.
    pub fn output_names(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::UnitreeG1 => &G1_OUTPUTS,
            Kind::Soma => &SOMA_OUTPUTS,
        }
    }

    /// The upstream rig asset (`.npz`) of this model.
    pub fn rig_asset(&self) -> &'static [u8] {
        match self.kind {
            Kind::UnitreeG1 => G1_RIG,
            Kind::Soma => SOMA_RIG,
        }
    }

    /// Folder of this model inside `checkpoints/MotionBricks` (the Hugging Face layout).
    pub fn hf_folder(&self) -> &'static str {
        match self.kind {
            Kind::UnitreeG1 => "g1",
            Kind::Soma => "soma",
        }
    }

    /// Width of the SOMA `local_poses` input: joint positions `(J-1)*3` + 6D rotations `J*6`.
    pub fn local_poses_dim(&self) -> i32 {
        (self.num_joints - 1) * 3 + self.num_joints * 6
    }
}

/// `checkpoints/MotionBricks` of this repository, or `$MOTIONBRICKS_CHECKPOINTS`.
pub fn checkpoints_root() -> PathBuf {
    std::env::var_os("MOTIONBRICKS_CHECKPOINTS").map(PathBuf::from).unwrap_or_else(|| {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        crate_dir.parent().unwrap_or(crate_dir).join("checkpoints/MotionBricks")
    })
}

/// The Hugging Face checkpoint folder of model `name` under [`checkpoints_root`]. An unknown
/// name maps to `<root>/<name>`, so loading it then fails with the "unknown model" error.
pub fn checkpoint_dir(name: &str) -> PathBuf {
    let folder = get_spec(name).map(|s| s.hf_folder()).unwrap_or(name);
    checkpoints_root().join(folder)
}

// ---- Precision ----

/// Inference precision, by the names both ports use: `float32`, `float16`, `bfloat16`, `int8`.
/// INT8 keeps FP32 activations and stores the transformer projections affine-quantized
/// (8 bits, groups of 64), like the Python port's `quantize(8)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Float32,
    Float16,
    BFloat16,
    Int8,
}

impl Precision {
    pub const ALL: [Precision; 4] = [Precision::Float32, Precision::Float16, Precision::BFloat16, Precision::Int8];

    pub fn name(self) -> &'static str {
        match self {
            Precision::Float32 => "float32",
            Precision::Float16 => "float16",
            Precision::BFloat16 => "bfloat16",
            Precision::Int8 => "int8",
        }
    }

    /// Element type of the activations (INT8 computes in float32).
    pub fn compute_dtype(self) -> mlx_rs::Dtype {
        match self {
            Precision::Float32 | Precision::Int8 => mlx_rs::Dtype::Float32,
            Precision::Float16 => mlx_rs::Dtype::Float16,
            Precision::BFloat16 => mlx_rs::Dtype::Bfloat16,
        }
    }

    pub fn is_quantized(self) -> bool {
        self == Precision::Int8
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Precision {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "float32" | "fp32" => Precision::Float32,
            "float16" | "fp16" => Precision::Float16,
            "bfloat16" | "bf16" => Precision::BFloat16,
            "int8" => Precision::Int8,
            other => anyhow::bail!("unknown precision {other:?}; choose float32, float16, bfloat16 or int8"),
        })
    }
}

impl TryFrom<&str> for Precision {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        s.parse()
    }
}

impl TryFrom<&String> for Precision {
    type Error = anyhow::Error;

    fn try_from(s: &String) -> anyhow::Result<Self> {
        s.parse()
    }
}

impl TryFrom<String> for Precision {
    type Error = anyhow::Error;

    fn try_from(s: String) -> anyhow::Result<Self> {
        s.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_names_round_trip() {
        for p in Precision::ALL {
            assert_eq!(p.name().parse::<Precision>().unwrap(), p);
            assert_eq!(Precision::try_from(p.name()).unwrap(), p);
            assert_eq!(p.to_string(), p.name());
        }
        assert_eq!(Precision::try_from("bf16").unwrap(), Precision::BFloat16);
        let err = "int4".parse::<Precision>().unwrap_err().to_string();
        assert!(err.contains("float32, float16, bfloat16 or int8"), "{err}");
    }

    #[test]
    fn contract_per_model() {
        let (g1, soma) = (get_spec("g1").unwrap(), get_spec("soma").unwrap());
        assert_eq!((g1.name, g1.hf_folder(), g1.input_names().len(), g1.output_names().len()), ("unitree_g1", "g1", 6, 3));
        assert_eq!((soma.name, soma.hf_folder(), soma.input_names().len(), soma.output_names().len()), ("soma", "soma", 7, 2));
        assert_eq!(soma.local_poses_dim(), 267);
        assert!(g1.rig_asset().starts_with(b"PK") && soma.rig_asset().starts_with(b"PK"));
        assert!(MASK_INPUTS.iter().all(|m| g1.input_names().contains(m) && soma.input_names().contains(m)));
        assert!(checkpoint_dir("g1").ends_with("g1") && checkpoint_dir("unitree_g1").ends_with("g1"));
        assert!(checkpoint_dir("typo").ends_with("typo"), "unknown names must not map to a real model");
    }
}
