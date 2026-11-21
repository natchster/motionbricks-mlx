# SPDX-License-Identifier: Apache-2.0
# Hyper-parameters transcribed from NVlabs/motionbricks tools/onnx/models/{g1,soma}/model.py.
"""Architecture specs of the two released MotionBricks models."""

from __future__ import annotations

from dataclasses import dataclass

MIN_TOKENS = 6
MAX_TOKENS = 16
DOWN_T = 2
FRAMES_PER_TOKEN = 2**DOWN_T  # 4 poses per motion token
MAX_FRAMES = MAX_TOKENS * FRAMES_PER_TOKEN  # 64
NUM_TOKEN_CHOICES = MAX_TOKENS - MIN_TOKENS + 1  # 11 (6..16)
MASKED_NUM_TOKENS = MAX_TOKENS + 2  # "let the model choose"
CONTEXT_AND_TARGET = 2 * FRAMES_PER_TOKEN  # 8 conditioning slots


@dataclass(frozen=True)
class RootSpec:
    n_embd: int = 512
    n_head: int = 16
    n_layers_shared: int = 6
    n_layers_root_token: int = 2
    pose_feat_dim: int = 256
    local_root_feat_dim: int = 64
    global_root_feat_dim: int = 64
    input_feat_mlp_num_layers: int = 2
    width: int = 512
    depth: int = 4
    dilation_growth_rate: int = 3


@dataclass(frozen=True)
class PoseSpec:
    n_embd: int = 1024
    n_head: int = 16
    n_layers: int = 16
    pose_token_mlp_num_layers: int = 2
    root_feat_width: int = 256
    pose_feat_width: int = 640
    token_length_feat_width: int = 128


@dataclass(frozen=True)
class VQSpec:
    num_heads: int
    codes_per_head: int
    code_dim: int  # all heads together
    width: int = 512
    depth: int = 4
    dilation_growth_rate: int = 3

    @property
    def code_dim_per_head(self) -> int:
        return self.code_dim // self.num_heads


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_joints: int
    root: RootSpec
    pose: PoseSpec
    vq: VQSpec
    final_root_mode: str  # "from_pose_module" (G1) | "from_height_constraint_violation" (SOMA)
    # Checkpoint layout on huggingface.co/nvidia/MotionBricks, relative to the model folder.
    checkpoint_files: dict[str, str]
    stats_files: tuple[str, str]
    state_prefixes: dict[str, str]


G1 = ModelSpec(
    name="unitree_g1",
    num_joints=34,
    root=RootSpec(n_layers_shared=3, n_layers_root_token=3),
    pose=PoseSpec(),
    vq=VQSpec(num_heads=8, codes_per_head=10, code_dim=256),  # 10**8 overall codes
    final_root_mode="from_pose_module",
    checkpoint_files={
        "pose": "out/motionbricks_pose/version_1/checkpoints/model-step=2000000.safetensors",
        "root": "out/motionbricks_root/version_1/checkpoints/model-step=2000000.safetensors",
        "vqvae": "out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.safetensors",
    },
    stats_files=(
        "out/motionbricks_pose/version_1/stats/motion/mean.npy",
        "out/motionbricks_pose/version_1/stats/motion/std.npy",
    ),
    state_prefixes={"pose": "backbone_net.", "root": "backbone_net.", "vqvae": "pose_net."},
)

_SOMA_DIR = "soma-v6-nova-4n8g-ea43dd8-fp32-v1"
SOMA = ModelSpec(
    name="soma",
    num_joints=30,
    root=RootSpec(n_layers_shared=6, n_layers_root_token=2),
    pose=PoseSpec(),
    vq=VQSpec(num_heads=5, codes_per_head=16, code_dim=160),  # 16**5 = 1,048,576 overall codes
    final_root_mode="from_height_constraint_violation",
    checkpoint_files={
        "pose": f"{_SOMA_DIR}/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
        "root": f"{_SOMA_DIR}/mmm-v6-soma-root-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
        "vqvae": f"{_SOMA_DIR}/vqvae-v6-soma-nova-4n8g-ea43dd8-fp32-v1/version_0/checkpoints/model-step=1500000.safetensors",
    },
    stats_files=(
        f"{_SOMA_DIR}/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/mean.npy",
        f"{_SOMA_DIR}/mmm-v6-soma-pose-nova-4n8g-ea43dd8-fp32-v1/version_0/stats/motion/std.npy",
    ),
    state_prefixes={"pose": "mmm_net.", "root": "mmm_net.", "vqvae": "pose_net."},
)

SPECS = {"unitree_g1": G1, "g1": G1, "soma": SOMA}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose 'soma' or 'unitree_g1'.") from None
