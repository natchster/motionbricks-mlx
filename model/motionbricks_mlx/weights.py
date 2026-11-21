# SPDX-License-Identifier: Apache-2.0
"""Load the released PyTorch SafeTensors checkpoints into the MLX modules.

PyTorch -> MLX name mapping is mechanical:

* the checkpoint's wrapper prefix (``backbone_net.`` / ``mmm_net.`` / ``pose_net.``) is
  replaced by the submodule (``root.`` / ``pose.`` / ``vqvae.``);
* leading underscores are dropped from every path component, because MLX treats
  ``_private`` attributes as non-parameters (``_proj_input`` -> ``proj_input``);
* Conv1d kernels are transposed from ``(out, in, k)`` to MLX's channels-last ``(out, k, in)``;
* training-only buffers (codebook EMA statistics, ``initted`` flags) are dropped.

The checkpoints are licensed under the NVIDIA Open Model Agreement; nothing here
redistributes them.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .config import ModelSpec

IGNORED_SUFFIXES = ("initted", "cluster_size", "embed_avg")
FORMAT_VERSION = "1"


def _mlx_name(torch_name: str) -> str:
    name = ".".join(part.lstrip("_") for part in torch_name.split("."))
    # nn.MultiheadAttention packs q/k/v as raw parameters; the port uses an nn.Linear.
    return name.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight").replace(
        "self_attn.in_proj_bias", "self_attn.in_proj.bias")


def read_checkpoints(checkpoint_dir: str | Path, spec: ModelSpec) -> tuple[dict[str, mx.array], np.ndarray, np.ndarray]:
    """Read the three SafeTensors files + stats from a Hugging Face model folder
    (e.g. ``checkpoints/MotionBricks/soma``). Returns (torch-named state, mean, std)."""
    root = Path(checkpoint_dir).expanduser()
    state: dict[str, mx.array] = {}
    for kind, rel in spec.checkpoint_files.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {path}. Run sandbox/setup.sh or `hf download nvidia/MotionBricks`.")
        prefix = spec.state_prefixes[kind]
        target = {"pose": "pose.", "root": "root.", "vqvae": "vqvae."}[kind]
        for name, value in mx.load(str(path)).items():
            if name.startswith(prefix):
                state[target + name.removeprefix(prefix)] = value
    mean = np.load(root / spec.stats_files[0], allow_pickle=False)
    std = np.load(root / spec.stats_files[1], allow_pickle=False)
    return state, mean, std


def convert_state(core, torch_state: dict[str, mx.array]) -> dict[str, mx.array]:
    """Rename/transposes a torch-named state for ``core`` (a ``MotionBricksCore``), strictly."""
    expected = dict(tree_flatten(core.parameters()))
    converted: dict[str, mx.array] = {}
    for name, value in torch_state.items():
        if name.endswith(IGNORED_SUFFIXES):
            continue
        key = _mlx_name(name)
        if key not in expected:
            raise KeyError(f"Unexpected checkpoint tensor {name!r} (mapped to {key!r}).")
        if value.ndim == 3 and key.endswith(".weight"):  # Conv1d kernel
            value = value.transpose(0, 2, 1)
        if value.shape != expected[key].shape:
            raise ValueError(f"Shape mismatch for {key}: checkpoint {value.shape}, model {expected[key].shape}.")
        converted[key] = value.astype(mx.float32)
    missing = sorted(set(expected) - set(converted))
    if missing:
        raise KeyError(f"Checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    return converted


def save_converted(path: str | Path, spec: ModelSpec, params: dict[str, mx.array], mean, std, dtype=mx.float32) -> Path:
    """Write one self-contained MLX file (params + normalization stats)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.astype(dtype) for k, v in params.items()}
    arrays["stats.mean"] = mx.array(np.asarray(mean, np.float32))
    arrays["stats.std"] = mx.array(np.asarray(std, np.float32))
    mx.save_safetensors(str(path), arrays, metadata={"model": spec.name, "format": FORMAT_VERSION})
    return path


def read_converted(path: str | Path) -> tuple[str, dict[str, mx.array], np.ndarray, np.ndarray]:
    arrays, metadata = mx.load(str(path), return_metadata=True)
    if metadata.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a motionbricks-mlx v{FORMAT_VERSION} weight file.")
    mean = np.array(arrays.pop("stats.mean"))
    std = np.array(arrays.pop("stats.std"))
    return metadata["model"], arrays, mean, std


def assign(core, params: dict[str, mx.array]) -> None:
    core.update(tree_unflatten(list(params.items())))
