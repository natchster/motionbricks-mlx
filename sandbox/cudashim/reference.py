# SPDX-License-Identifier: Apache-2.0
"""The upstream PyTorch implementation as a reference oracle.

Loads NVIDIA's own exporter models (``sandbox/upstream/tools/onnx``, pinned commit) from
the Hugging Face checkpoints and runs them with PyTorch on the CPU (exact FP32, the
same path the ONNX export traces) or on Apple's GPU through MPS. Forward hooks capture
each sub-network's inputs and outputs so the MLX port can be checked stage by stage.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from motionbricks_mlx import hub
from motionbricks_mlx.config import canonical_name

from .paths import CHECKPOINTS, UPSTREAM

__all__ = ["CHECKPOINTS", "UPSTREAM", "available", "build", "capture", "checkpoint_dir", "fixture_meta", "import_exporter",
           "kind", "require_checkpoints", "run", "torch_inputs", "upstream_commit", "write_meta"]


def available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (UPSTREAM / "tools" / "onnx" / "models").is_dir()


def kind(name: str) -> str:
    """Upstream's short model name (``"g1"`` / ``"soma"``); ValueError for an unknown model."""
    return "g1" if canonical_name(name) == "unitree_g1" else "soma"


def checkpoint_dir(name: str) -> Path:
    return hub.checkpoint_dir(name, CHECKPOINTS)


def require_checkpoints(name: str) -> None:
    """Fail early, with the fix, when a model's checkpoint files are missing."""
    missing = [e.path for e in hub.files_for(name) if not (CHECKPOINTS / e.path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {canonical_name(name)} checkpoint file(s) missing under {CHECKPOINTS} "
            f"(e.g. {missing[0]}): run `motionbricks-mlx download --dir {CHECKPOINTS}` or sandbox/setup.sh")


def upstream_commit() -> str | None:
    """The commit the upstream checkout is at (None without a checkout)."""
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fixture_meta(generator: str, device: str, **extra) -> dict:
    """What produced a set of golden fixtures: the oracle's device and versions. Fixtures are only
    bit-reproducible with the same versions, and only CPU fixtures are the FP32 reference."""
    import torch

    meta = {"generator": generator, "device": device, "upstream_commit": upstream_commit(),
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.machine()}
    for module in ("warp", "onnxruntime"):
        if module in sys.modules:
            meta[module] = sys.modules[module].__version__
    meta.update(extra)
    return meta


def write_meta(directory: Path, meta: dict) -> Path:
    path = Path(directory) / "meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def import_exporter():
    """Import upstream ``tools/onnx`` under its installed name ``motionbricks_onnx``."""
    if "motionbricks_onnx" in sys.modules:
        return sys.modules["motionbricks_onnx"]
    root = UPSTREAM / "tools" / "onnx"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} missing: run sandbox/setup.sh first")
    spec = importlib.util.spec_from_file_location(
        "motionbricks_onnx", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["motionbricks_onnx"] = module
    spec.loader.exec_module(module)
    return module


def build(name: str, device: str = "cpu"):
    """Build the upstream batched model (``BatchedG1`` / ``BatchedSOMA``) in eval mode."""
    import torch

    short = kind(name)
    import_exporter()
    require_checkpoints(name)
    from motionbricks_onnx.checkpoints import checkpoint_root, load_sources
    from motionbricks_onnx.checksums import CHECKPOINTS as SUMS

    sources = load_sources(checkpoint_root(checkpoint_dir(name), SUMS[short]), SUMS[short])
    if short == "g1":
        from motionbricks_onnx.models.g1 import build_g1 as builder
    else:
        from motionbricks_onnx.models.soma import build_soma as builder
    model = builder(sources).eval()
    # The exported graphs never take PyTorch's nested-tensor "fast path" (it zeroes padded
    # rows), so neither does the reference.
    torch.backends.mha.set_fastpath_enabled(False)
    return model.to(device)


def torch_inputs(name: str, request: dict[str, np.ndarray], device: str = "cpu"):
    import torch

    short = kind(name)
    import_exporter()
    if short == "g1":
        from motionbricks_onnx.models.g1 import INPUT_NAMES
    else:
        from motionbricks_onnx.models.soma import INPUT_NAMES
    return [torch.from_numpy(np.ascontiguousarray(request[k])).to(device) for k in INPUT_NAMES]


def _networks(model, name):
    """(root backbone, pose backbone, vqvae) modules of an upstream batched model."""
    if kind(name) == "g1":
        inferencer = model.core.inferencer
        return inferencer._root_model.backbone_net, inferencer._pose_model.backbone_net, inferencer._vqvae_pose_model
    inferencer = model.model
    return inferencer._mmm_root_model.mmm_net, inferencer._mmm_pose_model.mmm_net, inferencer._vqvae_pose_model


@contextmanager
def capture(model, name):
    """Record the tensors entering/leaving the three networks during one forward."""
    import torch

    root_net, pose_net, vqvae = _networks(model, name)
    record: dict[str, np.ndarray] = {}

    def to_np(x):
        return x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x

    def root_hook(module, args, kwargs, output):
        names = ("global_root", "has_global", "local_root", "has_local", "poses", "has_poses", "num_tokens")
        for key, value in zip(names, args):
            record[f"root_in.{key}"] = to_np(value)
        record["root_in.allowed"] = to_np(kwargs["allowed_pred_num_tokens"])
        record["root_out.num_token_logits"] = to_np(output["num_token_logits"])
        record["root_out.pred_num_tokens"] = to_np(output["pred_num_tokens"])
        record["root_out.pred_global_root_values"] = to_np(output["pred_global_root_values"])

    def pose_hook(module, args, kwargs, output):
        names = ("pose_tokens", "root_cond", "pose_cond", "has_pose_cond", "num_tokens")
        for key, value in zip(names, args):
            record[f"pose_in.{key}"] = to_np(value)
        record["pose_out.pose_logits"] = to_np(output["pose_logits"])

    original_decode = vqvae.forward_decoder

    def decode_spy(x, target_cond, has_target_cond=None, external_cond=None, use_overall_indices=True, token_mask=None):
        record["vq_in.tokens"] = to_np(x)
        record["vq_in.target_cond"] = to_np(target_cond)
        record["vq_in.has_target_cond"] = to_np(has_target_cond)
        record["vq_in.external_cond"] = to_np(external_cond)
        record["vq_in.token_mask"] = to_np(token_mask)
        out = original_decode(x, target_cond, has_target_cond, external_cond, use_overall_indices, token_mask)
        record["vq_out.recon_state"] = to_np(out["recon_state"])
        return out

    handles = [
        root_net.register_forward_hook(root_hook, with_kwargs=True),
        pose_net.register_forward_hook(pose_hook, with_kwargs=True),
    ]
    vqvae.forward_decoder = decode_spy
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()
        vqvae.forward_decoder = original_decode


def run(model, name: str, request: dict[str, np.ndarray], device: str = "cpu", intermediates: bool = False):
    """Run one request through the upstream model; return ONNX-named outputs (+ captures)."""
    import torch

    output_names = ("pred_qpos", "pred_num_tokens", "pred_foot_contacts") if kind(name) == "g1" else (
        "pred_global_motions", "pred_num_tokens")
    with torch.inference_mode():
        if intermediates:
            with capture(model, name) as record:
                outputs = model(*torch_inputs(name, request, device))
        else:
            record = {}
            outputs = model(*torch_inputs(name, request, device))
    result = {k: v.detach().to("cpu").numpy() for k, v in zip(output_names, outputs)}
    result.update(record)
    return result
