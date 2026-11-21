# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/{g1,soma}/model.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""Top-level models with the exact tensor interface of the released ONNX graphs
(``mmm_v6_g1_*.onnx`` / ``mmm_v6_soma_*.onnx``), so code written against ONNX Runtime or
TensorRT can switch to MLX by swapping the session object.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import weights as W
from .config import ModelSpec, get_spec
from .inference import MotionBricksCore
from .representation import MotionRepresentation
from .rig import G1Rig, SomaRetarget

G1_INPUTS = ("context_qpos", "target_qpos", "has_global_root_values", "has_local_root_values", "has_local_poses",
             "allowed_pred_num_tokens")
G1_OUTPUTS = ("pred_qpos", "pred_num_tokens", "pred_foot_contacts")
SOMA_INPUTS = ("global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
               "local_poses", "has_local_poses", "allowed_pred_num_tokens")
SOMA_OUTPUTS = ("pred_global_motions", "pred_num_tokens")
MASK_INPUTS = {"has_global_root_values", "has_local_root_values", "has_local_poses", "allowed_pred_num_tokens"}

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def load_rig_arrays(name: str) -> dict[str, np.ndarray]:
    filename = {"unitree_g1": "g1_model_rig.npz", "soma": "soma_model_rig.npz"}[name]
    with resources.files("motionbricks_mlx").joinpath("assets", filename).open("rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            return dict(data)


class MotionBricks(nn.Module):
    """A MotionBricks model (SOMA or Unitree G1) behind the ONNX tensor interface."""

    def __init__(self, spec: ModelSpec, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.spec = spec
        rep = MotionRepresentation(spec.num_joints, mean, std)
        self.core = MotionBricksCore(spec, rep)
        arrays = load_rig_arrays(spec.name)
        if spec.name == "unitree_g1":
            self.input_names, self.output_names = G1_INPUTS, G1_OUTPUTS
            self._rig = G1Rig(arrays, rep.global_.indices)
        else:
            self.input_names, self.output_names = SOMA_INPUTS, SOMA_OUTPUTS
            self._retarget = SomaRetarget(arrays, spec.num_joints)
            ric = (spec.num_joints - 1) * 3
            self._local_rot = (ric, ric + spec.num_joints * 6)
            g_rot = rep.global_.indices["global_rot_data"]
            self._global_rot = (int(g_rot[0]), int(g_rot[-1]) + 1)
        self.num_inference_steps = 1
        self._compiled = None

    # -- precision -----------------------------------------------------------------------
    @property
    def dtype(self):
        return self.core.compute_dtype

    def set_compute_dtype(self, dtype) -> "MotionBricks":
        """FP32 (default) or FP16/BF16 for the three networks. Geometry, normalization and the
        public I/O stay FP32, mirroring the upstream mixed-FP16 ONNX export."""
        dtype = DTYPES.get(dtype, dtype)
        self.core.root.set_dtype(dtype)
        self.core.pose.set_dtype(dtype)
        self.core.vqvae.set_dtype(dtype)
        self.core.compute_dtype = dtype
        self._compiled = None
        return self

    def quantize(self, bits: int = 8, group_size: int = 64) -> "MotionBricks":
        """Experimental: affine-quantize the transformer projections (q/k/v/out + FFN) of the
        root and pose networks. This is a *memory* option: 8-bit cuts the transformer weights
        about 4x, but it is not faster. Batch-1 latency is dispatch-bound, and at larger batch
        MLX's quantized GEMM is slower than FP32 (see model/benchmarks). Durations matched
        FP32 on all test requests at 8 bits; check agreement for your own inputs."""
        for transformer in (self.core.pose.transformer_model, self.core.root.shared_transformer_model,
                            self.core.root.root_token_transformer_model):
            nn.quantize(transformer, group_size=group_size, bits=bits)
        self.quantization = {"bits": bits, "group_size": group_size}
        self._compiled = None
        return self

    # -- forward ---------------------------------------------------------------------------
    def __call__(self, *inputs):
        """Positional inputs in ``input_names`` order (float32 / bool); outputs in ``output_names`` order."""
        steps = self.num_inference_steps
        if self.spec.name == "unitree_g1":
            context, target, has_g, has_l, has_p, allowed = inputs
            global_root, local_root, local_poses = self._rig.preprocess(context, target)
            pred, num_tokens = self.core.predict(global_root, has_g, local_root, has_l, local_poses, has_p, allowed,
                                                 num_inference_steps=steps)
            qpos, contacts = self._rig.postprocess(pred)
            return qpos, num_tokens, contacts
        global_root, has_g, local_root, has_l, local_poses, has_p, allowed = inputs
        a, b = self._local_rot
        local_poses = mx.concatenate(
            [local_poses[..., :a], self._retarget.capture_to_standard(local_poses[..., a:b]), local_poses[..., b:]], axis=-1
        )
        pred, num_tokens = self.core.predict(global_root, has_g, local_root, has_l, local_poses, has_p, allowed,
                                             num_inference_steps=steps)
        a, b = self._global_rot
        pred = mx.concatenate([pred[..., :a], self._retarget.standard_to_capture(pred[..., a:b]), pred[..., b:]], axis=-1)
        return pred, num_tokens

    def prepare(self, inputs: dict) -> list[mx.array]:
        """Convert a dict of NumPy/MLX arrays (ONNX names, INT32 masks) to positional MLX inputs."""
        out = []
        for name in self.input_names:
            value = mx.array(inputs[name]) if not isinstance(inputs[name], mx.array) else inputs[name]
            out.append(value != 0 if name in MASK_INPUTS else value.astype(mx.float32))
        return out

    def forward_compiled(self, *inputs):
        if self._compiled is None:
            self._compiled = mx.compile(self.__call__)
        return self._compiled(*inputs)

    def run(self, inputs: dict, compiled: bool = True) -> dict[str, np.ndarray]:
        """ONNX-Runtime-style call: dict of arrays in, dict of NumPy arrays out."""
        args = self.prepare(inputs)
        outputs = self.forward_compiled(*args) if compiled else self(*args)
        mx.eval(outputs)
        return {name: np.array(value) for name, value in zip(self.output_names, outputs)}


def _build(spec: ModelSpec, mean, std) -> MotionBricks:
    return MotionBricks(spec, mean, std)


def from_checkpoints(name: str, checkpoint_dir: str | Path, dtype="float32") -> MotionBricks:
    """Build from the Hugging Face SafeTensors folder (``.../MotionBricks/soma`` or ``.../g1``)."""
    spec = get_spec(name)
    torch_state, mean, std = W.read_checkpoints(checkpoint_dir, spec)
    model = _build(spec, mean, std)
    W.assign(model.core, W.convert_state(model.core, torch_state))
    model.eval()
    return model.set_compute_dtype(dtype)


def from_converted(path: str | Path, dtype="float32") -> MotionBricks:
    """Build from a single converted file written by :func:`convert`."""
    name, params, mean, std = W.read_converted(path)
    model = _build(get_spec(name), mean, std)
    W.assign(model.core, {k: v.astype(mx.float32) for k, v in params.items()})
    model.eval()
    return model.set_compute_dtype(dtype)


def convert(name: str, checkpoint_dir: str | Path, output: str | Path, dtype="float32") -> Path:
    spec = get_spec(name)
    torch_state, mean, std = W.read_checkpoints(checkpoint_dir, spec)
    model = _build(spec, mean, std)
    params = W.convert_state(model.core, torch_state)
    return W.save_converted(output, spec, params, mean, std, DTYPES.get(dtype, dtype))


def load(name_or_path: str | Path, checkpoint_dir: str | Path | None = None, dtype="float32") -> MotionBricks:
    """``load("soma", "checkpoints/MotionBricks/soma")`` or ``load("weights/soma.safetensors")``."""
    if checkpoint_dir is not None:
        return from_checkpoints(str(name_or_path), checkpoint_dir, dtype)
    return from_converted(name_or_path, dtype)
