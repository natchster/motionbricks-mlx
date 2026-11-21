# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/{g1,soma}/model.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""Top-level models with the exact tensor interface of the released ONNX graphs
(``mmm_v6_g1_*.onnx`` / ``mmm_v6_soma_*.onnx``), so code written against ONNX Runtime or
TensorRT can switch to MLX by swapping the session object.
"""

from __future__ import annotations

import operator
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import kernels
from . import weights as W
from .config import CONTEXT_AND_TARGET, FRAMES_PER_TOKEN, NUM_TOKEN_CHOICES, SPECS, ModelSpec, canonical_name, get_spec
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


def resolve_dtype(dtype) -> mx.Dtype:
    """``"float32"`` / ``"float16"`` / ``"bfloat16"`` (or the MLX dtype itself) -> MLX dtype."""
    if isinstance(dtype, mx.Dtype):
        if dtype in DTYPES.values():
            return dtype
    elif isinstance(dtype, str) and dtype in DTYPES:
        return DTYPES[dtype]
    raise ValueError(f"Unsupported compute dtype {dtype!r}; choose one of {', '.join(DTYPES)} "
                     "(for INT8 weights, call .quantize(8) on a loaded model).")


def load_rig_arrays(name: str) -> dict[str, np.ndarray]:
    filename = {"unitree_g1": "g1_model_rig.npz", "soma": "soma_model_rig.npz"}[canonical_name(name)]
    with resources.files("motionbricks_mlx").joinpath("assets", filename).open("rb") as stream, \
            np.load(stream, allow_pickle=False) as data:
        return dict(data)


def _input_dims(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    """Trailing (non-batch) shape of every ONNX input."""
    masks = {name: (CONTEXT_AND_TARGET,) for name in MASK_INPUTS} | {"allowed_pred_num_tokens": (NUM_TOKEN_CHOICES,)}
    if spec.name == "unitree_g1":
        qpos = (FRAMES_PER_TOKEN, kernels.G1_QPOS)
        return masks | {"context_qpos": qpos, "target_qpos": qpos}
    pose_dim = (spec.num_joints - 1) * 3 + spec.num_joints * 6  # joint positions + 6D rotations
    return masks | {"global_root_values": (CONTEXT_AND_TARGET, 5), "local_root_values": (CONTEXT_AND_TARGET, 4),
                    "local_poses": (CONTEXT_AND_TARGET, pose_dim)}


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
        self._input_dims = MappingProxyType(_input_dims(spec))  # not a dict: nn.Module would make it a child
        self._quant_bits: int | None = None
        self._quant_group_size: int | None = None
        self._compiled = None
        self.num_inference_steps = 1

    # -- settings that the compiled graph depends on ----------------------------------------
    # `forward_compiled` traces `__call__` once and replays it, with the parameters and these
    # Python-level settings baked in. Every way of changing them therefore drops the trace.
    @property
    def num_inference_steps(self) -> int:
        """MaskGIT decoding steps (1, the release configuration). Setting it drops the compiled graph."""
        return self._num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, steps: int) -> None:
        steps = operator.index(steps)
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {steps}.")
        self._num_inference_steps = steps
        self._compiled = None

    @property
    def dtype(self):
        return self.core.compute_dtype

    @property
    def quantization(self) -> dict[str, int] | None:
        """``{"bits": ..., "group_size": ...}`` after :meth:`quantize`, otherwise ``None``."""
        if self._quant_bits is None:
            return None
        return {"bits": self._quant_bits, "group_size": self._quant_group_size}

    def set_compute_dtype(self, dtype) -> MotionBricks:
        """FP32 (default) or FP16/BF16 for the three networks. Geometry, normalization and the
        public I/O stay FP32, mirroring the upstream mixed-FP16 ONNX export."""
        dtype = resolve_dtype(dtype)
        self.core.root.set_dtype(dtype)
        self.core.pose.set_dtype(dtype)
        self.core.vqvae.set_dtype(dtype)
        self.core.compute_dtype = dtype
        self._materialize()
        return self

    def set_dtype(self, dtype, predicate=None) -> MotionBricks:
        """Same as :meth:`set_compute_dtype`. Overrides ``nn.Module.set_dtype``, which would cast
        every parameter but leave the activations, ``dtype`` and the compiled graph behind."""
        if predicate is not None:
            raise TypeError("MotionBricks.set_dtype takes no predicate: the precision applies to the three networks.")
        return self.set_compute_dtype(dtype)

    def update(self, parameters: dict, strict: bool = True) -> MotionBricks:
        """``nn.Module.update`` (also behind ``load_weights``), then evaluate and recompile."""
        super().update(parameters, strict=strict)
        self._materialize()
        return self

    def update_modules(self, modules: dict, strict: bool = True) -> MotionBricks:
        """``nn.Module.update_modules`` (also behind ``nn.quantize(model)``), then evaluate and recompile."""
        super().update_modules(modules, strict=strict)
        self._materialize()
        return self

    def _materialize(self) -> None:
        """Evaluate the (lazily cast / quantized / loaded) parameters now. Otherwise the first
        `mx.compile` trace captures those lazy ops and every compiled call recomputes them: about
        4 ms per call for the ~200M-parameter FP16/BF16 cast."""
        mx.eval(self.parameters())
        self._compiled = None

    def quantize(self, bits: int = 8, group_size: int = 64) -> MotionBricks:
        """Experimental: affine-quantize the transformer projections (q/k/v/out + FFN) of the
        root and pose networks. 8-bit cuts the transformer weights about 4x. On the M4 Pro it was
        measured on, batch 1 is bound by streaming those weights, so it is also the fastest option
        there, and at larger batches MLX's quantized GEMM is slower than FP32 (see model/benchmarks);
        other Macs may differ. Durations matched
        FP32 on all test requests at 8 bits; check agreement for your own inputs.

        A model is quantized once; load it again for other settings."""
        if self._quant_bits is not None:
            raise RuntimeError(f"The model is already quantized ({self.quantization}); load it again to re-quantize.")
        for transformer in (self.core.pose.transformer_model, self.core.root.shared_transformer_model,
                            self.core.root.root_token_transformer_model):
            nn.quantize(transformer, group_size=group_size, bits=bits)
        self._quant_bits, self._quant_group_size = bits, group_size
        self._materialize()
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
        """Convert a dict of NumPy/MLX arrays (ONNX names, INT32 masks) to positional MLX inputs.

        Every input must have its ONNX shape with the same batch size ``N >= 1``, e.g. ``[N, 8]``
        for the masks and ``[N, 11]`` for ``allowed_pred_num_tokens``; otherwise ``ValueError``."""
        out = []
        batch = None
        for name in self.input_names:
            try:
                value = inputs[name]
            except KeyError:
                raise KeyError(f"Missing input {name!r}; the {self.spec.name} model takes {list(self.input_names)}.") from None
            if not isinstance(value, mx.array):
                value = mx.array(value)
            shape = value.shape
            if batch is None:
                batch = shape[0] if shape else 0
                if batch < 1:
                    raise ValueError(f"Input {name!r} has shape {shape}; expected a batch of at least one request.")
            expected = (batch, *self._input_dims[name])
            if shape != expected:
                raise ValueError(f"Input {name!r} has shape {shape}; expected {expected} "
                                 f"(batch {batch} from {self.input_names[0]!r}).")
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


def from_checkpoints(name: str, checkpoint_dir: str | Path, dtype="float32") -> MotionBricks:
    """Build from the Hugging Face SafeTensors folder (``.../MotionBricks/soma`` or ``.../g1``)."""
    spec, dtype = get_spec(name), resolve_dtype(dtype)
    torch_state, mean, std = W.read_checkpoints(checkpoint_dir, spec)
    model = MotionBricks(spec, mean, std)
    W.assign(model.core, W.convert_state(model.core, torch_state))
    model.eval()
    return model.set_compute_dtype(dtype)


def from_converted(path: str | Path, dtype="float32") -> MotionBricks:
    """Build from a single converted file written by :func:`convert`."""
    dtype = resolve_dtype(dtype)
    name, params, mean, std = W.read_converted(path)
    model = MotionBricks(get_spec(name), mean, std)
    W.assign(model.core, {k: v.astype(mx.float32) for k, v in params.items()})
    model.eval()
    return model.set_compute_dtype(dtype)


def convert(name: str, checkpoint_dir: str | Path, output: str | Path, dtype="float32") -> Path:
    spec, dtype = get_spec(name), resolve_dtype(dtype)
    torch_state, mean, std = W.read_checkpoints(checkpoint_dir, spec)
    model = MotionBricks(spec, mean, std)
    params = W.convert_state(model.core, torch_state)
    return W.save_converted(output, spec, params, mean, std, dtype)


def load(name_or_path: str | Path, checkpoint_dir: str | Path | None = None, dtype="float32") -> MotionBricks:
    """``load("soma", "checkpoints/MotionBricks/soma")`` or ``load("weights/soma.safetensors")``."""
    if checkpoint_dir is not None:
        return from_checkpoints(str(name_or_path), checkpoint_dir, dtype)
    path = Path(name_or_path).expanduser()
    if not path.is_file():
        if str(name_or_path) in SPECS:
            raise ValueError(f"load({str(name_or_path)!r}) needs the checkpoint folder as well, e.g. "
                             f"load({str(name_or_path)!r}, 'checkpoints/MotionBricks/...'); or pass a file "
                             "written by convert().")
        raise FileNotFoundError(f"No converted weight file at {path} (write one with motionbricks_mlx.convert, "
                                "or pass a model name and its checkpoint folder).")
    return from_converted(path, dtype)
