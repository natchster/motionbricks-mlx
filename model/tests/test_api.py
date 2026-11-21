"""Weight-free tests of the public API boundary: input validation, dtypes, loading, settings."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import motionbricks_mlx as mb
from motionbricks_mlx import examples, weights
from motionbricks_mlx.config import G1, SOMA, canonical_name
from motionbricks_mlx.model import MotionBricks, resolve_dtype


def dual_dim(spec) -> int:
    j = spec.num_joints
    return 9 + (j - 1) * 3 + j * 6 + j * 3 + 4


@pytest.fixture(scope="module", params=[SOMA, G1], ids=["soma", "unitree_g1"])
def model(request):
    """Randomly initialized (lazy, never evaluated here): enough for everything but inference."""
    spec = request.param
    return MotionBricks(spec, np.zeros(dual_dim(spec), np.float32), np.ones(dual_dim(spec), np.float32))


def test_prepare_accepts_the_onnx_interface(model):
    args = model.prepare(examples.request(model.spec.name, batch=3))
    assert len(args) == len(model.input_names)
    for name, value in zip(model.input_names, args):
        assert value.shape[0] == 3
        assert value.dtype == (mx.bool_ if name.startswith(("has_", "allowed_")) else mx.float32)


@pytest.mark.parametrize("change", ["first", "mask_batch", "allowed_width", "empty"])
def test_prepare_rejects_wrong_shapes(model, change):
    request = examples.request(model.spec.name, batch=2)
    first = model.input_names[0]
    if change == "first":
        request[first] = request[first][:, :2]
    elif change == "mask_batch":
        request["has_local_poses"] = request["has_local_poses"][:1]
    elif change == "allowed_width":
        request["allowed_pred_num_tokens"] = np.ones((2, 12), np.int32)
    else:
        request = {k: v[:0] for k, v in request.items()}
    with pytest.raises(ValueError, match="Input '"):
        model.prepare(request)


def test_prepare_names_a_missing_input(model):
    request = examples.request(model.spec.name)
    del request["has_local_poses"]
    with pytest.raises(KeyError, match="has_local_poses"):
        model.prepare(request)


def test_model_names():
    assert canonical_name("g1") == canonical_name("unitree_g1") == "unitree_g1"
    assert examples.request("g1").keys() == examples.request("unitree_g1").keys()
    with pytest.raises(ValueError, match="Unknown model"):
        examples.request("G1")


def test_dtypes():
    assert resolve_dtype("bfloat16") == mx.bfloat16 and resolve_dtype(mx.float16) == mx.float16
    for bad in ("fp16", "int8", mx.int8, None):
        with pytest.raises(ValueError, match="Unsupported compute dtype"):
            resolve_dtype(bad)


def test_bad_dtype_fails_before_any_work(model, tmp_path):
    with pytest.raises(ValueError, match="Unsupported compute dtype"):
        model.set_compute_dtype("fp16")
    assert model.dtype == mx.float32
    with pytest.raises(ValueError, match="Unsupported compute dtype"):
        mb.from_checkpoints("soma", tmp_path, dtype="int8")  # before looking for the checkpoints


def test_load_explains_what_it_needs(tmp_path):
    with pytest.raises(ValueError, match="needs the checkpoint folder"):
        mb.load("soma")
    with pytest.raises(FileNotFoundError, match="No converted weight file"):
        mb.load(tmp_path / "missing.safetensors")


def test_settings_drop_the_compiled_graph(model):
    model._compiled = object()  # stands in for a traced graph
    model.num_inference_steps = 1
    assert model._compiled is None and model.num_inference_steps == 1
    for bad in (0, -1):
        with pytest.raises(ValueError):
            model.num_inference_steps = bad
    with pytest.raises(TypeError):
        model.num_inference_steps = 2.5
    with pytest.raises(TypeError, match="no predicate"):
        model.set_dtype(mx.float16, predicate=lambda _: True)
    assert model.quantization is None
    assert "quantization" not in model.parameters()


def test_weights_load_strictly(model):
    params = dict(tree_flatten(model.core.parameters()))
    missing = dict(params)
    missing.pop(next(iter(missing)))
    misshaped = dict(params)
    key = next(k for k, v in params.items() if v.ndim == 2)
    misshaped[key] = mx.zeros((3, 3))
    extra = dict(params) | {"root.not_a_parameter": mx.zeros((1,))}
    for bad, message in ((missing, "Missing 1 parameters"), (misshaped, "Expected shape"), (extra, "not in model")):
        with pytest.raises(ValueError, match=message):
            weights.assign(model.core, bad)
