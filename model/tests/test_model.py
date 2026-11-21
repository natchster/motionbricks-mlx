"""Behavioural tests of the full MLX model (need the checkpoints, not torch)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import motionbricks_mlx as mb
from motionbricks_mlx import examples
from mlx_helpers import CHECKPOINTS, HF_FOLDERS, fresh_model, load_model

pytestmark = pytest.mark.weights


def test_interface_matches_onnx_contract(model_name):
    model = load_model(model_name)
    out = model.run(examples.request(model_name, batch=2))
    if model_name == "unitree_g1":
        assert out["pred_qpos"].shape == (2, 64, 36) and out["pred_foot_contacts"].shape == (2, 64, 4)
    else:
        assert out["pred_global_motions"].shape == (2, 64, 366)
    assert out["pred_num_tokens"].dtype == np.int32 and out["pred_num_tokens"].shape == (2,)
    assert all(np.isfinite(v).all() for v in out.values())


def test_compiled_equals_eager(model_name):
    model = load_model(model_name)
    request = examples.request(model_name, batch=3, seed=5)
    eager, compiled = model.run(request, compiled=False), model.run(request, compiled=True)
    for key in eager:
        assert np.abs(eager[key].astype(np.float64) - compiled[key]).max() < 1e-5


def test_batch_rows_are_independent(model_name):
    model = load_model(model_name)
    request = examples.request(model_name, batch=4, seed=7)
    batched = model.run(request)
    for i in range(4):
        single = model.run({k: v[i:i + 1] for k, v in request.items()})
        assert batched["pred_num_tokens"][i] == single["pred_num_tokens"][0]
        # Different batch sizes pick different matmul tilings (accumulation order), so allow
        # float32 noise at the same level as the PyTorch parity tolerance.
        for key in batched:
            assert np.abs(batched[key][i] - single[key][0]).max() < 2e-3


def test_allowed_token_counts_are_respected(model_name):
    model = load_model(model_name)
    request = examples.request(model_name, batch=6, seed=11)
    allowed = np.zeros((6, 11), np.int32)
    for i in range(6):
        allowed[i, [i, 10 - i]] = 1
    request["allowed_pred_num_tokens"] = allowed
    counts = model.run(request)["pred_num_tokens"]
    for i, count in enumerate(counts):
        assert count - 6 in (i, 10 - i)


def test_forced_duration(model_name):
    model = load_model(model_name)
    args = model.prepare(examples.request(model_name, batch=2))
    if model_name == "unitree_g1":
        g, loc, p = model._rig.preprocess(args[0], args[1])
        core_args = (g, args[2], loc, args[3], p, args[4], args[5])
    else:
        core_args = tuple(args)
    _, tokens = model.core.predict(*core_args, num_tokens=mx.array([7, 13]))
    assert np.array(tokens).tolist() == [7, 13]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_tracks_fp32(model_name, dtype):
    fp32 = load_model(model_name)
    low = load_model(model_name, dtype)
    assert low.dtype == mb.model.DTYPES[dtype]
    request = examples.request(model_name, batch=8, seed=3)
    a, b = fp32.run(request), low.run(request)
    agree = (a["pred_num_tokens"] == b["pred_num_tokens"]).mean()
    assert agree >= 7 / 8, f"{dtype}: only {agree:.0%} of durations match FP32"  # at most one flip
    key = "pred_qpos" if model_name == "unitree_g1" else "pred_global_motions"
    same = a["pred_num_tokens"] == b["pred_num_tokens"]
    assert np.isfinite(b[key]).all()
    # Where the duration matches, the motion stays close (tokens may still flip locally).
    assert np.median(np.abs(a[key][same] - b[key][same])) < 0.05


def test_converted_weights_round_trip(tmp_path, model_name):
    original = load_model(model_name)  # skips without checkpoints
    path = mb.convert(model_name, CHECKPOINTS / HF_FOLDERS[model_name], tmp_path / "w.safetensors")
    reloaded = mb.load(path)
    request = examples.request(model_name, batch=2, seed=4)
    a, b = original.run(request), reloaded.run(request)
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_multi_step_decoding_reaches_the_compiled_graph(model_name):
    """``num_inference_steps`` is read while tracing: setting it must re-trace (compiled = eager)."""
    model = load_model(model_name)
    request = examples.request(model_name, batch=2, seed=9)
    one_step = model.run(request)
    model.num_inference_steps = 3
    try:
        compiled, eager = model.run(request), model.run(request, compiled=False)
    finally:
        model.num_inference_steps = 1
    assert all(np.isfinite(v).all() for v in compiled.values())
    assert any(not np.array_equal(one_step[key], compiled[key]) for key in compiled)
    for key in compiled:
        assert np.abs(eager[key].astype(np.float64) - compiled[key]).max() < 1e-5
    assert all(np.array_equal(one_step[key], value) for key, value in model.run(request).items())


def test_weight_updates_reach_the_compiled_graph(model_name):
    model = load_model(model_name)
    request = examples.request(model_name, batch=2, seed=4)
    before = model.run(request)
    bias = model.core.root.proj_local_pose.bias
    model.update({"core": {"root": {"proj_local_pose": {"bias": bias + 1.0}}}})
    try:
        changed = model.run(request)
    finally:
        model.update({"core": {"root": {"proj_local_pose": {"bias": bias}}}})
    assert any(not np.array_equal(before[key], changed[key]) for key in before)
    assert all(np.array_equal(before[key], value) for key, value in model.run(request).items())


def test_set_dtype_is_set_compute_dtype(model_name):
    """``nn.Module.set_dtype`` must not bypass the networks-only cast or the compiled graph."""
    model = fresh_model(model_name)
    request = examples.request(model_name, batch=2, seed=6)
    model.run(request)  # trace in FP32 first
    model.set_dtype(mx.bfloat16)
    assert model.dtype == mx.bfloat16
    want = load_model(model_name, "bfloat16").run(request)
    for key, value in model.run(request).items():
        assert np.array_equal(value, want[key])


def test_quantized_8bit_tracks_fp32(model_name):
    fp32 = load_model(model_name)
    q8 = fresh_model(model_name).quantize(bits=8)
    assert q8.quantization == {"bits": 8, "group_size": 64} and "quantization" not in q8.parameters()
    with pytest.raises(RuntimeError, match="already quantized"):
        q8.quantize(bits=4)
    request = examples.request(model_name, batch=16, seed=21)
    a, b = fp32.run(request), q8.run(request)
    same = a["pred_num_tokens"] == b["pred_num_tokens"]
    assert same.mean() >= 15 / 16  # at most one flip
    assert all(np.isfinite(v).all() for v in b.values())
    key = "pred_qpos" if model_name == "unitree_g1" else "pred_global_motions"
    assert np.median(np.abs(a[key][same] - b[key][same])) < 0.05


def test_quantized_4bit_runs(model_name):
    q4 = fresh_model(model_name).quantize(bits=4)
    assert q4.quantization == {"bits": 4, "group_size": 64}
    out = q4.run(examples.request(model_name, batch=4, seed=21))
    assert all(np.isfinite(v).all() for v in out.values())
    assert np.all((out["pred_num_tokens"] >= 6) & (out["pred_num_tokens"] <= 16))
