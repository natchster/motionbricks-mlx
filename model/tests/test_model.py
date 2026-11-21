"""Behavioural tests of the full MLX model (need the checkpoints, not torch)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import motionbricks_mlx as mb
from motionbricks_mlx import examples
from mlx_helpers import CHECKPOINTS, HF_FOLDERS, load_model

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
        g, l, p = model._rig.preprocess(args[0], args[1])
        core_args = (g, args[2], l, args[3], p, args[4], args[5])
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
    assert agree >= 0.75, f"{dtype}: only {agree:.0%} of durations match FP32"
    key = "pred_qpos" if model_name == "unitree_g1" else "pred_global_motions"
    same = a["pred_num_tokens"] == b["pred_num_tokens"]
    assert np.isfinite(b[key]).all()
    # Where the duration matches, the motion stays close (tokens may still flip locally).
    assert np.median(np.abs(a[key][same] - b[key][same])) < 0.05


def test_converted_weights_round_trip(tmp_path, model_name):
    path = mb.convert(model_name, CHECKPOINTS / HF_FOLDERS[model_name], tmp_path / "w.safetensors")
    reloaded = mb.load(path)
    request = examples.request(model_name, batch=2, seed=4)
    a, b = load_model(model_name).run(request), reloaded.run(request)
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_multi_step_decoding_runs(model_name):
    model = load_model(model_name)
    model.num_inference_steps = 3
    model._compiled = None
    try:
        out = model.run(examples.request(model_name, batch=2, seed=9))
        assert all(np.isfinite(v).all() for v in out.values())
    finally:
        model.num_inference_steps = 1
        model._compiled = None


def test_quantized_8bit_tracks_fp32(model_name):
    fp32 = load_model(model_name)
    q8 = mb.load(model_name, CHECKPOINTS / HF_FOLDERS[model_name]).quantize(bits=8)
    request = examples.request(model_name, batch=16, seed=21)
    a, b = fp32.run(request), q8.run(request)
    assert (a["pred_num_tokens"] == b["pred_num_tokens"]).mean() >= 0.75
    assert all(np.isfinite(v).all() for v in b.values())
