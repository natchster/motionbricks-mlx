"""The Rust ONNX exporter (tools/onnx) against the original torch-traced exports.

Builds the `motionbricks-onnx` binary, exports both models from the verified checkpoints,
then checks the results three ways: onnx.checker, upstream's own interface validation, and
ONNX Runtime runs of the original and the Rust graph on identical inputs (the golden
fixtures plus a fresh random batch), comparing outputs elementwise.
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnxruntime as ort  # noqa: E402
from cudashim.paths import ASSETS, CHECKPOINTS, GOLDEN, ROOT  # noqa: E402

CRATE = ROOT / "tools/onnx"
MODELS = ("soma", "unitree_g1")
# FP32 originals to compare against: the locally exported files from sandbox/setup.sh.
ORIGINAL = {name: ASSETS / f"onnx/mmm_v6_{kind}_fp32.onnx" for name, kind in (("soma", "soma"), ("unitree_g1", "g1"))}


def have_checkpoints(name: str) -> bool:
    return (CHECKPOINTS / name).is_dir() and any((CHECKPOINTS / name).rglob("*.safetensors"))


@pytest.fixture(scope="module")
def rust_exporter():
    """The release binary, built once per test run (skipped when cargo is unavailable)."""
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=CRATE, check=True,
                       capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"cannot build tools/onnx: {error}")
    return CRATE / "target/release/motionbricks-onnx"


@pytest.fixture(scope="module", params=MODELS)
def rust_graph(request, rust_exporter, tmp_path_factory):
    """The Rust export of one model (FP32), validated with onnx.checker."""
    name = request.param
    if not have_checkpoints("g1" if name == "unitree_g1" else "soma"):
        pytest.skip(f"needs checkpoints/MotionBricks/{name} (run sandbox/setup.sh)")
    out = tmp_path_factory.mktemp("onnx_rs") / f"{name}.onnx"
    start = time.perf_counter()
    subprocess.run([str(rust_exporter), "--model", name, "--checkpoints", str(CHECKPOINTS / ("g1" if name == "unitree_g1" else "soma")),
                    "--output", str(out)], check=True, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - start
    onnx.checker.check_model(str(out), full_check=True)
    print(f"\n{name}: rust export in {elapsed:.1f}s, {out.stat().st_size / 1e6:.0f} MB")
    return name, out


def upstream_validate(path, name):
    """Upstream's own interface validation (export.validate_output), if the checkout is here."""
    from cudashim import reference
    if not reference.available():
        return  # covered separately by test_interface_table below
    reference.import_exporter()
    from motionbricks_onnx.export import validate_output
    validate_output(str(path), reference.kind(name))


def test_interface_table(rust_graph):
    """The exported model passes upstream's validate_output (names, dtypes, shapes, batch dim)."""
    name, out = rust_graph
    upstream_validate(out, name)


def load_golden_inputs(name: str):
    """Golden fixture inputs with ONNX names (`in.*` keys), batch as recorded."""
    case = GOLDEN / name / "random.npz"
    if not case.is_file():
        pytest.skip("needs sandbox/golden (run sandbox/setup.sh)")
    data = np.load(case)
    inputs = {key[3:]: data[key] for key in data.files if key.startswith("in.")}
    return inputs, data


def random_inputs(name: str, batch: int, seed: int):
    """A fresh random request with the ONNX input contract (mirrors make_golden's request draws)."""
    rng = np.random.default_rng(seed)
    if name == "unitree_g1":
        inputs = {
            "context_qpos": rng.normal(0, 0.5, (batch, 4, 36)).astype(np.float32),
            "target_qpos": rng.normal(0, 0.5, (batch, 4, 36)).astype(np.float32),
            "local_poses": None,
        }
    else:
        inputs = {
            "global_root_values": rng.normal(0, 0.5, (batch, 8, 5)).astype(np.float32),
            "local_root_values": rng.normal(0, 0.5, (batch, 8, 4)).astype(np.float32),
            "local_poses": rng.normal(0, 0.5, (batch, 8, 267)).astype(np.float32),
        }
    masks = {
        "has_global_root_values": (rng.random((batch, 8)) > 0.3),
        "has_local_root_values": (rng.random((batch, 8)) > 0.3),
        "has_local_poses": (rng.random((batch, 8)) > 0.3),
    }
    allowed = np.zeros((batch, 11), dtype=np.int32)
    allowed[:, 4:] = 1  # allow the longer durations, like the golden "random" case
    inputs.update({k: v.astype(np.int32) for k, v in masks.items()})
    inputs["allowed_pred_num_tokens"] = allowed
    if name == "unitree_g1":
        del inputs["local_poses"]
    return inputs


def run(path, inputs):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values = session.run(None, inputs)
    return {out.name: value for out, value in zip(session.get_outputs(), values, strict=True)}


def compare(name: str, reference_out: dict, rust_out: dict, atol: float):
    assert reference_out.keys() == rust_out.keys()
    worst = {}
    for key, ref in reference_out.items():
        got = rust_out[key]
        assert ref.shape == got.shape and ref.dtype == got.dtype, key
        diff = np.abs(ref.astype(np.float64) - got.astype(np.float64)).max() if ref.size else 0.0
        worst[key] = float(diff)
        assert diff <= atol, f"{name} {key}: max abs diff {diff:g} > {atol:g}"
    return worst


@pytest.mark.parametrize("source", ("golden", "random"))
def test_matches_original_onnx(rust_graph, source):
    """The Rust graph reproduces the original exported graph's outputs under ONNX Runtime."""
    name, out = rust_graph
    original = ORIGINAL[name]
    if not original.is_file():
        pytest.skip("needs sandbox/assets/onnx (run sandbox/setup.sh)")
    if source == "golden":
        inputs, _ = load_golden_inputs(name)
    else:
        inputs = random_inputs(name, batch=3, seed=7)
        inputs = {k: v for k, v in inputs.items() if k in {i.name for i in onnx.load(str(original), load_external_data=False).graph.input}}
    ref = run(original, inputs)
    got = run(out, inputs)
    worst = compare(name, ref, got, atol=2e-3)
    print(f"\n{name} ({source}): {json.dumps(worst)}")


def test_matches_golden_predictions(rust_graph):
    """Sanity: the Rust graph stays within the golden fixtures' tolerance on golden inputs."""
    name, out = rust_graph
    inputs, golden = load_golden_inputs(name)
    got = run(out, inputs)
    expected = {key: golden[key] for key in golden.files if key.startswith("pred_")}
    compare(name, expected, got, atol=3e-3)


def test_fp16_export(rust_exporter, tmp_path):
    """The mixed-FP16 export is a valid ONNX model with the same interface (ORT runs it on GPU
    runtimes only; CPU EP lacks FP16 Conv kernels, so this checks structure, not numerics)."""
    name = "soma"
    if not have_checkpoints("soma"):
        pytest.skip("needs checkpoints/MotionBricks/soma (run sandbox/setup.sh)")
    out = tmp_path / "soma_fp16.onnx"
    subprocess.run([str(rust_exporter), "--model", name, "--checkpoints", str(CHECKPOINTS / "soma"),
                    "--precision", "fp16", "--output", str(out)], check=True, capture_output=True, text=True, timeout=600)
    onnx.checker.check_model(str(out), full_check=True)
    upstream_validate(out, name)
    graph = onnx.load(str(out), load_external_data=False).graph
    halves = sum(t.raw_data.__len__() for t in graph.initializer if t.data_type == onnx.TensorProto.FLOAT16)
    assert halves > 100e6, "the transformer weights should be FP16"
