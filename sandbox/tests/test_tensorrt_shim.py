"""TensorRT-style code (including NVIDIA's own plan builder) running on ONNX Runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import cudashim  # noqa: E402
from cudashim.warp_cpu import UPSTREAM, upstream_on_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Locally exported by sandbox/scripts/export_onnx.py, else the released LFS file.
G1_ONNX = next((p for p in (ROOT / "sandbox/assets/onnx/mmm_v6_g1_fp32.onnx",
                            UPSTREAM / "motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx")
                if p.is_file() and p.stat().st_size > 10_000), ROOT / "missing.onnx")
GOLDEN = ROOT / "sandbox/golden/unitree_g1/trace.npz"


@pytest.fixture(autouse=True)
def trt():
    previous = sys.modules.get("tensorrt")
    module = cudashim.install_tensorrt(force=True)
    yield module
    if previous is None:
        sys.modules.pop("tensorrt", None)
    else:
        sys.modules["tensorrt"] = previous


def tiny_onnx(path: Path):
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [3, 2], [1, 2, 3, 4, 5, 6])
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "tiny", [x], [y], [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_named_tensor_flow(trt, tmp_path):
    tiny_onnx(tmp_path / "tiny.onnx")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    assert trt.OnnxParser(network, logger).parse_from_file(str(tmp_path / "tiny.onnx"))
    network.get_input(0).shape = (4, 3)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    assert [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] == ["x", "y"]
    assert tuple(engine.get_tensor_shape("y")) == (4, 2)
    context = engine.create_execution_context()
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = np.zeros((4, 2), np.float32)
    context.set_tensor_address("x", x.ctypes.data)
    context.set_tensor_address("y", y.ctypes.data)
    assert context.execute_async_v3(0)
    assert np.allclose(y, x @ np.arange(1, 7, dtype=np.float32).reshape(3, 2))


@pytest.mark.skipif(not (G1_ONNX.is_file() and G1_ONNX.stat().st_size > 10_000), reason="run sandbox/scripts/export_onnx.py")
def test_upstream_builder_runs_motionbricks_onnx():
    """NVIDIA's ``build_serialized_engine`` + ``inspect_engine_tensors`` with the shim, then a
    TensorRT-style launch on the released G1 graph, checked against the PyTorch golden."""
    if not GOLDEN.is_file():
        pytest.skip("run sandbox/scripts/make_golden.py")
    import tensorrt as trt

    with upstream_on_path():
        from motionbricks.inference.builder import TensorRTBuildOptions, build_serialized_engine, create_logger
        from motionbricks.inference.types import TensorMode, inspect_engine_tensors
    logger = create_logger("WARNING")
    plan = build_serialized_engine(G1_ONNX, TensorRTBuildOptions(batch_size=2, enable_tf32=False), logger)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    specs = inspect_engine_tensors(engine)
    golden = dict(np.load(GOLDEN))
    context = engine.create_execution_context()
    buffers = {}
    for spec in specs:
        if spec.mode == TensorMode.INPUT:
            buffers[spec.name] = np.ascontiguousarray(golden[f"in.{spec.name}"].astype(spec.dtype))
        else:
            buffers[spec.name] = np.zeros(spec.shape, spec.dtype)
        assert context.set_tensor_address(spec.name, buffers[spec.name].ctypes.data)
    assert context.execute_async_v3(0)
    assert np.array_equal(buffers["pred_num_tokens"], golden["pred_num_tokens"])
    for i, n in enumerate(golden["pred_num_tokens"]):
        assert np.abs(buffers["pred_qpos"][i, :4 * n] - golden["pred_qpos"][i, :4 * n]).max() < 5e-4
