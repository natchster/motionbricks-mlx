# SPDX-License-Identifier: Apache-2.0
"""A ``tensorrt`` stand-in backed by ONNX Runtime, for TensorRT-10-style Python code on macOS.

TensorRT ships no macOS wheels, and NVIDIA's runtime libraries cannot be translated (they
are closed binaries). What *can* be emulated is the API contract. This module implements
the subset of the TensorRT 10 Python API that the MotionBricks runtime uses: Builder /
OnnxParser / BuilderConfig / Runtime / ICudaEngine / IExecutionContext with the named-tensor
``set_tensor_address`` + ``execute_async_v3`` flow.

* A "plan" is the ONNX graph with its batch dimension specialized, plus a JSON header.
* "Device pointers" are host addresses: Apple silicon has unified memory, so a Warp CPU
  array's ``ptr`` or a NumPy array's ``ctypes.data`` is the buffer the GPU would see.
* Execution runs on ONNX Runtime's CPU EP (bit-faithful FP32, the documented TensorRT
  parity reference) or CoreML EP (``CUDASHIM_TRT_PROVIDER=coreml``, uses the GPU/ANE).
* ``BuilderFlag.FP16`` / ``TF32`` are recorded in the plan but do not change the math: an FP16
  build runs in FP32 and logs a warning, so it cannot stand in for checking FP16 accuracy.

Install as the importable ``tensorrt`` module with :func:`cudashim.install_tensorrt`.
"""

from __future__ import annotations

import ctypes
import enum
import json
import os

import numpy as np

__version__ = "10.8.0.43+cudashim.ort"
_MAGIC = b"CUDASHIM-TRT-PLAN\x00"


class _Severity(enum.IntEnum):
    INTERNAL_ERROR = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4


class ILogger:
    Severity = _Severity

    def __init__(self, *args, **kwargs):
        pass

    def log(self, severity, msg):  # pragma: no cover - overridden by users
        pass


class Logger(ILogger):
    INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE = (_Severity(i) for i in range(5))

    def __init__(self, min_severity=_Severity.WARNING):
        super().__init__()
        self.min_severity = min_severity

    def log(self, severity, msg):
        if int(severity) <= int(self.min_severity):
            print(f"[cudashim.tensorrt] {severity.name}: {msg}")


class IProgressMonitor:
    def __init__(self, *args, **kwargs):
        pass


class DataType(enum.IntEnum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    INT64 = 10


_NP = {DataType.FLOAT: np.float32, DataType.HALF: np.float16, DataType.INT8: np.int8, DataType.INT32: np.int32,
       DataType.BOOL: np.bool_, DataType.INT64: np.int64}
_FROM_ONNX = {1: DataType.FLOAT, 10: DataType.HALF, 3: DataType.INT8, 6: DataType.INT32, 9: DataType.BOOL, 7: DataType.INT64}


def nptype(dtype: DataType):
    return _NP[DataType(dtype)]


class TensorIOMode(enum.IntEnum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2


class TensorLocation(enum.IntEnum):
    DEVICE = 0
    HOST = 1


class BuilderFlag(enum.IntEnum):
    FP16 = 0
    TF32 = 7


class MemoryPoolType(enum.IntEnum):
    WORKSPACE = 0


def init_libnvinfer_plugins(logger, namespace):
    return True


class ITensor:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: DataType):
        self.name, self.shape, self.dtype = name, tuple(shape), dtype


class INetworkDefinition:
    """Inputs are mutable (``tensor.shape = ...``); outputs re-resolve like TensorRT's shape
    propagation: once every input is static, one dry run on ONNX Runtime fixes them."""

    def __init__(self):
        self.inputs: list[ITensor] = []
        self.parsed_outputs: list[ITensor] = []
        self.onnx_model = None
        self._resolved_key = None
        self._resolved: tuple[bytes, list[ITensor]] | None = None

    num_inputs = property(lambda self: len(self.inputs))
    num_outputs = property(lambda self: len(self.parsed_outputs))

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        resolved = self.resolve()
        return (resolved[1] if resolved else self.parsed_outputs)[index]

    def resolve(self):
        """Specialize the graph to the current input shapes; None while any input is dynamic."""
        key = tuple(t.shape for t in self.inputs)
        if any(d < 0 for shape in key for d in shape):
            return None
        if key != self._resolved_key:
            by_name = {t.name: t for t in self.inputs}
            for value in self.onnx_model.graph.input:
                if value.name in by_name:
                    for dim, size in zip(value.type.tensor_type.shape.dim, by_name[value.name].shape):
                        dim.Clear()
                        dim.dim_value = int(size)
            onnx_bytes = self.onnx_model.SerializeToString()
            session = _session(onnx_bytes, hand_off=True)
            results = session.run(None, {t.name: _dummy(t) for t in self.inputs})
            outputs = [ITensor(o.name, r.shape, _from_onnx_np(r.dtype)) for o, r in zip(session.get_outputs(), results)]
            self._resolved_key, self._resolved = key, (onnx_bytes, outputs)
        return self._resolved


class OnnxParser:
    def __init__(self, network: INetworkDefinition, logger):
        self.network, self.logger, self._errors = network, logger, []

    num_errors = property(lambda self: len(self._errors))

    def get_error(self, index):
        return self._errors[index]

    def parse_from_file(self, path) -> bool:
        import onnx

        try:
            model = onnx.load(str(path))
        except Exception as exc:
            self._errors.append(f"cannot load {path}: {exc}")
            return False
        return self._populate(model)

    def parse(self, data: bytes) -> bool:
        import onnx

        try:
            model = onnx.load_from_string(bytes(data))
        except Exception as exc:
            self._errors.append(f"cannot parse ONNX bytes: {exc}")
            return False
        return self._populate(model)

    def _populate(self, model) -> bool:
        def tensors(values):
            out = []
            for value in values:
                tt = value.type.tensor_type
                dims = tuple(d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim)
                out.append(ITensor(value.name, dims, _FROM_ONNX[tt.elem_type]))
            return out

        initializers = {init.name for init in model.graph.initializer}
        self.network.inputs = tensors(v for v in model.graph.input if v.name not in initializers)
        self.network.parsed_outputs = tensors(model.graph.output)
        self.network.onnx_model = model
        return True


class IBuilderConfig:
    def __init__(self):
        self.flags = {BuilderFlag.TF32}
        self.builder_optimization_level = 3
        self.progress_monitor = None
        self.pool_limits = {}

    def set_flag(self, flag):
        self.flags.add(BuilderFlag(flag))

    def clear_flag(self, flag):
        self.flags.discard(BuilderFlag(flag))

    def get_flag(self, flag):
        return BuilderFlag(flag) in self.flags

    def set_memory_pool_limit(self, pool, size):
        self.pool_limits[int(pool)] = int(size)


def _providers():
    choice = os.environ.get("CUDASHIM_TRT_PROVIDER", "cpu").lower()
    if choice == "coreml":
        return [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# The session of the latest shape-resolving dry run, kept for the engine built from the same
# graph so a build + deserialize creates one ONNX Runtime session, not two: [(providers, bytes, session)].
_handoff: list = []


def _session(onnx_bytes: bytes, hand_off: bool = False):
    providers = _providers()
    if _handoff:
        held_providers, held_bytes, held = _handoff[0]
        if held_providers == providers and len(held_bytes) == len(onnx_bytes) and held_bytes == onnx_bytes:
            _handoff.clear()
            return held
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(onnx_bytes, sess_options=options, providers=providers)
    if hand_off:
        _handoff[:] = [(providers, onnx_bytes, session)]
    return session


def _warn(logger, message: str) -> None:
    log = getattr(logger, "log", None)
    if callable(log):
        log(_Severity.WARNING, message)


def _dummy(tensor: ITensor):
    if tensor.dtype in (DataType.INT32, DataType.INT64, DataType.BOOL, DataType.INT8):
        return np.ones(tensor.shape, nptype(tensor.dtype))
    value = np.zeros(tensor.shape, nptype(tensor.dtype))
    if value.ndim == 3 and value.shape[-1] >= 7:
        value[..., 6] = 1.0  # keep quaternion-carrying inputs finite
    return value


class Builder:
    def __init__(self, logger):
        self.logger = logger

    def create_network(self, flags=0):
        return INetworkDefinition()

    def create_builder_config(self):
        return IBuilderConfig()

    def build_serialized_network(self, network: INetworkDefinition, config: IBuilderConfig):
        resolved = network.resolve()
        if resolved is None:
            dynamic = [t.name for t in network.inputs if any(d < 0 for d in t.shape)]
            raise ValueError(f"inputs are still dynamic: {dynamic}")
        onnx_bytes, outputs = resolved
        if config.get_flag(BuilderFlag.FP16):
            _warn(self.logger, "BuilderFlag.FP16 is recorded but not emulated: ONNX Runtime runs this plan in FP32")
        header = {
            "inputs": [[t.name, list(t.shape), int(t.dtype)] for t in network.inputs],
            "outputs": [[t.name, list(t.shape), int(t.dtype)] for t in outputs],
            "fp16": config.get_flag(BuilderFlag.FP16),
            "tf32": config.get_flag(BuilderFlag.TF32),
        }
        raw = json.dumps(header).encode()
        return memoryview(_MAGIC + len(raw).to_bytes(8, "little") + raw + onnx_bytes)


def _from_onnx_np(dtype) -> DataType:
    for key, value in _NP.items():
        if np.dtype(value) == np.dtype(dtype):
            return key
    raise TypeError(dtype)


class ICudaEngine:
    def __init__(self, header: dict, onnx_bytes: bytes):
        self._io = {}
        for name, shape, dtype in header["inputs"]:
            self._io[name] = (TensorIOMode.INPUT, tuple(shape), DataType(dtype))
        for name, shape, dtype in header["outputs"]:
            self._io[name] = (TensorIOMode.OUTPUT, tuple(shape), DataType(dtype))
        self._names = list(self._io)
        self.header = header
        self.session = _session(onnx_bytes)

    num_io_tensors = property(lambda self: len(self._names))

    def get_tensor_name(self, index):
        return self._names[index]

    def get_tensor_mode(self, name):
        return self._io[name][0]

    def get_tensor_shape(self, name):
        return self._io[name][1]

    def get_tensor_dtype(self, name):
        return self._io[name][2]

    def get_tensor_location(self, name):
        return TensorLocation.DEVICE

    def create_execution_context(self):
        return IExecutionContext(self)


class IExecutionContext:
    def __init__(self, engine: ICudaEngine):
        self.engine = engine
        self.addresses: dict[str, int] = {}

    def set_tensor_address(self, name, address) -> bool:
        """False (like TensorRT) for an unknown name, or a null address for a non-empty tensor."""
        if name not in self.engine._io:
            return False
        if int(address) == 0 and int(np.prod(self.engine._io[name][1])) > 0:
            return False
        self.addresses[name] = int(address)
        return True

    def get_tensor_address(self, name):
        return self.addresses.get(name, 0)

    def _view(self, name) -> np.ndarray:
        _, shape, dtype = self.engine._io[name]
        np_dtype = np.dtype(nptype(dtype))
        nbytes = int(np.prod(shape)) * np_dtype.itemsize
        buffer = (ctypes.c_char * nbytes).from_address(self.addresses[name])
        return np.frombuffer(buffer, dtype=np_dtype).reshape(shape)

    def execute_async_v3(self, stream_handle=0) -> bool:
        """Synchronous on the host; ``stream_handle`` is accepted and ignored."""
        missing = [n for n in self.engine._names if not self.addresses.get(n) and int(np.prod(self.engine._io[n][1])) > 0]
        if missing:
            raise RuntimeError(f"tensor addresses not set (or null): {missing}")
        feeds = {}
        for name in self.engine._names:
            if self.engine._io[name][0] == TensorIOMode.INPUT:
                feeds[name] = self._view(name).copy()
        outputs = self.engine.session.run(None, feeds)
        for meta, value in zip(self.engine.session.get_outputs(), outputs):
            view = self._view(meta.name)
            np.copyto(view, value.astype(view.dtype, copy=False))
        return True

    def execute_v2(self, bindings) -> bool:
        for name, address in zip(self.engine._names, bindings):
            self.addresses[name] = int(address)  # null addresses are rejected by execute_async_v3
        return self.execute_async_v3(0)


class Runtime:
    def __init__(self, logger):
        self.logger = logger

    def deserialize_cuda_engine(self, plan) -> ICudaEngine | None:
        data = bytes(plan)
        if not data.startswith(_MAGIC):
            return None
        offset = len(_MAGIC)
        size = int.from_bytes(data[offset:offset + 8], "little")
        header = json.loads(data[offset + 8:offset + 8 + size])
        return ICudaEngine(header, data[offset + 8 + size:])
