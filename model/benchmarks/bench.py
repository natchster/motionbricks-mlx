"""Latency / throughput of MotionBricks on Apple silicon across backends.

Backends (all run the same request batch, see motionbricks_mlx.examples):

* ``mlx-fp32`` / ``mlx-fp16`` / ``mlx-bf16``  MLX port, one compiled graph, Metal GPU
* ``mlx-int8``                              MLX, 8-bit transformer weights (experimental)
* ``mlx-cpu``                               MLX port on the CPU stream
* ``torch-cpu``                             NVIDIA's upstream PyTorch model, CPU, FP32
* ``torch-mps``                             same model via cudashim (``.cuda()`` -> MPS)
* ``ort-cpu``                               upstream FP32 ONNX graph on ONNX Runtime (TensorRT stand-in)
* ``ort-coreml``                            same graph on ONNX Runtime's CoreML EP (slow to build)

    uv run python model/benchmarks/bench.py --models soma unitree_g1 --batches 1 8 32

By default each backend is timed in one block, one backend after the other. ``--rounds N``
interleaves them instead: for every batch size, each backend runs ``--iters`` timed calls in
turn, N rounds with a rotating start, and the statistics pool all N rounds. Thermal drift then
spreads evenly over the backends, which matters when comparing close numbers (FP16 vs BF16).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sandbox"))

import mlx.core as mx  # noqa: E402

import motionbricks_mlx as mb  # noqa: E402
from motionbricks_mlx import examples  # noqa: E402

CKPT = {"unitree_g1": ROOT / "checkpoints/MotionBricks/g1", "soma": ROOT / "checkpoints/MotionBricks/soma"}
ONNX = {  # exported locally by sandbox/scripts/export_onnx.py (same graph as the LFS release)
    "unitree_g1": ROOT / "sandbox/assets/onnx/mmm_v6_g1_fp32.onnx",
    "soma": ROOT / "sandbox/assets/onnx/mmm_v6_soma_fp32.onnx",
}
# Upstream docs/onnx_inference.md: TensorRT 10.8 plan alone, RTX 4090, CUDA graph, ms per batch.
RTX4090_TRT = {
    ("soma", "fp32"): {1: 2.50, 8: 2.96, 16: 3.74, 32: 5.28},
    ("soma", "fp16"): {1: 1.28, 8: 1.54, 16: 1.87, 32: 3.93},
    ("unitree_g1", "fp32"): {1: 2.67, 8: 3.78, 16: 5.17, 32: 7.81},
    ("unitree_g1", "fp16"): {1: 1.63, 8: 1.91, 16: 2.36, 32: 3.38},
}


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    ms = np.array(times) * 1e3
    return {"median_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)), "p90_ms": float(np.percentile(ms, 90))}


def sample(fn, warmup, iters):
    """The loop of ``timeit``, returning the raw times in ms (pooled across ``--rounds``)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.array(times) * 1e3


def mlx_runner(name, dtype, device="gpu", bits=None):
    model = mb.load(name, CKPT[name], dtype=dtype)
    if bits:
        model.quantize(bits=bits)

    def make(batch):
        args = model.prepare(examples.request(name, batch=batch))
        stream = mx.gpu if device == "gpu" else mx.cpu

        def run():
            with mx.stream(stream):
                mx.eval(model.forward_compiled(*args))

        return run

    return make


def torch_runner(name, device):
    import torch

    from cudashim import reference, torch_cuda

    torch.set_num_threads(max(1, torch.get_num_threads()))
    if device == "mps":
        torch_cuda.enable()
        model = reference.build(name, "cpu").cuda()
        dev = "cuda"
    else:
        model = reference.build(name, "cpu")
        dev = "cpu"

    def make(batch):
        inputs = reference.torch_inputs(name, examples.request(name, batch=batch), dev)

        def run():
            with torch.inference_mode():
                model(*inputs)
            if device == "mps":
                torch.mps.synchronize()

        return run

    return make


def ort_runner(name, provider):
    import onnxruntime as ort

    path = ONNX[name]
    if not path.is_file() or path.stat().st_size < 10_000:
        raise FileNotFoundError(f"{path} (run sandbox/scripts/export_onnx.py)")
    providers = ["CPUExecutionProvider"]
    if provider == "coreml":
        providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}), *providers]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)

    def make(batch):
        feeds = examples.request(name, batch=batch)

        def run():
            session.run(None, feeds)

        return run

    return make


BACKENDS = {
    "mlx-fp32": lambda n: mlx_runner(n, "float32"),
    "mlx-fp16": lambda n: mlx_runner(n, "float16"),
    "mlx-bf16": lambda n: mlx_runner(n, "bfloat16"),
    "mlx-int8": lambda n: mlx_runner(n, "float32", bits=8),
    "mlx-cpu": lambda n: mlx_runner(n, "float32", "cpu"),
    "torch-cpu": lambda n: torch_runner(n, "cpu"),
    "torch-mps": lambda n: torch_runner(n, "mps"),
    "ort-cpu": lambda n: ort_runner(n, "cpu"),
    "ort-coreml": lambda n: ort_runner(n, "coreml"),
}
SLOW = {"torch-cpu", "mlx-cpu", "ort-cpu", "ort-coreml"}


def host():
    sysctl = lambda key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()  # noqa: E731
    gpu_cores = subprocess.run(
        "system_profiler SPDisplaysDataType | awk -F': ' '/Total Number of Cores/{print $2; exit}'",
        shell=True, capture_output=True, text=True).stdout.strip()
    return {"chip": sysctl("machdep.cpu.brand_string"), "memory_gb": int(sysctl("hw.memsize")) // 2**30,
            "gpu_cores": gpu_cores, "macos": platform.mac_ver()[0], "mlx": mx.__version__}


def setup(name, backend):
    """``(make, setup seconds)``, or None (printed) when the backend is unavailable."""
    try:
        start = time.perf_counter()
        make = BACKENDS[backend](name)
    except Exception as exc:
        print(f"{name:10s} {backend:10s} unavailable: {exc}")
        return None
    return make, time.perf_counter() - start


def interleaved(name, args):
    """``--rounds``: every backend in turn for each batch size, pooling the rounds."""
    makers = {}
    for backend in args.backends:
        made = setup(name, backend)
        if made is not None:
            makers[backend] = made
    results = []
    for batch in args.batches:
        runs, first_ms = {}, {}
        for backend, (make, _) in makers.items():
            runs[backend] = make(batch)
            first = time.perf_counter()
            runs[backend]()  # includes graph compilation / session warm-up
            first_ms[backend] = (time.perf_counter() - first) * 1e3
        order = list(runs)
        times = {backend: [] for backend in order}
        for r in range(args.rounds):
            for backend in order[r % len(order):] + order[:r % len(order)]:
                iters = max(5, args.iters // 4) if backend in SLOW else args.iters
                times[backend].append(sample(runs[backend], args.warmup if backend not in SLOW else 2, iters))
        for backend in order:
            ms = np.concatenate(times[backend])
            stats = {"median_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)),
                     "p90_ms": float(np.percentile(ms, 90))}
            stats.update(model=name, backend=backend, batch=batch, first_call_ms=first_ms[backend],
                         setup_s=makers[backend][1], requests_per_s=batch / stats["median_ms"] * 1e3, rounds=args.rounds)
            results.append(stats)
            print(f"{name:10s} {backend:10s} N={batch:3d}  median {stats['median_ms']:8.2f} ms  "
                  f"p90 {stats['p90_ms']:8.2f} ms  {stats['requests_per_s']:9.0f} req/s  (first call {first_ms[backend]:7.0f} ms)")
    if "torch-mps" in makers:
        from cudashim import torch_cuda

        torch_cuda.disable()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["soma", "unitree_g1"])
    parser.add_argument("--backends", nargs="+", default=["mlx-fp32", "mlx-fp16", "mlx-bf16", "torch-mps", "ort-cpu", "torch-cpu"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=1,
                        help="interleave the backends over this many rounds per batch size (default 1: one after the other)")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    info = host()
    print(f"# {info['chip']} ({info['gpu_cores']}-core GPU), {info['memory_gb']} GB, macOS {info['macos']}, MLX {info['mlx']}")
    results = []
    for name in args.models:
        if args.rounds > 1:
            results += interleaved(name, args)
            continue
        for backend in args.backends:
            try:
                start = time.perf_counter()
                make = BACKENDS[backend](name)
                setup_s = time.perf_counter() - start
            except Exception as exc:
                print(f"{name:10s} {backend:10s} unavailable: {exc}")
                continue
            for batch in args.batches:
                run = make(batch)
                first = time.perf_counter()
                run()  # includes graph compilation / session warm-up
                first_ms = (time.perf_counter() - first) * 1e3
                iters = max(5, args.iters // 4) if backend in SLOW else args.iters
                stats = timeit(run, args.warmup if backend not in SLOW else 2, iters)
                stats.update(model=name, backend=backend, batch=batch, first_call_ms=first_ms, setup_s=setup_s,
                             requests_per_s=batch / stats["median_ms"] * 1e3)
                results.append(stats)
                print(f"{name:10s} {backend:10s} N={batch:3d}  median {stats['median_ms']:8.2f} ms  "
                      f"p90 {stats['p90_ms']:8.2f} ms  {stats['requests_per_s']:9.0f} req/s  (first call {first_ms:7.0f} ms)")
            if backend == "torch-mps":
                from cudashim import torch_cuda

                torch_cuda.disable()
    output = args.output or ROOT / "model/benchmarks/results" / f"{date.today()}-{info['chip'].replace(' ', '-')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"host": info, "results": results, "reference_rtx4090_tensorrt_ms": {
        f"{m}/{p}": v for (m, p), v in RTX4090_TRT.items()}}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
