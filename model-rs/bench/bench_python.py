"""Benchmark of the Python MLX port, with the same methodology and JSON schema as the Rust
`mbrs-bench` binary (see src/bin/bench.rs). Needs only the model package.

    uv run --package motionbricks-mlx python model-rs/bench/bench_python.py --model soma --json out.json
    uv run --package motionbricks-mlx python model-rs/bench/bench_python.py --op-cost   # Python cost per lazy op

Each call is split into prepare (input conversion), graph (building the lazy graph: compiled
replay or eager tracing), eval (wall time, and the thread's CPU time inside ``mx.eval``, which
separates CPU encoding from waiting on the GPU) and to_host (NumPy conversion).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import motionbricks_mlx as mb
from motionbricks_mlx import hub

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"


def load_request(name, batch):
    with np.load(HERE / "requests" / f"{name}_N{batch}.npz") as data:
        return dict(data)


def copy_out(outputs):
    return [np.array(o) for o in outputs]


def python_op_cost(n=20000):
    """Microseconds Python spends building one lazy MLX graph node (no evaluation)."""
    a = mx.ones((4, 4))
    mx.eval(a)
    t0 = time.perf_counter()
    x = a
    for _ in range(n):
        x = x + a
    t1 = time.perf_counter()
    return round(1e6 * (t1 - t0) / n, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="soma", choices=("soma", "unitree_g1"))
    p.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16", "int8"))
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stream", type=int, default=200)
    p.add_argument("--json")
    p.add_argument("--op-cost", action="store_true",
                   help="separate mode: only measure the Python cost of building one lazy MLX op, then exit")
    a = p.parse_args()

    if a.op_cost:
        us = python_op_cost()
        print(f"Python cost to build one lazy MLX op: {us} us")
        if a.json:
            Path(a.json).write_text(json.dumps({"runtime": "python", "python_us_per_lazy_op": us}, indent=2))
        return

    start = time.perf_counter()
    model = mb.load(a.model, hub.checkpoint_dir(a.model, CHECKPOINTS), dtype="float32" if a.dtype == "int8" else a.dtype)
    if a.dtype == "int8":
        model.quantize(bits=8)  # FP32 activations, INT8 transformer projections
    load_s = time.perf_counter() - start
    report = {}
    keys = ("prepare", "graph", "eval_wall", "eval_cpu", "to_host", "total", "core")
    for batch in a.batches:
        request = load_request(a.model, batch)
        for mode in ("compiled", "eager"):
            samples = {k: [] for k in keys}
            first_ms = 0.0
            for it in range(a.warmup + a.iters):
                t0 = time.perf_counter()
                inputs = model.prepare(request)
                t1 = time.perf_counter()
                outputs = model.forward_compiled(*inputs) if mode == "compiled" else model(*inputs)
                t2 = time.perf_counter()
                c0 = time.thread_time()
                mx.eval(outputs)
                c1 = time.thread_time()
                t3 = time.perf_counter()
                copy_out(outputs)
                t4 = time.perf_counter()
                if it == 0:
                    first_ms = (t4 - t0) * 1e3
                if it >= a.warmup:
                    values = (t1 - t0, t2 - t1, t3 - t2, c1 - c0, t4 - t3, t4 - t0, t3 - t1)
                    for k, v in zip(keys, values):
                        samples[k].append(v * 1e3)
            row = {f"{k}_ms": float(np.median(v)) for k, v in samples.items()}
            for k in ("total", "core"):
                row[f"{k}_p10_ms"] = float(np.percentile(samples[k], 10, method="nearest"))
                row[f"{k}_p90_ms"] = float(np.percentile(samples[k], 90, method="nearest"))
            row["first_call_ms"] = first_ms
            print(f"python {a.model:<10} {a.dtype:<8} {mode:<8} N={batch:<3} core {row['core_ms']:7.2f} ms  "
                  f"total {row['total_ms']:7.2f} ms  (graph {row['graph_ms']:5.2f}, eval {row['eval_wall_ms']:6.2f} / "
                  f"cpu {row['eval_cpu_ms']:5.2f})")
            report[f"{a.model}/{a.dtype}/N={batch}/{mode}"] = row

    if a.stream > 0:
        inputs = model.prepare(load_request(a.model, 1))
        for _ in range(a.warmup):
            mx.eval(model.forward_compiled(*inputs))
        t = time.perf_counter()
        for _ in range(a.stream):
            out = model.forward_compiled(*inputs)
            mx.eval(out)
            copy_out(out)
        sequential = a.stream / (time.perf_counter() - t)
        t = time.perf_counter()
        pending = None
        for _ in range(a.stream):
            out = model.forward_compiled(*inputs)
            mx.async_eval(out)
            if pending is not None:
                copy_out(pending)
            pending = out
        copy_out(pending)
        pipelined = a.stream / (time.perf_counter() - t)
        print(f"python {a.model:<10} {a.dtype:<8} stream of N=1 requests: sequential {sequential:6.1f} req/s, "
              f"pipelined {pipelined:6.1f} req/s")
        report[f"{a.model}/{a.dtype}/stream"] = {"sequential_req_s": sequential, "pipelined_req_s": pipelined}

    if a.json:
        Path(a.json).write_text(json.dumps({"runtime": "python", "load_s": load_s, "results": report}, indent=2))


if __name__ == "__main__":
    main()
