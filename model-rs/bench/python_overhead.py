"""Where does one MotionBricks generation spend its time in the Python MLX port?

Splits a call into: Python input prep, building the lazy graph (compiled replay vs eager
tracing), mx.eval (MLX C++ scheduling + Metal encoding + GPU wait), and NumPy conversion.
Thread CPU time inside mx.eval separates CPU encoding work from blocking on the GPU.

    uv run --all-packages python model-rs/bench/python_overhead.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import motionbricks_mlx as mb
from motionbricks_mlx import examples

ROOT = Path(__file__).resolve().parents[2]
CKPT = {"soma": ROOT / "checkpoints/MotionBricks/soma", "unitree_g1": ROOT / "checkpoints/MotionBricks/g1"}


def median_ms(samples):
    return 1e3 * statistics.median(samples)


def profile(model, inputs, compiled: bool, iters: int):
    parts = {k: [] for k in ("prepare", "graph", "eval_wall", "eval_cpu", "to_numpy", "total")}
    for i in range(iters + 5):
        t0 = time.perf_counter()
        args = model.prepare(inputs)
        t1 = time.perf_counter()
        outs = model.forward_compiled(*args) if compiled else model(*args)
        t2 = time.perf_counter()
        c0 = time.thread_time()
        mx.eval(outs)
        c1 = time.thread_time()
        t3 = time.perf_counter()
        _ = [np.array(o) for o in outs]
        t4 = time.perf_counter()
        if i >= 5:
            for k, v in zip(parts, (t1 - t0, t2 - t1, t3 - t2, c1 - c0, t4 - t3, t4 - t0)):
                parts[k].append(v)
    return {k: round(median_ms(v), 3) for k, v in parts.items()}


def python_op_cost(n=20000):
    """Per-op cost of building lazy graph nodes from Python (no evaluation)."""
    a = mx.ones((4, 4))
    mx.eval(a)
    t0 = time.perf_counter()
    x = a
    for _ in range(n):
        x = x + a
    t1 = time.perf_counter()
    return round(1e6 * (t1 - t0) / n, 3)  # microseconds per op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["soma", "unitree_g1"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--json")
    args = parser.parse_args()
    report = {"python_us_per_lazy_op": python_op_cost(), "models": {}}
    print(f"Python cost to build one lazy MLX op: {report['python_us_per_lazy_op']} us")
    for name in args.models:
        model = mb.load(name, CKPT[name])
        for batch in args.batches:
            inputs = examples.request(name, batch=batch, seed=1)
            for compiled in (True, False):
                r = profile(model, inputs, compiled, args.iters)
                key = f"{name}/N={batch}/{'compiled' if compiled else 'eager'}"
                report["models"][key] = r
                print(key, r)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
