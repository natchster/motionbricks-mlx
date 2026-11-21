"""Benchmark of the Python MLX port, with the same methodology and JSON schema as the Rust
`mbrs-bench` binary (see src/bin/bench.rs).

    uv run --all-packages python model-rs/bench/bench_python.py --model soma --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import motionbricks_mlx as mb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CKPT = {"soma": ROOT / "checkpoints/MotionBricks/soma", "unitree_g1": ROOT / "checkpoints/MotionBricks/g1"}


def load_request(name, batch):
    with np.load(HERE / "requests" / f"{name}_N{batch}.npz") as data:
        return dict(data)


def copy_out(outputs):
    return [np.array(o) for o in outputs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="soma")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stream", type=int, default=200)
    p.add_argument("--json")
    p.add_argument("--eval-params", action="store_true",
                   help="evaluate the (dtype-cast) parameters before the first compiled call; works around the\n                         lazy-cast issue of set_compute_dtype without changing the package")
    p.add_argument("--contiguous-weights", action="store_true",
                   help="experiment: store Conv1d kernels row-contiguously (package code unchanged)")
    a = p.parse_args()

    start = time.perf_counter()
    model = mb.load(a.model, CKPT[a.model], dtype="float32" if a.dtype == "int8" else a.dtype)
    if a.dtype == "int8":
        model.quantize(bits=8)  # FP32 activations, INT8 transformer projections
    if a.contiguous_weights:
        from mlx.utils import tree_flatten, tree_unflatten
        convs = [(k, mx.contiguous(v)) for k, v in tree_flatten(model.core.parameters())
                 if v.ndim == 3 and k.endswith(".weight")]
        mx.eval([v for _, v in convs])
        model.core.update(tree_unflatten(convs))
    if a.eval_params:
        mx.eval(model.parameters())
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
