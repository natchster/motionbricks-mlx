# SPDX-License-Identifier: Apache-2.0
"""``motionbricks-mlx`` command line: download checkpoints, convert them, run a request, quick benchmark."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from . import examples, hub
from .model import DTYPES, convert, load


def _load(args):
    if args.checkpoints:
        model = load(args.model, args.checkpoints, dtype=args.dtype)
    else:
        model = load(args.weights, dtype=args.dtype)
    if getattr(args, "quantize", None):
        model.quantize(bits=args.quantize)
    return model


def cmd_download(args):
    models = ("soma", "unitree_g1") if args.model == "all" else (args.model,)
    dest = hub.download(models, args.dir)
    for model in models:
        print(f"{model}: {hub.checkpoint_dir(model, dest)}")


def cmd_convert(args):
    path = convert(args.model, args.checkpoints, args.output, dtype=args.dtype)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_generate(args):
    model = _load(args)
    inputs = examples.request(model.spec.name, batch=args.batch, seed=args.seed)
    out = model.run(inputs)
    for name, value in out.items():
        print(f"{name}: shape={value.shape} dtype={value.dtype}")
    for i, count in enumerate(out["pred_num_tokens"]):
        print(f"request {i}: {int(count)} tokens -> {4 * int(count)} frames ({4 * int(count) / 30:.2f} s)")


def cmd_bench(args):
    model = _load(args)
    for batch in args.batch:
        inputs = model.prepare(examples.request(model.spec.name, batch=batch, seed=0))
        for _ in range(args.warmup):
            mx.eval(model.forward_compiled(*inputs))
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            mx.eval(model.forward_compiled(*inputs))
            times.append(time.perf_counter() - start)
        ms = np.array(times) * 1e3
        print(f"{model.spec.name} {args.dtype} N={batch:3d}: median {np.median(ms):7.2f} ms  p90 {np.percentile(ms, 90):7.2f} ms"
              f"  ({batch / np.median(ms) * 1e3:8.0f} req/s)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="motionbricks-mlx", description="MotionBricks on MLX")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p, need_source=True):
        p.add_argument("--model", choices=("soma", "unitree_g1"), default="soma")
        p.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
        if need_source:
            p.add_argument("--quantize", type=int, choices=(4, 8), help="experimental: quantize transformer weights")
            src = p.add_mutually_exclusive_group(required=True)
            src.add_argument("--checkpoints", help="HF folder, e.g. checkpoints/MotionBricks/soma")
            src.add_argument("--weights", help="converted .safetensors from `motionbricks-mlx convert`")

    p = sub.add_parser("download", help=f"download and verify the checkpoints ({hub.REPO} @ {hub.REVISION[:7]})")
    p.add_argument("--model", choices=("all", "soma", "unitree_g1"), default="all")
    p.add_argument("--dir", default="checkpoints/MotionBricks", help="destination (default: checkpoints/MotionBricks)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("convert", help="convert HF checkpoints into one MLX .safetensors file")
    model_args(p, need_source=False)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("generate", help="run a synthetic request and print the outputs")
    model_args(p)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("bench", help="time compiled inference")
    model_args(p)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
