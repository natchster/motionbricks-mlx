"""Per-stage time of the MLX graph (compiled per stage), to see where a custom runtime could win.

Each stage is compiled on its own and timed over a pipelined loop (async_eval, 1 sync at the
end), which approximates GPU time per call. Inputs are the golden stage inputs tiled to N
(``sandbox/golden``, from ``sandbox/scripts/make_golden.py``; only the files, not the sandbox's
Python packages, are needed).

    uv run --package motionbricks-mlx python model-rs/bench/stage_profile.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import motionbricks_mlx as mb
from motionbricks_mlx import hub

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"
GOLDEN = ROOT / "sandbox" / "golden"


def tile(a, n):
    reps = -(-n // a.shape[0])
    return np.concatenate([a] * reps)[:n]


def timed(fn, args, iters=60):
    f = mx.compile(fn)
    for _ in range(5):
        mx.eval(f(*args))
    start = time.perf_counter()
    outs = []
    for _ in range(iters):
        o = f(*args)
        mx.async_eval(o)
        outs.append(o)
    mx.eval(outs[-1])
    return (time.perf_counter() - start) / iters * 1e3


def main():
    for name in ("soma", "unitree_g1"):
        golden = GOLDEN / name / "random.npz"
        if not golden.is_file():
            raise SystemExit(f"{golden} missing: generate it with sandbox/setup.sh (sandbox/scripts/make_golden.py)")
        model = mb.load(name, hub.checkpoint_dir(name, CHECKPOINTS))
        model.eval()
        mx.eval(model.parameters())
        core = model.core
        g = dict(np.load(golden))
        for n in (1, 32):
            t = {k: mx.array(tile(v, n)) for k, v in g.items() if "." in k}
            mask = {k: v != 0 for k, v in t.items()}
            root_args = [t["root_in.global_root"], mask["root_in.has_global"], t["root_in.local_root"], mask["root_in.has_local"],
                         t["root_in.poses"], mask["root_in.has_poses"], mask["root_in.allowed"],
                         t["root_in.num_tokens"].reshape(-1)]
            pose_args = [t["pose_in.pose_tokens"].astype(mx.int32), t["pose_in.root_cond"], t["pose_in.pose_cond"],
                         mask["pose_in.has_pose_cond"], t["pose_in.num_tokens"].astype(mx.int32).reshape(-1)]
            vq_args = [t["vq_in.tokens"].astype(mx.int32), t["vq_in.target_cond"], mask["vq_in.has_target_cond"],
                       t["vq_in.external_cond"], mask["vq_in.token_mask"]]
            full_inputs = model.prepare({k[3:]: tile(v, n) for k, v in g.items() if k.startswith("in.")})
            # Bound methods, compiled and timed right away (no closures over the loop variables).
            res = {
                "root network": timed(core.root.__call__, root_args),
                "pose network": timed(core.pose.__call__, pose_args),
                "pose transformer only": timed(core.pose.transformer_model.__call__,
                                               [mx.random.normal((n, 16, 1024)), mx.arange(16)[None] >= mx.full((n, 1), 12)]),
                "vq decoder": timed(core.vqvae.decode, vq_args),
                "whole model": timed(model.__call__, full_inputs),
            }
            parts = res["root network"] + res["pose network"] + res["vq decoder"]
            res["rest (rig, root conversions, glue)"] = res["whole model"] - parts
            print(f"{name} N={n}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in res.items()))


if __name__ == "__main__":
    main()
