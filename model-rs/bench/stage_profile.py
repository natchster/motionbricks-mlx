"""Per-stage time of the MLX graph (compiled per stage), to see where a custom runtime could win.

Each stage is compiled on its own and timed over a pipelined loop (async_eval, 1 sync at the
end), which approximates GPU time per call. Inputs are the golden stage inputs tiled to N.

    uv run --all-packages python model-rs/bench/stage_profile.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import motionbricks_mlx as mb

ROOT = Path(__file__).resolve().parents[2]
CKPT = {"soma": ROOT / "checkpoints/MotionBricks/soma", "unitree_g1": ROOT / "checkpoints/MotionBricks/g1"}


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
        model = mb.load(name, CKPT[name])
        model.eval()
        mx.eval(model.parameters())
        core = model.core
        g = dict(np.load(ROOT / "sandbox/golden" / name / "random.npz"))
        for n in (1, 32):
            t = {k: mx.array(tile(v, n)) for k, v in g.items() if "." in k}
            m = lambda k: t[k] != 0  # noqa: E731
            root_args = [t["root_in.global_root"], m("root_in.has_global"), t["root_in.local_root"], m("root_in.has_local"),
                         t["root_in.poses"], m("root_in.has_poses"), m("root_in.allowed"),
                         t["root_in.num_tokens"].reshape(-1)]
            pose_args = [t["pose_in.pose_tokens"].astype(mx.int32), t["pose_in.root_cond"], t["pose_in.pose_cond"],
                         m("pose_in.has_pose_cond"), t["pose_in.num_tokens"].astype(mx.int32).reshape(-1)]
            vq_args = [t["vq_in.tokens"].astype(mx.int32), t["vq_in.target_cond"], m("vq_in.has_target_cond"),
                       t["vq_in.external_cond"], m("vq_in.token_mask")]
            full_inputs = model.prepare({k[3:]: tile(v, n) for k, v in g.items() if k.startswith("in.")})
            res = {
                "root network": timed(lambda *a: core.root(*a), root_args),
                "pose network": timed(lambda *a: core.pose(*a), pose_args),
                "pose transformer only": timed(lambda x, pad: core.pose.transformer_model(x, pad),
                                               [mx.random.normal((n, 16, 1024)), mx.arange(16)[None] >= mx.full((n, 1), 12)]),
                "vq decoder": timed(lambda *a: core.vqvae.decode(*a), vq_args),
                "whole model": timed(lambda *a: model(*a), full_inputs),
            }
            parts = res["root network"] + res["pose network"] + res["vq decoder"]
            res["rest (rig, root conversions, glue)"] = res["whole model"] - parts
            print(f"{name} N={n}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in res.items()))


if __name__ == "__main__":
    main()
