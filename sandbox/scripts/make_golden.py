"""Generate golden fixtures with NVIDIA's upstream PyTorch model (the reference oracle).

For each model and case it stores the ONNX-named inputs, the outputs, and the tensors
entering/leaving the root network, pose network and VQ-VAE decoder. The MLX test-suite
compares against these without needing torch at test time. ``meta.json`` next to them records
the device and versions that produced them.

Only CPU fixtures are the FP32 reference the tests expect, so ``--device mps`` writes to
``sandbox/golden-mps/`` unless ``--out`` says otherwise.

    uv run --all-packages python sandbox/scripts/make_golden.py [--device cpu|mps] [--out DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from cudashim import reference
from cudashim.paths import GOLDEN
from motionbricks_mlx import examples
from motionbricks_mlx.config import canonical_name

MODELS = ("unitree_g1", "soma")
CASES = ("trace", "random", "one_target", "restricted", "chained")


def trace_request(name: str) -> dict[str, np.ndarray]:
    """Upstream ``tools/onnx/export.py::example_inputs`` (batch of two, eye(11)[[0, 10]])."""
    allowed = np.eye(11, dtype=np.int32)[[0, 10]]
    masks = np.ones((2, 8), np.int32)
    masks[:, 4:6] = 0
    if canonical_name(name) == "unitree_g1":
        context = np.zeros((2, 4, 36), np.float32)
        context[..., 2] = 0.75
        context[..., 6] = 1.0
        target = context.copy()
        target[..., 0] = 1.0
        return {"context_qpos": context, "target_qpos": target, "has_global_root_values": masks,
                "has_local_root_values": masks.copy(), "has_local_poses": masks.copy(), "allowed_pred_num_tokens": allowed}
    global_root = np.zeros((2, 8, 5), np.float32)
    global_root[..., 1] = 1.0
    global_root[..., 3] = 1.0
    local_root = np.zeros((2, 8, 4), np.float32)
    local_root[..., 3] = 1.0
    local_poses = np.zeros((2, 8, 267), np.float32)
    local_poses[..., 87:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), 30)
    return {"global_root_values": global_root, "has_global_root_values": masks, "local_root_values": local_root,
            "has_local_root_values": masks.copy(), "local_poses": local_poses, "has_local_poses": masks.copy(),
            "allowed_pred_num_tokens": allowed}


def one_target(request):
    out = {k: v.copy() for k, v in request.items()}
    for key in ("has_global_root_values", "has_local_poses"):
        out[key][:, 5:] = 0
    out["has_local_root_values"][:, 3:] = 0
    out["has_local_root_values"][:, :3] = 1
    return out


def restricted_durations(request):
    out = {k: v.copy() for k, v in request.items()}
    out["allowed_pred_num_tokens"][:] = 0
    out["allowed_pred_num_tokens"][:, 4:7] = 1  # 10..12 tokens
    return out


def _local_root_windows(xyz_heading_height):
    """Local root features for two 4-frame windows, like the upstream preprocessors."""
    x, height, z, heading = xyz_heading_height
    windows = []
    for start in (0, 4):
        sl = slice(start, start + 3)
        nx = slice(start + 1, start + 4)
        yaw = (np.mod(heading[:, nx] - heading[:, sl] + np.pi, 2 * np.pi) - np.pi) * 30
        first = np.stack([yaw, (x[:, nx] - x[:, sl]) * 30, (z[:, nx] - z[:, sl]) * 30, height[:, sl]], -1)
        last = np.concatenate([np.zeros_like(first[:, :1, :3]), height[:, start + 3:start + 4, None]], -1)
        windows.append(np.concatenate([first, last], 1))
    return np.concatenate(windows, 1).astype(np.float32)


def chained_request(name, previous_inputs, previous_outputs):
    """An on-manifold request: context and target sampled from a generated motion."""
    g1 = canonical_name(name) == "unitree_g1"
    counts = previous_outputs["pred_num_tokens"].astype(int)
    key = "pred_qpos" if g1 else "pred_global_motions"
    frames = []
    for i, n in enumerate(counts):
        last = 4 * n
        index = np.r_[8:12, last - 4:last]
        frames.append(previous_outputs[key][i, index])
    frames = np.stack(frames).astype(np.float32)
    batch = frames.shape[0]
    ones = np.ones((batch, 8), np.int32)
    has_local = ones.copy()
    has_local[:, [3, 7]] = 0
    allowed = np.ones((batch, 11), np.int32)
    if g1:
        return {"context_qpos": frames[:, :4], "target_qpos": frames[:, 4:], "has_global_root_values": ones,
                "has_local_root_values": has_local, "has_local_poses": ones.copy(), "allowed_pred_num_tokens": allowed}
    heading = np.arctan2(frames[..., 4], frames[..., 3])
    local_root = _local_root_windows((frames[..., 0], frames[..., 1], frames[..., 2], heading))
    return {"global_root_values": frames[..., :5], "has_global_root_values": ones, "local_root_values": local_root,
            "has_local_root_values": has_local, "local_poses": frames[..., 5:272], "has_local_poses": ones.copy(),
            "allowed_pred_num_tokens": allowed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--out", type=Path, help="output root (default: sandbox/golden, or sandbox/golden-<device>)")
    args = parser.parse_args()
    out_root = args.out or (GOLDEN if args.device == "cpu" else GOLDEN.with_name(f"golden-{args.device}"))
    for name in args.models:
        start = time.perf_counter()
        model = reference.build(name, args.device)
        print(f"{name}: built upstream reference in {time.perf_counter() - start:.1f}s")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        random_req = examples.request(name, batch=4, seed=1)
        cases = {
            "trace": trace_request(name),
            "random": random_req,
            "one_target": one_target(examples.request(name, batch=3, seed=2)),
            "restricted": restricted_durations(examples.request(name, batch=3, seed=3)),
        }
        results = {}
        for case, request in cases.items():
            results[case] = reference.run(model, name, request, args.device, intermediates=True)
            np.savez_compressed(out_dir / f"{case}.npz", **{f"in.{k}": v for k, v in request.items()}, **results[case])
            print(f"  {case}: tokens {results[case]['pred_num_tokens'].tolist()}")
        request = chained_request(name, random_req, results["random"])
        chained = reference.run(model, name, request, args.device, intermediates=True)
        np.savez_compressed(out_dir / "chained.npz", **{f"in.{k}": v for k, v in request.items()}, **chained)
        print(f"  chained: tokens {chained['pred_num_tokens'].tolist()}")
        reference.write_meta(out_dir, reference.fixture_meta("sandbox/scripts/make_golden.py", args.device, cases=list(CASES)))


if __name__ == "__main__":
    main()
