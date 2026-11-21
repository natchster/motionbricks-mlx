# SPDX-License-Identifier: Apache-2.0
"""Synthetic but well-formed requests, for smoke tests and benchmarks.

The trace inputs mirror upstream ``tools/onnx/export.py::example_inputs``; ``random_request``
builds plausible varied requests (a standing pose moving forward with a turn) so that
benchmarks do not just replay one degenerate input.
"""

from __future__ import annotations

import numpy as np

from .config import CONTEXT_AND_TARGET, NUM_TOKEN_CHOICES


def _masks(batch, num_targets=4):
    has = np.zeros((batch, CONTEXT_AND_TARGET), np.int32)
    has[:, : 4 + num_targets] = 1
    has_local = has.copy()
    has_local[:, 3] = 0  # velocities unknown at the last context pose and the last target pose
    has_local[:, 3 + num_targets] = 0
    return has, has_local


def g1_request(batch: int = 1, seed: int = 0, distance: float = 1.0, turn: float = 0.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    context = np.zeros((batch, 4, 36), np.float32)
    context[..., 2] = 0.75  # pelvis height (Z up)
    context[..., 6] = 1.0  # identity quaternion, XYZW
    context[..., 7:] = rng.normal(0, 0.05, size=(batch, 1, 29)).astype(np.float32)
    target = context.copy()
    target[..., 0] = distance + rng.uniform(-0.2, 0.2, size=(batch, 1))
    target[..., 1] = rng.uniform(-0.3, 0.3, size=(batch, 1))
    yaw = turn + rng.uniform(-0.3, 0.3, size=(batch, 1))
    target[..., 5] = np.sin(yaw / 2)
    target[..., 6] = np.cos(yaw / 2)
    has, has_local = _masks(batch)
    return {
        "context_qpos": context,
        "target_qpos": target,
        "has_global_root_values": has,
        "has_local_root_values": has_local,
        "has_local_poses": has.copy(),
        "allowed_pred_num_tokens": np.ones((batch, NUM_TOKEN_CHOICES), np.int32),
    }


def soma_request(batch: int = 1, seed: int = 0, distance: float = 1.5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(8, dtype=np.float32)
    global_root = np.zeros((batch, 8, 5), np.float32)
    speed = rng.uniform(0.5, 1.5, size=(batch, 1)).astype(np.float32)
    global_root[..., 0] = np.where(t < 4, t * speed / 30, distance + (t - 4) * speed / 30)
    global_root[..., 1] = 0.95  # hip height (Y up)
    heading = rng.uniform(-0.5, 0.5, size=(batch, 1))
    global_root[..., 3] = np.cos(np.where(t < 4, 0, heading))
    global_root[..., 4] = np.sin(np.where(t < 4, 0, heading))
    local_root = np.zeros((batch, 8, 4), np.float32)
    local_root[..., 1] = speed
    local_root[..., 3] = 0.95
    local_poses = np.zeros((batch, 8, 267), np.float32)
    local_poses[..., :87] = rng.normal(0, 0.3, size=(batch, 1, 87))
    local_poses[..., 87:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), 30)
    local_poses[..., 87:] += rng.normal(0, 0.02, size=(batch, 1, 180)).astype(np.float32)
    has, has_local = _masks(batch)
    return {
        "global_root_values": global_root,
        "has_global_root_values": has,
        "local_root_values": local_root,
        "has_local_root_values": has_local,
        "local_poses": local_poses,
        "has_local_poses": has.copy(),
        "allowed_pred_num_tokens": np.ones((batch, NUM_TOKEN_CHOICES), np.int32),
    }


def request(name: str, batch: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    return g1_request(batch, seed) if name in ("unitree_g1", "g1") else soma_request(batch, seed)
