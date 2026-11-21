"""Shared helpers for the MLX test-suite (not named conftest: sandbox/tests has one too)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"
GOLDEN = ROOT / "sandbox" / "golden"
HF_FOLDERS = {"unitree_g1": "g1", "soma": "soma"}
MODELS = ("unitree_g1", "soma")
CASES = ("trace", "random", "one_target", "restricted", "chained")


def have_checkpoints(name: str) -> bool:
    return any((CHECKPOINTS / HF_FOLDERS[name]).rglob("*.safetensors"))


@lru_cache(maxsize=None)
def load_model(name: str, dtype: str = "float32"):
    import motionbricks_mlx as mb

    if not have_checkpoints(name):
        pytest.skip(f"{name} checkpoints missing (run sandbox/setup.sh)")
    return mb.load(name, CHECKPOINTS / HF_FOLDERS[name], dtype=dtype)


def load_golden(name: str, case: str) -> dict[str, np.ndarray]:
    path = GOLDEN / name / f"{case}.npz"
    if not path.is_file():
        pytest.skip(f"golden fixture {path} missing (run sandbox/scripts/make_golden.py)")
    return dict(np.load(path))


def golden_inputs(golden: dict) -> dict[str, np.ndarray]:
    return {k[3:]: v for k, v in golden.items() if k.startswith("in.")}

