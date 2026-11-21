"""Shared helpers for the MLX test-suite (not named conftest: sandbox/tests has one too).

Checkpoints: ``checkpoints/MotionBricks`` (``motionbricks-mlx download``), or
``$MOTIONBRICKS_CHECKPOINTS``. Golden fixtures from the PyTorch reference: ``sandbox/golden``
(``sandbox/setup.sh``), or ``$MOTIONBRICKS_GOLDEN``. Tests needing either skip without them,
or fail with ``MOTIONBRICKS_REQUIRE_FIXTURES=1`` (see conftest.py).
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = Path(os.environ.get("MOTIONBRICKS_CHECKPOINTS", ROOT / "checkpoints" / "MotionBricks"))
GOLDEN = Path(os.environ.get("MOTIONBRICKS_GOLDEN", ROOT / "sandbox" / "golden"))
HF_FOLDERS = {"unitree_g1": "g1", "soma": "soma"}
MODELS = ("unitree_g1", "soma")
CASES = ("trace", "random", "one_target", "restricted", "chained")


def have_checkpoints(name: str) -> bool:
    return (CHECKPOINTS / HF_FOLDERS[name]).is_dir() and any((CHECKPOINTS / HF_FOLDERS[name]).rglob("*.safetensors"))


def fresh_model(name: str, dtype: str = "float32"):
    """A new, uncached model, for tests that change it; skips without the checkpoints."""
    import motionbricks_mlx as mb

    if not have_checkpoints(name):
        pytest.skip(f"{name} checkpoints missing (run `motionbricks-mlx download`)")
    return mb.load(name, CHECKPOINTS / HF_FOLDERS[name], dtype=dtype)


@cache
def load_model(name: str, dtype: str = "float32"):
    return fresh_model(name, dtype)


def load_golden(name: str, case: str) -> dict[str, np.ndarray]:
    path = GOLDEN / name / f"{case}.npz"
    if not path.is_file():
        pytest.skip(f"golden fixture {path} missing (made by the sandbox: sandbox/setup.sh)")
    return dict(np.load(path))


def golden_inputs(golden: dict) -> dict[str, np.ndarray]:
    return {k[3:]: v for k, v in golden.items() if k.startswith("in.")}

