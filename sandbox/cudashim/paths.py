# SPDX-License-Identifier: Apache-2.0
"""Where the sandbox keeps its (gitignored) material, in one place."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "sandbox"
UPSTREAM = SANDBOX / "upstream"  # NVlabs/motionbricks at the pinned commit (fetch_assets.py)
ASSETS = SANDBOX / "assets"  # locally exported ONNX graphs (export_onnx.py)
GOLDEN = SANDBOX / "golden"  # reference fixtures (make_golden.py, make_runtime_golden.py)
NATIVE = SANDBOX / "native"
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"  # holds soma/ and g1/ (motionbricks_mlx.hub)
