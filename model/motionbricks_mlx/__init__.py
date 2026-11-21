# SPDX-License-Identifier: Apache-2.0
"""MLX port of NVIDIA MotionBricks for Apple silicon.

    import motionbricks_mlx as mb
    model = mb.load("soma", "checkpoints/MotionBricks/soma")       # from the HF checkpoints
    outputs = model.run(inputs)                                     # ONNX-style dict in / dict out

``mb.download()`` fetches and verifies the checkpoints; ``mb.runtime`` is the Runtime API
(engines, tokens, animation buffers, motion players), see docs/runtime_api.md.
"""

from .config import FRAMES_PER_TOKEN, MAX_FRAMES, MAX_TOKENS, MIN_TOKENS, G1, SOMA, get_spec
from .model import MotionBricks, convert, from_checkpoints, from_converted, load
from . import hub, runtime  # noqa: E402
from .hub import download

__version__ = "0.1.0"
__all__ = [
    "FRAMES_PER_TOKEN", "MAX_FRAMES", "MAX_TOKENS", "MIN_TOKENS", "G1", "SOMA", "MotionBricks",
    "convert", "download", "from_checkpoints", "from_converted", "get_spec", "hub", "load", "runtime",
]
