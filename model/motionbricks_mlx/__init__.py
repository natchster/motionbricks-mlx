# SPDX-License-Identifier: Apache-2.0
"""MLX port of NVIDIA MotionBricks for Apple silicon.

    import motionbricks_mlx as mb
    model = mb.load("soma", "checkpoints/MotionBricks/soma")       # from the HF checkpoints
    outputs = model.run(inputs)                                     # ONNX-style dict in / dict out
"""

from .config import FRAMES_PER_TOKEN, MAX_FRAMES, MAX_TOKENS, MIN_TOKENS, G1, SOMA, get_spec
from .model import MotionBricks, convert, from_checkpoints, from_converted, load

__version__ = "0.1.0"
__all__ = [
    "FRAMES_PER_TOKEN", "MAX_FRAMES", "MAX_TOKENS", "MIN_TOKENS", "G1", "SOMA", "MotionBricks",
    "convert", "from_checkpoints", "from_converted", "get_spec", "load",
]
