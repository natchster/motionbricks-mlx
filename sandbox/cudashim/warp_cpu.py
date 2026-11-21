# SPDX-License-Identifier: Apache-2.0
"""Run NVIDIA Warp kernels written for CUDA on the Apple-silicon CPU.

Warp compiles the same ``@wp.kernel`` Python source to LLVM for the CPU when no CUDA
device exists, so kernels need no translation. What differs on macOS is the *runtime*
around them: there are no streams, CUDA graphs, memory pools or ``cuda:0`` device.
:func:`cuda_device_alias` lets code that selects the device through ``wp.set_device("cuda:0")``
/ ``wp.get_device("cuda")`` (and so ``wp.ScopedDevice("cuda")`` and allocations on the default
device) run unchanged by aliasing those names to ``cpu``. An explicit ``device="cuda"`` argument
to an allocation or launch (``wp.zeros(..., device="cuda")``) is *not* aliased: Warp resolves it
internally and raises, so such code needs ``device="cpu"``.
"""

from __future__ import annotations

import contextlib
import sys

import warp as wp

from .paths import UPSTREAM

__all__ = ["UPSTREAM", "cuda_device_alias", "device_report", "upstream_on_path"]


@contextlib.contextmanager
def upstream_on_path():
    """Temporarily import the pinned upstream ``motionbricks`` package."""
    path = str(UPSTREAM)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


@contextlib.contextmanager
def cuda_device_alias():
    """Within the block, ``wp.get_device("cuda[:0]")`` / ``wp.set_device("cuda:0")`` mean the CPU.

    Explicit ``device="cuda"`` arguments to allocations and launches still raise (see module doc)."""
    wp.init()
    original_get = wp.get_device
    original_set = wp.set_device

    def remap(ident):
        if isinstance(ident, str) and ident.split(":")[0] == "cuda":
            return "cpu"
        return ident

    wp.get_device = lambda ident=None: original_get(remap(ident))
    wp.set_device = lambda ident: original_set(remap(ident))
    try:
        yield wp.get_device("cpu")
    finally:
        wp.get_device, wp.set_device = original_get, original_set


def device_report() -> dict:
    wp.init()
    return {
        "warp": wp.__version__,
        "devices": [d.alias for d in wp.get_devices()],
        "cuda_available": wp.is_cuda_available(),
    }
