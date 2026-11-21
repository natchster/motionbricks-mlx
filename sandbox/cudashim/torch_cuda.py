# SPDX-License-Identifier: Apache-2.0
"""Run CUDA-targeting PyTorch code on Apple silicon by redirecting ``cuda`` to ``mps``.

PyTorch's macOS wheels have no CUDA backend at all (``.cuda()`` raises "Torch not
compiled with CUDA enabled"). This shim rewrites the *device*, not the kernels: every
``device="cuda[:N]"`` / ``.cuda()`` / ``.to("cuda")`` lands on the Metal (MPS) backend, and
the ``torch.cuda`` utility API (sync, events, seeds, memory stats, autocast) is mapped onto
``torch.mps``; ``torch.autocast("cuda")``, ``torch.amp.autocast("cuda")`` and the deprecated
``torch.cuda.amp.autocast`` autocast on the target device. Kernels that MPS lacks fall back to
CPU when ``PYTORCH_ENABLE_MPS_FALLBACK=1`` (set by :func:`enable`). FP64 does not exist on
Apple GPUs.

    from cudashim import torch_cuda
    with torch_cuda.enabled():            # or torch_cuda.enable() for the whole process
        x = torch.randn(4, 4, device="cuda")   # -> mps:0
"""

from __future__ import annotations

import contextlib
import os
import platform
from types import SimpleNamespace

import torch
from torch.overrides import TorchFunctionMode

_state: dict = {"mode": None, "saved": None, "target": None, "saved_autocast": None}


def _target() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _is_cuda(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.split(":")[0] == "cuda"
    return False


def _remap(value):
    return _state["target"] if _is_cuda(value) else value


class CudaToMpsMode(TorchFunctionMode):
    """Rewrites CUDA device arguments of *every* torch call to the shim target."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if "device" in kwargs:
            kwargs["device"] = _remap(kwargs["device"])
        if func is torch.Tensor.cuda or getattr(func, "__name__", "") == "cuda":
            return args[0].to(_state["target"], non_blocking=kwargs.get("non_blocking", False))
        if func is torch.Tensor.pin_memory:
            return args[0]  # unified memory: nothing to pin
        if args and any(_is_cuda(a) for a in args[1:]):
            args = (args[0], *(_remap(a) for a in args[1:]))
        return func(*args, **kwargs)


class _NullStream:
    """Stand-in for ``torch.cuda.Stream``; MPS runs one in-order command queue."""

    def __init__(self, *args, **kwargs):
        self.device = _state["target"]

    def synchronize(self):
        _sync()

    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        event = event or _event_cls()(enable_timing=False)
        event.record()
        return event

    def query(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sync():
    if _state["target"].type == "mps":
        torch.mps.synchronize()


def _event_cls():
    return torch.mps.Event if _state["target"].type == "mps" else _CpuEvent


class _CpuEvent:
    def __init__(self, enable_timing=False, **_):
        self.t = None

    def record(self, stream=None):
        import time

        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass

    def elapsed_time(self, end):
        return (end.t - self.t) * 1e3


def _device_properties(device=None):
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return SimpleNamespace(
        name=f"{platform.processor() or 'arm'} Apple GPU (cudashim->{_state['target'].type})",
        total_memory=total, major=8, minor=0, multi_processor_count=1, is_integrated=True,
    )


def _patches():
    mps = _state["target"].type == "mps"
    return {
        "is_available": lambda: True,
        "is_initialized": lambda: True,
        "init": lambda: None,
        "device_count": lambda: 1,
        "current_device": lambda: 0,
        "set_device": lambda device: None,
        "get_device_name": lambda device=None: _device_properties().name,
        "get_device_capability": lambda device=None: (8, 0),
        "get_device_properties": _device_properties,
        "is_bf16_supported": lambda *a, **k: True,
        "synchronize": lambda device=None: _sync(),
        "empty_cache": (torch.mps.empty_cache if mps else (lambda: None)),
        "manual_seed": (torch.mps.manual_seed if mps else torch.manual_seed),
        "manual_seed_all": (torch.mps.manual_seed if mps else torch.manual_seed),
        "memory_allocated": (lambda device=None: torch.mps.current_allocated_memory()) if mps else (lambda device=None: 0),
        "max_memory_allocated": (lambda device=None: torch.mps.driver_allocated_memory()) if mps else (lambda device=None: 0),
        "reset_peak_memory_stats": lambda device=None: None,
        "Event": _event_cls(),
        "Stream": _NullStream,
        "current_stream": lambda device=None: _NullStream(),
        "default_stream": lambda device=None: _NullStream(),
        "stream": lambda stream=None: contextlib.nullcontext(stream),
        "device": lambda device=None: contextlib.nullcontext(),
    }


def _autocast_on(target_type: str, base: type) -> type:
    """``base`` (``torch.autocast``) with ``device_type="cuda"`` meaning the shim target.

    Without it, ``torch.autocast("cuda")`` silently does nothing on a Mac: the shim reports CUDA
    as available, so torch doesn't even warn, and the code runs in full precision."""

    class autocast(base):
        def __init__(self, device_type, *args, **kwargs):
            super().__init__(target_type if device_type == "cuda" else device_type, *args, **kwargs)

    autocast.__qualname__ = autocast.__name__ = "autocast"
    autocast.__module__ = __name__
    return autocast


def enable() -> torch.device:
    """Install the shim process-wide. Returns the device that ``cuda`` now means."""
    if _state["mode"] is not None:
        return _state["target"]
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _state["target"] = _target()
    patches = _patches()
    _state["saved"] = {name: getattr(torch.cuda, name, None) for name in patches}
    for name, value in patches.items():
        setattr(torch.cuda, name, value)
    target_type = _state["target"].type
    _state["saved_autocast"] = {
        "cuda.amp": torch.cuda.amp.autocast if hasattr(torch.cuda, "amp") else None,
        "torch": torch.autocast,
        "amp": torch.amp.autocast,
    }
    patched = _autocast_on(target_type, torch.autocast)
    torch.autocast = torch.amp.autocast = patched
    if hasattr(torch.cuda, "amp"):
        torch.cuda.amp.autocast = lambda *a, **k: patched(target_type, *a, **k)
    _state["mode"] = CudaToMpsMode()
    _state["mode"].__enter__()
    return _state["target"]


def disable() -> None:
    if _state["mode"] is None:
        return
    _state["mode"].__exit__(None, None, None)
    for name, value in _state["saved"].items():
        if value is None:
            delattr(torch.cuda, name)
        else:
            setattr(torch.cuda, name, value)
    saved = _state["saved_autocast"]
    torch.autocast, torch.amp.autocast = saved["torch"], saved["amp"]
    if saved["cuda.amp"] is not None:
        torch.cuda.amp.autocast = saved["cuda.amp"]
    _state.update(mode=None, saved=None, target=None, saved_autocast=None)


@contextlib.contextmanager
def enabled():
    device = enable()
    try:
        yield device
    finally:
        disable()
