# SPDX-License-Identifier: Apache-2.0
"""``cudashim-doctor``: report which CUDA substitutes work on this Mac."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .paths import CHECKPOINTS, NATIVE, UPSTREAM

XCODE = Path("/Applications/Xcode.app/Contents/Developer")


def _run(cmd, env=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def report() -> list[tuple[str, str, str]]:
    rows = []
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem = int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30
    rows.append(("host", "ok", f"{chip}, {mem:.0f} GB unified memory, macOS {platform.mac_ver()[0]}"))
    rows.append(("nvidia cuda", "absent", "no NVIDIA driver/GPU on Apple silicon; CUDA binaries cannot run"))

    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rows.append(("mlx (metal)", "ok", f"mlx {mx.__version__}, {info.get('device_name', '?')}, "
                     f"max buffer {info.get('max_buffer_length', 0) / 2**30:.0f} GB"))
    except Exception as exc:
        rows.append(("mlx (metal)", "missing", str(exc)))

    try:
        import torch

        mps = torch.backends.mps.is_available()
        rows.append(("torch.cuda -> mps", "ok" if mps else "cpu-only", f"torch {torch.__version__}, mps={mps}"))
    except Exception as exc:
        rows.append(("torch.cuda -> mps", "missing", str(exc)))

    try:
        import onnxruntime as ort

        rows.append(("tensorrt -> onnxruntime", "ok", f"ort {ort.__version__}: {', '.join(ort.get_available_providers())}"))
    except Exception as exc:
        rows.append(("tensorrt -> onnxruntime", "missing", str(exc)))

    try:
        from .warp_cpu import device_report

        w = device_report()
        rows.append(("warp kernels -> cpu", "ok", f"warp {w['warp']}, devices {w['devices']}, cuda={w['cuda_available']}"))
    except Exception as exc:
        rows.append(("warp kernels -> cpu", "missing", str(exc)))

    # Like the CuMetal installer and tests: aim xcrun at Xcode per process when xcode-select points
    # at the Command Line Tools, which have no Metal compiler.
    metal = _run(["xcrun", "-sdk", "macosx", "-f", "metal"])
    if not metal and XCODE.is_dir():
        metal = _run(["xcrun", "-sdk", "macosx", "-f", "metal"], env={**os.environ, "DEVELOPER_DIR": str(XCODE)})
        metal = metal and f"{metal} (DEVELOPER_DIR={XCODE})"
    rows.append(("xcode metal compiler", "ok" if metal else "missing", metal or "install Xcode + `xcodebuild -downloadComponent MetalToolchain`"))
    cumetal = shutil.which("cumetal") or next(iter((NATIVE / "toolchains").glob("cumetal*/bin/cumetal")), None)
    rows.append(("cuda c++ -> metal (cumetal)", "ok" if cumetal else "not installed",
                 str(cumetal) if cumetal else "optional: sandbox/native/install_cumetal.sh"))

    rows.append(("upstream reference", "ok" if (UPSTREAM / "tools/onnx/models").is_dir() else "missing", str(UPSTREAM)))
    try:
        from motionbricks_mlx import hub

        # Size check only (cheap); `motionbricks-mlx download --dir ...` verifies SHA-256 too.
        entries = hub.manifest()
        found = sum((CHECKPOINTS / e.path).is_file() and (CHECKPOINTS / e.path).stat().st_size == e.size for e in entries)
        rows.append(("checkpoints", "ok" if found == len(entries) else "incomplete",
                     f"{found}/{len(entries)} files of the right size in {CHECKPOINTS}"))
    except Exception as exc:
        rows.append(("checkpoints", "unknown", str(exc)))
    return rows


def main():
    rows = report()
    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{name:<{width}}  {status:<13} {detail}")


if __name__ == "__main__":
    main()
