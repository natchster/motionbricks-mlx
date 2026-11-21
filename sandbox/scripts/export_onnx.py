"""Export the FP32 ONNX graphs locally with NVIDIA's own exporter (tools/onnx/export.py).

Equivalent to ``uv run --project tools/onnx motionbricks-onnx --model ...`` upstream, minus
its exact-version pin check (it pins torch 2.14.0+cpu / numpy 1.26.4, which do not resolve
on macOS). The upstream graph + interface validation still runs. Faster than pulling the
released 760 MB files from Git LFS, and a good smoke test of the sandbox itself.

    uv run --all-packages python sandbox/scripts/export_onnx.py --models unitree_g1 soma
"""

from __future__ import annotations

import argparse
import time

from cudashim import reference
from cudashim.paths import ASSETS

OUT = ASSETS / "onnx"
MODELS = ("unitree_g1", "soma")


def onnx_path(name: str):
    return OUT / f"mmm_v6_{reference.kind(name)}_fp32.onnx"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    exporter = reference.import_exporter()
    from motionbricks_onnx.export import export_fp32, validate_output

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        kind, target = reference.kind(name), onnx_path(name)
        start = time.perf_counter()
        model = reference.build(name, "cpu")
        staged = target.with_suffix(".tmp.onnx")
        try:
            export_fp32(kind, model, staged)
            validate_output(staged, kind)
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)  # a failed export leaves no half-written graph
        print(f"{name}: {target} ({target.stat().st_size / 1e6:.0f} MB) in {time.perf_counter() - start:.0f}s "
              f"[exporter {exporter.__version__}]")


if __name__ == "__main__":
    main()
