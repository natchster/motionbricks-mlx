#!/usr/bin/env python3
"""Benchmark the Rust ONNX exporter against the original torch-tracing exporter.

For both models: wall time and peak RSS of each export, the node/initializer counts and
file size of each graph, and ONNX Runtime session-creation time for each graph. Prints a
Markdown table for tools/onnx/README.md and docs/user_guide.md.

    uv run --all-packages python tools/onnx/bench_export.py [--models soma g1]

Peak RSS is the RUSAGE_CHILDREN delta around each export subprocess, so the two exporters
are measured the same way.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import onnx
import onnxruntime as ort

from cudashim.paths import CHECKPOINTS, ROOT

CRATE = ROOT / "tools/onnx"
BINARY = CRATE / "target/release/motionbricks-onnx"
# label -> (upstream kind, checkpoint folder); the Rust CLI takes unitree_g1 for g1.
MODELS = {"soma": ("soma", "soma"), "g1": ("g1", "g1")}


def timed(command: list[str]) -> tuple[float, float]:
    """Run `command`; return (wall seconds, peak child RSS in MB).

    A fresh wrapper process per measurement: RUSAGE_CHILDREN's ru_maxrss is the max over
    all waited-for children, so wrapping each run gives the per-export peak. The wrapper's
    own RSS is not counted.
    """
    wrapper = (
        "import json, resource, subprocess, sys, time\n"
        "start = time.perf_counter()\n"
        "subprocess.run(sys.argv[1:], check=True)\n"
        "print(json.dumps({'wall': time.perf_counter() - start,\n"
        "                  'rss_mb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6}))\n"
    )
    out = subprocess.run([sys.executable, "-c", wrapper, *map(str, command)],
                         check=True, capture_output=True, text=True).stdout
    result = json.loads(out.strip().splitlines()[-1])
    return result["wall"], result["rss_mb"]


def export_rust(kind: str, checkpoints: Path, out: Path) -> tuple[float, float]:
    cli_model = "unitree_g1" if kind == "g1" else "soma"
    return timed([str(BINARY), "--model", cli_model, "--checkpoints", str(checkpoints), "--output", str(out)])


def export_python(kind: str, checkpoints: Path, out: Path) -> tuple[float, float]:
    """Upstream's own exporter (through the sandbox's reference oracle), same staged write."""
    script = """
import sys
from cudashim import reference
reference.import_exporter()
from motionbricks_onnx.export import export_fp32, validate_output
kind, name, ckpt, out = sys.argv[1:5]
model = reference.build(kind, "cpu")
export_fp32(kind, model, out)
validate_output(out, kind)
"""
    return timed([sys.executable, "-c", script, kind, kind, str(checkpoints), str(out)])


def graph_stats(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    weight_bytes = sum(len(t.raw_data) for t in graph.initializer)
    start = time.perf_counter()
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    session_s = time.perf_counter() - start
    return {
        "nodes": len(graph.node),
        "initializers": len(graph.initializer),
        "file_mb": path.stat().st_size / 1e6,
        "weight_mb": weight_bytes / 1e6,
        "ort_session_s": session_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    if not BINARY.is_file():
        subprocess.run(["cargo", "build", "--release"], cwd=CRATE, check=True)
    rows = []
    for kind in args.models:
        _, folder = MODELS[kind]
        checkpoints = CHECKPOINTS / folder
        with tempfile.TemporaryDirectory(prefix="onnx-bench-") as tmp:
            rust_out = Path(tmp) / "rust.onnx"
            python_out = Path(tmp) / "python.onnx"
            rust_s, rust_rss = export_rust(kind, checkpoints, rust_out)
            python_s, python_rss = export_python(kind, checkpoints, python_out)
            rust = graph_stats(rust_out)
            python = graph_stats(python_out)
        rows.append({"model": kind, "rust": {"export_s": rust_s, "export_rss_mb": rust_rss, **rust},
                     "python": {"export_s": python_s, "export_rss_mb": python_rss, **python}})
    print("| model | exporter | export s | peak RSS GB | nodes | file MB | ORT session s |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        for exporter in ("rust", "python"):
            r = row[exporter]
            print(f"| {row['model']} | {exporter} | {r['export_s']:.1f} | {r['export_rss_mb'] / 1e3:.2f} "
                  f"| {r['nodes']} | {r['file_mb']:.0f} | {r['ort_session_s']:.2f} |")
    print(json.dumps(rows, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
