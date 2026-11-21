"""Quick paired Python-MLX vs Rust-Metal comparison (one round, core time) for development.

    python3 model-rs/bench/quick_compare.py --dtypes float32 bfloat16 --batches 1 8 32

The runtime that goes first alternates from row to row, so thermal drift does not favour one
side in every row. For numbers to publish, use run_all.sh (several interleaved rounds).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ("soma", "unitree_g1")
BATCHES = (1, 2, 4, 8, 16, 32)  # the request files make_requests.py writes
ENV = {**os.environ, "DEVELOPER_DIR": os.environ.get("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")}


def run(cmd, cwd=ROOT):
    """Run a step; on failure show its output (it is hidden on success) and stop."""
    result = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit(f"failed ({result.returncode}): {' '.join(map(str, cmd))}")


def prepare():
    """What run_all.sh does first: build mbrs-bench, sync the model package once, write the requests."""
    run(["cargo", "build", "--release", "--quiet"], cwd=ROOT / "model-rs")
    run(["uv", "sync", "--inexact", "--package", "motionbricks-mlx", "--quiet"])
    if not all((HERE / "requests" / f"{m}_N{n}.npz").is_file() for m in MODELS for n in BATCHES):
        run(["uv", "run", "--no-sync", "python", "model-rs/bench/make_requests.py"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtypes", nargs="+", default=["float32", "float16", "bfloat16", "int8"],
                   choices=("float32", "float16", "bfloat16", "int8"))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--batches", nargs="+", default=["1", "8", "32"], choices=[str(n) for n in BATCHES])
    p.add_argument("--iters", default="20")
    a = p.parse_args()
    prepare()
    print(f"{'model':<11}{'dtype':<10}" + "".join(f"| N={n:<3} py / rs (ratio) " for n in a.batches))
    with tempfile.TemporaryDirectory() as tmp:
        rows = [(dtype, model) for dtype in a.dtypes for model in a.models]
        for row, (dtype, model) in enumerate(rows):
            pj, rj = Path(tmp) / f"py_{model}_{dtype}.json", Path(tmp) / f"rs_{model}_{dtype}.json"
            python = ["uv", "run", "--no-sync", "python", "model-rs/bench/bench_python.py", "--model", model, "--dtype", dtype,
                      "--batches", *a.batches, "--iters", a.iters, "--stream", "0", "--json", str(pj)]
            rust = ["model-rs/target/release/mbrs-bench", "--model", model, "--dtype", dtype, "--batches", *a.batches,
                    "--iters", a.iters, "--stream", "0", "--json", str(rj)]
            for cmd in (python, rust) if row % 2 == 0 else (rust, python):
                run(cmd)
            py, rs = json.loads(pj.read_text())["results"], json.loads(rj.read_text())["results"]
            cells = []
            for n in a.batches:
                x = py[f"{model}/{dtype}/N={n}/compiled"]["total_ms"]
                y = rs[f"{model}/{dtype}/N={n}/metal"]["total_ms"]
                cells.append(f"| {x:6.2f} / {y:6.2f} ({x / y:4.2f}x) ")
            print(f"{model:<11}{dtype:<10}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
