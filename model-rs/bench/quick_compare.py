"""Quick paired Python-MLX vs Rust-Metal comparison (one round, core time) for development.

    python3 model-rs/bench/quick_compare.py --dtypes float32 bfloat16 --batches 1 8 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtypes", nargs="+", default=["float32", "float16", "bfloat16", "int8"])
    p.add_argument("--models", nargs="+", default=["soma", "unitree_g1"])
    p.add_argument("--batches", nargs="+", default=["1", "8", "32"])
    p.add_argument("--iters", default="20")
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp())
    print(f"{'model':<11}{'dtype':<10}" + "".join(f"| N={n:<3} py / rs (ratio) " for n in a.batches))
    for dtype in a.dtypes:
        for model in a.models:
            pj, rj = tmp / f"py_{model}_{dtype}.json", tmp / f"rs_{model}_{dtype}.json"
            run(["uv", "run", "--all-packages", "python", "model-rs/bench/bench_python.py", "--model", model, "--dtype", dtype,
                 "--batches", *a.batches, "--iters", a.iters, "--stream", "0", "--json", str(pj)])
            run(["model-rs/target/release/mbrs-bench", "--model", model, "--dtype", dtype, "--batches", *a.batches,
                 "--iters", a.iters, "--stream", "0", "--json", str(rj)])
            py, rs = json.loads(pj.read_text())["results"], json.loads(rj.read_text())["results"]
            cells = []
            for n in a.batches:
                x = py[f"{model}/{dtype}/N={n}/compiled"]["total_ms"]
                y = rs[f"{model}/{dtype}/N={n}/metal"]["total_ms"]
                cells.append(f"| {x:6.2f} / {y:6.2f} ({x / y:4.2f}x) ")
            print(f"{model:<11}{dtype:<10}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
