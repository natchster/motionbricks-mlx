"""Aggregate bench/results/raw/*.json (written by run_all.sh) into markdown tables.

Each round ran Python and Rust back to back, so ratios are computed *within* a round and then
summarized (median and range across rounds); absolute numbers are medians of the per-round medians.

    python3 model-rs/bench/report.py model-rs/bench/results/raw --table metal  > model-rs/bench/results/report_metal.md
    python3 model-rs/bench/report.py model-rs/bench/results/raw --table mlx-rs > model-rs/bench/results/report_mlx-rs.md

``--table metal``: Python MLX vs Rust mlx-rs vs the Rust custom Metal runtime, end to end (the
mlx-rs port has no INT8 mode, so its INT8 cells are empty). ``--table mlx-rs`` (the default):
Python vs mlx-rs core, graph-build, eval-CPU and end-to-end time, plus the request streams.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DTYPE_ORDER = ["float32", "bfloat16", "float16", "int8"]
MODELS = ["soma", "unitree_g1"]
TIMES, DASH = "\u00d7", "\u2013"  # the published tables use the typographic multiplication sign and en dash


def load_rounds(raw: Path):
    """((dtype, round, model) -> runtime -> results, 1-min load averages) from a raw directory."""
    runs = defaultdict(dict)
    loads = []
    for path in sorted(raw.glob("*.json")):
        dtype, rnd, rest = path.stem.split("_", 2)
        model, runtime = rest.rsplit("_", 1)
        runs[(dtype, int(rnd[1:]), model)][runtime] = json.loads(path.read_text())["results"]
        load_file = path.with_suffix(".load")
        if load_file.exists():
            loads.append(float(load_file.read_text()))
    return runs, loads


def ordered_dtypes(runs) -> list[str]:
    return sorted({k[0] for k in runs}, key=lambda d: DTYPE_ORDER.index(d) if d in DTYPE_ORDER else len(DTYPE_ORDER))


def fmt_ratio(values):
    m = statistics.median(values)
    return f"{m:.2f}{TIMES} ({min(values):.2f}{DASH}{max(values):.2f})"


def median_of(rounds, runtime: str, key: str, field: str) -> float:
    return statistics.median(r[runtime][key][field] for r in rounds)


def table_mlx_rs(runs, loads) -> None:
    load_text = (f"median {statistics.median(loads):.1f}, range {min(loads):.1f}{DASH}{max(loads):.1f}" if loads
                 else "not recorded")
    print(f"Rounds: {len({k[1] for k in runs})} per dtype; 1-min load average during runs: {load_text}.\n")
    for dtype in ordered_dtypes(runs):
        for model in MODELS:
            rounds = [v for k, v in runs.items() if k[0] == dtype and k[2] == model and len(v) == 2]
            # mlx-rs rows only (the Metal runtime is the `metal` table; mlx-rs has no INT8 mode)
            if not rounds or not any(k.endswith("/compiled") for r in rounds for k in r["rust"]):
                continue
            print(f"#### {model}, {dtype}: median ms per batched call (Rust ÷ Python, median and range over {len(rounds)} rounds)\n")
            print("| N | mode | Python core | Rust core | core ratio | Python graph build | Rust graph build | "
                  "Python eval CPU | Rust eval CPU | Python end-to-end | Rust end-to-end |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            keys = sorted({k for r in rounds for k in r["rust"] if "/N=" in k and all(k in q["python"] for q in rounds)},
                          key=lambda k: (int(re.search(r"N=(\d+)", k).group(1)), k.endswith("eager")))
            for key in keys:
                n = re.search(r"N=(\d+)", key).group(1)
                mode = key.rsplit("/", 1)[1]
                ratio = [r["rust"][key]["core_ms"] / r["python"][key]["core_ms"] for r in rounds]
                cells = [f"{median_of(rounds, rt, key, field):.2f}" for field in ("core_ms", "graph_ms", "eval_cpu_ms", "total_ms")
                         for rt in ("python", "rust")]
                print(f"| {n} | {mode} | {cells[0]} | {cells[1]} | {fmt_ratio(ratio)} | {cells[2]} | {cells[3]} | "
                      f"{cells[4]} | {cells[5]} | {cells[6]} | {cells[7]} |")
            skey = f"{model}/{dtype}/stream"
            if all(skey in r["rust"] and skey in r["python"] for r in rounds):
                print("\nStream of single (N=1) requests, requests/s:\n")
                print("| runtime | sequential | pipelined (`async_eval`) | pipelining gain |")
                print("|---|---|---|---|")
                for rt in ("python", "rust"):
                    seq = statistics.median(r[rt][skey]["sequential_req_s"] for r in rounds)
                    pip = statistics.median(r[rt][skey]["pipelined_req_s"] for r in rounds)
                    print(f"| {rt} | {seq:.0f} | {pip:.0f} | {pip / seq:.2f}{TIMES} |")
            print()


def table_metal(runs, loads) -> None:
    rounds = sorted({k[1] for k in runs})
    load_text = (f"median {statistics.median(loads):.1f} (range {min(loads):.1f}-{max(loads):.1f})" if loads
                 else "not recorded")
    print(f"{len(rounds)} interleaved rounds; 1-min load average {load_text}.\n")
    for dtype, model in [(d, m) for d in ordered_dtypes(runs) for m in MODELS]:
        rs = [runs[(dtype, r, model)] for r in rounds if len(runs.get((dtype, r, model), {})) == 2]
        if not rs:
            continue
        print(f"#### {model} {dtype}: median ms per batched call, end to end (inputs in, outputs on the host)\n")
        print("| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |")
        print("|---|---|---|---|---|---|---|")
        has_rs = all(f"{model}/{dtype}/N=1/compiled" in r["rust"] for r in rs)
        for n in (1, 2, 4, 8, 16, 32):
            kc = f"{model}/{dtype}/N={n}/compiled"
            km = f"{model}/{dtype}/N={n}/metal"
            if not all(kc in r["python"] and km in r["rust"] for r in rs):
                continue
            py = [r["python"][kc]["total_ms"] for r in rs]
            rm = [r["rust"][kc]["total_ms"] for r in rs] if has_rs else [float("nan")] * len(rs)
            mt = [r["rust"][km]["total_ms"] for r in rs]
            gpu = [r["rust"][km]["gpu_ms"] for r in rs]
            sp = [a / b for a, b in zip(py, mt)]
            sr = [a / b for a, b in zip(rm, mt)]
            rs_cell = f"{statistics.median(rm):.2f}" if has_rs else "—"
            sr_cell = f"{statistics.median(sr):.2f}x ({min(sr):.2f}-{max(sr):.2f})" if has_rs else "—"
            print(f"| {n} | {statistics.median(py):.2f} | {rs_cell} | **{statistics.median(mt):.2f}** | "
                  f"{statistics.median(sp):.2f}x ({min(sp):.2f}-{max(sp):.2f}) | {sr_cell} | {statistics.median(gpu):.2f} |")
        sk = f"{model}/{dtype}/stream"
        if has_rs and all(sk in r["python"] and sk in r["rust"] for r in rs):
            py_seq = statistics.median(r["python"][sk]["sequential_req_s"] for r in rs)
            py_pip = statistics.median(r["python"][sk]["pipelined_req_s"] for r in rs)
            rm_pip = statistics.median(r["rust"][sk]["pipelined_req_s"] for r in rs)
            mt_seq = statistics.median(r["rust"][sk]["metal_sequential_req_s"] for r in rs)
            print(f"\nStream of single requests (req/s): Python sequential {py_seq:.0f}, Python pipelined {py_pip:.0f}, "
                  f"mlx-rs pipelined {rm_pip:.0f}, **Metal runtime sequential {mt_seq:.0f}**\n")
        else:
            print()


TABLES = {"mlx-rs": table_mlx_rs, "metal": table_metal}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", nargs="?", type=Path, default=Path(__file__).resolve().parent / "results" / "raw",
                        help="directory of run_all.sh results (default: bench/results/raw)")
    parser.add_argument("--table", choices=sorted(TABLES), default="mlx-rs")
    args = parser.parse_args(argv)
    runs, loads = load_rounds(args.raw)
    if not runs:
        raise SystemExit(f"no results in {args.raw} (run model-rs/bench/run_all.sh first)")
    TABLES[args.table](runs, loads)


if __name__ == "__main__":
    main()
