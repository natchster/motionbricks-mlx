"""Aggregate bench/results/raw/*.json (written by run_all.sh) into markdown tables.

Each round ran Python and Rust back to back, so the Rust/Python ratio is computed *within* a
round and then summarized (median and range across rounds); absolute numbers are medians of the
per-round medians.

    python3 model-rs/bench/report.py > model-rs/bench/results/report.md
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import sys

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "raw"


def load():
    runs = defaultdict(dict)  # (dtype, round, model) -> runtime -> results
    loads = []
    for path in sorted(RAW.glob("*.json")):
        dtype, rnd, rest = path.stem.split("_", 2)
        model, runtime = rest.rsplit("_", 1)
        runs[(dtype, int(rnd[1:]), model)][runtime] = json.loads(path.read_text())["results"]
        load_file = path.with_suffix(".load")
        if load_file.exists():
            loads.append(float(load_file.read_text()))
    return runs, loads


def fmt_ratio(values):
    m = statistics.median(values)
    return f"{m:.2f}× ({min(values):.2f}–{max(values):.2f})"


def main():
    runs, loads = load()
    dtypes = sorted({k[0] for k in runs}, key=lambda d: d != "float32")
    models = ["soma", "unitree_g1"]
    print(f"Rounds: {len({k[1] for k in runs})} per dtype; 1-min load average during runs: "
          f"median {statistics.median(loads):.1f}, range {min(loads):.1f}–{max(loads):.1f}.\n")
    for dtype in dtypes:
        for model in models:
            rounds = [v for k, v in runs.items() if k[0] == dtype and k[2] == model and len(v) == 2]
            if not rounds:
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
                med = lambda rt, f: statistics.median(r[rt][key][f] for r in rounds)  # noqa: E731
                ratio = [r["rust"][key]["core_ms"] / r["python"][key]["core_ms"] for r in rounds]
                print(f"| {n} | {mode} | {med('python', 'core_ms'):.2f} | {med('rust', 'core_ms'):.2f} | {fmt_ratio(ratio)} | "
                      f"{med('python', 'graph_ms'):.2f} | {med('rust', 'graph_ms'):.2f} | "
                      f"{med('python', 'eval_cpu_ms'):.2f} | {med('rust', 'eval_cpu_ms'):.2f} | "
                      f"{med('python', 'total_ms'):.2f} | {med('rust', 'total_ms'):.2f} |")
            skey = f"{model}/{dtype}/stream"
            if all(skey in r["rust"] and skey in r["python"] for r in rounds):
                print("\nStream of single (N=1) requests, requests/s:\n")
                print("| runtime | sequential | pipelined (`async_eval`) | pipelining gain |")
                print("|---|---|---|---|")
                for rt in ("python", "rust"):
                    seq = statistics.median(r[rt][skey]["sequential_req_s"] for r in rounds)
                    pip = statistics.median(r[rt][skey]["pipelined_req_s"] for r in rounds)
                    print(f"| {rt} | {seq:.0f} | {pip:.0f} | {pip / seq:.2f}× |")
            print()


if __name__ == "__main__":
    main()
