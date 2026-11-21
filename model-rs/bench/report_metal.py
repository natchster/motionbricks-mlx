"""Tables for the Python MLX vs Rust (mlx-rs) vs Rust custom Metal runtime rounds.

    python3 model-rs/bench/report_metal.py model-rs/bench/results/raw_metal
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RAW = Path(sys.argv[1])
runs = defaultdict(dict)
loads = []
for path in sorted(RAW.glob("*.json")):
    dtype, rnd, rest = path.stem.split("_", 2)
    model, runtime = rest.rsplit("_", 1)
    runs[(int(rnd[1:]), model)][runtime] = json.loads(path.read_text())["results"]
    lf = path.with_suffix(".load")
    if lf.exists():
        loads.append(float(lf.read_text()))
rounds = sorted({k[0] for k in runs})
print(f"{len(rounds)} interleaved rounds; 1-min load average median {st.median(loads):.1f} (range {min(loads):.1f}-{max(loads):.1f}).\n")
for model in ("soma", "unitree_g1"):
    rs = [runs[(r, model)] for r in rounds if len(runs[(r, model)]) == 2]
    print(f"#### {model} FP32: median ms per batched call, end to end (inputs in, outputs on the host)\n")
    print("| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |")
    print("|---|---|---|---|---|---|---|")
    for n in (1, 2, 4, 8, 16, 32):
        kc = f"{model}/float32/N={n}/compiled"
        km = f"{model}/float32/N={n}/metal"
        if not all(kc in r["python"] and km in r["rust"] for r in rs):
            continue
        py = [r["python"][kc]["total_ms"] for r in rs]
        rm = [r["rust"][kc]["total_ms"] for r in rs]
        mt = [r["rust"][km]["total_ms"] for r in rs]
        gpu = [r["rust"][km]["gpu_ms"] for r in rs]
        sp = [a / b for a, b in zip(py, mt)]
        sr = [a / b for a, b in zip(rm, mt)]
        print(f"| {n} | {st.median(py):.2f} | {st.median(rm):.2f} | **{st.median(mt):.2f}** | {st.median(sp):.2f}x ({min(sp):.2f}-{max(sp):.2f}) | {st.median(sr):.2f}x ({min(sr):.2f}-{max(sr):.2f}) | {st.median(gpu):.2f} |")
    sk = f"{model}/float32/stream"
    if all(sk in r["python"] and sk in r["rust"] for r in rs):
        py_seq = st.median(r["python"][sk]["sequential_req_s"] for r in rs)
        py_pip = st.median(r["python"][sk]["pipelined_req_s"] for r in rs)
        rm_pip = st.median(r["rust"][sk]["pipelined_req_s"] for r in rs)
        mt_seq = st.median(r["rust"][sk]["metal_sequential_req_s"] for r in rs)
        print(f"\nStream of single requests (req/s): Python sequential {py_seq:.0f}, Python pipelined {py_pip:.0f}, "
              f"mlx-rs pipelined {rm_pip:.0f}, **Metal runtime sequential {mt_seq:.0f}**\n")
