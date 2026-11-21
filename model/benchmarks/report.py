"""Render a bench.py JSON into Markdown tables (latency, throughput, vs. RTX 4090 TensorRT).

    uv run python model/benchmarks/report.py model/benchmarks/results/<file>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = {
    "mlx-fp32": "**MLX FP32** (GPU)", "mlx-fp16": "**MLX FP16** (GPU)", "mlx-bf16": "**MLX BF16** (GPU)",
    "mlx-int8": "MLX INT8 weights (GPU, experimental)", "mlx-cpu": "MLX FP32 (CPU)",
    "torch-mps": "upstream PyTorch, `.cuda()`→MPS via cudashim", "torch-cpu": "upstream PyTorch, CPU",
    "ort-cpu": "ONNX Runtime CPU (TensorRT shim backend)", "ort-coreml": "ONNX Runtime CoreML EP",
}


def main(path: str) -> None:
    data = json.loads(Path(path).read_text())
    host, rows = data["host"], data["results"]
    batches = sorted({r["batch"] for r in rows})
    print(f"Host: {host['chip']} ({host['gpu_cores']}-core GPU), {host['memory_gb']} GB, macOS {host['macos']}, "
          f"MLX {host['mlx']}. Median of timed runs after warm-up; full statistics in the JSON.\n")
    for model in ("soma", "unitree_g1"):
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        backends = list(dict.fromkeys(r["backend"] for r in sub))
        cell = {(r["backend"], r["batch"]): r for r in sub}
        print(f"#### {model}: latency per batch in ms (requests/s)\n")
        print("| backend | " + " | ".join(f"N={b}" for b in batches) + " |")
        print("|---|" + "---|" * len(batches))
        for backend in backends:
            values = []
            for b in batches:
                r = cell.get((backend, b))
                values.append(f"{r['median_ms']:.1f} ({r['requests_per_s']:,.0f})" if r else "—")
            print(f"| {LABELS.get(backend, backend)} | " + " | ".join(values) + " |")
        ref = data.get("reference_rtx4090_tensorrt_ms", {})
        for precision in ("fp32", "fp16"):
            values = ref.get(f"{model}/{precision}")
            if values:
                cells = [f"{values[str(b)]:.2f} ({b / values[str(b)] * 1e3:,.0f})" if str(b) in values else "—" for b in batches]
                print(f"| *RTX 4090 + TensorRT {precision.upper()} (upstream docs, GPU time only)* | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
