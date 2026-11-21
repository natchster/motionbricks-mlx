# Benchmarks

`bench.py` times one batched generation, from the ONNX-style inputs to the decoded outputs, for
each backend on the *same* requests (`motionbricks_mlx.examples`). MLX runs as one `mx.compile`
graph. The torch and ONNX Runtime rows run NVIDIA's upstream PyTorch model and its exported
FP32 graph through the sandbox. Results: [`results/`](results/). `report.py` renders them.

```sh
uv run --all-packages python model/benchmarks/bench.py                       # default backends, N = 1 8 16 32
uv run --all-packages python model/benchmarks/bench.py --backends mlx-fp32 mlx-int8 --batches 1
uv run --all-packages python model/benchmarks/bench.py --backends mlx-fp16 mlx-bf16 --rounds 5   # interleaved
uv run --all-packages python model/benchmarks/report.py model/benchmarks/results/<file>.json
```

By default each backend is timed in one block, after the previous one. `--rounds N` interleaves
them instead (every backend in turn, N rounds per batch size, statistics pooled), so thermal drift
of up to about 10% spreads evenly when comparing close numbers.

The torch and ONNX Runtime backends need the sandbox (`sandbox/setup.sh`). The MLX port alone can
be timed without it: `uv run --package motionbricks-mlx motionbricks-mlx bench --model soma
--checkpoints checkpoints/MotionBricks/soma --batch 1 8 32`.

The Rust ports have their own interleaved Python-vs-Rust harness in
[`model-rs/bench`](../../model-rs/bench). The Rust Metal runtime's single-request latency is
4.0 ms in BF16 and 5.2 ms in FP32 (see the
[user guide](../../docs/user_guide.md#5-benchmarks)).

## Results

Host: Mac mini (2024, `Mac16,11`), Apple M4 Pro with a 14-core CPU (10P + 4E) and 20-core GPU,
64 GB, macOS 27.0 (26A428); Python 3.12.11, MLX 0.32.4, PyTorch 2.15.0, ONNX Runtime 1.32.0. Run
on 2026-11-21 on a quiet machine. Each cell is the median of 40 timed runs after warm-up (10 for
PyTorch on the CPU); full statistics are in the JSON.

#### soma: latency per batch in ms (requests/s)

| backend | N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| **MLX FP32** (GPU) | 10.5 (95) | 15.9 (504) | 24.7 (648) | 42.7 (750) |
| **MLX FP16** (GPU) | 8.1 (123) | 13.8 (579) | 21.6 (740) | 37.7 (848) |
| **MLX BF16** (GPU) | 8.1 (123) | 13.8 (578) | 21.7 (738) | 37.8 (848) |
| MLX INT8 weights (GPU, experimental) | 7.9 (126) | 15.2 (525) | 24.7 (647) | 43.9 (730) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 32.1 (31) | 44.0 (182) | 59.5 (269) | 91.2 (351) |
| ONNX Runtime CPU (TensorRT shim backend) | 13.5 (74) | 58.2 (137) | 110.4 (145) | 210.6 (152) |
| upstream PyTorch, CPU | 33.4 (30) | 86.0 (93) | 526.3 (30) | 867.4 (37) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | 2.50 (400) | 2.96 (2,703) | 3.74 (4,278) | 5.28 (6,061) |
| *RTX 4090 + TensorRT FP16 (upstream docs, GPU time only)* | 1.28 (781) | 1.54 (5,195) | 1.87 (8,556) | 3.93 (8,142) |

#### unitree_g1: latency per batch in ms (requests/s)

| backend | N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| **MLX FP32** (GPU) | 10.5 (95) | 15.8 (506) | 24.7 (647) | 42.7 (749) |
| **MLX FP16** (GPU) | 8.3 (121) | 13.8 (581) | 21.7 (739) | 37.8 (848) |
| **MLX BF16** (GPU) | 8.3 (121) | 13.8 (581) | 21.6 (739) | 37.8 (848) |
| MLX INT8 weights (GPU, experimental) | 8.0 (126) | 15.3 (523) | 25.0 (639) | 46.2 (693) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 37.0 (27) | 49.6 (161) | 65.8 (243) | 99.0 (323) |
| ONNX Runtime CPU (TensorRT shim backend) | 14.1 (71) | 59.9 (134) | 110.3 (145) | 213.6 (150) |
| upstream PyTorch, CPU | 33.3 (30) | 83.8 (96) | 533.2 (30) | 913.9 (35) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | 2.67 (375) | 3.78 (2,116) | 5.17 (3,095) | 7.81 (4,097) |
| *RTX 4090 + TensorRT FP16 (upstream docs, GPU time only)* | 1.63 (613) | 1.91 (4,188) | 2.36 (6,780) | 3.38 (9,467) |

MLX's CPU backend (`--backends mlx-cpu`) is not competitive: about 160 ms at N=1 and 7 req/s,
after a 13–48 s first compile. For CPU-only use, ONNX Runtime is the better choice.

## Reading the numbers

- **Batch 1 is GPU-bound, not CPU-bound.** A Metal System Trace of a compiled N=1 call shows 16
  command buffers and the GPU busy 99% of the time; MLX's CPU encoding (~2 ms) overlaps with it.
  About 3 ms of GPU time is streaming the ~800 MB of FP32 weights at the M4 Pro's 273 GB/s. The
  rest is the latency of about 500 small, dependent kernels.
- **So smaller weights help at N=1.** BF16/FP16 take 8.1–8.3 ms and INT8 7.9–8.0 ms, against
  10.5 ms for FP32. An earlier version of this page reported the opposite. That was a bug:
  lazily cast or quantized weights were captured by `mx.compile` and recomputed on every call.
  It is fixed in `MotionBricks.set_compute_dtype` / `quantize`.
- **Throughput comes from batching.** Going from N=1 to N=8 costs +50% latency for 8× the work.
  At N=32, BF16 sustains 848 generations/s. One generation drives a character for 0.6–1.9 s, so
  a single M4 Pro can keep **hundreds of characters** regenerating in real time. INT8 is slower
  than BF16 at large batch because MLX's quantized GEMM is compute-bound there.
- **MLX vs the other Apple-silicon options.** MLX is the fastest at every batch size:
  - against the upstream PyTorch model on MPS: 3–4.5× faster at N=1 and 2.4× at N=32 (BF16);
  - against ONNX Runtime on the CPU (the TensorRT stand-in): 4.9–5.7× the throughput at N=32.
- **Against an RTX 4090 with TensorRT** (upstream figures, GPU time only, CUDA graph replay),
  MLX on the M4 Pro is about 4× slower at N=1 and 5.5–8× at N=32, comparing FP32 with FP32
  (5–11× comparing BF16 with TensorRT FP16). Those gaps track the hardware: memory bandwidth
  (≈1008 vs 273 GB/s, 3.7×) bounds small batches, and compute bounds large ones.

## Optimizations applied, with measured effect

| Change | Effect |
|---|---|
| `mx.compile` of the whole static-shape graph | single graph replay; compiled output equals eager to ≤ 6e-8 |
| `mx.split` instead of `qkv[i]` after a transpose (it lowered to 3 Gather copy kernels/layer × 24) | SOMA N=1 12.7 → 10.4 ms, N=32 519 → 644 req/s |
| Evaluate cast / quantized parameters before the first compiled call | BF16 N=1 12.2 → 8.1 ms; INT8 N=1 12.4 → 7.9 ms |
| Rig geometry as custom Metal kernels (`kernels.py`), one thread per pose | G1 N=1 11.3 → 10.5 ms, G1 N=32 48.0 → 42.7 ms, SOMA N=32 46.0 → 42.7 ms |
| Larger Metal command buffers (`MLX_MAX_OPS_PER_BUFFER`) | **slower** (17 ms): the GPU starts later, losing CPU/GPU overlap; not used |
| Row-contiguous Conv1d weights | no measurable effect; not used |

Beyond MLX itself, the Rust Metal runtime in [`model-rs`](../../model-rs/README.md) replaces the
generic graph executor with a model-specific dispatch plan and kernels: about 2× faster at N=1
and 1.04–1.09× at N=32, at every precision.
