# Benchmarks

`bench.py` times one batched generation, from the ONNX-style inputs to the decoded outputs, for
each backend on the *same* requests (`motionbricks_mlx.examples`). MLX runs as one `mx.compile`
graph. The torch and ONNX Runtime rows run NVIDIA's upstream PyTorch model and its exported
FP32 graph through the sandbox. Results: [`results/`](results/). `report.py` renders them.

```sh
uv run --all-packages python model/benchmarks/bench.py                       # default backends, N = 1 8 16 32
uv run --all-packages python model/benchmarks/bench.py --backends mlx-fp32 mlx-cpu --batches 1
uv run --all-packages python model/benchmarks/report.py model/benchmarks/results/<file>.json
```

## Results

Host: Mac mini (2024, `Mac16,11`), Apple M4 Pro with a 14-core CPU (10P + 4E) and 20-core GPU, 64 GB, macOS 27.0 (26A428); Python 3.12.11, MLX 0.32.4, PyTorch 2.15.0, ONNX Runtime 1.32.0. Median of timed runs after warm-up; full statistics in the JSON.

#### soma: latency per batch in ms (requests/s)

| backend | N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| **MLX FP32** (GPU) | 10.4 (97) | 16.4 (488) | 26.1 (612) | 49.7 (644) |
| **MLX FP16** (GPU) | 11.9 (84) | 18.7 (427) | 27.2 (589) | 46.4 (689) |
| **MLX BF16** (GPU) | 12.2 (82) | 18.8 (426) | 27.2 (589) | 45.0 (711) |
| MLX INT8 weights (GPU, experimental) | 12.4 (81) | 21.0 (381) | 32.6 (491) | 56.3 (569) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 32.8 (30) | 44.8 (179) | 60.3 (265) | 91.6 (349) |
| ONNX Runtime CPU (TensorRT shim backend) | 13.6 (74) | 58.2 (137) | 106.4 (150) | 206.5 (155) |
| upstream PyTorch, CPU | 33.4 (30) | 85.6 (93) | 523.5 (31) | 882.2 (36) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | 2.50 (400) | 2.96 (2,703) | 3.74 (4,278) | 5.28 (6,061) |
| *RTX 4090 + TensorRT FP16 (upstream docs, GPU time only)* | 1.28 (781) | 1.54 (5,195) | 1.87 (8,556) | 3.93 (8,142) |

#### unitree_g1: latency per batch in ms (requests/s)

| backend | N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| **MLX FP32** (GPU) | 11.2 (89) | 17.3 (463) | 26.9 (594) | 47.7 (671) |
| **MLX FP16** (GPU) | 12.8 (78) | 19.1 (418) | 27.6 (580) | 45.4 (705) |
| **MLX BF16** (GPU) | 12.8 (78) | 19.1 (418) | 27.6 (580) | 45.4 (705) |
| MLX INT8 weights (GPU, experimental) | 12.8 (78) | 21.0 (380) | 31.2 (513) | 53.9 (593) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 36.6 (27) | 49.0 (163) | 65.5 (244) | 98.3 (325) |
| ONNX Runtime CPU (TensorRT shim backend) | 13.9 (72) | 58.5 (137) | 107.9 (148) | 276.3 (116) |
| upstream PyTorch, CPU | 39.5 (25) | 87.1 (92) | 591.1 (27) | 966.3 (33) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | 2.67 (375) | 3.78 (2,116) | 5.17 (3,095) | 7.81 (4,097) |
| *RTX 4090 + TensorRT FP16 (upstream docs, GPU time only)* | 1.63 (613) | 1.91 (4,188) | 2.36 (6,780) | 3.38 (9,467) |

MLX's CPU backend (`--backends mlx-cpu`) is not competitive: about 160 ms at N=1 and 7 req/s,
after a 13–48 s first compile. For CPU-only use, ONNX Runtime is the better choice.

## Reading the numbers

- **MLX vs. the other Apple-silicon options.** MLX is the fastest at every batch size. Against
  the upstream PyTorch model on the GPU (MPS), it is 2.9–3.3× faster at N=1 and 1.9–2.1× faster
  at N=32. Against ONNX Runtime on the CPU (the TensorRT stand-in), it is 1.2–1.3× faster at
  N=1 and 4–6× faster at N=32. Against PyTorch on the CPU, it is up to 20× faster.
- **Batch 1 is dispatch-bound, not bandwidth-bound.** A generation is a chain of about 500
  dependent Metal kernels (24 transformer layers, two dilated-conv decoders, rig math). The CPU
  spends about 8 ms of the 10–11 ms encoding them. So at N=1:
  - FP16, BF16 and INT8 don't help; they are slightly slower because MLX uses a less
    bandwidth-efficient half-precision GEMM for 16-row matrices;
  - ONNX Runtime on the CPU, with no GPU dispatch at all, comes within 25 %.
- **Throughput comes from batching.** Going from N=1 to N=8 costs +60 % latency for 8× the work.
  At N=32, BF16 sustains about 700 generations/s. One generation drives a character for 0.6–1.9 s,
  so a single M4 Pro can keep **hundreds of characters** regenerating in real time.
- **INT8 (`model.quantize(8)`) is a memory option, not a speed option.** It cuts the transformer
  weights about 4× (pose network ~546 → ~140 MB). In this graph the quantized GEMMs are slower
  than FP32 at every batch size measured.
- **Against an RTX 4090 with TensorRT** (upstream figures, GPU time only, CUDA graph replay),
  MLX on the M4 Pro is about 4× slower at N=1 and about 9–10× at N=32. Those gaps track the
  hardware: memory bandwidth (≈1008 vs 273 GB/s, 3.7×) bounds small batches, and FP32 compute
  (≈82 vs ≈7 TFLOPS measured, ~11×) bounds large ones.

## Optimizations applied, with measured effect

| Change | Effect |
|---|---|
| `mx.compile` of the whole static-shape graph | single graph replay; compiled output equals eager to ≤ 6e-8 |
| `mx.split` instead of `qkv[i]` after a transpose (it lowered to 3 Gather copy kernels/layer × 24) | SOMA N=1 12.7 → 10.4 ms, N=32 519 → 644 req/s |
| Batched change of basis in G1 forward kinematics (90 tiny matmuls → 2) | G1 N=1 11.5 → 11.2 ms |
| Larger Metal command buffers (`MLX_MAX_OPS_PER_BUFFER`) | **slower** (17 ms): the GPU starts later, losing CPU/GPU overlap; not used |

Next candidates, all aimed at cutting kernel count:
- level-wise (tree-depth) G1 forward kinematics;
- fewer mask multiplies in the conv decoders;
- fusing the per-frame `where`/`concatenate` chains in the root network.
