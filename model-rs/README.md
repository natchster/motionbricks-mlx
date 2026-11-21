# model-rs — MotionBricks in Rust

Two Rust implementations of [`model/`](../model) (`motionbricks_mlx`, the MLX port of NVIDIA's
MotionBricks). Both run the two released models, SOMA and Unitree G1, and both pass the same
golden parity tests against NVIDIA's PyTorch model as the Python port.

| Runtime | What it is | Speed vs the Python MLX port (Mac mini M4 Pro, FP32) |
|---|---|---|
| `motionbricks_rs::MotionBricks` | an op-for-op port on [mlx-rs](https://github.com/oxiglade/mlx-rs) 0.34 (MLX 0.32.2): the same graph, the same kernels, and the shared rig kernels (`../model/motionbricks_mlx/assets/rig_kernels.metal`) | the same (0.98–1.01×) |
| **`motionbricks_rs::metal::model::MetalModel`** | **a model-specific Metal runtime** built from what MLX does generically: a static dispatch plan, custom kernels, and MLX's own GEMM kernels where they win | **1.49–1.50× at N=1**, 1.20× at N=2, 1.11× at N=4, 1.06× at N=8; parity at N=16; 0.97–0.98× at N=32 |

The repository README explains when to use which. The Metal runtime is tied to this model
implementation and to Apple-silicon GPUs, in exchange for the lowest single-request latency.

## Build, test, run

```sh
cd model-rs
cargo build --release                      # first build compiles MLX (~6 min); needs Xcode + its Metal toolchain
cargo test --release -- --test-threads=1   # both runtimes vs PyTorch + rig kernels vs ops (needs checkpoints/ + sandbox/golden/)
cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 bench/requests/unitree_g1_N8.npz
cargo run --release --example generate       -- soma ../checkpoints/MotionBricks/soma bench/requests/soma_N8.npz
OUT=$PWD/bench/results/raw_fp32 bench/run_all.sh 4 float32 && python3 bench/report_metal.py bench/results/raw_fp32
```

```rust
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 64)?; // max batch 64
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?; // ONNX input names, NumPy-style arrays
```

`.cargo/config.toml` sets `DEVELOPER_DIR` to Xcode and keeps mlx-sys's `mlx.metallib` (175 MB) in
`target/mlx-metal/`. The Metal runtime loads that same metallib at run time for MLX's steel GEMM
kernels.

## 1. How MLX and mlx-c execute one generation

Both ports pay for the same machinery. Python or Rust only builds the graph; MLX's C++ core
executes it.

- **mlx-c** is a thin C ABI over MLX's C++ API. Every op call allocates a graph node and a stream
  handle and returns an opaque `mlx_array`; mlx-rs wraps each one in a `Result`.
- **Evaluation** (`backend/metal/eval.cpp`, `device.cpp`):
  - the scheduler walks the graph in topological order and calls each primitive's `eval_gpu`;
  - that encodes one dispatch into a `MTLDispatchTypeConcurrent` compute encoder;
  - MLX tracks the input and output buffers of every op and inserts a memory barrier when an op
    reads a buffer an earlier op wrote;
  - it commits a command buffer every 40 ops on this GPU class (`applegpu_g16s`);
  - buffers come from a caching allocator.
- **Kernels** come from the precompiled `mlx.metallib`: steel GEMM, conv, SDPA, normalization,
  reductions, gather and copy. With `mx.compile`, chains of elementwise ops are fused into
  JIT-generated Metal kernels.
- **GEMM selection** (`matmul.cpp`):
  - `gemv` / `gemv_wide` for M ≤ ~15 rows;
  - split-K steel when the output is small and K is large: `(M/16)(N/16) ≤ 2048` and `K ≥ max(M, N)`;
  - otherwise the regular steel kernel, which on this GPU uses 64×64 tiles.

  So at batch 1 the 16-row matrices are padded into 64-row tiles.

What this costs MotionBricks, as GPU time per stage (MLX, pipelined calls, ms). The profile was
measured before the Python port adopted the rig kernels this analysis motivated:

| | SOMA N=1 | G1 N=1 | SOMA N=32 | G1 N=32 |
|---|---|---|---|---|
| pose transformer (16 × 1024) | 4.2 | 4.2 | 24.3 | 24.5 |
| root network | 2.4 | 1.9 | 9.6 | 9.1 |
| VQ-VAE decoder | 1.5 | 1.5 | 5.1 | 5.3 |
| rig and glue | ~0 | **2.4** | 2.4 | **7.1** |

- **Pose transformer at N=1:** it streams 512 MB of weights at about 120 GB/s, 44% of the
  M4 Pro's 273 GB/s.
- **G1 rig:** the 30-joint forward kinematics and 34-joint decoding become hundreds of tiny
  batched 3×3 matmuls and gathers, each padded into a GEMM tile.
- **CPU:** MLX spends about 2 ms per call encoding about 500 dispatches. A Metal System Trace shows
  16 command buffers per call and the GPU busy 99% of each call at N=1.

## 2. The test: is a model-specific runtime faster?

The prototype covered the largest sub-network, the 16-layer pose transformer with the real SOMA
weights. It used:
- one serial command buffer and preallocated buffers;
- 7 dispatches per layer: QKV GEMM, attention reading the packed QKV directly, output GEMM,
  add+LayerNorm, FFN GEMM+ReLU, FFN GEMM, add+LayerNorm.

Each step was checked against mlx-rs (relative error ≤ 1.3e-6):

| Pose transformer, Metal / MLX time | N=1 | N=2 | N=4 | N=32 |
|---|---|---|---|---|
| v1: naive skinny + tiled GEMMs | 0.92× slower | 0.38× | 0.16× | 0.18× |
| v2: simdgroup-MMA GEMM, 16-row tiles, split-K, unrolled fragments | **1.62× faster** | 1.31× | 0.82× | 0.79× |
| v3: + MLX's split-K/steel kernels for M > 32, shared-memory attention | **1.64×** | 1.24× | 1.03× | 0.91–0.96× |

GEMM by GEMM, on identical shapes run as dependent chains:

| M (rows) | MLX steel | own `gemm_mma` |
|---|---|---|
| 16 (batch 1) | 104.7 µs | **41–44 µs** (2.4×) |
| 64 | 104.6 µs | 96.6 µs |
| 128 | **149.7 µs** | 175 µs |
| 512 (batch 32) | **524.7 µs** (6.1 TFLOPS) | 605 µs |

The prototype cleared the >5% bar at batch 1 by a wide margin, so the full runtime was built.
It combines both kernel families, mine where M ≤ 32 and MLX's where M is large, and it adds
everything MLX cannot do for a model it doesn't know.

## 3. The Metal runtime (`src/metal/`)

- **A static plan:** one generation is a fixed list of 370–391 dispatches (N=1) in a single
  serial command buffer. The CPU encodes it in **0.10–0.14 ms**, versus MLX's ~2 ms. There is no
  graph, allocator or hazard tracking.
- **`gemm_mma`** (`kernels.metal`): one simdgroup-MMA template with tiles from 16×32 to 64×64,
  split-K with a reduce, and a generalized operand loader and epilogue. Most of MLX's ~500 kernels
  fold into it:
  - **A side:** implicit dilated Conv1d im2col with 2× nearest upsampling, concatenation of up to
    three sources, ReLU and row masks;
  - **output side:** ReLU / LeakyReLU, residual with a row modulus (position tables), output mask,
    predicated row writes (the decoder's "overwrite the target frames") and the root network's
    "missing-embedding blend".
- **MLX's steel kernels**, loaded from `mlx.metallib` and dispatched exactly as MLX does
  (`GEMMParams` / `GEMMSpiltKParams` mirrored, the same split-K rule), handle large plain GEMMs.
  Large convolutions and other non-plain GEMMs are materialized into a scratch matrix and then run
  on steel.
- **Model kernels** (`model.metal`), one thread per item:
  - G1 forward kinematics and feature extraction (one thread per sequence, all 30 joints);
  - joint-angle decoding with `matrix_quat`;
  - the SOMA T-pose retarget;
  - normalization, the duration head and argmax, dense conditions, `global_root_to_local` with
    the last-frame patch;
  - pose argmax with the codebook lookup, root integration with SOMA's height-violation rule, and
    re-anchoring.
- **Constant folding:** in the release configuration (one MaskGIT step, every input token
  masked), the pose network's token-embedding MLP has a constant input. Its output is computed
  once at load time.

GPU time per stage after the rewrite (ms):

| | SOMA N=1 | G1 N=1 | SOMA N=32 | G1 N=32 |
|---|---|---|---|---|
| rig, input and glue | 0.16 | 0.26 | 0.17 | 0.35 |
| root network | 2.26 | 1.88 | 10.65 | 10.37 |
| pose network | 3.41 | 3.42 | 26.65 | 27.53 |
| VQ-VAE decoder | 0.94 | 0.91 | 5.42 | 5.69 |
| **whole generation** | **6.76** | **6.50** | 44.0 | 44.7 |

## 4. Correctness

`tests/metal_parity.rs` and `tests/parity.rs` cover both runtimes. They use the 5 golden request
families per model and the Python suite's tolerances. Every sub-network is teacher-forced with
PyTorch's exact inputs, then the full pipeline runs end to end.

| Metal runtime vs PyTorch FP32 | SOMA | G1 |
|---|---|---|
| root duration logits / root path (relative) | ≤ 2.1e-6 / ≤ 3.6e-6 | ≤ 2.5e-6 / ≤ 2.7e-6 |
| pose logits (relative), argmax tokens | ≤ 8.8e-6, identical | ≤ 5.4e-6, identical |
| VQ-VAE decoder (relative) | ≤ 3.7e-6 | ≤ 1.2e-5 |
| durations, end to end | identical, 5/5 | identical, 5/5 |
| outputs, end to end (absolute) | root ≤ 3.2e-6, features ≤ 1.2e-4, rotations ≤ 7.9e-4 | `pred_qpos` ≤ 1.5e-4, contacts ≤ 1.7e-3 |

## 5. Benchmarks

Machine: Mac mini (2024, `Mac16,11`), Apple M4 Pro (14-core CPU, 20-core GPU), 64 GB, macOS 27.0.
4 interleaved rounds were run on 2026-11-21 on a quiet machine (load median 1.9). Times are the
median per batched call, end to end: inputs from the host, outputs back on the host. Ratios are
paired within rounds; the range is in parentheses. The Python port includes this release's
optimizations: parameters evaluated before compiling, and rig kernels.

#### SOMA, FP32

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal vs Python | Metal GPU time |
|---|---|---|---|---|---|
| 1 | 10.70 ms | 10.62 ms | **7.14 ms** | **1.50×** (1.50–1.53) | 6.77 ms |
| 2 | 11.08 | 11.01 | **9.23** | 1.20× (1.19–1.20) | 8.85 |
| 4 | 12.01 | 11.95 | **10.79** | 1.11× (1.11–1.12) | 10.40 |
| 8 | 16.00 | 15.93 | **15.06** | 1.06× (1.06–1.07) | 14.65 |
| 16 | 25.09 | 25.00 | 25.06 | 1.00× (0.99–1.01) | 24.62 |
| 32 | 45.48 | 45.57 | 46.66 | 0.97× (0.97–0.98) | 46.18 |

#### Unitree G1, FP32

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal vs Python | Metal GPU time |
|---|---|---|---|---|---|
| 1 | 10.55 ms | 10.50 ms | **7.09 ms** | **1.49×** (1.48–1.51) | 6.72 ms |
| 2 | 10.90 | 10.86 | **9.10** | 1.20× (1.18–1.20) | 8.73 |
| 4 | 11.87 | 11.82 | **10.72** | 1.11× (1.10–1.12) | 10.33 |
| 8 | 15.83 | 15.74 | **14.93** | 1.06× (1.06–1.06) | 14.53 |
| 16 | 24.80 | 24.84 | 24.88 | 1.00× (0.99–1.02) | 24.48 |
| 32 | 44.85 | 45.00 | 46.01 | 0.98× (0.97–0.98) | 45.60 |

Streams of single requests: the Metal runtime sustains 140/141 req/s sequentially (SOMA/G1).
Python and mlx-rs reach 94/95 sequentially and 150/154 with `async_eval` pipelining.

Before this release's Python optimizations, the Metal runtime was also 4–16% faster at N=8–32.
That margin came mostly from its one-thread rig kernels. Once the same kernels went into the
Python port (and into the mlx-rs port), large batches reached parity: the transformer GEMMs
dominate there, and both runtimes use MLX's steel kernels for them.

mlx-rs vs Python, FP32 and BF16 (`bench/results/report_fp32_mlx-rs.md`,
`report_bf16.md`): compiled 0.98–1.01×; eager 0.97–0.99×, because Rust builds the eager graph in
0.31 ms against Python's 0.55 ms. The language does not matter once the graph is compiled. The
Metal-runtime rounds are in `bench/results/report_fp32.md`.

## 6. Limits

- **FP32 only.** BF16/FP16 kernels are not written yet. In MLX, BF16 already cuts N=1 to about
  8.2 ms, so a BF16 Metal runtime should be faster again.
- **Release configuration only.** It runs one MaskGIT step with model-chosen durations; forced
  durations are wired to the root kernels but only used by the stage tests. Multi-step decoding
  and other options remain available in the mlx-rs port.
- **Batch size** is capped by `max_batch` at construction (buffers are preallocated).
- **Version coupling.** The steel kernels and their parameter layouts are those of MLX 0.32.2 as
  built by mlx-sys 0.7.0, so an MLX upgrade needs the mirrored structs re-checked.
- **Large batches:** at N ≥ 16 the optimized Python port is equal or up to 3% faster, and Python
  BF16 is the throughput choice (848 req/s at N=32). The gains are for single requests and small
  batches, which are MotionBricks' real-time use. Sequential request streams are not yet
  double-buffered: 140 req/s, against pipelined MLX's 150.

## Layout

| Path | Contents |
|---|---|
| `src/metal/mod.rs` | device, kernel library (own source + `mlx.metallib`), buffers, serial recorder |
| `src/metal/kernels.metal` | `gemm_mma`, split-K reduce, materialize/epilogue, attention, add+LayerNorm |
| `src/metal/model.metal` | rig and glue kernels |
| `src/metal/transformer.rs` | GEMM planning (own kernels / MLX steel / split-K / materialized), transformer stack |
| `src/metal/model.rs` | weights, buffers, and the per-stage encoders of a generation |
| `src/*.rs` | the mlx-rs port (`config`, `weights`, `layers`, `networks`, `representation`, `inference`, `rig`, `model`); `kernels.rs` runs the shared rig kernels |
| `tests/` | `parity.rs` (mlx-rs port, rig kernels vs ops), `metal_parity.rs` (Metal runtime) |
| `src/bin/` | `mbrs-bench` (benchmark), `proto-transformer`, `gemm-sweep`, `metal-stages`, `metal-debug` (development tools) |
| `bench/` | interleaved Python-vs-Rust harness, reports and raw results |

## Sources

- [MLX](https://github.com/ml-explore/mlx) source (v0.32.2, read from the mlx-sys build):
  `backend/metal/{eval,device,matmul}.cpp`, `kernels/steel/gemm/*`
- [mlx-rs](https://github.com/oxiglade/mlx-rs) ([docs.rs](https://docs.rs/mlx-rs)), [objc2-metal](https://docs.rs/objc2-metal)
- [Apple: indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md),
  [MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer.md)
- [Kernel fusion in MLX: what `mx.compile` and `mx.fast.*` do](https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html)

License: Apache-2.0, like the repository. The rig assets embedded with `include_bytes!` are
NVIDIA's (Apache-2.0); MLX's kernels are loaded at run time from the MLX build (MIT); mlx-rs is
MIT or Apache-2.0.
