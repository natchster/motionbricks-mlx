# motionbricks-mlx

NVIDIA's [MotionBricks](https://github.com/NVlabs/motionbricks) (real-time motion generation for
characters and robots, SIGGRAPH 2026) on Apple silicon. It comes as a Python MLX port and as a
Rust implementation with a model-specific Metal runtime.

> [!NOTE]
> **This is a quick port and may contain issues.** It passes its full test suites, which check
> both versions stage by stage against NVIDIA's own PyTorch implementation, and it has also been
> tested by a human. If you find a discrepancy with upstream, please open an issue. This is an
> independent community project, not affiliated with or endorsed by NVIDIA.

| Folder | What it is |
|---|---|
| [`model/`](model/README.md) | **`motionbricks_mlx`** (Python), a native MLX port of the complete SOMA and Unitree G1 inference graphs. It has the same tensor interface as the upstream ONNX models, supports FP32, FP16, BF16 and INT8, and depends only on `mlx` + `numpy`. |
| [`model-rs/`](model-rs/README.md) | **`motionbricks-rs`** (Rust). It contains an op-for-op port on mlx-rs and a **model-specific Metal runtime**, the fastest way to run a single request. |
| [`sandbox/`](sandbox/README.md) | **`cudashim`**, a layered CUDA-emulation sandbox for M-series Macs. It maps `torch.cuda`→MPS, TensorRT→ONNX Runtime and Warp kernels→CPU, and optionally compiles CUDA C++→Metal through CuMetal. It also runs NVIDIA's upstream model as the reference oracle for both versions. |

### Ported and tested version

| | Version |
|---|---|
| Code | [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) **v0.2.0**, commit [`54fc563aeaaf819a753d72cc9fdc6713989cdd43`](https://github.com/NVlabs/motionbricks/tree/54fc563aeaaf819a753d72cc9fdc6713989cdd43) |
| Weights | [nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks) on Hugging Face, revision [`3cb362d9b95aa40462abcb4d8cfae9ffb6d97410`](https://huggingface.co/nvidia/MotionBricks/tree/3cb362d9b95aa40462abcb4d8cfae9ffb6d97410) |
| SOMA checkpoints | MotionBricks v6 release `soma-v6-nova-4n8g-ea43dd8-fp32-v1`: pose, root and VQ-VAE networks at training step 1,500,000 |
| Unitree G1 checkpoints | release `out/motionbricks_{pose,root,vqvae}/version_1`: pose, root and VQ-VAE networks at training step 2,000,000 |
| Reference ONNX graphs | `mmm_v6_soma_fp32.onnx` and `mmm_v6_g1_fp32.onnx`, exported from the code and weights above with upstream's own exporter |

`sandbox/scripts/fetch_assets.py` pins both the commit and the revision. It also verifies every
checkpoint file against the SHA-256 list in upstream's `tools/onnx/checksums.py`.

## Benchmarks

All numbers come from a **Mac mini (2024, model `Mac16,11`) with an Apple M4 Pro**: 14-core CPU
(10 performance + 4 efficiency cores), 20-core GPU and 64 GB of unified memory. The software was
macOS 27.0 (build 26A428), Python 3.12.11 with MLX 0.32.4, Rust 1.100 with mlx-rs 0.34 (MLX
0.32.2), PyTorch 2.15.0 and ONNX Runtime 1.32.0. All runs used the quiet machine (load average
1–4) on 2026-11-21.

The table shows the median time per batched generation, with requests/s in parentheses:

| Backend | SOMA N=1 | SOMA N=32 | G1 N=1 | G1 N=32 |
|---|---|---|---|---|
| **Rust Metal runtime, FP32** (`model-rs`) | **7.14 ms** (140) | 46.58 ms (687) | **7.08 ms** (141) | 45.99 ms (696) |
| Python MLX, FP32 | 10.67 ms (94) | 45.42 ms (705) | 10.50 ms (95) | 44.80 ms (714) |
| Python MLX, BF16 | 8.12 ms (123) | **37.75 ms (848)** | 8.28 ms (121) | **37.76 ms (848)** |
| Python MLX, INT8 weights | 7.93 ms (126) | 43.85 ms (730) | 7.96 ms (126) | 46.20 ms (693) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 32.08 ms (31) | 91.19 ms (351) | 37.04 ms (27) | 98.97 ms (323) |
| upstream ONNX on ONNX Runtime CPU (TensorRT shim) | 13.54 ms (74) | 210.6 ms (152) | 14.06 ms (71) | 213.6 ms (150) |
| upstream PyTorch, CPU | 33.41 ms (30) | 867 ms (37) | 33.29 ms (30) | 914 ms (35) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | *2.50 ms* | *5.28 ms* | *2.67 ms* | *7.81 ms* |

Where the numbers come from:
- **Rust Metal runtime and Python FP32 rows:** the same interleaved rounds, so the comparison
  between them is paired; [Part 2](#part-2--the-rust-optimized-model-model-rs) has every batch size.
- **Other rows:** the Python harness `model/benchmarks/bench.py`
  ([full tables](model/benchmarks/README.md)).

What the table shows:
- **Lowest latency:** the Rust Metal runtime takes **7.1 ms for a single request**. That is
  **1.5× faster** than Python MLX FP32, and faster than Python BF16 or INT8. Its advantage shrinks
  with batch size, reaching parity at N=16.
- **Highest throughput:** Python MLX **BF16** delivers 848 req/s at N=32.
- **Versus other Mac options:** MLX is 2.4–4.5× faster than NVIDIA's PyTorch model on the GPU
  (MPS). At batch 32 it delivers 4.6–5.6× the throughput of ONNX Runtime on the CPU.

---

## Part 1 — Using the Python model (`motionbricks_mlx`)

### Setup

```sh
sandbox/setup.sh                                   # uv env, pinned upstream, verified weights, ONNX, golden fixtures
uv run motionbricks-mlx generate --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1 --batch 4
```

Only the weights are required for inference: `uv sync`, then `hf download nvidia/MotionBricks
--local-dir checkpoints/MotionBricks --revision 3cb362d9b95aa40462abcb4d8cfae9ffb6d97410`. The
rest of `setup.sh` is for the tests and benchmarks (Part 3).

```python
import motionbricks_mlx as mb
from motionbricks_mlx import examples

model = mb.load("soma", "checkpoints/MotionBricks/soma", dtype="bfloat16")  # float32 | float16 | bfloat16
out = model.run(examples.request("soma", batch=8))       # dict of NumPy in (ONNX input names) -> dict of NumPy out
frames = [out["pred_global_motions"][i, : 4 * n] for i, n in enumerate(out["pred_num_tokens"])]

mb.convert("unitree_g1", "checkpoints/MotionBricks/g1", "weights/g1.safetensors")   # one-file MLX weights
g1 = mb.load("weights/g1.safetensors").quantize(bits=8)  # INT8 transformer weights
```

The CLI also has `convert` and `bench` subcommands. See [model/README.md](model/README.md) for
the architecture and design notes.

### Choosing precision and batching

| Goal | Use | Why |
|---|---|---|
| exact reference behaviour | FP32 (default) | matches PyTorch FP32 to ≤ 1.2e-5 per sub-network |
| single requests, lowest latency in Python | **INT8** (7.9 ms) or **BF16** (8.1 ms) | batch 1 is bound by streaming the ~800 MB of FP32 weights; smaller weights help directly |
| many characters at once | **BF16 with batching** | N=32 costs about 38 ms, which is 848 generations/s |
| a stream of single requests | `mx.async_eval` pipelining | encode request i+1 while the GPU runs request i: 94 → 150 req/s in FP32 |

INT8 and BF16 chose the same durations as FP32 on every test request; check agreement for your
own inputs. Two Python optimizations were applied for this release:
- **Weights evaluated up front:** the cast or quantized weights are evaluated before compiling, so
  each compiled call no longer recomputes about 200 M parameter casts. BF16 went from 12.2 to
  8.1 ms and INT8 from 12.4 to 7.9 ms at N=1.
- **Rig as custom Metal kernels:** the rig math runs in `mx.fast.metal_kernel` kernels, shared
  with the Rust port in `assets/rig_kernels.metal`. That makes G1 7–11% faster and SOMA 7% faster
  at N=32.

### Correctness

Every sub-network is teacher-forced with PyTorch's exact inputs from golden fixtures:
- **Sub-networks:** root ≤ 4e-6, pose logits ≤ 8e-6 with identical argmax tokens, VQ-VAE
  ≤ 1.2e-5 relative.
- **End to end:** durations are identical on all 5 request families per model.

The details are in [model/README.md](model/README.md#verification).

---

## Part 2 — The Rust optimized model (`model-rs`)

`model-rs` holds two Rust implementations that share the weight loader and the parity tests:

| | `motionbricks_rs::MotionBricks` (mlx-rs port) | **`motionbricks_rs::metal::model::MetalModel`** (Metal runtime) |
|---|---|---|
| what it is | the Python graph op for op on mlx-rs: same MLX kernels | a fixed dispatch plan written for these two networks: own Metal kernels plus MLX's steel GEMMs |
| speed | same as Python (0.98–1.00× compiled) | **1.5× faster at N=1**, faster up to N=8, parity at N=16, 2–3% slower at N=32 |
| precisions | FP32, FP16, BF16 | FP32 only |
| configurations | everything the Python port supports | the release configuration only |

### Best speed, at the price of generality

The Metal runtime is fast *because* it isn't general:
- **Static plan:** a generation is one precomputed list of about 380 dispatches in a single command
  buffer. The CPU encodes it in 0.1 ms, where MLX builds and schedules a graph in about 2 ms.
- **Skinny GEMMs:** 16-row GEMM tiles and split-K are sized for single-request shapes, and they
  stream the weights at close to the memory bandwidth.
- **Fused work:** convolutions, masks, concatenations and blends are folded into the matrix
  multiplies, and the rig runs as one thread per pose.
- **Precomputed constants:** the pose network's token embedding is computed once at load time,
  because the release configuration makes it constant.

What that ties it to:
- **This exact model implementation.** The kernels and the dispatch plan encode the SOMA/G1
  shapes, the post-norm transformer, the `DoubleCondDecoder` layout and the rig. It only supports
  one MaskGIT step with model-chosen durations, in FP32, with a batch size capped when the model
  is built. Any architecture or configuration change means editing kernels and plans, not Python.
- **The device.** It runs on Apple-silicon Metal GPUs only and needs Xcode with its Metal
  toolchain to build. Tile shapes and the MLX steel variant were chosen and measured on an M4 Pro
  (`applegpu_g16s`). Other M-series chips will run it, but the speedup isn't measured there.
- **The MLX build.** It loads MLX's precompiled steel GEMM kernels from the `mlx.metallib` of the
  mlx-sys build and mirrors their parameter structs. That couples it to MLX 0.32.2 / mlx-sys 0.7.0.

So the Python port (or the mlx-rs port) is the flexible choice: any precision, batching for
throughput, easy changes. The Metal runtime is the choice for **latency-critical, single-request
inference on known Apple hardware**, such as a robot control loop or a game character, where
1.5× matters.

FP32, median time per call end to end (inputs in, outputs on the host), from 4 interleaved
rounds; the range of the paired ratio across rounds is in parentheses:

| N | Python MLX | Rust mlx-rs | **Rust Metal runtime** | Metal vs Python, SOMA | Metal vs Python, G1 |
|---|---|---|---|---|---|
| 1 | 10.70 / 10.55 ms | 10.62 / 10.50 | **7.14 / 7.09** | **1.50×** (1.50–1.53) | **1.49×** (1.48–1.51) |
| 2 | 11.08 / 10.90 | 11.01 / 10.86 | **9.23 / 9.10** | 1.20× | 1.20× |
| 4 | 12.01 / 11.87 | 11.95 / 11.82 | **10.79 / 10.72** | 1.11× | 1.11× |
| 8 | 16.00 / 15.83 | 15.93 / 15.74 | **15.06 / 14.93** | 1.06× | 1.06× |
| 16 | 25.09 / 24.80 | 25.00 / 24.84 | 25.06 / 24.88 | 1.00× | 1.00× |
| 32 | 45.48 / 44.85 | 45.57 / 45.00 | 46.66 / 46.01 | 0.97× | 0.98× |

(Cells give SOMA / G1.) For a stream of single requests the Metal runtime sustains 140 req/s
without pipelining. Pipelined MLX reaches 150 req/s.

### Usage

```sh
cd model-rs
cargo build --release      # first build compiles MLX from source (~6 min); needs Xcode + Metal toolchain
cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 bench/requests/unitree_g1_N8.npz
```

```rust
use motionbricks_rs::metal::model::MetalModel;
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 64)?; // max batch 64
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?;  // ONNX input names, NumPy-style arrays
```

`model-rs/.cargo/config.toml` points `DEVELOPER_DIR` at Xcode, so `xcode-select` can stay on the
Command Line Tools. It also keeps the 175 MB `mlx.metallib` inside `model-rs/target/`. The design,
the MLX analysis that motivated it, and per-stage profiles are in
[model-rs/README.md](model-rs/README.md).

### Correctness

`cargo test --release -- --test-threads=1` runs the same golden fixtures and tolerances as the
Python suite against both Rust runtimes. For the Metal runtime, every sub-network stays within
≤ 1.2e-5 relative and durations and pose tokens are identical. End to end, SOMA rotations are
within ≤ 7.9e-4 and G1 `pred_qpos` within ≤ 1.5e-4.

---

## Part 3 — Benchmarking and development (`sandbox/`)

### The CUDA sandbox

Nothing can run NVIDIA's CUDA binaries on Apple silicon. `sandbox/cudashim` therefore swaps each
CUDA layer for a Mac equivalent at its API, which lets NVIDIA's own code run unchanged as the
reference:

| CUDA layer | Mac replacement | Verified with |
|---|---|---|
| `torch.cuda`, `.cuda()` | PyTorch on MPS | NVIDIA's PyTorch MotionBricks moved with `.cuda()` matches the CPU golden outputs |
| TensorRT | ONNX Runtime behind a TensorRT 10 API | NVIDIA's own engine-build code runs and matches PyTorch |
| Warp CUDA kernels | Warp's CPU backend | NVIDIA's kernels compile and run unchanged |
| raw CUDA C++ (`.cu`, cuBLAS) | CuMetal CUDA→Metal compiler (optional) | 3 programs compile and run **on the M4 Pro GPU** (vector add, shared-memory reduction, cuBLAS SGEMM plus a custom kernel) |

CuMetal was rechecked for this release: Xcode 27.2 (27B52), Metal Toolchain 27B47, Homebrew LLVM
23.1.6 and CuMetal 0.8.0. All three CuMetal tests pass. They skip on machines without the
toolchain, which needs full Xcode: `sandbox/native/install_cumetal.sh`. Research notes, sources
and limits are in [sandbox/README.md](sandbox/README.md).

### Development workflow

```sh
sandbox/setup.sh                              # env + pinned upstream + verified checkpoints + ONNX + golden fixtures
uv run --all-packages pytest                  # Python: 87 tests (70 model, 17 sandbox incl. 3 CuMetal)
(cd model-rs && cargo test --release -- --test-threads=1)   # Rust: both runtimes vs the golden fixtures
```

- **Golden fixtures:** `sandbox/scripts/make_golden.py` records the inputs and outputs of every
  sub-network from NVIDIA's PyTorch model, for 5 request families per model. Both test suites
  teacher-force each stage with them, so an error localizes to one module.
- **Python benchmarks:** `uv run --all-packages python model/benchmarks/bench.py` times MLX at
  each precision, PyTorch on MPS and CPU, and ONNX Runtime. It writes JSON to
  `model/benchmarks/results/`, and `report.py` renders the tables.
- **Python vs Rust:**
  1. `OUT=$PWD/model-rs/bench/results/raw model-rs/bench/run_all.sh 4 float32` runs interleaved
     rounds of Python MLX, mlx-rs and the Metal runtime on byte-identical requests.
  2. `model-rs/bench/report_metal.py` and `report.py` summarize them.
- **Profiling tools in `model-rs`:**
  - `metal-stages`: GPU time per stage;
  - `gemm-sweep`: GEMM tile and split-K tuning;
  - `proto-transformer`: the transformer prototype against MLX;
  - `bench/stage_profile.py` and `bench/python_overhead.py`: MLX-side profiles.

Lessons from profiling, which apply to both versions:
1. **N=1 is GPU-bound, not CPU-bound.** A Metal System Trace shows the GPU busy 99% of a call.
   About 3 ms of that is weight streaming, and the rest is the latency of about 500 small,
   dependent kernels.
2. **The host language doesn't matter.** Rust and Python match on MLX, because MLX's C++ core
   does the work.
3. **What pays off:** fewer, better-shaped kernels (the Metal runtime), smaller weights
   (BF16/INT8), batching, and pipelining.

## License

**Apache-2.0** ([LICENSE](LICENSE)), the same license as upstream's code, so both ports stay
license-compatible with MotionBricks. [NOTICE](NOTICE):
- carries NVIDIA's copyright and marks the modified files;
- credits the third-party code upstream inherited (T2M-GPT, MotionGPT,
  vector-quantize-pytorch), with their licenses in [licenses/](licenses/);
- lists the Rust dependencies, which are fetched, not vendored.

The **weights are not included**. They are under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
and are downloaded by `sandbox/setup.sh` from
[huggingface.co/nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks). Upstream sample
motions and robot meshes are not redistributed either.

## Citation

```bibtex
@article{wang2026motionbricks,
  title   = {MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives},
  author  = {Wang, Tingwu and Dionne, Olivier and De Ruyter, Michael and Minor, David and Rempe, Davis and Zhao, Kaifeng and Petrovich, Mathis and Yuan, Ye and Li, Chenran and Luo, Zhengyi and Robison, Brian and Blackwell, Xavier and Antoniazzi, Bernardo and Peng, Xue Bin and Zhu, Yuke and Yuen, Simon},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026}, volume = {45}, number = {4}, doi = {10.1145/3811334}
}
```
