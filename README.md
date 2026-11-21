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
| [`model/`](model/README.md) | **`motionbricks_mlx`** (Python), a native MLX port of the complete SOMA and Unitree G1 inference graphs. It has the same tensor interface as the upstream ONNX models, supports FP32, FP16, BF16 and INT8, and depends only on `mlx` + `numpy`. It also includes upstream's **Runtime API**: engines, motion players, recorders, target poses and clips. |
| [`model-rs/`](model-rs/README.md) | **`motionbricks-rs`** (Rust). It contains an op-for-op port on mlx-rs and a **model-specific Metal runtime** (FP32, FP16, BF16, INT8), the fastest way to run MotionBricks at every batch size. It also includes the **Runtime API**, running on either backend. |
| [`sandbox/`](sandbox/README.md) | **`cudashim`**, a layered CUDA-emulation sandbox for M-series Macs. It maps `torch.cuda`→MPS, TensorRT→ONNX Runtime and Warp kernels→CPU, and optionally compiles CUDA C++→Metal through CuMetal. It also runs NVIDIA's upstream model as the reference oracle for both versions. |

### Quick start: three independent paths

The models don't need the sandbox. Pick the path you need; each one downloads the weights
(~1.6 GB from `nvidia/MotionBricks` at the pinned revision) and verifies every file's SHA-256.

| Path | Needs | Commands |
|---|---|---|
| **Python model only** ([Part 1](#part-1--using-the-python-model-motionbricks_mlx)) | Python 3.12, [uv](https://docs.astral.sh/uv/) (or pip); installs only `mlx` + `numpy` | `uv sync --package motionbricks-mlx`<br>`uv run --package motionbricks-mlx motionbricks-mlx download`<br>`uv run --package motionbricks-mlx motionbricks-mlx generate --model soma --checkpoints checkpoints/MotionBricks/soma --batch 4` |
| **Rust model only** ([Part 2](#part-2--the-rust-optimized-model-model-rs)) | Rust, Xcode with its Metal toolchain; no Python | `model-rs/scripts/download_weights.sh`<br>`cd model-rs && cargo build --release`<br>`cargo run --release --example generate_metal -- soma ../checkpoints/MotionBricks/soma 4 bfloat16` |
| **Full sandbox** ([Part 3](#part-3--benchmarking-and-development-sandbox)): NVIDIA's reference, golden fixtures, every test, benchmarks | uv; about 5 GB of disk | `sandbox/setup.sh`<br>`uv run --all-packages pytest`<br>`(cd model-rs && cargo test --release -- --test-threads=1)` |

The model test suites run on the first two paths too. Without the sandbox they run every test
except the parity tests against NVIDIA's code, which need the sandbox's golden fixtures and report
themselves as skipped.

**Runtime API.** Both ports also carry upstream's Runtime API, which turns generations into
continuous character motion:
- engines that generate from four context poses and one to four target poses;
- motion players and recorders that chain generations;
- target placement, speed scaling and SOMA IK pins;
- BVH/CSV clip loading.

The guides are [model/docs/runtime_api.md](model/docs/runtime_api.md) (Python) and
[model-rs/docs/runtime_api.md](model-rs/docs/runtime_api.md) (Rust). Add `--clips` to `download`, or
run `download_weights.sh clips`, for the sample clips its examples use.

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
0.6–4) on 2026-11-21.

The table shows the median time per batched generation, with requests/s in parentheses:

| Backend | SOMA N=1 | SOMA N=32 | G1 N=1 | G1 N=32 |
|---|---|---|---|---|
| **Rust Metal runtime, BF16** (`model-rs`) | **4.05 ms** (247) | **35.21 ms (909)** | **4.04 ms** (248) | **35.42 ms (903)** |
| Rust Metal runtime, INT8 weights | **4.03 ms** (248) | 42.78 ms (748) | 4.06 ms (246) | 42.15 ms (759) |
| Rust Metal runtime, FP32 | 5.18 ms (193) | 42.91 ms (746) | 5.18 ms (193) | 43.72 ms (732) |
| Python MLX, BF16 | 8.25 ms (121) | 37.86 ms (845) | 8.30 ms (121) | 37.76 ms (848) |
| Python MLX, INT8 weights | 8.01 ms (125) | 46.37 ms (690) | 7.96 ms (126) | 45.64 ms (701) |
| Python MLX, FP32 | 10.72 ms (93) | 45.18 ms (708) | 10.55 ms (95) | 45.38 ms (705) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 32.08 ms (31) | 91.19 ms (351) | 37.04 ms (27) | 98.97 ms (323) |
| upstream ONNX on ONNX Runtime CPU (TensorRT shim) | 13.54 ms (74) | 210.6 ms (152) | 14.06 ms (71) | 213.6 ms (150) |
| upstream PyTorch, CPU | 33.41 ms (30) | 867 ms (37) | 33.29 ms (30) | 914 ms (35) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU time only)* | *2.50 ms* | *5.28 ms* | *2.67 ms* | *7.81 ms* |

Where the numbers come from:
- **Rust Metal runtime and Python MLX rows:** the same interleaved rounds, so the comparison
  between them is paired. [Part 2](#part-2--the-rust-optimized-model-model-rs) has every batch size
  and FP16.
- **Other rows:** the Python harness `model/benchmarks/bench.py`
  ([full tables](model/benchmarks/README.md)). They were measured for the previous release on the
  same machine and software; those backends haven't changed since.

What the table shows:
- **Lowest latency:** the Rust Metal runtime takes **4.0 ms for a single request** in BF16,
  FP16 or INT8, and 5.2 ms in FP32. That is about **2× faster** than Python MLX at the same
  precision.
- **Highest throughput:** the Rust Metal runtime in **BF16** delivers 903–909 req/s at N=32,
  against Python BF16's 845–848.
- **Every batch size and precision:** the Metal runtime is faster than Python MLX in all 144 paired
  comparisons (2 models × 4 precisions × 6 batch sizes × 3 rounds). The smallest margin is 1.02×,
  for FP32 at N=32.
- **Versus other Mac options:** at its fastest precision, Python MLX is 2.4–4.7× faster than
  NVIDIA's PyTorch model on the GPU (MPS), and the Rust Metal runtime 2.6–9.2× faster. At batch 32
  they deliver 5.6–5.7× and 6.0× the throughput of ONNX Runtime on the CPU.

---

## Part 1 — Using the Python model (`motionbricks_mlx`)

### Setup

```sh
uv sync --package motionbricks-mlx                      # only mlx + numpy (or: pip install ./model)
uv run --package motionbricks-mlx motionbricks-mlx download             # both models; --model soma|unitree_g1
uv run --package motionbricks-mlx motionbricks-mlx generate --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1 --batch 4
uv run --package motionbricks-mlx --extra test pytest model/tests       # 30 tests; +40 parity tests with the sandbox
```

`download` fetches the checkpoints over plain HTTPS from the revision this port was tested with.
It checks each file's size and SHA-256 against upstream's own checksum table
(`model/motionbricks_mlx/assets/checkpoints.sha256`) and skips files that are already verified.
`--package motionbricks-mlx` keeps the sandbox's PyTorch, ONNX Runtime and Warp out of the
environment. If you already ran `sandbox/setup.sh` (Part 3), the weights and the full environment
are in place, and `uv run motionbricks-mlx ...` works too.

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

### Runtime API

`motionbricks_mlx.runtime` is upstream's Runtime API on MLX, with upstream's names and behavior:
engines, tokens, animation buffers, `MotionRecorder`, motion players, target placement, IK pins and
clip loading. Poses are NumPy arrays and inference is the MLX model in any precision:

```python
from motionbricks_mlx import runtime as rt

with rt.MotionBricksUnitreeG1Engine.from_checkpoints("checkpoints/MotionBricks/g1", dtype="bfloat16") as engine:
    context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
    context.update_from_poses(context_joint_q)          # four poses of 36 joint coordinates
    target.update_from_poses(target_joint_q)            # one to four poses
    engine.generate_animation(context, target, output)  # output.num_frames valid frames at 30 Hz
    pose = output.sample(0.5)
```

`model/examples/generate_once.py` and `locomotion_loop.py` are the ports of upstream's two examples.
The guide is [model/docs/runtime_api.md](model/docs/runtime_api.md).

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

`model/tests` has 109 tests:
- **10 weight-free unit tests** (every block against NumPy derivations);
- **20 behavioural tests** that need the checkpoints;
- **40 parity tests** against NVIDIA's PyTorch model, which need the sandbox's golden fixtures;
- **39 Runtime API tests** (`test_runtime.py`): weight-free unit tests, step-by-step parity with
  upstream's own runtime kernels, and end-to-end generation.

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
| what it is | the Python graph op for op on mlx-rs: same MLX kernels | a fixed dispatch plan written for these two networks, on its own Metal kernels |
| speed | same as Python (0.99–1.01× the time, compiled) | **about 2× faster at N=1**; 1.8–1.9× at N=2, 1.4× at N=4, 1.2–1.3× at N=8, 1.07–1.18× at N=16, 1.04–1.09× at N=32 (INT8 at N=2–4: 1.35–1.6×) |
| precisions | FP32, FP16, BF16 | FP32, FP16, BF16, INT8 (the same modes as the Python port) |
| configurations | everything the Python port supports | the release configuration only |

### Best speed, at the price of generality

The Metal runtime is fast *because* it isn't general:
- **Static plan:** a generation is one precomputed list of 240–400 dispatches. The CPU encodes it in
  about 0.12 ms and commits it in chunks, so the GPU starts almost at once. MLX builds and
  schedules a graph of about 500 ops in about 2 ms.
- **Tuned GEMMs:** one GEMM kernel with 11 tile shapes and split-K. A plan table measured on this
  Mac picks the best tile and split for every GEMM the two models launch. Skinny 16-row tiles
  stream the weights at close to the memory bandwidth at batch 1.
- **Fused work:** convolutions, masks, concatenations, blends and INT8 dequantization are folded
  into the matrix multiplies. Attention runs on simdgroup matrices straight from the packed QKV,
  and the rig runs as one thread per pose.
- **Precomputed constants:** the pose network's token embedding is computed once at load time,
  because the release configuration makes it constant.

What that ties it to:
- **This exact model implementation.** The kernels and the dispatch plan encode the SOMA/G1
  shapes, the post-norm transformer, the `DoubleCondDecoder` layout and the rig. It only supports
  one MaskGIT step with model-chosen durations, with a batch size capped when the model is built.
  Any architecture or configuration change means editing kernels and plans, not Python.
- **The device.** It runs on Apple-silicon Metal GPUs only and needs Xcode with its Metal
  toolchain to build. The GEMM plan table was measured on this M4 Pro and is used only on an
  M4 Pro. Other M-series chips fall back to a fitted rule; it runs everywhere, but the speedups
  above are only measured here. `gemm-tune` regenerates the table for another Mac.

So the Python port (or the mlx-rs port) is the flexible choice: every configuration, easy changes,
the same speed on any Mac. The Metal runtime is the choice for the **fastest inference on known
Apple hardware**: about 2× lower latency for single requests, such as a robot control loop or a game
character, and still the fastest option for batched throughput.

SOMA, median time per call end to end (inputs in, outputs on the host), from 3 interleaved rounds
per precision. Each cell gives Python MLX → **Rust Metal runtime** in ms, and the speedup:

| N | FP32 | BF16 | FP16 | INT8 weights |
|---|---|---|---|---|
| 1 | 10.72 → **5.18** (2.06×) | 8.25 → **4.05** (2.04×) | 8.24 → **4.04** (2.04×) | 8.01 → **4.03** (1.99×) |
| 2 | 11.06 → **5.98** (1.85×) | 8.89 → **4.66** (1.91×) | 8.87 → **4.70** (1.88×) | 8.48 → **5.26** (1.61×) |
| 4 | 12.02 → **8.28** (1.45×) | 10.00 → **6.92** (1.45×) | 9.99 → **6.99** (1.43×) | 10.75 → **7.93** (1.36×) |
| 8 | 15.99 → **13.08** (1.20×) | 13.94 → **10.90** (1.28×) | 13.93 → **10.97** (1.27×) | 15.37 → **12.82** (1.20×) |
| 16 | 24.90 → **22.99** (1.08×) | 21.77 → **18.38** (1.18×) | 21.78 → **19.08** (1.14×) | 25.26 → **23.01** (1.10×) |
| 32 | 45.18 → **42.91** (1.04×) | 37.86 → **35.21** (1.08×) | 37.86 → **36.49** (1.04×) | 46.37 → **42.78** (1.09×) |

- **Unitree G1:** within 0.04× of these ratios. Both models, the per-round ranges and the mlx-rs
  port's times are in
  [`model-rs/bench/results/report_metal.md`](model-rs/bench/results/report_metal.md).
- **Streams of single requests:** the Metal runtime sustains 193–194 req/s in FP32 and 247–251 req/s
  in BF16/FP16 without pipelining. Pipelined Python reaches 149–153 and 198–199.

What changed since the first release, which was FP32 only and 0.97× Python's speed at N=32:
- **FP16, BF16 and INT8 kernels**, with the same modes as the Python port.
- **Attention on simdgroup matrices:** 1.21 → 0.48 ms per pose transformer at N=32.
- **Tile-aligned fast loaders for convs, concatenations and masks:** −1.2 ms of GEMM time at BF16
  N=32.
- **A tuned GEMM plan per launch:** up to −7% of GEMM time at N=8.
- **Chunked command-buffer commits:** the GPU starts while the CPU is still encoding, which saves
  0.16–0.25 ms per call (4–6% at N=1).
- **No MLX kernels at run time:** the steel GEMMs the first release borrowed from `mlx.metallib`
  are replaced by its own kernel, which matches them.

### Usage

```sh
model-rs/scripts/download_weights.sh      # curl + shasum against the same checksum table; or: all|soma|unitree_g1 [dir]
cd model-rs
cargo build --release      # first build compiles MLX from source (~6 min); needs Xcode + Metal toolchain
cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 8 bfloat16   # batch 8, synthetic request
cargo run --release --example generate -- soma ../checkpoints/MotionBricks/soma 8 int8                 # the mlx-rs port
cargo test --release -- --test-threads=1  # 64 tests; parity tests skip without the sandbox's fixtures
```

The examples take a batch size (a synthetic request from `motionbricks_rs::examples`, the Rust
version of the Python generator) or a request `.npz` with the ONNX input names.

**Runtime API.** `motionbricks_rs::runtime` provides the same engines, motion players, recorders,
target helpers and clip loaders as the Python port. Each engine runs on a `Backend`, either the
Metal runtime (any precision) or the mlx-rs port:

```rust
use motionbricks_rs::{metal::model::MetalModel, runtime::*};
let model = MetalModel::from_checkpoints_with("unitree_g1", "checkpoints/MotionBricks/g1", 1, "bfloat16")?;
let mut engine = MotionBricksUnitreeG1Engine::new(Some(model), Transform::IDENTITY)?;
let (mut context, mut target, mut output) = (engine.create_input(), engine.create_input(), engine.create_output());
context.update_from_poses(&context_joint_q)?;
target.update_from_poses(&target_joint_q)?;
engine.generate_animation(&context, &target, &mut output, &MotionBricksInferenceConfig::default())?;
```

On the Metal runtime, a whole generation, including encoding, decoding and endpoint correction,
takes about 7 ms in FP32 and 5 ms in BF16. `cargo run --release --example locomotion_loop`
and `generate_once` are the ports of upstream's examples. The guide is
[model-rs/docs/runtime_api.md](model-rs/docs/runtime_api.md).

```rust
use motionbricks_rs::metal::model::MetalModel;
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 64)?; // FP32, max batch 64
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?;  // ONNX input names, NumPy-style arrays
let fast = MetalModel::from_checkpoints_with("soma", "checkpoints/MotionBricks/soma", 64, "bfloat16")?; // float16 | bfloat16 | int8
let forced = model.run_with_durations(&inputs, Some(&[8, 12]))?;   // one fixed duration (6..=16 tokens) per request
motionbricks_rs::convert("soma", "checkpoints/MotionBricks/soma", "weights/soma.safetensors", mlx_rs::Dtype::Float32)?;
let from_file = MetalModel::from_converted("weights/soma.safetensors", 64, "float32")?;  // same file format as Python's convert
```

`model-rs/.cargo/config.toml` points `DEVELOPER_DIR` at Xcode, so `xcode-select` can stay on the
Command Line Tools. It also keeps the mlx-rs port's 175 MB `mlx.metallib` inside
`model-rs/target/`. The Metal runtime compiles its own kernels at load time and doesn't use MLX
kernels. The design, the MLX analysis that motivated it, and per-stage profiles are in
[model-rs/README.md](model-rs/README.md).

### Correctness

`cargo test --release -- --test-threads=1` mirrors the Python suite file by file, for both Rust
runtimes, with the same requests, checks and tolerances:

| Python (`model/tests`) | mlx-rs port (`model-rs/tests`) | Metal runtime (`model-rs/tests`) |
|---|---|---|
| `test_parity.py`: root, pose, VQ-VAE, end to end × 5 cases × 2 models | `parity.rs`: the same 4 stages × 2 models, each test over all 5 cases (end to end eager and compiled) | `metal_parity.rs`: the same, each stage teacher-forced through the runtime's buffers |
| `test_model.py`: ONNX contract, compiled = eager, batch independence, allowed durations, forced durations, FP16/BF16, converted files, multi-step decoding, INT8 | `model.rs`: all 9, for both models | `metal_model.rs`: the same, except compiled = eager becomes "chunked commits = one command buffer" and multi-step decoding (not in the release-only plan) becomes refusing oversized batches |
| `test_layers.py`: 10 weight-free unit tests of the building blocks and the rig kernels | `layers.rs`: the same blocks against float64 references, plus the attention path MLX's fused kernels don't cover | `metal_kernels.rs`: every GEMM loader and epilogue under all 33 tile/split plans (442 launches), half-precision GEMMs, and a full transformer layer in every precision with both attention kernels |

`tests/runtime.rs` mirrors `test_runtime.py` (23 tests) and runs the end-to-end generations on both
backends. Altogether that gives 88 Rust tests. Rust's test harness has no skip state, so tests
without their checkpoints or fixtures print `SKIPPED` and pass.

Porting `compiled = eager` found a real bug: compiled mlx-rs runs could flip pose tokens. MLX 0.32.2
(the version mlx-rs links) writes scalar constants into compiled kernels with too few digits, which
broke the fallback attention MLX uses for the root networks' head dim 32. The port now runs that
fallback itself with the scale stored as an array. Compiled and eager are bitwise identical again,
as in Python.
- **Metal runtime, FP32:** every sub-network stays within ≤ 1.2e-5 relative, and durations and pose
  tokens are identical. End to end, SOMA rotations are within ≤ 9.4e-4 and G1 `pred_qpos` within
  ≤ 1.6e-4.
- **FP16, BF16 and INT8:** the Python port's criteria apply: at least 75% of durations agree with FP32,
  and the median deviation is below 0.05. On the golden requests they pick the same duration as FP32
  on 15–16 of 16 requests. The median deviation is 1–3e-3 for FP16, 1–2e-2 for BF16 and 0.7–3.5e-3
  for INT8. That matches the Python port at the same precision: on the SOMA `random` request, BF16
  deviates by 4.4e-2 in Rust and 4.5e-2 in Python.

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
uv run --all-packages pytest                  # Python: 128 tests (109 model, 19 sandbox incl. 3 CuMetal)
(cd model-rs && cargo test --release -- --test-threads=1)   # Rust: 88 tests, both runtimes vs the golden fixtures
```

- **Golden fixtures:** `sandbox/scripts/make_golden.py` records the inputs and outputs of every
  sub-network from NVIDIA's PyTorch model, for 5 request families per model. Both test suites
  teacher-force each stage with them, so an error localizes to one module.
- **Runtime fixtures:** `sandbox/scripts/make_runtime_golden.py` runs upstream's own Runtime API
  code on Warp's CPU backend: engine encode/decode/correction methods, recorder, target helpers,
  IK and clip loaders. NVIDIA's PyTorch model stands in for TensorRT. Both ports' runtime tests
  check each step against these fixtures.
- **Python benchmarks:** `uv run --all-packages python model/benchmarks/bench.py` times MLX at
  each precision, PyTorch on MPS and CPU, and ONNX Runtime. It writes JSON to
  `model/benchmarks/results/`, and `report.py` renders the tables.
- **Python vs Rust:**
  1. `OUT=$PWD/model-rs/bench/results/raw model-rs/bench/run_all.sh 3 "float32 bfloat16 float16 int8"`
     runs interleaved rounds of Python MLX, mlx-rs and the Metal runtime on byte-identical requests.
  2. `model-rs/bench/report_metal.py` and `report.py` summarize them.
  3. `model-rs/bench/quick_compare.py` gives a single paired pass for quick checks.
- **Profiling and tuning tools in `model-rs`:**
  - `metal-stages <precision> [N...]`: GPU time per stage;
  - `gemm-tune <model> <precision> [N...]`: replays every GEMM launch of a generation under every
    tile and split. It reports the table, rule and best plans, and `TUNE_TABLE=<file>` writes a
    plan table;
  - `gemm-shapes`, `gemm-sweep`: the GEMM launches of a generation, and a synthetic sweep;
  - `metal-precision`: an FP16/BF16/INT8 run against FP32, stage by stage;
  - `proto-transformer <precision>`: the pose transformer against MLX;
  - `bench/stage_profile.py` and `bench/python_overhead.py`: MLX-side profiles.

Lessons from profiling, which apply to both versions:
1. **N=1 is GPU-bound, not CPU-bound.** A Metal System Trace shows the GPU busy 99% of a call.
   About 3 ms of that is weight streaming, and the rest is the latency of about 500 small,
   dependent kernels.
2. **The host language doesn't matter.** Rust and Python match on MLX, because MLX's C++ core
   does the work.
3. **What pays off:** fewer, better-shaped kernels (the Metal runtime), smaller weights
   (BF16/INT8), batching, and pipelining.
4. **At large batches, measure in pairs.** Sustained GPU load raises the M4 Pro's GPU time at N=32
   by up to 10% within a few minutes, so only interleaved runs compare two runtimes fairly.

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
