# motionbricks-mlx user guide

This guide covers motionbricks-mlx in full:
- installation paths and what each one needs;
- the Python and Rust models and how they differ;
- the Runtime API;
- benchmarks, tuning and profiling;
- how both ports are checked against NVIDIA's code;
- the development sandbox and license details.

The [README](../README.md) is the short version.

1. [Install](#1-install)
2. [The Python model (`motionbricks_mlx`)](#2-the-python-model-motionbricks_mlx)
3. [The Rust implementations (`model-rs`)](#3-the-rust-implementations-model-rs)
4. [The Runtime API](#4-the-runtime-api)
5. [Benchmarks](#5-benchmarks)
6. [Tuning and profiling](#6-tuning-and-profiling)
7. [The development sandbox (`sandbox/`)](#7-the-development-sandbox-sandbox)
8. [License details](#8-license-details)

---

## 1. Install

### Three independent paths

The models don't need the sandbox. Each path downloads the weights (about 1.6 GB from
`nvidia/MotionBricks` at the pinned revision) and verifies every file's SHA-256.

| Path | Needs | Commands |
|---|---|---|
| **Python model only** ([section 2](#2-the-python-model-motionbricks_mlx)) | Python 3.12, [uv](https://docs.astral.sh/uv/) (or pip); installs only `mlx` + `numpy` | `uv sync --package motionbricks-mlx`<br>`uv run --package motionbricks-mlx motionbricks-mlx download`<br>`uv run --package motionbricks-mlx motionbricks-mlx generate --model soma --checkpoints checkpoints/MotionBricks/soma --batch 4` |
| **Rust model only** ([section 3](#3-the-rust-implementations-model-rs)) | Rust, Xcode with its Metal toolchain; no Python | `model-rs/scripts/download_weights.sh`<br>`cd model-rs && cargo build --release`<br>`cargo run --release --example generate_metal -- soma ../checkpoints/MotionBricks/soma 4 bfloat16` |
| **Full sandbox** ([section 7](#7-the-development-sandbox-sandbox)): NVIDIA's reference, golden fixtures, every test, all benchmarks | uv; about 5 GB of disk | `sandbox/setup.sh`<br>`uv run --all-packages pytest`<br>`(cd model-rs && cargo test --release -- --test-threads=1)` |

The model test suites also run on the first two paths. Without the sandbox, the parity tests
against NVIDIA's code have no fixtures and are skipped:
- **Python:** 95 of the 181 model tests run with the checkpoints, and 69 run without them.
- **Rust:** tests without their inputs print `SKIPPED` (on stderr) and pass, because Rust's test
  harness has no skip state. `MOTIONBRICKS_REQUIRE_FIXTURES=1` makes them fail instead.

### Sample clips

The [Runtime API](#4-the-runtime-api) examples start from four of upstream's sample motion clips:
a walk and a jog for each rig. Two commands fetch them into `checkpoints/motions`:
- **Python:** `motionbricks-mlx download --clips`, or `--only-clips` for the clips alone.
- **Rust:** `model-rs/scripts/download_weights.sh clips`.

The clips come from upstream's Git LFS at the pinned commit. Each one is checked against
`model/motionbricks_mlx/assets/clips.sha256`, and the sandbox tests that this manifest matches
upstream's LFS pointers. The clips are NVIDIA sample data under the NVIDIA Sample Data Evaluation
License, and this repository does not redistribute them.

### Ported and tested version

| | Version |
|---|---|
| Code | [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) **v0.2.0**, commit [`54fc563aeaaf819a753d72cc9fdc6713989cdd43`](https://github.com/NVlabs/motionbricks/tree/54fc563aeaaf819a753d72cc9fdc6713989cdd43) |
| Weights | [nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks) on Hugging Face, revision [`3cb362d9b95aa40462abcb4d8cfae9ffb6d97410`](https://huggingface.co/nvidia/MotionBricks/tree/3cb362d9b95aa40462abcb4d8cfae9ffb6d97410) |
| SOMA checkpoints | MotionBricks v6 release `soma-v6-nova-4n8g-ea43dd8-fp32-v1`: pose, root and VQ-VAE networks at training step 1,500,000 |
| Unitree G1 checkpoints | release `out/motionbricks_{pose,root,vqvae}/version_1`: pose, root and VQ-VAE networks at training step 2,000,000 |
| Reference ONNX graphs | `mmm_v6_soma_fp32.onnx` and `mmm_v6_g1_fp32.onnx`, exported from the code and weights above with upstream's own exporter (or with the Rust port of it, [`tools/onnx`](../tools/onnx/README.md)) |

The download commands and `sandbox/scripts/fetch_assets.py` pin both the commit and the revision.
They verify every checkpoint file against `model/motionbricks_mlx/assets/checkpoints.sha256`, and
the sandbox tests that this table equals upstream's `tools/onnx/checksums.py`.

---

## 2. The Python model (`motionbricks_mlx`)

### Setup

```sh
uv sync --package motionbricks-mlx                      # only mlx + numpy (or: pip install ./model)
uv run --package motionbricks-mlx motionbricks-mlx download             # both models; --model soma|unitree_g1, --clips
uv run --package motionbricks-mlx motionbricks-mlx generate --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1 --batch 4
uv run --package motionbricks-mlx --extra test pytest model/tests       # 181 tests; 95 run without the sandbox
```

How `download` works:
- **Source:** the checkpoints come over plain HTTPS from the revision this port was tested with.
- **Verification:** it checks each file's size and SHA-256 against upstream's own checksum table
  (`model/motionbricks_mlx/assets/checkpoints.sha256`) and skips files that are already verified.
- **Interruptions:** each file streams into a `.part` file and is renamed only once its size and
  SHA-256 match, so a partial file never looks complete. A stalled connection fails after 60 s, and
  errors name the file. Rerun to retry; `download_weights.sh` also resumes from its `.part` file.
- **Environment:** `--package motionbricks-mlx` keeps the sandbox's PyTorch, ONNX Runtime and Warp
  out of the environment. If you already ran `sandbox/setup.sh` ([section 7](#7-the-development-sandbox-sandbox)),
  the weights and the full environment are in place, and `uv run motionbricks-mlx ...` works too.

```python
import motionbricks_mlx as mb
from motionbricks_mlx import examples

model = mb.load("soma", "checkpoints/MotionBricks/soma", dtype="bfloat16")  # float32 | float16 | bfloat16
out = model.run(examples.request("soma", batch=8))       # dict of NumPy in (ONNX input names) -> dict of NumPy out
frames = [out["pred_global_motions"][i, : 4 * n] for i, n in enumerate(out["pred_num_tokens"])]

mb.convert("unitree_g1", "checkpoints/MotionBricks/g1", "weights/g1.safetensors")   # one-file MLX weights
g1 = mb.load("weights/g1.safetensors").quantize(bits=8)  # INT8 transformer weights
```

The tensor interface is identical to upstream's ONNX models: the same names, shapes and INT32
masks. Code written against ONNX Runtime or TensorRT can switch by swapping the session object.
`run` checks every input's shape on the host and raises `ValueError` naming the input. An
unsupported dtype string, or `mb.load("soma")` without the checkpoint folder, fails with a message
that says what to pass.
The CLI also has `convert` and `bench` subcommands. The architecture, tensor shapes and design
notes are in [model/README.md](../model/README.md).

### Choosing precision and batching

| Goal | Use | Why |
|---|---|---|
| exact reference behaviour | FP32 (default) | matches PyTorch FP32 to ≤ 1.2e-5 per sub-network |
| single requests, lowest latency in Python | **INT8** (7.9 ms) or **BF16** (8.1 ms) | batch 1 is bound by streaming the ~800 MB of FP32 weights; smaller weights help directly |
| many characters at once | **BF16 with batching** | N=32 costs about 38 ms, which is 848 generations/s |
| a stream of single requests | `mx.async_eval` pipelining (below) | encode request i+1 while the GPU runs request i: about 98 → 155 req/s in FP32, 125 → 200 in BF16 |

`run` waits for each result. To pipeline a stream, queue request i on the GPU and read request
i-1 while it runs:

```python
import mlx.core as mx
import numpy as np

pending = None
for request in requests:                                        # dicts of ONNX-named arrays, batch 1
    outputs = model.forward_compiled(*model.prepare(request))
    mx.async_eval(outputs)                                      # starts on the GPU, returns at once
    if pending is not None:
        handle(dict(zip(model.output_names, map(np.array, pending))))   # waits only for request i-1
    pending = outputs
handle(dict(zip(model.output_names, map(np.array, pending))))
```

INT8 and BF16 chose the same durations as FP32 on every test request; check agreement for your
own inputs. Two Python optimizations were applied in the second release:
- **Weights evaluated up front:** the cast or quantized weights are evaluated before compiling, so
  each compiled call no longer recomputes about 200 M parameter casts. BF16 went from 12.2 to
  8.1 ms and INT8 from 12.4 to 7.9 ms at N=1.
- **Rig as custom Metal kernels:** the rig math runs in `mx.fast.metal_kernel` kernels, shared
  with the Rust port in `assets/rig_kernels.metal`. That makes G1 7–11% faster and SOMA 7% faster
  at N=32.

### Correctness

`model/tests` has 181 tests:
- **18 weight-free unit tests** (every block against NumPy derivations, the rig kernels against
  the op-based rig, multi-step re-masking against a port of upstream's sampler);
- **21 weight-free API tests** (input validation, dtype and loading errors, strict weight loading);
- **26 behavioural tests** that need the checkpoints;
- **42 parity tests** against NVIDIA's PyTorch model, which need the sandbox's golden fixtures
  (2 of them run end to end on the CPU stream);
- **66 Runtime API tests** (`test_runtime.py`): 22 weight-free, 36 step-by-step parity checks with
  upstream's own runtime code, and 8 end-to-end generations;
- **8 offline download tests** (`test_hub.py`: verification, size and checksum errors, cleanup).

Every sub-network is teacher-forced with PyTorch's exact inputs from the golden fixtures:
- **Sub-networks:** root ≤ 4e-6, pose logits ≤ 8e-6 with identical argmax tokens, VQ-VAE
  ≤ 1.2e-5 relative.
- **End to end:** durations are identical on all 5 request families per model.

`MOTIONBRICKS_REQUIRE_FIXTURES=1` turns every skip into a failure, so a complete run cannot pass
by skipping. The per-stage table is in [model/README.md](../model/README.md#verification).

---

## 3. The Rust implementations (`model-rs`)

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
  and the rig runs as one thread per pose and batch item. Each transformer layer's
  Linear → +residual → LayerNorm pairs are fused into the split-K reduce
  (`gemm_reduce_ln`: partials in, LayerNorm out, one dispatch per pair), which removes 48
  dispatches and the intermediate round trips at N=1 (~3–7% lower single-request GPU time).
- **Precomputed constants:** the pose network's token embedding is computed once at load time,
  because the release configuration makes it constant.

What that ties it to:
- **This exact model implementation.** The kernels and the dispatch plan encode the SOMA/G1
  shapes, the post-norm transformer, the `DoubleCondDecoder` layout and the rig. It only supports
  one MaskGIT step with model-chosen durations, with a batch size capped when the model is built.
  Any architecture or configuration change means editing kernels and plans, not Python.
- **The device.** It runs on Apple-silicon Metal GPUs only and needs Xcode with its Metal
  toolchain to build. The GEMM plan table was measured on an M4 Pro and is used only there.
  Other M-series chips fall back to a fitted rule; it runs everywhere, but the speedups above
  are only measured on the M4 Pro. [Section 6](#6-tuning-and-profiling) shows how to tune
  another Mac.

So the Python port (or the mlx-rs port) is the flexible choice: every configuration, easy changes,
the same speed on any Mac. The Metal runtime is the choice for the **fastest inference on known
Apple hardware**: about 2× lower latency for single requests, such as a robot control loop or a game
character, and still the fastest option for batched throughput.

### Usage

```sh
model-rs/scripts/download_weights.sh      # curl + shasum against the same checksum table; or: all|soma|unitree_g1|clips [dir]
cd model-rs
cargo build --release      # first build compiles MLX from source (~6 min); needs Xcode + Metal toolchain
cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 8 bfloat16   # batch 8, synthetic request
cargo run --release --example generate -- soma ../checkpoints/MotionBricks/soma 8 int8                 # the mlx-rs port
cargo test --release -- --test-threads=1  # 128 tests; parity tests print SKIPPED without the sandbox's fixtures
```

The examples take either of two inputs:
- a batch size, which builds a synthetic request with `motionbricks_rs::examples` (the Rust
  version of the Python generator);
- a request `.npz` with the ONNX input names.

```rust
use motionbricks_rs::metal::model::MetalModel;
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 64)?; // FP32, max batch 64
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?;  // ONNX input names, NumPy-style arrays
let fast = MetalModel::from_checkpoints_with("soma", "checkpoints/MotionBricks/soma", 64, "bfloat16")?; // float16 | bfloat16 | int8
let forced = model.run_with_durations(&inputs, Some(&[8, 12]))?;   // one fixed duration (6..=16 tokens) per request
motionbricks_rs::convert("soma", "checkpoints/MotionBricks/soma", "weights/soma.safetensors", mlx_rs::Dtype::Float32)?;
let from_file = MetalModel::from_converted("weights/soma.safetensors", 64, "float32")?;  // same file format as Python's convert
let mlx = motionbricks_rs::MotionBricks::load("soma", "checkpoints/MotionBricks/soma", "int8")?; // the mlx-rs port: float32 | float16 | bfloat16 | int8
```

`motionbricks_rs::Precision` names the four precisions; both runtimes accept it or its name. The
mlx-rs port checks the input count and shapes before it builds a graph, and
`forward_with_durations` checks forced durations (6..=16 tokens).

Build setup:
- **Xcode:** `model-rs/.cargo/config.toml` points `DEVELOPER_DIR` at Xcode, so `xcode-select` can
  stay on the Command Line Tools.
- **MLX kernels:** the same config keeps the mlx-rs port's 175 MB `mlx.metallib` inside
  `model-rs/target/`. The Metal runtime compiles its own kernels at load time and doesn't use MLX
  kernels.

The design, the MLX analysis that motivated it, and per-stage profiles are in
[model-rs/README.md](../model-rs/README.md).

### Correctness

`cargo test --release -- --test-threads=1` mirrors the Python suite file by file, for both Rust
runtimes, with the same requests, checks and tolerances:

| Python (`model/tests`) | mlx-rs port (`model-rs/tests`) | Metal runtime (`model-rs/tests`) |
|---|---|---|
| `test_parity.py`: root, pose, VQ-VAE, end to end × 5 cases × 2 models | `parity.rs`: the same 4 stages × 2 models, each test over all 5 cases (end to end eager and compiled) | `metal_parity.rs`: the same, each stage teacher-forced through the runtime's buffers |
| `test_model.py`: ONNX contract, compiled = eager, batch independence, allowed durations, forced durations, FP16/BF16, converted files, multi-step decoding, INT8 | `model.rs`: all 9 (compiled = eager also bitwise), plus malformed inputs being errors, for both models | `metal_model.rs`: the same, except compiled = eager becomes "chunked commits = one command buffer" and multi-step decoding (not in the release-only plan) becomes refusing oversized batches, plus malformed inputs and runtime-only checks (re-encoding, untuned batch sizes, plan rule) |
| `test_layers.py`: 18 weight-free unit tests of the building blocks and the rig kernels | `layers.rs`: the same blocks against float64 references, plus the attention path MLX's fused kernels don't cover | `metal_kernels.rs`: every GEMM loader and epilogue under all 33 tile/split plans (442 launches), half-precision GEMMs, and a full transformer layer in every precision with both attention kernels; also the fallback plan rule at other GPU core counts |
| `test_runtime.py`: the Runtime API ([section 4](#4-the-runtime-api)) | `runtime.rs`: the same checks; the end-to-end generations run on both backends | (shared with mlx-rs in `runtime.rs`) |

That gives 128 Rust tests, including 15 unit tests inside the crate (NPY parsing, the model
contract, ops, bench statistics), plus one ignored CPU benchmark of the runtime.

Porting `compiled = eager` found a real bug: compiled mlx-rs runs could flip pose tokens. MLX 0.32.2
(the version mlx-rs links) writes scalar constants into compiled kernels with too few digits, which
broke the fallback attention MLX uses for the root networks' head dim 32. The port now runs that
fallback itself with the scale stored as an array. Compiled and eager are bitwise identical again,
as in Python.

Results:
- **Metal runtime, FP32:** every sub-network stays within ≤ 1.2e-5 relative, and durations and pose
  tokens are identical. End to end, SOMA rotations are within ≤ 9.4e-4 and G1 `pred_qpos` within
  ≤ 1.6e-4.
- **FP16, BF16 and INT8:** the Python port's criteria apply: at least 75% of durations agree with FP32,
  and the median deviation is below 0.05. On the golden requests they pick the same duration as FP32
  on 15–16 of 16 requests. The median deviation is 1–3e-3 for FP16, 1–2e-2 for BF16 and 0.7–3.5e-3
  for INT8. That matches the Python port at the same precision: on the SOMA `random` request, BF16
  deviates by 4.4e-2 in Rust and 4.5e-2 in Python.

### Exporting the ONNX graphs (`tools/onnx`)

`tools/onnx` ports NVIDIA's ONNX exporter to Rust: it emits the same `mmm_v6_{g1,soma}_fp32`
graphs (and the mixed-FP16 variants) from the verified checkpoints, without PyTorch or the
pinned export environment. The exporter builds the graph statically — every dimension except
the batch size is fixed by the architecture — so it needs no tracing and produces much leaner
graphs. The CLI mirrors upstream's:

```sh
(cd tools/onnx && cargo build --release)
tools/onnx/target/release/motionbricks-onnx --model soma --checkpoints checkpoints/MotionBricks/soma
```

Against upstream's torch-tracing exporter on this machine: **2.3–2.5 s vs 3.7–3.8 s** per
export at **2.2–2.3 GB vs 3.4–3.6 GB** peak RSS, emitting **1,726/2,447 nodes instead of
6,993/8,662** (the traced graphs spend most nodes on run-time shape arithmetic), which also
makes ONNX Runtime sessions load faster (0.30–0.32 s vs 0.39–0.52 s). ONNX Runtime runs of
both graphs on identical inputs agree to FP32 accumulation noise (≤ 1.4e-4 on the motion
outputs, durations exact), and the FP16 conversion halves the same initializers as upstream's
`precision.py`. Details: [tools/onnx/README.md](../tools/onnx/README.md); correctness:
`sandbox/tests/test_onnx_rs.py`, benchmark: `tools/onnx/bench_export.py`.

### The applications (`app-rs`)

`app-rs` ports the four upstream applications to Rust: the **Inbetweener** and **Scene
Preview** demos and the **Smart Primitive** and **Scene** editors. They run on a small
wgpu viewer (`src/viewer/`, with the same logging-shaped API as upstream's Newton
ViewerGL) and — unlike upstream, which needs an NVIDIA GPU with TensorRT — they call **the
Rust model and Rust Runtime API** (`model-rs`): the Metal runtime behind
`motionbricks_rs::runtime::Backend`, plus the Runtime API's engines, motion players,
target-pose placement and clip IO. No Python, TensorRT, Warp or ONNX Runtime is involved.

```sh
cd app-rs && cargo build --release     # the first build also compiles MLX (~6 min)
cargo run --release --bin inbetweener -- --rig soma
cargo run --release --bin scene-preview -- --scene assets/scenes/soma_gym.scene.json
cargo run --release --bin smart-primitive-editor -- assets/smart_primitives/soma/sit.smart_primitive.json
cargo run --release --bin scene-editor -- --scene assets/scenes/soma_gym.scene.json
```

The scene and Smart Primitive documents are byte-compatible with upstream's
(`motionbricks.scene` / `motionbricks.smart_primitive.authoring` v1, tested round-trip),
so scenes and primitives authored in one implementation open in the other. Clips come
from `checkpoints/motions` (the `motionbricks-mlx download --clips` set) or the upstream
checkout via `--assets` — the sample clips stay under their NVIDIA Sample Data Evaluation
License and are never committed (see [app-rs/assets/README.md](../app-rs/assets/README.md)).
For tests and screenshots the viewer also has an offscreen headless mode
(`viewer::HeadlessViewer`; `viewer-demo --headless out.ppm --frames 30`).
Details and per-app notes: [app-rs/README.md](../app-rs/README.md).

---

## 4. The Runtime API

The model generates one transition at a time. Upstream's Runtime API is the library that turns
those transitions into continuous character motion, and both ports include it, with upstream's
names and behavior:

| | Python | Rust |
|---|---|---|
| Module | `motionbricks_mlx.runtime` | `motionbricks_rs::runtime` |
| Poses | NumPy float32 arrays | `Transform` and `Vec<f32>` |
| Inference | the MLX model, any precision | a `Backend`: the Metal runtime (any precision), the mlx-rs port, or your own |
| Guide | [model/docs/runtime_api.md](../model/docs/runtime_api.md) | [model-rs/docs/runtime_api.md](../model-rs/docs/runtime_api.md) |

What it covers:
- **Engines** for SOMA and the G1. They align the context and target to a heading, encode the
  model inputs, run inference, decode back to world space, drop the boundary frames and correct the
  endpoints.
- **Tokens and animation buffers:** four-pose context and target tokens, and buffers with time
  sampling, root offsets and foot contacts.
- **Chaining:** `MotionRecorder` keeps the poses you showed, and `GeneratedMotionPlayer` /
  `GeneratedDOFMotionPlayer` play generated motion and chain the next generation. A generation that
  fails leaves the player on its previous motion.
- **Targets:** placement on the ground with a facing direction, speed scaling, and SOMA IK pins
  that ease hands and feet onto the target. Your app applies the pins each frame, as upstream's demo
  apps do ([how](../model/docs/runtime_api.md#target-pose-ik)).
- **Clips:** BVH (SOMA) and CSV (G1) loading, and model-space conventions (`model_to_runtime`).

Each generation takes four context poses and one to four target poses. The model chooses 6 to 16
tokens, which gives 18 to 58 valid frames at 30 Hz (0.6 to 1.9 s).

```python
from motionbricks_mlx import runtime as rt

with rt.MotionBricksUnitreeG1Engine.from_checkpoints("checkpoints/MotionBricks/g1", dtype="bfloat16") as engine:
    context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
    context.update_from_poses(context_joint_q)          # four poses of 36 joint coordinates
    target.update_from_poses(target_joint_q)            # one to four poses
    engine.generate_animation(context, target, output)  # output.num_frames valid frames at 30 Hz
    pose = output.sample(0.5)
```

```rust
use motionbricks_rs::{metal::model::MetalModel, runtime::*};
let model = MetalModel::from_checkpoints_with("unitree_g1", "checkpoints/MotionBricks/g1", 1, "bfloat16")?;
let mut engine = MotionBricksUnitreeG1Engine::new(Some(model), Transform::IDENTITY)?;
let (mut context, mut target, mut output) = (engine.create_input(), engine.create_input(), engine.create_output());
context.update_from_poses(&context_joint_q)?;
target.update_from_poses(&target_joint_q)?;
engine.generate_animation(&context, &target, &mut output, &MotionBricksInferenceConfig::default())?;
```

Both ports include upstream's two examples:
- **`generate_once`** generates from clip poses, then chains a second generation from the last
  four poses it played.
- **`locomotion_loop`** steers a jog around a left turn with a motion player.

```sh
uv run --package motionbricks-mlx motionbricks-mlx download --clips
uv run --package motionbricks-mlx python model/examples/locomotion_loop.py --model soma --dtype bfloat16
(cd model-rs && cargo run --release --example locomotion_loop -- unitree_g1 float32)
```

On the Metal runtime, a whole generation takes about 7 ms in FP32 and 5 ms in BF16 at batch 1. That
includes encoding, inference, decoding and endpoint correction.

How it is verified:
- **Fixtures:** `sandbox/scripts/make_runtime_golden.py` runs upstream's own runtime code on Warp's
  CPU backend, with NVIDIA's PyTorch model in place of TensorRT, and records four generations per
  rig.
- **Step by step:** both ports match encoding, decoding, endpoint correction, clip loading,
  recorder resampling, target placement and speed, and buffer sampling and looping within 2e-5,
  and IK pins within 5e-5.
- **End to end:** every generation picks upstream's duration, and its poses stay within 3e-3.

Not ported: upstream's viewer, renderers, blend spaces, Smart Primitives and demo apps.

---

## 5. Benchmarks

All numbers come from a **Mac mini (2024, model `Mac16,11`) with an Apple M4 Pro**:
- 14-core CPU (10 performance + 4 efficiency cores), 20-core GPU, 64 GB of unified memory;
- macOS 27.0 (build 26A428), Python 3.12.11 with MLX 0.32.4, Rust 1.100 with mlx-rs 0.34 (MLX
  0.32.2), PyTorch 2.15.0 and ONNX Runtime 1.32.0;
- a quiet machine (load average 0.6–4), measured on 2026-11-21.

### Across backends

Median time per batched generation, with requests/s in parentheses:

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
  between them is paired. The next table has every batch size and FP16.
- **Other rows:** the Python harness `model/benchmarks/bench.py`
  ([full tables](../model/benchmarks/README.md)). They were measured for the previous release on the
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

### Python MLX vs the Rust Metal runtime

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
  [`model-rs/bench/results/report_metal.md`](../model-rs/bench/results/report_metal.md).
- **Streams of single requests:** the Metal runtime sustains 193–194 req/s in FP32 and 247–251 req/s
  in BF16/FP16 without pipelining. Pipelined Python reaches 149–153 and 198–199.

What changed since the first Rust release, which was FP32 only and 0.97× Python's speed at N=32:
- **FP16, BF16 and INT8 kernels**, with the same modes as the Python port.
- **Attention on simdgroup matrices:** 1.21 → 0.48 ms per pose transformer at N=32.
- **Tile-aligned fast loaders for convs, concatenations and masks:** −1.2 ms of GEMM time at BF16
  N=32.
- **A tuned GEMM plan per launch:** up to −7% of GEMM time at N=8.
- **Chunked command-buffer commits:** the GPU starts while the CPU is still encoding, which saves
  0.16–0.25 ms per call (4–6% at N=1).
- **No MLX kernels at run time:** the steel GEMMs the first release borrowed from `mlx.metallib`
  are replaced by its own kernel, which matches them.

### Running the benchmarks

| What | Needs | Command |
|---|---|---|
| Python MLX at chosen batch sizes | the Python model | `uv run --package motionbricks-mlx motionbricks-mlx bench --model soma --checkpoints checkpoints/MotionBricks/soma --batch 1 8 32` |
| both Rust runtimes (mlx-rs and Metal) | the Rust model | `cd model-rs && cargo run --release --bin mbrs-bench -- --model soma --dtype bfloat16 --batches 1 8 32` |
| Python vs Rust, interleaved rounds on byte-identical requests | both models; the Python side needs only the model package, synced once before the first round (`uv sync --inexact --package motionbricks-mlx`, which leaves an installed sandbox alone) | `model-rs/bench/run_all.sh 3 "float32 bfloat16 float16 int8"` (raw JSON in `model-rs/bench/results/raw`, or `OUT=...`), then `python3 model-rs/bench/report.py model-rs/bench/results/raw --table metal` (or `--table mlx-rs`) |
| one quick paired pass | the same; the runtime that goes first alternates from row to row | `python3 model-rs/bench/quick_compare.py --dtypes float32 bfloat16 --batches 1 8 32` |
| every backend: MLX, PyTorch MPS and CPU, ONNX Runtime | the sandbox | `uv run --all-packages python model/benchmarks/bench.py` (JSON in `model/benchmarks/results/`), then `model/benchmarks/report.py` for the tables |

**At large batches, compare only interleaved pairs.** Sustained GPU load raises the M4 Pro's GPU
time at N=32 by up to 10% within a few minutes. Separate runs of two runtimes therefore measure
the machine's temperature as much as the code. `run_all.sh` alternates the order every round, and
the reports pair runs within a round.

---

## 6. Tuning and profiling

### Other Apple silicon Macs (not tested)

> [!IMPORTANT]
> Every number in this guide comes from one Mac: a Mac mini with an M4 Pro (20-core GPU, 273 GB/s).
> The adaptations below let the code run sensibly on M2, M3, M4 and M5 chips in base, Pro, Max and
> Ultra form. **They have not been tested or benchmarked on any of those Macs.** On this M4 Pro
> they change nothing: the same plans, the same speed.

**What adapts by itself:**
- **Python port and mlx-rs port:**
  - MLX picks its own kernels for each GPU.
  - On M5 chips, MLX 0.30 and later uses the GPU's Neural Accelerators on macOS 26.2 and later. Both ports link MLX 0.32.
  - The custom rig kernels launch one thread per row or element, so they don't depend on the core count.
- **Metal runtime (`model-rs`): what it reads at load.** It reads the GPU's name, architecture, Apple
  GPU family and core count. Metal exposes neither the core count nor the memory bandwidth. The core
  count therefore comes from the IORegistry (`gpu-core-count`, the number System Information shows),
  and `MB_GPU_CORES=<n>` overrides it. `mbrs-bench` (on stderr) and `gemm-tune` print what they found,
  for example: `GPU: Apple M4 Pro (applegpu_g16s, Apple9 family, 20 GPU cores from IORegistry); GEMM
  plans: built-in table plans/apple_m4_pro.txt`.
- **Metal runtime: how GEMMs are planned.**
  - The built-in plan table is used on every M4 Pro, the 16- and 20-core versions alike.
  - Elsewhere, the fallback rule sizes each GEMM grid by threadgroups per GPU core: at least 8, and 6
    after split-K (16 and 12 with INT8 weights).
  - It allows up to 8 K-splits on GPUs of up to 20 cores, 16 up to 40 cores and 32 beyond. A GEMM of a
    single request can then still occupy an Ultra.
  - On a 20-core GPU, this reproduces the fitted plans exactly.
- **Metal runtime: what it checks at load.**
  - The GPU is Apple7 (M1) or later.
  - It has 32 KB of threadgroup memory.
  - Its SIMD-groups are 32 wide.
  - Every pipeline accepts the 256-thread threadgroups the dispatches use.

  Each of these holds on every Apple GPU family through Apple11 ([Metal feature set
  tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)). If one ever fails, loading
  stops with an error instead of producing wrong results.
- **Nothing in the code depends on memory bandwidth.**

**What to expect (inferred, not measured):**
- **Single requests:** these are bound by memory traffic and launch latency, so their latency
  should scale roughly with memory bandwidth more than with core count.
- **Large batches:** these scale more with GPU cores.
- **Metal runtime vs Python MLX:** the 2× advantage at batch 1 was measured on this M4 Pro only. On
  M5 chips, MLX's GEMMs use the Neural Accelerators and the Metal runtime's do not: its
  simdgroup-matrix kernels don't use Metal 4 tensor operations. So MLX may close the gap at large
  batches there. `mbrs-bench` times both runtimes side by side.
- **Plans tuned for your Mac:** for the best Metal runtime plans on another Mac, measure a table for
  it (next section). That replaces the fallback rule with plans timed on your GPU.

| Chip | GPU family | GPU cores | Memory bandwidth |
|---|---|---|---|
| [M2](https://support.apple.com/111867) / [Pro, Max](https://support.apple.com/111340) / [Ultra](https://support.apple.com/111835) | Apple8 | 8–10 / 16–19, 30–38 / 60–76 | 100 / 200, 400 / 800 GB/s |
| [M3](https://support.apple.com/118551) / [Pro, Max](https://support.apple.com/117736) / [Ultra](https://support.apple.com/122211) | Apple9 | 8–10 / 14–18, 30–40 / 60–80 | 100 / 150, 300–400 / 819 GB/s |
| [M4](https://support.apple.com/122209) / [Pro, Max](https://support.apple.com/121553) | Apple9 | 8–10 / 16–20, 32–40 | 120 / 273, 410–546 GB/s |
| [M5](https://www.apple.com/macbook-air/specs/) / [Pro, Max](https://www.apple.com/macbook-pro/specs/) / [Ultra](https://www.apple.com/mac-studio/specs/) | Apple10 | 8–10 / 16–20, 32–40 / 64–80 | 153 / 307, 460–614 / 1,200 GB/s |

There is no M4 Ultra. M6 chips (Apple11) are detected the same way, also untested.

### Tuning the Metal runtime for another Mac

The GEMM plan table (`model-rs/src/metal/plans/apple_m4_pro.txt`) is used only on GPUs named
"Apple M4 Pro". It was measured on the 20-core GPU. Other Macs use the fallback rule scaled to
their core count (previous section). On the M4 Pro, that rule makes GEMMs 1–7% slower than the
table. To measure a table for your own GPU:

```sh
uv run --package motionbricks-mlx python model-rs/bench/make_requests.py   # the requests gemm-tune replays
cd model-rs
for p in float32 float16 bfloat16 int8; do for m in soma unitree_g1; do
  TUNE_TABLE=$PWD/my_mac_plans.txt cargo run --release --bin gemm-tune -- $m $p 1 2 4 8 16 32
done; done
MB_PLAN_TABLE=$PWD/my_mac_plans.txt cargo run --release --bin mbrs-bench -- --model soma --dtype bfloat16 --batches 1 8 32
```

How the tuning works:
- **Measurement:** `gemm-tune` replays every distinct GEMM launch of a generation, with its real
  shapes and fused loaders, under all tile and split plans. It prints the best plans next to the
  table's and the rule's choices.
- **The table:** with `TUNE_TABLE`, each run appends its best plans to the file. When a launch
  appears more than once, its last line wins, so re-tuning into the same file overrides older plans.
- **Using it:** `MB_PLAN_TABLE=<file>` makes the runtime use that file, and `MB_PLAN_TABLE=0`
  turns tuned plans off (an empty value counts as unset). Loading checks every line (16 fields,
  a compiled tile, at least one split) and fails with the line number of a malformed one.

To make a table the default for a GPU, add it next to `apple_m4_pro.txt` and select it in
`plan_table` (`src/metal/transformer.rs`).

### Profiling tools

In `model-rs` (`cargo run --release --bin <tool> -- ...`; the two development tools also need
`--features dev-tools`):
- `gemm-tune <model> <precision> [N...]`: the GEMM plans, as above; `gemm-tune --list ...` only
  lists the GEMM launches of a generation;
- `metal-stages <precision> [N...]` (dev-tools): GPU time per stage of the Metal runtime;
- `metal-precision <model> <precision>` (dev-tools): an FP16/BF16/INT8 run against FP32, stage by
  stage, on the sandbox's `random` golden case;
- `bench/stage_profile.py`: GPU time per stage of the MLX graph (reads the golden fixture files);
- `bench/bench_python.py`: each Python MLX call split into input prep, graph build, `mx.eval`
  (wall and CPU time) and host copy; `--op-cost` measures Python's cost to build one lazy MLX op.

### Lessons from profiling

These apply to both versions:
1. **N=1 is GPU-bound, not CPU-bound.** A Metal System Trace shows the GPU busy 99% of a call.
   About 3 ms of that is weight streaming, and the rest is the latency of about 500 small,
   dependent kernels.
2. **The host language doesn't matter.** Rust and Python match on MLX, because MLX's C++ core
   does the work.
3. **What pays off:** fewer, better-shaped kernels (the Metal runtime), smaller weights
   (BF16/INT8), batching, and pipelining.
4. **At large batches, measure in pairs**, because of the thermal drift described above.

---

## 7. The development sandbox (`sandbox/`)

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

CuMetal was rechecked for this release:
- **Toolchain:** Xcode 27.2 (27B52), Metal Toolchain 27B47, Homebrew LLVM 23.1.6 and CuMetal 0.8.0.
- **Tests:** all three CuMetal tests pass. They skip on machines without the toolchain.
- **Install:** the toolchain needs full Xcode; run `sandbox/native/install_cumetal.sh`.

Research notes, sources and limits are in [sandbox/README.md](../sandbox/README.md).

### Development workflow

```sh
sandbox/setup.sh                              # env + pinned upstream + verified checkpoints + ONNX + golden + runtime fixtures (rerun-safe; --force regenerates)
uv run --all-packages pytest                  # Python: 219 tests (181 model, 38 sandbox incl. 3 CuMetal)
(cd model-rs && cargo test --release -- --test-threads=1)   # Rust: 128 tests, both runtimes vs the fixtures
```

- **Golden fixtures:** `sandbox/scripts/make_golden.py` records the inputs and outputs of every
  sub-network from NVIDIA's PyTorch model on the CPU, for 5 request families per model. Both test
  suites teacher-force each stage with them, so an error localizes to one module. `meta.json`
  records the device and versions that produced them. `--device mps` writes to
  `sandbox/golden-mps/`, and `--out DIR` (in both fixture scripts) writes elsewhere, for example to
  check that a generator change keeps every array bit-identical.
- **Runtime fixtures:** `sandbox/scripts/make_runtime_golden.py` runs upstream's own Runtime API
  code on Warp's CPU backend: engine encode/decode/correction methods, recorder, target helpers,
  IK and clip loaders. NVIDIA's PyTorch model stands in for TensorRT. Both ports' runtime tests
  check each step against these fixtures.
- **Locations:** `MOTIONBRICKS_CHECKPOINTS`, `MOTIONBRICKS_GOLDEN`, `MOTIONBRICKS_RUNTIME_GOLDEN`
  and `MOTIONBRICKS_CLIPS` point both test suites at other locations.

---

## 8. License details

**Apache-2.0** ([LICENSE](../LICENSE)), the same license as upstream's code, so both ports stay
license-compatible with MotionBricks. [NOTICE](../NOTICE) covers the rest:
- **Upstream:** it carries NVIDIA's copyright and marks the modified files, including the Runtime
  API ports.
- **Inherited third-party code:** T2M-GPT, MotionGPT, vector-quantize-pytorch, and Daniel Holden's
  two-bone IK. Their licenses are in [licenses/](../licenses/).
- **Rust dependencies:** fetched by Cargo, not vendored.

The two Runtime API guides are adapted from upstream's `docs/runtime_api.md` under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each page carries the attribution and
a note of what changed.

Not included:
- **Weights:** under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  `motionbricks-mlx download`, `model-rs/scripts/download_weights.sh` and `sandbox/setup.sh` fetch
  them from [huggingface.co/nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks).
- **Sample motion clips:** under the NVIDIA Sample Data Evaluation License; downloaded on request,
  as above.
- **Robot meshes:** not redistributed either.
