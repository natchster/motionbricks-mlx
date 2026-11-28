# model-rs — MotionBricks in Rust

Two Rust implementations of [`model/`](../model) (`motionbricks_mlx`, the MLX port of NVIDIA's
MotionBricks). Both run the two released models, SOMA and Unitree G1, and both pass the same
golden parity tests against NVIDIA's PyTorch model as the Python port.

| Runtime | What it is | Speed vs the Python MLX port at the same precision (Mac mini M4 Pro) |
|---|---|---|
| `motionbricks_rs::MotionBricks` | an op-for-op port on [mlx-rs](https://github.com/oxiglade/mlx-rs) 0.34 (MLX 0.32.2): the same graph, the same kernels, and the shared rig kernels (`../model/motionbricks_mlx/assets/rig_kernels.metal`); FP32, FP16, BF16 | the same (0.99–1.01× the time, compiled) |
| **`motionbricks_rs::metal::model::MetalModel`** | **a model-specific Metal runtime** built from what MLX does generically: a static dispatch plan, its own fused kernels (GEMM + residual + LayerNorm in one reduce), and GEMM plans tuned per launch; FP32, FP16, BF16, INT8 | **2.12–2.31× at N=1**, 1.70–2.05× at N=2, 1.38–1.49× at N=4, 1.20–1.28× at N=8, 1.09–1.18× at N=16, 1.02–1.08× at N=32 |

The [user guide](../docs/user_guide.md#3-the-rust-implementations-model-rs) explains when to use which. The Metal runtime is tied to this model
implementation and to Apple-silicon GPUs, in exchange for the lowest latency.

Both also run upstream's **Runtime API** (`motionbricks_rs::runtime`). It provides engines,
endpoint correction, tokens, animation buffers, the motion recorder, motion players, target
placement, IK pins, and BVH/CSV clips. See [Runtime API](#runtime-api).

## Build, test, run

Neither runtime needs Python or the sandbox:

```sh
scripts/download_weights.sh                # checkpoints into ../checkpoints/MotionBricks (curl + shasum, verified)
cargo build --release                      # first build compiles MLX (~6 min); needs Xcode + its Metal toolchain
cargo run --release --example generate_metal -- unitree_g1 ../checkpoints/MotionBricks/g1 8 bfloat16
cargo run --release --example generate       -- soma ../checkpoints/MotionBricks/soma 8 int8
cargo test --release -- --test-threads=1   # 128 tests; the parity tests also need the sandbox's fixtures
```

- **Weights:** `download_weights.sh [all|soma|unitree_g1] [dir]` reads the same checksum table as
  the Python port's `motionbricks-mlx download`
  (`../model/motionbricks_mlx/assets/checkpoints.sha256`).
- **Clips:** `download_weights.sh clips [dir]` fetches the Runtime API examples' four sample clips
  into `../checkpoints/motions`. They come from upstream's Git LFS at the pinned commit and are
  verified against `../model/motionbricks_mlx/assets/clips.sha256`. They are NVIDIA sample data
  under the NVIDIA Sample Data Evaluation License.
- **Requests:** the examples take a batch size, which builds a synthetic request with
  `motionbricks_rs::examples` (the Rust version of the Python generator), or a request `.npz`.
- **Benchmarks** compare against Python on byte-identical `.npz` requests, so they need the Python
  environment (the model package only): `bench/run_all.sh 3 "float32 bfloat16"` and
  `python3 bench/report.py bench/results/raw --table metal`.

```rust
use motionbricks_rs::metal::model::MetalModel;
let model = MetalModel::from_checkpoints("unitree_g1", "checkpoints/MotionBricks/g1", 64)?;  // FP32, max batch 64
let inputs = motionbricks_rs::examples::request("unitree_g1", 4, 0)?;                       // or npy::load_npz(...)
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?;                       // ONNX input names
let fast = MetalModel::from_checkpoints_with("soma", "checkpoints/MotionBricks/soma", 64, "bfloat16")?; // or float16, int8
let forced = model.run_with_durations(&inputs, Some(&[8, 8, 12, 16]))?;                      // fixed durations

let mlx = motionbricks_rs::MotionBricks::load("soma", "checkpoints/MotionBricks/soma", "int8")?; // mlx-rs port; INT8 = Python's quantize(8)
motionbricks_rs::convert("soma", "checkpoints/MotionBricks/soma", "weights/soma.safetensors", mlx_rs::Dtype::Float32)?;
```

`.cargo/config.toml` sets `DEVELOPER_DIR` to Xcode and keeps mlx-sys's `mlx.metallib` (175 MB),
which the mlx-rs port needs, in `target/mlx-metal/`. The Metal runtime compiles its own kernels
when a model is loaded and needs nothing from MLX at run time. It still loads the checkpoints and
quantizes INT8 weights with mlx-rs.

## Runtime API

`motionbricks_rs::runtime` ports upstream's Runtime API with the same behavior as the Python port's
`motionbricks_mlx.runtime`. Its engines run on a `Backend`: `MetalModel` (any precision),
`runtime::MlxBackend` (the mlx-rs port), or your own implementation.

```rust
use motionbricks_rs::{metal::model::MetalModel, runtime::*};
let model = MetalModel::from_checkpoints_with("unitree_g1", "../checkpoints/MotionBricks/g1", 1, "bfloat16")?;
let engine = MotionBricksUnitreeG1Engine::new(Some(model), model_to_runtime(ModelRig::UnitreeG1))?;
let mut player = GeneratedDofMotionPlayer::new(engine, &reference_joint_q, 32)?;
player.seed_history_from_pose(&start_joint_q)?;
player.generate(&place_target_dof(ground, direction, &jog), &TargetPoseInferenceParams::new(4, 4), None)?;
let (joint_q, foot_contacts) = player.sample(1.0 / 60.0)?;   // each update: sample, change, record
player.record_pose(1.0 / 60.0, &joint_q)?;
```

```sh
scripts/download_weights.sh clips
cargo run --release --example generate_once   -- soma bfloat16
cargo run --release --example locomotion_loop -- unitree_g1 float32
```

On the Metal runtime, a whole generation (encode, inference, decode, endpoint correction) takes
about 7 ms in FP32 and 5 ms in BF16 at batch 1. `locomotion_loop` makes 14 generations in 9 s of
steered motion. The guide is [docs/runtime_api.md](docs/runtime_api.md).

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
The first release combined both kernel families, mine where M ≤ 32 and MLX's where M is large.
The current one uses only its own kernels: `gemm_mma` was restructured like steel (below) and now
matches or beats it at every size, so convolutions, masks and activations stay fused at large
batch sizes too.

## 3. The Metal runtime (`src/metal/`)

- **A static plan:** one generation is a fixed list of 236–350 dispatches (most at N=1, where
  split-K adds reduce passes). They are recorded into serial command buffers and committed in
  chunks (after 8, 40 and 120 dispatches), so the GPU starts on the first stage while the CPU
  encodes the rest. That saves 0.16–0.25 ms per call over one command buffer. Encoding takes
  0.11–0.16 ms, versus MLX's ~2 ms. There is no graph or allocator: every buffer is allocated at
  load time, and so is every GEMM pipeline variant the plans can pick for batch 1..=`max_batch`
  (each batch is recorded once and discarded unrun), so no generation compiles a kernel.
- **One element type `T`** (`float`, `half` or `bfloat`) for all network tensors, chosen when the
  kernels are compiled. Accumulation, softmax, LayerNorm statistics, masks and the rig are float.
  **INT8** keeps FP32 activations and stores the transformer projections in MLX's affine format
  (8 bits, groups of 64), exactly like the Python port's `quantize(8)`.
- **`gemm_mma`** (`kernels.metal`): one simdgroup-MMA template with 11 tile shapes from 16×32 to
  64×64, and split-K with a reduce. It is structured like MLX's steel GEMM:
  - simdgroup tiles interleaved across the threadgroup tile;
  - fragments loaded lane-wise from threadgroup memory (stored as `T`, accumulated as float);
  - serpentine multiply-accumulate order;
  - per-thread load pointers fixed for the whole K loop.

  Function constants specialize the operand loader per launch, so most of MLX's ~500 kernels fold
  into this one:
  - **A side:** implicit dilated Conv1d with 2× nearest upsampling, concatenation of up to three
    sources, ReLU and row masks. When the tile is aligned, the conv tap and concat segment are
    resolved once per K tile instead of per element;
  - **B side:** dense `T` weights, or INT8 dequantized on the fly (`w = scale · q + bias`, one
    group per K tile);
  - **output side:** ReLU / LeakyReLU, residual with a row modulus (position tables), output mask,
    predicated row writes (the decoder's "overwrite the target frames") and the root network's
    "missing-embedding blend".
- **GEMM plans:** `plans/apple_m4_pro.txt` holds the measured best tile and split count for every
  GEMM launch of both models at N = 1, 2, 4, 8, 16 and 32, per precision (843 entries).
  `gemm-tune` measures them by replaying each launch with its real shape and loader flags. On
  other GPUs, or for other batch sizes, a rule fitted on the same data picks the plan:
  - 32×64 tiles for FP32, 16×32 for FP16/BF16 and 64-row tiles for INT8 weights;
  - then K splits until the grid fills the GPU, counted in threadgroups per GPU core. The core
    count comes from the IORegistry (`MB_GPU_CORES` overrides it). That scaling is untested on
    Macs other than the M4 Pro; see the
    [user guide](../docs/user_guide.md#other-apple-silicon-macs-not-tested).

  `MB_PLAN_TABLE=0` disables the table, and `MB_PLAN_TABLE=<file>` loads a table tuned on
  another Mac. Every line is checked (malformed ones fail with their line number); a launch
  listed twice uses its last line. The split-K partial buffer is sized at load from `max_batch`
  and the plans' split counts, so every plan resolves as with the 64 MB maximum.
- **Attention** on simdgroup matrices: one threadgroup per (sequence, head) and one simdgroup per
  8 queries. It reads Q/K/V straight from the packed QKV projection, and computes the masked
  softmax in threadgroup memory. At N=32 it takes 0.48 ms for the pose transformer's 16 layers,
  against 1.21 ms for the earlier one-lane-per-key kernel.
- **Fused normalize** (`gemm_reduce_ln`, `kernels.metal`): the post-norm transformer's
  `Linear → +residual → LayerNorm` pairs (48 of them at N=1) run as two dispatches instead of
  three: the GEMM always writes split-K partials (the `fc_partial` function constant forces the
  partial path even for plans that don't split K), then one row-wise reduce per pair sums the
  partials, adds the bias and residual, and LayerNorms the row — 1024 threads per row, float4
  loads, threadgroup reductions. That saves the dispatch and the intermediate tensor's round trip
  per pair, worth ~3–7% of single-request GPU time (and it rounds the GEMM output *less* often
  than the unfused order). Two alternatives measured and rejected: forcing split-K off loses more
  in the GEMM than the reduce saves, and a 256-thread row kernel can't stream the partials fast
  enough (the reduce dominated the fusion).
- **Model kernels** (`model.metal`), one thread per item:
  - G1 forward kinematics and feature extraction (one thread per pose per sequence — 8 per
    sequence — so a single request runs 8 threads instead of 1; the local-root rows, which need
    the next pose's heading, follow in the tiny `g1_preprocess_lr` dispatch);
  - joint-angle decoding with `matrix_quat`;
  - the SOMA T-pose retarget;
  - normalization, the duration head and argmax, dense conditions, `global_root_to_local` with
    the last-frame patch;
  - pose argmax with the codebook lookup, root integration with SOMA's height-violation rule, and
    re-anchoring.
- **Constant folding:** in the release configuration (one MaskGIT step, every input token
  masked), the pose network's token-embedding MLP has a constant input. Its output is computed
  once at load time, rounded through `T` like the network would.

GPU time per stage (ms, `metal-stages`):

| SOMA | N=1 FP32 | N=1 BF16 | N=1 INT8 | N=32 FP32 | N=32 BF16 | N=32 INT8 |
|---|---|---|---|---|---|---|
| input, glue and rig | 0.15 | 0.26 | 0.14 | 0.18 | 0.18 | 0.18 |
| root network | 1.51 | 1.17 | 1.26 | 9.35 | 8.10 | 9.54 |
| pose network | 2.94 | 2.07 | 1.95 | 24.20 | 20.78 | 25.61 |
| VQ-VAE decoder | 0.47 | 0.39 | 0.46 | 4.87 | 4.40 | 4.86 |
| **whole generation** | **4.97** | **3.84** | **3.83** | 38.64 | 33.47 | 40.31 |

G1 differs by at most 0.3 ms: its rig costs 0.1–0.2 ms more and its root network 0.14–0.26 ms
less. At N=1 the pose network streams its 512 MB of FP32 transformer weights in 2.9 ms (about
180 GB/s); BF16 and INT8 halve and quarter that traffic. At N=32 every precision is
compute-bound, and INT8 pays for dequantizing inside the GEMM.

## 4. Correctness

The test files mirror the Python suite (`../model/tests`) one to one, for both runtimes, with the
same requests, checks and tolerances:

| File | Counterpart | Tests | Needs |
|---|---|---|---|
| `tests/parity.rs` (mlx-rs) | `test_parity.py` | root, pose, VQ-VAE, end to end (eager + compiled), × 2 models, each over all 5 golden cases | checkpoints + golden fixtures |
| `tests/metal_parity.rs` (Metal) | `test_parity.py` | the same 8; each stage teacher-forced through the runtime's buffers | checkpoints + golden fixtures |
| `tests/model.rs` (mlx-rs) | `test_model.py` | ONNX contract, compiled = eager, batch independence, allowed durations, forced durations, FP16/BF16 vs FP32, converted-file round trip, multi-step decoding, INT8 vs FP32; × 2 models | checkpoints |
| `tests/metal_model.rs` (Metal) | `test_model.py` | the same, with "chunked commits = one command buffer, runs repeat bitwise" for compiled = eager and "oversized batches are refused" for multi-step decoding, which the release-only plan doesn't run; plus re-encoding without a new upload, malformed inputs, an untuned batch size at `max_batch`, no pipeline compiles after loading, and the fitted plan rule against the table; × 2 models | checkpoints |
| `tests/layers.rs` (mlx-rs) | `test_layers.py` | sincos table, transformer layer vs float64 PyTorch semantics (fused and fallback attention), ResConv masking and residual, decoder shapes and target override, codebook lookup and digits, rotation round trips, root integration, FP16/BF16 attention, rig kernels vs ops | nothing |
| `tests/metal_kernels.rs` (Metal) | none (Python runs MLX's kernels) | every `gemm_mma` loader and epilogue under all 33 tile/split plans (442 launches) vs a CPU reference; FP16/BF16 GEMMs; a transformer layer in all 4 precisions with both attention kernels vs float64; plan-table parsing (malformed lines, the built-in table in full), buffer type and bounds checks, the INT8 loader guard | nothing |
| `tests/runtime.rs` (Runtime API) | `test_runtime.py` | 6 weight-free (transforms, tokens, recorder resampling, buffer sampling, config, player timing with a fixed backend); 13 parity with upstream's runtime code (encode, decode + endpoint correction, clips, recorder, target placement and speed, buffer sampling and looping, × 2 models, plus IK pins); 4 end to end (Metal and mlx-rs × 2 models), each over all 4 recorded generations | weight-free: nothing; parity: the runtime fixtures and clips; end to end: also checkpoints |

`MOTIONBRICKS_CHECKPOINTS`, `MOTIONBRICKS_GOLDEN`, `MOTIONBRICKS_RUNTIME_GOLDEN` and
`MOTIONBRICKS_CLIPS` override the locations. Tests whose inputs
are missing print `SKIPPED` on stderr and pass, because Rust's harness has no skip state;
`MOTIONBRICKS_REQUIRE_FIXTURES=1` turns every skip into a failure.

Porting `test_compiled_equals_eager` exposed a real difference: in the mlx-rs port (and only
there), compiled runs differed from eager ones and could flip G1 pose tokens. MLX 0.32.2, which
mlx-rs 0.34 links, writes scalar constants into compiled kernels with too few digits. Its
composite attention fallback, used for head dims the fused kernels lack (the root networks' 32),
multiplies by such a scalar. `layers.rs` now runs that fallback itself, op for op, with the scale
stored as a `(head_dim,)` array. Eager results are bitwise unchanged, and compiled = eager holds
bitwise, as in Python (`tests/model.rs` checks it bit for bit).

Golden-fixture parity of the Metal runtime:

| Metal runtime (FP32) vs PyTorch FP32 | SOMA | G1 |
|---|---|---|
| root duration logits / root path (relative) | ≤ 1.3e-6 / ≤ 2.5e-6 | ≤ 3.0e-6 / ≤ 2.2e-6 |
| pose logits (relative), argmax tokens | ≤ 7.4e-6, identical | ≤ 5.8e-6, identical |
| VQ-VAE decoder (relative) | ≤ 4.4e-6 | ≤ 1.0e-5 |
| durations, end to end | identical, 5/5 | identical, 5/5 |
| outputs, end to end (absolute) | root ≤ 3.3e-6, features ≤ 9.7e-5, rotations ≤ 9.4e-4 | `pred_qpos` ≤ 1.6e-4, contacts ≤ 1.8e-3 |

FP16, BF16 and INT8 are checked against the FP32 Metal runtime with the Python suite's criteria:
at least 75% of durations agree, and the median deviation is below 0.05 where they agree.

| vs FP32 (all golden requests) | SOMA | G1 |
|---|---|---|
| FP16: durations agree, median deviation | 16/16, 1.4e-3 | 16/16, 2.7e-3 |
| BF16 | 16/16, 1.9e-2 | 16/16, 1.4e-2 |
| INT8 weights | 15/16, 7.4e-4 | 16/16, 3.5e-3 |

These are the same magnitudes as the Python port at the same precision. On the SOMA `random`
request, `metal-precision` measures a BF16 deviation of 4.4e-2 where Python's is 4.5e-2, and both
pick the same durations.

## 5. Benchmarks

Machine: Mac mini (2024, `Mac16,11`), Apple M4 Pro (14-core CPU, 20-core GPU), 64 GB, macOS 27.0.
`bench/run_all.sh 3 "float32 bfloat16 float16 int8"` ran 3 interleaved rounds per precision on
2026-11-27, on a quiet machine (load median 2.3). The order of Python and Rust alternates every
round. Sustained load raises the GPU time at N=32 by up to 10% within minutes, so only paired
ratios are meaningful there. Times are the median per batched call, end to end: inputs from the
host, outputs back on the host. Each cell gives Python MLX (compiled) → **Metal runtime** in ms,
and the paired speedup. Per-round ranges, GPU times and the mlx-rs port are in
[`bench/results/report_metal.md`](bench/results/report_metal.md).

#### SOMA

| N | FP32 | BF16 | FP16 | INT8 weights |
|---|---|---|---|---|
| 1 | 10.61 → **5.00** (2.12×) | 8.29 → **3.76** (2.25×) | 8.14 → **3.78** (2.15×) | 7.96 → **3.81** (2.26×) |
| 2 | 11.02 → **5.70** (1.93×) | 8.98 → **4.42** (2.05×) | 8.81 → **4.50** (1.97×) | 8.47 → **5.03** (1.72×) |
| 4 | 11.92 → **8.08** (1.48×) | 10.12 → **6.76** (1.48×) | 9.93 → **6.78** (1.47×) | 10.66 → **7.74** (1.38×) |
| 8 | 15.95 → **12.95** (1.23×) | 14.09 → **11.17** (1.28×) | 13.86 → **10.95** (1.26×) | 15.27 → **12.70** (1.21×) |
| 16 | 25.30 → **22.56** (1.12×) | 22.19 → **19.06** (1.18×) | 21.77 → **18.96** (1.15×) | 25.60 → **23.19** (1.11×) |
| 32 | 46.02 → **42.88** (1.07×) | 38.80 → **37.35** (1.04×) | 37.86 → **36.61** (1.03×) | 46.75 → **43.48** (1.08×) |

#### Unitree G1

| N | FP32 | BF16 | FP16 | INT8 weights |
|---|---|---|---|---|
| 1 | 10.50 → **4.88** (2.15×) | 8.25 → **3.64** (2.26×) | 8.22 → **3.57** (2.31×) | 7.85 → **3.71** (2.12×) |
| 2 | 10.88 → **5.59** (1.95×) | 8.78 → **4.36** (2.01×) | 8.73 → **4.39** (2.00×) | 8.32 → **4.88** (1.70×) |
| 4 | 11.84 → **7.96** (1.49×) | 9.93 → **6.68** (1.49×) | 9.88 → **6.71** (1.47×) | 10.60 → **7.63** (1.39×) |
| 8 | 15.82 → **13.08** (1.21×) | 13.74 → **10.76** (1.28×) | 13.72 → **10.85** (1.26×) | 15.23 → **12.55** (1.22×) |
| 16 | 25.38 → **23.31** (1.09×) | 21.90 → **19.00** (1.15×) | 21.64 → **19.64** (1.11×) | 24.79 → **22.49** (1.10×) |
| 32 | 46.03 → **43.67** (1.05×) | 38.17 → **36.08** (1.05×) | 38.44 → **37.83** (1.02×) | 45.60 → **42.57** (1.08×) |

- **Every round:** all 144 paired comparisons favour the Metal runtime at N ≤ 16 (2.12× or more
  at N=1). At N=32 both runtimes are GPU-throughput-bound and the pair is within noise
  (0.98–1.08×).
- **Streams of single requests:** the Metal runtime sustains 200–205 req/s in FP32 and 265–280 in
  BF16/FP16 sequentially. Python and mlx-rs reach 95 (FP32) or 120–123 (BF16/FP16)
  sequentially, and 151–155 or 185–200 with `async_eval` pipelining.
- **mlx-rs vs Python** ([`bench/results/report_mlx-rs.md`](bench/results/report_mlx-rs.md)):
  compiled, mlx-rs takes 0.99–1.01× Python's time; eager 0.97–1.01×, because Rust builds the eager
  graph in 0.3–0.4 ms against Python's 0.5–0.7 ms. The language does not matter once the graph is
  compiled.

## 6. Limits

- **Release configuration only.** It runs one MaskGIT step, with durations chosen by the model or
  fixed per request (`run_with_durations`). Multi-step decoding and other options remain
  available in the mlx-rs port.
- **Batch size** is capped by `max_batch` at construction (buffers are preallocated). Requests
  with more rows, or with inputs whose shapes disagree, are refused with an error. Loading
  compiles the GEMM pipelines of every batch size up to `max_batch`, so a large `max_batch`
  loads more slowly.
- **Tuned on one GPU.** The plan table was measured on an M4 Pro (20-core GPU) and is used on
  GPUs named "Apple M4 Pro". Other Apple GPUs
  use the fitted rule; here its GEMMs take 1–7% longer than the table's. Run `gemm-tune` with
  `TUNE_TABLE=<file>`, then `MB_PLAN_TABLE=<file>`, to tune another machine.
- **INT8 is a weight format, not faster math.** Like in MLX, the activations stay FP32 and the
  projections are dequantized inside the GEMM. INT8 matches BF16 at N=1 (4.0 ms), because weight
  streaming dominates there. From N=2 on, BF16 or FP16 is faster.

## Layout

| Path | Contents |
|---|---|
| `src/metal/mod.rs` | device, precision, kernel library (compiled with `typedef ... T`), buffers, recorder with chunked commits |
| `src/metal/kernels.metal` | `gemm_mma` (+ INT8 loader), split-K reduce, attention (simdgroup MMA and scalar fallback), add+LayerNorm |
| `src/metal/model.metal` | rig and glue kernels |
| `src/metal/transformer.rs` | GEMM planning (tuned table, fitted rule, split-K, kernel specialization), INT8 weights, transformer stack |
| `src/metal/plans/apple_m4_pro.txt` | tuned GEMM plans for the M4 Pro |
| `src/metal/weights.rs` | weight blocks (linear, conv, FC block, decoder), the folded token embedding, rig tables |
| `src/metal/model.rs` | load-time validation and warm-up, buffers, input checks, and the per-stage encoders of a generation |
| `src/runtime/` | the Runtime API: `math` (Warp-convention transforms), `contract`, `animation`, `tokens`, `engine` (`Backend`, both engines), `player`, `target_pose` (incl. two-bone IK), `io` (BVH, CSV, spaces) |
| `src/*.rs` | the mlx-rs port (`config`, `weights` incl. `save_converted`, `layers` incl. INT8 `Linear`, `networks`, `representation`, `inference`, `rig`, `model`); `kernels.rs` runs the shared rig kernels; `examples.rs` builds synthetic requests |
| `tests/` | see [Correctness](#4-correctness): `parity.rs`, `model.rs`, `layers.rs` (mlx-rs port); `metal_parity.rs`, `metal_model.rs`, `metal_kernels.rs` (Metal runtime); `runtime.rs` (Runtime API); `common/` (shared helpers, float64 references) |
| `scripts/download_weights.sh` | checkpoint and sample-clip download + SHA-256 verification without Python |
| `examples/` | `generate_metal.rs`, `generate.rs`: one generation from a batch size or a request `.npz`; `generate_once.rs`, `locomotion_loop.rs` (+ `support/runtime_rig.rs`): upstream's Runtime API examples |
| `docs/runtime_api.md` | the Runtime API guide |
| `src/bin/` | `mbrs-bench` (benchmark; `--runtime both\|mlx\|metal`), `gemm-tune` (replay and tune every GEMM launch, write plan tables; `--list` lists a generation's GEMMs); with `--features dev-tools`: `metal-stages` (GPU time per stage), `metal-precision` (FP16/BF16/INT8 vs FP32 per stage) |
| `bench/` | interleaved Python-vs-Rust harness (`run_all.sh`, `report.py --table metal\|mlx-rs`), `quick_compare.py` (one paired pass), reports and raw results |

## Sources

- [MLX](https://github.com/ml-explore/mlx) source (v0.32.2, read from the mlx-sys build):
  `backend/metal/{eval,device,matmul}.cpp`, `kernels/steel/gemm/*`
- [mlx-rs](https://github.com/oxiglade/mlx-rs) ([docs.rs](https://docs.rs/mlx-rs)), [objc2-metal](https://docs.rs/objc2-metal)
- [Apple: indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md),
  [MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer.md)
- [Kernel fusion in MLX: what `mx.compile` and `mx.fast.*` do](https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html)

License: Apache-2.0, like the repository. The rig assets embedded with `include_bytes!` are
NVIDIA's (Apache-2.0). The two-bone IK in `src/runtime/target_pose.rs` derives, through upstream,
from Daniel Holden's MIT-licensed code (`../licenses/Daniel-Holden-MIT-LICENSE.txt`). `gemm_mma` follows the tiling scheme of MLX's steel GEMM (MIT). mlx-rs is
MIT or Apache-2.0.
