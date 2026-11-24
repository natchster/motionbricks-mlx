# motionbricks-mlx

NVIDIA's [MotionBricks](https://github.com/NVlabs/motionbricks) (SIGGRAPH 2026) on Apple silicon.
MotionBricks generates real-time motion for characters and robots. You give it a character's last
few poses and one to four target poses, and it generates the motion in between, 0.6 to 1.9 s at
30 Hz. It runs for two rigs: the **SOMA** human skeleton and the **Unitree G1** humanoid.

This repository runs it on the Apple GPU in three ways:
- a **Python** port on MLX;
- a **Rust** implementation with a model-specific Metal runtime, about **2× faster** than MLX for
  single requests;
- upstream's **Runtime API** in both, which turns single generations into continuous motion.

> [!NOTE]
> **This is a quick port and may contain issues.** Both versions pass test suites that check them
> stage by stage against NVIDIA's own code, and a human has tried them too. If you find a
> discrepancy with upstream, please open an issue. This is an independent community project, not
> affiliated with or endorsed by NVIDIA.

| Folder | What it is |
|---|---|
| [`model/`](model/README.md) | **`motionbricks_mlx`**, the Python MLX port of both inference graphs plus the Runtime API. It has the same tensor interface as upstream's ONNX models, runs in FP32, FP16, BF16 or INT8, and depends only on `mlx` and `numpy`. |
| [`model-rs/`](model-rs/README.md) | **`motionbricks-rs`**: an op-for-op mlx-rs port and a **model-specific Metal runtime** (FP32, FP16, BF16, INT8), plus the Runtime API on either one. |
| [`sandbox/`](sandbox/README.md) | **`cudashim`**, a CUDA-emulation layer that runs NVIDIA's original PyTorch, TensorRT and Warp code on a Mac. It is the reference both ports are tested against, and it is only needed for development. |
| [`tools/onnx/`](tools/onnx/README.md) | **`motionbricks-onnx`**, NVIDIA's ONNX exporter in Rust: emits the same FP32/mixed-FP16 batched graphs as the torch-tracing original, from the verified checkpoints, without PyTorch — and with ~4× fewer nodes. |
| [`app-rs/`](app-rs/README.md) | **`motionbricks-app`**, the four upstream applications in Rust — Inbetweener, Scene Preview, Smart Primitive Editor, Scene Editor — on a small wgpu viewer. They call the Rust model and Runtime API (`model-rs`), never Python or TensorRT. |

It is ported from upstream v0.2.0 (commit `54fc563`) with the weights at revision `3cb362d`.

**[User guide →](docs/user_guide.md)**: installation paths, design, full benchmarks, correctness,
tuning and the sandbox.

## Performance

Median time per generation on a Mac mini with an Apple M4 Pro (20-core GPU), with requests/s in
parentheses:

| Backend | SOMA N=1 | SOMA N=32 | G1 N=1 | G1 N=32 |
|---|---|---|---|---|
| **Rust Metal runtime, BF16** | **4.05 ms** (247) | **35.21 ms** (909) | **4.04 ms** (248) | **35.42 ms** (903) |
| Rust Metal runtime, FP32 | 5.18 ms (193) | 42.91 ms (746) | 5.18 ms (193) | 43.72 ms (732) |
| Python MLX, BF16 | 8.25 ms (121) | 37.86 ms (845) | 8.30 ms (121) | 37.76 ms (848) |
| Python MLX, FP32 | 10.72 ms (93) | 45.18 ms (708) | 10.55 ms (95) | 45.38 ms (705) |
| upstream PyTorch on the GPU (MPS) | 32.08 ms (31) | 91.19 ms (351) | 37.04 ms (27) | 98.97 ms (323) |
| *RTX 4090 + TensorRT FP32 (upstream, GPU time only)* | *2.50 ms* | *5.28 ms* | *2.67 ms* | *7.81 ms* |

- **Metal runtime vs Python MLX:** the Metal runtime is faster in all 144 paired comparisons
  (2 models × 4 precisions × 6 batch sizes × 3 rounds). The margin runs from 2.06× at batch 1 down
  to 1.02× at batch 32.
- **Python MLX vs upstream:** Python MLX is 2.4–4.7× faster than NVIDIA's PyTorch model on the
  same GPU.

[All numbers, the setup and INT8/FP16](docs/user_guide.md#5-benchmarks) are in the user guide.

## Use the model directly

The model takes the same named tensors as upstream's ONNX files and returns the generated motion.
Each path below downloads the checkpoints (about 1.6 GB) and verifies every file's SHA-256. No
sandbox is needed.

**Python** (Python 3.12 and [uv](https://docs.astral.sh/uv/), or `pip install ./model`):

```sh
uv sync --package motionbricks-mlx
uv run --package motionbricks-mlx motionbricks-mlx download
```

```python
import motionbricks_mlx as mb
from motionbricks_mlx import examples

model = mb.load("soma", "checkpoints/MotionBricks/soma", dtype="bfloat16")  # float32 | float16 | bfloat16
out = model.run(examples.request("soma", batch=8))     # NumPy in (ONNX input names) -> NumPy out
frames = [out["pred_global_motions"][i, : 4 * n] for i, n in enumerate(out["pred_num_tokens"])]
g1 = mb.load("unitree_g1", "checkpoints/MotionBricks/g1").quantize(bits=8)   # INT8 weights
```

**Rust** (Rust plus Xcode with its Metal toolchain; no Python):

```sh
model-rs/scripts/download_weights.sh
cd model-rs && cargo build --release      # the first build compiles MLX (~6 min)
cargo run --release --example generate_metal -- soma ../checkpoints/MotionBricks/soma 4 bfloat16
```

```rust
use motionbricks_rs::metal::model::MetalModel;
let model = MetalModel::from_checkpoints_with("unitree_g1", "checkpoints/MotionBricks/g1", 64, "bfloat16")?; // max batch 64
let (pred_qpos, pred_num_tokens, foot_contacts) = model.run(&inputs)?;   // ONNX input names
```

The guide covers the CLI (`generate`, `convert`, `bench`), converted weight files, forced durations
and the mlx-rs port: [Python](docs/user_guide.md#2-the-python-model-motionbricks_mlx) and
[Rust](docs/user_guide.md#3-the-rust-implementations-model-rs).

## Use the Runtime API

The Runtime API keeps upstream's names and behavior. Engines generate from context and target
poses, and motion players and recorders chain generations into continuous motion. It also includes
target placement and speed scaling, SOMA IK pins, and BVH/CSV clip loading.

```python
from motionbricks_mlx import runtime as rt

with rt.MotionBricksUnitreeG1Engine.from_checkpoints("checkpoints/MotionBricks/g1", dtype="bfloat16") as engine:
    context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
    context.update_from_poses(context_joint_q)          # the last four poses shown
    target.update_from_poses(target_joint_q)            # one to four target poses
    engine.generate_animation(context, target, output)  # output.num_frames frames at 30 Hz
    pose = output.sample(0.5)
```

Upstream's two examples are ported. `locomotion_loop` steers a jog around a turn; on the Metal
runtime each generation takes about 7 ms in FP32 and 5 ms in BF16:

```sh
uv run --package motionbricks-mlx motionbricks-mlx download --clips    # the sample clips the examples start from
uv run --package motionbricks-mlx python model/examples/locomotion_loop.py --model soma
(cd model-rs && cargo run --release --example locomotion_loop -- unitree_g1 bfloat16)
```

API guides: [Python](model/docs/runtime_api.md) · [Rust](model-rs/docs/runtime_api.md).

## Tune and benchmark

**Pick a precision.**

| Goal | Use |
|---|---|
| exact reference behaviour | FP32 (matches PyTorch to ≤ 1.2e-5 per sub-network) |
| lowest latency for single requests | the Rust Metal runtime in BF16, FP16 or INT8 (4.0 ms); in Python, INT8 or BF16 (about 8 ms) |
| many characters at once | BF16 with batching (903–909 req/s at N=32 on the Metal runtime) |
| a stream of single requests | the Metal runtime (247–251 req/s in BF16), or Python with `mx.async_eval` pipelining |

**Benchmark.**

```sh
uv run --package motionbricks-mlx motionbricks-mlx bench --model soma --checkpoints checkpoints/MotionBricks/soma --batch 1 8 32
(cd model-rs && cargo run --release --bin mbrs-bench -- --model soma --dtype bfloat16 --batches 1 8 32)   # mlx-rs + Metal
model-rs/bench/run_all.sh 3 "float32 bfloat16"    # Python vs Rust, interleaved rounds (uses the full uv workspace)
```

Compare runtimes only in interleaved rounds: under sustained load the M4 Pro's GPU time drifts by
up to 10%.

**Tune the Metal runtime for your Mac.** The built-in GEMM plan table is measured on an M4 Pro.
Other Macs use a fitted rule scaled to their GPU core count. That adaptation to M2–M5 base, Pro,
Max and Ultra chips is [untested](docs/user_guide.md#other-apple-silicon-macs-not-tested): all
results here come from one M4 Pro. `gemm-tune` measures a table for your GPU, and `MB_PLAN_TABLE`
loads it:

```sh
cd model-rs
TUNE_TABLE=$PWD/my_mac.txt cargo run --release --bin gemm-tune -- soma bfloat16 1 8 32   # repeat per model and precision
MB_PLAN_TABLE=$PWD/my_mac.txt cargo run --release --bin mbrs-bench -- --model soma --dtype bfloat16 --batches 1 8 32
```

**Check correctness.**

```sh
sandbox/setup.sh                                            # NVIDIA's reference model, fixtures (~5 GB)
uv run --all-packages pytest                                # 219 Python tests
(cd model-rs && cargo test --release -- --test-threads=1)   # 128 Rust tests
```

The model suites also run without the sandbox, and skip only the parity tests against NVIDIA's code.
The guide covers [tuning and the profiling tools](docs/user_guide.md#6-tuning-and-profiling) and
[how the results are checked](docs/user_guide.md#3-the-rust-implementations-model-rs).

## Documentation

| Document | Contents |
|---|---|
| [User guide](docs/user_guide.md) | installation paths, both models in depth, the Runtime API, full benchmarks, tuning, correctness, the sandbox, license details |
| Runtime API guides: [Python](model/docs/runtime_api.md), [Rust](model-rs/docs/runtime_api.md) | engines, tokens, chaining, players, target poses, clips, differences from upstream |
| [model/README.md](model/README.md) | the MLX port: architecture, tensor interface, design notes, per-stage parity |
| [model-rs/README.md](model-rs/README.md) | the Metal runtime: MLX analysis, kernel design, per-stage profiles |
| [sandbox/README.md](sandbox/README.md) | CUDA emulation on Apple silicon: research, setup, limits |
| [tools/onnx/README.md](tools/onnx/README.md) | the Rust ONNX exporter: usage, design, comparison with the original exporter |

## License

The code is **Apache-2.0** ([LICENSE](LICENSE)), like upstream's. [NOTICE](NOTICE) carries
NVIDIA's copyright and credits the third-party code.

Not included:
- **Weights:** under the NVIDIA Open Model Agreement. The download commands fetch them from
  [Hugging Face](https://huggingface.co/nvidia/MotionBricks).
- **Sample clips:** under NVIDIA's Sample Data Evaluation License. They are downloaded on request.

The Runtime API guides are adapted from upstream's docs under CC BY 4.0. The user guide has the
[details](docs/user_guide.md#8-license-details).

## Citation

```bibtex
@article{wang2026motionbricks,
  title   = {MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives},
  author  = {Wang, Tingwu and Dionne, Olivier and De Ruyter, Michael and Minor, David and Rempe, Davis and Zhao, Kaifeng and Petrovich, Mathis and Yuan, Ye and Li, Chenran and Luo, Zhengyi and Robison, Brian and Blackwell, Xavier and Antoniazzi, Bernardo and Peng, Xue Bin and Zhu, Yuke and Yuen, Simon},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026}, volume = {45}, number = {4}, doi = {10.1145/3811334}
}
```
