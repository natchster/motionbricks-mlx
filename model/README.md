# model/ — MotionBricks on MLX

`motionbricks_mlx` is a port to [MLX](https://github.com/ml-explore/mlx) of NVIDIA's
[MotionBricks](https://github.com/NVlabs/motionbricks) (SIGGRAPH 2026) for both released models:
**SOMA** (30-joint human) and **Unitree G1** (34-joint humanoid). It runs on the Apple GPU
through Metal. Its only dependencies are `mlx` and `numpy`.

## What is ported

The complete batched inference graph, i.e. everything inside upstream's ONNX files:

```
context + target poses ──► [G1: forward kinematics | SOMA: T-pose retarget]
   ──► Root module    : 2-stage transformer picks the duration (6–16 tokens) + conv decoder → 64-frame root path
   ──► Pose module    : 16-layer masked (MaskGIT) transformer → multi-head codebook tokens
   ──► VQ-VAE decoder : product-codebook lookup → conditioned dilated-conv upsampler → motion features
   ──► local→global root integration, SOMA height-violation fallback, re-anchoring
   ──► [G1: joint-angle decoding + foot contacts | SOMA: retarget back]
```

The tensor interface is **identical to the ONNX models** (same names, shapes and INT32 masks),
so code written against ONNX Runtime or TensorRT can switch by swapping the session object.

Upstream's **Runtime API** is ported too, as `motionbricks_mlx.runtime`. It is the library that
turns single generations into continuous character motion: engines, endpoint correction, tokens,
animation buffers, the motion recorder, motion players, target placement, IK pins and clip loading.
The viewer, editors and demo apps are not ported. See [Runtime API](#runtime-api).

| | SOMA | Unitree G1 |
|---|---|---|
| inputs | `global_root_values [N,8,5]`, `local_root_values [N,8,4]`, `local_poses [N,8,267]`, 3 masks `[N,8]`, `allowed_pred_num_tokens [N,11]` | `context_qpos [N,4,36]`, `target_qpos [N,4,36]`, 3 masks `[N,8]`, `allowed_pred_num_tokens [N,11]` |
| outputs | `pred_global_motions [N,64,366]`, `pred_num_tokens [N]` | `pred_qpos [N,64,36]`, `pred_num_tokens [N]`, `pred_foot_contacts [N,64,4]` |
| parameters | ≈ 200 M (pose 136.5 M, root 40 M, VQ-VAE 23 M) | ≈ 195 M |

## Usage

```python
import motionbricks_mlx as mb
from motionbricks_mlx import examples

model = mb.load("soma", "checkpoints/MotionBricks/soma")       # HF SafeTensors folder
out = model.run(examples.request("soma", batch=8))              # dict of NumPy in → dict of NumPy out
frames = [out["pred_global_motions"][i, : 4 * n] for i, n in enumerate(out["pred_num_tokens"])]

mb.convert("unitree_g1", "checkpoints/MotionBricks/g1", "weights/g1.safetensors")   # one-file MLX weights
g1 = mb.load("weights/g1.safetensors", dtype="float16")         # float32 | float16 | bfloat16
g1.quantize(bits=8)                                             # experimental: ~4x smaller transformer weights, fastest at N=1
```

`run` (and `prepare`) checks every input's ONNX shape on the host and raises `ValueError` naming
the input, instead of letting the rig kernels read out of bounds. Model names are `soma` and
`unitree_g1` (alias `g1`) everywhere. For a stream of single requests, overlap host and GPU work
with `mx.async_eval` (see the [user guide](../docs/user_guide.md#choosing-precision-and-batching)).

Install and get the weights (MLX + NumPy only, no sandbox):

```sh
uv sync --package motionbricks-mlx                                      # from the repository root; or: pip install ./model
uv run --package motionbricks-mlx motionbricks-mlx download             # checkpoints/MotionBricks/{soma,g1}, SHA-256 verified
```

From the command line:

```sh
uv run --package motionbricks-mlx motionbricks-mlx generate --model soma --checkpoints checkpoints/MotionBricks/soma --batch 4
uv run --package motionbricks-mlx motionbricks-mlx bench    --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1 --batch 1 8 32
uv run --package motionbricks-mlx motionbricks-mlx convert  --model soma --checkpoints checkpoints/MotionBricks/soma --output weights/soma.safetensors
```

`--package motionbricks-mlx` installs only this package, not the sandbox. `download`
(`motionbricks_mlx.hub`) fetches `nvidia/MotionBricks` at revision `3cb362d` over plain HTTPS. It
checks every file against upstream's size and SHA-256 table (`assets/checkpoints.sha256`) and skips
files that are already verified. `--model soma|unitree_g1` and `--dir PATH` narrow or move it. The
weights are licensed under the NVIDIA Open Model Agreement and are not part of this repository.
`--clips` (or `--only-clips`) also fetches the four sample clips used by the Runtime API examples
into `checkpoints/motions`. They come from upstream's Git LFS at the pinned commit, are verified
against `assets/clips.sha256`, and are NVIDIA sample data under the NVIDIA Sample Data Evaluation
License.

## Runtime API

`motionbricks_mlx.runtime` keeps upstream's names and behavior, with NumPy poses and MLX inference
in place of Warp arrays and TensorRT:

```python
import motionbricks_mlx as mb
from motionbricks_mlx import runtime as rt

clip_path = "checkpoints/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh"
skeleton = rt.load_skeleton_from_bvh(clip_path, remove_root=True)   # Hips-rooted, no USD needed
clip = rt.load_bvh(clip_path, skeleton)
with rt.MotionBricksSOMAEngine(mb.load("soma", "checkpoints/MotionBricks/soma", dtype="bfloat16"),
                               model_to_runtime=rt.model_to_runtime("soma")) as engine:
    player = rt.GeneratedMotionPlayer(engine, skeleton)
    player.seed_history_from_pose(start_pose)
    player.generate(target_poses, rt.TargetPoseInferenceParams(transition_early_by_frames=4))
    player.sample(1 / 60, pose, foot_contacts)      # writes the next pose in place
    player.record_pose(1 / 60, pose)                # record what you show, then chain
```

```sh
uv run --package motionbricks-mlx motionbricks-mlx download --clips
uv run --package motionbricks-mlx python model/examples/generate_once.py --model unitree_g1
uv run --package motionbricks-mlx python model/examples/locomotion_loop.py --model soma --dtype bfloat16
```

The guide, adapted from upstream's `docs/runtime_api.md`, is
[docs/runtime_api.md](docs/runtime_api.md). It covers engines, spaces, generation settings,
chaining, players and the differences from upstream. The Rust port has the same API on its Metal
runtime ([model-rs/docs/runtime_api.md](../model-rs/docs/runtime_api.md)).

## Design notes

- **Channels-last everywhere.** PyTorch's Conv1d is `(B, C, T)`; MLX convolutions are natively
  `(B, T, C)`. The port keeps one layout end to end, which removes every `transpose(1, 2)` of the
  original, and transposes Conv1d kernels `(out, in, k) → (out, k, in)` once at load time.
- **One implementation for both rigs.** Upstream's SOMA and G1 network files differ only in
  hyper-parameters and mask dtypes (INT32 vs bool). Here they share `networks.py`; the
  differences live in `config.py` and the rig wrappers.
- **Static shapes → one compiled graph.** Upstream pads to 16 tokens / 64 frames and masks, which
  is what made it exportable to ONNX. The same property lets `mx.compile` trace the whole
  generation, including the data-dependent duration argmax, into a single graph.
- **Strict, mechanical weight loading** (`weights.py`). Wrapper prefixes are remapped, leading
  underscores dropped (MLX treats `_attr` as private), PyTorch's packed `in_proj_weight` is split
  into an `nn.Linear`, and the load fails on any missing, unexpected or misshaped tensor.
- **Parameters are evaluated before compiling.** Casting to FP16/BF16 and `quantize` are lazy in
  MLX. If the first `mx.compile` trace captured them, every compiled call would recompute about
  200 M casts (≈ 4 ms). `set_compute_dtype` and `quantize` therefore evaluate the parameters.
- **Anything the compiled graph bakes in drops it.** The trace captures the parameters and
  `num_inference_steps`. Setting `num_inference_steps`, `set_compute_dtype` (and the inherited
  `set_dtype`, which is routed to it), `quantize`, `update` and `load_weights` all re-evaluate the
  parameters where needed and force a fresh trace.
- **The rig runs as custom Metal kernels** (`kernels.py`, source in `assets/rig_kernels.metal`).
  G1 forward kinematics, joint-angle decoding and the SOMA T-pose retarget are hundreds of tiny
  3×3 products as MLX ops, but one thread per pose as kernels. The op-based versions remain the CPU
  fallback and the test oracle.
- **Mixed precision mirrors upstream.** `float16`/`bfloat16` apply to the three networks only;
  normalization, root integration and rig geometry stay FP32, like upstream's mixed-FP16 ONNX.

## Verification

`uv run --package motionbricks-mlx --extra test pytest model/tests` (or `pip install
"./model[test]" && pytest model/tests`) runs the suite without the sandbox:

| Tests | Need | Without them |
|---|---|---|
| 18 weight-free unit tests (`test_layers.py`) and 21 API tests (`test_api.py`) | nothing | always run |
| 26 behavioural tests (`test_model.py`) | the checkpoints (`motionbricks-mlx download`) | skipped |
| 42 parity tests (`test_parity.py`), 2 of them on the CPU stream | the golden fixtures from NVIDIA's PyTorch model (`sandbox/setup.sh`) | skipped |
| 39 Runtime API tests (`test_runtime.py`): 6 weight-free, 25 parity with upstream's runtime, 8 end to end | weight-free: nothing; parity: the runtime fixtures (`sandbox/setup.sh`) and the clips (`download --clips`); end to end: also the checkpoints | skipped |

`MOTIONBRICKS_CHECKPOINTS`, `MOTIONBRICKS_GOLDEN`, `MOTIONBRICKS_RUNTIME_GOLDEN` and
`MOTIONBRICKS_CLIPS` point the tests at other locations. `MOTIONBRICKS_REQUIRE_FIXTURES=1` turns
every skip into a failure, so a complete run cannot pass by skipping. The parity
tests compare against NVIDIA's own PyTorch implementation, using golden fixtures recorded by
`sandbox/scripts/make_golden.py`. Every sub-network is **teacher-forced** with the exact tensors the
PyTorch run produced, so an error localizes to one module.

| Stage (5 request families × 2 models) | Max relative error vs PyTorch FP32 |
|---|---|
| Root network: duration logits / root path | ≤ 2e-6 / ≤ 4e-6 |
| Pose transformer logits | ≤ 8e-6, **100 % identical argmax tokens** |
| VQ-VAE decoder | ≤ 1.2e-5 |
| End to end: chosen durations | **identical in every case** |
| End to end: G1 `pred_qpos` | ≤ 6e-5 abs (m / rad) |
| End to end: SOMA root / positions / rotations | ≤ 1e-6 / ≤ 1e-5 / ≤ 6e-4 abs |

The SOMA rotation error comes from the 6D→matrix Gram-Schmidt in the T-pose retarget, which
amplifies float32 noise; 6e-4 is about 0.03°. The request families are upstream's ONNX trace
inputs, randomized requests, one-target requests, restricted durations, and on-manifold
"chained" requests built from generated motion.

Other tests cover:

- weight-free unit tests of every block against NumPy derivations, including multi-step MaskGIT
  re-masking against a NumPy port of upstream's sampler;
- input validation, dtype and loading errors, and strict weight loading;
- batch-row independence, `mx.compile` = eager, forced durations and allowed-duration masks;
- settings and weight updates reaching the compiled graph;
- converted-file round trips and multi-step MaskGIT decoding;
- FP16, BF16 and INT8 durations against FP32 (at most one flip per test batch), and 4-bit weights;
- the custom rig kernels against the op-based reference implementation (batch 1, 3 and 33), their
  shape checks, and the packed-rig offsets in `rig_kernels.metal`;
- end to end on the CPU stream, which uses the op-based rig.

The Runtime API tests compare with upstream's own runtime code.
`sandbox/scripts/make_runtime_golden.py` runs upstream's engine methods, recorder, target helpers,
IK corrector and clip loaders on Warp's CPU backend, with NVIDIA's PyTorch model in place of
TensorRT, for four generations per rig. Results:
- Encoding, decoding, endpoint correction, clip loading, recorder resampling, target placement and
  speed scaling, and buffer sampling and looping match within 2e-5, and IK pins within 5e-5.
- End to end, every generation picks upstream's duration, and its poses stay within 3e-3.

## Performance (Mac mini 2024, Apple M4 Pro: 14-core CPU, 20-core GPU, 64 GB)

| | N=1 latency | N=32 throughput |
|---|---|---|
| SOMA, MLX FP32 / BF16 / INT8 | 10.5 / 8.1 / 7.9 ms | 750 / 848 / 730 req/s |
| G1, MLX FP32 / BF16 / INT8 | 10.5 / 8.3 / 8.0 ms | 749 / 848 / 693 req/s |

That is 3–4.5× faster than NVIDIA's PyTorch model on MPS at batch 1, and 4.9–5.7× the throughput
of ONNX Runtime at batch 32. Full tables, the comparison with an RTX 4090 + TensorRT, and a profile
of where the time goes: [benchmarks/README.md](benchmarks/README.md). For the lowest single-request
latency (4.0 ms in BF16, about 2× faster than MLX at every precision), see the Rust Metal runtime in
[`model-rs`](../model-rs/README.md).
