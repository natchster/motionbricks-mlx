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
The Warp/TensorRT *runtime* is not ported: the playback engines, Newton viewer, editors and
endpoint correction are application code built around this graph.

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
g1.quantize(bits=8)                                             # experimental: ~4x smaller transformer weights (not faster)
```

From the command line:

```sh
uv run motionbricks-mlx generate --model soma --checkpoints checkpoints/MotionBricks/soma --batch 4
uv run motionbricks-mlx bench    --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1 --batch 1 8 32
uv run motionbricks-mlx convert  --model soma --checkpoints checkpoints/MotionBricks/soma --output weights/soma.safetensors
```

Get the weights with `sandbox/setup.sh`, or `hf download nvidia/MotionBricks --local-dir
checkpoints/MotionBricks`. They are licensed under the NVIDIA Open Model Agreement and are not
part of this repository.

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
- **Mixed precision mirrors upstream.** `float16`/`bfloat16` apply to the three networks only;
  normalization, root integration and rig geometry stay FP32, like upstream's mixed-FP16 ONNX.

## Verification

The tests run against NVIDIA's own PyTorch implementation (golden fixtures recorded by
`sandbox/scripts/make_golden.py`): `uv run --all-packages pytest model/tests`. Every sub-network is
**teacher-forced** with the exact tensors the PyTorch run produced, so an error localizes to one
module.

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

- weight-free unit tests of every block against NumPy derivations;
- batch-row independence, `mx.compile` = eager, forced durations and allowed-duration masks;
- converted-file round trips and multi-step MaskGIT decoding;
- FP16, BF16 and INT8 durations against FP32.

## Performance (Mac mini 2024, Apple M4 Pro: 14-core CPU, 20-core GPU, 64 GB)

| | N=1 latency | N=32 throughput |
|---|---|---|
| SOMA, MLX FP32 / BF16 | 10.4 / 12.2 ms | 644 / 711 req/s |
| G1, MLX FP32 / BF16 | 11.2 / 12.8 ms | 671 / 705 req/s |

That is 3× faster than NVIDIA's PyTorch model on MPS at batch 1, and 4–6× the throughput of
ONNX Runtime at batch 32. Full tables, the comparison with an RTX 4090 + TensorRT, and a profile
of where the time goes: [benchmarks/README.md](benchmarks/README.md).
