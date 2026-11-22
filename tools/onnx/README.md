# tools/onnx — `motionbricks-onnx`, the ONNX exporter in Rust

A Rust port of NVIDIA's ONNX exporter ([upstream `tools/onnx`](https://github.com/NVlabs/motionbricks/tree/main/tools/onnx)),
a drop-in replacement for `uv run --project tools/onnx motionbricks-onnx` — **without PyTorch,
torch tracing, or the pinned export environment**. It emits the batched inference graphs
(`mmm_v6_{g1,soma}_{fp32,fp16}.onnx`) directly from the verified checkpoints.

```sh
cd tools/onnx && cargo build --release
target/release/motionbricks-onnx --model soma --checkpoints ../../checkpoints/MotionBricks/soma
target/release/motionbricks-onnx --model unitree_g1 --checkpoints ../../checkpoints/MotionBricks/g1 --precision fp16
```

The CLI mirrors upstream's: `--model {soma|unitree_g1}`, `--checkpoints DIR`,
`--precision {fp32,fp16}` (fp32 default), `--output PATH` (default
`mmm_v6_{g1|soma}_{fp32|fp16}.onnx`), `--force`, `--verbose`. Checkpoints are verified
against the shared `model/motionbricks_mlx/assets/checkpoints.sha256` manifest before
export, like upstream's verify-then-export, and the output is staged atomically.

## How it works

The torch exporter builds the graph by *tracing* a forward pass, deferring every shape
computation into the graph itself (the exported graphs spend most of their ~7,000–8,700
nodes on `Shape`/`Gather`/`Unsqueeze` plumbing). These graphs are single-shot — context
and target poses in, 64 frames out in one pass — and every tensor dimension except the
leading batch size is fixed by the architecture. This crate instead builds the graph
**statically**: each emitter (the transformers, the VQ-VAE decoder, the representation
conversions, the G1 rig, the SOMA retarget) mirrors the verified
[`model-rs`](../../model-rs/src) op sequences and resolves all shape arithmetic at export
time. Only genuinely data-dependent values — frame indices derived from
`pred_num_tokens` — become runtime nodes. Constant work the tracer performs per export
(the step-0 MaskGIT remasking, weight transposes, the rig's window offsets) is folded at
build time.

The FP16 conversion (`--precision fp16`) ports upstream's `precision.py`: the two
transformers and the VQ-VAE decoder run in FP16, LayerNormalization/Softmax/Resize and all
motion geometry stay in FP32, and boundary casts are inserted. It needs no ONNX shape
inference — the emitter's own static registry knows every value's dtype and shape.

## Compared with the original exporter

Measured on a Mac mini M4 Pro with the checkpoints above
(`uv run --all-packages python tools/onnx/bench_export.py`):

| model | exporter | export s | peak RSS GB | nodes | file MB | ORT session s |
|---|---|---|---|---|---|---|
| soma | **rust** | **2.5** | **2.3** | **1,726** | 757 | **0.30** |
| soma | python (torch trace) | 3.8 | 3.6 | 6,993 | 758 | 0.39 |
| g1 | **rust** | **2.3** | **2.2** | **2,447** | 733 | **0.32** |
| g1 | python (torch trace) | 3.7 | 3.4 | 8,662 | 734 | 0.52 |

(The Python time is for `export_fp32` + validation alone; upstream's `export_release` also
verifies checksums, which the Rust time already includes. The graphs are dominated by ~735–757
MB of weights, so the file sizes match.)

**Correctness** (`sandbox/tests/test_onnx_rs.py`): every export passes `onnx.checker` and
upstream's own interface validation, and ONNX Runtime runs of the original and the Rust
graphs on identical inputs agree to FP32 accumulation noise:

| output | max abs diff vs the original graph |
|---|---|
| `pred_global_motions` (soma) | 1.4e-4 |
| `pred_qpos` (g1) | 1.3e-5 |
| `pred_foot_contacts` (g1) | 9.7e-5 |
| `pred_num_tokens` (both) | 0 (exact) |

The FP16 conversion halves the same initializers as upstream's `precision.py` (378.119 MB
vs 378.081 MB FP16 payload; the 38 KB delta is small fp32 constants whose consumer sets
differ between the two graph shapes).

## Layout

| File | Contents |
|---|---|
| `src/graph.rs`, `src/ops.rs` | the graph IR and typed op emitters, with static shape rules |
| `src/weights.rs`, `src/npy.rs` | verified SafeTensors/stats loading; NPY/NPZ reader (shared with `model-rs`) |
| `src/layers.rs`, `src/networks.rs` | transformer blocks, MHA, conv decoder; the three networks |
| `src/representation.rs` | feature layouts, normalization, global/local root conversion |
| `src/rig_g1.rs`, `src/soma.rs` | the G1 rig pre/post-processing; the SOMA T-pose retarget |
| `src/inference.rs`, `src/emit.rs` | the prediction graph and the per-model wiring |
| `src/fp16.rs` | the mixed-FP16 conversion (upstream `precision.py`) |
| `src/validate.rs` | the interface checks (upstream `validate_output`) |
| `proto/onnx.proto3` | the ONNX message definitions, vendored from onnx 1.23.2 (Apache-2.0) |

## License

Apache-2.0, like the rest of the repository; the emitters port NVIDIA's Apache-2.0
exporter models. `proto/onnx.proto3` is from the ONNX project (Apache-2.0, see
[licenses/onnx-Apache-2.0.txt](../../licenses/onnx-Apache-2.0.txt)). The exported graphs
contain the MotionBricks weights and are not redistributed.
