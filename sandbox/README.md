# sandbox/ — CUDA emulation on Apple silicon

A reproducible environment for running CUDA-targeting code (specifically NVIDIA's MotionBricks
stack) on an M-series Mac, plus the **reference oracle** that the MLX port in `../model` is
verified against.

## The short answer from the research

**No tool can execute NVIDIA's CUDA binaries on a Mac.** There is no NVIDIA driver or GPU,
eGPUs are unsupported on Apple silicon, and TensorRT, cuDNN and `libtorch_cuda` are closed
binaries that cannot be translated. What *does* work is substituting each CUDA layer at its
**API boundary**. The best sandbox is therefore layered:

| CUDA layer used by MotionBricks | Substitute on Apple silicon | Fidelity | Module |
|---|---|---|---|
| `torch.cuda` / `.cuda()` (exporter, research code) | PyTorch **MPS** (Metal), CPU fallback for missing ops | same kernels' math; no FP64 on Apple GPUs | `cudashim/torch_cuda.py` |
| **TensorRT** (`tensorrt-cu12`, no macOS wheels) | **ONNX Runtime** CPU EP (exact FP32) or CoreML EP, behind a TensorRT-10-compatible API | upstream docs: every TensorRT plan "matched ONNX Runtime" | `cudashim/tensorrt_ort.py` |
| **NVIDIA Warp** CUDA kernels | Warp's **LLVM CPU backend**: same `@wp.kernel` source, no translation | bit-identical semantics; no streams, graphs or mempools | `cudashim/warp_cpu.py` |
| raw **CUDA C++** (`.cu`, cuBLAS) | **CuMetal** CUDA→Metal compiler + runtime (optional) | research-grade subset; FP64 emulated | `native/` |
| (reference) NVIDIA's PyTorch model | upstream `tools/onnx` models, pinned commit | is the reference | `cudashim/reference.py` |

### Options evaluated (Nov 2026)

**Python stack**, ranked:

1. **warp-lang 1.19 CPU.** Its macOS wheels "support CPU execution but not Metal acceleration".
   `wp.get_device("cuda")` raises, so it doesn't fall back silently. Streams, graph capture,
   mempools and IPC are CUDA-only. Tile ops run on CPU with `block_dim=1`.
2. **ONNX Runtime 1.32, CPU EP.** The TensorRT replacement. MotionBricks itself documents ORT as
   the parity reference.
3. **ONNX Runtime CoreML EP.** Uses GPU and ANE. FP16 accumulation can change outputs, so gate it
   against CPU EP results.
4. **PyTorch 2.15 MPS.** No fp64 or complex128. Use `PYTORCH_ENABLE_MPS_FALLBACK=1`. `.cuda()`
   raises without a shim.
5. triton-cpu (work in progress) or third-party triton-msl. MotionBricks doesn't need either.

Ruled out:

- **CuMetal and mvcc for Python:** they cannot load TensorRT or `libtorch_cuda`.
- **Docker and Podman:** krunkit/libkrun Venus gives Vulkan *compute* only, and Docker Desktop's
  VMM documents no GPU access. Neither gives a CUDA path.

**Raw CUDA C++**, ranked:

| # | Project | Status | Notes |
|---|---|---|---|
| 1 | **CuMetal** (Lulzx/cuda-metal) v0.8.0 | Apache-2.0; 217★, 5 contributors, active | Runtime + driver API subsets, PTX import, NVRTC, streams/events/graphs, shared memory, warp ops, **cuBLAS/cuFFT/cuRAND subsets**. FP64 emulated. Needs full Xcode + Metal toolchain. |
| 2 | mvcc (doximity/mvcc) v1.3.0 | Apache-2.0; single author | 132 runtime entry points, no PTX loading/NVRTC, **no cuBLAS/cuDNN**. macOS 26+. Good second compiler for cross-checking. |
| 3 | chipStar 1.3 | mature HIP/CUDA→SPIR-V | On macOS it runs only on the **CPU** (PoCL). The M4 GPU (KosmicKrisp Vulkan) is in unreleased CI only. |
| ✗ | bare-cuda-metal, cuda4mac, CUDAM, elementalcollision/mvcc | — | Days-old single-author repos, committed binaries, no license, a 404 package, or a 0★ copy of mvcc. |
| ✗ | ZLUDA on macOS / HIP-CPU | — | ZLUDA: "unlikely to ever happen". HIP-CPU: archived. |

Supply-chain note: CuMetal ships through a third-party Homebrew tap with 0★ and one owner, and a
formula is Ruby that runs at install time. `native/install_cumetal.sh` instead builds the
**reviewed v0.8.0 source tarball** into a repo-local prefix. Both tarballs are SHA-256 pinned, the
`DYLD` binary-injection shim stays off, and nothing touches the system `PATH`.

Sources:
- [TensorRT wheels](https://pypi.org/project/tensorrt-cu12-bindings/10.8.0.43/#files)
- [Warp README](https://github.com/NVIDIA/warp/blob/main/README.md)
- [ORT CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [PyTorch MPS notes](https://docs.pytorch.org/docs/2.15/notes/mps.html)
- [Podman GPU](https://podman-desktop.io/docs/podman/gpu), [Docker VMM](https://docs.docker.com/desktop/features/vmm/)
- [CuMetal](https://github.com/Lulzx/cuda-metal) ([known gaps](https://github.com/Lulzx/cuda-metal/blob/main/docs/known-gaps.md))
- [mvcc](https://github.com/doximity/mvcc), [chipStar 1.3](https://github.com/CHIP-SPV/chipStar/releases/tag/v1.3.0)
- [ZLUDA FAQ](https://zluda.readthedocs.io/stable/faq.html), [HIP-CPU](https://github.com/ROCm/HIP-CPU)
- [MotionBricks ONNX notes](https://github.com/NVlabs/motionbricks/blob/main/docs/onnx_inference.md)

## Setup

The sandbox is only needed for NVIDIA's reference model, the golden fixtures, the parity tests and
the cross-backend benchmarks. The Python and Rust models install and run without it (see the
repository README's quick start).

```sh
sandbox/setup.sh                 # ~1.6 GB checkpoints + 1.5 GB local ONNX export, all verified
uv run --all-packages cudashim-doctor
```

`setup.sh` does the following:

1. `uv sync --all-packages`, which installs Python 3.12, MLX, PyTorch (MPS), ONNX Runtime and Warp.
2. Clones NVlabs/motionbricks at the pinned v0.2.0 commit into `sandbox/upstream/`.
3. Fetches the Git-LFS rig assets through GitHub's media endpoint and checks each one against
   its LFS pointer's SHA-256, so git-lfs isn't needed.
4. Downloads the checkpoints from `nvidia/MotionBricks` with the model package's own downloader
   (`motionbricks_mlx.hub`, the same one `motionbricks-mlx download` uses). It first checks that
   the package's checksum table matches upstream's `checksums.py`.
5. Exports the ONNX graphs with **NVIDIA's own exporter**, including its `onnx.checker` and
   interface validation. This replaces the slow 760 MB LFS downloads.
6. Records golden fixtures with the upstream PyTorch model.

Optional raw-CUDA layer. It needs full Xcode, not just the Command Line Tools:

```sh
# 1. Install Xcode from the App Store, then accept its license once (or open Xcode and click Agree):
sudo /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -license accept
# 2. Build dependencies, then CuMetal into sandbox/native/toolchains/:
brew install cmake llvm lz4 zstd
sandbox/native/install_cumetal.sh && source sandbox/native/env.sh
cumetalc sandbox/native/samples/vector_add.cu -o /tmp/va && CUMETAL_TRACE_GPU=1 /tmp/va
```

`xcode-select` can stay pointed at the Command Line Tools. The installer, `env.sh` and the
tests aim `xcrun` at `/Applications/Xcode.app` through `DEVELOPER_DIR`, per process only. The
installer also downloads Xcode's Metal toolchain component
(`xcodebuild -downloadComponent MetalToolchain`) if it is missing. Tested with Xcode 27.2
(27B52), Metal Toolchain 27B47, Homebrew LLVM 23.1.6 and CuMetal 0.8.0.

## Usage

```python
import cudashim
cudashim.enable_all()          # `import tensorrt` -> ONNX Runtime shim; cuda -> mps

import torch
x = torch.randn(1024, 1024, device="cuda")      # lands on mps:0
model = MyNet().cuda()                           # NVIDIA-style code, unchanged
torch.cuda.synchronize(); torch.cuda.Event(enable_timing=True)

import tensorrt as trt                           # Builder/OnnxParser/Runtime/execute_async_v3
```

On Apple silicon, "device pointers" are host addresses (unified memory). The TensorRT shim
therefore reads and writes the buffers whose `ptr` you pass to `set_tensor_address`, whether
NumPy arrays or Warp CPU arrays.

## What is verified (`uv run --all-packages pytest sandbox/tests`)

| Test | Proves |
|---|---|
| `test_assets.py` | The model package's checkpoint manifest (`model/motionbricks_mlx/assets/checkpoints.sha256`), which the model-only download paths verify against, equals upstream's `checksums.py`. |
| `test_torch_cuda_shim.py` | `device="cuda"`, `.cuda()`, `Module.cuda()`, events, autocast, sync. Results land on MPS and match CPU numerics. |
| `test_reference_on_mps.py` | **NVIDIA's PyTorch MotionBricks, moved with `.cuda()`**, runs on the Apple GPU and matches the CPU golden outputs. |
| `test_tensorrt_shim.py` | **NVIDIA's own `build_serialized_engine` + `inspect_engine_tensors`** produce a plan from the G1 graph. A named-tensor `execute_async_v3` launch matches the PyTorch golden outputs. |
| `test_warp_kernels.py` | **NVIDIA's Warp CUDA kernels** (condition masks, token padding) compile and run unchanged on the CPU, and `wp.set_device("cuda:0")` code runs under the alias. |
| `test_native_cuda.py` | **Raw CUDA C++ compiled by CuMetal runs on the Apple GPU.** Three programs: a grid-stride vector add, a shared-memory + warp-shuffle reduction with `atomicAdd`, and a cuBLAS SGEMM followed by a custom bias + ReLU kernel. Each must match a host reference, and every custom kernel must report `device=apple_gpu` in CuMetal's GPU trace. cuBLAS runs as an MPS matrix multiply, which doesn't trace, hence the extra kernel. Skips if CuMetal isn't installed. |

## Limits (honest list)

- **No CUDA graphs or streams.** The shims are synchronous. Upstream's `InferenceEngine` class
  itself is not runnable, because it asserts `wp.get_device().is_cuda` and captures CUDA graphs.
  Its builder and plan paths are, as tested above.
- **The TensorRT shim is an API emulation, not an optimizer.** FP16 and TF32 flags are recorded
  but don't change ONNX Runtime's math. For speed on Apple silicon, use the MLX port.
- **No FP64 on Apple GPUs.** CuMetal emulates FP64 in software, and MPS rejects float64 tensors.
- CuMetal is research-grade: self-reported results and open bugs (e.g. `__shfl_xor_sync` with
  width < 32). Treat its output like any unverified compiler's.
