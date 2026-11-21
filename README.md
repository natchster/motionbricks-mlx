# motionbricks-mlx

NVIDIA's [MotionBricks](https://github.com/NVlabs/motionbricks) (real-time motion generation for
characters and robots, SIGGRAPH 2026) on Apple silicon.

> [!NOTE]
> **This is a quick port and may contain issues.** It passes its full test suite, which checks it
> stage by stage against NVIDIA's own PyTorch implementation, and it has also been tested by a
> human. If you find a discrepancy with upstream, please open an issue. This is an independent
> community project, not affiliated with or endorsed by NVIDIA.

The repo has two parts:

| Folder | What it is |
|---|---|
| [`model/`](model/README.md) | **`motionbricks_mlx`**, a native MLX port of the complete SOMA + Unitree G1 inference graphs. It has the same tensor interface as the upstream ONNX models, verified against NVIDIA's PyTorch implementation, and depends only on `mlx` + `numpy`. |
| [`sandbox/`](sandbox/README.md) | **`cudashim`**, a researched, layered CUDA-emulation sandbox for M-series Macs: `torch.cuda`→MPS, TensorRT→ONNX Runtime, Warp kernels→CPU, and optional CUDA C++→Metal via CuMetal. It also runs NVIDIA's upstream model as the reference oracle for the port. |

```
motionbricks-mlx/
├── model/
│   ├── motionbricks_mlx/     # layers, networks, inference, rig, weights, CLI (+ Apache-2.0 rig assets)
│   ├── tests/                # unit + golden-parity + behaviour tests
│   └── benchmarks/           # cross-backend latency/throughput harness + results
├── sandbox/
│   ├── cudashim/             # torch_cuda, tensorrt_ort, warp_cpu, reference, doctor
│   ├── native/               # pinned CuMetal installer, CUDA samples (vector add, reduction, cuBLAS)
│   ├── scripts/              # fetch_assets (verified), export_onnx (NVIDIA exporter), make_golden
│   ├── tests/                # each emulation layer exercised on real upstream code
│   └── setup.sh              # one-shot bootstrap
├── LICENSE  NOTICE  licenses/
└── pyproject.toml            # uv workspace (model + sandbox), Python 3.12
```

## Quick start

```sh
sandbox/setup.sh                             # env + pinned upstream + verified checkpoints + ONNX + golden fixtures
uv run --all-packages pytest                 # MLX tests + sandbox tests
uv run motionbricks-mlx generate --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1
uv run --all-packages python model/benchmarks/bench.py
```

## Ported and tested version

Everything below was ported from, and tested against, exactly these upstream versions:

| | Version |
|---|---|
| Code | [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) **v0.2.0**, commit [`54fc563aeaaf819a753d72cc9fdc6713989cdd43`](https://github.com/NVlabs/motionbricks/tree/54fc563aeaaf819a753d72cc9fdc6713989cdd43) |
| Weights | [nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks) on Hugging Face, revision [`3cb362d9b95aa40462abcb4d8cfae9ffb6d97410`](https://huggingface.co/nvidia/MotionBricks/tree/3cb362d9b95aa40462abcb4d8cfae9ffb6d97410) |
| SOMA checkpoints | MotionBricks v6 release `soma-v6-nova-4n8g-ea43dd8-fp32-v1`: pose, root and VQ-VAE networks at training step 1,500,000 |
| Unitree G1 checkpoints | release `out/motionbricks_{pose,root,vqvae}/version_1`: pose, root and VQ-VAE networks at training step 2,000,000 |
| Reference ONNX graphs | `mmm_v6_soma_fp32.onnx`, `mmm_v6_g1_fp32.onnx`, exported from the code and weights above with upstream's own exporter |

`sandbox/scripts/fetch_assets.py` pins both the commit and the revision. It also verifies every
checkpoint file against the SHA-256 list in upstream's `tools/onnx/checksums.py`. All 10 files
match at the revision above.

## Results

Correctness tests and benchmarks were run on a **Mac mini (2024, model `Mac16,11`) with an Apple
M4 Pro chip**: 14-core CPU (10 performance + 4 efficiency cores), 20-core GPU and 64 GB of unified
memory. It ran macOS 27.0 (build 26A428), Python 3.12.11, MLX 0.32.4, PyTorch 2.15.0 and ONNX
Runtime 1.32.0.

**Correctness.** Both models were checked on five request families each. Every sub-network
matches NVIDIA's PyTorch FP32 to ≤ 1.2e-5 relative error. Every predicted pose token's argmax and
every chosen motion duration is identical. 69 model tests and 14 sandbox tests pass. The 3 tests
for the optional CuMetal layer (raw CUDA C++ on Metal) skip unless that toolchain is built with
`sandbox/native/install_cumetal.sh`, which needs full Xcode.

**Speed.** Median time per batched generation on that Mac mini, with requests per second in
parentheses:

| | SOMA N=1 | SOMA N=32 | G1 N=1 | G1 N=32 |
|---|---|---|---|---|
| **MLX FP32** (this port) | **10.4 ms** (97) | 49.7 ms (644) | **11.2 ms** (89) | 47.7 ms (671) |
| **MLX BF16** (this port) | 12.2 ms (82) | **45.0 ms (711)** | 12.8 ms (78) | **45.4 ms (705)** |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 32.8 ms (30) | 91.6 ms (349) | 36.6 ms (27) | 98.3 ms (325) |
| upstream ONNX on ONNX Runtime CPU (TensorRT shim) | 13.6 ms (74) | 206.5 ms (155) | 13.9 ms (72) | 276.3 ms (116) |
| upstream PyTorch, CPU | 33.4 ms (30) | 882 ms (36) | 39.5 ms (25) | 966 ms (33) |
| *RTX 4090 + TensorRT FP32 (upstream docs, GPU only)* | *2.50 ms* | *5.28 ms* | *2.67 ms* | *7.81 ms* |

MLX is the fastest option on the Mac at every batch size: 3× faster than PyTorch-MPS at N=1, and
4–6× the throughput of ONNX Runtime at N=32. Batch 1 is bound by kernel dispatch rather than by
bandwidth, so batching is nearly free. The RTX 4090 row is NVIDIA's published figure, not
measured here. Tables for N = 1, 8, 16 and 32, the FP16 and INT8 rows, and the analysis are in
[model/benchmarks](model/benchmarks/README.md). The raw measurements are in
[`model/benchmarks/results/`](model/benchmarks/results/).

## License

**Apache-2.0** ([LICENSE](LICENSE)), the same license as upstream's code, so the port stays
license-compatible with MotionBricks. [NOTICE](NOTICE) carries NVIDIA's copyright, marks the
modified files, and credits the third-party code that upstream inherited (T2M-GPT, MotionGPT,
vector-quantize-pytorch), whose license texts are in [licenses/](licenses/).

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
