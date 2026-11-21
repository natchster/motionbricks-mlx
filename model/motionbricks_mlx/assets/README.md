# Assets

## Rig assets

`g1_model_rig.npz` and `soma_model_rig.npz` are copied unchanged from
[NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) (`tools/onnx/assets/`, commit
`54fc563`). They hold rig offsets and conversion data prepared by NVIDIA. Per the upstream
LICENSE ("NVIDIA RIG ASSETS"), NVIDIA's contributions to these files are licensed under the
Apache License, Version 2.0. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

## Other files

- `rig_kernels.metal`: Metal transcriptions of the rig geometry, shared by the Python port
  (`kernels.py`) and the Rust ports.
- `checkpoints.sha256`: size and SHA-256 of every released checkpoint file, copied from upstream
  `tools/onnx/checksums.py` (Apache-2.0, NVIDIA). `motionbricks-mlx download`,
  `model-rs/scripts/download_weights.sh` and the sandbox verify downloads against it.
