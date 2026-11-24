# app-rs assets

`scenes/` and `smart_primitives/` are NVIDIA-authored documents copied unchanged from
upstream [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) (`assets/`, commit
54fc563). Like upstream's other assets they are dual-licensed Apache-2.0 / CC-BY-4.0 (see
the repository LICENSE, section "NVIDIA-AUTHORED DOCUMENTATION AND MEDIA", and NOTICE).

The apps also use, but do not redistribute:

- **Motion clips** (BVH/CSV), under the NVIDIA Sample Data Evaluation License: download
  them with `motionbricks-mlx download --clips` (or `model-rs/scripts/download_weights.sh
  clips`) into `checkpoints/motions/`, then pass `--clips checkpoints/motions`.
- **The SOMA skeletal mesh** (`motionbricks/assets/human/soma_base_skel_minimal_no_root.usd`,
  Apache-2.0) and the **Unitree G1 robot description** (`motionbricks/assets/robots/unitree_g1/`,
  BSD 3-Clause, see `licenses/unitree-LICENSE.txt`): fetch from the upstream repository with
  Git LFS, then pass `--assets <upstream checkout>` (see `app-rs/README.md`).
