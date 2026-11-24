# app-rs — the MotionBricks applications in Rust

Rust ports of NVIDIA's four [MotionBricks](https://github.com/NVlabs/motionbricks)
applications (upstream `app/`, commit 54fc563), running natively on Apple silicon:

| Binary | Upstream | What it does |
|---|---|---|
| `inbetweener` | `app/inbetweener_demo.py` | place start and target poses, chain generations through them |
| `scene-preview` | `app/scene_preview.py` | drive a character through a scene with locomotion and Smart Primitive interactions |
| `smart-primitive-editor` | `app/smart_primitive_editor.py` | author `.smart_primitive.json` documents |
| `scene-editor` | `app/scene_editor.py` | author `.scene.json` documents |

## The apps call the Rust model and the Rust Runtime API — never Python

The generation apps run the neural model through
[`motionbricks_rs`](../model-rs/README.md), the Rust implementation in `model-rs/`:

- **The model** is the model-specific **Metal runtime**
  (`motionbricks_rs::metal::model::MetalModel`) — the same ~4 ms (BF16) / ~5 ms (FP32)
  engine the `model-rs` benchmarks measure — behind the Runtime API's `Backend` trait.
- **The Runtime API** (`motionbricks_rs::runtime`, a port of upstream's
  `motionbricks/model` + `motionbricks/runtime` + `motionbricks/animation` +
  `motionbricks/io`) provides the engines (`MotionBricksSomaEngine`,
  `MotionBricksUnitreeG1Engine`), motion players, target-pose placement and SOMA IK,
  and BVH/CSV clip loading.
- There is no TensorRT, ONNX Runtime, Warp, or Python anywhere in these apps. (For
  comparison runs, upstream's TensorRT/ONNX path remains available in
  [`sandbox/`](../sandbox/README.md).)

Everything the apps draw goes through `viewer`, a small wgpu viewer with the same
logging-shaped API as upstream's Newton ViewerGL (`log_mesh`, `log_lines`, `log_shapes`,
`log_gizmo`, ground plane, orbit camera, picking, transform gizmo, imgui panels with
docking).

## Layout

| Module | Contents |
|---|---|
| `src/viewer/` | the wgpu viewer: window, cameras, primitive logging, CPU skinning, picking, translate gizmo, imgui harness, and an offscreen headless mode (`viewer::HeadlessViewer`, `src/viewer/headless.rs`) |
| `src/documents/` | `.scene.json` and `.smart_primitive.json` IO (exact upstream formats, atomic writes) |
| `src/rig/` | SOMA skinned mesh (`assets/soma_mesh.bin`, converted from upstream's Apache-2.0 USD by `scripts/convert_soma_mesh.py`), G1 MJCF + STL meshes, clip registry |
| `src/runtime/` | the app runtime: character playback, state machine, locomotion modes and blend spaces, Smart Primitive behaviors, interaction probe |
| `src/bin/` | the four apps above plus `viewer-demo` |
| `assets/` | NVIDIA-authored scene and Smart Primitive documents (Apache-2.0/CC-BY-4.0, see `assets/README.md`) |

## Running

```sh
cd app-rs && cargo build --release        # the first build also compiles MLX (~6 min)
```

The generation apps need the checkpoints (about 1.6 GB, fetched by
`model-rs/scripts/download_weights.sh` or `motionbricks-mlx download`), and clips from
`motionbricks-mlx download --clips` (NVIDIA Sample Data Evaluation License — fetched on
demand, never committed). The G1 meshes and the original SOMA USD live in the upstream
checkout (see `assets/README.md`); the skinned SOMA mesh is pre-converted and committed.

## The apps

All four apps share the viewer's controls: drag to orbit, scroll to zoom, click to
pick, drag the gizmo arrows to translate the selection. The editors keep their
documents in the File menu (New / Open / Save / Save As) and track unsaved changes.

| Binary | Run | Notes |
|---|---|---|
| `inbetweener` | `cargo run --release --bin inbetweener -- --rig soma` | place start and target poses on the timeline and chain generations through them; panel-driven, `--rig soma\|unitree_g1`, `--precision fp32\|bf16` |
| `scene-preview` | `cargo run --release --bin scene-preview -- --scene assets/scenes/soma_gym.scene.json` | drive the character with **WASD**/arrows; **Space** full speed, **I** interact, **O** Forward/Strafing, **P** Crouch, **[** cycle locomotion style |
| `smart-primitive-editor` | `cargo run --release --bin smart-primitive-editor -- assets/smart_primitives/soma/sit.smart_primitive.json` | author `.smart_primitive.json`: node hierarchy with Target Pose, Interaction Box and Pivot components, plus target pose sequences |
| `scene-editor` | `cargo run --release --bin scene-editor -- --scene assets/scenes/soma_gym.scene.json` | author `.scene.json`: Smart Primitive instances, Solid Cubes, and the Character Start |

The generation apps (`inbetweener`, `scene-preview`) need the checkpoints; the
editors and `viewer-demo` run without them.

## Headless captures

`viewer::HeadlessViewer` renders the same logged primitives offscreen (no window,
no imgui) and reads the frame back as an `RgbaImage` — used by the viewer tests
and available for screenshots:

```sh
cargo run --release --bin viewer-demo -- --headless frame.ppm --frames 30   # P6 PPM image
```

## License

Apache-2.0, like the rest of the repository; the apps port NVIDIA's Apache-2.0
`app/` code (see NOTICE). Unitree G1 robot assets are BSD 3-Clause
(`licenses/unitree-LICENSE.txt`).
