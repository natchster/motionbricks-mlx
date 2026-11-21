"""Golden fixtures for the Runtime API, from upstream's own code (the reference oracle).

Upstream's engines need TensorRT and CUDA, but every step around inference is a Warp kernel that
also runs on Warp's CPU backend. This script drives the real upstream engine methods
(``_encode_guide_poses``, ``_decode_generated_poses``, ``_post_process_generated_poses``) on Warp
CPU, with inference done by NVIDIA's PyTorch model in place of TensorRT. It also records upstream's
clip loaders, recorder sampling, target placement, speed scaling, IK pins and buffer sampling.

The Python and Rust runtime tests teacher-force each step against these fixtures.

    uv run --all-packages python sandbox/scripts/make_runtime_golden.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "sandbox" / "upstream"
OUT = ROOT / "sandbox" / "golden" / "runtime"
sys.path.insert(0, str(ROOT / "sandbox"))
sys.path.insert(0, str(UPSTREAM))

import cudashim  # noqa: E402

cudashim.enable_all()  # `import tensorrt` -> the ONNX Runtime shim, so upstream's engine modules import

import warp as wp  # noqa: E402

wp.init()
wp.set_device("cpu")

from cudashim import reference  # noqa: E402
from motionbricks.animation.animation_buffer import AnimationBuffer  # noqa: E402
from motionbricks.animation.skeleton import Skeleton  # noqa: E402
from motionbricks.io.bvh import load_bvh, load_skeleton_from_bvh  # noqa: E402
from motionbricks.io.csv import load_csv  # noqa: E402
from motionbricks.model.config import MotionBricksInferenceConfig  # noqa: E402
from motionbricks.model.soma.buffers import PoseToken  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402
from motionbricks.model.unitree_g1.buffers import DOFToken  # noqa: E402
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine  # noqa: E402
from motionbricks.runtime.motion_recorder import MotionRecorder  # noqa: E402
from motionbricks.runtime.target_pose import (  # noqa: E402
    IKPin,
    TargetPoseIKCorrector,
    adjust_target_dof_speed,
    adjust_target_pose_speed,
    create_soma_target_pose_ik_rig,
    place_target_dof,
    place_target_poses,
)
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

CLIPS = {
    "soma": ("assets/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh",
             "assets/motions/soma/locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534.bvh"),
    "unitree_g1": ("assets/motions/unitree_g1/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.csv",
                   "assets/motions/unitree_g1/locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534.csv"),
}


def fetch_clips() -> None:
    spec = importlib.util.spec_from_file_location("fetch_assets", ROOT / "sandbox/scripts/fetch_assets.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    for paths in CLIPS.values():
        for rel in paths:
            fetch.fetch_lfs(rel)


def no_root_skeleton(path: Path) -> Skeleton:
    """The BVH skeleton without its identity ``Root`` joint (upstream's "no_root" SOMA skeleton)."""
    full = load_skeleton_from_bvh(str(path))
    ref = full.reference_local_transforms.numpy()
    parents = [p - 1 if p > 0 else -1 for p in full._np_parent_indices[1:]]
    root, hips = wp.transform(*ref[0, :3], *ref[0, 3:]), wp.transform(*ref[1, :3], *ref[1, 3:])
    ref = ref[1:].copy()
    ref[0] = np.array(wp.mul(root, hips))
    return Skeleton(full.joint_names[1:], parents, ref)


class Oracle:
    """An upstream engine with inference by NVIDIA's PyTorch model instead of TensorRT."""

    def __init__(self, engine_cls, model_to_runtime, torch_model, name):
        class OracleEngine(engine_cls):  # inference properties replaced; everything else is upstream's
            stream = None
            device = wp.get_device("cpu")

        engine = OracleEngine.__new__(OracleEngine)
        engine._model_to_runtime = wp.transform(model_to_runtime)
        engine._runtime_to_model = wp.transform_inverse(engine._model_to_runtime)
        engine._alignment = wp.empty(1, dtype=wp.transform)
        engine._generated_frame_count = wp.empty(1, dtype=wp.int32)
        engine._tensors = {spec.name: wp.zeros(spec.shape, dtype=wp.dtype_from_numpy(spec.dtype)) for spec in engine.contract.tensor_specs}
        engine.device_tensor = lambda tensor: engine._tensors[tensor]
        engine._ensure_open = lambda: None
        engine._initialize_pose_layout()
        self.engine, self.torch_model, self.name = engine, torch_model, name

    def generate(self, context, target, output, config, record: dict, prefix: str) -> None:
        e = self.engine
        e._prepare_generation(context, target, output)
        e._encode_guide_poses(context, target, config)
        inputs = {spec.name: e._tensors[spec.name].numpy().copy() for spec in e.contract.tensor_specs if spec.mode.value == "input"}
        outputs = reference.run(self.torch_model, self.name, inputs)
        for key, value in outputs.items():
            e._tensors[key].assign(np.asarray(value).astype(e._tensors[key].numpy().dtype))
        e._decode_generated_poses(output)
        count = int(e._generated_frame_count.numpy()[0])
        record[f"{prefix}.decoded"] = output.poses.numpy()[:count].copy()
        record[f"{prefix}.decoded_contacts"] = output.foot_contacts.numpy()[:count].copy()
        if config.correct_generated_diff:
            e._post_process_generated_poses(context, target, output)
        record[f"{prefix}.generated"] = output.poses.numpy()[:count].copy()
        record[f"{prefix}.frame_count"] = np.array(count, np.int32)
        record[f"{prefix}.alignment"] = e._alignment.numpy()[0].copy()
        for key, value in inputs.items():
            record[f"{prefix}.in.{key}"] = value
        for key, value in outputs.items():
            record[f"{prefix}.out.{key}"] = np.asarray(value)


def placement(yaw_degrees: float, x: float, y: float) -> wp.transform:
    return wp.transform(wp.vec3(x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.radians(yaw_degrees)))


def window(clip, start: float, offset, n: int = 4) -> np.ndarray:
    poses = []
    for i in range(n):
        if isinstance(clip, AnimationBuffer):
            pose = wp.empty(clip.skeleton.num_joints, dtype=wp.transform)
        else:
            pose = wp.empty(clip.num_coordinates, dtype=wp.float32)
        clip.sample(start + i / 30.0, pose, root_offset=offset)
        poses.append(pose.numpy())
    return np.stack(poses)


def main() -> None:
    fetch_clips()
    OUT.mkdir(parents=True, exist_ok=True)
    mujoco = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
    for name in ("soma", "unitree_g1"):
        rec: dict[str, np.ndarray] = {}
        walk_path, jog_path = (UPSTREAM / p for p in CLIPS[name])
        torch_model = reference.build(name)
        if name == "soma":
            skeleton = no_root_skeleton(walk_path)
            walk, jog = load_bvh(str(walk_path), skeleton), load_bvh(str(jog_path), skeleton)
            rec["skeleton.parents"] = skeleton._np_parent_indices.copy()
            rec["skeleton.reference"] = skeleton.reference_local_transforms.numpy()
            # Joint names as UTF-8 bytes joined by newlines (readable without NumPy string arrays).
            rec["skeleton.names_utf8"] = np.frombuffer("\n".join(skeleton.joint_names).encode(), np.uint8).astype(np.int32)
            m2r_default = mujoco
            engine_cls, token = MotionBricksSOMAEngine, (lambda: PoseToken(skeleton.num_joints))
            new_output = lambda e: e.create_output(skeleton)  # noqa: E731
        else:
            walk, jog = load_csv(walk_path), load_csv(jog_path)
            m2r_default = wp.transform_identity()
            engine_cls, token = MotionBricksUnitreeG1Engine, (lambda: DOFToken(29))
            new_output = lambda e: e.create_output()  # noqa: E731
        # Clip loading: a few frames of each clip as upstream's loader produced them.
        for label, clip in (("walk", walk), ("jog", jog)):
            rec[f"clip.{label}.frames"] = clip.poses.numpy()[[0, 1, 2, 100, 500, clip.num_frames - 1]].copy()
            rec[f"clip.{label}.frame_index"] = np.array([0, 1, 2, 100, 500, clip.num_frames - 1], np.int32)
            rec[f"clip.{label}.num_frames"] = np.array(clip.num_frames, np.int32)
            rec[f"clip.{label}.sample_rate"] = np.array(clip.sample_rate, np.float32)

        # Generations: (label, model_to_runtime, context clip/time/placement, target clip/time/placement, poses, config).
        far = placement(70.0, 15.0, -8.0)
        scenarios = [
            ("walk", m2r_default, walk, 1.0, None, walk, 1.8, None, 4, MotionBricksInferenceConfig()),
            ("jog_one_pose", m2r_default, jog, 2.0, far, jog, 3.0, far, 1,
             MotionBricksInferenceConfig(allowed_token_counts=(6, 7, 8, 9, 10), heading_alignment=(0.6, 0.8))),
            ("no_correction", m2r_default, walk, 2.0, None, walk, 2.6, placement(-30.0, 0.5, 0.2), 2,
             MotionBricksInferenceConfig(correct_generated_diff=False, allowed_token_counts=(12, 13, 14, 15, 16))),
            ("model_to_runtime", wp.mul(placement(25.0, -3.0, 4.0), m2r_default), jog, 1.2, placement(25.0, -3.0, 4.0),
             jog, 2.4, placement(25.0, -3.0, 4.0), 3, MotionBricksInferenceConfig()),
        ]
        for label, m2r, cclip, ctime, coff, tclip, ttime, toff, n, config in scenarios:
            oracle = Oracle(engine_cls, m2r, torch_model, name)
            cof = wp.mul(coff, m2r) if coff is not None else m2r
            tof = wp.mul(toff, m2r) if toff is not None else m2r
            context, target = token(), token()
            ctx = window(cclip, ctime - 3 / 30.0, cof)
            tgt = window(tclip, ttime, tof)[:n]
            context.update_from_poses(wp.array(ctx, dtype=context.poses.dtype))
            target.update_from_poses(wp.array(tgt, dtype=target.poses.dtype))
            rec[f"gen.{label}.model_to_runtime"] = np.array(m2r, np.float32)
            rec[f"gen.{label}.context"] = ctx
            rec[f"gen.{label}.target"] = tgt
            rec[f"gen.{label}.allowed"] = np.array(config.allowed_token_counts, np.int32)
            rec[f"gen.{label}.heading"] = np.array(config.heading_alignment, np.float32)
            rec[f"gen.{label}.correct"] = np.array(config.correct_generated_diff)
            oracle.generate(context, target, new_output(oracle.engine), config, rec, f"gen.{label}")
            print(f"{name} {label}: {int(rec[f'gen.{label}.frame_count'])} frames")

        # Recorder: irregular update steps, then four context poses resampled at 30 Hz.
        dts = np.array([1 / 60, 1 / 45, 1 / 120, 1 / 30, 1 / 90, 1 / 60, 1 / 24, 1 / 75, 1 / 60, 1 / 50, 1 / 100, 1 / 60], np.float32)
        poses = window(walk, 0.5, m2r_default, len(dts) + 1)
        history = MotionRecorder(wp.array(poses[0], dtype=wp.transform if name == "soma" else wp.float32), capacity=8)
        for dt, pose in zip(dts, poses[1:]):
            history.record(float(dt), wp.array(pose, dtype=history.poses.dtype))
        tok = token()
        tok.update_from_recorder(history)
        rec["recorder.poses"], rec["recorder.dts"] = poses, dts
        rec["recorder.history"], rec["recorder.delta_times"] = history.poses.numpy(), history.delta_times.copy()
        rec["recorder.token"] = tok.poses.numpy()

        # Target placement and speed scaling.
        targets = window(jog, 1.5, m2r_default)
        rec["target.poses"] = targets
        placed = wp.empty_like(wp.array(targets, dtype=tok.poses.dtype))
        place = place_target_poses if name == "soma" else place_target_dof
        place(wp.vec3(3.0, -1.5, 0.0), wp.vec3(0.3, 0.9, 0.0), wp.array(targets, dtype=tok.poses.dtype), placed)
        rec["target.placed"] = placed.numpy()
        for speed in (0.0, 1.3):
            scaled = wp.array(targets, dtype=tok.poses.dtype)
            (adjust_target_pose_speed if name == "soma" else adjust_target_dof_speed)(scaled, speed)
            rec[f"target.speed_{speed:g}"] = scaled.numpy()

        # Buffer sampling with a root offset, foot contacts, and make_loopable.
        offset = placement(40.0, 1.0, 2.0)
        for t in (0.0, 0.37, 1.234, 99.0):
            pose = wp.empty(skeleton.num_joints if name == "soma" else walk.num_coordinates,
                            dtype=wp.transform if name == "soma" else wp.float32)
            walk.sample(t, pose, root_offset=offset)
            rec[f"sample.{t:g}"] = pose.numpy()
        loop = walk.copy()
        loop.num_frames = 40
        loop.make_loopable()
        rec["loop.before"] = walk.poses.numpy()[:40].copy()
        rec["loop.after"] = loop.poses.numpy()[:40].copy()

        if name == "soma":
            corrector = TargetPoseIKCorrector(skeleton, create_soma_target_pose_ik_rig(skeleton))
            target_pose = wp.array(window(jog, 2.2, mujoco, 1)[0], dtype=wp.transform)
            current = window(walk, 3.1, mujoco, 1)[0]
            rec["ik.target"], rec["ik.current"] = target_pose.numpy(), current
            for label, mask, weight in (("hands", IKPin.LEFT_HAND | IKPin.RIGHT_FOOT, 0.7), ("all", IKPin.ALL, 1.0)):
                corrector.capture_pins(target_pose, mask)
                pose = wp.array(current, dtype=wp.transform)
                corrector.apply(pose, weight)
                rec[f"ik.{label}.mask"], rec[f"ik.{label}.weight"] = np.array(int(mask), np.int32), np.array(weight, np.float32)
                rec[f"ik.{label}.result"] = pose.numpy()

        np.savez_compressed(OUT / f"{name}.npz", **rec)
        print(f"wrote {OUT / f'{name}.npz'} ({len(rec)} arrays)")


if __name__ == "__main__":
    main()
