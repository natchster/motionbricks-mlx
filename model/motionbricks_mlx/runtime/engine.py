# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (model/engine.py,
# model/generation.py, model/soma/engine.py, model/soma/kernels.py, model/unitree_g1/engine.py,
# model/unitree_g1/kernels.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Engines: poses in world space -> model inputs -> MLX inference -> generated motion.

Upstream's engines run a TensorRT plan on a CUDA stream; these run a ``motionbricks_mlx`` model
(any precision) and keep the same flow and math: align the context and target to a heading,
encode them into the ONNX-named inputs, run the model, decode the prediction back to world
space, drop the boundary frames, and correct the endpoints.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import transforms as T
from .animation import AnimationBuffer, DOFAnimationBuffer, Skeleton, read_root, write_root
from .contract import (
    FOOT_CONTACT_COUNT,
    MODEL_BOUNDARY_FRAMES,
    MODEL_CONTEXT_AND_TARGET_POSES,
    MODEL_MAX_VALID_FRAMES,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_MAX_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    ROOT_COORDINATE_COUNT,
    SOMA_HEADING_JOINTS,
    SOMA_JOINT_NAMES,
    UNITREE_G1_JOINT_NAMES,
    MotionBricksInferenceConfig,
    ModelRig,
    valid_frame_count,
)
from .tokens import DOFToken, PoseToken

F = np.float32
GENERATION_STAGES = ("encode", "inference", "decode", "post_process")
# G1 root position filter at 30 Hz: keeps the gait below 4 Hz, removes jitter above 8 Hz.
ROOT_POSITION_FILTER = np.array((-0.0465, 0.0639, 0.2827, 0.3998, 0.2827, 0.0639, -0.0465), F)
# G1 joint-angle endpoint correction fades in and out over this time at each end.
JOINT_ANGLE_FADE_SECONDS = 0.3


class GenerationTimer:
    """Wall-clock time of each stage of the last generation, in ``milliseconds``."""

    def __init__(self) -> None:
        self.milliseconds = dict.fromkeys(GENERATION_STAGES, 0.0)
        self._timed: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        yield
        self._timed[name] = (time.perf_counter() - start) * 1e3

    def read(self) -> None:
        for name in GENERATION_STAGES:
            self.milliseconds[name] = self._timed.get(name, 0.0)
        self._timed.clear()


def conditions(num_target_poses: int, config: MotionBricksInferenceConfig) -> dict[str, np.ndarray]:
    """Which context/target slots are given, and which token counts the model may choose.

    Local root values stay unmarked at the last context and last target pose, where their
    derivatives are unknown.
    """
    index = np.arange(MODEL_CONTEXT_AND_TARGET_POSES)
    given = (index < MODEL_POSES_PER_TOKEN + num_target_poses).astype(np.int32)
    local = np.where((index == MODEL_POSES_PER_TOKEN - 1) | (index == MODEL_POSES_PER_TOKEN + num_target_poses - 1), 0, given)
    bits = config.allowed_token_bits()
    allowed = np.array([(bits >> i) & 1 for i in range(MODEL_MAX_TOKEN_COUNT - MODEL_MIN_TOKEN_COUNT + 1)], np.int32)
    return {
        "has_global_root_values": given[None].copy(),
        "has_local_root_values": local[None].astype(np.int32),
        "has_local_poses": given[None].copy(),
        "allowed_pred_num_tokens": allowed[None],
    }


def check_generation_buffers(context, target, output) -> None:
    if context.num_poses != MODEL_POSES_PER_TOKEN:
        raise ValueError(f"Context token must contain {MODEL_POSES_PER_TOKEN} poses.")
    if not 1 <= target.num_poses <= MODEL_POSES_PER_TOKEN:
        raise ValueError(f"Target token must contain 1 to {MODEL_POSES_PER_TOKEN} poses.")
    if output.sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"Output must use a sample rate of {MODEL_SAMPLE_RATE:g} Hz.")
    if output.capacity < MODEL_MAX_VALID_FRAMES:
        raise ValueError(f"Output capacity must be at least {MODEL_MAX_VALID_FRAMES} frames.")
    if output.foot_contacts is None:
        raise ValueError("Output must have foot_contacts. create_output() allocates them.")


class MotionBricksEngine:
    """Generate motion from four context poses and one to four target poses (batch 1).

    Args:
        model: A loaded ``motionbricks_mlx.MotionBricks`` of the engine's rig, in any precision
            (``mb.load(...)``, optionally ``.quantize(8)``), or any backend with
            ``run(inputs: dict) -> dict`` on the ONNX tensor names. ``None`` builds an engine
            without inference, whose encode/decode stages still work.
        model_to_runtime: Transform ``[7]`` from model space to world space (default identity).
        compiled: Run the model as one ``mx.compile`` graph (default).
    """

    rig: ModelRig
    joint_names: tuple[str, ...]

    def __init__(self, model, *, model_to_runtime=T.IDENTITY, compiled: bool = True) -> None:
        spec = getattr(model, "spec", None)
        if spec is not None and spec.name != self.rig.value:
            raise ValueError(f"{type(self).__name__} needs a {self.rig.value} model, got {spec.name}.")
        self.model = model
        self.compiled = compiled
        self._model_to_runtime = np.asarray(model_to_runtime, F).copy()
        self._runtime_to_model = T.transform_inverse(self._model_to_runtime)
        self.timer = GenerationTimer()
        self.alignment = T.IDENTITY.copy()
        self.generated_frame_count = 0
        self.stream = None  # upstream API compatibility: there are no streams here
        self.device = "gpu"
        self._open = True

    @classmethod
    def from_checkpoints(cls, checkpoint_dir: str | Path, *, dtype: str = "float32", quantize: int | None = None, **kwargs):
        """Load the rig's model from its checkpoint folder and wrap it in an engine."""
        from ..model import load

        model = load(cls.rig.value, checkpoint_dir, dtype=dtype)
        if quantize:
            model.quantize(bits=quantize)
        return cls(model, **kwargs)

    @property
    def model_to_runtime(self) -> np.ndarray:
        return self._model_to_runtime.copy()

    def close(self) -> None:
        self._open = False

    def synchronize(self) -> None:
        """Generation is synchronous here; kept for source compatibility with upstream."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if not self._open:
            raise RuntimeError("The engine is closed.")

    def run_model(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Inference on the ONNX-named inputs (the step upstream runs with TensorRT)."""
        if self.model is None:
            raise RuntimeError("This engine has no model (it was built with model=None).")
        if hasattr(self.model, "spec"):  # a motionbricks_mlx.MotionBricks
            return self.model.run(inputs, compiled=self.compiled)
        return self.model.run(inputs)

    def generate_animation(self, context, target, output, *, config: MotionBricksInferenceConfig | None = None, stream=None):
        """Generate motion at 30 Hz into ``output`` and return it.

        ``output.num_frames`` gives the valid frames (18 to 58) and ``output.foot_contacts`` the
        predicted contacts. ``timer.milliseconds`` then holds the time of each stage.
        """
        self._ensure_open()
        config = config or MotionBricksInferenceConfig()
        self._prepare_generation(context, target, output)
        output.num_frames = 0
        with self.timer.stage("encode"):
            inputs = self.encode_inputs(context, target, config)
        with self.timer.stage("inference"):
            prediction = self.run_model(inputs)
        with self.timer.stage("decode"):
            self.decode_prediction(prediction, output)
        if config.correct_generated_diff:
            with self.timer.stage("post_process"):
                self.correct_endpoints(context, target, output)
        self.timer.read()
        token_count = int(prediction["pred_num_tokens"][0])
        if token_count not in config.allowed_token_counts:
            raise RuntimeError(f"Model selected {token_count} tokens, outside the allowed set {config.allowed_token_counts}.")
        output.num_frames = valid_frame_count(token_count)
        return output

    def _frame_count(self, prediction, frames_in_prediction: int) -> int:
        raw = min(max(int(prediction["pred_num_tokens"][0]) * MODEL_POSES_PER_TOKEN, 0), frames_in_prediction)
        return max(0, raw - 2 * MODEL_BOUNDARY_FRAMES)

    # Steps supplied by each rig (public, so they can be checked one at a time).
    def _prepare_generation(self, context, target, output) -> None:
        raise NotImplementedError

    def encode_inputs(self, context, target, config: MotionBricksInferenceConfig) -> dict[str, np.ndarray]:
        """Align the context and target, and return the model's ONNX-named inputs (batch 1)."""
        raise NotImplementedError

    def decode_prediction(self, prediction: dict[str, np.ndarray], output) -> int:
        """Decode the prediction into ``output`` (world space, boundary frames dropped); returns the frame count."""
        raise NotImplementedError

    def correct_endpoints(self, context, target, output) -> None:
        """Match the generated motion's first and last frames to the last context and first target pose."""
        raise NotImplementedError


# --- SOMA ------------------------------------------------------------------------------------


def _heading_from_hip_positions(right, left) -> np.ndarray:
    """Heading about +Y from the leg joints (training's ``compute_heading_method="hips_pos"``)."""
    right, left = np.asarray(right, F), np.asarray(left, F)
    across = np.stack([right[..., 0] - left[..., 0], np.zeros_like(right[..., 0]), right[..., 2] - left[..., 2]], -1)
    length = np.sqrt((across * across).sum(-1, keepdims=True))
    across = across / np.where(length > 1.0e-6, length, 1)
    forward = np.cross(np.array([0.0, 1.0, 0.0], F), across)
    return np.where(length[..., 0] > 1.0e-6, np.arctan2(forward[..., 0], forward[..., 2]), 0.0).astype(F)


def _rotation_from_6d(x_raw, y_raw) -> np.ndarray:
    x_axis = T.normalize(x_raw)
    z_axis = T.normalize(np.cross(x_axis, y_raw))
    y_axis = np.cross(z_axis, x_axis)
    return T.quat_from_matrix(T.matrix_from_cols(x_axis, y_axis, z_axis))


class MotionBricksSOMAEngine(MotionBricksEngine):
    """SOMA engine: poses are one local transform per skeleton joint, root (``Hips``) first.

    The skeleton may hold more joints than the model's 30 (fingers, ...): those keep their
    reference transforms in generated motion. The model's joints are found by name.
    """

    rig = ModelRig.SOMA
    joint_names = SOMA_JOINT_NAMES

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._pose_skeleton = None

    def create_input(self, skeleton: Skeleton) -> PoseToken:
        """A context or target token for poses of ``skeleton``."""
        self._ensure_open()
        self._bind_pose_skeleton(skeleton)
        return PoseToken(skeleton.num_joints)

    def create_output(self, skeleton: Skeleton) -> AnimationBuffer:
        """An empty buffer with room for the longest generated motion (58 frames)."""
        self._ensure_open()
        self._bind_pose_skeleton(skeleton)
        poses = np.broadcast_to(skeleton.reference_local_transforms, (MODEL_MAX_VALID_FRAMES, skeleton.num_joints, 7)).copy()
        return AnimationBuffer(skeleton, MODEL_SAMPLE_RATE, poses, num_frames=0,
                               foot_contacts=np.zeros((MODEL_MAX_VALID_FRAMES, FOOT_CONTACT_COUNT), F))

    def _prepare_generation(self, context: PoseToken, target: PoseToken, output: AnimationBuffer) -> None:
        self._bind_pose_skeleton(output.skeleton)
        for name, token in (("Context", context), ("Target", target)):
            if token.num_joints != output.skeleton.num_joints:
                raise ValueError(f"{name} token must contain {output.skeleton.num_joints} joints per pose.")
        check_generation_buffers(context, target, output)

    def _bind_pose_skeleton(self, skeleton: Skeleton) -> None:
        if skeleton is self._pose_skeleton:
            return
        model_to_pose = [skeleton.joint_index(name) for name in self.joint_names]
        missing = [n for n, i in zip(self.joint_names, model_to_pose) if i == -1]
        if missing:
            raise ValueError("The skeleton is missing model joints: " + ", ".join(missing))
        if model_to_pose[0] != 0:
            raise ValueError("The skeleton must store the root (Hips) at index zero.")
        pose_to_model = np.full(skeleton.num_joints, -1, np.int32)
        for model_joint, pose_joint in enumerate(model_to_pose):
            parent = skeleton.joint_parent(pose_joint)
            # Model joints outside the connected hierarchy keep their reference transforms.
            if parent == -1 or pose_to_model[parent] >= 0:
                pose_to_model[pose_joint] = model_joint
        self._model_to_pose = np.asarray(model_to_pose, np.int32)
        self._pose_to_model = pose_to_model
        self._right_leg = skeleton.joint_index(SOMA_HEADING_JOINTS[0])
        self._left_leg = skeleton.joint_index(SOMA_HEADING_JOINTS[1])
        self._pose_skeleton = skeleton

    def encode_inputs(self, context: PoseToken, target: PoseToken, config: MotionBricksInferenceConfig) -> dict[str, np.ndarray]:
        sk = self._pose_skeleton
        world = np.concatenate([
            T.compute_world_poses(context.poses, sk.parent_indices, self._runtime_to_model),
            T.compute_world_poses(target.poses, sk.parent_indices, self._runtime_to_model),
        ])  # [8, J, 7] in SOMA model space
        last = MODEL_POSES_PER_TOKEN - 1
        hips = world[last, 0, :3]
        current = _heading_from_hip_positions(world[last, self._right_leg, :3], world[last, self._left_leg, :3])
        desired = math.atan2(config.heading_alignment[0], config.heading_alignment[1])
        self.alignment = T.transform([hips[0], 0.0, hips[2]], T.quat_from_axis_angle([0.0, 1.0, 0.0], F(current - desired)))
        align = self.alignment

        def aligned(p):  # subtract before rotating to keep precision far from the origin
            return T.quat_rotate_inv(align[3:], p - align[:3])

        frames = np.arange(MODEL_CONTEXT_AND_TARGET_POSES)
        nxt = np.where(frames == last, frames, np.minimum(frames + 1, MODEL_CONTEXT_AND_TARGET_POSES - 1))
        hips_now, hips_next = aligned(world[:, 0, :3]), aligned(world[nxt, 0, :3])
        heading = _heading_from_hip_positions(aligned(world[:, self._right_leg, :3]), aligned(world[:, self._left_leg, :3]))
        heading_next = heading[nxt]
        delta = np.arctan2(np.sin(heading_next - heading), np.cos(heading_next - heading)).astype(F)

        global_root = np.stack([hips_now[:, 0], hips_now[:, 1], hips_now[:, 2], np.cos(heading), np.sin(heading)], -1)
        local_root = np.stack([delta * F(MODEL_SAMPLE_RATE), (hips_next[:, 0] - hips_now[:, 0]) * F(MODEL_SAMPLE_RATE),
                               (hips_next[:, 2] - hips_now[:, 2]) * F(MODEL_SAMPLE_RATE), hips_now[:, 1]], -1)
        jm = len(self.joint_names)
        joints = world[:, self._model_to_pose]  # [8, 30, 7]
        positions = aligned(joints[..., :3]) - np.stack([hips_now[:, 0], np.zeros(8, F), hips_now[:, 2]], -1)[:, None]
        rotations = T.quat_mul(T.quat_inverse(align[3:]), joints[..., 3:])
        x_axis = T.quat_rotate(rotations, np.array([1.0, 0.0, 0.0], F))
        y_axis = T.quat_rotate(rotations, np.array([0.0, 1.0, 0.0], F))
        local_poses = np.concatenate([positions[:, 1:].reshape(8, (jm - 1) * 3),
                                      np.concatenate([x_axis, y_axis], -1).reshape(8, jm * 6)], -1)
        inputs = {
            "global_root_values": global_root.astype(F)[None],
            "local_root_values": local_root.astype(F)[None],
            "local_poses": local_poses.astype(F)[None],
        }
        inputs.update(conditions(target.num_poses, config))
        return inputs

    def decode_prediction(self, prediction: dict[str, np.ndarray], output: AnimationBuffer) -> int:
        sk = self._pose_skeleton
        pred = np.asarray(prediction["pred_global_motions"], F)[0]  # [64, 366]
        count = self._frame_count(prediction, pred.shape[0])
        self.generated_frame_count = count
        jm = len(self.joint_names)
        raw = pred[MODEL_BOUNDARY_FRAMES: MODEL_BOUNDARY_FRAMES + count]
        output.foot_contacts[:count] = np.clip(raw[:, -FOOT_CONTACT_COUNT:], 0.0, 1.0)
        # Layout: global root [5], child joint positions [3 (J-1)], global 6D rotations [6 J], ...
        rot_offset = 5 + (jm - 1) * 3
        six = raw[:, rot_offset: rot_offset + 6 * jm].reshape(count, jm, 6)
        rotations = _rotation_from_6d(six[..., :3], six[..., 3:])  # [count, 30, 4]
        hips = T.transform_mul(self.alignment, T.transform(raw[:, :3], rotations[:, 0]))
        output.poses[:count, 0] = T.transform_mul(self._model_to_runtime, hips)
        ref = sk.reference_local_transforms
        for joint in range(1, sk.num_joints):
            model_joint = self._pose_to_model[joint]
            if model_joint < 0:
                output.poses[:count, joint] = ref[joint]
                continue
            parent_model = self._pose_to_model[sk.parent_indices[joint]]
            local = T.quat_mul(T.quat_inverse(rotations[:, parent_model]), rotations[:, model_joint])
            output.poses[:count, joint] = T.transform(np.broadcast_to(ref[joint, :3], (count, 3)), T.quat_normalize(local))
        return count

    def correct_endpoints(self, context: PoseToken, target: PoseToken, output: AnimationBuffer) -> None:
        count = self.generated_frame_count
        if count < 2:
            return
        p, last = output.poses, count - 1
        start_root = T.compute_root_endpoint_delta(p[0, 0], context.poses[-1, 0])
        end_root = T.compute_root_endpoint_delta(p[last, 0], target.poses[0, 0])
        start = T.compute_delta_transform(p[0, 1:], context.poses[-1, 1:])
        end = T.compute_delta_transform(p[last, 1:], target.poses[0, 1:])
        n = start.shape[0]
        ew = (np.arange(count, dtype=F) / F(last)).astype(F)
        sw = (F(1.0) - ew).astype(F)
        p[:count, 0] = T.apply_root_endpoint_correction(p[:count, 0], start_root, end_root, ew)
        corrected = T.apply_delta_transform(p[:count, 1:], start, np.repeat(sw[:, None], n, 1))
        p[:count, 1:] = T.apply_delta_transform(corrected, end, np.repeat(ew[:, None], n, 1))


# --- Unitree G1 ------------------------------------------------------------------------------


class MotionBricksUnitreeG1Engine(MotionBricksEngine):
    """G1 engine: poses are 36 joint coordinates (root XYZ, XYZW quaternion, 29 angles in radians).

    Model space is Z up with +X forward. The root position of the decoded motion is filtered
    over neighboring frames, and joint-angle endpoint corrections fade over 0.3 s at each end.
    """

    rig = ModelRig.UNITREE_G1
    joint_names = UNITREE_G1_JOINT_NAMES

    def create_input(self) -> DOFToken:
        self._ensure_open()
        return DOFToken(len(self.joint_names))

    def create_output(self) -> DOFAnimationBuffer:
        self._ensure_open()
        poses = np.zeros((MODEL_MAX_VALID_FRAMES, ROOT_COORDINATE_COUNT + len(self.joint_names)), F)
        poses[:, 6] = 1.0
        return DOFAnimationBuffer(self.joint_names, MODEL_SAMPLE_RATE, poses, num_frames=0,
                                  foot_contacts=np.zeros((MODEL_MAX_VALID_FRAMES, FOOT_CONTACT_COUNT), F))

    def _prepare_generation(self, context: DOFToken, target: DOFToken, output: DOFAnimationBuffer) -> None:
        n = len(self.joint_names)
        for name, token in (("Context", context), ("Target", target)):
            if token.num_dofs != n:
                raise ValueError(f"{name} token must contain {n} joint angles per pose.")
        if output.num_dofs != n:
            raise ValueError(f"Output must contain {n} joint angles per frame.")
        check_generation_buffers(context, target, output)

    def encode_inputs(self, context: DOFToken, target: DOFToken, config: MotionBricksInferenceConfig) -> dict[str, np.ndarray]:
        root = T.transform_mul(self._runtime_to_model, read_root(context.poses[MODEL_POSES_PER_TOKEN - 1]))
        forward = T.transform_vector(root, np.array([1.0, 0.0, 0.0], F))
        current = math.atan2(float(forward[1]), float(forward[0]))
        desired = math.atan2(config.heading_alignment[1], config.heading_alignment[0])
        self.alignment = T.transform([root[0], root[1], 0.0], T.quat_from_axis_angle([0.0, 0.0, 1.0], F(current - desired)))
        align = self.alignment
        inputs = {}
        for token, name in ((context, "context_qpos"), (target, "target_qpos")):
            roots = T.transform_mul(self._runtime_to_model, read_root(token.poses))
            position = T.quat_rotate_inv(align[3:], roots[:, :3] - align[:3])
            rotation = T.quat_mul(T.quat_inverse(align[3:]), roots[:, 3:])
            qpos = token.poses.copy()
            write_root(T.transform(np.stack([position[:, 0], position[:, 1], roots[:, 2]], -1), rotation), qpos)
            inputs[name] = qpos[None]
        inputs.update(conditions(target.num_poses, config))
        return inputs

    def decode_prediction(self, prediction: dict[str, np.ndarray], output: DOFAnimationBuffer) -> int:
        pred = np.asarray(prediction["pred_qpos"], F)[0]  # [64, 36]
        contacts = np.asarray(prediction["pred_foot_contacts"], F)[0]
        raw_count = min(max(int(prediction["pred_num_tokens"][0]) * MODEL_POSES_PER_TOKEN, 0), pred.shape[0])
        count = max(0, raw_count - 2 * MODEL_BOUNDARY_FRAMES)
        self.generated_frame_count = count
        if count == 0:
            return 0
        radius = len(ROOT_POSITION_FILTER) // 2
        roots = read_root(pred)
        raw = np.arange(count) + MODEL_BOUNDARY_FRAMES
        output.foot_contacts[:count] = np.clip(contacts[raw], 0.0, 1.0)
        position = np.zeros((count, 3), F)
        for tap, weight in enumerate(ROOT_POSITION_FILTER):  # boundary frames give the neighbors
            position += weight * roots[np.clip(raw + tap - radius, 0, raw_count - 1), :3]
        root = T.transform_mul(self.alignment, T.transform(position, roots[raw, 3:]))
        root = T.transform_mul(self._model_to_runtime, T.transform(np.stack([root[:, 0], root[:, 1], position[:, 2]], -1), root[:, 3:]))
        write_root(root, output.poses[:count])
        output.poses[:count, ROOT_COORDINATE_COUNT:] = pred[raw, ROOT_COORDINATE_COUNT:]
        return count

    def correct_endpoints(self, context: DOFToken, target: DOFToken, output: DOFAnimationBuffer) -> None:
        count = self.generated_frame_count
        if count < 2:
            return
        p, last = output.poses, count - 1
        deltas = np.zeros((2, p.shape[1]), F)
        for row, frame, guide in ((0, 0, context.poses[-1]), (1, last, target.poses[0])):
            write_root(T.compute_root_endpoint_delta(read_root(p[frame]), read_root(guide)), deltas[row])
            # Joint angle differences are scalar offsets, without wrapping at +-pi.
            deltas[row, ROOT_COORDINATE_COUNT:] = guide[ROOT_COORDINATE_COUNT:] - p[frame, ROOT_COORDINATE_COUNT:]
        start_root, end_root = read_root(deltas[0]), read_root(deltas[1])
        # Cap the fade so the start and end correction windows do not overlap.
        fade = min(F(JOINT_ANGLE_FADE_SECONDS * MODEL_SAMPLE_RATE), F(0.5) * F(last))
        frames = np.arange(count, dtype=F)
        ew = (frames / F(last)).astype(F)
        write_root(T.apply_root_endpoint_correction(read_root(p[:count]), start_root, end_root, ew), p[:count])
        sw_angle = T.quadratic_fade_weight(frames / fade)[:, None]
        ew_angle = T.quadratic_fade_weight((F(last) - frames) / fade)[:, None]
        p[:count, ROOT_COORDINATE_COUNT:] += deltas[0, ROOT_COORDINATE_COUNT:] * sw_angle + deltas[1, ROOT_COORDINATE_COUNT:] * ew_angle
