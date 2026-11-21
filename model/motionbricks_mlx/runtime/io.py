# SPDX-License-Identifier: Apache-2.0
# Runtime API of motionbricks_mlx, ported from NVlabs/motionbricks (io/bvh.py, io/csv.py,
# utils/space_conversion_utils.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Clips (BVH for SOMA, CSV for the G1) and the rig-space conventions."""

from __future__ import annotations

import csv
from enum import IntEnum, auto
from itertools import chain
from pathlib import Path

import numpy as np

from . import transforms as T
from .animation import AnimationBuffer, DOFAnimationBuffer, Skeleton, create_animation_buffer_for_skeleton
from .contract import ROOT_COORDINATE_COUNT, ModelRig

F = np.float32
CSV_SAMPLE_RATE = 120.0  # robot motion CSV clips; the files do not store it
_CSV_ROOT_TITLE = ["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"]


# --- BVH ---------------------------------------------------------------------------------------


def _open_bvh(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Could not find BVH file: {path}.")
    if path.suffix.lower() != ".bvh":
        raise ValueError(f"Expected a BVH file: {path}.")
    return path.open(encoding="utf-8")


def _read_hierarchy(source, remove_namespace: bool):
    names, parents, offsets, channels = [], [], [], []
    stack: list[int] = []
    in_end_site = False
    for line in source:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "MOTION":
            break
        if tokens[0] in ("ROOT", "JOINT"):
            names.append(tokens[1].rsplit(":", 1)[-1] if remove_namespace else tokens[1])
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
        elif tokens[:2] == ["End", "Site"]:
            in_end_site = True  # the end of a chain, not another animated joint
        elif tokens[0] == "OFFSET" and not in_end_site:
            offsets[stack[-1]] = [float(v) for v in tokens[1:4]]
        elif tokens[0] == "CHANNELS":
            listed = tokens[2:]
            if int(tokens[1]) != len(listed):
                raise ValueError(f"BVH joint {names[stack[-1]]!r} declares {tokens[1]} channels but lists {len(listed)}.")
            channels[stack[-1]] = listed
        elif tokens[0] == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
    reference = np.zeros((len(names), 7), F)
    reference[:, :3] = np.asarray(offsets, F) * F(0.01)
    reference[:, 6] = 1.0
    return Skeleton(names, parents, reference), channels


def _euler_to_quaternion(angles_deg: np.ndarray, order: list[int]) -> np.ndarray:
    """Compose per-axis rotations in the file's channel order (``q = q0 * q1 * q2``)."""
    q = np.broadcast_to(T.IDENTITY_QUAT, angles_deg.shape[:-1] + (4,)).copy()
    for i, axis_index in enumerate(order):
        axis = np.zeros(3, F)
        axis[axis_index] = 1.0
        q = T.quat_mul(q, T.quat_normalize(T.quat_from_axis_angle(axis, np.radians(angles_deg[..., i]).astype(F))))
    return T.quat_normalize(q)


def load_skeleton_from_bvh(file_path, *, remove_namespace: bool = True, remove_root: bool = False) -> Skeleton:
    """The joint hierarchy and reference pose of a BVH file (offsets in cm become meters).

    ``remove_root`` drops a leading ``Root`` joint that only holds the identity, making the
    ``Hips``-rooted skeleton the SOMA engine expects (upstream reads the same skeleton from
    ``soma_base_skel_minimal_no_root.usd``).
    """
    with _open_bvh(file_path) as source:
        skeleton, _ = _read_hierarchy(source, remove_namespace)
    return _without_root(skeleton) if remove_root else skeleton


def _without_root(skeleton: Skeleton) -> Skeleton:
    if skeleton.num_joints < 2 or skeleton.joint_parent(1) != 0 or any(skeleton.joint_parent(j) == -1 for j in range(1, skeleton.num_joints)):
        raise ValueError("remove_root needs a single root joint with a child at index 1.")
    parents = [p - 1 if p > 0 else -1 for p in skeleton.parent_indices[1:]]
    reference = skeleton.reference_local_transforms[1:].copy()
    reference[0] = T.transform_mul(skeleton.reference_local_transforms[0], reference[0])
    return Skeleton(skeleton.joint_names[1:], parents, reference)


def load_bvh(file_path, skeleton: Skeleton | None = None, *, remove_namespace: bool = True) -> AnimationBuffer:
    """Load a BVH clip: positions cm -> m, rotation channels in degrees, in the order listed.

    ``skeleton`` remaps the clip onto another skeleton by joint name (joints missing from the file
    keep its reference transforms); ``None`` keeps the file's skeleton.
    """
    with _open_bvh(file_path) as source:
        file_skeleton, joint_channels = _read_hierarchy(source, remove_namespace)
        frame_count, sample_rate = 0, None
        for line in source:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "Frames:":
                frame_count = int(tokens[1])
            elif tokens[:2] == ["Frame", "Time:"]:
                sample_rate = round(1.0 / float(tokens[2]), 2)
                break
        if sample_rate is None:
            raise ValueError(f"BVH frame time is missing: {file_path}.")
        samples = np.loadtxt(source, dtype=F, ndmin=2)
    channel_count = sum(len(c) for c in joint_channels)
    if samples.shape != (frame_count, channel_count):
        raise ValueError(f"BVH motion has shape {samples.shape}. Expected {frame_count} frames and {channel_count} channels.")
    poses = np.broadcast_to(file_skeleton.reference_local_transforms, (frame_count, file_skeleton.num_joints, 7)).copy()
    column = 0
    for joint, joint_channels_ in enumerate(joint_channels):
        rotation_columns, order = [], []
        for channel in joint_channels_:
            channel = channel.lower()
            if channel.endswith("position"):
                poses[:, joint, "xyz".index(channel[0])] = samples[:, column] * F(0.01)
            elif channel.endswith("rotation"):
                rotation_columns.append(column)
                order.append("xyz".index(channel[0]))
            column += 1
        if rotation_columns:
            poses[:, joint, 3:] = _euler_to_quaternion(samples[:, rotation_columns], order)
    clip = AnimationBuffer(file_skeleton, sample_rate, poses)
    return create_animation_buffer_for_skeleton(clip, skeleton) if skeleton is not None else clip


# --- CSV ---------------------------------------------------------------------------------------


def _quat_from_euler_xyz_extrinsic(angles_rad: np.ndarray) -> np.ndarray:
    """``scipy.spatial.transform.Rotation.from_euler("xyz", ...)`` (extrinsic X, then Y, then Z)."""
    ex, ey, ez = (np.zeros(3, F) for _ in range(3))
    ex[0], ey[1], ez[2] = 1.0, 1.0, 1.0
    qx = T.quat_from_axis_angle(ex, angles_rad[..., 0])
    qy = T.quat_from_axis_angle(ey, angles_rad[..., 1])
    qz = T.quat_from_axis_angle(ez, angles_rad[..., 2])
    return T.quat_normalize(T.quat_mul(qz, T.quat_mul(qy, qx)))  # scipy composes qz * qy * qx


def _euler_xyz_extrinsic_from_quat(q: np.ndarray) -> np.ndarray:
    """Inverse of the above (``as_euler("xyz")``), for writing CSV files."""
    x, y, z, w = (q[..., i].astype(np.float64) for i in range(4))
    rx = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    ry = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    rz = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([rx, ry, rz], -1)


def load_csv(file_path, sample_rate: float = CSV_SAMPLE_RATE, *, device=None) -> DOFAnimationBuffer:
    """Load a robot motion CSV (cm, extrinsic XYZ degrees, joint degrees) as joint coordinates
    (m, XYZW quaternion, radians), keeping the file's joint column order."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
        if header is None or header[: len(_CSV_ROOT_TITLE)] != _CSV_ROOT_TITLE or len(header) <= len(_CSV_ROOT_TITLE):
            raise ValueError(f"{path}: expected the columns Frame and root_translateX to root_rotateZ, then one column per joint angle.")
        names = tuple(header[len(_CSV_ROOT_TITLE):])
        rows = (line for line in source if line.strip())
        first = next(rows, None)
        if first is None:
            raise ValueError(f"{path}: CSV contains no motion frames.")
        data = np.loadtxt(chain((first,), rows), delimiter=",", ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(f"{path}: row width does not match the CSV header.")
    joint_q = np.empty((len(data), ROOT_COORDINATE_COUNT + len(names)), F)
    joint_q[:, :3] = data[:, 1:4] * 0.01
    joint_q[:, 3:ROOT_COORDINATE_COUNT] = _quat_from_euler_xyz_extrinsic(np.radians(data[:, 4:7]))
    joint_q[:, ROOT_COORDINATE_COUNT:] = np.radians(data[:, len(_CSV_ROOT_TITLE):])
    if not np.isfinite(joint_q).all():
        raise ValueError(f"{path}: motion coordinates must be finite.")
    return DOFAnimationBuffer(names, sample_rate, joint_q)


def save_csv(file_path, buffer: DOFAnimationBuffer) -> None:
    """Write the frames in use of ``buffer`` as a robot motion CSV."""
    if buffer.num_frames == 0:
        raise ValueError("buffer must contain at least one frame in use.")
    frames = buffer.poses[: buffer.num_frames]
    euler = np.degrees(_euler_xyz_extrinsic_from_quat(frames[:, 3:ROOT_COORDINATE_COUNT]))
    with open(file_path, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(_CSV_ROOT_TITLE + list(buffer.joint_names))
        for i, pose in enumerate(frames):
            writer.writerow([i, *(pose[:3] * 100.0), *euler[i], *np.degrees(pose[ROOT_COORDINATE_COUNT:])])


# --- spaces ------------------------------------------------------------------------------------


class FacingDirectionType(IntEnum):
    """How converted characters face, following Maya or MuJoCo conventions."""

    MAYA = auto()
    MUJOCO = auto()


class SpaceConverter:
    """Convert transforms from Y-up source data to a Z-up world."""

    def __init__(self, facing_direction: FacingDirectionType) -> None:
        if facing_direction == FacingDirectionType.MUJOCO:
            self.converter = T.quat_from_axis_angle([1.0, 0.0, 0.0], F(np.radians(90.0)))
        elif facing_direction == FacingDirectionType.MAYA:
            q1 = T.quat_from_axis_angle([0.0, 1.0, 0.0], F(np.radians(90.0)))
            q2 = T.quat_from_axis_angle([0.0, 0.0, 1.0], F(np.radians(90.0)))
            self.converter = T.quat_mul(q1, q2)
        else:
            self.converter = T.IDENTITY_QUAT.copy()

    def convert_transform(self, transform) -> np.ndarray:
        return T.transform_mul(T.transform(np.zeros(3, F), self.converter), transform)


def model_to_runtime(rig: ModelRig | str) -> np.ndarray:
    """Model space -> Z-up world space: SOMA (Y up) turns into Z up facing like the G1 (MuJoCo
    convention); the G1 model space is already Z up, so its transform is the identity."""
    if ModelRig(rig) is ModelRig.SOMA:
        return SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(T.IDENTITY)
    return T.IDENTITY.copy()
