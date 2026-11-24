# SPDX-License-Identifier: Apache-2.0
# Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
# Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
"""Convert the SOMA skeletal mesh (binary USD) into app-rs's little-endian mesh format.

One-off build step for app-rs/assets/soma_mesh.bin; the USD file is Apache-2.0 (upstream
"NVIDIA RIG ASSETS"), and the converted file keeps that license. Run:

    uv run --with usd-core python app-rs/scripts/convert_soma_mesh.py \
        [upstream_usd] [output]

Layout (all little-endian, strings u8-length-prefixed UTF-8):
    magic "MBM1" | u32 num_vertices | u32 num_triangles
    f32 positions[3V] | f32 normals[3V] | u32 indices[3T]
    u32 num_joints | str joint_names[J] | i32 parents[J] (-1 = root)
    f32 inverse_bind_matrices[16J] (row-major 4x4)
    f32 rest_local_transforms[16J] (row-major 4x4)
    u16 joint_indices[4V] | f32 joint_weights[4V]
    f32 geom_bind_transform[16] (row-major 4x4)

All positions and matrix translations are converted to meters with the stage's
metersPerUnit (the SOMA USD is authored in centimeters), and the skin's joint
indices are remapped from the mesh binding's own joint list onto the skeleton's
joint list by name, as upstream motionbricks/io/usd.py does when loading.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdSkel


def main() -> None:
    usd_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "sandbox/upstream/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "app-rs/assets/soma_mesh.bin")
    stage = Usd.Stage.Open(str(usd_path))
    skel = mesh = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton) and skel is None:
            skel = UsdSkel.Skeleton(prim)
        if prim.IsA(UsdGeom.Mesh) and mesh is None:
            mesh = UsdGeom.Mesh(prim)
    assert skel and mesh, "expected one skeleton and one skinned mesh"

    joints = list(skel.GetJointsAttr().Get())
    # Parents from the joint paths (each path is /root/.../name).
    parents = []
    for joint in joints:
        parent_path = joint.rpartition("/")[0]
        parents.append(joints.index(parent_path) if parent_path in joints else -1)

    bind_world = np.asarray(skel.GetBindTransformsAttr().Get(), dtype=np.float64)
    assert bind_world.size, "skeleton has no bindTransforms"
    inv_bind = np.linalg.inv(bind_world.reshape(-1, 4, 4))
    rest_local = np.array(skel.GetRestTransformsAttr().Get(), dtype=np.float64)
    geom_bind = np.array(UsdSkel.BindingAPI(mesh.GetPrim()).GetGeomBindTransformAttr().Get(), dtype=np.float64) \
        if UsdSkel.BindingAPI(mesh.GetPrim()).GetGeomBindTransformAttr().HasValue() else np.eye(4)

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.uint32)
    # Triangulate (the faces are quads at most): fan per face.
    tris = []
    at = 0
    for count in counts:
        face = indices[at:at + count]
        tris.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        at += count
    tris = np.asarray(tris, dtype=np.uint32).reshape(-1)
    normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float32) if mesh.GetNormalsAttr().HasValue() else None
    if normals is None or len(normals) != len(points):
        # Recompute per-vertex normals from the triangles.
        normals = np.zeros_like(points)
        p = points[tris.reshape(-1, 3)]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # [T, 3] per-face
        for corner in range(3):
            np.add.at(normals, tris.reshape(-1, 3)[:, corner], n)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30

    binding = UsdSkel.BindingAPI(mesh.GetPrim())
    joint_indices = np.asarray(binding.GetJointIndicesPrimvar().Get(), dtype=np.uint16)
    joint_weights = np.asarray(binding.GetJointWeightsPrimvar().Get(), dtype=np.float32)
    element_size = binding.GetJointIndicesPrimvar().GetElementSize()
    joint_indices = joint_indices.reshape(len(points), element_size)
    joint_weights = joint_weights.reshape(len(points), element_size)

    # Joint indices refer to the mesh binding's own joint list, which can differ from the
    # skeleton's (the SOMA binding omits the unskinned eye joints). Remap them onto the
    # skeleton's joints by name, like upstream's USD loader.
    names = [joint.rpartition("/")[2] for joint in joints]
    bound = binding.GetJointsAttr().Get()
    if bound:
        lut = np.array([names.index(str(j).rpartition("/")[2]) for j in bound], dtype=np.uint16)
        joint_indices = lut[joint_indices]

    # The stage is authored in centimeters; convert positions and matrix translations to
    # meters so the file matches the BVH clips and the G1 rig.
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    points = points * scale
    inv_bind[:, 3, :3] *= scale
    rest_local[:, 3, :3] *= scale
    geom_bind[3, :3] *= scale
    # Normalize to 4 influences, padded with zeros.
    ji = np.zeros((len(points), 4), dtype=np.uint16)
    jw = np.zeros((len(points), 4), dtype=np.float32)
    take = min(4, element_size)
    ji[:, :take] = joint_indices[:, :take]
    jw[:, :take] = joint_weights[:, :take]
    jw_sum = jw.sum(axis=1, keepdims=True)
    np.divide(jw, jw_sum, out=jw, where=jw_sum > 0)

    def put(buf, fmt, *values):
        buf.extend(struct.pack("<" + fmt, *values))

    buf = bytearray(b"MBM1")
    put(buf, "II", len(points), len(tris) // 3)
    buf.extend(points.astype("<f4").tobytes())
    buf.extend(normals.astype("<f4").tobytes())
    buf.extend(tris.astype("<u4").tobytes())
    put(buf, "I", len(joints))
    for joint in joints:
        name = joint.rpartition("/")[2].encode()
        put(buf, "B", len(name))
        buf.extend(name)
    buf.extend(np.asarray(parents, dtype="<i4").tobytes())
    buf.extend(np.ascontiguousarray(inv_bind, dtype="<f8").astype("<f4").tobytes())
    buf.extend(np.ascontiguousarray(rest_local, dtype="<f8").astype("<f4").tobytes())
    buf.extend(ji.astype("<u2").tobytes())
    buf.extend(jw.astype("<f4").tobytes())
    buf.extend(np.ascontiguousarray(geom_bind, dtype="<f8").astype("<f4").tobytes())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf)
    print(f"{out_path}: {len(points)} vertices, {len(tris) // 3} triangles, {len(joints)} joints, {len(buf) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
