// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Triangle mesh container and small geometry builders (cube, wire box, arrow head).

use glam::Vec3;

use crate::viewer::render::MeshVertex;

/// A plain indexed triangle mesh with per-vertex normals.
///
/// If `indices` is empty the positions are treated as a flat triangle soup.
/// If `normals` is shorter than `positions`, missing normals default to +Y.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Per-vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices into `positions` (3 per triangle).
    pub indices: Vec<u32>,
}

impl Mesh {
    /// An axis-aligned cube centered at `center` (24 vertices, 36 indices, flat normals).
    pub fn cube(center: Vec3, half_extents: Vec3) -> Mesh {
        let e = half_extents;
        let mut positions = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        // (axis index of the face normal, sign)
        let faces = [
            (0usize, 1.0f32),
            (0, -1.0),
            (1, 1.0),
            (1, -1.0),
            (2, 1.0),
            (2, -1.0),
        ];
        for (axis, sign) in faces {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            let base = positions.len() as u32;
            for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let mut p = center;
                p[axis] += sign * e[axis];
                p[u] += su * e[u];
                p[v] += sv * e[v];
                positions.push(p);
                normals.push(n);
            }
            // Winding doesn't matter (meshes are drawn double-sided) but keep it consistent.
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Mesh { positions, normals, indices }
    }
}

/// The 12 edges of an axis-aligned box as line segments.
pub fn wire_box_segments(center: Vec3, half_extents: Vec3) -> [[Vec3; 2]; 12] {
    let c = center;
    let e = half_extents;
    let corner = |sx: f32, sy: f32, sz: f32| c + Vec3::new(sx * e.x, sy * e.y, sz * e.z);
    [
        // Bottom face.
        [corner(-1.0, -1.0, -1.0), corner(1.0, -1.0, -1.0)],
        [corner(1.0, -1.0, -1.0), corner(1.0, -1.0, 1.0)],
        [corner(1.0, -1.0, 1.0), corner(-1.0, -1.0, 1.0)],
        [corner(-1.0, -1.0, 1.0), corner(-1.0, -1.0, -1.0)],
        // Top face.
        [corner(-1.0, 1.0, -1.0), corner(1.0, 1.0, -1.0)],
        [corner(1.0, 1.0, -1.0), corner(1.0, 1.0, 1.0)],
        [corner(1.0, 1.0, 1.0), corner(-1.0, 1.0, 1.0)],
        [corner(-1.0, 1.0, 1.0), corner(-1.0, 1.0, -1.0)],
        // Verticals.
        [corner(-1.0, -1.0, -1.0), corner(-1.0, 1.0, -1.0)],
        [corner(1.0, -1.0, -1.0), corner(1.0, 1.0, -1.0)],
        [corner(1.0, -1.0, 1.0), corner(1.0, 1.0, 1.0)],
        [corner(-1.0, -1.0, 1.0), corner(-1.0, 1.0, 1.0)],
    ]
}

/// Cone geometry for an arrow head: returns the point where the shaft should stop
/// (`head_base`) and the cone as a world-space triangle soup (8 side + 8 base
/// triangles = 48 vertices) with outward normals. Color is left to the caller; the
/// returned vertices carry a placeholder color.
pub(crate) fn arrow_head(start: Vec3, end: Vec3) -> (Vec3, Vec<MeshVertex>) {
    const SEGMENTS: usize = 8;
    let axis = end - start;
    let len = axis.length();
    if len < 1e-6 {
        return (end, Vec::new());
    }
    let dir = axis / len;
    let head_len = (0.18 * len).clamp(0.02, 0.3);
    let radius = 0.3 * head_len;
    let head_base = end - dir * head_len;
    // Orthonormal basis around `dir`.
    let side = if dir.y.abs() < 0.99 { dir.cross(Vec3::Y) } else { dir.cross(Vec3::X) }.normalize();
    let up = side.cross(dir);

    let ring: Vec<Vec3> = (0..SEGMENTS)
        .map(|i| {
            let a = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            head_base + (a.cos() * side + a.sin() * up) * radius
        })
        .collect();
    let push = |verts: &mut Vec<MeshVertex>, p: Vec3, n: Vec3| {
        verts.push(MeshVertex { position: p.to_array(), normal: n.to_array(), color: [0; 4] });
    };
    let mut out = Vec::with_capacity(SEGMENTS * 2 * 3);
    for i in 0..SEGMENTS {
        let a = ring[i];
        let b = ring[(i + 1) % SEGMENTS];
        // Side triangle (apex at `end`); normal is the face normal, good enough at this size.
        let n = (b - a).cross(end - a).normalize_or(dir);
        push(&mut out, a, n);
        push(&mut out, b, n);
        push(&mut out, end, n);
        // Base fan triangle facing away from the arrow.
        push(&mut out, b, -dir);
        push(&mut out, a, -dir);
        push(&mut out, head_base, -dir);
    }
    (head_base, out)
}

/// A screen-space-aligned quad (two triangles) spanning axes `u`/`v` of a basis,
/// used by the translate gizmo's plane handles. Returns 6 vertices.
pub(crate) fn plane_quad(origin: Vec3, u: Vec3, v: Vec3, color: [f32; 4]) -> Vec<MeshVertex> {
    let n = u.cross(v).normalize_or(Vec3::Y);
    let color = super::render::pack_color(color);
    let mut out = Vec::with_capacity(6);
    let corners = [origin, origin + u, origin + u + v, origin + v];
    for p in [corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]] {
        out.push(MeshVertex { position: p.to_array(), normal: n.to_array(), color });
    }
    out
}
