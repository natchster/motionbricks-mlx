// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! CPU ray casting against the logged triangle soup.

use glam::Vec3;

use crate::viewer::render::MeshVertex;

/// A world-space ray. `dir` does not need to be normalized, but [`Pick::distance`]
/// is measured in units of `dir`, so a normalized direction gives world units.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin.
    pub origin: Vec3,
    /// Ray direction (normalized when coming from [`crate::viewer::Frame::screen_ray`]).
    pub dir: Vec3,
}

/// Result of a ray-triangle pick.
#[derive(Clone, Copy, Debug)]
pub struct Pick {
    /// World-space hit point.
    pub point: Vec3,
    /// Geometric (face) normal at the hit; oriented against the ray.
    pub normal: Vec3,
    /// Distance from the ray origin in units of `ray.dir`.
    pub distance: f32,
    /// Index of the logged mesh instance that was hit (in log order this frame).
    pub mesh_index: usize,
}

/// Möller–Trumbore intersection. Returns the ray parameter `t` and barycentric (u, v).
pub(crate) fn ray_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-8;
    let e1 = b - a;
    let e2 = c - a;
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None; // Ray parallel to the triangle (either side).
    }
    let inv = 1.0 / det;
    let s = ray.origin - a;
    let u = s.dot(p) * inv;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > EPS).then_some((t, u, v))
}

/// Nearest hit of `ray` against a flat world-space triangle soup (3 vertices per
/// triangle), as produced by the viewer's mesh logging. `mesh_index` is filled in by
/// the caller; it is left as 0 here.
pub(crate) fn ray_triangles(ray: &Ray, soup: &[MeshVertex]) -> Option<Pick> {
    let mut best: Option<Pick> = None;
    for tri in soup.chunks_exact(3) {
        let (a, b, c) = (Vec3::from(tri[0].position), Vec3::from(tri[1].position), Vec3::from(tri[2].position));
        if let Some((t, _u, _v)) = ray_triangle(ray, a, b, c) {
            if best.as_ref().is_none_or(|p| t < p.distance) {
                let mut normal = (b - a).cross(c - a).normalize_or(Vec3::Y);
                if normal.dot(ray.dir) > 0.0 {
                    normal = -normal; // Face the ray, regardless of winding.
                }
                best = Some(Pick {
                    point: ray.origin + ray.dir * t,
                    normal,
                    distance: t,
                    mesh_index: 0,
                });
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::mesh::Mesh;
    use crate::viewer::Color;
    use glam::Mat4;

    /// Expand a mesh into a world-space soup, mirroring what `Frame::log_mesh` does.
    fn soup_of(mesh: &Mesh, transform: Mat4) -> Vec<MeshVertex> {
        let color: Color = [1.0; 4];
        let mut out = Vec::new();
        for &i in &mesh.indices {
            let p = transform.transform_point3(mesh.positions[i as usize]);
            out.push(MeshVertex { position: p.to_array(), normal: [0.0; 3], color: crate::viewer::render::pack_color(color) });
        }
        out
    }

    #[test]
    fn pick_cube_face() {
        // Unit cube at the origin, ray flying down -Z from (0, 0, 5).
        let cube = Mesh::cube(Vec3::ZERO, Vec3::ONE);
        let soup = soup_of(&cube, Mat4::IDENTITY);
        let ray = Ray { origin: Vec3::new(0.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        let pick = ray_triangles(&ray, &soup).expect("ray through the cube face must hit");
        assert!((pick.distance - 4.0).abs() < 1e-5, "distance: {}", pick.distance);
        assert!(pick.point.abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-5));
        assert!(pick.normal.abs_diff_eq(Vec3::Z, 1e-5), "normal: {:?}", pick.normal);

        // A ray that misses reports nothing.
        let miss = Ray { origin: Vec3::new(10.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        assert!(ray_triangles(&miss, &soup).is_none());
    }

    #[test]
    fn pick_respects_transform() {
        // Cube shifted to (0, 0, -10): the same ray hits 14 units out.
        let cube = Mesh::cube(Vec3::ZERO, Vec3::ONE);
        let soup = soup_of(&cube, Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)));
        let ray = Ray { origin: Vec3::new(0.0, 0.0, 5.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        let pick = ray_triangles(&ray, &soup).unwrap();
        assert!((pick.distance - 14.0).abs() < 1e-4, "distance: {}", pick.distance);
    }
}
