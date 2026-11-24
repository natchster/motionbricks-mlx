// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Orbit camera matching the Newton ViewerGL defaults.

use glam::{Mat4, Vec2, Vec3};

use crate::viewer::picking::Ray;

/// Orbit camera: the eye circles `target` at `distance`, parameterized by `yaw`
/// (radians around +Y, 0 looks towards -Z... see below) and `pitch` (radians above the
/// horizon, clamped to just under ±90°).
///
/// Eye position: `target + distance * (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))`.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    /// Look-at point.
    pub target: Vec3,
    /// Orbit angle around the vertical axis, radians.
    pub yaw: f32,
    /// Elevation angle, radians (clamped to roughly ±89°).
    pub pitch: f32,
    /// Distance from the eye to `target`.
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 1.0, 0.0),
            yaw: 0.6,
            pitch: 0.35,
            distance: 4.0,
        }
    }
}

impl OrbitCamera {
    /// Vertical field of view, radians.
    pub const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
    /// Near clip plane.
    pub const NEAR: f32 = 0.01;
    /// Far clip plane.
    pub const FAR: f32 = 2000.0;

    /// World-space eye position.
    pub fn eye(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        self.target + self.distance * Vec3::new(cp * sy, sp, cp * cy)
    }

    /// World-to-view matrix.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// View-projection matrix (wgpu clip space, depth 0..1) for a viewport aspect ratio.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(Self::FOV_Y, aspect.max(1e-6), Self::NEAR, Self::FAR) * self.view()
    }

    /// Smoothly move the look-at point towards `target` (character-follow style).
    /// `stiffness` is per second: higher snappier.
    pub fn follow(&mut self, target: Vec3, stiffness: f32, dt: f32) {
        let k = 1.0 - (-stiffness.max(0.0) * dt).exp();
        self.target = self.target.lerp(target, k);
    }

    /// Orbit by yaw/pitch deltas in radians.
    pub(crate) fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        const LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.yaw = (self.yaw + dyaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + dpitch).clamp(-LIMIT, LIMIT);
    }

    /// Pan in the view plane by a pixel delta (scaled with distance).
    pub(crate) fn pan(&mut self, delta: Vec2) {
        let scale = self.distance * 0.0018;
        let eye = self.eye();
        let fwd = (self.target - eye).normalize_or(Vec3::Z);
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        self.target += (-delta.x * right + delta.y * up) * scale;
    }

    /// Dolly by a scroll delta in lines (positive zooms in).
    pub(crate) fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.12).exp()).clamp(0.05, 500.0);
    }
}

/// Ray from the camera through a pixel (logical pixels, top-left origin) of a viewport
/// of `viewport` logical pixels.
pub(crate) fn screen_ray(camera: &OrbitCamera, viewport: Vec2, pixel: Vec2) -> Ray {
    let ndc = Vec3::new(
        2.0 * pixel.x / viewport.x.max(1.0) - 1.0,
        1.0 - 2.0 * pixel.y / viewport.y.max(1.0),
        0.0,
    );
    let inv = camera.view_proj(viewport.x / viewport.y.max(1.0)).inverse();
    let near = inv.project_point3(ndc);
    let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    Ray { origin: near, dir: (far - near).normalize_or(Vec3::Z) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_proj_basis() {
        // Camera looking straight down -Z at a target on the origin-ish.
        let cam = OrbitCamera { target: Vec3::ZERO, yaw: 0.0, pitch: 0.0, distance: 5.0 };
        // Eye sits at +Z from the target.
        assert!(cam.eye().abs_diff_eq(Vec3::new(0.0, 0.0, 5.0), 1e-6));
        let vp = cam.view_proj(16.0 / 9.0);
        // The target projects to the NDC center.
        let c = vp.project_point3(Vec3::ZERO);
        assert!(c.truncate().abs_diff_eq(Vec2::ZERO, 1e-5), "center: {c:?}");
        // +X world projects to +X NDC, +Y world to +Y NDC.
        assert!(vp.project_point3(Vec3::X).x > 0.0);
        assert!(vp.project_point3(Vec3::Y).y > 0.0);
        // The target is between near and far in NDC depth.
        assert!(c.z > 0.0 && c.z < 1.0);
        // Pitch=90° looks straight down: eye above the target.
        let top = OrbitCamera { target: Vec3::ZERO, yaw: 0.0, pitch: std::f32::consts::FRAC_PI_2, distance: 2.0 };
        assert!(top.eye().abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-5));
    }

    #[test]
    fn screen_ray_through_center_hits_target() {
        let cam = OrbitCamera { target: Vec3::new(1.0, 2.0, 3.0), yaw: 0.7, pitch: -0.3, distance: 6.5 };
        let viewport = Vec2::new(1280.0, 720.0);
        let ray = screen_ray(&cam, viewport, viewport / 2.0);
        // The ray origin is on the near plane, just in front of the eye.
        assert!(ray.origin.distance(cam.eye()) < 0.05, "origin: {:?}", ray.origin);
        // The target lies exactly on the ray (zero perpendicular distance).
        let to_target = cam.target - ray.origin;
        let perp = (to_target - ray.dir * to_target.dot(ray.dir)).length();
        assert!(perp < 1e-3, "target off the ray by {perp}");
        // ... at roughly `distance` units from the origin.
        assert!((to_target.dot(ray.dir) - cam.distance).abs() < 0.05);
    }
}
