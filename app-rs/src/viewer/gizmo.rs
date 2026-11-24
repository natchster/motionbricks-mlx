// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Minimal ImGuizmo-style translate gizmo, drawn with the viewer's own log calls.
//!
//! Three axis arrows (X red, Y green, Z blue) plus three small plane quads (YZ, XZ,
//! XY). Hover and dragging are resolved with the camera ray from [`Frame::screen_ray`]:
//! dragging an axis projects the mouse ray onto the axis line, dragging a plane quad
//! intersects it with that plane, and the delta moves the transform's translation.
//! Rotation rings are intentionally not part of v1.

use glam::{Mat4, Vec3};

use crate::viewer::mesh;
use crate::viewer::{Color, Frame, MeshInstance};

const AXIS_COLORS: [Color; 3] = [
    [0.85, 0.25, 0.25, 1.0], // X: red
    [0.35, 0.8, 0.3, 1.0],   // Y: green
    [0.3, 0.45, 0.9, 1.0],   // Z: blue
];
const HIGHLIGHT: Color = [1.0, 0.9, 0.3, 1.0];
/// Axis arrow length as a fraction of the camera distance (constant screen size).
const SCREEN_SCALE: f32 = 0.18;
/// Plane quad offset / size, as a fraction of the axis length.
const PLANE_OFFSET: f32 = 0.3;
const PLANE_SIZE: f32 = 0.2;
/// Hover tolerance for axes, as a fraction of the axis length.
const HOVER_TOL: f32 = 0.08;

/// Which handle of the gizmo is hovered or dragged.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Handle {
    /// Axis index 0/1/2 = X/Y/Z.
    Axis(usize),
    /// Plane whose normal is axis 0/1/2 (YZ / XZ / XY).
    Plane(usize),
}

/// Persistent (cross-frame) gizmo drag state. One active drag at a time.
#[derive(Clone, Debug, Default)]
pub(crate) struct GizmoState {
    active_id: String,
    handle: Option<Handle>,
    /// World-space grab point on the axis line / plane at drag start.
    grab: Vec3,
    /// Transform translation at drag start.
    start_translation: Vec3,
    /// Last frame index in which the active id was driven.
    last_frame: u64,
}

impl GizmoState {
    /// Forget drags whose owner stopped calling `gizmo_translate`.
    pub fn prune(&mut self, frame_index: u64) {
        if self.handle.is_some() && self.last_frame + 1 < frame_index {
            self.handle = None;
            self.active_id.clear();
        }
    }
}

const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// Closest-point parameters between a ray (ro, rd) and a line (lo, ld), both
/// directions normalized. Returns `(s_ray, t_line)`.
fn closest_ray_line(ro: Vec3, rd: Vec3, lo: Vec3, ld: Vec3) -> (f32, f32) {
    let w0 = ro - lo;
    let b = rd.dot(ld);
    let d = rd.dot(w0);
    let e = ld.dot(w0);
    let denom = 1.0 - b * b;
    if denom < 1e-8 {
        // Nearly parallel: project the ray origin onto the line.
        return (0.0, e);
    }
    ((b * e - d) / denom, (e - b * d) / denom)
}

/// Intersect a ray with a plane through `origin` with `normal`. Returns the hit point.
fn ray_plane(origin: Vec3, normal: Vec3, ro: Vec3, rd: Vec3) -> Option<Vec3> {
    let denom = rd.dot(normal);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = (origin - ro).dot(normal) / denom;
    (t > 0.0).then(|| ro + rd * t)
}

/// The gizmo body. See [`Frame::gizmo_translate`].
pub(crate) fn gizmo_translate(frame: &mut Frame, id: &str, transform: &mut Mat4) -> bool {
    let pos = transform.w_axis.truncate();
    let cam_eye = frame.camera.eye();
    let scale = (cam_eye - pos).length().max(0.01) * SCREEN_SCALE;
    let tol = HOVER_TOL * scale;

    let ray = frame.input.cursor.map(|c| frame.screen_ray(c));
    let st = frame.gizmo_state.clone();
    let left_held = frame.input.mouse_held(winit::event::MouseButton::Left);
    let left_pressed = frame.input.mouse_pressed(winit::event::MouseButton::Left);

    let mut dragging: Option<Handle> = None;
    let mut hovered: Option<Handle> = None;

    // 1. Continue an active drag owned by this id.
    if st.handle.is_some() && st.active_id == id {
        if left_held {
            let handle = st.handle.unwrap();
            dragging = Some(handle);
            if let Some(ray) = ray {
                let delta = match handle {
                    Handle::Axis(i) => {
                        let (_, t) = closest_ray_line(ray.origin, ray.dir, pos, AXES[i]);
                        let p_now = pos + AXES[i] * t;
                        AXES[i] * (p_now - st.grab).dot(AXES[i])
                    }
                    Handle::Plane(i) => ray_plane(pos, AXES[i], ray.origin, ray.dir)
                        .map(|p| p - st.grab)
                        .unwrap_or(Vec3::ZERO),
                };
                let t = st.start_translation + delta;
                *transform = Mat4::from_cols(
                    transform.x_axis,
                    transform.y_axis,
                    transform.z_axis,
                    t.extend(transform.w_axis.w),
                );
            }
        } else {
            // Released: drop the drag immediately so a re-click starts fresh.
            frame.gizmo_state.handle = None;
        }
    }

    // 2. Otherwise resolve hover, and start a drag on click.
    if dragging.is_none() {
        if let Some(ray) = ray {
            let mut best: Option<(Handle, f32, Vec3)> = None; // (handle, score, grab point)
            for (i, axis) in AXES.iter().enumerate() {
                let (s, t) = closest_ray_line(ray.origin, ray.dir, pos, *axis);
                let d = (ray.origin + ray.dir * s - (pos + *axis * t)).length();
                if (0.0..=scale).contains(&t) && d < tol && s > 0.0 {
                    let better = best.as_ref().is_none_or(|(_, bd, _)| d < *bd);
                    if better {
                        best = Some((Handle::Axis(i), d, pos + *axis * t));
                    }
                }
            }
            // Plane quads: normal axis i, spanned by the other two axes.
            for i in 0..3 {
                let (a, b) = (AXES[(i + 1) % 3], AXES[(i + 2) % 3]);
                if let Some(hit) = ray_plane(pos, AXES[i], ray.origin, ray.dir) {
                    let la = (hit - pos).dot(a);
                    let lb = (hit - pos).dot(b);
                    let (o, q) = (PLANE_OFFSET * scale, PLANE_SIZE * scale);
                    if (o..=o + q).contains(&la) && (o..=o + q).contains(&lb) {
                        // Planes win ties against axes at zero distance.
                        let better = best.as_ref().is_none_or(|(_, bd, _)| 0.0 < *bd);
                        if better {
                            best = Some((Handle::Plane(i), 0.0, hit));
                        }
                    }
                }
            }
            if let Some((handle, _, grab)) = best {
                hovered = Some(handle);
                if left_pressed {
                    frame.gizmo_state = GizmoState {
                        active_id: id.to_owned(),
                        handle: Some(handle),
                        grab,
                        start_translation: pos,
                        last_frame: frame.frame_index,
                    };
                    dragging = Some(handle);
                    hovered = None;
                }
            }
        }
    }

    if let Some(handle) = dragging {
        frame.gizmo_state.active_id = id.to_owned();
        frame.gizmo_state.handle = Some(handle);
        frame.gizmo_state.last_frame = frame.frame_index;
    }
    if dragging.is_some() || hovered.is_some() {
        frame.gizmo_captured = true;
    }

    // 3. Draw: axis arrows + plane quads, highlighting the active handle.
    let active = dragging.or(hovered);
    for (i, axis) in AXES.iter().enumerate() {
        let hot = active == Some(Handle::Axis(i));
        frame.log_arrow(pos, pos + *axis * scale, if hot { HIGHLIGHT } else { AXIS_COLORS[i] });
    }
    for i in 0..3 {
        let (a, b) = (AXES[(i + 1) % 3], AXES[(i + 2) % 3]);
        let (o, q) = (PLANE_OFFSET * scale, PLANE_SIZE * scale);
        let hot = active == Some(Handle::Plane(i));
        let mut c = if hot { HIGHLIGHT } else { AXIS_COLORS[i] };
        c[3] = if hot { 0.9 } else { 0.45 };
        let quad = mesh::plane_quad(pos + (a + b) * o, a * q, b * q, c);
        let base = frame.mesh_verts.len();
        frame.mesh_verts.extend(quad);
        frame
            .mesh_instances
            .push(MeshInstance { start: base, count: frame.mesh_verts.len() - base });
    }

    dragging.is_some()
}
