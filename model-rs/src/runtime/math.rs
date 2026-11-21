// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (motionbricks/utils/*).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Rigid transforms and quaternions with Warp's conventions: a quaternion is `[x, y, z, w]`, a
//! transform is a position and a rotation (`wp.transform`). The formulas follow Warp's native code
//! (`quat_rotate`, `quat_slerp`, `quat_from_matrix`, ...), so results match upstream's kernels to
//! float32 rounding, including which of `q` and `-q` comes out.

use std::ops::{Add, Mul, Neg, Sub};

pub type Quat = [f32; 4];
pub const QUAT_IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3(pub [f32; 3]);

impl Vec3 {
    pub const ZERO: Vec3 = Vec3([0.0; 3]);
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }
    pub fn x(self) -> f32 {
        self.0[0]
    }
    pub fn y(self) -> f32 {
        self.0[1]
    }
    pub fn z(self) -> f32 {
        self.0[2]
    }
    pub fn dot(self, o: Vec3) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        let (a, b) = (self.0, o.0);
        Vec3([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    /// Zero length gives zero (Warp's `kEps` is 0).
    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 0.0 { Vec3([self.0[0] / l, self.0[1] / l, self.0[2] / l]) } else { Vec3::ZERO }
    }
    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self + (o - self) * t
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

/// Zero length gives the identity.
pub fn quat_normalize(q: Quat) -> Quat {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 0.0 { [q[0] / l, q[1] / l, q[2] / l, q[3] / l] } else { QUAT_IDENTITY }
}

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + b[3] * a[0] + (a[1] * b[2] - a[2] * b[1]),
        a[3] * b[1] + b[3] * a[1] + (a[2] * b[0] - a[0] * b[2]),
        a[3] * b[2] + b[3] * a[2] + (a[0] * b[1] - a[1] * b[0]),
        a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]),
    ]
}

/// Conjugate, like `wp.quat_inverse` (not renormalized).
pub fn quat_inverse(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn quat_rotate(q: Quat, x: Vec3) -> Vec3 {
    let c = 2.0 * q[3] * q[3] - 1.0;
    let d = 2.0 * (q[0] * x.0[0] + q[1] * x.0[1] + q[2] * x.0[2]);
    let w2 = q[3] * 2.0;
    Vec3([
        x.0[0] * c + q[0] * d + (q[1] * x.0[2] - q[2] * x.0[1]) * w2,
        x.0[1] * c + q[1] * d + (q[2] * x.0[0] - q[0] * x.0[2]) * w2,
        x.0[2] * c + q[2] * d + (q[0] * x.0[1] - q[1] * x.0[0]) * w2,
    ])
}

pub fn quat_rotate_inv(q: Quat, x: Vec3) -> Vec3 {
    let c = 2.0 * q[3] * q[3] - 1.0;
    let d = 2.0 * (q[0] * x.0[0] + q[1] * x.0[1] + q[2] * x.0[2]);
    let w2 = q[3] * 2.0;
    Vec3([
        x.0[0] * c + q[0] * d - (q[1] * x.0[2] - q[2] * x.0[1]) * w2,
        x.0[1] * c + q[1] * d - (q[2] * x.0[0] - q[0] * x.0[2]) * w2,
        x.0[2] * c + q[2] * d - (q[0] * x.0[1] - q[1] * x.0[0]) * w2,
    ])
}

pub fn quat_from_axis_angle(axis: Vec3, angle: f32) -> Quat {
    let half = angle * 0.5;
    let s = half.sin();
    [axis.0[0] * s, axis.0[1] * s, axis.0[2] * s, half.cos()]
}

pub fn quat_to_axis_angle(q: Quat) -> (Vec3, f32) {
    let v = Vec3([q[0], q[1], q[2]]);
    let axis = if q[3] < 0.0 { -v.normalize() } else { v.normalize() };
    (axis, 2.0 * v.length().atan2(q[3].abs()))
}

/// `wp.quat_slerp`: `q0 * axis_angle(inverse(q0) * q1) ^ t`, along the shorter arc.
pub fn quat_slerp(q0: Quat, q1: Quat, t: f32) -> Quat {
    let (axis, angle) = quat_to_axis_angle(quat_mul(quat_inverse(q0), q1));
    quat_mul(q0, quat_from_axis_angle(axis, t * angle))
}

/// `wp.quat_from_matrix` for a rotation matrix given by its columns (same branches).
pub fn quat_from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Quat {
    // m[row][col]
    let m = [[c0.0[0], c1.0[0], c2.0[0]], [c0.0[1], c1.0[1], c2.0[1]], [c0.0[2], c1.0[2], c2.0[2]]];
    let tr = m[0][0] + m[1][1] + m[2][2];
    let (x, y, z, w);
    if tr >= 0.0 {
        let h = (tr + 1.0).sqrt();
        w = 0.5 * h;
        let h = 0.5 / h;
        x = (m[2][1] - m[1][2]) * h;
        y = (m[0][2] - m[2][0]) * h;
        z = (m[1][0] - m[0][1]) * h;
    } else {
        let mut k = 0;
        if m[1][1] > m[0][0] {
            k = 1;
        }
        if m[2][2] > m[k][k] {
            k = 2;
        }
        if k == 0 {
            let h = ((m[0][0] - (m[1][1] + m[2][2])) + 1.0).sqrt();
            x = 0.5 * h;
            let h = 0.5 / h;
            y = (m[0][1] + m[1][0]) * h;
            z = (m[2][0] + m[0][2]) * h;
            w = (m[2][1] - m[1][2]) * h;
        } else if k == 1 {
            let h = ((m[1][1] - (m[2][2] + m[0][0])) + 1.0).sqrt();
            y = 0.5 * h;
            let h = 0.5 / h;
            z = (m[1][2] + m[2][1]) * h;
            x = (m[0][1] + m[1][0]) * h;
            w = (m[0][2] - m[2][0]) * h;
        } else {
            let h = ((m[2][2] - (m[0][0] + m[1][1])) + 1.0).sqrt();
            z = 0.5 * h;
            let h = 0.5 / h;
            x = (m[2][0] + m[0][2]) * h;
            y = (m[1][2] + m[2][1]) * h;
            w = (m[1][0] - m[0][1]) * h;
        }
    }
    quat_normalize([x, y, z, w])
}

pub fn quat_between_vectors(a: Vec3, b: Vec3) -> Quat {
    let (a, b) = (a.normalize(), b.normalize());
    let c = a.cross(b);
    quat_normalize([c.0[0], c.0[1], c.0[2], 1.0 + a.dot(b)])
}

/// A rigid transform: position and rotation (`wp.transform`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub p: Vec3,
    pub q: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Transform::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform { p: Vec3::ZERO, q: QUAT_IDENTITY };

    pub const fn new(p: Vec3, q: Quat) -> Self {
        Transform { p, q }
    }
    /// From `[px, py, pz, qx, qy, qz, qw]`.
    pub fn from_slice(v: &[f32]) -> Self {
        Transform { p: Vec3([v[0], v[1], v[2]]), q: [v[3], v[4], v[5], v[6]] }
    }
    pub fn to_array(self) -> [f32; 7] {
        [self.p.0[0], self.p.0[1], self.p.0[2], self.q[0], self.q[1], self.q[2], self.q[3]]
    }
    pub fn mul(self, b: Transform) -> Transform {
        Transform { p: self.p + quat_rotate(self.q, b.p), q: quat_mul(self.q, b.q) }
    }
    pub fn inverse(self) -> Transform {
        let q = quat_inverse(self.q);
        Transform { p: -quat_rotate(q, self.p), q }
    }
    pub fn point(self, x: Vec3) -> Vec3 {
        self.p + quat_rotate(self.q, x)
    }
    pub fn vector(self, x: Vec3) -> Vec3 {
        quat_rotate(self.q, x)
    }
}

/// Positions lerp, rotations slerp (`pose_utils.blend_transform`).
pub fn blend_transform(a: Transform, b: Transform, t: f32) -> Transform {
    Transform { p: a.p.lerp(b.p, t), q: quat_slerp(a.q, b.q, t) }
}

pub fn compute_delta_transform(src: Transform, dst: Transform) -> Transform {
    src.inverse().mul(dst)
}

pub fn apply_delta_transform(t: Transform, delta: Transform, weight: f32) -> Transform {
    t.mul(blend_transform(Transform::IDENTITY, delta, weight))
}

/// World-space position offset and post-multiplied rotation from a generated root to a guide root.
pub fn compute_root_endpoint_delta(generated: Transform, guide: Transform) -> Transform {
    let r = quat_mul(quat_inverse(quat_normalize(generated.q)), quat_normalize(guide.q));
    Transform { p: guide.p - generated.p, q: quat_normalize(r) }
}

/// The start correction fades out as the end correction fades in; positions blend linearly.
pub fn apply_root_endpoint_correction(root: Transform, start: Transform, end: Transform, end_weight: f32) -> Transform {
    let start_weight = 1.0 - end_weight;
    let r = quat_mul(quat_mul(root.q, quat_slerp(QUAT_IDENTITY, start.q, start_weight)), quat_slerp(QUAT_IDENTITY, end.q, end_weight));
    Transform { p: root.p + start.p * start_weight + end.p * end_weight, q: quat_normalize(r) }
}

pub fn quadratic_fade_weight(value: f32) -> f32 {
    let t = value.clamp(0.0, 1.0);
    let rem = 1.0 - t;
    if t < 0.5 { 1.0 - 2.0 * t * t } else { 2.0 * rem * rem }
}

/// Compose local joint transforms through the hierarchy (each parent before its children).
pub fn compute_world_poses(local: &[Transform], parents: &[i32], root: Transform) -> Vec<Transform> {
    let mut world = Vec::with_capacity(local.len());
    world.push(root.mul(local[0]));
    for j in 1..local.len() {
        let parent = world[parents[j] as usize];
        world.push(parent.mul(local[j]));
    }
    world
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_slerp() {
        let q = quat_normalize([0.3, -0.2, 0.5, 0.7]);
        let t = Transform::new(Vec3::new(1.0, 2.0, -3.0), q);
        let i = t.mul(t.inverse());
        assert!(i.p.length() < 1e-5 && (i.q[3].abs() - 1.0).abs() < 1e-5);
        let half = quat_slerp(QUAT_IDENTITY, q, 0.5);
        let full = quat_mul(half, half);
        assert!((full.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>().abs() - 1.0).abs() < 1e-5);
    }
}
