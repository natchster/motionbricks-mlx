// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The probe box ahead of a character and the oriented-box overlap test, ported from
//! upstream `app/runtime/common/interaction.py` and the box parts of
//! `motionbricks/runtime/smart_primitive.py`. Where upstream launches Warp kernels,
//! this port runs plain loops with the same math.

use glam::{Mat3, Vec3};
use motionbricks_rs::runtime::math::{self as rtm, Transform};

use super::motion::to_rt;

/// Oriented box for interaction boxes and the probe box (upstream `InteractionBox`).
#[derive(Clone, Copy, Debug)]
pub struct InteractionBox {
    /// Transform of the box center.
    pub transform: Transform,
    /// Half size along each of the box's axes, in meters.
    pub half_extents: Vec3,
}

/// How far ahead a character looks for Smart Primitive interaction boxes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionProbe {
    /// Width of the probe box in meters.
    pub box_width: f32,
    /// Smallest depth of the box in meters.
    pub box_min_depth: f32,
    /// Height of the box in meters while standing.
    pub box_height: f32,
    /// Distance from the character to the near end of the box, in meters.
    pub near_distance: f32,
    /// Seconds of travel at the desired speed that the box reaches.
    pub lookahead_time: f32,
    /// Speed in m/s above which the box reaches further.
    pub extra_lookahead_min_speed: f32,
    /// Extra seconds of travel for the speed above `extra_lookahead_min_speed`.
    pub extra_lookahead_time: f32,
}

impl InteractionProbe {
    /// Return the near and far ends of the probe on the ground.
    pub fn segment(&self, ground_position: Vec3, direction: Vec3, speed: f32) -> (Vec3, Vec3) {
        let mut reach = speed * self.lookahead_time;
        reach += (speed - self.extra_lookahead_min_speed).max(0.0) * self.extra_lookahead_time;
        let far = (self.near_distance + self.box_min_depth).max(reach);
        (ground_position + direction * self.near_distance, ground_position + direction * far)
    }

    /// The box that covers the segment, standing on the ground (upstream's
    /// `write_box` kernel). The box's local X is its width, Y its depth, Z its height.
    pub fn box_for_segment(&self, segment: (Vec3, Vec3), height_scale: f32) -> InteractionBox {
        let height = self.box_height * height_scale;
        let direction = segment.1 - segment.0;
        let depth = direction.length();
        let center = Vec3::new(0.5 * (segment.0.x + segment.1.x), 0.5 * (segment.0.y + segment.1.y), 0.5 * height);
        let yaw = direction.y.atan2(direction.x) - 0.5 * std::f32::consts::PI;
        InteractionBox {
            transform: Transform::new(to_rt(center), rtm::quat_from_axis_angle(rtm::Vec3::new(0.0, 0.0, 1.0), yaw)),
            half_extents: 0.5 * Vec3::new(self.box_width, depth, height),
        }
    }
}

/// Test two oriented boxes on every separating axis (upstream `_boxes_overlap`).
/// Touching boxes overlap. A small bias keeps nearly parallel edges from reporting a
/// false gap.
pub fn boxes_overlap(a: &InteractionBox, b: &InteractionBox) -> bool {
    let quat = |q: rtm::Quat| glam::Quat::from_xyzw(q[0], q[1], q[2], q[3]);
    let rotation_a = Mat3::from_quat(quat(a.transform.q));
    // Express b's axes and center in a's local space.
    let r = rotation_a.transpose() * Mat3::from_quat(quat(b.transform.q));
    let t = rotation_a.transpose() * (super::motion::to_glam(b.transform.p) - super::motion::to_glam(a.transform.p));
    let (ea, eb) = (a.half_extents, b.half_extents);
    let at = |i: usize, j: usize| r.row(i)[j].abs() + 1.0e-6;
    let rt = |i: usize, j: usize| r.row(i)[j];

    for i in 0..3 {
        if t[i].abs() > ea[i] + eb[0] * at(i, 0) + eb[1] * at(i, 1) + eb[2] * at(i, 2) {
            return false;
        }
    }
    for j in 0..3 {
        let distance = (t[0] * rt(0, j) + t[1] * rt(1, j) + t[2] * rt(2, j)).abs();
        if distance > ea[0] * at(0, j) + ea[1] * at(1, j) + ea[2] * at(2, j) + eb[j] {
            return false;
        }
    }
    for i in 0..3 {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        for j in 0..3 {
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let radius_a = ea[i1] * at(i2, j) + ea[i2] * at(i1, j);
            let radius_b = eb[j1] * at(i, j2) + eb[j2] * at(i, j1);
            if (t[i2] * rt(i1, j) - t[i1] * rt(i2, j)).abs() > radius_a + radius_b {
                return false;
            }
        }
    }
    true
}

/// Test every interaction box against the character's probe box (upstream
/// `compute_interaction_box_overlaps`). Touching boxes count as overlapping.
pub fn compute_interaction_box_overlaps(interaction_boxes: &[InteractionBox], probe_box: &InteractionBox) -> Vec<bool> {
    interaction_boxes.iter().map(|b| boxes_overlap(b, probe_box)).collect()
}
