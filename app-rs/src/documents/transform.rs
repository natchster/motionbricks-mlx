// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The transform shared by every document, ported from upstream's `wp.transform`
//! JSON handling in `app/runtime/common/io/document_io.py`.

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::{DocumentError, Result};

/// How far a quaternion's norm may drift from 1 before a document is rejected.
/// Mirrors `validate_transforms()` in upstream `document_io.py`.
pub const QUATERNION_NORM_TOLERANCE: f32 = 1.0e-3;

/// Angles smaller than this are written as 0, upstream's `clean_display_value()`
/// in `app/authoring/common.py` (`_DISPLAY_ZERO_EPSILON = 5.0e-4`).
const DISPLAY_ZERO_EPSILON_DEGREES: f32 = 5.0e-4;

/// A rigid transform: translation plus an XYZW quaternion.
///
/// Serializes exactly like upstream's `serialize_transform()`:
/// `{"translation": [x, y, z], "rotation_xyzw": [x, y, z, w]}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation_xyzw: Quat,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation_xyzw: Quat::IDENTITY,
    };

    pub const fn new(translation: Vec3, rotation_xyzw: Quat) -> Self {
        Self {
            translation,
            rotation_xyzw,
        }
    }

    /// Require finite values and a unit quaternion (within 1e-3), like upstream's
    /// `validate_transforms()`.
    pub fn validate(&self, location: &str) -> Result<()> {
        if !self.translation.is_finite() || !self.rotation_xyzw.is_finite() {
            return Err(DocumentError::new(format!(
                "{location} contains non-finite values."
            )));
        }
        if (self.rotation_xyzw.length() - 1.0).abs() > QUATERNION_NORM_TOLERANCE {
            return Err(DocumentError::new(format!(
                "{location} contains a non-unit quaternion."
            )));
        }
        Ok(())
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation_xyzw, self.translation)
    }

    /// Convert a matrix back to a rigid transform.
    ///
    /// `wp.transform` has no scale, so a matrix whose basis columns are not unit
    /// length (within [`QUATERNION_NORM_TOLERANCE`]) is rejected.
    pub fn from_mat4(matrix: &Mat4) -> Result<Self> {
        let (scale, rotation_xyzw, translation) = matrix.to_scale_rotation_translation();
        if (scale - Vec3::ONE).abs().max_element() > QUATERNION_NORM_TOLERANCE {
            return Err(DocumentError::new(
                "transform matrices must not contain scale.".to_string(),
            ));
        }
        Ok(Self::new(translation, rotation_xyzw))
    }

    /// Compose two transforms, upstream's `wp.mul(lhs, rhs)`: apply `rhs` first.
    pub fn mul(&self, rhs: &Self) -> Self {
        Self::new(
            self.translation + self.rotation_xyzw * rhs.translation,
            self.rotation_xyzw * rhs.rotation_xyzw,
        )
    }

    pub fn inverse(&self) -> Self {
        let rotation_xyzw = self.rotation_xyzw.inverse();
        Self::new(rotation_xyzw * (-self.translation), rotation_xyzw)
    }

    /// Roll, pitch, yaw in degrees, recomputed from the quaternion.
    ///
    /// Mirrors upstream's sync rule (`stable_rotation_degrees()` with no reference
    /// angles in `app/authoring/common.py`): `wp.quat_to_rpy()` converted to
    /// degrees, with values below 5e-4 degrees snapped to 0. The quaternion is
    /// authoritative; the stored `euler_degrees` fields are derived display data.
    pub fn euler_degrees(&self) -> Vec3 {
        let (x, y, z, w) = (
            self.rotation_xyzw.x,
            self.rotation_xyzw.y,
            self.rotation_xyzw.z,
            self.rotation_xyzw.w,
        );
        // wp.quat_to_rpy(): roll about x, pitch about y, yaw about z.
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        Vec3::new(
            clean_display_value(roll.to_degrees()),
            clean_display_value(pitch.to_degrees()),
            clean_display_value(yaw.to_degrees()),
        )
    }

    /// The inverse of [`Self::euler_degrees`], upstream's `wp.quat_rpy()`:
    /// roll, pitch, yaw applied in that order about the fixed x, y, z axes.
    pub fn quat_from_euler_degrees(euler_degrees: Vec3) -> Quat {
        Quat::from_axis_angle(Vec3::Z, euler_degrees.z.to_radians())
            * Quat::from_axis_angle(Vec3::Y, euler_degrees.y.to_radians())
            * Quat::from_axis_angle(Vec3::X, euler_degrees.x.to_radians())
    }
}

fn clean_display_value(value: f32) -> f32 {
    if value.abs() < DISPLAY_ZERO_EPSILON_DEGREES {
        0.0
    } else {
        value
    }
}
