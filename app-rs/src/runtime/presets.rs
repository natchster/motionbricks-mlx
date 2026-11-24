// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Default locomotion modes, idle clips, sitting clips, and probe sizes for each rig,
//! ported from upstream `app/runtime/soma/presets.py` and
//! `app/runtime/unitree_g1/presets.py`.
//!
//! Paths are relative to the app's asset root (upstream's `ASSETS_ROOT`); the bins
//! resolve them.

use std::path::PathBuf;

use glam::Vec2;

use super::interaction::InteractionProbe;
use super::motion::{
    CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, DIRECTIONAL_TARGET_VALUES, LINEAR_LOCOMOTION_MODE, LocomotionMode, StyleModeOptions,
    style_locomotion_mode,
};
use super::blender::BlendSpaceType;

fn directional_target_values() -> Vec<(String, Vec2)> {
    DIRECTIONAL_TARGET_VALUES.iter().map(|(name, value)| (name.to_string(), Vec2::new(value[0], value[1]))).collect()
}

fn checked(mode: Result<LocomotionMode, anyhow::Error>) -> LocomotionMode {
    mode.expect("the preset locomotion modes are valid")
}

/// SOMA presets (upstream `app/runtime/soma/presets.py`).
pub mod soma {
    use super::*;

    const SMART_PRIMITIVE_ROOT: &str = "smart_primitives/soma";

    pub const DEFAULT_IDLE_CLIPS: [&str; 4] = [
        "motions/soma/locomotion/idle/idle_loop_001__A268.bvh",
        "motions/soma/locomotion/idle/neutral_looking_around_R_001__A542.bvh",
        "motions/soma/locomotion/idle/warm_up_neck_001__A359.bvh",
        "motions/soma/locomotion/idle/legs_relax_002__A407.bvh",
    ];
    pub const DEFAULT_CROUCH_IDLE_CLIPS: [&str; 1] = ["motions/soma/locomotion/idle/crouch_idle_001__A529.bvh"];

    /// The rig's looping sitting clip, for `SittingBehavior`.
    pub const SITTING_CLIP: &str = "motions/soma/sitting/sit_on_chair_loop.bvh";

    const NORMAL_DEFAULT_SPEED: f32 = 2.3;
    const NORMAL_MAX_SPEED: f32 = 4.5;
    const STRAFE_DEFAULT_SPEED: f32 = 2.3;
    const STRAFE_MAX_SPEED: f32 = 3.0;
    const CROUCH_DEFAULT_SPEED: f32 = 1.5;
    const CROUCH_MAX_SPEED: f32 = 2.5;

    pub fn linear_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            LINEAR_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs1_linear.smart_primitive.json")),
            BlendSpaceType::Linear1D,
            vec![
                ("Idle".to_string(), Vec2::ZERO),
                ("Walk".to_string(), Vec2::new(NORMAL_DEFAULT_SPEED / NORMAL_MAX_SPEED, 0.0)),
                ("Jog".to_string(), Vec2::new(3.0 / NORMAL_MAX_SPEED, 0.0)),
                ("Run".to_string(), Vec2::X),
            ],
            NORMAL_MAX_SPEED,
            NORMAL_DEFAULT_SPEED,
            0.8 * NORMAL_DEFAULT_SPEED,
            10,
            (8..=14).collect(),
            0.97,
            1.0,
            "Forward",
        ))
    }

    pub fn directional_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            DIRECTIONAL_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs2_directional.smart_primitive.json")),
            BlendSpaceType::Directional2D,
            directional_target_values(),
            STRAFE_MAX_SPEED,
            STRAFE_DEFAULT_SPEED,
            0.8 * STRAFE_DEFAULT_SPEED,
            8,
            (8..=14).collect(),
            0.95,
            0.9,
            "Strafing",
        ))
    }

    pub fn crouch_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            CROUCH_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs2_directional_crouch.smart_primitive.json")),
            BlendSpaceType::Directional2D,
            directional_target_values(),
            CROUCH_MAX_SPEED,
            CROUCH_DEFAULT_SPEED,
            CROUCH_DEFAULT_SPEED * 0.8,
            8,
            (8..=14).collect(),
            0.9,
            0.8,
            "Crouching",
        ))
    }

    pub fn style_locomotion_modes() -> Vec<LocomotionMode> {
        let style_path = PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_style_variations_bs1_linear.smart_primitive.json"));
        let defaults = StyleModeOptions::default();
        vec![
            checked(style_locomotion_mode(
                "Skipping",
                style_path.clone(),
                1.3,
                1.45,
                StyleModeOptions { position_scale: 1.32, allowed_token_counts: (6..=13).collect(), ..defaults.clone() },
            )),
            checked(style_locomotion_mode("Vogue", style_path.clone(), 1.2, 1.3, StyleModeOptions { position_scale: 1.3, ..defaults.clone() })),
            checked(style_locomotion_mode(
                "Injured Torso",
                style_path.clone(),
                0.7,
                0.8,
                StyleModeOptions { position_scale: 1.2, transition_early_by_frames: 6, allowed_token_counts: (8..=16).collect(), ..defaults.clone() },
            )),
            checked(style_locomotion_mode("Injured Leg", style_path.clone(), 0.7, 0.8, StyleModeOptions { position_scale: 1.2, ..defaults.clone() })),
            checked(style_locomotion_mode("Zombie", style_path.clone(), 0.7, 0.85, StyleModeOptions { position_scale: 1.3, ..defaults.clone() })),
            checked(style_locomotion_mode("Combat", style_path.clone(), 0.7, 1.0, StyleModeOptions { position_scale: 1.7, ..defaults.clone() })),
        ]
    }

    pub fn default_locomotion_modes() -> Vec<LocomotionMode> {
        let mut modes = vec![linear_locomotion(), directional_locomotion(), crouch_locomotion()];
        modes.extend(style_locomotion_modes());
        modes
    }

    pub fn interaction_probe() -> InteractionProbe {
        InteractionProbe {
            box_width: 1.0,
            box_min_depth: 0.75,
            box_height: 1.6,
            near_distance: 0.15,
            lookahead_time: 1.0,
            extra_lookahead_min_speed: 2.3,
            extra_lookahead_time: 0.25,
        }
    }
}

/// Unitree G1 presets (upstream `app/runtime/unitree_g1/presets.py`).
pub mod unitree_g1 {
    use super::*;

    const SMART_PRIMITIVE_ROOT: &str = "smart_primitives/unitree_g1";

    pub const DEFAULT_IDLE_CLIPS: [&str; 3] = [
        "motions/unitree_g1/locomotion/idle/idle_loop_001__A268.csv",
        "motions/unitree_g1/locomotion/idle/neutral_looking_around_R_001__A542.csv",
        "motions/unitree_g1/locomotion/idle/legs_relax_002__A407.csv",
    ];
    pub const DEFAULT_CROUCH_IDLE_CLIPS: [&str; 1] = ["motions/unitree_g1/locomotion/idle/crouch_idle_001__A529.csv"];

    /// The rig's looping sitting clip, for `SittingBehavior`.
    pub const SITTING_CLIP: &str = "motions/unitree_g1/sitting/sit_on_chair_loop.csv";

    pub fn linear_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            LINEAR_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs1_linear.smart_primitive.json")),
            BlendSpaceType::Linear1D,
            vec![
                ("Idle".to_string(), Vec2::ZERO),
                ("Walk".to_string(), Vec2::new(1.3 / 3.4, 0.0)),
                ("Jog".to_string(), Vec2::new(2.0 / 3.4, 0.0)),
                ("Run".to_string(), Vec2::X),
            ],
            3.4,
            1.3,
            1.3,
            10,
            (8..=14).collect(),
            1.0,
            1.0,
            "Forward",
        ))
    }

    pub fn directional_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            DIRECTIONAL_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs2_directional.smart_primitive.json")),
            BlendSpaceType::Directional2D,
            directional_target_values(),
            2.3,
            1.3,
            1.3,
            10,
            (8..=14).collect(),
            1.0,
            1.0,
            "Strafing",
        ))
    }

    pub fn crouch_locomotion() -> LocomotionMode {
        checked(LocomotionMode::new(
            CROUCH_LOCOMOTION_MODE,
            PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_bs2_directional_crouch.smart_primitive.json")),
            BlendSpaceType::Directional2D,
            directional_target_values(),
            1.5,
            0.8,
            0.8,
            10,
            (8..=14).collect(),
            1.0,
            1.0,
            "Crouching",
        ))
    }

    pub fn style_locomotion_modes() -> Vec<LocomotionMode> {
        let style_path = PathBuf::from(format!("{SMART_PRIMITIVE_ROOT}/locomotion_style_variations_bs1_linear.smart_primitive.json"));
        let defaults = StyleModeOptions::default();
        vec![
            checked(style_locomotion_mode("Skipping", style_path.clone(), 1.3, 1.45, StyleModeOptions { transition_early_by_frames: 0, ..defaults.clone() })),
            checked(style_locomotion_mode("Vogue", style_path.clone(), 0.8, 0.9, StyleModeOptions { position_scale: 1.3, ..defaults.clone() })),
            checked(style_locomotion_mode("Injured Torso", style_path.clone(), 0.4, 0.45, defaults.clone())),
            checked(style_locomotion_mode("Injured Leg", style_path.clone(), 0.4, 0.6, defaults.clone())),
            checked(style_locomotion_mode("Zombie", style_path.clone(), 0.4, 0.45, defaults.clone())),
            checked(style_locomotion_mode("Combat", style_path.clone(), 0.65, 0.75, defaults.clone())),
        ]
    }

    pub fn default_locomotion_modes() -> Vec<LocomotionMode> {
        let mut modes = vec![linear_locomotion(), directional_locomotion(), crouch_locomotion()];
        modes.extend(style_locomotion_modes());
        modes
    }

    pub fn interaction_probe() -> InteractionProbe {
        InteractionProbe {
            box_width: 0.79,
            box_min_depth: 0.59,
            box_height: 1.26,
            near_distance: 0.12,
            lookahead_time: 1.0,
            extra_lookahead_min_speed: 1.3,
            extra_lookahead_time: 0.25,
        }
    }
}
