// SPDX-License-Identifier: Apache-2.0
// Ported from NVlabs/motionbricks docs/examples/locomotion_loop.py. Original: Copyright (c) 2026
// NVIDIA CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
//! Move a character along a steered path with a generation loop. No viewer, no apps.
//!
//! Each update plays the generated motion, records the pose it showed, and decides whether to
//! generate again. The target poses are one set of jog poses, placed ahead of the character along
//! the steering direction, which turns from +X to +Y, so the path bends left.
//!
//!     cargo run --release --example locomotion_loop -- [soma|unitree_g1] [float32|float16|bfloat16|int8]

#[path = "support/runtime_rig.rs"]
mod runtime_rig;

use std::time::Instant;

use anyhow::Result;
use motionbricks_rs::runtime::math::quat_from_axis_angle;
use motionbricks_rs::runtime::*;
use runtime_rig::*;

const UPDATE_RATE: f32 = 60.0;
const DURATION: f32 = 9.0;
/// The jog poses that serve as the target. Any clip time in the jog works.
const TARGET_TIME: f32 = 1.8;
/// The target sits this many seconds of travel ahead of the character.
const LEAD_SECONDS: f32 = 1.0;
/// The transition point, in frames before the end of each generation.
const TRANSITION_EARLY_BY: usize = 4;
/// Generate again when the steering direction turns by more than this many degrees.
const TURN_THRESHOLD: f32 = 15.0;
/// The steering holds +X, turns to +Y between these two times, then holds.
const TURN: (f32, f32) = (2.0, 4.0);

/// The desired ground direction at a time. An application reads its controls here.
fn steering(t: f32) -> Vec3 {
    let blend = ((t - TURN.0) / (TURN.1 - TURN.0)).clamp(0.0, 1.0);
    let angle = std::f32::consts::FRAC_PI_2 * blend;
    Vec3::new(angle.cos(), angle.sin(), 0.0)
}

/// The transform that removes the clip's own placement and heading (origin, facing +X).
fn canonical(roots: &[Vec3]) -> Transform {
    let (first, last) = (roots[0], roots[roots.len() - 1]);
    let heading = (last.y() - first.y()).atan2(last.x() - first.x());
    Transform::new(Vec3::new(first.x(), first.y(), 0.0), quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), heading)).inverse()
}

/// Shared loop: `generate(position, direction)` runs a generation, `step(dt)` samples and records
/// the shown pose and returns its root position and the time to the transition point.
fn run_loop(mut generate: impl FnMut(Vec3, Vec3) -> Result<()>, mut step: impl FnMut(f32) -> Result<(Vec3, f32)>, speed: f32, start: Vec3) -> Result<()> {
    let dt = 1.0 / UPDATE_RATE;
    let (mut position, mut to_transition) = (start, 0.0);
    let mut generated: Option<Vec3> = None;
    let (mut generations, mut busy) = (0, 0.0);
    for step_index in 0..(DURATION * UPDATE_RATE).round() as usize {
        let t = step_index as f32 * dt;
        let direction = steering(t);
        let turned = generated.is_some_and(|g| direction.dot(g).min(1.0).acos().to_degrees() > TURN_THRESHOLD);
        if generated.is_none() || turned || to_transition <= dt {
            let ground = Vec3::new(position.x(), position.y(), 0.0) + direction * (speed * LEAD_SECONDS);
            let s = Instant::now();
            generate(ground, direction)?;
            busy += s.elapsed().as_secs_f64();
            generated = Some(direction);
            generations += 1;
        }
        // Apply target pose IK, physics, or other changes to the pose inside `step`, before recording.
        (position, to_transition) = step(dt)?;
        if step_index % UPDATE_RATE as usize == 0 {
            println!("t = {t:4.1} s  steering [{:.2} {:.2}]  root [{:.2} {:.2}]", direction.x(), direction.y(), position.x(), position.y());
        }
    }
    println!("{generations} generations in {DURATION:.0} s, {:.1} ms each", busy / generations as f64 * 1e3);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).map_or("soma", String::as_str);
    let precision = args.get(2).map_or("float32", String::as_str);
    let backend = load_model(model, precision)?;
    let settings = TargetPoseInferenceParams::new(TRANSITION_EARLY_BY, 4);
    match load_clip(model, JOG)? {
        Clip::Soma { clip, skeleton } => {
            // The ground speed to move at, in m/s; the target poses are scaled to it.
            let speed = 1.9;
            let raw = soma_window(&clip, TARGET_TIME, Transform::IDENTITY);
            let mut jog = soma_window(&clip, TARGET_TIME, canonical(&raw.iter().map(|p| p[0].p).collect::<Vec<_>>()));
            adjust_target_pose_speed(&mut jog, speed)?;
            let engine = MotionBricksSomaEngine::new(Some(backend), model_to_runtime(ModelRig::Soma))?;
            let player = std::cell::RefCell::new(GeneratedMotionPlayer::new(engine, &skeleton, 32)?);
            // Start at the origin in the jog pose, as if the character had held it.
            player.borrow_mut().seed_history_from_pose(&jog[MODEL_POSES_PER_TOKEN - 1])?;
            run_loop(
                |ground, direction| player.borrow_mut().generate(&place_target_poses(ground, direction, &jog), &settings, None),
                |dt| {
                    let mut p = player.borrow_mut();
                    let (pose, _contacts) = p.sample(dt)?;
                    p.record_pose(dt, &pose)?;
                    Ok((pose[0].p, p.time_to_transition()))
                },
                speed,
                jog[MODEL_POSES_PER_TOKEN - 1][0].p,
            )
        }
        Clip::G1(clip) => {
            let speed = 1.45; // a higher G1 speed makes its feet slide
            let raw = dof_window(&clip, TARGET_TIME, Transform::IDENTITY);
            let mut jog = dof_window(&clip, TARGET_TIME, canonical(&raw.iter().map(|q| read_root(q).p).collect::<Vec<_>>()));
            adjust_target_dof_speed(&mut jog, speed)?;
            let engine = MotionBricksUnitreeG1Engine::new(Some(backend), model_to_runtime(ModelRig::UnitreeG1))?;
            let reference = clip.sample(0.0, Transform::IDENTITY)?;
            let player = std::cell::RefCell::new(GeneratedDofMotionPlayer::new(engine, reference, 32)?);
            player.borrow_mut().seed_history_from_pose(&jog[MODEL_POSES_PER_TOKEN - 1])?;
            run_loop(
                |ground, direction| player.borrow_mut().generate(&place_target_dof(ground, direction, &jog), &settings, None),
                |dt| {
                    let mut p = player.borrow_mut();
                    let (pose, _contacts) = p.sample(dt)?;
                    p.record_pose(dt, &pose)?;
                    Ok((read_root(&pose).p, p.time_to_transition()))
                },
                speed,
                read_root(&jog[MODEL_POSES_PER_TOKEN - 1]).p,
            )
        }
    }
}
