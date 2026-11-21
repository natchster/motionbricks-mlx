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

/// What the loop drives: either rig's player with its placed target poses.
trait Character {
    /// Generate toward the target poses placed at `ground`, facing `direction`.
    fn generate(&mut self, ground: Vec3, direction: Vec3) -> Result<()>;
    /// Sample and record the shown pose; return its root position and the time to the transition point.
    fn step(&mut self, dt: f32) -> Result<(Vec3, f64)>;
}

/// A player, the jog target poses it places, and the pose buffer reused every update.
struct Runner<E: MotionBricksEngine> {
    player: GeneratedPlayer<E>,
    jog: Vec<Vec<E::Elem>>,
    pose: Vec<E::Elem>,
    settings: TargetPoseInferenceParams,
}

impl<B: Backend> Character for Runner<MotionBricksSomaEngine<B>> {
    fn generate(&mut self, ground: Vec3, direction: Vec3) -> Result<()> {
        self.player.generate(&place_target_poses(ground, direction, &self.jog), &self.settings, None)
    }
    fn step(&mut self, dt: f32) -> Result<(Vec3, f64)> {
        let _contacts = self.player.sample_into(dt, &mut self.pose)?;
        // Apply target pose IK, physics, or other changes to the pose here, before recording.
        self.player.record_pose(dt, &self.pose)?;
        Ok((self.pose[0].p, self.player.time_to_transition()))
    }
}

impl<B: Backend> Character for Runner<MotionBricksUnitreeG1Engine<B>> {
    fn generate(&mut self, ground: Vec3, direction: Vec3) -> Result<()> {
        self.player.generate(&place_target_dof(ground, direction, &self.jog), &self.settings, None)
    }
    fn step(&mut self, dt: f32) -> Result<(Vec3, f64)> {
        let _contacts = self.player.sample_into(dt, &mut self.pose)?;
        self.player.record_pose(dt, &self.pose)?;
        Ok((read_root(&self.pose).p, self.player.time_to_transition()))
    }
}

/// Run the update loop: generate when there is no motion, the steering turned, or the transition
/// point is near; then sample and record.
fn run_loop(character: &mut impl Character, speed: f32, start: Vec3) -> Result<()> {
    let dt = 1.0 / UPDATE_RATE;
    let (mut position, mut to_transition) = (start, 0.0);
    let mut generated: Option<Vec3> = None;
    let (mut generations, mut busy) = (0, 0.0);
    for step_index in 0..(DURATION * UPDATE_RATE).round() as usize {
        let t = step_index as f32 * dt;
        let direction = steering(t);
        let turned = generated.is_some_and(|g| direction.dot(g).min(1.0).acos().to_degrees() > TURN_THRESHOLD);
        if generated.is_none() || turned || to_transition <= dt as f64 {
            let ground = Vec3::new(position.x(), position.y(), 0.0) + direction * (speed * LEAD_SECONDS);
            let s = Instant::now();
            character.generate(ground, direction)?;
            busy += s.elapsed().as_secs_f64();
            generated = Some(direction);
            generations += 1;
        }
        (position, to_transition) = character.step(dt)?;
        if step_index % UPDATE_RATE as usize == 0 {
            println!("t = {t:4.1} s  steering [{:.2} {:.2}]  root [{:.2} {:.2}]", direction.x(), direction.y(), position.x(), position.y());
        }
    }
    println!("{generations} generations in {DURATION:.0} s, {:.1} ms each", busy / generations as f64 * 1e3);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let rig = ModelRig::parse(args.get(1).map_or("soma", String::as_str))?;
    let precision = args.get(2).map_or("float32", String::as_str);
    let backend = load_model(rig, precision)?;
    let settings = TargetPoseInferenceParams::new(TRANSITION_EARLY_BY, 4);
    match load_clip(rig, JOG)? {
        Clip::Soma { clip, skeleton } => {
            // The ground speed to move at, in m/s; the target poses are scaled to it.
            let speed = 1.9;
            let raw = soma_window(&clip, TARGET_TIME, Transform::IDENTITY)?;
            let mut jog = soma_window(&clip, TARGET_TIME, canonical(&raw.iter().map(|p| p[0].p).collect::<Vec<_>>()))?;
            adjust_target_pose_speed(&mut jog, speed)?;
            let engine = MotionBricksSomaEngine::new(Some(backend), model_to_runtime(ModelRig::Soma))?;
            let mut player = GeneratedMotionPlayer::new(engine, &skeleton, 32)?;
            // Start at the origin in the jog pose, as if the character had held it.
            let start = jog[MODEL_POSES_PER_TOKEN - 1].clone();
            player.seed_history_from_pose(&start)?;
            run_loop(&mut Runner { player, jog, pose: start.clone(), settings }, speed, start[0].p)
        }
        Clip::G1(clip) => {
            let speed = 1.45; // a higher G1 speed makes its feet slide
            let raw = dof_window(&clip, TARGET_TIME, Transform::IDENTITY)?;
            let mut jog = dof_window(&clip, TARGET_TIME, canonical(&raw.iter().map(|q| read_root(q).p).collect::<Vec<_>>()))?;
            adjust_target_dof_speed(&mut jog, speed)?;
            let engine = MotionBricksUnitreeG1Engine::new(Some(backend), model_to_runtime(ModelRig::UnitreeG1))?;
            let reference = clip.sample(0.0, Transform::IDENTITY)?;
            let mut player = GeneratedDofMotionPlayer::new(engine, &reference, 32)?;
            let start = jog[MODEL_POSES_PER_TOKEN - 1].clone();
            player.seed_history_from_pose(&start)?;
            run_loop(&mut Runner { player, jog, pose: start.clone(), settings }, speed, read_root(&start).p)
        }
    }
}
