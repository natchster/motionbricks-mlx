// SPDX-License-Identifier: Apache-2.0
// Ported from NVlabs/motionbricks docs/examples/generate_once.py. Original: Copyright (c) 2026
// NVIDIA CORPORATION & AFFILIATES, Apache-2.0. Modifications Copyright 2026 The motionbricks-mlx Authors.
//! Generate motion between clip poses, then chain a second generation. No viewer, no apps.
//!
//! Each generation takes four context poses and one to four target poses. The first context and
//! both targets come from one walk clip. The second generation reads its context from the last four
//! poses of the first, which is the loop that makes continuous motion.
//!
//!     cargo run --release --example generate_once -- [soma|unitree_g1] [float32|float16|bfloat16|int8]

#[path = "support/runtime_rig.rs"]
mod runtime_rig;

use anyhow::Result;
use motionbricks_rs::runtime::*;
use runtime_rig::*;

/// The context ends at this clip time. Each generation targets the clip poses at the next time.
const CONTEXT_END: f32 = 1.0;
const TARGET_TIMES: [f32; 2] = [1.8, 3.4];

fn report(frames: usize, roots: &[Transform], timer: &GenerationTimer) {
    let (a, b) = (roots[0].p.0, roots[frames - 1].p.0);
    println!(
        "generated {frames} frames ({:.2} s), root from [{:.2} {:.2} {:.2}] to [{:.2} {:.2} {:.2}]  [ms: encode {:.2}, inference {:.2}, decode {:.2}, post_process {:.2}]",
        frames as f32 / MODEL_SAMPLE_RATE, a[0], a[1], a[2], b[0], b[1], b[2], timer.encode, timer.inference, timer.decode, timer.post_process
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).map_or("soma", String::as_str);
    let precision = args.get(2).map_or("float32", String::as_str);
    let backend = load_model(model, precision)?;
    let config = MotionBricksInferenceConfig::default();
    let context_start = CONTEXT_END - (MODEL_POSES_PER_TOKEN - 1) as f32 / MODEL_SAMPLE_RATE;
    match load_clip(model, WALK)? {
        Clip::Soma { clip, skeleton } => {
            let mut engine = MotionBricksSomaEngine::new(Some(backend), model_to_runtime(ModelRig::Soma))?;
            let (mut context, mut target, mut output) = (engine.create_input(&skeleton)?, engine.create_input(&skeleton)?, engine.create_output(&skeleton)?);
            let mut context_poses = soma_window(&clip, context_start, Transform::IDENTITY);
            for t in TARGET_TIMES {
                context.update_from_poses(&context_poses)?;
                target.update_from_poses(&soma_window(&clip, t, Transform::IDENTITY))?;
                engine.generate_animation(&context, &target, &mut output, &config)?;
                // The output buffer has room for the longest result. Only num_frames frames are valid.
                let frames = output.num_frames;
                report(frames, &output.root_transforms(), &engine.timer);
                // Play the whole motion, so the last four poses are what the character showed.
                context_poses = (frames - MODEL_POSES_PER_TOKEN..frames).map(|f| output.frame(f).to_vec()).collect();
            }
        }
        Clip::G1(clip) => {
            let mut engine = MotionBricksUnitreeG1Engine::new(Some(backend), model_to_runtime(ModelRig::UnitreeG1))?;
            let (mut context, mut target, mut output) = (engine.create_input(), engine.create_input(), engine.create_output());
            let mut context_poses = dof_window(&clip, context_start, Transform::IDENTITY);
            for t in TARGET_TIMES {
                context.update_from_poses(&context_poses)?;
                target.update_from_poses(&dof_window(&clip, t, Transform::IDENTITY))?;
                engine.generate_animation(&context, &target, &mut output, &config)?;
                let frames = output.num_frames;
                report(frames, &output.root_transforms(), &engine.timer);
                context_poses = (frames - MODEL_POSES_PER_TOKEN..frames).map(|f| output.frame(f).to_vec()).collect();
            }
        }
    }
    Ok(())
}
