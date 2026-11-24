// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Viewer demo: ground grid, a spinning (gizmo-movable) cube, a two-bone skeleton drawn
//! with lines, a circling arrow, and an imgui panel with fps + cube color.
//!
//! Run with: `cargo run --bin viewer-demo`
//! Headless screenshot: `cargo run --bin viewer-demo -- --headless frame.ppm --frames 30`
//! (renders offscreen — no window — and writes a P6 PPM image).

use std::path::PathBuf;

use anyhow::Result;
use glam::{Mat4, Vec3};
use lexopt::prelude::*;
use motionbricks_app::viewer::{Frame, HeadlessViewer, Mesh, RgbaImage, Viewer, ViewerConfig};

const CUBE_COLOR: [f32; 4] = [0.85, 0.55, 0.2, 1.0];

/// The demo's 3D scene at time `t` (the imgui panel is windowed-only).
fn draw_scene(frame: &mut Frame, cube: &Mesh, t: f32, cube_transform: &mut Mat4, cube_color: [f32; 4]) {
    // Ground plane.
    frame.log_ground();

    // Spinning cube: the mesh is centered at the origin, so `translation * spin`
    // rotates it about its own (gizmo-controlled) center.
    let spin = Mat4::from_rotation_y(t * 0.8);
    frame.log_mesh(cube, *cube_transform * spin, cube_color);

    // Two-bone "skeleton" swinging at the hip.
    let hip = Vec3::new(1.5, 1.2, 0.0);
    let swing = (t * 1.7).sin() * 0.6;
    let knee = hip + Vec3::new(swing.sin() * 0.6, -0.6, 0.0);
    let foot = knee + Vec3::new(0.0, -0.6, 0.15 * (t * 3.4).cos());
    frame.log_lines(&[[hip, knee], [knee, foot]], [0.9, 0.9, 0.95, 1.0]);
    for joint in [hip, knee, foot] {
        frame.log_wire_box(joint, Vec3::splat(0.05), [0.95, 0.35, 0.4, 1.0]);
    }

    // Arrow circling the skeleton.
    let a0 = Vec3::new(-1.5, 0.2, 0.0);
    let a1 = a0 + Vec3::new(t.cos(), 0.4 + 0.3 * (t * 2.0).sin(), t.sin());
    frame.log_arrow(a0, a1, [0.4, 0.85, 0.9, 1.0]);

    // Translate gizmo drives the cube (a no-op headless: there is no input).
    frame.gizmo_translate("demo-cube", cube_transform);
}

/// Render `frames` frames of the demo offscreen and capture the last one.
fn headless_capture(frames: u32) -> Result<RgbaImage> {
    let cube = Mesh::cube(Vec3::ZERO, Vec3::splat(0.5));
    let mut cube_transform = Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
    let mut viewer = HeadlessViewer::new(1280, 720)?;
    let dt = 1.0 / 60.0;
    let mut image = None;
    for frame_index in 0..frames.max(1) {
        let t = frame_index as f32 * dt;
        image = Some(viewer.capture(dt, |frame| draw_scene(frame, &cube, t, &mut cube_transform, CUBE_COLOR))?);
    }
    Ok(image.expect("at least one frame ran"))
}

fn main() -> Result<()> {
    let mut headless = false;
    let mut out = PathBuf::from("frame.ppm");
    let mut frames = 30;
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Long("headless") => headless = true,
            Long("frames") => frames = parser.value()?.parse()?,
            Long("help") => {
                eprintln!("Usage: viewer-demo [--headless [OUT.ppm]] [--frames N]");
                return Ok(());
            }
            Value(value) => out = PathBuf::from(value.string()?),
            _ => return Err(arg.unexpected().into()),
        }
    }

    if headless {
        let image = headless_capture(frames)?;
        image.write_ppm(&out)?;
        eprintln!("viewer-demo: wrote {} ({}x{}, {frames} frames)", out.display(), image.width, image.height);
        return Ok(());
    }

    let viewer = Viewer::new(ViewerConfig {
        title: "MotionBricks viewer demo".into(),
        width: 1280,
        height: 720,
    })?;

    let cube = Mesh::cube(Vec3::ZERO, Vec3::splat(0.5));
    let start = std::time::Instant::now();
    let mut cube_transform = Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
    let mut cube_color = CUBE_COLOR;
    let mut frames = 0u64;

    viewer.run(move |frame, ui| {
        let t = start.elapsed().as_secs_f32();
        frames += 1;
        draw_scene(frame, &cube, t, &mut cube_transform, cube_color);

        // imgui panel.
        ui.window("Viewer demo").build(|| {
            ui.text(format!("fps: {:.1}", 1.0 / frame.dt().max(1e-4)));
            ui.text(format!("frame: {frames}"));
            ui.separator();
            ui.color_edit4("cube color", &mut cube_color);
            ui.separator();
            ui.text("left-drag: orbit, middle/right-drag: pan, wheel: zoom");
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offscreen capture, or None when the machine has no GPU (skip then).
    fn capture() -> Option<RgbaImage> {
        match headless_capture(2) {
            Ok(image) => Some(image),
            Err(error) => {
                eprintln!("skipping headless demo test (no GPU): {error:#}");
                None
            }
        }
    }

    #[test]
    fn headless_capture_renders_the_demo_scene() {
        let Some(image) = capture() else { return };
        assert_eq!((image.width, image.height), (1280, 720));
        // The ground grid fills much of the lower half, so a good share of the
        // pixels is not the background.
        let background = motionbricks_app::viewer::clear_color_rgba();
        assert!(image.count_not(background) > 5_000, "the scene draws visibly");
        // The cube's orange and the skeleton's light lines appear somewhere.
        let cube_rgb = [221, 144, 55]; // ~[0.85, 0.55, 0.2] shaded, generous tolerance
        assert!(image.contains_color(cube_rgb, 90), "the cube color appears");
        assert!(image.contains_color([229, 229, 242], 60), "the skeleton lines appear");
    }

    #[test]
    fn headless_capture_is_deterministic() {
        let Some(first) = capture() else { return };
        let Some(second) = capture() else { return };
        assert_eq!(first, second, "same frames, same pixels");
    }
}
