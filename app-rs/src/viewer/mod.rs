// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! wgpu-based viewer with the same logging-shaped API as upstream's Newton `ViewerGL`.
//!
//! The MotionBricks applications never render directly: once per frame they *log* primitives
//! (meshes, skeleton lines, boxes, arrows, a ground plane) into a [`Frame`] and build their
//! imgui panels. [`Viewer::run`] drives the winit event loop and hands the app a fresh
//! `Frame` plus the `imgui::Ui` every frame.

pub mod camera;
pub mod gizmo;
pub mod headless;
pub mod mesh;
pub mod picking;
pub mod render;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use glam::{Mat3, Mat4, Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

pub use camera::OrbitCamera;
pub use headless::{RgbaImage, clear_color_rgba};
pub use mesh::Mesh;
pub use picking::{Pick, Ray};

/// RGBA color, linear-space components in `0.0..=1.0`.
pub type Color = [f32; 4];

/// Linear-blend skinning data for [`Frame::log_skinned_mesh`].
///
/// `weights` and `weight_values` are per-vertex: up to four joint indices (into
/// `bone_transforms` / `inverse_bind`) and their blend weights, which should sum to 1.
#[derive(Clone, Debug)]
pub struct Skin {
    /// Parent joint index per joint, `-1` for roots. Not used by LBS itself, but kept so
    /// apps can round-trip the upstream skin description verbatim.
    pub joint_parents: Vec<i32>,
    /// One inverse bind-pose matrix per joint.
    pub inverse_bind: Vec<Mat4>,
    /// Per-vertex joint indices (up to 4 influences).
    pub weights: Vec<[u16; 4]>,
    /// Per-vertex blend weights matching `weights`.
    pub weight_values: Vec<[f32; 4]>,
}

/// Window configuration for [`Viewer::new`].
pub struct ViewerConfig {
    /// Window title.
    pub title: String,
    /// Initial window width in logical pixels.
    pub width: u32,
    /// Initial window height in logical pixels.
    pub height: u32,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self { title: "MotionBricks".into(), width: 1280, height: 720 }
    }
}

/// Per-frame input snapshot handed to the app through [`Frame::input`].
///
/// "pressed" means the key/button went down during this frame; "held" means it is
/// currently down. All coordinates are logical pixels with the origin at the top-left
/// of the window's client area.
#[derive(Clone, Debug, Default)]
pub struct Input {
    keys_pressed: HashSet<KeyCode>,
    keys_held: HashSet<KeyCode>,
    mouse_pressed: HashSet<MouseButton>,
    mouse_held: HashSet<MouseButton>,
    /// Cursor position in logical pixels (top-left origin), if inside the window.
    pub cursor: Option<Vec2>,
    /// Cursor movement accumulated this frame, in logical pixels.
    pub cursor_delta: Vec2,
    /// Scroll wheel delta accumulated this frame, in lines (`y` is vertical).
    pub scroll_delta: Vec2,
}

impl Input {
    /// A key that went down this frame.
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
    /// A key that is currently held down.
    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }
    /// A mouse button that went down this frame.
    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_pressed.contains(&button)
    }
    /// A mouse button that is currently held down.
    pub fn mouse_held(&self, button: MouseButton) -> bool {
        self.mouse_held.contains(&button)
    }
    /// All keys currently held down.
    pub fn held_keys(&self) -> impl Iterator<Item = KeyCode> + '_ {
        self.keys_held.iter().copied()
    }

    /// Reset the per-frame input deltas. Called at the *end* of each frame, after the
    /// camera controls and the app handler have consumed the events that arrived since
    /// the previous frame.
    fn finish_frame(&mut self) {
        self.keys_pressed.clear();
        self.mouse_pressed.clear();
        self.cursor_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    fn key_event(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.keys_pressed.insert(code);
                self.keys_held.insert(code);
            }
            ElementState::Released => {
                self.keys_held.remove(&code);
            }
        }
    }

    fn mouse_event(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.mouse_pressed.insert(button);
                self.mouse_held.insert(button);
            }
            ElementState::Released => {
                self.mouse_held.remove(&button);
            }
        }
    }
}

/// One logged mesh instance, used for CPU picking against the world-space triangle soup.
pub(crate) struct MeshInstance {
    /// First vertex of this instance in the triangle soup.
    start: usize,
    /// Number of vertices (always a multiple of 3).
    count: usize,
}

/// Per-frame context handed to the app's frame handler.
///
/// Log buffers are cleared automatically at the start of every frame; camera, input and
/// gizmo drag state persist across frames.
pub struct Frame {
    dt: f32,
    camera: OrbitCamera,
    input: Input,
    /// Viewport size in logical pixels (matches [`Input::cursor`] coordinates).
    viewport: Vec2,
    /// World-space triangle soup: `(position, normal, color)` per vertex, 3 per triangle.
    mesh_verts: Vec<render::MeshVertex>,
    /// World-space line soup: `(position, color)` per vertex, 2 per segment.
    line_verts: Vec<render::LineVertex>,
    /// One entry per `log_mesh` / `log_skinned_mesh` / `log_solid_box` call, for picking.
    mesh_instances: Vec<MeshInstance>,
    /// Scratch for `log_skinned_mesh`: the skinned positions and normals of the
    /// last skinned mesh, reused across calls so skinning never allocates.
    skinned_positions: Vec<Vec3>,
    skinned_normals: Vec<Vec3>,
    gizmo_state: gizmo::GizmoState,
    /// Set by the gizmo while hovered/dragging; suppresses camera controls next frame.
    gizmo_captured: bool,
    prev_gizmo_captured: bool,
    /// Set by `Frame::capture_camera`; suppresses camera controls next frame.
    app_captured: bool,
    prev_app_captured: bool,
    frame_index: u64,
}

impl Frame {
    fn new() -> Self {
        Self {
            dt: 1.0 / 60.0,
            camera: OrbitCamera::default(),
            input: Input::default(),
            viewport: Vec2::ZERO,
            mesh_verts: Vec::new(),
            line_verts: Vec::new(),
            mesh_instances: Vec::new(),
            skinned_positions: Vec::new(),
            skinned_normals: Vec::new(),
            gizmo_state: gizmo::GizmoState::default(),
            gizmo_captured: false,
            prev_gizmo_captured: false,
            app_captured: false,
            prev_app_captured: false,
            frame_index: 0,
        }
    }

    fn begin_frame(&mut self, dt: f32) {
        self.dt = dt;
        self.mesh_verts.clear();
        self.line_verts.clear();
        self.mesh_instances.clear();
        self.prev_gizmo_captured = self.gizmo_captured;
        self.gizmo_captured = false;
        self.prev_app_captured = self.app_captured;
        self.app_captured = false;
        self.gizmo_state.prune(self.frame_index);
        self.frame_index += 1;
    }

    /// Seconds elapsed since the previous frame.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// The orbit camera. Apps may reposition it freely; the built-in mouse controls
    /// (left-drag orbit, middle/right-drag pan, scroll zoom) keep working afterwards.
    pub fn camera(&mut self) -> &mut OrbitCamera {
        &mut self.camera
    }

    /// Current per-frame input snapshot.
    pub fn input(&self) -> &Input {
        &self.input
    }

    /// Log a mesh with a world transform and a flat color (drawn double-sided, lambert).
    pub fn log_mesh(&mut self, mesh: &Mesh, transform: Mat4, color: Color) {
        push_mesh(&mut self.mesh_verts, &mut self.mesh_instances, &mesh.positions, &mesh.normals, &mesh.indices, transform, color);
    }

    /// Log a skinned mesh: CPU linear-blend skinning with `bone_transforms` (world-space
    /// joint matrices), then drawn like [`Frame::log_mesh`]. Per vertex,
    /// `P' = sum_k w_k * bone_transforms[j_k] * inverse_bind[j_k] * P`.
    ///
    /// The skinned vertices go through the frame's scratch buffers: this call
    /// performs no heap allocation after warmup.
    pub fn log_skinned_mesh(
        &mut self,
        mesh: &Mesh,
        skin: &Skin,
        bone_transforms: &[Mat4],
        transform: Mat4,
        color: Color,
    ) {
        self.skinned_positions.clear();
        self.skinned_positions.extend_from_slice(&mesh.positions);
        self.skinned_normals.clear();
        self.skinned_normals.extend_from_slice(&mesh.normals);
        skin_vertices(&mut self.skinned_positions, &mut self.skinned_normals, skin, bone_transforms);
        // Transform each vertex once, not once per triangle corner (~6x fewer).
        let normal_mat = Mat3::from_mat4(transform.inverse().transpose());
        for position in &mut self.skinned_positions {
            *position = transform.transform_point3(*position);
        }
        for normal in &mut self.skinned_normals {
            *normal = normal_mat.mul_vec3(*normal).normalize_or_zero();
        }
        let start = self.mesh_verts.len();
        let vertex = |i: usize| render::MeshVertex {
            position: self.skinned_positions[i].to_array(),
            normal: self.skinned_normals.get(i).copied().unwrap_or(Vec3::Y).to_array(),
            color,
        };
        self.mesh_verts.reserve(if mesh.indices.is_empty() { self.skinned_positions.len() } else { mesh.indices.len() });
        if mesh.indices.is_empty() {
            self.mesh_verts.extend((0..self.skinned_positions.len()).map(vertex));
        } else {
            self.mesh_verts.extend(mesh.indices.iter().map(|&i| vertex(i as usize)));
        }
        let count = self.mesh_verts.len() - start;
        if count > 0 {
            self.mesh_instances.push(MeshInstance { start, count });
        }
    }

    /// Log world-space line segments (drawn 1 px wide).
    pub fn log_lines(&mut self, segments: &[[Vec3; 2]], color: Color) {
        self.line_verts.reserve(segments.len() * 2);
        for [a, b] in segments {
            self.line_verts.push(render::LineVertex { position: a.to_array(), color });
            self.line_verts.push(render::LineVertex { position: b.to_array(), color });
        }
    }

    /// Log the 12 edges of an axis-aligned box.
    pub fn log_wire_box(&mut self, center: Vec3, half_extents: Vec3, color: Color) {
        self.log_lines(&mesh::wire_box_segments(center, half_extents), color);
    }

    /// Log a filled axis-aligned box (shaded like a mesh).
    pub fn log_solid_box(&mut self, center: Vec3, half_extents: Vec3, color: Color) {
        let cube = Mesh::cube(center, half_extents);
        self.log_mesh(&cube, Mat4::IDENTITY, color);
    }

    /// Log an arrow from `start` to `end`: a line plus a small solid cone head at `end`.
    pub fn log_arrow(&mut self, start: Vec3, end: Vec3, color: Color) {
        let (head_base, cone) = mesh::arrow_head(start, end);
        if cone.is_empty() {
            return; // Degenerate (zero-length) arrow.
        }
        self.log_lines(&[[start, head_base]], color);
        let base = self.mesh_verts.len();
        self.mesh_verts
            .extend(cone.into_iter().map(|mut v| {
                v.color = color;
                v
            }));
        self.mesh_instances
            .push(MeshInstance { start: base, count: self.mesh_verts.len() - base });
    }

    /// Log the ground plane: a y=0 grid with 1 m spacing covering roughly 50 x 50 m.
    pub fn log_ground(&mut self) {
        const HALF: i32 = 25;
        let mut segments = Vec::with_capacity((2 * HALF * 2 + 2) as usize * 2);
        for i in -HALF..=HALF {
            let f = i as f32;
            let h = HALF as f32;
            segments.push([Vec3::new(f, 0.0, -h), Vec3::new(f, 0.0, h)]);
            segments.push([Vec3::new(-h, 0.0, f), Vec3::new(h, 0.0, f)]);
        }
        let color = [0.32, 0.32, 0.36, 1.0];
        self.log_lines(&segments, color);
        // Highlight the world X (red-ish) and Z (blue-ish) axes, like upstream's viewer.
        let h = HALF as f32;
        self.log_lines(
            &[[Vec3::new(-h, 0.0, 0.0), Vec3::new(h, 0.0, 0.0)]],
            [0.55, 0.25, 0.25, 1.0],
        );
        self.log_lines(
            &[[Vec3::new(0.0, 0.0, -h), Vec3::new(0.0, 0.0, h)]],
            [0.25, 0.35, 0.6, 1.0],
        );
    }

    /// Ray from the camera through a pixel (logical pixels, top-left origin).
    pub fn screen_ray(&self, pixel: Vec2) -> Ray {
        camera::screen_ray(&self.camera, self.viewport, pixel)
    }

    /// Nearest ray-triangle hit over all meshes logged this frame.
    pub fn pick_mesh(&self, ray: &Ray) -> Option<Pick> {
        let mut best: Option<Pick> = None;
        for (mesh_index, inst) in self.mesh_instances.iter().enumerate() {
            let soup = &self.mesh_verts[inst.start..inst.start + inst.count];
            if let Some(mut pick) = picking::ray_triangles(ray, soup) {
                pick.mesh_index = mesh_index;
                if best.as_ref().is_none_or(|b| pick.distance < b.distance) {
                    best = Some(pick);
                }
            }
        }
        best
    }

    /// Minimal ImGuizmo-style translate gizmo, drawn with the viewer's own primitives.
    ///
    /// Shows three colored axis arrows (X red, Y green, Z blue) and three small plane
    /// quads (YZ, XZ, XY) at `transform`'s translation. Hovering highlights a handle;
    /// left-dragging it moves `transform` along the axis / in the plane. Returns `true`
    /// while this gizmo is being dragged. While hovered or dragged the built-in camera
    /// controls are suppressed. Axis locks belong in the app's own imgui panel.
    pub fn gizmo_translate(&mut self, id: &str, transform: &mut Mat4) -> bool {
        gizmo::gizmo_translate(self, id, transform)
    }

    /// Suppress the built-in camera controls (e.g. while the app runs its own drag
    /// interaction). Call every frame while the interaction is active.
    pub fn capture_camera(&mut self) {
        self.app_captured = true;
    }
}

/// CPU linear-blend skinning: rewrite `mesh`'s positions and normals in place.
fn skin_vertices(positions: &mut [Vec3], normals: &mut [Vec3], skin: &Skin, bone_transforms: &[Mat4]) {
    let identity = Mat4::IDENTITY;
    let skin_mat = |v: usize| -> Mat4 {
        let mut m = Mat4::ZERO;
        for k in 0..4 {
            let w = skin.weight_values[v][k];
            if w == 0.0 {
                continue;
            }
            let j = skin.weights[v][k] as usize;
            let bone = bone_transforms.get(j).copied().unwrap_or(identity);
            let inv_bind = skin.inverse_bind.get(j).copied().unwrap_or(identity);
            m += bone * inv_bind * w;
        }
        m
    };
    for (v, position) in positions.iter_mut().enumerate() {
        let m = skin_mat(v);
        *position = m.transform_point3(*position);
        if let Some(n) = normals.get_mut(v) {
            // Bone transforms are rigid, so the skin matrix's 3x3 part rotates
            // normals closely enough for debug lambert shading — much cheaper
            // than an inverse-transpose per vertex.
            *n = Mat3::from_mat4(m).mul_vec3(*n).normalize_or_zero();
        }
    }
}

/// Transform indexed (or plain, when `indices` is empty) vertex data into the
/// triangle soup — the shared half of [`Frame::log_mesh`].
fn push_mesh(
    soup: &mut Vec<render::MeshVertex>,
    instances: &mut Vec<MeshInstance>,
    positions: &[Vec3],
    normals: &[Vec3],
    indices: &[u32],
    transform: Mat4,
    color: Color,
) {
    let start = soup.len();
    soup.reserve(if indices.is_empty() { positions.len() } else { indices.len() });
    let normal_mat = transform.inverse().transpose();
    let mut push = |i: u32| {
        let p = positions[i as usize];
        let n = normals.get(i as usize).copied().unwrap_or(Vec3::Y);
        let wp = transform.transform_point3(p);
        let wn = normal_mat.transform_point3(n).normalize_or_zero();
        soup.push(render::MeshVertex {
            position: wp.to_array(),
            normal: wn.to_array(),
            color,
        });
    };
    if indices.is_empty() {
        for i in 0..positions.len() as u32 {
            push(i);
        }
    } else {
        for &i in indices {
            push(i);
        }
    }
    let count = soup.len() - start;
    if count > 0 {
        instances.push(MeshInstance { start, count });
    }
}

/// The viewer: window, wgpu device and imgui context, plus the frame loop.
pub struct Viewer {
    config: ViewerConfig,
}

/// A viewer without a window: runs frame handlers against the same [`Frame`]
/// logging API and captures offscreen renders (no imgui, no input events —
/// the gizmo and mouse camera controls do nothing headless). For tests and
/// command-line screenshot runs.
pub struct HeadlessViewer {
    gpu: headless::OffscreenGpu,
    frame: Frame,
}

impl HeadlessViewer {
    /// Create the offscreen renderer (a GPU is required, as for the windowed viewer).
    pub fn new(width: u32, height: u32) -> Result<HeadlessViewer> {
        let gpu = pollster::block_on(headless::OffscreenGpu::new(width, height))?;
        let mut frame = Frame::new();
        frame.viewport = Vec2::new(width as f32, height as f32);
        Ok(HeadlessViewer { gpu, frame })
    }

    /// Run one frame: clear the log buffers, call `handler` with the frame
    /// (there is no imgui `Ui` headless), draw the logged primitives into the
    /// offscreen texture, and read back the capture.
    pub fn capture(&mut self, dt: f32, handler: impl FnOnce(&mut Frame)) -> Result<RgbaImage> {
        self.frame.begin_frame(dt);
        handler(&mut self.frame);
        pollster::block_on(headless::capture(
            &mut self.gpu,
            &self.frame.camera,
            &self.frame.mesh_verts,
            &self.frame.line_verts,
        ))
    }
}

impl Viewer {
    /// Create a viewer. The actual window and GPU device are created lazily inside
    /// [`Viewer::run`], on the winit `resumed` event, as required by winit 0.30.
    pub fn new(config: ViewerConfig) -> Result<Viewer> {
        Ok(Viewer { config })
    }

    /// Run the frame loop. `handler` runs once per frame: it logs primitives into the
    /// [`Frame`] and builds the imgui UI. Never returns; exits the process when the
    /// window is closed.
    pub fn run(self, handler: impl FnMut(&mut Frame, &mut imgui::Ui) + 'static) -> ! {
        let event_loop = EventLoop::new().expect("failed to create winit event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App { config: self.config, handler, state: None, frame: Frame::new() };
        event_loop.run_app(&mut app).expect("winit event loop failed");
        std::process::exit(0);
    }
}

/// GPU + window + imgui state, created on `resumed`.
struct AppState {
    window: Arc<Window>,
    gpu: render::Gpu,
    imgui: imgui::Context,
    platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
    last_frame: Instant,
}

struct App<H: FnMut(&mut Frame, &mut imgui::Ui)> {
    config: ViewerConfig,
    handler: H,
    state: Option<AppState>,
    frame: Frame,
}

impl<H: FnMut(&mut Frame, &mut imgui::Ui)> App<H> {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width as f64,
                self.config.height as f64,
            ));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let gpu = pollster::block_on(render::Gpu::new(window.clone()))?;

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        {
            let io = imgui.io_mut();
            io.config_flags.insert(imgui::ConfigFlags::DOCKING_ENABLE);
        }
        let mut platform = imgui_winit_support::WinitPlatform::new(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );
        let imgui_renderer = imgui_wgpu::Renderer::new(
            &mut imgui,
            &gpu.device,
            &gpu.queue,
            imgui_wgpu::RendererConfig {
                texture_format: gpu.surface_format(),
                depth_format: None,
                ..Default::default()
            },
        );

        self.frame.viewport = Vec2::new(self.config.width as f32, self.config.height as f32);
        self.state = Some(AppState {
            window,
            gpu,
            imgui,
            platform,
            imgui_renderer,
            last_frame: Instant::now(),
        });
        Ok(())
    }

    fn redraw(&mut self) -> Result<()> {
        let state = self.state.as_mut().context("redraw before init")?;
        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32();
        state.last_frame = now;

        // 1. New frame: clear log buffers and per-frame input.
        self.frame.begin_frame(dt);

        // 2. Built-in camera controls, gated on imgui capture and gizmo/app capture.
        let camera_allowed = {
            let io = state.imgui.io();
            !io.want_capture_mouse
                && !self.frame.prev_gizmo_captured
                && !self.frame.prev_app_captured
        };
        if camera_allowed {
            apply_camera_controls(&mut self.frame);
        }

        // 3. imgui frame + app handler.
        state.platform.prepare_frame(state.imgui.io_mut(), &state.window)?;
        let ui = state.imgui.frame();
        (self.handler)(&mut self.frame, ui);
        state.platform.prepare_render(ui, &state.window);
        let draw_data = state.imgui.render();

        // 4. Draw 3D, then imgui on top.
        let AppState { gpu, imgui_renderer, .. } = state;
        gpu.render(
            &self.frame.camera,
            &self.frame.mesh_verts,
            &self.frame.line_verts,
            |device, queue, rpass| {
                imgui_renderer
                    .render(draw_data, queue, device, rpass)
                    .expect("imgui render failed");
            },
        )?;

        // Per-frame input deltas are consumed; reset them for the events that will
        // arrive before the next redraw.
        self.frame.input.finish_frame();
        state.window.request_redraw();
        Ok(())
    }
}

/// Newton ViewerGL camera defaults: left-drag orbits, middle/right-drag pans, scroll zooms.
fn apply_camera_controls(frame: &mut Frame) {
    let input = &frame.input;
    let cam = &mut frame.camera;
    if input.mouse_held(MouseButton::Left) {
        cam.orbit(-input.cursor_delta.x * 0.01, input.cursor_delta.y * 0.01);
    }
    if input.mouse_held(MouseButton::Middle) || input.mouse_held(MouseButton::Right) {
        cam.pan(input.cursor_delta);
    }
    let scroll = input.scroll_delta.y;
    if scroll != 0.0 {
        cam.zoom(scroll);
    }
}

impl<H: FnMut(&mut Frame, &mut imgui::Ui)> ApplicationHandler for App<H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            if let Err(e) = self.init(event_loop) {
                eprintln!("viewer init failed: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        if window_id != state.window.id() {
            return;
        }
        // Feed imgui first (WinitPlatform forwards keys/pointer to imgui, which sets
        // want_capture_* that the camera controls consult above).
        state.platform.handle_event(
            state.imgui.io_mut(),
            &state.window,
            &winit::event::Event::<()>::WindowEvent { window_id, event: event.clone() },
        );
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    let scale = state.window.scale_factor() as f32;
                    self.frame.viewport =
                        Vec2::new(size.width as f32 / scale, size.height as f32 / scale);
                    state.gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.frame.input.key_event(code, event.state);
                }
            }
            WindowEvent::MouseInput { state: s, button, .. } => {
                self.frame.input.mouse_event(button, s);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = state.window.scale_factor() as f32;
                let p = Vec2::new(position.x as f32 / scale, position.y as f32 / scale);
                let prev = self.frame.input.cursor;
                self.frame.input.cursor_delta += prev.map_or(Vec2::ZERO, |q| p - q);
                self.frame.input.cursor = Some(p);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(h, v) => Vec2::new(h, v),
                    MouseScrollDelta::PixelDelta(p) => {
                        Vec2::new(p.x as f32, p.y as f32) / 16.0 // ~ px per line
                    }
                };
                self.frame.input.scroll_delta += d;
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.redraw() {
                    eprintln!("frame failed: {e:#}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod bench_timing {
    #[test]
    #[ignore = "timing only"]
    fn time_log_skinned_mesh_soma() {
        use crate::rig::soma::SomaRig;
        use crate::viewer::Frame;
        use std::time::Instant;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let soma = SomaRig::load(root.join("checkpoints/motions")).expect("soma loads");
        let skin = soma.viewer_skin();
        let local = soma.skeleton.reference_local_transforms().to_vec();
        let bones = soma.evaluate(&local, soma.model_to_runtime).expect("bones");
        let mut frame = Frame::new();
        let start = Instant::now();
        let iterations = 200;
        for _ in 0..iterations {
            frame.begin_frame(1.0 / 60.0);
            frame.log_skinned_mesh(&soma.mesh.mesh, &skin, &bones, glam::Mat4::IDENTITY, [1.0; 4]);
        }
        let elapsed = start.elapsed();
        eprintln!("log_skinned_mesh x{iterations}: {:?} ({:?}/call)", elapsed, elapsed / iterations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn wire_box_geometry_counts() {
        let segments = mesh::wire_box_segments(Vec3::ZERO, Vec3::ONE);
        assert_eq!(segments.len(), 12, "a box has 12 edges");
        // Every endpoint is a box corner (each coordinate is +/- half extent).
        for [a, b] in &segments {
            for p in [a, b] {
                assert!(
                    p.to_array().iter().all(|c| (c.abs() - 1.0).abs() < 1e-6),
                    "not a corner: {p:?}"
                );
            }
        }
    }

    #[test]
    fn arrow_geometry_counts() {
        let start = Vec3::ZERO;
        let end = Vec3::new(0.0, 1.0, 0.0);
        let (head_base, cone) = mesh::arrow_head(start, end);
        // 8 side triangles + 8 base triangles, 3 verts each.
        assert_eq!(cone.len(), 16 * 3);
        // Head base sits on the arrow axis, short of `end`.
        assert!((head_base - end).length() > 0.0);
        assert!(head_base.x.abs() < 1e-6 && head_base.z.abs() < 1e-6);
    }

    #[test]
    fn lbs_two_bone_hand_computed() {
        // Bind pose: joint 0 at the origin, joint 1 at (1, 0, 0). Pose: joint 1 rotated
        // 90 degrees about Z around its own origin. A vertex at (1, 1, 0) bound to
        // joint 1 lands at the origin; its +X normal becomes +Y.
        let skin = Skin {
            joint_parents: vec![-1, 0],
            inverse_bind: vec![Mat4::IDENTITY, Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))],
            weights: vec![[1, 0, 0, 0]],
            weight_values: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        let bone_transforms = vec![
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)) * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
        ];
        let mut mesh = Mesh {
            positions: vec![Vec3::new(1.0, 1.0, 0.0)],
            normals: vec![Vec3::X],
            indices: vec![],
        };
        skin_vertices(&mut mesh.positions, &mut mesh.normals, &skin, &bone_transforms);
        assert!(mesh.positions[0].abs_diff_eq(Vec3::ZERO, 1e-5), "got {:?}", mesh.positions[0]);
        assert!(mesh.normals[0].abs_diff_eq(Vec3::Y, 1e-5), "got {:?}", mesh.normals[0]);

        // 50/50 blend between the identity joint 0 and joint 1: midpoint.
        let mut mesh = Mesh {
            positions: vec![Vec3::new(1.0, 1.0, 0.0)],
            normals: vec![Vec3::X],
            indices: vec![],
        };
        let skin = Skin {
            weights: vec![[0, 1, 0, 0]],
            weight_values: vec![[0.5, 0.5, 0.0, 0.0]],
            ..skin
        };
        skin_vertices(&mut mesh.positions, &mut mesh.normals, &skin, &bone_transforms);
        assert!(
            mesh.positions[0].abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), 1e-5),
            "got {:?}",
            mesh.positions[0]
        );
    }
}
