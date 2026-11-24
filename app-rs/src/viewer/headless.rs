// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Offscreen rendering for headless runs: the same debug-draw pipelines as the
//! windowed viewer ([`super::render`]), drawing into an `Rgba8UnormSrgb`
//! texture that is read back to the CPU instead of presented.
//!
//! There is no window, surface, or imgui overlay: a capture contains exactly
//! what the app logged into the [`super::Frame`] that frame.

use anyhow::{Context as _, Result};

use super::camera::OrbitCamera;
use super::render::{self, LineVertex, MeshVertex, Pipelines};

/// The offscreen target format: sRGB, like the swapchain formats the windowed
/// viewer prefers.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// One captured frame, RGBA8, row-major, top-left origin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// The RGBA color of pixel `(x, y)`; (0, 0) is the top-left corner.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * self.width + x) * 4) as usize;
        self.pixels[offset..offset + 4].try_into().expect("pixel offset is in range")
    }

    /// Pixels not equal to `color`.
    pub fn count_not(&self, color: [u8; 4]) -> usize {
        self.pixels.chunks_exact(4).filter(|pixel| *pixel != color).count()
    }

    /// Whether any pixel's RGB channels are within `tolerance` of `rgb`.
    pub fn contains_color(&self, rgb: [u8; 3], tolerance: u8) -> bool {
        self.pixels.chunks_exact(4).any(|pixel| {
            (0..3).all(|channel| pixel[channel].abs_diff(rgb[channel]) <= tolerance)
        })
    }

    /// Write the capture as a binary PPM (P6) image — a dependency-free
    /// format every image tool reads (the alpha channel is dropped).
    pub fn write_ppm(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut bytes = Vec::with_capacity((self.width * self.height * 3) as usize + 32);
        bytes.extend_from_slice(format!("P6\n{} {}\n255\n", self.width, self.height).as_bytes());
        for pixel in self.pixels.chunks_exact(4) {
            bytes.extend_from_slice(&pixel[..3]);
        }
        std::fs::write(path.as_ref(), bytes).with_context(|| format!("could not write {}", path.as_ref().display()))
    }
}

/// The expected clear color as captured bytes, for assertions: the target is
/// an sRGB texture, so the fixed-function pipeline encodes the linear clear
/// color to sRGB on write.
pub fn clear_color_rgba() -> [u8; 4] {
    let to_byte = |linear: f64| {
        let srgb = if linear > 0.003_130_8 { 1.055 * linear.powf(1.0 / 2.4) - 0.055 } else { 12.92 * linear };
        (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    [
        to_byte(render::CLEAR_COLOR.r),
        to_byte(render::CLEAR_COLOR.g),
        to_byte(render::CLEAR_COLOR.b),
        255,
    ]
}

/// The offscreen renderer: device, pipelines, color/depth targets, and the
/// growable vertex buffers (mirroring [`super::render::Gpu`] minus the surface).
pub(crate) struct OffscreenGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    pipelines: Pipelines,
    mesh_vb: wgpu::Buffer,
    mesh_vb_len: usize,
    line_vb: wgpu::Buffer,
    line_vb_len: usize,
}

impl OffscreenGpu {
    pub async fn new(width: u32, height: u32) -> Result<OffscreenGpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable wgpu adapter for headless rendering")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("wgpu device request failed")?;

        let width = width.max(1);
        let height = height.max(1);
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let depth_view = render::create_depth(&device, width, height);
        let pipelines = render::create_pipelines(&device, OFFSCREEN_FORMAT);
        let mesh_vb = render::growable_buffer(&device, "mesh vb", 1 << 16);
        let line_vb = render::growable_buffer(&device, "line vb", 1 << 14);

        Ok(OffscreenGpu {
            device,
            queue,
            width,
            height,
            color,
            color_view,
            depth_view,
            pipelines,
            mesh_vb,
            mesh_vb_len: 1 << 16,
            line_vb,
            line_vb_len: 1 << 14,
        })
    }

    /// Draw the logged vertex streams into the offscreen texture (the 3D pass
    /// of [`super::render::Gpu::render`], without the imgui overlay).
    fn render(&mut self, camera: &OrbitCamera, mesh_verts: &[MeshVertex], line_verts: &[LineVertex]) {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.queue.write_buffer(&self.pipelines.uniform_buffer, 0, &render::uniforms(camera, aspect));

        let mesh_bytes = render::as_bytes(mesh_verts);
        if mesh_bytes.len() > self.mesh_vb_len {
            self.mesh_vb_len = mesh_bytes.len().next_power_of_two();
            self.mesh_vb = render::growable_buffer(&self.device, "mesh vb", self.mesh_vb_len as u64);
        }
        if !mesh_bytes.is_empty() {
            self.queue.write_buffer(&self.mesh_vb, 0, mesh_bytes);
        }
        let line_bytes = render::as_bytes(line_verts);
        if line_bytes.len() > self.line_vb_len {
            self.line_vb_len = line_bytes.len().next_power_of_two();
            self.line_vb = render::growable_buffer(&self.device, "line vb", self.line_vb_len as u64);
        }
        if !line_bytes.is_empty() {
            self.queue.write_buffer(&self.line_vb, 0, line_bytes);
        }

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("headless") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(render::CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            rpass.set_bind_group(0, &self.pipelines.uniform_bind_group, &[]);
            if !mesh_verts.is_empty() {
                rpass.set_pipeline(&self.pipelines.mesh_pipeline);
                rpass.set_vertex_buffer(0, self.mesh_vb.slice(..));
                rpass.draw(0..mesh_verts.len() as u32, 0..1);
            }
            if !line_verts.is_empty() {
                rpass.set_pipeline(&self.pipelines.line_pipeline);
                rpass.set_vertex_buffer(0, self.line_vb.slice(..));
                rpass.draw(0..line_verts.len() as u32, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Copy the offscreen texture to the CPU (unpadding rows to the wgpu
    /// 256-byte copy alignment).
    async fn readback(&self) -> Result<RgbaImage> {
        let row_bytes = self.width * 4;
        let padded = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u32);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless readback"),
            size: (padded * self.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .context("headless readback poll failed")?;
        receiver.recv().context("headless readback map dropped")?.context("headless readback map failed")?;
        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((row_bytes * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(RgbaImage { width: self.width, height: self.height, pixels })
    }
}

/// Render one frame offscreen and read it back.
pub(crate) async fn capture(
    gpu: &mut OffscreenGpu,
    camera: &OrbitCamera,
    mesh_verts: &[MeshVertex],
    line_verts: &[LineVertex],
) -> Result<RgbaImage> {
    gpu.render(camera, mesh_verts, line_verts);
    gpu.readback().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::{Mesh, OrbitCamera};
    use glam::{Mat4, Vec3};

    /// A renderer for tests, or None when the machine has no GPU adapter
    /// (the tests skip then: they verify pixels, which needs real hardware).
    fn renderer() -> Option<OffscreenGpu> {
        match pollster::block_on(OffscreenGpu::new(320, 240)) {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                eprintln!("skipping headless render test (no GPU): {error:#}");
                None
            }
        }
    }

    /// The vertex streams of one frame containing the given meshes and lines,
    /// built through the same `Frame` logging path the apps use.
    fn log_frame(meshes: &[(Mat4, [f32; 4])], lines: &[([Vec3; 2], [f32; 4])]) -> (Vec<MeshVertex>, Vec<LineVertex>) {
        let mut frame = super::super::Frame::new();
        frame.begin_frame(1.0 / 60.0);
        let cube = Mesh::cube(Vec3::ZERO, Vec3::splat(0.5));
        for (transform, color) in meshes {
            frame.log_mesh(&cube, *transform, *color);
        }
        for (segment, color) in lines {
            frame.log_lines(&[*segment], *color);
        }
        (frame.mesh_verts, frame.line_verts)
    }

    fn capture_frame(gpu: &mut OffscreenGpu, meshes: &[(Mat4, [f32; 4])], lines: &[([Vec3; 2], [f32; 4])]) -> RgbaImage {
        let (mesh_verts, line_verts) = log_frame(meshes, lines);
        let camera = OrbitCamera::default();
        pollster::block_on(capture(gpu, &camera, &mesh_verts, &line_verts)).expect("capture")
    }

    #[test]
    fn empty_scene_clears_to_the_background() {
        let Some(mut gpu) = renderer() else { return };
        let image = capture_frame(&mut gpu, &[], &[]);
        assert_eq!((image.width, image.height), (320, 240));
        assert_eq!(image.pixels.len(), 320 * 240 * 4);
        assert_eq!(image.count_not(clear_color_rgba()), 0, "every pixel is the clear color");
    }

    #[test]
    fn a_cube_draws_its_color_over_the_center() {
        let Some(mut gpu) = renderer() else { return };
        // The default camera looks at (0, 1, 0) from 4 m away: a 1 m cube
        // centered there covers the image center.
        let image = capture_frame(&mut gpu, &[(Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), [1.0, 0.0, 0.0, 1.0])], &[]);
        let changed = image.count_not(clear_color_rgba());
        assert!(changed > 1_000, "the cube covers a visible area: {changed} pixels");
        let center = image.pixel(160, 120);
        assert!(center[0] > center[1] && center[0] > center[2], "the center is red-ish: {center:?}");
        assert!(center[0] > 60, "lambert shading keeps the red channel visible: {center:?}");
    }

    #[test]
    fn nearer_meshes_win_the_depth_test() {
        let Some(mut gpu) = renderer() else { return };
        // The default camera looks at the origin from +Z-ish: the red cube sits
        // between the camera and the blue one, fully covering it.
        let camera = OrbitCamera::default();
        let image = {
            let near = (Mat4::from_translation(camera.eye() * 0.5), [1.0, 0.0, 0.0, 1.0]);
            let far = (Mat4::from_scale(Vec3::splat(2.0)), [0.0, 0.0, 1.0, 1.0]);
            let (mesh_verts, line_verts) = log_frame(&[far, near], &[]);
            pollster::block_on(capture(&mut gpu, &camera, &mesh_verts, &line_verts)).expect("capture")
        };
        let center = image.pixel(160, 120);
        assert!(center[0] > center[2], "the nearer red cube wins: {center:?}");
        assert!(!image.contains_color([0, 0, 200], 60), "the hidden blue cube shows nowhere");
    }

    #[test]
    fn lines_draw_single_pixel_segments() {
        let Some(mut gpu) = renderer() else { return };
        let with_lines = capture_frame(&mut gpu, &[], &[([Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)], [0.0, 1.0, 0.0, 1.0])]);
        let changed = with_lines.count_not(clear_color_rgba());
        assert!(changed > 50, "a 4 m line crosses a visible stretch: {changed} pixels");
        assert!(with_lines.contains_color([0, 200, 0], 80), "the line is green-ish");
    }

    #[test]
    fn captures_are_deterministic() {
        let Some(mut gpu) = renderer() else { return };
        let scene = [(Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)), [0.85, 0.55, 0.2, 1.0])];
        let first = capture_frame(&mut gpu, &scene, &[]);
        let second = capture_frame(&mut gpu, &scene, &[]);
        assert_eq!(first, second, "same scene, same pixels");
    }

    #[test]
    fn ppm_writes_a_readable_header() {
        let dir = std::env::temp_dir().join(format!("motionbricks-ppm-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.ppm");
        let image = RgbaImage { width: 2, height: 1, pixels: vec![255, 0, 0, 255, 0, 255, 0, 255] };
        image.write_ppm(&path).expect("writes the file");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..11], b"P6\n2 1\n255\n");
        assert_eq!(&bytes[11..], &[255, 0, 0, 0, 255, 0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
