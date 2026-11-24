// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! wgpu device/surface management and the two debug-draw pipelines (triangles + lines).
//!
//! Everything the apps log during a frame is baked into two world-space vertex streams
//! on the CPU and uploaded with `queue.write_buffer`, so a frame costs two draw calls
//! plus the imgui overlay pass.

use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};
use glam::Vec3;
use winit::window::Window;

use crate::viewer::camera::OrbitCamera;

/// Clear color: Newton's viewer is a dark grey.
pub(crate) const CLEAR_COLOR: wgpu::Color = wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 };
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One vertex of the triangle stream: position + normal + color (40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// World-space normal.
    pub normal: [f32; 3],
    /// Linear RGBA color.
    pub color: [f32; 4],
}

/// One vertex of the line stream: position + color (28 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LineVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Linear RGBA color.
    pub color: [f32; 4],
}

pub(crate) fn as_bytes<T>(slice: &[T]) -> &[u8] {
    // Safe for repr(C) POD vertex structs: no padding in these layouts, any byte
    // pattern is valid, and the source slice outlives the call.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};
struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) col: vec4<f32>,
};

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
    var out: MeshOut;
    out.clip = u.view_proj * vec4<f32>(in.pos, 1.0);
    out.nrm = in.nrm;
    out.col = in.col;
    return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    // Lambert with a fixed key light plus ambient. Meshes are drawn double-sided
    // (upstream debug geometry has arbitrary winding), so light both faces.
    let n = normalize(in.nrm);
    let l = normalize(u.light_dir.xyz);
    let diff = abs(dot(n, l));
    let rgb = in.col.rgb * (0.3 + 0.7 * diff);
    return vec4<f32>(rgb, in.col.a);
}

struct LineIn {
    @location(0) pos: vec3<f32>,
    @location(1) col: vec4<f32>,
};
struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
};

@vertex
fn vs_line(in: LineIn) -> LineOut {
    var out: LineOut;
    out.clip = u.view_proj * vec4<f32>(in.pos, 1.0);
    out.col = in.col;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.col;
}
"#;

/// Owns the wgpu surface/device and the debug-draw pipelines.
pub(crate) struct Gpu {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    pipelines: Pipelines,
    mesh_vb: wgpu::Buffer,
    mesh_vb_len: usize,
    line_vb: wgpu::Buffer,
    line_vb_len: usize,
}

/// The debug-draw pipelines and their shared uniform (view-projection + light),
/// created for one color target format. Both the windowed [`Gpu`] and the
/// offscreen headless renderer use these.
pub(crate) struct Pipelines {
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub mesh_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
}

/// Create the pipelines and uniform bindings for `format` (swapchain format
/// for the windowed viewer, offscreen texture format for headless).
pub(crate) fn create_pipelines(device: &wgpu::Device, format: wgpu::TextureFormat) -> Pipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer uniforms"),
        size: 80, // mat4x4 + vec4
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer uniform layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewer uniform bind group"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer pipeline layout"),
        bind_group_layouts: &[Some(&uniform_layout)],
        immediate_size: 0,
    });

    let mesh_pipeline = make_pipeline(
        device,
        &pipeline_layout,
        &shader,
        "vs_mesh",
        "fs_mesh",
        format,
        wgpu::PrimitiveTopology::TriangleList,
        &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
        }],
    );
    let line_pipeline = make_pipeline(
        device,
        &pipeline_layout,
        &shader,
        "vs_line",
        "fs_line",
        format,
        wgpu::PrimitiveTopology::LineList,
        &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        }],
    );

    Pipelines { uniform_buffer, uniform_bind_group, mesh_pipeline, line_pipeline }
}

/// The view-projection + key light uniform block (80 bytes) both renderers upload.
pub(crate) fn uniforms(camera: &OrbitCamera, aspect: f32) -> [u8; 80] {
    let vp = camera.view_proj(aspect);
    let light: Vec3 = Vec3::new(0.5, 0.8, 0.6).normalize();
    let mut uniforms = Vec::with_capacity(80);
    for f in vp.to_cols_array() {
        uniforms.extend_from_slice(&f.to_le_bytes());
    }
    for f in [light.x, light.y, light.z, 0.0] {
        uniforms.extend_from_slice(&f.to_le_bytes());
    }
    uniforms.try_into().expect("mat4x4 + vec4 is 80 bytes")
}

impl Gpu {
    pub async fn new(window: Arc<Window>) -> Result<Gpu> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window).context("create wgpu surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable wgpu adapter")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("wgpu device request failed")?;

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow!("surface is incompatible with the adapter"))?;
        // Prefer an sRGB swapchain format: our shader outputs sRGB-encoded colors.
        if let Some(format) = surface
            .get_capabilities(&adapter)
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
        {
            config.format = format;
        }
        surface.configure(&device, &config);

        let depth_view = create_depth(&device, config.width, config.height);
        let pipelines = create_pipelines(&device, config.format);

        let mesh_vb = growable_buffer(&device, "mesh vb", 1 << 16);
        let line_vb = growable_buffer(&device, "line vb", 1 << 14);

        Ok(Gpu {
            surface,
            device,
            queue,
            config,
            depth_view,
            pipelines,
            mesh_vb,
            mesh_vb_len: 1 << 16,
            line_vb,
            line_vb_len: 1 << 14,
        })
    }

    /// Swapchain format, needed by the imgui renderer.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, self.config.width, self.config.height);
    }

    /// Draw one frame: upload the vertex streams, run the 3D pass, then hand the render
    /// pass to `draw_ui` for the imgui overlay.
    pub fn render(
        &mut self,
        camera: &OrbitCamera,
        mesh_verts: &[MeshVertex],
        line_verts: &[LineVertex],
        draw_ui: impl FnOnce(&wgpu::Device, &wgpu::Queue, &mut wgpu::RenderPass<'_>),
    ) -> Result<()> {
        // Uniforms: view-projection + key light direction.
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        self.queue.write_buffer(&self.pipelines.uniform_buffer, 0, &uniforms(camera, aspect));

        // Upload the two vertex streams (growing the buffers when needed).
        let mesh_bytes = as_bytes(mesh_verts);
        if mesh_bytes.len() > self.mesh_vb_len {
            self.mesh_vb_len = mesh_bytes.len().next_power_of_two();
            self.mesh_vb = growable_buffer(&self.device, "mesh vb", self.mesh_vb_len as u64);
        }
        if !mesh_bytes.is_empty() {
            self.queue.write_buffer(&self.mesh_vb, 0, mesh_bytes);
        }
        let line_bytes = as_bytes(line_verts);
        if line_bytes.len() > self.line_vb_len {
            self.line_vb_len = line_bytes.len().next_power_of_two();
            self.line_vb = growable_buffer(&self.device, "line vb", self.line_vb_len as u64);
        }
        if !line_bytes.is_empty() {
            self.queue.write_buffer(&self.line_vb, 0, line_bytes);
        }

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(frame) | Cst::Suboptimal(frame) => frame,
            // Lost/outdated swapchain (e.g. mid-resize): reconfigure and skip a frame.
            Cst::Lost | Cst::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            // Timeout / occluded / validation: skip this frame.
            _ => return Ok(()),
        };
        let view = frame.texture.create_view(&Default::default());

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewer") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
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
        // UI overlay: own pass, load the 3D result, no depth.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            draw_ui(&self.device, &self.queue, &mut rpass);
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }
}

pub(crate) fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}

pub(crate) fn growable_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    format: wgpu::TextureFormat,
    topology: wgpu::PrimitiveTopology,
    buffers: &[wgpu::VertexBufferLayout<'_>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vs),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Debug meshes are double-sided upstream; never cull.
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: Default::default(),
        cache: None,
    })
}
