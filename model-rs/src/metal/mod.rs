// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! A minimal, model-specific Metal runtime: one device/queue, kernels compiled once from
//! `kernels.metal`, preallocated buffers, and a serial compute encoder per generation.
//! Compared with MLX there is no graph, no allocator, no dependency tracking and no per-op
//! scheduling: the whole call is a fixed list of dispatches recorded into one command buffer.

pub mod model;
pub mod transformer;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

pub const KERNELS: &str = concat!(include_str!("kernels.metal"), "\n", include_str!("model.metal"));
/// MLX's precompiled kernel library from the mlx-sys build (see .cargo/config.toml).
pub const MLX_METALLIB: &str = concat!(env!("MLX_RS_METAL_PATH"), "/mlx.metallib");

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// A shared-storage (unified memory) float buffer.
pub struct Buf {
    pub raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize, // in f32 elements
}

impl Buf {
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr() as *const f32, self.len) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr() as *mut f32, self.len) }
    }
    pub fn bytes_mut(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
}

pub struct Gpu {
    pub device: Retained<Device>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    mlx_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RefCell<HashMap<String, Pipeline>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("no command queue")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNELS), None)
            .map_err(|e| anyhow!("Metal compile error: {}", e.localizedDescription()))?;
        let mlx_library = device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&NSString::from_str(MLX_METALLIB)))
            .map_err(|e| anyhow!("loading {MLX_METALLIB}: {}", e.localizedDescription()))?;
        Ok(Gpu { device, queue, library, mlx_library, pipelines: RefCell::new(HashMap::new()) })
    }

    /// A kernel from MLX's metallib specialized with boolean function constants (index, value).
    pub fn mlx_pipeline(&self, name: &str, consts: &[(usize, bool)]) -> Result<Pipeline> {
        let key = format!("mlx:{name}:{consts:?}");
        if let Some(p) = self.pipelines.borrow().get(&key) {
            return Ok(p.clone());
        }
        let values = MTLFunctionConstantValues::new();
        for &(index, value) in consts {
            let mut v = value;
            unsafe { values.setConstantValue_type_atIndex(NonNull::from(&mut v).cast(), MTLDataType::Bool, index as _) };
        }
        let f = self
            .mlx_library
            .newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
            .map_err(|e| anyhow!("mlx kernel {name}: {}", e.localizedDescription()))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow!("pipeline {name}: {}", e.localizedDescription()))?;
        self.pipelines.borrow_mut().insert(key, p.clone());
        Ok(p)
    }

    pub fn pipeline(&self, name: &str) -> Result<Pipeline> {
        if let Some(p) = self.pipelines.borrow().get(name) {
            return Ok(p.clone());
        }
        let f = self.library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel {name}"))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow!("pipeline {name}: {}", e.localizedDescription()))?;
        self.pipelines.borrow_mut().insert(name.to_string(), p.clone());
        Ok(p)
    }

    pub fn buffer(&self, len: usize) -> Buf {
        let raw = self
            .device
            .newBufferWithLength_options((len.max(1) * 4) as _, MTLResourceOptions::StorageModeShared)
            .expect("buffer allocation");
        Buf { raw, len }
    }

    pub fn upload(&self, data: &[f32]) -> Buf {
        let b = self.buffer(data.len());
        b.as_mut_slice().copy_from_slice(data);
        b
    }

    /// A byte buffer (e.g. bool masks), length in bytes rounded up to f32 elements.
    pub fn upload_bytes(&self, data: &[u8]) -> Buf {
        let b = self.buffer(data.len().div_ceil(4));
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.bytes_mut(), data.len()) };
        b
    }

    pub fn command_buffer(&self) -> CommandBuffer {
        self.queue.commandBuffer().expect("command buffer")
    }
}

/// Records dispatches into one serial compute encoder (implicit barrier between dispatches,
/// which is what a strictly sequential network needs).
pub struct Recorder {
    pub cb: CommandBuffer,
    enc: Encoder,
    pub dispatches: usize,
}

pub enum Arg<'a> {
    Buf(&'a Buf, usize), // buffer, byte offset
    Bytes(&'a [u8]),
}

pub fn as_bytes<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

impl Recorder {
    pub fn new(gpu: &Gpu) -> Self {
        let cb = gpu.command_buffer();
        let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).expect("encoder");
        Recorder { cb, enc, dispatches: 0 }
    }

    pub fn dispatch(&mut self, pso: &Pipeline, args: &[Arg], groups: (usize, usize, usize), threads: (usize, usize, usize)) {
        self.bind(pso, args);
        let g = MTLSize { width: groups.0 as _, height: groups.1 as _, depth: groups.2 as _ };
        let t = MTLSize { width: threads.0 as _, height: threads.1 as _, depth: threads.2 as _ };
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        self.dispatches += 1;
    }

    fn bind(&mut self, pso: &Pipeline, args: &[Arg]) {
        unsafe {
            self.enc.setComputePipelineState(pso);
            for (i, a) in args.iter().enumerate() {
                match a {
                    Arg::Buf(b, off) => self.enc.setBuffer_offset_atIndex(Some(&b.raw), *off as _, i as _),
                    Arg::Bytes(bytes) => {
                        self.enc.setBytes_length_atIndex(NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(), bytes.len() as _, i as _)
                    }
                }
            }
        }
    }

    /// `dispatchThreads` (non-uniform grid), for kernels without bounds checks.
    pub fn dispatch_threads(&mut self, pso: &Pipeline, args: &[Arg], grid: (usize, usize, usize), threads: (usize, usize, usize)) {
        self.bind(pso, args);
        let g = MTLSize { width: grid.0 as _, height: grid.1 as _, depth: grid.2 as _ };
        let t = MTLSize { width: threads.0 as _, height: threads.1 as _, depth: threads.2 as _ };
        self.enc.dispatchThreads_threadsPerThreadgroup(g, t);
        self.dispatches += 1;
    }

    /// Dynamic threadgroup memory for the next dispatch (`[[threadgroup(index)]]`).
    pub fn threadgroup_memory(&mut self, bytes: usize, index: usize) {
        unsafe { self.enc.setThreadgroupMemoryLength_atIndex(bytes.div_ceil(16) * 16, index) };
    }

    /// End encoding and commit; returns the command buffer (call `wait`).
    pub fn commit(self) -> CommandBuffer {
        self.enc.endEncoding();
        self.cb.commit();
        self.cb
    }
}

pub fn wait(cb: &CommandBuffer) -> f64 {
    cb.waitUntilCompleted();
    (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 // ms of GPU time
}
