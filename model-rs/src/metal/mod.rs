// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! A minimal, model-specific Metal runtime: one device/queue, kernels compiled once from
//! `kernels.metal` + `model.metal` for one network element type, preallocated buffers, and a serial
//! compute encoder per generation. Compared with MLX there is no graph, no allocator, no
//! dependency tracking and no per-op scheduling: the whole call is a fixed list of dispatches
//! recorded into one command buffer.

pub mod model;
pub mod transformer;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

pub const KERNELS: &str = concat!(include_str!("kernels.metal"), "\n", include_str!("model.metal"));

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Element type of the network tensors (`T` in the kernels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

impl Precision {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "float32" | "fp32" | "int8" => Precision::F32,
            "float16" | "fp16" => Precision::F16,
            "bfloat16" | "bf16" => Precision::BF16,
            other => bail!("unsupported precision {other}"),
        })
    }
    pub fn size(self) -> usize {
        if self == Precision::F32 { 4 } else { 2 }
    }
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "float32",
            Precision::F16 => "float16",
            Precision::BF16 => "bfloat16",
        }
    }
    fn metal(self) -> &'static str {
        match self {
            Precision::F32 => "float",
            Precision::F16 => "half",
            Precision::BF16 => "bfloat",
        }
    }
    /// Round-to-nearest-even conversion, like MLX's `astype`.
    pub fn encode(self, data: &[f32]) -> Vec<u8> {
        match self {
            Precision::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Precision::F16 => data.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            Precision::BF16 => data.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        }
    }
    pub fn decode(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            Precision::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Precision::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
            Precision::BF16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        }
    }
}

/// A shared-storage (unified memory) buffer of `len` elements of `elem` bytes.
pub struct Buf {
    pub raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
    pub elem: usize,
    /// Element type for T-typed buffers (None: float32 / uint32 / raw).
    pub prec: Option<Precision>,
}

impl Buf {
    /// The contents as f32 (float32 buffers only).
    pub fn as_slice(&self) -> &[f32] {
        assert_eq!(self.elem, 4);
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr() as *const f32, self.len) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice(&self) -> &mut [f32] {
        assert_eq!(self.elem, 4);
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr() as *mut f32, self.len) }
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr() as *const u8, self.len * self.elem) }
    }
    pub fn bytes_mut(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
    /// First `n` elements converted to f32 (any precision).
    pub fn to_f32(&self, n: usize) -> Vec<f32> {
        match self.prec {
            Some(p) => p.decode(&self.bytes()[..n * self.elem]),
            None => self.as_slice()[..n].to_vec(),
        }
    }
    /// Write f32 values (converted to the buffer's precision) at the start.
    pub fn write_f32(&self, data: &[f32]) {
        let bytes = self.prec.unwrap_or(Precision::F32).encode(data);
        assert!(bytes.len() <= self.len * self.elem, "write past buffer end");
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.bytes_mut(), bytes.len()) };
    }
}

/// Device, queue, kernel libraries and a pipeline cache. Cheap to clone (shared handles).
#[derive(Clone)]
pub struct Gpu {
    pub device: Retained<Device>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub prec: Precision,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Rc<RefCell<HashMap<String, Pipeline>>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        Self::with_precision(Precision::F32)
    }

    pub fn with_precision(prec: Precision) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("no command queue")?;
        let source = format!("#include <metal_stdlib>\ntypedef {} T;\n{KERNELS}", prec.metal());
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|e| anyhow!("Metal compile error: {}", e.localizedDescription()))?;
        Ok(Gpu { device, queue, prec, library, pipelines: Rc::new(RefCell::new(HashMap::new())) })
    }

    /// The Metal device name, e.g. "Apple M4 Pro".
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// One of our kernels specialized with boolean function constants (index, value).
    pub fn specialized(&self, name: &str, consts: &[(usize, bool)]) -> Result<Pipeline> {
        let key = format!("{name}:{consts:?}");
        if let Some(p) = self.pipelines.borrow().get(&key) {
            return Ok(p.clone());
        }
        let values = MTLFunctionConstantValues::new();
        for &(index, value) in consts {
            let mut v = value;
            unsafe { values.setConstantValue_type_atIndex(NonNull::from(&mut v).cast(), MTLDataType::Bool, index as _) };
        }
        let f = self
            .library
            .newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
            .map_err(|e| anyhow!("kernel {name}: {}", e.localizedDescription()))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow!("pipeline {name}: {}", e.localizedDescription()))?;
        self.pipelines.borrow_mut().insert(key, p.clone());
        Ok(p)
    }

    pub fn pipeline(&self, name: &str) -> Result<Pipeline> {
        // Always through the constant-values path: kernels declaring function constants require it.
        self.specialized(name, &[])
    }

    fn raw(&self, bytes: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.device.newBufferWithLength_options(bytes.max(16) as _, MTLResourceOptions::StorageModeShared).expect("buffer allocation")
    }

    /// A float32 buffer of `len` elements.
    pub fn buffer(&self, len: usize) -> Buf {
        Buf { raw: self.raw(len * 4), len, elem: 4, prec: None }
    }

    /// A buffer of `len` network elements (`T`).
    pub fn buffer_t(&self, len: usize) -> Buf {
        Buf { raw: self.raw(len * self.prec.size()), len, elem: self.prec.size(), prec: Some(self.prec) }
    }

    pub fn upload(&self, data: &[f32]) -> Buf {
        let b = self.buffer(data.len());
        b.as_mut_slice().copy_from_slice(data);
        b
    }

    /// Upload f32 data converted to `T`.
    pub fn upload_t(&self, data: &[f32]) -> Buf {
        let b = self.buffer_t(data.len());
        b.write_f32(data);
        b
    }

    /// A raw 32-bit buffer (ints, packed INT8 weights).
    pub fn upload_u32(&self, data: &[u32]) -> Buf {
        let b = Buf { raw: self.raw(data.len() * 4), len: data.len(), elem: 4, prec: None };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.bytes_mut(), data.len() * 4) };
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
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Command buffers already committed by `flush` (same queue: they run in order).
    sent: Vec<CommandBuffer>,
    /// Dispatch counts at which `auto_flush` commits the work recorded so far.
    flush_at: Vec<usize>,
}

/// The command buffers of one recording, in commit order.
pub struct Submitted(Vec<CommandBuffer>);

pub enum Arg<'a> {
    /// Buffer and *element* offset.
    Buf(&'a Buf, usize),
    Bytes(&'a [u8]),
}

pub fn as_bytes<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

impl Recorder {
    pub fn new(gpu: &Gpu) -> Self {
        let cb = gpu.command_buffer();
        let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).expect("encoder");
        Recorder { cb, enc, dispatches: 0, queue: gpu.queue.clone(), sent: Vec::new(), flush_at: Vec::new() }
    }

    /// Commit early at these dispatch counts, so the GPU starts while the CPU is still encoding
    /// (one command buffer would leave the GPU idle for the whole encode).
    pub fn with_flushes(mut self, at: &[usize]) -> Self {
        self.flush_at = at.iter().rev().copied().collect();
        self
    }

    /// Commit what has been recorded and continue in a new command buffer.
    pub fn flush(&mut self) {
        self.enc.endEncoding();
        self.cb.commit();
        let cb = self.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).expect("encoder");
        self.enc = enc;
        self.sent.push(std::mem::replace(&mut self.cb, cb));
    }

    fn auto_flush(&mut self) {
        if self.flush_at.last() == Some(&self.dispatches) {
            self.flush_at.pop();
            self.flush();
        }
    }

    pub fn dispatch(&mut self, pso: &Pipeline, args: &[Arg], groups: (usize, usize, usize), threads: (usize, usize, usize)) {
        self.bind(pso, args);
        let g = MTLSize { width: groups.0 as _, height: groups.1 as _, depth: groups.2 as _ };
        let t = MTLSize { width: threads.0 as _, height: threads.1 as _, depth: threads.2 as _ };
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        self.dispatches += 1;
        self.auto_flush();
    }

    fn bind(&mut self, pso: &Pipeline, args: &[Arg]) {
        unsafe {
            self.enc.setComputePipelineState(pso);
            for (i, a) in args.iter().enumerate() {
                match a {
                    Arg::Buf(b, off) => self.enc.setBuffer_offset_atIndex(Some(&b.raw), (*off * b.elem) as _, i as _),
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
        self.auto_flush();
    }

    /// Dynamic threadgroup memory for the next dispatch (`[[threadgroup(index)]]`).
    pub fn threadgroup_memory(&mut self, bytes: usize, index: usize) {
        unsafe { self.enc.setThreadgroupMemoryLength_atIndex(bytes.div_ceil(16) * 16, index) };
    }

    /// End encoding and commit; returns the submitted command buffers (call `wait`).
    pub fn commit(mut self) -> Submitted {
        self.enc.endEncoding();
        self.cb.commit();
        self.sent.push(self.cb);
        Submitted(self.sent)
    }
}

/// Wait for the recording to finish; returns its GPU time in ms (first start to last end).
pub fn wait(s: &Submitted) -> f64 {
    let last = s.0.last().expect("command buffer");
    last.waitUntilCompleted();
    (last.GPUEndTime() - s.0[0].GPUStartTime()) * 1e3
}
