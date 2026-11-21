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
mod weights;

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
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType, MTLFunctionConstantValues,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

pub const KERNELS: &str = concat!(include_str!("kernels.metal"), "\n", include_str!("model.metal"));

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Element type of the network tensors (`T` in the kernels). INT8 mode computes in [`Precision::F32`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

impl From<crate::config::Precision> for Precision {
    fn from(p: crate::config::Precision) -> Self {
        use crate::config::Precision as P;
        match p {
            P::Float32 | P::Int8 => Precision::F32,
            P::Float16 => Precision::F16,
            P::BFloat16 => Precision::BF16,
        }
    }
}

impl Precision {
    /// Element type for a precision name (`int8` computes in float32).
    pub fn parse(name: &str) -> Result<Self> {
        Ok(name.parse::<crate::config::Precision>()?.into())
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
            Precision::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Precision::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Precision::BF16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        }
    }
}

/// What a buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// float32 (inputs, masks, rig, outputs).
    F32,
    /// int32 (durations, tokens, rig indices).
    I32,
    /// Raw 32-bit words (packed INT8 weights).
    U32,
    /// The network element type `T`.
    T(Precision),
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::T(p) => p.size(),
            _ => 4,
        }
    }
}

/// A shared-storage (unified memory) buffer of `len` elements. The GPU writes it behind Rust's
/// back, so the host only copies in and out (no references into the contents are handed out);
/// read results after [`wait`]ing for the work that writes them.
pub struct Buf {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    /// Bytes per element (`kind.size()`, kept for the per-dispatch offset computation).
    elem: usize,
    kind: Elem,
}

impl Buf {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn kind(&self) -> Elem {
        self.kind
    }
    /// The Metal buffer.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
    fn ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
    /// Copy `bytes` to byte offset `at` (bounds-checked).
    fn write_bytes(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.len * self.elem, "write past buffer end ({} + {} > {} bytes)", at, bytes.len(), self.len * self.elem);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr().add(at), bytes.len()) };
    }
    /// Copy of the first `n` elements' bytes (bounds-checked).
    fn read_bytes(&self, n: usize) -> Vec<u8> {
        assert!(n <= self.len, "read past buffer end ({n} > {} elements)", self.len);
        let mut out = vec![0u8; n * self.elem];
        unsafe { std::ptr::copy_nonoverlapping(self.ptr(), out.as_mut_ptr(), out.len()) };
        out
    }
    /// Copy of the first `n` 4-byte elements (bounds-checked; one allocation, one copy).
    fn read_words<W: Copy>(&self, n: usize) -> Vec<W> {
        assert!(n <= self.len, "read past buffer end ({n} > {} elements)", self.len);
        assert_eq!(std::mem::size_of::<W>(), self.elem);
        let mut out = Vec::<W>::with_capacity(n);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr() as *const W, out.as_mut_ptr(), n);
            out.set_len(n);
        }
        out
    }
    /// First `n` elements converted to f32 (float32 or `T` buffers).
    pub fn to_f32(&self, n: usize) -> Vec<f32> {
        match self.kind {
            Elem::F32 => self.read_words(n),
            Elem::T(p) => p.decode(&self.read_bytes(n)),
            k => panic!("to_f32 on a {k:?} buffer"),
        }
    }
    /// First `n` elements of an int32 buffer.
    pub fn to_i32(&self, n: usize) -> Vec<i32> {
        assert_eq!(self.kind, Elem::I32, "to_i32 on a {:?} buffer", self.kind);
        self.read_words(n)
    }
    /// Write f32 values (converted to the buffer's element type) at the start.
    pub fn write_f32(&self, data: &[f32]) {
        self.write_f32_at(0, data);
    }
    /// Write f32 values (converted to the buffer's element type) from element `offset`.
    pub fn write_f32_at(&self, offset: usize, data: &[f32]) {
        let p = match self.kind {
            Elem::F32 => Precision::F32,
            Elem::T(p) => p,
            k => panic!("write_f32 on a {k:?} buffer"),
        };
        if p == Precision::F32 {
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
            self.write_bytes(offset * 4, bytes);
        } else {
            self.write_bytes(offset * self.elem, &p.encode(data));
        }
    }
    /// Write f32 values produced by `values` from element `offset` of a float32 buffer, without an
    /// intermediate vector. `count` is the number of values (checked against the buffer first).
    pub fn fill_f32(&self, offset: usize, count: usize, values: impl Iterator<Item = f32>) {
        assert_eq!(self.kind, Elem::F32, "fill_f32 on a {:?} buffer", self.kind);
        assert!(offset + count <= self.len, "write past buffer end ({offset} + {count} > {} elements)", self.len);
        let dst = unsafe { (self.ptr() as *mut f32).add(offset) };
        let mut written = 0;
        for v in values.take(count) {
            unsafe { dst.add(written).write(v) };
            written += 1;
        }
        assert_eq!(written, count, "fill_f32: iterator ended early");
    }
    /// Write int32 values at the start of an int32 buffer.
    pub fn write_i32(&self, data: &[i32]) {
        assert_eq!(self.kind, Elem::I32, "write_i32 on a {:?} buffer", self.kind);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        self.write_bytes(0, bytes);
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

    fn alloc(&self, len: usize, kind: Elem) -> Buf {
        let bytes = (len * kind.size()).max(16);
        let raw = self.device.newBufferWithLength_options(bytes as _, MTLResourceOptions::StorageModeShared).unwrap_or_else(|| panic!("allocating a {bytes}-byte Metal buffer"));
        Buf { raw, len, elem: kind.size(), kind }
    }

    /// A float32 buffer of `len` elements.
    pub fn buffer(&self, len: usize) -> Buf {
        self.alloc(len, Elem::F32)
    }

    /// An int32 buffer of `len` elements.
    pub fn buffer_i32(&self, len: usize) -> Buf {
        self.alloc(len, Elem::I32)
    }

    /// A buffer of `len` network elements (`T`).
    pub fn buffer_t(&self, len: usize) -> Buf {
        self.alloc(len, Elem::T(self.prec))
    }

    pub fn upload(&self, data: &[f32]) -> Buf {
        let b = self.buffer(data.len());
        b.write_f32(data);
        b
    }

    /// Upload f32 data converted to `T`.
    pub fn upload_t(&self, data: &[f32]) -> Buf {
        let b = self.buffer_t(data.len());
        b.write_f32(data);
        b
    }

    pub fn upload_i32(&self, data: &[i32]) -> Buf {
        let b = self.buffer_i32(data.len());
        b.write_i32(data);
        b
    }

    /// A raw 32-bit buffer (packed INT8 weights).
    pub fn upload_u32(&self, data: &[u32]) -> Buf {
        let b = self.alloc(data.len(), Elem::U32);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        b.write_bytes(0, bytes);
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

/// The command buffers of one recording, in commit order. [`wait`] for them before reading
/// results or writing inputs again.
#[must_use = "wait for the submitted work before reading its results or reusing its buffers"]
pub struct Submitted(Vec<CommandBuffer>);

impl Submitted {
    /// Wait for the recording to finish; its GPU time in ms (first start to last end), or the
    /// error of a command buffer that did not complete (GPU fault, timeout, out of memory).
    pub fn wait(&self) -> Result<f64> {
        let last = self.0.last().expect("command buffer");
        last.waitUntilCompleted();
        for (i, cb) in self.0.iter().enumerate() {
            let status = cb.status();
            if status != MTLCommandBufferStatus::Completed {
                let why = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_else(|| "no error reported".into());
                bail!("Metal command buffer {i} of {} did not complete ({status:?}): {why}", self.0.len());
            }
        }
        Ok((last.GPUEndTime() - self.0[0].GPUStartTime()) * 1e3)
    }
}

pub enum Arg<'a> {
    /// Buffer and *element* offset.
    Buf(&'a Buf, usize),
    Bytes(&'a [u8]),
}

/// Plain-old-data kernel parameters: `Copy`, no padding, no pointers, so every byte is initialized.
///
/// # Safety
/// Implement only for types whose every byte is initialized (no padding) and that hold no references.
pub(crate) unsafe trait Pod: Copy + 'static {}
unsafe impl Pod for i32 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for f32 {}
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

pub(crate) fn as_bytes<T: Pod>(v: &T) -> &[u8] {
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

    /// Dynamic threadgroup memory for the next dispatch (`[[threadgroup(index)]]`).
    pub fn threadgroup_memory(&mut self, bytes: usize, index: usize) {
        unsafe { self.enc.setThreadgroupMemoryLength_atIndex(bytes.div_ceil(16) * 16, index) };
    }

    /// End encoding and commit; returns the submitted command buffers (call `wait`).
    #[must_use = "wait for the submitted work before reading its results or reusing its buffers"]
    pub fn commit(mut self) -> Submitted {
        self.enc.endEncoding();
        self.cb.commit();
        self.sent.push(self.cb);
        Submitted(self.sent)
    }

    /// End encoding and drop the recording without running it (nothing was flushed yet).
    pub fn discard(self) {
        assert!(self.sent.is_empty(), "discard after a flush: part of the recording already ran");
        self.enc.endEncoding();
    }
}

/// Wait for the recording to finish; returns its GPU time in ms (first start to last end).
/// Panics if a command buffer failed; [`Submitted::wait`] returns the error instead.
pub fn wait(s: &Submitted) -> f64 {
    s.wait().unwrap_or_else(|e| panic!("{e:#}"))
}
