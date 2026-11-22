// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
fn main() {
    // The vendored ONNX message definitions (proto/onnx.proto3, Apache-2.0, from the
    // onnx 1.23.2 release pinned by upstream's exporter). protoc comes from
    // protoc-bin-vendored, so no system protobuf is needed.
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    prost_build::Config::new()
        .protoc_executable(protoc)
        .compile_protos(&["proto/onnx.proto3"], &["proto"])
        .expect("compiling proto/onnx.proto3");
}
