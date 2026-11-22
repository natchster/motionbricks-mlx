// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The graph emitters: checkpoint tensors in, a complete [`Graph`] out. The wiring
//! mirrors `model-rs/src/model.rs` (`forward_with`): G1 runs the rig pre/post-processing
//! around the core prediction; SOMA retargets the rotation slices around it. The INT32
//! mask inputs become bools with a `Cast` (nonzero is true, upstream's `!= 0`).

use anyhow::Result;

use crate::config::{G1_QPOS_DIM, Kind, ModelSpec};
use crate::graph::{DType, Dim, Graph};
use crate::inference::Core;
use crate::ops::{cast, concat, slice_axis};
use crate::representation::Rep;
use crate::weights::Loaded;

/// Emit the whole batched inference graph of `spec` (FP32; the FP16 conversion is a
/// separate pass over the result).
pub fn emit_graph(spec: &'static ModelSpec, loaded: &mut Loaded) -> Result<Graph> {
    let mut g = Graph::new();
    let rep = Rep::load(&mut g, spec, &loaded.mean, &loaded.std)?;
    let core = Core::load(&mut g, &mut loaded.weights, spec, rep)?;
    let mask = |g: &mut Graph, name: &str| {
        let x = g.input(name, DType::I32, &[Dim::Batch, 8.into()]);
        cast(g, "/pre", &x, DType::Bool)
    };
    match spec.kind {
        Kind::G1 => {
            let context = g.input("context_qpos", DType::F32, &[Dim::Batch, 4.into(), G1_QPOS_DIM.into()]);
            let target = g.input("target_qpos", DType::F32, &[Dim::Batch, 4.into(), G1_QPOS_DIM.into()]);
            let has_global = mask(&mut g, "has_global_root_values");
            let has_local = mask(&mut g, "has_local_root_values");
            let has_poses = mask(&mut g, "has_local_poses");
            let allowed = g.input("allowed_pred_num_tokens", DType::I32, &[Dim::Batch, 11.into()]);

            let rig = crate::rig_g1::G1Rig::load(&mut g, spec)?;
            let qpos = concat(&mut g, "/pre", &[&context, &target], 1);
            let (global_root, local_root, local_poses) = rig.emit_preprocess(&mut g, "/pre", &qpos)?;
            let pred = core.emit(&mut g, &global_root, &has_global, &local_root, &has_local, &local_poses, &has_poses, &allowed)?;
            let qpos_out = rig.emit_postprocess(&mut g, "/post", &pred.global_features)?;
            let global_dim = core.rep.layout.global_dim;
            let contacts = slice_axis(&mut g, "/post", &pred.global_features, 2, global_dim - 4, global_dim);
            g.output_as("pred_qpos", &qpos_out);
            g.output_as("pred_num_tokens", &pred.num_tokens);
            g.output_as("pred_foot_contacts", &contacts);
        }
        Kind::Soma => {
            let global_root = g.input("global_root_values", DType::F32, &[Dim::Batch, 8.into(), 5.into()]);
            let has_global = mask(&mut g, "has_global_root_values");
            let local_root = g.input("local_root_values", DType::F32, &[Dim::Batch, 8.into(), 4.into()]);
            let has_local = mask(&mut g, "has_local_root_values");
            let local_poses = g.input("local_poses", DType::F32, &[Dim::Batch, 8.into(), spec.local_poses_dim().into()]);
            let has_poses = mask(&mut g, "has_local_poses");
            let allowed = g.input("allowed_pred_num_tokens", DType::I32, &[Dim::Batch, 11.into()]);

            let retarget = crate::soma::SomaRetarget::load(&mut g, spec)?;
            let standard = retarget.emit_to_standard(&mut g, "/pre", &local_poses);
            let pred = core.emit(&mut g, &global_root, &has_global, &local_root, &has_local, &standard, &has_poses, &allowed)?;
            let motions = retarget.emit_to_capture(&mut g, "/post", &pred.global_features);
            g.output_as("pred_global_motions", &motions);
            g.output_as("pred_num_tokens", &pred.num_tokens);
        }
    }
    Ok(g)
}
