// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Top-level models with the tensor interface of the released ONNX graphs
//! (ported from `motionbricks_mlx/model.py`).

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use anyhow::Result;
use mlx_rs::error::Exception;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};

use crate::config::{Kind, ModelSpec, get_spec};
use crate::inference::Core;
use crate::npy::{Npy, load_npz_bytes};
use crate::ops::*;
use crate::representation::MotionRepresentation;
use crate::rig::{G1Rig, SomaRetarget};
use crate::weights::{Loaded, Weights, read_checkpoints_with, read_converted};

pub const G1_INPUTS: [&str; 6] = [
    "context_qpos", "target_qpos", "has_global_root_values", "has_local_root_values", "has_local_poses",
    "allowed_pred_num_tokens",
];
pub const G1_OUTPUTS: [&str; 3] = ["pred_qpos", "pred_num_tokens", "pred_foot_contacts"];
pub const SOMA_INPUTS: [&str; 7] = [
    "global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values", "local_poses",
    "has_local_poses", "allowed_pred_num_tokens",
];
pub const SOMA_OUTPUTS: [&str; 2] = ["pred_global_motions", "pred_num_tokens"];
const MASK_INPUTS: [&str; 4] = ["has_global_root_values", "has_local_root_values", "has_local_poses", "allowed_pred_num_tokens"];

// Upstream rig assets (Apache-2.0, NVIDIA), shared with the Python package.
const G1_RIG: &[u8] = include_bytes!("../../model/motionbricks_mlx/assets/g1_model_rig.npz");
const SOMA_RIG: &[u8] = include_bytes!("../../model/motionbricks_mlx/assets/soma_model_rig.npz");

enum Rig {
    G1(G1Rig),
    Soma { retarget: SomaRetarget, local_rot: (i32, i32), global_rot: (i32, i32) },
}

pub struct MotionBricks {
    pub spec: ModelSpec,
    pub core: Core,
    rig: Rig,
    pub num_inference_steps: usize,
}

pub fn parse_dtype(name: &str) -> Result<Dtype> {
    Ok(match name {
        "float32" | "fp32" => Dtype::Float32,
        "float16" | "fp16" => Dtype::Float16,
        "bfloat16" | "bf16" => Dtype::Bfloat16,
        other => anyhow::bail!("unsupported dtype {other}"),
    })
}

impl MotionBricks {
    /// Build from the Hugging Face SafeTensors folder (`checkpoints/MotionBricks/{soma,g1}`).
    pub fn from_checkpoints(name: &str, dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        Self::from_checkpoints_with(name, dir, dtype, false)
    }

    /// Like [`Self::from_checkpoints`]; `contiguous_conv` stores Conv1d kernels contiguously.
    pub fn from_checkpoints_with(name: &str, dir: impl AsRef<Path>, dtype: Dtype, contiguous_conv: bool) -> Result<Self> {
        let spec = get_spec(name)?;
        let loaded = read_checkpoints_with(dir, &spec, contiguous_conv)?;
        Self::build(spec, loaded, dtype)
    }

    /// Build from a single file written by the Python port's `motionbricks_mlx.convert`.
    pub fn from_converted(path: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let (name, loaded) = read_converted(path)?;
        Self::build(get_spec(&name)?, loaded, dtype)
    }

    fn build(spec: ModelSpec, loaded: Loaded, dtype: Dtype) -> Result<Self> {
        let rep = MotionRepresentation::new(spec.num_joints, &loaded.mean, &loaded.std)?;
        let rig = match spec.kind {
            Kind::UnitreeG1 => Rig::G1(G1Rig::new(
                &load_npz_bytes(G1_RIG)?,
                &rep.global.range("global_rot_data"),
                &rep.global.range("global_root_pos"),
                &rep.global.range("foot_contacts"),
            )?),
            Kind::Soma => {
                let j = spec.num_joints;
                let ric = (j - 1) * 3;
                let g_rot = rep.global.range("global_rot_data");
                Rig::Soma {
                    retarget: SomaRetarget::new(&load_npz_bytes(SOMA_RIG)?, j)?,
                    local_rot: (ric, ric + j * 6),
                    global_rot: (g_rot[0], g_rot[g_rot.len() - 1] + 1),
                }
            }
        };
        let mut weights = Weights::new(loaded.tensors, dtype);
        let core = Core::load(&mut weights, spec.clone(), rep)?;
        weights.finish()?;
        // Materialize the dtype casts once, so they are not part of every generation.
        let params = weights.into_used();
        mlx_rs::transforms::eval(params.iter())?;
        Ok(MotionBricks { spec, core, rig, num_inference_steps: 1 })
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        match self.spec.kind {
            Kind::UnitreeG1 => &G1_INPUTS,
            Kind::Soma => &SOMA_INPUTS,
        }
    }

    pub fn output_names(&self) -> &'static [&'static str] {
        match self.spec.kind {
            Kind::UnitreeG1 => &G1_OUTPUTS,
            Kind::Soma => &SOMA_OUTPUTS,
        }
    }

    /// Positional inputs in `input_names` order (float32 / bool) -> outputs in `output_names` order.
    /// Builds the lazy graph (eager mode); evaluate with `mlx_rs::transforms::eval`.
    pub fn forward(&self, inputs: &[Array]) -> R<Vec<Array>> {
        let steps = self.num_inference_steps;
        match &self.rig {
            Rig::G1(rig) => {
                let [context, target, has_g, has_l, has_p, allowed] = inputs else {
                    return Err(Exception::custom("G1 expects 6 inputs"));
                };
                let (global_root, local_root, local_poses) = rig.preprocess(context, target)?;
                let (pred, num_tokens) =
                    self.core.predict(&global_root, has_g, &local_root, has_l, &local_poses, has_p, allowed, steps)?;
                let (qpos, contacts) = rig.postprocess(&pred)?;
                Ok(vec![qpos, num_tokens, contacts])
            }
            Rig::Soma { retarget, local_rot: (a, b), global_rot: (ga, gb) } => {
                let [global_root, has_g, local_root, has_l, local_poses, has_p, allowed] = inputs else {
                    return Err(Exception::custom("SOMA expects 7 inputs"));
                };
                let local_poses = cat(
                    &[
                        &slice(local_poses, -1, 0, *a)?,
                        &retarget.capture_to_standard(&slice(local_poses, -1, *a, *b)?)?,
                        &slice(local_poses, -1, *b, END)?,
                    ],
                    -1,
                )?;
                let (pred, num_tokens) =
                    self.core.predict(global_root, has_g, local_root, has_l, &local_poses, has_p, allowed, steps)?;
                let pred = cat(
                    &[
                        &slice(&pred, -1, 0, *ga)?,
                        &retarget.standard_to_capture(&slice(&pred, -1, *ga, *gb)?)?,
                        &slice(&pred, -1, *gb, END)?,
                    ],
                    -1,
                )?;
                Ok(vec![pred, num_tokens])
            }
        }
    }

    /// ONNX-named NumPy-style arrays -> positional MLX inputs (masks `!= 0`, floats float32).
    pub fn prepare(&self, inputs: &HashMap<String, Npy>) -> Result<Vec<Array>> {
        self.input_names()
            .iter()
            .map(|name| {
                let value = inputs.get(*name).ok_or_else(|| anyhow::anyhow!("missing input {name}"))?;
                Ok(if MASK_INPUTS.contains(name) { value.i32_array().ne(i(0))? } else { value.f32_array() })
            })
            .collect()
    }
}

/// `mx.compile` of the whole generation. The model is shared by the closure, so its weights are
/// captured as constants of the compiled graph (as in the Python port).
pub fn compiled(model: Rc<MotionBricks>) -> impl FnMut(&[Array]) -> R<Vec<Array>> {
    let mut f = compile(move |args: &[Array]| model.forward(args), None);
    move |args: &[Array]| f(args)
}
