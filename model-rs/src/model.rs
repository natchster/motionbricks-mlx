// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Top-level models with the tensor interface of the released ONNX graphs
//! (ported from `motionbricks_mlx/model.py`).

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result};
use mlx_rs::error::Exception;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};

use crate::config::{
    CONTEXT_AND_TARGET, G1_QPOS_DIM, Kind, MAX_TOKENS, MIN_TOKENS, ModelSpec, NUM_TOKEN_CHOICES, Precision, get_spec, is_mask_input,
};
// The ONNX names moved to `config`; re-exported here for existing imports.
pub use crate::config::{G1_INPUTS, G1_OUTPUTS, SOMA_INPUTS, SOMA_OUTPUTS};
use crate::inference::Core;
use crate::networks::GLOBAL_ROOT_DIM;
use crate::npy::{Npy, load_npz_bytes};
use crate::ops::*;
use crate::representation::MotionRepresentation;
use crate::rig::{G1Rig, SomaRetarget};
use crate::weights::{Loaded, Weights, read_checkpoints, read_converted};

enum Rig {
    G1(G1Rig),
    Soma { retarget: SomaRetarget, local_rot: (i32, i32), global_rot: (i32, i32) },
}

pub struct MotionBricks {
    pub spec: ModelSpec,
    pub core: Core,
    rig: Rig,
    /// MaskGIT steps per generation (release configuration: 1; must be at least 1).
    pub num_inference_steps: usize,
    /// `(bits, group_size)` once [`MotionBricks::quantize`] ran.
    pub quantization: Option<(i32, i32)>,
}

/// `float32`/`fp32`, `float16`/`fp16`, `bfloat16`/`bf16` -> MLX dtype (INT8 is not a dtype: use
/// [`MotionBricks::load`] with [`Precision::Int8`]).
pub fn parse_dtype(name: &str) -> Result<Dtype> {
    let precision: Precision = name.parse()?;
    anyhow::ensure!(!precision.is_quantized(), "unsupported dtype {name}; load INT8 with MotionBricks::load");
    Ok(precision.compute_dtype())
}

fn invalid(msg: String) -> Exception {
    Exception::custom(msg)
}

impl MotionBricks {
    /// Build from the Hugging Face SafeTensors folder (`checkpoints/MotionBricks/{soma,g1}`).
    pub fn from_checkpoints(name: &str, dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let spec = get_spec(name)?;
        let loaded = read_checkpoints(dir, &spec)?;
        Self::build(spec, loaded, dtype)
    }

    /// Build from the checkpoint folder at a [`Precision`] (`"float32"`, `"float16"`,
    /// `"bfloat16"`, `"int8"`, or the enum). INT8 loads float32 weights and runs
    /// [`Self::quantize`]`(8, 64)`, like the Python port's `quantize(8)`.
    pub fn load<P>(name: &str, dir: impl AsRef<Path>, precision: P) -> Result<Self>
    where
        P: TryInto<Precision, Error: Into<anyhow::Error>>,
    {
        let precision = precision.try_into().map_err(Into::into)?;
        let mut model = Self::from_checkpoints(name, dir, precision.compute_dtype())?;
        if precision.is_quantized() {
            model.quantize(8, 64)?;
        }
        Ok(model)
    }

    /// Build from a single file written by the Python port's `motionbricks_mlx.convert`.
    pub fn from_converted(path: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let (name, loaded) = read_converted(path)?;
        Self::build(get_spec(&name)?, loaded, dtype)
    }

    fn build(spec: ModelSpec, loaded: Loaded, dtype: Dtype) -> Result<Self> {
        let rep = MotionRepresentation::new(spec.num_joints, &loaded.mean, &loaded.std)?;
        let rig_asset = load_npz_bytes(spec.rig_asset()).context("reading the embedded rig asset")?;
        let rig = match spec.kind {
            Kind::UnitreeG1 => Rig::G1(G1Rig::new(
                &rig_asset,
                &rep.global.range("global_rot_data"),
                &rep.global.range("global_root_pos"),
                &rep.global.range("foot_contacts"),
            )?),
            Kind::Soma => {
                let j = spec.num_joints;
                let ric = (j - 1) * 3;
                let g_rot = rep.global.range("global_rot_data");
                Rig::Soma {
                    retarget: SomaRetarget::new(&rig_asset, j)?,
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
        Ok(MotionBricks { spec, core, rig, num_inference_steps: 1, quantization: None })
    }

    /// Affine-quantize the transformer projections of the pose and root networks, like the Python
    /// port's `MotionBricks.quantize(bits, group_size)` (INT8: `quantize(8, 64)`).
    pub fn quantize(&mut self, bits: i32, group_size: i32) -> Result<()> {
        if let Some((b, g)) = self.quantization {
            anyhow::bail!("the model is already quantized ({b} bits, groups of {g})");
        }
        let core = &mut self.core;
        for t in [&mut core.pose.transformer, &mut core.root.shared, &mut core.root.root_token] {
            t.quantize(group_size, bits)?;
        }
        // Evaluate the quantized weights once, so compiled calls don't recompute them.
        let params: Vec<&Array> = [&core.pose.transformer, &core.root.shared, &core.root.root_token].iter().flat_map(|t| t.parameters()).collect();
        mlx_rs::transforms::eval(params)?;
        self.quantization = Some((bits, group_size));
        Ok(())
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        self.spec.input_names()
    }

    pub fn output_names(&self) -> &'static [&'static str] {
        self.spec.output_names()
    }

    /// Shape of input `name` for a batch of `b` requests.
    fn input_shape(&self, name: &str, b: i32) -> Vec<i32> {
        let slots = CONTEXT_AND_TARGET;
        match name {
            "context_qpos" | "target_qpos" => vec![b, slots / 2, G1_QPOS_DIM],
            "global_root_values" => vec![b, slots, GLOBAL_ROOT_DIM],
            "local_root_values" => vec![b, slots, 4],
            "local_poses" => vec![b, slots, self.spec.local_poses_dim()],
            "allowed_pred_num_tokens" => vec![b, NUM_TOKEN_CHOICES],
            _ => vec![b, slots], // the per-slot masks
        }
    }

    /// Input count, shapes (one batch size throughout) and `num_inference_steps >= 1`: checked on
    /// the host from array metadata before any graph is built, so malformed inputs are errors
    /// instead of out-of-bounds reads in the rig kernels. Under [`compiled`] this runs only when
    /// a new input shape is traced.
    fn check_inputs(&self, inputs: &[Array], num_tokens: Option<&Array>) -> R<()> {
        let names = self.input_names();
        if inputs.len() != names.len() {
            return Err(invalid(format!("{} expects {} inputs ({}), got {}", self.spec.name, names.len(), names.join(", "), inputs.len())));
        }
        if self.num_inference_steps == 0 {
            return Err(invalid("num_inference_steps must be at least 1".into()));
        }
        let b = inputs[0].shape().first().copied().ok_or_else(|| invalid(format!("{} must have a batch axis", names[0])))?;
        for (name, x) in names.iter().zip(inputs) {
            let want = self.input_shape(name, b);
            if x.shape() != want.as_slice() {
                return Err(invalid(format!("input {name} has shape {:?}; expected {want:?}", x.shape())));
            }
        }
        match num_tokens {
            Some(n) if n.shape() != [b] => Err(invalid(format!("num_tokens has shape {:?}; expected [{b}]", n.shape()))),
            _ => Ok(()),
        }
    }

    /// Positional inputs in `input_names` order (float32 / bool) -> outputs in `output_names` order.
    /// Builds the lazy graph (eager mode); evaluate with `mlx_rs::transforms::eval`.
    pub fn forward(&self, inputs: &[Array]) -> R<Vec<Array>> {
        self.forward_with(inputs, None)
    }

    /// Like [`Self::forward`]; `num_tokens` `(B,)` int32 forces the durations. Its values are not
    /// checked (they may still be on the GPU) and must be in 6..=16; [`Self::forward_with_durations`]
    /// checks durations given on the host.
    pub fn forward_with(&self, inputs: &[Array], num_tokens: Option<&Array>) -> R<Vec<Array>> {
        self.check_inputs(inputs, num_tokens)?;
        let steps = self.num_inference_steps;
        match &self.rig {
            Rig::G1(rig) => {
                let [context, target, has_g, has_l, has_p, allowed] = inputs else {
                    return Err(invalid("G1 expects 6 inputs".into()));
                };
                let (global_root, local_root, local_poses) = rig.preprocess(context, target)?;
                let (pred, num_tokens) =
                    self.core.predict_with(&global_root, has_g, &local_root, has_l, &local_poses, has_p, allowed, steps, num_tokens)?;
                let (qpos, contacts) = rig.postprocess(&pred)?;
                Ok(vec![qpos, num_tokens, contacts])
            }
            Rig::Soma { retarget, local_rot: (a, b), global_rot: (ga, gb) } => {
                let [global_root, has_g, local_root, has_l, local_poses, has_p, allowed] = inputs else {
                    return Err(invalid("SOMA expects 7 inputs".into()));
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
                    self.core.predict_with(global_root, has_g, local_root, has_l, &local_poses, has_p, allowed, steps, num_tokens)?;
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

    /// [`Self::forward_with`] with durations given on the host, checked to be one per request
    /// and in 6..=16 tokens.
    pub fn forward_with_durations(&self, inputs: &[Array], durations: &[i32]) -> R<Vec<Array>> {
        if let Some(d) = durations.iter().find(|d| !(MIN_TOKENS..=MAX_TOKENS).contains(*d)) {
            return Err(invalid(format!("duration {d} is outside {MIN_TOKENS}..={MAX_TOKENS} tokens")));
        }
        let n = i32::try_from(durations.len()).map_err(|_| invalid("too many durations".into()))?;
        self.forward_with(inputs, Some(&Array::from_slice(durations, &[n])))
    }

    /// ONNX-named NumPy-style arrays -> positional MLX inputs: masks are `!= 0` in their own dtype
    /// (like the Python port), the other inputs float32. Shapes are checked by [`Self::forward`].
    pub fn prepare(&self, inputs: &HashMap<String, Npy>) -> Result<Vec<Array>> {
        self.input_names()
            .iter()
            .map(|name| {
                let value = inputs.get(*name).ok_or_else(|| anyhow::anyhow!("missing input {name}"))?;
                value.check().with_context(|| format!("input {name}"))?;
                Ok(if is_mask_input(name) { value.bool_array() } else { value.f32_array() })
            })
            .collect()
    }
}

/// Convert the Hugging Face checkpoint folder of `name` into one file that both ports load
/// (`motionbricks_mlx.convert`). `dtype` is the stored precision.
pub fn convert(name: &str, dir: impl AsRef<Path>, output: impl AsRef<Path>, dtype: Dtype) -> Result<std::path::PathBuf> {
    let spec = get_spec(name)?;
    let loaded = read_checkpoints(dir, &spec)?;
    crate::weights::save_converted(&output, spec.name, &loaded, dtype)?;
    Ok(output.as_ref().to_path_buf())
}

/// `mx.compile` of the whole generation. The model is shared by the closure, so its weights are
/// captured as constants of the compiled graph (as in the Python port).
pub fn compiled(model: Rc<MotionBricks>) -> impl FnMut(&[Array]) -> R<Vec<Array>> {
    let mut f = compile(move |args: &[Array]| model.forward(args), None);
    move |args: &[Array]| f(args)
}
