// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Weight loading, ported from `motionbricks_mlx/weights.py`.
//!
//! Two sources are supported:
//! * the Hugging Face SafeTensors folder (`checkpoints/MotionBricks/{soma,g1}`), with the same
//!   mechanical PyTorch -> MLX renaming and Conv1d transposes as the Python port;
//! * a single converted file written by `motionbricks_mlx.convert` (Python) or [`save_converted`]
//!   (Rust, `motionbricks_rs::convert`): the same format, readable by both ports.
//!
//! Loading is strict: every tensor must be consumed by the model, except the VQ-VAE encoder,
//! which inference never runs (the Python port keeps it for completeness only).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::{Array, Dtype};

use crate::config::ModelSpec;
use crate::npy::Npy;
use crate::ops::R;

const IGNORED_SUFFIXES: [&str; 3] = ["initted", "cluster_size", "embed_avg"];
const UNUSED_PREFIXES: [&str; 1] = ["vqvae.encoder."];
const FORMAT_VERSION: &str = "1";

/// PyTorch name -> MLX name (`_proj_input.0` -> `proj_input.0`, packed MHA projection -> Linear).
pub fn mlx_name(torch_name: &str) -> String {
    let name: Vec<&str> = torch_name.split('.').map(|p| p.trim_start_matches('_')).collect();
    name.join(".")
        .replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
        .replace("self_attn.in_proj_bias", "self_attn.in_proj.bias")
}

/// Named tensors with use tracking. Network modules read through [`Weights::get`], which
/// casts to the compute dtype (like `set_compute_dtype` in the Python port).
pub struct Weights {
    tensors: HashMap<String, Array>,
    used: HashSet<String>,
    handed_out: Vec<Array>,
    pub dtype: Dtype,
}

impl Weights {
    pub fn new(tensors: HashMap<String, Array>, dtype: Dtype) -> Self {
        Weights { tensors, used: HashSet::new(), handed_out: Vec::new(), dtype }
    }

    /// Tensor `name` in the compute dtype.
    pub fn get(&mut self, name: &str) -> R<Array> {
        let t = self.raw(name)?;
        let t = if t.dtype() == self.dtype { t } else { t.as_dtype(self.dtype)? };
        self.handed_out.push(t.clone());
        Ok(t)
    }

    /// Tensor `name` as stored (float32).
    pub fn raw(&mut self, name: &str) -> R<Array> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| mlx_rs::error::Exception::custom(format!("missing weight {name}")))?
            .clone();
        self.used.insert(name.to_string());
        Ok(t)
    }

    /// Every array given to the model (after dtype casts), to evaluate once at build time.
    pub fn into_used(self) -> Vec<Array> {
        self.handed_out
    }

    /// Fails if any checkpoint tensor was not consumed by the model.
    pub fn finish(&self) -> Result<()> {
        let mut unused: Vec<&String> = self
            .tensors
            .keys()
            .filter(|k| !self.used.contains(*k) && !UNUSED_PREFIXES.iter().any(|p| k.starts_with(p)))
            .collect();
        unused.sort();
        if !unused.is_empty() {
            bail!("{} checkpoint tensors were not used, e.g. {:?}", unused.len(), &unused[..unused.len().min(5)]);
        }
        Ok(())
    }
}

pub struct Loaded {
    pub tensors: HashMap<String, Array>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Read the three SafeTensors files + stats of a Hugging Face model folder.
///
/// Conv1d kernels stay the strided transposed views the Python port keeps. (Storing them
/// row-contiguously was tried: numerically identical, and the default layout was kept.)
pub fn read_checkpoints(dir: impl AsRef<Path>, spec: &ModelSpec) -> Result<Loaded> {
    let dir = dir.as_ref();
    let mut tensors = HashMap::new();
    for (module, rel, prefix) in spec.checkpoint_files {
        let path = dir.join(rel);
        if !path.is_file() {
            bail!("missing checkpoint {} (run model-rs/scripts/download_weights.sh or `motionbricks-mlx download`)", path.display());
        }
        let state = Array::load_safetensors(&path).with_context(|| format!("loading {}", path.display()))?;
        for (name, value) in state {
            let Some(rest) = name.strip_prefix(prefix) else { continue };
            if IGNORED_SUFFIXES.iter().any(|s| rest.ends_with(s)) {
                continue;
            }
            let key = format!("{module}.{}", mlx_name(rest));
            // Conv1d kernels: PyTorch (out, in, k) -> MLX channels-last (out, k, in).
            let value = if value.ndim() == 3 && key.ends_with(".weight") {
                value.transpose_axes(&[0, 2, 1])?
            } else {
                value
            };
            let value = value.as_dtype(Dtype::Float32)?;
            tensors.insert(key, value);
        }
    }
    let mean = Npy::load(dir.join(spec.stats_files.0))?.to_f32();
    let std = Npy::load(dir.join(spec.stats_files.1))?.to_f32();
    Ok(Loaded { tensors, mean, std })
}

/// Write a converted single-file model: MLX-layout tensors in `dtype`, the normalization stats,
/// and `{"model": name, "format": "1"}` metadata (`motionbricks_mlx.weights.save_converted`).
pub fn save_converted(path: impl AsRef<Path>, name: &str, loaded: &Loaded, dtype: Dtype) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut arrays: Vec<(String, Array)> =
        loaded.tensors.iter().map(|(k, v)| Ok((k.clone(), v.as_dtype(dtype)?))).collect::<Result<_>>()?;
    arrays.push(("stats.mean".into(), Array::from_slice(&loaded.mean, &[loaded.mean.len() as i32])));
    arrays.push(("stats.std".into(), Array::from_slice(&loaded.std, &[loaded.std.len() as i32])));
    let metadata = HashMap::from([("model".to_string(), name.to_string()), ("format".to_string(), FORMAT_VERSION.to_string())]);
    Array::save_safetensors(arrays, &metadata, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Read a converted single-file model (`motionbricks_mlx.convert`). Returns the model name too.
pub fn read_converted(path: impl AsRef<Path>) -> Result<(String, Loaded)> {
    let path = path.as_ref();
    let (mut tensors, metadata) = Array::load_safetensors_with_metadata(path)
        .with_context(|| format!("loading {}", path.display()))?;
    if metadata.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        bail!("{} is not a motionbricks-mlx v{FORMAT_VERSION} weight file", path.display());
    }
    let name = metadata.get("model").context("converted file has no model name")?.clone();
    let take = |t: &mut HashMap<String, Array>, k: &str| -> Result<Vec<f32>> {
        let a = t.remove(k).with_context(|| format!("converted file lacks {k}"))?.as_dtype(Dtype::Float32)?;
        Ok(crate::ops::to_vec_f32(&a)?)
    };
    let mean = take(&mut tensors, "stats.mean")?;
    let std = take(&mut tensors, "stats.std")?;
    let tensors = tensors
        .into_iter()
        .map(|(k, v)| Ok((k, v.as_dtype(Dtype::Float32)?)))
        .collect::<std::result::Result<HashMap<_, _>, mlx_rs::error::Exception>>()?;
    Ok((name, Loaded { tensors, mean, std }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_names_map_to_mlx() {
        assert_eq!(mlx_name("_proj_input.0.weight"), "proj_input.0.weight");
        assert_eq!(mlx_name("layers.3.self_attn.in_proj_weight"), "layers.3.self_attn.in_proj.weight");
        assert_eq!(mlx_name("layers.3.self_attn.in_proj_bias"), "layers.3.self_attn.in_proj.bias");
        assert_eq!(mlx_name("transformer._layers.0.linear1.bias"), "transformer.layers.0.linear1.bias");
    }

    #[test]
    fn loading_is_strict() {
        let t = |names: &[&str]| -> HashMap<String, Array> { names.iter().map(|n| (n.to_string(), Array::from_f32(1.0))).collect() };
        let mut w = Weights::new(t(&["a.weight", "vqvae.encoder.x"]), Dtype::Float16);
        assert_eq!(w.get("a.weight").unwrap().dtype(), Dtype::Float16);
        assert!(w.get("missing").is_err());
        assert!(w.finish().is_ok(), "the unused VQ-VAE encoder is allowed");
        let mut w = Weights::new(t(&["a.weight", "b.bias"]), Dtype::Float32);
        w.get("a.weight").unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("b.bias"), "{err}");
    }
}
