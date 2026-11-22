// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Checkpoint loading: the three SafeTensors files plus the normalization stats of a
//! Hugging Face model folder, verified against the shared
//! `model/motionbricks_mlx/assets/checkpoints.sha256` manifest (like upstream's
//! verify-then-export). Tensors keep their PyTorch names and layouts — the emit helpers
//! handle transposes at graph-build time.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::Digest;

use crate::config::ModelSpec;
use crate::npy::Npy;

const IGNORED_SUFFIXES: [&str; 3] = ["initted", "cluster_size", "embed_avg"];
const UNUSED_PREFIXES: [&str; 1] = ["vqvae.encoder."];

/// The checksum manifest, shared by the Python package, model-rs and the sandbox.
const MANIFEST: &str = include_str!("../../../model/motionbricks_mlx/assets/checkpoints.sha256");

/// A checkpoint tensor (always float32 on disk).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Named tensors with use tracking. The emitters read through [`Weights::get`], and
/// [`Weights::finish`] enforces strict loading (every tensor consumed), like upstream's
/// `load_state_dict(strict=True)` — the VQ-VAE encoder is the documented exception.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    used: HashSet<String>,
}

/// PyTorch name -> the name the emitters use (`_proj_input.0` -> `proj_input.0`, packed
/// MHA projection -> Linear), the same mapping as `model-rs/src/weights.rs::mlx_name`.
pub fn emitter_name(torch_name: &str) -> String {
    let name: Vec<&str> = torch_name.split('.').map(|p| p.trim_start_matches('_')).collect();
    name.join(".")
        .replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
        .replace("self_attn.in_proj_bias", "self_attn.in_proj.bias")
}

impl Weights {
    /// Tensor `name` (marks it used).
    pub fn get(&mut self, name: &str) -> Result<&Tensor> {
        self.used.insert(name.to_string());
        self.tensors.get(name).with_context(|| format!("missing weight {name}"))
    }

    /// Fails if any checkpoint tensor was not consumed by the emitters.
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
    pub weights: Weights,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// The manifest's (sha256, size) for a checkpoint path relative to `checkpoints/MotionBricks`.
fn manifest_entry(rel: &str) -> Result<(&str, u64)> {
    for line in MANIFEST.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(hash), Some(size), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            bail!("malformed checkpoints.sha256 line: {line:?}");
        };
        if path == rel {
            return Ok((hash, size.parse().with_context(|| format!("bad size in {line:?}"))?));
        }
    }
    bail!("{rel} is not in checkpoints.sha256")
}

/// Read one verified file of the model folder.
fn read_verified(dir: &Path, hf_folder: &str, rel: &str) -> Result<Vec<u8>> {
    let path = dir.join(rel);
    if !path.is_file() {
        bail!(
            "missing checkpoint {} (run `motionbricks-mlx download`, model-rs/scripts/download_weights.sh or sandbox/setup.sh)",
            path.display()
        );
    }
    let key = format!("{hf_folder}/{rel}");
    let (hash, size) = manifest_entry(&key)?;
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() as u64 != size || sha256_hex(&bytes) != hash {
        bail!("{} fails the checkpoints.sha256 checksum; re-download it", path.display());
    }
    Ok(bytes)
}

/// Read and verify the three SafeTensors files + stats of a Hugging Face model folder.
pub fn read_checkpoints(dir: impl AsRef<Path>, spec: &ModelSpec) -> Result<Loaded> {
    let dir = dir.as_ref();
    let mut tensors = HashMap::new();
    for file in spec.checkpoints {
        let bytes = read_verified(dir, spec.kind_name, file.path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parsing {}", dir.join(file.path).display()))?;
        for (name, view) in st.tensors() {
            let Some(rest) = name.strip_prefix(file.prefix) else { continue };
            if IGNORED_SUFFIXES.iter().any(|s| rest.ends_with(s)) {
                continue;
            }
            if view.dtype() != safetensors::Dtype::F32 {
                bail!("{}: {name} is {:?}, expected F32", file.path, view.dtype());
            }
            let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
            let data = bytemuck::cast_slice(view.data()).to_vec();
            tensors.insert(format!("{}.{}", file.module, emitter_name(rest)), Tensor { shape, data });
        }
    }
    let mean = Npy::parse(&read_verified(dir, spec.kind_name, spec.stats_files.0)?)?.to_f32();
    let std = Npy::parse(&read_verified(dir, spec.kind_name, spec.stats_files.1)?)?.to_f32();
    Ok(Loaded { weights: Weights { tensors, used: HashSet::new() }, mean, std })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_names_map_to_emitter_names() {
        assert_eq!(emitter_name("_proj_input.0.weight"), "proj_input.0.weight");
        assert_eq!(emitter_name("layers.3.self_attn.in_proj_weight"), "layers.3.self_attn.in_proj.weight");
        assert_eq!(emitter_name("layers.3.self_attn.in_proj_bias"), "layers.3.self_attn.in_proj.bias");
        assert_eq!(emitter_name("transformer._layers.0.linear1.bias"), "transformer.layers.0.linear1.bias");
    }

    #[test]
    fn manifest_covers_all_checkpoint_files() {
        for spec in [crate::config::G1, crate::config::SOMA] {
            for file in spec.checkpoints {
                manifest_entry(&format!("{}/{}", spec.kind_name, file.path)).unwrap();
            }
            manifest_entry(&format!("{}/{}", spec.kind_name, spec.stats_files.0)).unwrap();
            manifest_entry(&format!("{}/{}", spec.kind_name, spec.stats_files.1)).unwrap();
        }
    }

    #[test]
    fn loading_is_strict() {
        let t = |names: &[&str]| -> HashMap<String, Tensor> {
            names.iter().map(|n| (n.to_string(), Tensor { shape: vec![1], data: vec![1.0] })).collect()
        };
        let mut w = Weights { tensors: t(&["a.weight", "vqvae.encoder.x"]), used: HashSet::new() };
        w.get("a.weight").unwrap();
        assert!(w.get("missing").is_err());
        assert!(w.finish().is_ok(), "the unused VQ-VAE encoder is allowed");
        let mut w = Weights { tensors: t(&["a.weight", "b.bias"]), used: HashSet::new() };
        w.get("a.weight").unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("b.bias"), "{err}");
    }
}
