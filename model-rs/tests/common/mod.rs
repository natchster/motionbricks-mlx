// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Shared test helpers, the Rust counterpart of `model/tests/mlx_helpers.py`.
//!
//! Checkpoints: `checkpoints/MotionBricks` (`model-rs/scripts/download_weights.sh`), or
//! `$MOTIONBRICKS_CHECKPOINTS`. Golden fixtures from the PyTorch reference: `sandbox/golden`
//! (`sandbox/setup.sh`), or `$MOTIONBRICKS_GOLDEN`. Tests that need either print `SKIPPED` and
//! pass without them (Rust's test harness has no skip state).
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use mlx_rs::Array;
use motionbricks_rs::npy::{Npy, load_npz};

pub const MODELS: [&str; 2] = ["unitree_g1", "soma"];
pub const CASES: [&str; 5] = ["trace", "random", "one_target", "restricted", "chained"];

/// One model on the GPU at a time (tests in a binary run on parallel threads).
static GPU: Mutex<()> = Mutex::new(());

pub fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn folder(name: &str) -> &'static str {
    if name == "soma" { "soma" } else { "g1" }
}

pub fn checkpoints_root() -> PathBuf {
    std::env::var_os("MOTIONBRICKS_CHECKPOINTS").map(PathBuf::from).unwrap_or_else(|| root().join("checkpoints/MotionBricks"))
}

pub fn golden_root() -> PathBuf {
    std::env::var_os("MOTIONBRICKS_GOLDEN").map(PathBuf::from).unwrap_or_else(|| root().join("sandbox/golden"))
}

/// The checkpoint folder of `name`, or `None` (after printing a skip notice) when it is missing.
pub fn checkpoints(name: &str) -> Option<PathBuf> {
    let dir = checkpoints_root().join(folder(name));
    let has = dir.is_dir() && std::fs::read_dir(&dir).map(|mut d| d.next().is_some()).unwrap_or(false);
    if !has {
        println!("SKIPPED: {name} checkpoints missing (run model-rs/scripts/download_weights.sh)");
        return None;
    }
    Some(dir)
}

/// Golden fixture of `name`/`case`, or `None` (after printing a skip notice) when it is missing.
pub fn golden(name: &str, case: &str) -> Option<HashMap<String, Npy>> {
    let path = golden_root().join(name).join(format!("{case}.npz"));
    if !path.is_file() {
        println!("SKIPPED: golden fixture {} missing (made by the sandbox: sandbox/setup.sh)", path.display());
        return None;
    }
    Some(load_npz(&path).expect("golden fixture loads"))
}

/// All five golden cases of `name`, or `None` if any is missing.
pub fn goldens(name: &str) -> Option<Vec<(&'static str, HashMap<String, Npy>)>> {
    CASES.iter().map(|case| golden(name, case).map(|g| (*case, g))).collect()
}

/// The `in.*` entries of a golden fixture, renamed to the ONNX input names.
pub fn golden_inputs(g: &HashMap<String, Npy>) -> HashMap<String, Npy> {
    g.iter().filter_map(|(k, v)| k.strip_prefix("in.").map(|n| (n.to_string(), v.clone()))).collect()
}

/// max |got - want| / max |want| over the kept elements (Python: `rel_err`).
pub fn rel_err(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let (mut num, mut den) = (0f64, 0f64);
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        if keep(k) {
            num = num.max((g as f64 - w as f64).abs());
            den = den.max((w as f64).abs());
        }
    }
    num / den.max(1e-12)
}

pub fn max_abs(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter().zip(want).enumerate().filter(|(k, _)| keep(*k)).map(|(_, (a, b))| (a - b).abs()).fold(0.0, f32::max)
}

pub fn argmax_rows(values: &[f32], width: usize) -> Vec<usize> {
    values.chunks(width).map(|r| r.iter().enumerate().fold(0, |b, (k, &x)| if x > r[b] { k } else { b })).collect()
}

pub fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Collects failures across cases so one failing case does not hide the others.
#[derive(Default)]
pub struct Failures(Vec<String>);

impl Failures {
    pub fn check(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        if !ok {
            self.0.push(msg());
        }
    }
    pub fn finish(self) {
        assert!(self.0.is_empty(), "{} failure(s):\n  {}", self.0.len(), self.0.join("\n  "));
    }
}

/// End-to-end tolerances of `test_parity.py::test_end_to_end`, on the frames inside each request's
/// chosen duration. `out`: `pred_qpos` (G1) or `pred_global_motions` (SOMA), flattened.
pub fn check_end_to_end(
    f: &mut Failures, name: &str, case: &str, label: &str, g: &HashMap<String, Npy>, out: &[f32], durations: &[i32], contacts: Option<&[f32]>,
) {
    let counts = g["pred_num_tokens"].to_i32();
    f.check(durations == counts.as_slice(), || format!("{case}/{label}: durations {durations:?} != {counts:?}"));
    let key = if name == "soma" { "pred_global_motions" } else { "pred_qpos" };
    let shape = &g[key].shape;
    let (frames, dim) = (shape[1] as usize, shape[2] as usize);
    let valid = |k: usize| (k / dim) % frames < 4 * counts[k / (dim * frames)] as usize;
    let want = g[key].to_f32();
    if name == "unitree_g1" {
        let q = max_abs(out, &want, valid);
        f.check(q < 5e-4, || format!("{case}/{label}: qpos max abs {q}"));
        let c = max_abs(contacts.unwrap(), &g["pred_foot_contacts"].to_f32(), |k| (k / 4) % frames < 4 * counts[k / (4 * frames)] as usize);
        f.check(c < 5e-3, || format!("{case}/{label}: contacts max abs {c}"));
    } else {
        // SOMA global features: root 0..5, rotations 92..272 (Gram-Schmidt, looser), the rest.
        let rot = 92..272;
        let root = max_abs(out, &want, |k| valid(k) && k % dim < 5);
        let rots = max_abs(out, &want, |k| valid(k) && rot.contains(&(k % dim)));
        let other = max_abs(out, &want, |k| valid(k) && k % dim >= 5 && !rot.contains(&(k % dim)));
        f.check(root < 1e-4, || format!("{case}/{label}: root max abs {root}"));
        f.check(other < 1e-3, || format!("{case}/{label}: features max abs {other}"));
        f.check(rots < 3e-3, || format!("{case}/{label}: rotations max abs {rots}"));
    }
}

/// Seeded uniform values in [-scale, scale) for weight-free tests.
pub struct Rand(u64);

impl Rand {
    pub fn new(seed: u64) -> Self {
        Rand(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// Random encoder-layer weights under `t.layers.{i}.*` (PyTorch names, MLX layout).
pub fn encoder_weights(rng: &mut Rand, layers: usize, d: usize, ffn: usize) -> HashMap<String, Vec<f32>> {
    let mut w = HashMap::new();
    for i in 0..layers {
        let p = |n: &str| format!("t.layers.{i}.{n}");
        for (name, n) in [
            ("self_attn.in_proj.weight", 3 * d * d),
            ("self_attn.in_proj.bias", 3 * d),
            ("self_attn.out_proj.weight", d * d),
            ("self_attn.out_proj.bias", d),
            ("linear1.weight", ffn * d),
            ("linear1.bias", ffn),
            ("linear2.weight", d * ffn),
            ("linear2.bias", d),
            ("norm1.weight", d),
            ("norm1.bias", d),
            ("norm2.weight", d),
            ("norm2.bias", d),
        ] {
            w.insert(p(name), rng.vec(n, 0.17));
        }
    }
    w
}

pub fn to_arrays(w: &HashMap<String, Vec<f32>>, d: usize, ffn: usize) -> HashMap<String, Array> {
    w.iter()
        .map(|(k, v)| {
            let shape: Vec<i32> = if k.ends_with("in_proj.weight") {
                vec![3 * d as i32, d as i32]
            } else if k.ends_with("linear1.weight") {
                vec![ffn as i32, d as i32]
            } else if k.ends_with("linear2.weight") {
                vec![d as i32, ffn as i32]
            } else if k.ends_with("out_proj.weight") {
                vec![d as i32, d as i32]
            } else {
                vec![v.len() as i32]
            };
            (k.clone(), Array::from_slice(v, &shape))
        })
        .collect()
}

/// PyTorch `TransformerEncoderLayer` (post-norm, ReLU) in float64; `pad[b * l + j]` = padded key.
pub fn reference_layer(x: &[f64], w: &HashMap<String, Vec<f32>>, layer: usize, b: usize, l: usize, d: usize, ffn: usize, heads: usize, pad: &[bool]) -> Vec<f64> {
    let g = |n: &str| w[&format!("t.layers.{layer}.{n}")].iter().map(|&v| v as f64).collect::<Vec<f64>>();
    let lin = |x: &[f64], wt: &[f64], bias: &[f64], n: usize, k: usize| -> Vec<f64> {
        let rows = x.len() / k;
        (0..rows * n).map(|o| (0..k).map(|c| x[(o / n) * k + c] * wt[(o % n) * k + c]).sum::<f64>() + bias[o % n]).collect()
    };
    let ln = |x: &[f64], wt: &[f64], bias: &[f64]| -> Vec<f64> {
        x.chunks(d)
            .flat_map(|r| {
                let mu = r.iter().sum::<f64>() / d as f64;
                let var = r.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / d as f64;
                r.iter().enumerate().map(move |(c, v)| (v - mu) / (var + 1e-5).sqrt() * wt[c] + bias[c]).collect::<Vec<_>>()
            })
            .collect()
    };
    let qkv = lin(x, &g("self_attn.in_proj.weight"), &g("self_attn.in_proj.bias"), 3 * d, d);
    let hd = d / heads;
    let mut att = vec![0f64; b * l * d];
    for bi in 0..b {
        for h in 0..heads {
            for i in 0..l {
                let q = |c: usize| qkv[(bi * l + i) * 3 * d + h * hd + c];
                let s: Vec<f64> = (0..l)
                    .map(|j| {
                        if pad[bi * l + j] {
                            f64::NEG_INFINITY
                        } else {
                            (0..hd).map(|c| q(c) * qkv[(bi * l + j) * 3 * d + d + h * hd + c]).sum::<f64>() / (hd as f64).sqrt()
                        }
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for c in 0..hd {
                    att[(bi * l + i) * d + h * hd + c] = (0..l).map(|j| e[j] / z * qkv[(bi * l + j) * 3 * d + 2 * d + h * hd + c]).sum();
                }
            }
        }
    }
    let o = lin(&att, &g("self_attn.out_proj.weight"), &g("self_attn.out_proj.bias"), d, d);
    let x1 = ln(&x.iter().zip(&o).map(|(a, c)| a + c).collect::<Vec<_>>(), &g("norm1.weight"), &g("norm1.bias"));
    let h1: Vec<f64> = lin(&x1, &g("linear1.weight"), &g("linear1.bias"), ffn, d).into_iter().map(|v| v.max(0.0)).collect();
    let f = lin(&h1, &g("linear2.weight"), &g("linear2.bias"), d, ffn);
    ln(&x1.iter().zip(&f).map(|(a, c)| a + c).collect::<Vec<_>>(), &g("norm2.weight"), &g("norm2.bias"))
}

