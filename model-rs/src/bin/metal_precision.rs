// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Compare a reduced-precision Metal run with the FP32 Metal run, stage by stage, on the
//! sandbox's `random` golden case. Development tool (`--features dev-tools`):
//!
//!     metal-precision [soma|unitree_g1] [float16|bfloat16|int8]
//!
//! Checkpoints: `$MOTIONBRICKS_CHECKPOINTS` or `checkpoints/MotionBricks`; golden fixtures:
//! `$MOTIONBRICKS_GOLDEN` or `sandbox/golden`.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use common::{median, positionals};
use motionbricks_rs::config::{Precision, checkpoint_dir, get_spec};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Buf, Recorder, wait};
use motionbricks_rs::npy::{Npy, load_npz};

/// Median and max |a - b| (non-finite differences count as 1e9).
fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut d: Vec<f64> = a.iter().zip(b).map(|(x, y)| { let v = (x - y).abs() as f64; if v.is_finite() { v } else { 1e9 } }).collect();
    let max = d.iter().copied().fold(0.0, f64::max);
    (median(&mut d), max)
}

fn main() -> Result<()> {
    let mut args = positionals()?.into_iter();
    let name = get_spec(&args.next().unwrap_or("soma".into()))?.name;
    let prec = args.next().unwrap_or("float16".into()).parse::<Precision>()?.name();
    let golden = std::env::var_os("MOTIONBRICKS_GOLDEN").map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../sandbox/golden")
    });
    let dir = checkpoint_dir(name);
    let a = MetalModel::from_checkpoints(name, &dir, 8)?;
    let b = MetalModel::from_checkpoints_with(name, &dir, 8, prec)?;
    let path = golden.join(name).join("random.npz");
    let g: HashMap<String, Npy> = load_npz(&path).with_context(|| format!("golden fixture {} (made by sandbox/setup.sh)", path.display()))?;
    let get = |k: &str| g.get(k).with_context(|| format!("golden fixture lacks {k}"));
    let inputs: HashMap<String, Npy> = g.iter().filter_map(|(k, v)| k.strip_prefix("in.").map(|s| (s.to_string(), v.clone()))).collect();
    // Teacher-forced root network on the golden root inputs: per-buffer NaN count and divergence.
    let n = 4usize;
    for m in [&a, &b] {
        let put = |buf: &Buf, k: &str| -> Result<()> { buf.write_f32(&get(k)?.to_f32()); Ok(()) };
        let mask = |buf: &Buf, k: &str| -> Result<()> { buf.write_f32(&get(k)?.to_bool().iter().map(|&x| x as i32 as f32).collect::<Vec<_>>()); Ok(()) };
        put(&m.b.gn, "root_in.global_root")?; put(&m.b.gn_t, "root_in.global_root")?;
        put(&m.b.ln, "root_in.local_root")?; put(&m.b.ln_t, "root_in.local_root")?;
        put(&m.b.pn, "root_in.poses")?; put(&m.b.pn_t, "root_in.poses")?;
        mask(&m.b.hg, "root_in.has_global")?; mask(&m.b.hl, "root_in.has_local")?; mask(&m.b.hp, "root_in.has_poses")?; mask(&m.b.allowed, "root_in.allowed")?;
        m.b.nt_in.write_i32(&get("root_in.num_tokens")?.to_i32());
        let mut rec = Recorder::new(&m.gpu);
        m.encode_root_a(&mut rec, &m.dims(n, true));
        wait(&rec.commit());
    }
    for (label, x, y) in [
        ("x1 (after shared tf)", a.b.x1.to_f32(n * 9 * 512), b.b.x1.to_f32(n * 9 * 512)),
        ("x2 (2nd tf input)", a.b.x2.to_f32(n * 25 * 512), b.b.x2.to_f32(n * 25 * 512)),
    ] {
        let nan = y.iter().filter(|v| !v.is_finite()).count();
        let (med, max) = stats(&x, &y.iter().map(|v| if v.is_finite() { *v } else { 1e9 }).collect::<Vec<_>>());
        println!("{label:<24} nonfinite {nan:<6} median |diff| {med:.2e}  max {max:.2e}");
    }
    let (oa, da, _) = a.run(&inputs)?;
    let (ob, db, _) = b.run(&inputs)?;
    let n = da.len();
    println!("durations fp32 {da:?} {prec} {db:?}");
    for (label, x, y) in [
        ("root path (normalized)", a.b.pr.to_f32(n * 64 * 5), b.b.pr.to_f32(n * 64 * 5)),
        ("pose logits", a.b.plog.to_f32(a.b.plog.len().min(n * 16 * 80)), b.b.plog.to_f32(b.b.plog.len().min(n * 16 * 80))),
        ("recon (normalized)", a.b.recon.to_f32(n * 64 * 365.min(a.b.recon.len() / (8 * 64))), b.b.recon.to_f32(n * 64 * 365.min(b.b.recon.len() / (8 * 64)))),
    ] {
        let (med, max) = stats(&x, &y);
        println!("{label:<24} median |diff| {med:.2e}  max {max:.2e}");
    }
    let per = oa.len() / n;
    let frames = 64;
    let dim = per / frames;
    let valid: Vec<usize> = (0..oa.len()).filter(|&k| (k / dim) % frames < 4 * da[k / per] as usize).collect();
    let (x, y): (Vec<f32>, Vec<f32>) = valid.iter().map(|&k| (oa[k], ob[k])).unzip();
    let (med, max) = stats(&x, &y);
    println!("output (valid frames)    median |diff| {med:.2e}  max {max:.2e}");
    let (med, max) = stats(&oa, &ob);
    println!("output (all frames)      median |diff| {med:.2e}  max {max:.2e}");
    Ok(())
}
