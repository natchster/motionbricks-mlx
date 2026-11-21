// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! List the GEMM launches of one generation (shape, conv/concat, plan) with counts (development tool).
use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use motionbricks_rs::metal::Recorder;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::npy::load_npz;

fn main() -> Result<()> {
    let name = std::env::args().nth(1).unwrap_or("soma".into());
    let prec = std::env::args().nth(2).unwrap_or("float32".into());
    let batch: usize = std::env::args().nth(3).map(|v| v.parse().unwrap()).unwrap_or(32);
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let folder = if name == "soma" { "soma" } else { "g1" };
    let m = MetalModel::from_checkpoints_with(&name, here.join("../checkpoints/MotionBricks").join(folder), 64, &prec)?;
    let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
    m.run(&req)?;
    *m.k.log.borrow_mut() = Some(Vec::new());
    let mut rec = Recorder::new(&m.gpu);
    m.encode(&mut rec, &m.dims(batch, false));
    let log = m.k.log.borrow_mut().take().unwrap();
    let mut counts: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for (p, q8) in &log {
        let key = format!(
            "M={:<5} N={:<5} K={:<5} taps={} up={} concat={} mask={} arelu={} act={} q8={} splits={}",
            p.m, p.n, p.k, p.conv_taps, p.conv_up, p.k1 > 0, p.has_mask, p.a_relu, p.act, q8, p.splits
        );
        let e = counts.entry(key).or_default();
        e.0 += 1;
        e.1 += 2.0 * p.m as f64 * p.n as f64 * p.k as f64 / 1e9;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
    for (k, (c, gf)) in v {
        println!("{c:>3}x  {gf:7.2} GFLOP  {k}");
    }
    Ok(())
}
