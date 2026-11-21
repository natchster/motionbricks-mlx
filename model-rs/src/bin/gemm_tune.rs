// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Replay every distinct GEMM launch of the model (real shapes and loader/epilogue flags, synthetic
//! data) under every tile/split plan, and compare the best plan with the runtime's heuristic.
//!
//!     gemm-tune [--list] <soma|unitree_g1> <float32|float16|bfloat16|int8> [batches...]
//!
//! `--list` only lists the GEMM launches of one generation (shape, conv/concat, plan) with
//! their counts and GFLOP, without timing them. Checkpoints: `$MOTIONBRICKS_CHECKPOINTS` or
//! `checkpoints/MotionBricks`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use common::Argv;
use motionbricks_rs::config::{Precision, checkpoint_dir, get_spec};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::transformer::{GemmCall, GemmParams, Plan, PlanKey, QWeight, TILES, Weight};
use motionbricks_rs::metal::{Recorder, wait};
use motionbricks_rs::npy::load_npz;

/// The GEMM launches of one generation of `batch` requests, as the runtime logged them.
fn launches(m: &MetalModel, batch: usize) -> Result<Vec<(GemmParams, bool)>> {
    *m.k.log.borrow_mut() = Some(Vec::new());
    let mut rec = Recorder::new(&m.gpu);
    m.encode(&mut rec, &m.dims(batch, false));
    wait(&rec.commit());
    m.k.log.borrow_mut().take().context("GEMM log was not recorded")
}

/// `--list`: distinct launches with counts and GFLOP, largest first.
fn list(log: &[(GemmParams, bool)]) {
    let mut counts: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for (p, q8) in log {
        let key = format!(
            "M={:<5} N={:<5} K={:<5} taps={} up={} concat={} mask={} arelu={} act={} q8={} splits={}",
            p.m, p.n, p.k, p.conv_taps, p.conv_up, p.k1 > 0, p.has_mask, p.a_relu, p.act, q8, p.splits
        );
        let e = counts.entry(key).or_default();
        e.0 += 1;
        e.1 += 2.0 * p.m as f64 * p.n as f64 * p.k as f64 / 1e9;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    for (k, (c, gf)) in v {
        println!("{c:>3}x  {gf:7.2} GFLOP  {k}");
    }
}

fn main() -> Result<()> {
    let mut argv = Argv::from_env();
    let mut positional = Vec::new();
    let mut list_only = false;
    while let Some(a) = argv.next() {
        if a == "--list" { list_only = true } else { positional.push(a) }
    }
    let mut positional = positional.into_iter();
    let name = get_spec(&positional.next().unwrap_or("soma".into()))?.name;
    let prec = positional.next().unwrap_or("float32".into()).parse::<Precision>()?.name();
    let batches = positional.map(|v| common::parse(&v, "batch size")).collect::<Result<Vec<usize>>>()?;
    let batches = if batches.is_empty() { vec![1, 2, 4, 8, 16, 32] } else { batches };
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut m = MetalModel::from_checkpoints_with(name, checkpoint_dir(name), 64, prec)?;
    println!("GPU: {}; GEMM plans: {}", m.gpu.info.summary(), m.k.plan_source);
    if list_only {
        for &batch in &batches {
            let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
            m.run(&req)?;
            println!("{name} {prec} N={batch}: GEMM launches of one generation");
            list(&launches(&m, batch)?);
        }
        return Ok(());
    }
    let gpu = m.gpu.clone();
    // TUNE_JSON=path: also write every (shape, plan, time) measurement, for fitting the heuristic offline.
    let mut dump = Vec::new();
    // TUNE_TABLE=path: append the best plan of every launch class (plan-table format).
    let mut table = Vec::new();
    for batch in batches {
        let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
        m.run(&req)?;
        let log = launches(&m, batch)?;
        // Distinct launches (plan fields cleared) with their counts.
        let mut uniq: BTreeMap<String, (GemmParams, bool, usize)> = BTreeMap::new();
        for (mut p, q8) in log {
            p.k_split = 0;
            p.splits = 0;
            uniq.entry(format!("{p:?}{q8}")).or_insert((p, q8, 0)).2 += 1;
        }
        let (mut tot_c, mut tot_h, mut tot_b) = (0.0, 0.0, 0.0);
        let mut rows = Vec::new();
        for (p, q8, count) in uniq.into_values() {
            let (mm, n, k) = (p.m as usize, p.n as usize, p.k as usize);
            let wa = k.max(p.lda as usize).max(p.lda2 as usize).max(p.lda3 as usize);
            let a = gpu.upload_t(&vec![0.01f32; mm * wa + 64]);
            let y = gpu.buffer_t(mm * (p.ldy as usize).max(n) + 64);
            let r = gpu.upload_t(&vec![0.0f32; mm * (p.ldr as usize).max(n) + 64]);
            let ones = gpu.upload(&vec![1.0f32; 2 * mm + 64]);
            let missing = gpu.upload_t(&vec![0.0f32; n + 64]);
            let bias = gpu.upload_t(&vec![0.0f32; n]);
            let w = if q8 {
                Weight::Q8(QWeight { q: gpu.upload_u32(&vec![0x80808080u32; n * k / 4]), scales: gpu.upload(&vec![0.001f32; n * k / 64]), biases: gpu.upload(&vec![-0.1f32; n * k / 64]) })
            } else {
                Weight::Dense(gpu.upload_t(&vec![0.02f32; n * k]))
            };
            let call = GemmCall {
                a: (&a, 0),
                w: &w,
                bias: &bias,
                y: (&y, 0),
                p,
                amask: Some(&ones),
                res: Some((&r, 0)),
                a2: Some(&a),
                a3: Some(&a),
                ymask: Some(&ones),
                missing: Some(&missing),
            };
            let reps = (2_000_000 / (mm * n * k / 1000 + 1)).clamp(5, 40);
            let time = |m: &mut MetalModel, plan: Option<Plan>| -> f64 {
                m.k.force = plan;
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let mut rec = Recorder::new(&gpu);
                    for _ in 0..reps {
                        m.k.run(&mut rec, &call);
                    }
                    best = best.min(wait(&rec.commit()) * 1e3 / reps as f64);
                }
                best
            };
            // Current plan (tuned table or heuristic), heuristic alone, then every plan.
            let current = time(&mut m, None);
            let hp = m.k.heuristic(mm, n, k, q8);
            let heuristic = time(&mut m, Some(hp));
            let mut all: Vec<(f64, Plan)> = Vec::new();
            for t in TILES {
                for splits in [1, 2, 4, 8] {
                    if (splits > 1 && k / splits < t.bk * 2) || (splits > 1 && splits * mm * n > m.k.partial.len()) {
                        continue;
                    }
                    let plan = Plan { tile: t, splits };
                    all.push((time(&mut m, Some(plan)), plan));
                }
            }
            // Re-time the four fastest twice more; keep each plan's minimum.
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            for c in all.iter_mut().take(4) {
                for _ in 0..2 {
                    c.0 = c.0.min(time(&mut m, Some(c.1)));
                }
            }
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            m.k.force = None;
            let best = *all.first().context("no GEMM plan applies to this launch")?;
            dump.push(format!(
                "{{\"batch\":{batch},\"count\":{count},\"m\":{mm},\"n\":{n},\"k\":{k},\"conv\":{},\"up\":{},\"concat\":{},\"q8\":{q8},\"plans\":[{}]}}",
                p.conv_taps,
                p.conv_up,
                p.k1 > 0,
                all.iter().map(|(v, pl)| format!("[{},{},{},{},{},{},{v:.2}]", pl.tile.bm, pl.tile.bn, pl.tile.bk, pl.tile.wm, pl.tile.wn, pl.splits)).collect::<Vec<_>>().join(",")
            ));
            let key = PlanKey::of(&p, q8);
            table.push(format!("{}  # {:.1} us (heuristic {:.1} us)", key.line(gpu.prec, &best.1), best.0, heuristic));
            let c = count as f64;
            tot_c += current * c;
            tot_h += heuristic * c;
            tot_b += best.0 * c;
            rows.push((current * c - best.0 * c, format!(
                "  {count:>2}x M={mm:<5} N={n:<4} K={k:<4} taps={} up={} cat={} q8={} | current {current:7.1} us | heuristic {:7.1} us ({}x{}x{} {}x{} s{}) | best {:7.1} us ({}x{}x{} {}x{} s{})",
                p.conv_taps, p.conv_up, p.k1 > 0, q8 as i32, heuristic, hp.tile.bm, hp.tile.bn, hp.tile.bk, hp.tile.wm, hp.tile.wn, hp.splits,
                best.0, best.1.tile.bm, best.1.tile.bn, best.1.tile.bk, best.1.tile.wm, best.1.tile.wn, best.1.splits
            )));
        }
        println!(
            "{name} {prec} N={batch}: GEMM total current {:.2} ms, heuristic {:.2} ms, best plans {:.2} ms",
            tot_c / 1e3, tot_h / 1e3, tot_b / 1e3
        );
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, r) in rows.iter().take(if std::env::var_os("TUNE_ALL").is_some() { usize::MAX } else { 12 }) {
            println!("{r}");
        }
    }
    if let Ok(path) = std::env::var("TUNE_TABLE") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        let g = &m.gpu.info;
        writeln!(f, "# {name} {prec}, measured on {} ({}, {} GPU cores)", g.name, g.architecture, g.cores)?;
        for l in &table {
            writeln!(f, "{l}")?;
        }
    }
    if let Ok(path) = std::env::var("TUNE_JSON") {
        std::fs::write(path, format!("[{}]", dump.join(",\n")))?;
    }
    Ok(())
}
