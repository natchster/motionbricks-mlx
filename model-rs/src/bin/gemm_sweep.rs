// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Time every GEMM tile/split configuration on the model's GEMM shapes, as a chain of dependent
//! dispatches (like inside the network), and report the best plan vs the heuristic.

use anyhow::Result;
use motionbricks_rs::metal::transformer::{GemmCall, GemmParams, Kernels, Plan, TILES, Weight};
use motionbricks_rs::metal::{Gpu, Precision, Recorder, wait};

fn main() -> Result<()> {
    // Usage: gemm-sweep [float32|float16|bfloat16|int8] [label filter]
    let dtype = std::env::args().nth(1).unwrap_or("float32".into());
    let filter = std::env::args().nth(2).unwrap_or_default();
    let int8 = dtype == "int8";
    let gpu = Gpu::with_precision(Precision::parse(&dtype)?)?;
    let mut k = Kernels::new(&gpu)?;
    // (label, M, N, K)
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("pose qkv N=4", 64, 3072, 1024),
        ("pose qkv N=1", 16, 3072, 1024),
        ("pose out N=1", 16, 1024, 1024),
        ("pose ff1 N=1", 16, 2048, 1024),
        ("pose ff2 N=1", 16, 1024, 2048),
        ("root qkv N=1 (25 tok)", 25, 1536, 512),
        ("root ff2 N=1 (25 tok)", 25, 512, 2048),
        ("conv k3 64fr N=1", 64, 512, 1536),
        ("pose qkv N=8", 128, 3072, 1024),
        ("pose qkv N=32", 512, 3072, 1024),
        ("pose ff2 N=32", 512, 1024, 2048),
        ("conv k3 64fr N=32", 2048, 512, 1536),
    ];
    let reps = 50;
    for (label, m, n, kk) in shapes.into_iter().filter(|s| s.0.contains(filter.as_str())) {
        let a = gpu.upload_t(&vec![0.01f32; m * kk]);
        let w = if int8 {
            use motionbricks_rs::metal::transformer::QWeight;
            Weight::Q8(QWeight { q: gpu.upload_u32(&vec![0x80808080u32; n * kk / 4]), scales: gpu.upload(&vec![0.001f32; n * kk / 64]), biases: gpu.upload(&vec![-0.1f32; n * kk / 64]) })
        } else {
            Weight::Dense(gpu.upload_t(&vec![0.02f32; n * kk]))
        };
        let b = gpu.upload_t(&vec![0.0f32; n]);
        let y = gpu.buffer_t(m * n);
        let p = GemmParams { m: m as i32, n: n as i32, k: kk as i32, lda: kk as i32, ldy: n as i32, ..Default::default() };
        let time = |k: &mut Kernels, plan: Option<Plan>| -> f64 {
            k.force = plan;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut rec = Recorder::new(&gpu);
                for _ in 0..reps {
                    k.run(&mut rec, &GemmCall::new(&a, &w, &b, &y, p));
                }
                best = best.min(wait(&rec.commit()) / reps as f64);
            }
            best
        };
        let heuristic = time(&mut k, None);
        k.force = None;
        let hp = k.plan(m, n, kk);
        let mut results = Vec::new();
        for t in TILES {
            for splits in [1, 2, 4, 8] {
                if kk / splits < t.bk * 2 || (splits > 1 && splits * m * n > k.partial.len) {
                    continue;
                }
                results.push((time(&mut k, Some(Plan { tile: t, splits })), t, splits));
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if std::env::var("SWEEP_ALL").is_ok() {
            for (t_, tl, s_) in &results {
                println!("    {:7.1} us  {}x{}x{} wm{} wn{} s{}", t_ * 1e3, tl.bm, tl.bn, tl.bk, tl.wm, tl.wn, s_);
            }
        }
        let (bt, btile, bs) = results[0];
        let es = gpu.prec.size() as f64;
        let bytes = (n * kk) as f64 * if int8 { 1.0 } else { es } + (m * kk + m * n) as f64 * es;
        let flops = 2.0 * (m * n * kk) as f64;
        println!(
            "{label:<22} M={m:<4} N={n:<4} K={kk:<4} steel/heuristic {:7.1} us ({}x{}x{} s{}) | best {:7.1} us ({}x{}x{} wm{} wn{} s{bs}) {:5.0} GB/s {:5.2} TFLOPS",
            heuristic * 1e3, hp.tile.bm, hp.tile.bn, hp.tile.bk, hp.splits,
            bt * 1e3, btile.bm, btile.bn, btile.bk, btile.wm, btile.wn,
            bytes / (bt * 1e-3) / 1e9, flops / (bt * 1e-3) / 1e12
        );
    }
    Ok(())
}
