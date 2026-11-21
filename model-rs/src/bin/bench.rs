// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Benchmark of the Rust port, with the same methodology as `bench/bench_python.py`.
//!
//! Per call it records: input preparation, building the lazy graph (compiled replay or eager),
//! `eval` wall time and the calling thread's CPU time inside `eval` (Metal encoding), and the
//! copy of the outputs to host vectors. It also measures a stream of single requests, run
//! back to back (`sequential`) or with the next request encoded while the GPU still runs the
//! previous one (`pipelined`, via `async_eval`).
//!
//!     cargo run --release --bin mbrs-bench -- --model soma --batches 1 8 16 32 --json out.json

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::transforms::{async_eval, eval};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::wait;
use motionbricks_rs::npy::load_npz;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled, parse_dtype};

fn thread_cpu() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn copy_out(model: &MotionBricks, outputs: &[Array]) -> Result<usize> {
    let mut n = 0;
    for (name, o) in model.output_names().iter().zip(outputs) {
        n += if *name == "pred_num_tokens" { to_vec_i32(o)?.len() } else { to_vec_f32(o)?.len() };
    }
    Ok(n)
}

struct Args {
    model: String,
    checkpoints: Option<PathBuf>,
    dtype: String,
    batches: Vec<usize>,
    iters: usize,
    warmup: usize,
    stream: usize,
    json: Option<PathBuf>,
    contiguous: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: "soma".into(),
        checkpoints: None,
        dtype: "float32".into(),
        batches: vec![1, 2, 4, 8, 16, 32],
        iters: 40,
        warmup: 5,
        stream: 200,
        json: None,
        contiguous: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 0;
    while k < argv.len() {
        let next = |k: usize| argv.get(k + 1).cloned().ok_or_else(|| anyhow::anyhow!("{} needs a value", argv[k]));
        match argv[k].as_str() {
            "--model" => a.model = next(k)?,
            "--checkpoints" => a.checkpoints = Some(next(k)?.into()),
            "--dtype" => a.dtype = next(k)?,
            "--iters" => a.iters = next(k)?.parse()?,
            "--warmup" => a.warmup = next(k)?.parse()?,
            "--stream" => a.stream = next(k)?.parse()?,
            "--json" => a.json = Some(next(k)?.into()),
            "--contiguous-weights" => {
                a.contiguous = true;
                k -= 1;
            }
            "--batches" => {
                a.batches.clear();
                while k + 1 < argv.len() && !argv[k + 1].starts_with("--") {
                    a.batches.push(argv[k + 1].parse()?);
                    k += 1;
                }
                k -= 1;
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        k += 2;
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let folder = if args.model == "soma" { "soma" } else { "g1" };
    let ckpt = args.checkpoints.clone().unwrap_or_else(|| here.join("../checkpoints/MotionBricks").join(folder));
    let load_start = Instant::now();
    // The mlx-rs port has no INT8 mode; its rows are skipped for --dtype int8.
    let mlx_dtype = if args.dtype == "int8" { "float32" } else { args.dtype.as_str() };
    let model = Rc::new(MotionBricks::from_checkpoints_with(&args.model, &ckpt, parse_dtype(mlx_dtype)?, args.contiguous)?);
    let run_mlx = args.dtype != "int8";
    let load_s = load_start.elapsed().as_secs_f64();
    let mut forward = compiled(model.clone());
    let metal = Some(MetalModel::from_checkpoints_with(&args.model, &ckpt, 64, &args.dtype)?);

    let mut report = BTreeMap::new();
    for &batch in &args.batches {
        let request = load_npz(here.join("bench/requests").join(format!("{}_N{batch}.npz", args.model)))?;
        for mode in ["compiled", "eager"].into_iter().filter(|_| run_mlx) {
            let keys = ["prepare", "graph", "eval_wall", "eval_cpu", "to_host", "total", "core"];
            let mut samples: BTreeMap<&str, Vec<f64>> = keys.iter().map(|k| (*k, Vec::new())).collect();
            let mut first_ms = 0.0;
            for it in 0..args.warmup + args.iters {
                let t0 = Instant::now();
                let inputs = model.prepare(&request)?;
                let t1 = Instant::now();
                let outputs = if mode == "compiled" { forward(&inputs)? } else { model.forward(&inputs)? };
                let t2 = Instant::now();
                let c0 = thread_cpu();
                eval(outputs.iter())?;
                let c1 = thread_cpu();
                let t3 = Instant::now();
                copy_out(&model, &outputs)?;
                let t4 = Instant::now();
                if it == 0 {
                    first_ms = (t4 - t0).as_secs_f64() * 1e3;
                }
                if it >= args.warmup {
                    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
                    let values = [ms(t0, t1), ms(t1, t2), ms(t2, t3), (c1 - c0) * 1e3, ms(t3, t4), ms(t0, t4), ms(t1, t3)];
                    for (k, v) in keys.iter().zip(values) {
                        samples.get_mut(k).unwrap().push(v);
                    }
                }
            }
            let mut row = BTreeMap::new();
            for (k, mut v) in samples {
                row.insert(format!("{k}_ms"), median(&mut v));
                if k == "total" || k == "core" {
                    row.insert(format!("{k}_p10_ms"), percentile(&mut v, 0.1));
                    row.insert(format!("{k}_p90_ms"), percentile(&mut v, 0.9));
                }
            }
            row.insert("first_call_ms".into(), first_ms);
            println!(
                "rust {:<10} {:<8} {:<8} N={batch:<3} core {:7.2} ms  total {:7.2} ms  (graph {:5.2}, eval {:6.2} / cpu {:5.2})",
                args.model, args.dtype, mode, row["core_ms"], row["total_ms"], row["graph_ms"], row["eval_wall_ms"], row["eval_cpu_ms"]
            );
            report.insert(format!("{}/{}/N={batch}/{mode}", args.model, args.dtype), row);
        }
        if let Some(mm) = &metal {
            // Custom Metal runtime: upload, encode one command buffer, wait, read back.
            let keys = ["prepare", "graph", "eval_wall", "eval_cpu", "to_host", "total", "core", "gpu"];
            let mut samples: BTreeMap<&str, Vec<f64>> = keys.iter().map(|k| (*k, Vec::new())).collect();
            let mut first_ms = 0.0;
            for it in 0..args.warmup + args.iters {
                let t0 = Instant::now();
                let n = mm.upload_inputs(&request)?;
                let t1 = Instant::now();
                let d = mm.dims(n, false);
                let mut rec = mm.recorder();
                mm.encode(&mut rec, &d);
                let cb = rec.commit();
                let t2 = Instant::now();
                let c0 = thread_cpu();
                let gpu = wait(&cb);
                let c1 = thread_cpu();
                let t3 = Instant::now();
                let out = mm.outputs(n);
                std::hint::black_box(&out);
                let t4 = Instant::now();
                if it == 0 {
                    first_ms = (t4 - t0).as_secs_f64() * 1e3;
                }
                if it >= args.warmup {
                    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
                    let values = [ms(t0, t1), ms(t1, t2), ms(t2, t3), (c1 - c0) * 1e3, ms(t3, t4), ms(t0, t4), ms(t1, t3), gpu];
                    for (k, v) in keys.iter().zip(values) {
                        samples.get_mut(k).unwrap().push(v);
                    }
                }
            }
            let mut row = BTreeMap::new();
            for (k, mut v) in samples {
                row.insert(format!("{k}_ms"), median(&mut v));
                if k == "total" || k == "core" {
                    row.insert(format!("{k}_p10_ms"), percentile(&mut v, 0.1));
                    row.insert(format!("{k}_p90_ms"), percentile(&mut v, 0.9));
                }
            }
            row.insert("first_call_ms".into(), first_ms);
            println!(
                "rust {:<10} {:<8} {:<8} N={batch:<3} core {:7.2} ms  total {:7.2} ms  (encode {:5.2}, wait {:6.2}, GPU {:6.2})",
                args.model, args.dtype, "metal", row["core_ms"], row["total_ms"], row["graph_ms"], row["eval_wall_ms"], row["gpu_ms"]
            );
            report.insert(format!("{}/{}/N={batch}/metal", args.model, args.dtype), row);
        }
    }

    // A stream of single requests: back to back, or pipelined (encode i+1 while the GPU runs i).
    if args.stream > 0 && run_mlx {
        let request = load_npz(here.join("bench/requests").join(format!("{}_N1.npz", args.model)))?;
        let inputs = model.prepare(&request)?;
        for _ in 0..args.warmup {
            eval(forward(&inputs)?.iter())?;
        }
        let t = Instant::now();
        for _ in 0..args.stream {
            let out = forward(&inputs)?;
            eval(out.iter())?;
            copy_out(&model, &out)?;
        }
        let sequential = args.stream as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut pending: Option<Vec<Array>> = None;
        for _ in 0..args.stream {
            let out = forward(&inputs)?;
            async_eval(out.iter())?;
            if let Some(prev) = pending.take() {
                copy_out(&model, &prev)?; // waits for the previous request only
            }
            pending = Some(out);
        }
        if let Some(prev) = pending {
            copy_out(&model, &prev)?;
        }
        let pipelined = args.stream as f64 / t.elapsed().as_secs_f64();
        println!("rust {:<10} {:<8} stream of N=1 requests: sequential {sequential:6.1} req/s, pipelined {pipelined:6.1} req/s", args.model, args.dtype);
        let mut row = BTreeMap::new();
        row.insert("sequential_req_s".to_string(), sequential);
        row.insert("pipelined_req_s".to_string(), pipelined);
        if let Some(mm) = &metal {
            for _ in 0..args.warmup {
                mm.run(&request)?;
            }
            let t = Instant::now();
            for _ in 0..args.stream {
                std::hint::black_box(mm.run(&request)?);
            }
            let seq = args.stream as f64 / t.elapsed().as_secs_f64();
            println!("rust {:<10} {:<8} metal stream of N=1 requests: sequential {seq:6.1} req/s", args.model, args.dtype);
            row.insert("metal_sequential_req_s".to_string(), seq);
        }
        report.insert(format!("{}/{}/stream", args.model, args.dtype), row);
    }

    if let Some(path) = args.json {
        let mut s = String::from("{\n");
        s += &format!("  \"runtime\": \"rust\", \"load_s\": {load_s:.3},\n  \"results\": {{\n");
        let rows: Vec<String> = report
            .iter()
            .map(|(k, row)| {
                let fields: Vec<String> = row.iter().map(|(f, v)| format!("\"{f}\": {v:.4}")).collect();
                format!("    \"{k}\": {{{}}}", fields.join(", "))
            })
            .collect();
        s += &rows.join(",\n");
        s += "\n  }\n}\n";
        std::fs::write(path, s)?;
    }
    Ok(())
}
