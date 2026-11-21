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
//!
//! Options: `--model soma|unitree_g1`, `--checkpoints DIR` (default: `$MOTIONBRICKS_CHECKPOINTS`
//! or `checkpoints/MotionBricks`), `--dtype float32|float16|bfloat16|int8`, `--batches N...`,
//! `--iters N`, `--warmup N`, `--stream N`, `--json PATH`, `--runtime both|mlx|metal`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Result, bail};
use common::{Argv, median, percentile};
use mlx_rs::Array;
use mlx_rs::transforms::{async_eval, eval};
use motionbricks_rs::config::{Precision, checkpoint_dir, get_spec};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::wait;
use motionbricks_rs::npy::load_npz;
use motionbricks_rs::ops::{to_vec_f32, to_vec_i32};
use motionbricks_rs::{MotionBricks, compiled};

fn thread_cpu() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
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
    /// Which runtimes to load and time: "both" (default), "mlx" or "metal".
    runtime: String,
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
        runtime: "both".into(),
    };
    let mut argv = Argv::from_env();
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--model" => a.model = argv.value(&flag)?,
            "--checkpoints" => a.checkpoints = Some(argv.value(&flag)?.into()),
            "--dtype" => a.dtype = argv.value(&flag)?,
            "--iters" => a.iters = argv.parsed(&flag)?,
            "--warmup" => a.warmup = argv.parsed(&flag)?,
            "--stream" => a.stream = argv.parsed(&flag)?,
            "--json" => a.json = Some(argv.value(&flag)?.into()),
            "--batches" => a.batches = argv.values(&flag)?,
            "--runtime" => a.runtime = argv.value(&flag)?,
            other => bail!("unknown argument {other}"),
        }
    }
    // Normalized names: "g1" -> "unitree_g1", "bf16" -> "bfloat16" (request files and report keys).
    a.model = get_spec(&a.model)?.name.to_string();
    a.dtype = a.dtype.parse::<Precision>()?.name().to_string();
    if a.iters == 0 {
        bail!("--iters must be at least 1");
    }
    if !["both", "mlx", "metal"].contains(&a.runtime.as_str()) {
        bail!("--runtime must be both, mlx or metal, not {:?}", a.runtime);
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ckpt = args.checkpoints.clone().unwrap_or_else(|| checkpoint_dir(&args.model));
    let precision: Precision = args.dtype.parse()?;
    let load_start = Instant::now();
    // The mlx-rs port's rows are skipped for --dtype int8 (only the Metal runtime is timed at
    // INT8); the float32 model is still loaded then, as in earlier versions of this benchmark.
    let mlx_precision = if precision.is_quantized() { Precision::Float32 } else { precision };
    let model = match args.runtime.as_str() {
        "metal" => None,
        _ => Some(Rc::new(MotionBricks::load(&args.model, &ckpt, mlx_precision)?)),
    };
    let run_mlx = model.is_some() && !precision.is_quantized();
    let load_s = load_start.elapsed().as_secs_f64();
    let mut forward = model.clone().map(compiled);
    let metal = match args.runtime.as_str() {
        "mlx" => None,
        _ => Some(MetalModel::from_checkpoints_with(&args.model, &ckpt, 64, args.dtype.as_str())?),
    };

    let mut report = BTreeMap::new();
    for &batch in &args.batches {
        let request = load_npz(here.join("bench/requests").join(format!("{}_N{batch}.npz", args.model)))?;
        for mode in ["compiled", "eager"].into_iter().filter(|_| run_mlx) {
            let (Some(model), Some(forward)) = (&model, forward.as_mut()) else { break };
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
                copy_out(model, &outputs)?;
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
    if args.stream > 0 && (run_mlx || args.runtime == "metal") {
        let request = load_npz(here.join("bench/requests").join(format!("{}_N1.npz", args.model)))?;
        let mut row = BTreeMap::new();
        if let (true, Some(model), Some(forward)) = (run_mlx, &model, forward.as_mut()) {
            let inputs = model.prepare(&request)?;
            for _ in 0..args.warmup {
                eval(forward(&inputs)?.iter())?;
        }
        let t = Instant::now();
        for _ in 0..args.stream {
            let out = forward(&inputs)?;
            eval(out.iter())?;
            copy_out(model, &out)?;
        }
        let sequential = args.stream as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut pending: Option<Vec<Array>> = None;
        for _ in 0..args.stream {
            let out = forward(&inputs)?;
            async_eval(out.iter())?;
            if let Some(prev) = pending.take() {
                copy_out(model, &prev)?; // waits for the previous request only
            }
            pending = Some(out);
        }
        if let Some(prev) = pending {
            copy_out(model, &prev)?;
        }
        let pipelined = args.stream as f64 / t.elapsed().as_secs_f64();
        println!("rust {:<10} {:<8} stream of N=1 requests: sequential {sequential:6.1} req/s, pipelined {pipelined:6.1} req/s", args.model, args.dtype);
        row.insert("sequential_req_s".to_string(), sequential);
        row.insert("pipelined_req_s".to_string(), pipelined);
        }
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

#[cfg(test)]
mod tests {
    use super::common::{median, percentile};

    #[test]
    fn statistics() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&mut []).is_nan());
        assert_eq!(median(&mut [1.0, f64::NAN, 0.0]), 1.0, "NaN sorts last instead of panicking");
        assert_eq!(percentile(&mut [5.0, 1.0, 3.0], 0.0), 1.0);
        assert_eq!(percentile(&mut [5.0, 1.0, 3.0], 1.0), 5.0);
        assert!(percentile(&mut [], 0.5).is_nan());
    }
}
