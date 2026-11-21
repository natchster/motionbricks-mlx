// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Prototype test of a model-specific Metal runtime against MLX on the largest sub-network: the
//! 16-layer pose transformer (dim 1024, 16 heads, FFN 2048, 16 tokens), real SOMA weights.
//! Checks numerics against the mlx-rs implementation and times both, sequential and pipelined.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use anyhow::Result;
use mlx_rs::transforms::compile::compile;
use mlx_rs::transforms::{async_eval, eval};
use mlx_rs::{Array, Dtype};
use motionbricks_rs::config::get_spec;
use motionbricks_rs::layers::TransformerEncoder;
use motionbricks_rs::metal::transformer::{Kernels, Scratch, Transformer};
use motionbricks_rs::metal::{Gpu, Recorder, wait};
use motionbricks_rs::ops::{i, to_vec_f32};
use motionbricks_rs::weights::{Weights, read_checkpoints};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let spec = get_spec("soma")?;
    let loaded = read_checkpoints(here.join("../checkpoints/MotionBricks/soma"), &spec)?;
    let prefix = "pose.transformer_model";
    let (layers, heads, len, d) = (16, 16, 16usize, 1024usize);

    let gpu = Gpu::new()?;
    let kernels = Kernels::new(&gpu)?;
    let metal = Transformer::load(&gpu, &loaded.tensors, prefix, layers, heads)?;
    let mut w = Weights::new(loaded.tensors.clone(), Dtype::Float32);
    let mlx_tf = Rc::new(TransformerEncoder::load(&mut w, prefix, layers, heads as i32)?);
    let tf = mlx_tf.clone();
    let mut mlx_fn = compile(move |a: &[Array]| Ok(vec![tf.forward(&a[0], Some(&a[1]))?]), None);

    let iters = 40;
    for batch in [1usize, 2, 4, 8, 16, 32] {
        let rows = batch * len;
        let mut seed = 12345u64;
        let x: Vec<f32> = (0..rows * d)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        let valid: Vec<usize> = (0..batch).map(|b| 6 + (b * 5) % 11).collect();
        let pad: Vec<u8> = (0..batch).flat_map(|b| (0..len).map(move |t| (t >= 6 + (b * 5) % 11) as u8)).collect();
        let _ = valid;

        // MLX reference.
        let xa = Array::from_slice(&x, &[batch as i32, len as i32, d as i32]);
        let pada = Array::from_slice(&pad.iter().map(|&v| v as i32).collect::<Vec<_>>(), &[batch as i32, len as i32]).ne(i(0))?;
        let want = to_vec_f32(&mlx_fn(&[xa.clone(), pada.clone()])?[0])?;

        // Metal.
        let xb = gpu.upload(&x);
        let padb = gpu.upload_bytes(&pad);
        let scratch = Scratch::new(&gpu, rows, d, metal.ffn);
        let mut rec = Recorder::new(&gpu);
        metal.encode(&kernels, &mut rec, &xb, Some(&padb), batch, len, &scratch);
        let dispatches = rec.dispatches;
        wait(&rec.commit());
        let got = xb.as_slice().to_vec();
        let num = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let den = want.iter().map(|v| v.abs()).fold(0f32, f32::max);

        // Timing: sequential (one call at a time, synchronous) and pipelined (async, sync at end).
        let mut mlx_seq = Vec::new();
        for it in 0..iters + 5 {
            let t = Instant::now();
            let o = mlx_fn(&[xa.clone(), pada.clone()])?;
            eval(o.iter())?;
            if it >= 5 {
                mlx_seq.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let t = Instant::now();
        let mut last = Vec::new();
        for _ in 0..iters {
            last = mlx_fn(&[xa.clone(), pada.clone()])?;
            async_eval(last.iter())?;
        }
        eval(last.iter())?;
        let mlx_pipe = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let (mut met_seq, mut met_gpu) = (Vec::new(), Vec::new());
        for it in 0..iters + 5 {
            let t = Instant::now();
            xb.as_mut_slice().copy_from_slice(&x);
            let mut rec = Recorder::new(&gpu);
            metal.encode(&kernels, &mut rec, &xb, Some(&padb), batch, len, &scratch);
            let g = wait(&rec.commit());
            if it >= 5 {
                met_seq.push(t.elapsed().as_secs_f64() * 1e3);
                met_gpu.push(g);
            }
        }
        let t = Instant::now();
        let mut cbs = Vec::new();
        for _ in 0..iters {
            let mut rec = Recorder::new(&gpu);
            metal.encode(&kernels, &mut rec, &xb, Some(&padb), batch, len, &scratch);
            cbs.push(rec.commit());
        }
        wait(cbs.last().unwrap());
        let met_pipe = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let part = |mask: u32| -> f64 {
            let mut v = Vec::new();
            for _ in 0..10 {
                let mut rec = Recorder::new(&gpu);
                metal.encode_parts(&kernels, &mut rec, &xb, Some(&padb), batch, len, &scratch, mask);
                v.push(wait(&rec.commit()));
            }
            median(v)
        };
        let (pg, pa, pn) = (part(1), part(2), part(4));
        println!("      GPU by family: GEMMs {pg:6.2} ms, attention {pa:6.2} ms, add+layernorm {pn:6.2} ms");
        println!(
            "N={batch:<2} rows={rows:<3} rel err {:.1e} | MLX seq {:6.2} ms pipe {:6.2} ms | Metal seq {:6.2} ms (GPU {:6.2}) pipe {:6.2} ms, {dispatches} dispatches | speedup seq {:.2}x pipe {:.2}x",
            num / den,
            median(mlx_seq.clone()),
            mlx_pipe,
            median(met_seq.clone()),
            median(met_gpu),
            met_pipe,
            median(mlx_seq) / median(met_seq),
            mlx_pipe / met_pipe
        );
    }
    Ok(())
}
