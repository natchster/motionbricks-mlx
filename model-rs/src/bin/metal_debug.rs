// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Bisect the Metal root network against mlx-rs on one golden case.
use std::path::PathBuf;

use anyhow::Result;
use mlx_rs::{Array, Dtype};
use motionbricks_rs::config::get_spec;
use motionbricks_rs::layers::{DoubleCondDecoder, TransformerEncoder};
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Buf, Recorder, wait};
use motionbricks_rs::npy::load_npz;
use motionbricks_rs::ops::{i, to_vec_f32};
use motionbricks_rs::weights::{Weights, read_checkpoints};

fn put(dst: &Buf, v: &[f32]) { dst.as_mut_slice()[..v.len()].copy_from_slice(v); }
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let n = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
    n / b.iter().map(|v| v.abs() as f64).fold(1e-12, f64::max)
}
fn arr(v: &[f32], s: &[i32]) -> Array { Array::from_slice(v, s) }

fn main() -> Result<()> {
    let name = std::env::args().nth(1).unwrap_or("soma".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let dir = root.join("checkpoints/MotionBricks").join(if name == "soma" { "soma" } else { "g1" });
    let m = MetalModel::from_checkpoints(&name, &dir, 8)?;
    let spec = get_spec(&name)?;
    let loaded = read_checkpoints(&dir, &spec)?;
    let mut w = Weights::new(loaded.tensors.clone(), Dtype::Float32);
    let tf = TransformerEncoder::load(&mut w, "root.root_token_transformer_model", spec.root.n_layers_root_token, 16)?;
    let dec = DoubleCondDecoder::load(&mut w, "root.conv_output", 2, 512, 4, 3)?;
    let g = load_npz(root.join("sandbox/golden").join(&name).join("trace.npz"))?;
    let n = 4usize;
    let b = &m.b;
    let mask = |k: &str| g[k].to_f32().into_iter().map(|x| (x != 0.0) as i32 as f32).collect::<Vec<_>>();
    put(&b.gn, &g["root_in.global_root"].to_f32());
    put(&b.ln, &g["root_in.local_root"].to_f32());
    put(&b.pn, &g["root_in.poses"].to_f32());
    put(&b.hg, &mask("root_in.has_global"));
    put(&b.hl, &mask("root_in.has_local"));
    put(&b.hp, &mask("root_in.has_poses"));
    put(&b.allowed, &mask("root_in.allowed"));
    let nt: Vec<f32> = g["root_in.num_tokens"].to_i32().iter().map(|&x| f32::from_bits(x as u32)).collect();
    put(&b.nt_in, &nt);
    let d = m.dims(n, true);
    let mut rec = Recorder::new(&m.gpu);
    m.encode_root_a(&mut rec, &d);
    wait(&rec.commit());
    let x2 = b.x2.as_slice()[..n * 25 * 512].to_vec();
    let pad2: Vec<i32> = b.pad2.as_slice()[..n * 25].iter().map(|&v| v as i32).collect();
    // transformer
    let y_ref = to_vec_f32(&tf.forward(&arr(&x2, &[n as i32, 25, 512]), Some(&Array::from_slice(&pad2, &[n as i32, 25]).ne(i(0))?))?)?;
    let mut rec = Recorder::new(&m.gpu);
    m.encode_root_b(&mut rec, &d, 0, true);
    wait(&rec.commit());
    println!("root-token transformer rel err {:.2e}", rel(&b.x2.as_slice()[..n * 25 * 512], &y_ref));
    // full decoder from the Metal decoder inputs
    let [h, h2, _t1] = m.scratch();
    let _ = (h, h2);
    let dec_in: Vec<f32> = (0..n).flat_map(|bb| y_ref[(bb * 25 + 9) * 512..(bb * 25 + 25) * 512].to_vec()).collect();
    let m16: Vec<i32> = b.m16.as_slice()[..n * 16].iter().map(|&v| v as i32).collect();
    let dhas: Vec<i32> = b.dhas_debug()[..n * 64].iter().map(|&v| v as i32).collect();
    let pr_ref = to_vec_f32(&dec.forward(
        &arr(&dec_in, &[n as i32, 16, 512]),
        &arr(&b.dframe_debug()[..n * 64 * 512], &[n as i32, 64, 512]),
        &arr(&b.droot_debug()[..n * 64 * 5], &[n as i32, 64, 5]),
        &Array::from_slice(&dhas, &[n as i32, 64]).ne(i(0))?,
        &Array::from_slice(&m16, &[n as i32, 16]).ne(i(0))?,
    )?)?;
    for limit in [usize::MAX] {
        let mut rec = Recorder::new(&m.gpu);
        m.encode_root_b(&mut rec, &d, limit, false);
        wait(&rec.commit());
    }
    println!("decoder (from identical inputs) rel err {:.2e}; vs golden {:.2e}", rel(&b.pr.as_slice()[..n * 64 * 5], &pr_ref), rel(&pr_ref, &g["root_out.pred_global_root_values"].to_f32()));
    Ok(())
}
