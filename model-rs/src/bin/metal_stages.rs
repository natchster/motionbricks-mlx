// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! GPU time per stage of the Metal runtime (each stage re-run on valid buffers from a full pass).
use std::path::PathBuf;

use anyhow::Result;
use motionbricks_rs::metal::model::MetalModel;
use motionbricks_rs::metal::{Recorder, wait};
use motionbricks_rs::npy::load_npz;

fn main() -> Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for name in ["soma", "unitree_g1"] {
        let folder = if name == "soma" { "soma" } else { "g1" };
        let m = MetalModel::from_checkpoints(name, here.join("../checkpoints/MotionBricks").join(folder), 64)?;
        for batch in [1usize, 32] {
            let req = load_npz(here.join("bench/requests").join(format!("{name}_N{batch}.npz")))?;
            m.run(&req)?;
            let d = m.dims(batch, false);
            let time = |f: &dyn Fn(&mut Recorder)| -> f64 {
                let mut v: Vec<f64> = (0..15).map(|_| { let mut r = Recorder::new(&m.gpu); f(&mut r); wait(&r.commit()) }).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[7]
            };
            let input = time(&|r| m.encode_input(r, &d));
            let root = time(&|r| m.encode_root(r, &d));
            let r2p = time(&|r| m.encode_root_to_pose(r, &d));
            let pose = time(&|r| m.encode_pose(r, &d));
            let vq = time(&|r| m.encode_vq(r, &d));
            let fin = time(&|r| m.encode_finish(r, &d));
            let all = time(&|r| m.encode(r, &d));
            let mut probe = Recorder::new(&m.gpu);
            m.encode(&mut probe, &d);
            let dispatches = probe.dispatches;
            wait(&probe.commit());
            println!("{name:<10} N={batch:<2} GPU ms: input {input:.2} | root {root:.2} | root->pose {r2p:.2} | pose {pose:.2} | vq {vq:.2} | finish {fin:.2} | whole {all:.2} | {dispatches} dispatches");
        }
    }
    Ok(())
}
