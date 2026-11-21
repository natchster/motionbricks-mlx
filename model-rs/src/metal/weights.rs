// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Loading MotionBricks onto the Metal runtime: weight blocks (linear, conv, FC block, decoder),
//! the release configuration's constant token embedding, and the packed rig tables.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::Array;

use super::transformer::{Weight, upload};
use super::{Buf, Gpu, Precision};
use crate::config::{Kind, ModelSpec};
use crate::npy::{Npy, load_npz_bytes};
use crate::ops::to_vec_f32;

/// A Linear layer: W[n][k] and b[n].
pub(super) struct Lin {
    pub w: Weight,
    pub b: Buf,
    pub n: usize,
    pub k: usize,
}

/// A Conv1d layer, channels-last: W[n][taps][cin] and b[n].
pub(super) struct Conv {
    pub w: Weight,
    pub b: Buf,
    pub n: usize,
    pub cin: usize,
    pub taps: usize,
    pub dil: usize,
}

/// `FCBlock` (layers.py): two leaky-ReLU layers and a projection.
pub(super) struct FcBlock {
    pub layers: Vec<Lin>,
    pub proj: Lin,
}

/// One upsampling level of a `DoubleCondDecoder`.
pub(super) struct Level {
    pub target: Lin,
    pub ext: Lin,
    pub res: Vec<(Conv, Conv)>,
    pub up: Conv,
}

/// `DoubleCondDecoder` (layers.py).
pub(super) struct Decoder {
    pub conv0: Conv,
    pub levels: Vec<Level>,
    pub post: Conv,
    pub out: Conv,
}

impl Decoder {
    /// Every conv and linear layer with a name for error messages.
    pub fn check(&self, name: &str, cin: usize, target_dim: usize, ext_dim: usize, cout: usize, width: usize) -> Result<()> {
        ensure!(self.conv0.cin == cin, "{name}: conv0 reads {} channels, expected {cin}", self.conv0.cin);
        ensure!(self.out.n == cout, "{name}: output conv writes {} channels, expected {cout}", self.out.n);
        ensure!(self.levels.len() == 2, "{name}: {} levels, expected 2", self.levels.len());
        let mut convs: Vec<&Conv> = vec![&self.conv0, &self.post, &self.out];
        for (i, l) in self.levels.iter().enumerate() {
            let fpp = 4 >> i;
            ensure!(l.target.k == target_dim, "{name}: level {i} target layer reads {}, expected {target_dim}", l.target.k);
            ensure!(l.ext.k == width + fpp * ext_dim, "{name}: level {i} external layer reads {}, expected {}", l.ext.k, width + fpp * ext_dim);
            ensure!(l.target.n <= width && l.ext.n <= width, "{name}: level {i} layers wider than {width}");
            convs.push(&l.up);
            for (a, b) in &l.res {
                convs.push(a);
                convs.push(b);
            }
        }
        for c in convs {
            ensure!(c.n <= width.max(cout) && c.cin <= width.max(cin), "{name}: conv {}x{}x{} exceeds the decoder buffers ({width} channels)", c.n, c.taps, c.cin);
            ensure!(c.taps % 2 == 1, "{name}: conv kernel size {} must be odd", c.taps);
        }
        Ok(())
    }
}

pub(super) struct Loader<'a> {
    pub gpu: &'a Gpu,
    pub t: &'a HashMap<String, Array>,
}

impl Loader<'_> {
    fn tensor(&self, name: &str) -> Result<&Array> {
        self.t.get(name).ok_or_else(|| anyhow!("missing weight {name}"))
    }
    pub fn buf(&self, name: &str) -> Result<Buf> {
        upload(self.gpu, self.t, name)
    }
    pub fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(to_vec_f32(self.tensor(name)?)?)
    }
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape().iter().map(|&d| d as usize).collect())
    }
    pub fn lin(&self, p: &str) -> Result<Lin> {
        let s = self.shape(&format!("{p}.weight"))?;
        ensure!(s.len() == 2, "{p}.weight: expected 2 dimensions, got {s:?}");
        Ok(Lin { w: Weight::Dense(self.buf(&format!("{p}.weight"))?), b: self.buf(&format!("{p}.bias"))?, n: s[0], k: s[1] })
    }
    pub fn conv(&self, p: &str, dil: usize) -> Result<Conv> {
        let s = self.shape(&format!("{p}.weight"))?; // (out, k, in), already channels-last
        ensure!(s.len() == 3, "{p}.weight: expected 3 dimensions, got {s:?}");
        Ok(Conv { w: Weight::Dense(self.buf(&format!("{p}.weight"))?), b: self.buf(&format!("{p}.bias"))?, n: s[0], taps: s[1], cin: s[2], dil })
    }
    pub fn fc(&self, p: &str) -> Result<FcBlock> {
        Ok(FcBlock { layers: vec![self.lin(&format!("{p}.fc_layers.0"))?, self.lin(&format!("{p}.fc_layers.1"))?], proj: self.lin(&format!("{p}.forward_projection"))? })
    }
    pub fn decoder(&self, p: &str) -> Result<Decoder> {
        let mut levels = Vec::new();
        for i in 0..2 {
            let res = (0..4)
                .map(|j| {
                    let dil = 3usize.pow(3 - j as u32);
                    Ok((self.conv(&format!("{p}.model.{}.0.model.{j}.conv1", 2 + i), dil)?, self.conv(&format!("{p}.model.{}.0.model.{j}.conv2", 2 + i), 1)?))
                })
                .collect::<Result<_>>()?;
            levels.push(Level {
                target: self.lin(&format!("{p}.target_cond_blocks.{}", 2 * i))?,
                ext: self.lin(&format!("{p}.external_cond_blocks.{}", 2 * i))?,
                res,
                up: self.conv(&format!("{p}.model.{}.2", 2 + i), 1)?,
            });
        }
        Ok(Decoder { conv0: self.conv(&format!("{p}.model.0"), 1)?, levels, post: self.conv(&format!("{p}.model.4"), 1)?, out: self.conv(&format!("{p}.model.6"), 1)? })
    }
    /// The first `n` rows of a (seq, 1, dim) position table, as `T`.
    pub fn pos_table(&self, name: &str, n: usize) -> Result<Buf> {
        let v = self.vec(name)?;
        let rows = self.shape(name)?[0];
        ensure!(rows >= n, "{name}: {rows} positions, need {n}");
        Ok(self.gpu.upload_t(&v[..n * (v.len() / rows)]))
    }
}

fn cpu_linear(x: &[f32], w: &[f32], b: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|j| b[j] + (0..k).map(|i| x[i] * w[j * k + i]).sum::<f32>()).collect()
}

/// The pose network's token embedding in the release configuration (every token masked at the
/// single step): FCBlock(pose_token_emb[h * (codes + 1) + mask_id] for each head), split into 4
/// frames (640 = 4 x 160). Weights and every intermediate are rounded to `T` like the network.
pub(super) fn token_const(l: &Loader, prec: Precision, heads: usize, codes: usize, cw: usize) -> Result<Vec<f32>> {
    let emb = l.vec("pose.pose_token_emb.weight")?;
    ensure!(emb.len() >= heads * (codes + 1) * cw, "pose.pose_token_emb.weight: {} values, need {}", emb.len(), heads * (codes + 1) * cw);
    let tok: Vec<f32> = (0..heads).flat_map(|h| emb[(h * (codes + 1) + codes) * cw..(h * (codes + 1) + codes + 1) * cw].to_vec()).collect();
    let fc = |p: &str| -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let s = l.shape(&format!("{p}.weight"))?;
        ensure!(s.len() == 2, "{p}.weight: expected 2 dimensions, got {s:?}");
        Ok((l.vec(&format!("{p}.weight"))?, l.vec(&format!("{p}.bias"))?, s[0], s[1]))
    };
    let (w0, b0, n0, k0) = fc("pose.proj_pose_token_emb.fc_layers.0")?;
    let (w1, b1, n1, k1) = fc("pose.proj_pose_token_emb.fc_layers.1")?;
    let (w2, b2, n2, k2) = fc("pose.proj_pose_token_emb.forward_projection")?;
    ensure!(k0 == tok.len() && k1 == n0 && k2 == n1, "pose.proj_pose_token_emb: layer shapes do not chain");
    let rt = |v: Vec<f32>| prec.decode(&prec.encode(&v));
    let leaky_t = |v: Vec<f32>| v.into_iter().map(|x| prec.decode(&prec.encode(&[x * 0.01]))[0].max(x)).collect::<Vec<f32>>();
    let (w0, b0, w1, b1, w2, b2) = (rt(w0), rt(b0), rt(w1), rt(b1), rt(w2), rt(b2));
    let h0 = leaky_t(rt(cpu_linear(&rt(tok), &w0, &b0, n0, k0)));
    let h1 = leaky_t(rt(cpu_linear(&h0, &w1, &b1, n1, k1)));
    Ok(rt(cpu_linear(&h1, &w2, &b2, n2, k2)))
}

/// Rig tables of `model.metal`: G1 floats/ints (see `G1Off` there) or the SOMA T-pose offsets.
pub(super) struct Rig {
    pub f: Buf,
    pub i: Buf,
    pub soma_capture: Buf,
    pub soma_standard: Buf,
}

fn entry<'a>(a: &'a HashMap<String, Npy>, name: &str, len: usize) -> Result<&'a Npy> {
    let v = a.get(name).ok_or_else(|| anyhow!("rig asset: missing {name}"))?;
    let n: usize = v.shape.iter().map(|&d| d as usize).product();
    ensure!(n == len, "rig asset: {name} has {n} values, expected {len}");
    Ok(v)
}

pub(super) fn rig(gpu: &Gpu, spec: &ModelSpec) -> Result<Rig> {
    let a = load_npz_bytes(spec.rig_asset()).context("rig asset")?;
    let j = spec.num_joints as usize;
    Ok(match spec.kind {
        Kind::UnitreeG1 => {
            // G1Off in model.metal: 29 hinges, 30 MuJoCo bodies, 34 model joints.
            ensure!(j == 34, "G1 rig tables are laid out for 34 model joints, got {j}");
            let mapping = entry(&a, "mujoco_indices_to_motion_indices", 29)?.to_i32();
            let q2t = entry(&a, "rot_offsets_q2t", 34 * 9)?.to_f32();
            let mut q2t_mapped = Vec::with_capacity(29 * 9);
            for &m in &mapping {
                let m = usize::try_from(m).ok().filter(|&m| m < 34).ok_or_else(|| anyhow!("rig asset: motion index {m} out of range"))?;
                q2t_mapped.extend_from_slice(&q2t[m * 9..m * 9 + 9]);
            }
            let f: Vec<f32> = [
                entry(&a, "mujoco_to_motion_matrix", 9)?.to_f32(),
                entry(&a, "mujoco_joint_axis_values_mujoco_space", 29 * 3)?.to_f32(),
                q2t_mapped,
                entry(&a, "capture_neutral_joints_mujoco", 30 * 3)?.to_f32(),
                entry(&a, "neutral_joints", 34 * 3)?.to_f32(),
                entry(&a, "rot_offsets_f2q", 34 * 9)?.to_f32(),
                entry(&a, "mujoco_joint_axis_values_motion_space", 29 * 3)?.to_f32(),
            ]
            .concat();
            let i: Vec<i32> = [
                entry(&a, "mujoco_joint_including_root_parent_list", 30)?.to_i32(),
                entry(&a, "joint_parents", 34)?.to_i32(),
                entry(&a, "motion_indices_to_mujoco_indices", 34)?.to_i32(),
                mapping,
            ]
            .concat();
            Rig { f: gpu.upload(&f), i: gpu.upload_i32(&i), soma_capture: gpu.buffer(4), soma_standard: gpu.buffer(4) }
        }
        Kind::Soma => Rig {
            f: gpu.buffer(4),
            i: gpu.buffer_i32(4),
            soma_capture: gpu.upload(&entry(&a, "capture_offsets", j * 9)?.to_f32()),
            soma_standard: gpu.upload(&entry(&a, "standard_offsets", j * 9)?.to_f32()),
        },
    })
}
