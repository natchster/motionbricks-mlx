// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Rig geometry around the network (ported from `motionbricks_mlx/rig.py`): Unitree G1 forward
//! kinematics and joint-angle decoding, SOMA capture <-> standard T-pose retarget. Float32.

use std::collections::HashMap;

use mlx_rs::{Array, ops};

use crate::kernels::RigKernels;
use crate::npy::Npy;
use crate::ops::*;

pub fn cross(a: &Array, b: &Array) -> R<Array> {
    let (ax, ay, az) = (index(a, -1, 0)?, index(a, -1, 1)?, index(a, -1, 2)?);
    let (bx, by, bz) = (index(b, -1, 0)?, index(b, -1, 1)?, index(b, -1, 2)?);
    stack(
        &[&(&(&ay * &bz) - &(&az * &by)), &(&(&az * &bx) - &(&ax * &bz)), &(&(&ax * &by) - &(&ay * &bx))],
        -1,
    )
}

pub fn normalize(v: &Array) -> R<Array> {
    ops::divide(v, ops::sqrt((v * v).sum_axis(-1, true)?)?)
}

/// 6D rotation (first two columns) -> 3x3 matrix with columns (x, y, z).
pub fn cont6d_to_matrix(six: &Array) -> R<Array> {
    let x = normalize(&slice(six, -1, 0, 3)?)?;
    let z = normalize(&cross(&x, &slice(six, -1, 3, 6)?)?)?;
    let y = cross(&z, &x)?;
    stack(&[&x, &y, &z], -1)
}

pub fn matrix_to_cont6d(m: &Array) -> R<Array> {
    cat(&[&index(m, -1, 0)?, &index(m, -1, 1)?], -1)
}

/// WXYZ quaternion -> rotation matrix (normalizing like upstream's `2 / |q|^2`).
pub fn quat_matrix(q: &Array) -> R<Array> {
    let (r, x, y, z) = (index(q, -1, 0)?, index(q, -1, 1)?, index(q, -1, 2)?, index(q, -1, 3)?);
    let s = ops::divide(f(2.0), (q * q).sum_axis(-1, None)?)?;
    let one = f(1.0);
    let rows = [
        &one - &(&s * &(&(&y * &y) + &(&z * &z))),
        &s * &(&(&x * &y) - &(&z * &r)),
        &s * &(&(&x * &z) + &(&y * &r)),
        &s * &(&(&x * &y) + &(&z * &r)),
        &one - &(&s * &(&(&x * &x) + &(&z * &z))),
        &s * &(&(&y * &z) - &(&x * &r)),
        &s * &(&(&x * &z) - &(&y * &r)),
        &s * &(&(&y * &z) + &(&x * &r)),
        &one - &(&s * &(&(&x * &x) + &(&y * &y))),
    ];
    let refs: Vec<&Array> = rows.iter().collect();
    let mut shape = q.shape()[..q.ndim() - 1].to_vec();
    shape.extend([3, 3]);
    stack(&refs, -1)?.reshape(&shape)
}

/// Rotation matrix -> WXYZ quaternion, choosing the best-conditioned of 4 candidates.
pub fn matrix_quat(m: &Array) -> R<Array> {
    let e = |r, c| index2(m, r, c);
    let (m00, m01, m02) = (e(0, 0)?, e(0, 1)?, e(0, 2)?);
    let (m10, m11, m12) = (e(1, 0)?, e(1, 1)?, e(1, 2)?);
    let (m20, m21, m22) = (e(2, 0)?, e(2, 1)?, e(2, 2)?);
    let one = f(1.0);
    let cands = stack(
        &[
            &(&(&(&one + &m00) + &m11) + &m22),
            &(&(&(&one + &m00) - &m11) - &m22),
            &(&(&(&one - &m00) + &m11) - &m22),
            &(&(&(&one - &m00) - &m11) + &m22),
        ],
        -1,
    )?;
    let a = ops::sqrt(ops::maximum(&cands, i(0))?)?;
    let a2 = |k| ops::power(index(&a, -1, k)?, i(2));
    let rows = stack(
        &[
            &stack(&[&a2(0)?, &(&m21 - &m12), &(&m02 - &m20), &(&m10 - &m01)], -1)?,
            &stack(&[&(&m21 - &m12), &a2(1)?, &(&m10 + &m01), &(&m02 + &m20)], -1)?,
            &stack(&[&(&m02 - &m20), &(&m10 + &m01), &a2(2)?, &(&m12 + &m21)], -1)?,
            &stack(&[&(&m10 - &m01), &(&m20 + &m02), &(&m21 + &m12), &a2(3)?], -1)?,
        ],
        -2,
    )?;
    let rows = ops::divide(&rows, unsq(&(&ops::maximum(&a, f(0.1))? * f(2.0)), -1)?)?;
    let best = ops::indexing::argmax_axis(&a, -1, None)?;
    let best = unsq(&unsq(&best, -1)?, -1)?;
    let mut shape = a.shape()[..a.ndim() - 1].to_vec();
    shape.extend([1, 4]);
    index(&rows.take_along_axis(ops::broadcast_to(&best, &shape)?, -2)?, -2, 0)
}

pub fn rotate(m: &Array, p: &Array) -> R<Array> {
    index(&ops::matmul(m, unsq(p, -1)?)?, -1, 0)
}

/// Wrap radians to [-pi, pi).
pub fn wrap(angle: &Array) -> R<Array> {
    let pi = std::f32::consts::PI;
    Ok(&ops::remainder(angle + f(pi), f(2.0 * pi))? - f(pi))
}

fn take_rows(a: &Array, axis: i32, idx: usize) -> R<Array> {
    index(a, axis, idx as i32)
}

/// Unitree G1: MuJoCo joint coordinates <-> the model's 34-joint skeleton features.
pub struct G1Rig {
    basis: Array,
    axes: Array,
    model_axes: Array,
    f2q: Array,
    rest: Array,
    neutral: Array,
    parents: Vec<i32>,
    model_parents: Vec<i32>,
    reverse: Vec<i32>,
    mapping: Array,
    q2t_mapped: Array,
    rotation_indices: Array,
    root_indices: Array,
    contact_indices: Array,
    parent_indices: Array,
    identity: Array,
    num_joints: i32,
    rf: Array,
    ri: Array,
    kernels: RigKernels,
}

impl G1Rig {
    /// `arrays`: upstream `g1_model_rig.npz`; `indices`: global-layout ranges by key.
    pub fn new(arrays: &HashMap<String, Npy>, rotation: &[i32], root_pos: &[i32], contacts: &[i32]) -> anyhow::Result<Self> {
        let get = |k: &str| arrays.get(k).ok_or_else(|| anyhow::anyhow!("rig asset lacks {k}"));
        let model_parents = get("joint_parents")?.to_i32();
        let mapping = get("mujoco_indices_to_motion_indices")?.i32_array();
        let q2t = get("rot_offsets_q2t")?.f32_array();
        let idx = |v: &[i32]| Array::from_slice(v, &[v.len() as i32]);
        // Packed constants for the rig kernels (layout in rig_kernels.metal HEADER).
        let q2t_v = get("rot_offsets_q2t")?.to_f32();
        let map_v = get("mujoco_indices_to_motion_indices")?.to_i32();
        let q2t_mapped: Vec<f32> = map_v.iter().flat_map(|&m| q2t_v[m as usize * 9..m as usize * 9 + 9].to_vec()).collect();
        let rf: Vec<f32> = [
            get("mujoco_to_motion_matrix")?.to_f32(),
            get("mujoco_joint_axis_values_mujoco_space")?.to_f32(),
            q2t_mapped,
            get("capture_neutral_joints_mujoco")?.to_f32(),
            get("neutral_joints")?.to_f32(),
            get("rot_offsets_f2q")?.to_f32(),
            get("mujoco_joint_axis_values_motion_space")?.to_f32(),
        ]
        .concat();
        let ri: Vec<i32> = [
            get("mujoco_joint_including_root_parent_list")?.to_i32(),
            model_parents.clone(),
            get("motion_indices_to_mujoco_indices")?.to_i32(),
            map_v.clone(),
        ]
        .concat();
        Ok(G1Rig {
            rf: Array::from_slice(&rf, &[rf.len() as i32]),
            ri: idx(&ri),
            kernels: RigKernels::new()?,
            basis: get("mujoco_to_motion_matrix")?.f32_array(),
            axes: get("mujoco_joint_axis_values_mujoco_space")?.f32_array(),
            model_axes: get("mujoco_joint_axis_values_motion_space")?.f32_array(),
            f2q: get("rot_offsets_f2q")?.f32_array(),
            rest: get("capture_neutral_joints_mujoco")?.f32_array(),
            neutral: get("neutral_joints")?.f32_array(),
            parents: get("mujoco_joint_including_root_parent_list")?.to_i32(),
            reverse: get("motion_indices_to_mujoco_indices")?.to_i32(),
            q2t_mapped: q2t.take_axis(&mapping, 0)?,
            mapping,
            rotation_indices: idx(rotation),
            root_indices: idx(root_pos),
            contact_indices: idx(contacts),
            parent_indices: idx(&model_parents[1..]),
            identity: Array::eye::<f32>(3, None, None)?,
            num_joints: model_parents.len() as i32,
            model_parents,
        })
    }

    /// `qpos (..., 36)` with a WXYZ root quaternion -> joint positions `(..., 34, 3)` and
    /// rotations `(..., 34, 3, 3)` in model space.
    pub fn fk(&self, qpos: &Array) -> R<(Array, Array)> {
        let dofs = slice(qpos, -1, 7, END)?;
        let half = &dofs / f(2.0);
        let quats = cat(&[&unsq(&cos(&half)?, -1)?, &(&unsq(&sin(&half)?, -1)? * &self.axes)], -1)?;
        let local = ops::matmul(&self.q2t_mapped, quat_matrix(&quats)?)?;
        let mut rotations = vec![quat_matrix(&slice(qpos, -1, 3, 7)?)?];
        let mut positions = vec![ops::add(slice(qpos, -1, 0, 3)?, take_rows(&self.rest, 0, 0)?)?];
        for joint in 1..self.parents.len() {
            let parent = self.parents[joint] as usize;
            let offset = &take_rows(&self.rest, 0, joint)? - &take_rows(&self.rest, 0, parent)?;
            positions.push(ops::add(&positions[parent], rotate(&rotations[parent], &offset)?)?);
            rotations.push(ops::matmul(&rotations[parent], index(&local, -3, joint as i32 - 1)?)?);
        }
        // Change of basis for all 30 robot joints at once.
        let p_all = rotate(&self.basis, &stack(&positions.iter().collect::<Vec<_>>(), -2)?)?;
        let r_all = ops::matmul(ops::matmul(&self.basis, stack(&rotations.iter().collect::<Vec<_>>(), -3)?)?, self.basis.t())?;
        let mut model_p: Vec<Array> = Vec::new();
        let mut model_r: Vec<Array> = Vec::new();
        for (joint, &robot_index) in self.reverse.iter().enumerate() {
            if robot_index >= 0 {
                model_p.push(index(&p_all, -2, robot_index)?);
                model_r.push(index(&r_all, -3, robot_index)?);
            } else {
                let parent = self.model_parents[joint] as usize;
                let offset = &take_rows(&self.neutral, 0, joint)? - &take_rows(&self.neutral, 0, parent)?;
                let p = ops::add(&model_p[parent], rotate(&model_r[parent], &offset)?)?;
                let r = ops::broadcast_to(&self.identity, model_r[parent].shape())?;
                model_p.push(p);
                model_r.push(r);
            }
        }
        Ok((stack(&model_p.iter().collect::<Vec<_>>(), -2)?, stack(&model_r.iter().collect::<Vec<_>>(), -3)?))
    }

    /// Global features `(B, T, D)` -> root position, root rotation, 29 joint angles.
    fn decode(&self, features: &Array) -> R<(Array, Array, Array)> {
        let (b, t) = (features.dim(0), features.dim(1));
        let r = cont6d_to_matrix(&features.take_axis(&self.rotation_indices, -1)?.reshape(&[b, t, self.num_joints, 6])?)?;
        let first = slice(&r, -3, 0, 1)?;
        let parents = cat(&[&ops::broadcast_to(&self.identity, first.shape())?, &r.take_axis(&self.parent_indices, -3)?], -3)?;
        let local = ops::matmul(&self.f2q, ops::matmul(parents.swap_axes(-1, -2)?, &r)?)?;
        let root_p = rotate(&self.basis.t(), &features.take_axis(&self.root_indices, -1)?)?;
        let root_r = ops::matmul(ops::matmul(self.basis.t(), index(&local, -3, 0)?)?, &self.basis)?;
        let hinge = local.take_axis(&self.mapping, -3)?;
        let angles = stack(
            &[
                &atan2(&index2(&hinge, 2, 1)?, &index2(&hinge, 2, 2)?)?,
                &atan2(&index2(&hinge, 0, 2)?, &index2(&hinge, 0, 0)?)?,
                &atan2(&index2(&hinge, 1, 0)?, &index2(&hinge, 1, 1)?)?,
            ],
            -1,
        )?;
        Ok((root_p, root_r, (&angles * &self.model_axes).sum_axis(-1, None)?))
    }

    /// Placed G1 poses -> `(global_root (B,8,5), local_root (B,8,4), local_poses (B,8,303))`, on the
    /// rig kernel (one thread per sequence).
    pub fn preprocess(&self, context_qpos: &Array, target_qpos: &Array) -> R<(Array, Array, Array)> {
        let mut o = self.kernels.g1_preprocess(context_qpos, target_qpos, &self.rf, &self.ri)?;
        let lp = o.pop().unwrap();
        let lr = o.pop().unwrap();
        Ok((o.pop().unwrap(), lr, lp))
    }

    /// Op-based reference of [`Self::preprocess`].
    pub fn preprocess_ops(&self, context_qpos: &Array, target_qpos: &Array) -> R<(Array, Array, Array)> {
        let poses = cat(&[context_qpos, target_qpos], 1)?;
        let poses = cat(
            &[&slice(&poses, -1, 0, 3)?, &slice(&poses, -1, 6, 7)?, &slice(&poses, -1, 3, 6)?, &slice(&poses, -1, 7, END)?],
            -1,
        )?; // XYZW -> WXYZ
        let (p, r) = self.fk(&poses)?;
        let root = index(&p, 2, 0)?; // (B, 8, 3)
        let r_root = index(&r, 2, 0)?; // (B, 8, 3, 3)
        let heading = atan2(&index2(&r_root, 0, 2)?, &index2(&r_root, 2, 2)?)?;
        let global_root = cat(&[&root, &unsq(&cos(&heading)?, -1)?, &unsq(&sin(&heading)?, -1)?], -1)?;
        let mut windows = Vec::new();
        for start in [0, 4] {
            let yaw_vel = &wrap(&(&slice(&heading, 1, start + 1, start + 4)? - &slice(&heading, 1, start, start + 3)?))? * i(30);
            let xz = stack(&[&index(&root, -1, 0)?, &index(&root, -1, 2)?], -1)?;
            let velocity = &(&slice(&xz, 1, start + 1, start + 4)? - &slice(&xz, 1, start, start + 3)?) * i(30);
            let height = |a, b| slice(&slice(&root, 1, a, b)?, -1, 1, 2);
            let first = cat(&[&unsq(&yaw_vel, -1)?, &velocity, &height(start, start + 3)?], -1)?;
            let last = cat(&[&ops::zeros_like(&slice(&slice(&first, 1, 0, 1)?, -1, 0, 3)?)?, &height(start + 3, start + 4)?], -1)?;
            windows.push(cat(&[&first, &last], 1)?);
        }
        let nonroot = &slice(&p, 2, 1, END)? - &(&unsq(&root, 2)? * Array::from_slice(&[1.0f32, 0.0, 1.0], &[3]));
        let six = cat(&[&index(&r, -1, 0)?, &index(&r, -1, 1)?], -1)?;
        let b = poses.dim(0);
        let local_poses = cat(&[&nonroot.reshape(&[b, 8, -1])?, &six.reshape(&[b, 8, -1])?], -1)?;
        Ok((global_root, cat(&[&windows[0], &windows[1]], 1)?, local_poses))
    }

    /// Global features -> `(pred_qpos (B,T,36) XYZW root quaternion, foot contacts (B,T,4))`, on the
    /// rig kernel (one thread per frame).
    pub fn postprocess(&self, pred_global: &Array) -> R<(Array, Array)> {
        let mut o = self.kernels.g1_postprocess(pred_global, &self.rf, &self.ri)?;
        let contacts = o.pop().unwrap();
        Ok((o.pop().unwrap(), contacts))
    }

    /// Op-based reference of [`Self::postprocess`].
    pub fn postprocess_ops(&self, pred_global: &Array) -> R<(Array, Array)> {
        let (root_p, root_r, hinges) = self.decode(pred_global)?;
        let q = matrix_quat(&root_r)?;
        let quat_xyzw = cat(&[&slice(&q, -1, 1, 4)?, &slice(&q, -1, 0, 1)?], -1)?;
        Ok((cat(&[&root_p, &quat_xyzw, &hinges], -1)?, pred_global.take_axis(&self.contact_indices, -1)?))
    }
}

/// Moves global joint rotations between the capture and the standard T pose (SOMA).
pub struct SomaRetarget {
    capture: Array,
    standard: Array,
    num_joints: i32,
    kernels: RigKernels,
}

impl SomaRetarget {
    pub fn new(arrays: &HashMap<String, Npy>, num_joints: i32) -> anyhow::Result<Self> {
        let get = |k: &str| arrays.get(k).ok_or_else(|| anyhow::anyhow!("rig asset lacks {k}"));
        Ok(SomaRetarget { capture: get("capture_offsets")?.f32_array(), standard: get("standard_offsets")?.f32_array(), num_joints, kernels: RigKernels::new()? })
    }

    fn apply(&self, six_flat: &Array, from: &Array, to: &Array) -> R<Array> {
        self.kernels.soma_retarget(six_flat, from, to, self.num_joints)
    }

    /// Op-based reference of the retarget.
    pub fn apply_ops(&self, six_flat: &Array, from: &Array, to: &Array) -> R<Array> {
        let shape = six_flat.shape().to_vec();
        let mut joint_shape = shape[..shape.len() - 1].to_vec();
        joint_shape.extend([self.num_joints, 6]);
        let m = cont6d_to_matrix(&six_flat.reshape(&joint_shape)?)?;
        let m = ops::matmul(ops::matmul(&m, from)?, to.swap_axes(-1, -2)?)?;
        matrix_to_cont6d(&m)?.reshape(&shape)
    }

    pub fn offsets(&self) -> (&Array, &Array) {
        (&self.capture, &self.standard)
    }

    pub fn capture_to_standard(&self, six_flat: &Array) -> R<Array> {
        self.apply(six_flat, &self.capture, &self.standard)
    }

    pub fn standard_to_capture(&self, six_flat: &Array) -> R<Array> {
        self.apply(six_flat, &self.standard, &self.capture)
    }
}
