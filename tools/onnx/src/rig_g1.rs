// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Unitree G1 rig emitter (upstream `tools/onnx/models/g1/rig.py`): MuJoCo qpos <->
//! model motion features. This is an op-for-op port of the `G1Rig::*_ops` reference
//! implementations in `model-rs/src/rig.rs`, except that every function appends ONNX
//! nodes to the graph instead of computing values. All tensor dimensions except the
//! leading batch are static and resolved here at emit time.
//!
//! Constant naming follows the traced modules: rig matrices live under `pre.` / `post.`,
//! shared matrices under `rig.`. The per-op index tensors (`Slice` bounds, `Gather`
//! indices, `Reshape` targets) get unique names from [`Graph::fresh`] because the fixed
//! `ops::c_vec` names would collide when a helper is used twice under one scope.

use anyhow::{Context, Result, ensure};

use crate::config::{CONTEXT_AND_TARGET, FPS, FRAMES_PER_TOKEN, G1_QPOS_DIM, ModelSpec};
use crate::graph::{Attr, Dim, Graph};
use crate::npy::load_npz_bytes;
use crate::ops::{self, c_f32, c_scalar};

/// MuJoCo joints (1 root + 29 hinges) and model skeleton joints of the G1 rig.
const NUM_MUJOCO: usize = 30;
const NUM_MODEL: usize = 34;
const NUM_HINGES: usize = 29;

// Layout of the 414-dim global motion features (upstream `representation.py`,
// `GlobalRootGlobalJoints` with 34 joints): root keys first (global_root_pos 3,
// global_root_heading 2), then the body keys in declaration order (ric_data 99,
// global_rot_data 204, local_vel 102, foot_contacts 4).
const GLOBAL_ROOT_DIM: i64 = 5;
const RIC_DIM: i64 = (NUM_MODEL as i64 - 1) * 3; // 99
const ROT6_DIM: i64 = NUM_MODEL as i64 * 6; // 204
const ROT6_START: i64 = GLOBAL_ROOT_DIM + RIC_DIM; // 104

/// The G1 rig: names of the constant initializers emitted at load plus the static index
/// tables (parent lists, index mappings) the emitters unroll over.
pub struct G1Rig {
    basis: String,
    basis_t: String,
    axes: String,
    q2t_mapped: String,
    rest0: String,
    fk_offset: Vec<String>,
    identity: String,
    horizontal: String,
    f2q: String,
    model_axes: String,
    parents: Vec<i64>,
    model_parents: Vec<i64>,
    reverse: Vec<i64>,
    mapping: Vec<i64>,
    parent_indices: Vec<i64>,
    voffset: Vec<Option<String>>,
    rotation_indices: Vec<i64>,
    root_indices: Vec<i64>,
}

impl G1Rig {
    /// Load the rig asset (`spec.rig_asset()`, upstream `g1_model_rig.npz`) and emit all
    /// constant initializers: basis change matrices, hinge axes (MuJoCo and motion
    /// space), the mapped q2t/f2q rotation offsets, the per-joint FK/rest offsets, and
    /// the identity matrix the four virtual joints expand.
    pub fn load(g: &mut Graph, spec: &ModelSpec) -> Result<G1Rig> {
        let arrays = load_npz_bytes(spec.rig_asset()).context("reading the G1 rig asset")?;
        let get = |key: &str| arrays.get(key).with_context(|| format!("rig asset lacks {key}"));
        let basis = get("mujoco_to_motion_matrix")?.to_f32();
        let axes = get("mujoco_joint_axis_values_mujoco_space")?.to_f32();
        let model_axes = get("mujoco_joint_axis_values_motion_space")?.to_f32();
        let q2t = get("rot_offsets_q2t")?.to_f32();
        let f2q = get("rot_offsets_f2q")?.to_f32();
        // `neutral_joints` is float64 in the asset; everything in the graph is f32.
        let rest = get("capture_neutral_joints_mujoco")?.to_f32();
        let neutral = get("neutral_joints")?.to_f32();
        let parents = get("mujoco_joint_including_root_parent_list")?.to_i32();
        let model_parents = get("joint_parents")?.to_i32();
        let reverse = get("motion_indices_to_mujoco_indices")?.to_i32();
        let mapping = get("mujoco_indices_to_motion_indices")?.to_i32();
        ensure!(basis.len() == 9 && axes.len() == NUM_HINGES * 3 && model_axes.len() == NUM_HINGES * 3);
        ensure!(q2t.len() == NUM_MODEL * 9 && f2q.len() == NUM_MODEL * 9);
        ensure!(rest.len() == NUM_MUJOCO * 3 && neutral.len() == NUM_MODEL * 3);
        ensure!(parents.len() == NUM_MUJOCO && model_parents.len() == NUM_MODEL && reverse.len() == NUM_MODEL);
        ensure!(mapping.len() == NUM_HINGES);

        let basis_t: Vec<f32> = (0..9).map(|i| basis[(i % 3) * 3 + i / 3]).collect();
        // q2t[mapping]: the per-hinge rotation offsets, in MuJoCo joint order.
        let q2t_mapped: Vec<f32> = mapping.iter().flat_map(|&m| q2t[m as usize * 9..m as usize * 9 + 9].to_vec()).collect();

        let basis = c_f32(g, "rig.basis", &[3, 3], &basis);
        let basis_t = c_f32(g, "rig.basis_t", &[3, 3], &basis_t);
        let axes = c_f32(g, "pre.axes", &[NUM_HINGES as i64, 3], &axes);
        let q2t_mapped = c_f32(g, "pre.q2t_mapped", &[NUM_HINGES as i64, 3, 3], &q2t_mapped);
        let rest0 = c_f32(g, "pre.rest0", &[3], &rest[..3]);
        let identity = c_f32(g, "rig.identity", &[3, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let horizontal = c_f32(g, "pre.horizontal", &[3], &[1.0, 0.0, 1.0]);
        let f2q = c_f32(g, "post.f2q", &[NUM_MODEL as i64, 3, 3], &f2q);
        let model_axes = c_f32(g, "post.model_axes", &[NUM_HINGES as i64, 3], &model_axes);

        // FK link offsets in MuJoCo space: rest[joint] - rest[parent], one [3] constant
        // per non-root MuJoCo joint (constant-folded, like upstream's traced buffers).
        let mut fk_offset = Vec::with_capacity(NUM_MUJOCO - 1);
        for joint in 1..NUM_MUJOCO {
            let parent = parents[joint] as usize;
            let offset: Vec<f32> = (0..3).map(|c| rest[joint * 3 + c] - rest[parent * 3 + c]).collect();
            fk_offset.push(c_f32(g, &format!("pre.fk_offset_{joint}"), &[3], &offset));
        }
        // Rest offsets of the virtual joints (neutral[joint] - neutral[parent], model space).
        let mut voffset: Vec<Option<String>> = vec![None; NUM_MODEL];
        for joint in 0..NUM_MODEL {
            if reverse[joint] < 0 {
                let parent = model_parents[joint] as usize;
                let offset: Vec<f32> = (0..3).map(|c| neutral[joint * 3 + c] - neutral[parent * 3 + c]).collect();
                voffset[joint] = Some(c_f32(g, &format!("pre.voffset_{joint}"), &[3], &offset));
            }
        }

        let parents: Vec<i64> = parents.iter().map(|&p| p as i64).collect();
        let model_parents: Vec<i64> = model_parents.iter().map(|&p| p as i64).collect();
        let reverse: Vec<i64> = reverse.iter().map(|&p| p as i64).collect();
        let mapping: Vec<i64> = mapping.iter().map(|&p| p as i64).collect();
        Ok(G1Rig {
            basis,
            basis_t,
            axes,
            q2t_mapped,
            rest0,
            fk_offset,
            identity,
            horizontal,
            f2q,
            model_axes,
            parent_indices: model_parents[1..].to_vec(),
            parents,
            model_parents,
            reverse,
            mapping,
            voffset,
            rotation_indices: (ROT6_START..ROT6_START + ROT6_DIM).collect(),
            root_indices: vec![0, 1, 2],
        })
    }

    /// qpos context+target `[B, 8, 36]` (root xyz, root quaternion XYZW, 29 hinges) ->
    /// `(global_root [B,8,5], local_root [B,8,4], local_poses [B,8,303])`.
    ///
    /// 303 = 99 root-relative horizontal-projected joint positions + 204 6D rotations.
    /// `scope` is the node-name prefix (`/pre`).
    pub fn emit_preprocess(&self, g: &mut Graph, scope: &str, qpos: &str) -> Result<(String, String, String)> {
        debug_assert_eq!(g.shape(qpos), &[Dim::Batch, Dim::Static(CONTEXT_AND_TARGET), Dim::Static(G1_QPOS_DIM)]);
        let last = g.shape(qpos).len() as i64 - 1;
        // XYZW -> WXYZ root quaternion.
        let pos = slice_at(g, scope, qpos, last, 0, 3);
        let qw = slice_at(g, scope, qpos, last, 6, 7);
        let qxyz = slice_at(g, scope, qpos, last, 3, 6);
        let dofs = slice_at(g, scope, qpos, last, 7, G1_QPOS_DIM);
        let poses = ops::concat(g, scope, &[&pos, &qw, &qxyz, &dofs], last);
        let (p, r) = self.emit_fk(g, scope, &poses);

        let root_s = slice_at(g, scope, &p, 2, 0, 1);
        let root = squeeze_at(g, scope, &root_s, &[2]); // [B,8,3]
        let r_root_s = slice_at(g, scope, &r, 2, 0, 1);
        let r_root = squeeze_at(g, scope, &r_root_s, &[2]); // [B,8,3,3]
        let h_y = mat_entry(g, scope, &r_root, 0, 2);
        let h_x = mat_entry(g, scope, &r_root, 2, 2);
        let heading = ops::atan2(g, scope, &h_y, &h_x); // [B,8]
        let cos_h = ops::cos(g, scope, &heading);
        let cos_h = unsqueeze_at(g, scope, &cos_h, &[last]);
        let sin_h = ops::sin(g, scope, &heading);
        let sin_h = unsqueeze_at(g, scope, &sin_h, &[last]);
        let global_root = ops::concat(g, scope, &[&root, &cos_h, &sin_h], last); // [B,8,5]

        // local_root: per 4-frame window (yaw velocity, 2D root velocity) at 30 fps and
        // the hip height; the last frame of each window carries zero velocity.
        let fps = c_scalar(g, scope, FPS as f32);
        let xz = gather_rows(g, scope, &root, &[0, 2], last); // [B,8,2]
        let mut windows = Vec::new();
        for start in [0, FRAMES_PER_TOKEN] {
            let h_next = slice_at(g, scope, &heading, 1, start + 1, start + FRAMES_PER_TOKEN);
            let h_prev = slice_at(g, scope, &heading, 1, start, start + FRAMES_PER_TOKEN - 1);
            let h_diff = ops::sub(g, scope, &h_next, &h_prev);
            let yaw_wrapped = wrap_angle(g, scope, &h_diff);
            let yaw_vel = ops::mul(g, scope, &yaw_wrapped, &fps); // [B,3]
            let v_next = slice_at(g, scope, &xz, 1, start + 1, start + FRAMES_PER_TOKEN);
            let v_prev = slice_at(g, scope, &xz, 1, start, start + FRAMES_PER_TOKEN - 1);
            let v_diff = ops::sub(g, scope, &v_next, &v_prev);
            let velocity = ops::mul(g, scope, &v_diff, &fps); // [B,3,2]
            let height_a = slice_at(g, scope, &root, 1, start, start + FRAMES_PER_TOKEN - 1);
            let height_a = slice_at(g, scope, &height_a, last, 1, 2); // [B,3,1]
            let yaw_col = unsqueeze_at(g, scope, &yaw_vel, &[last]);
            let first = ops::concat(g, scope, &[&yaw_col, &velocity, &height_a], last); // [B,3,4]
            // zeros_like(first[:, :1, :3]): Shape + ConstantOfShape, as the tracer emits.
            let first_row = slice_at(g, scope, &first, 1, 0, 1);
            let first_vel = slice_at(g, scope, &first_row, last, 0, 3);
            let vel_shape = ops::shape_of(g, scope, &first_vel);
            let zeros = ops::constant_of_shape(g, scope, &vel_shape, 0.0, &[Dim::Batch, Dim::Static(1), Dim::Static(3)]);
            let height_b = slice_at(g, scope, &root, 1, start + FRAMES_PER_TOKEN - 1, start + FRAMES_PER_TOKEN);
            let height_b = slice_at(g, scope, &height_b, last, 1, 2); // [B,1,1]
            let last_frame = ops::concat(g, scope, &[&zeros, &height_b], last); // [B,1,4]
            windows.push(ops::concat(g, scope, &[&first, &last_frame], 1)); // [B,4,4]
        }
        let local_root = ops::concat(g, scope, &[&windows[0], &windows[1]], 1); // [B,8,4]

        // local_poses: root-relative horizontal joint positions + the first two columns
        // of every joint rotation (6D), flattened.
        let nonroot_all = slice_at(g, scope, &p, 2, 1, NUM_MODEL as i64); // [B,8,33,3]
        let root_u = unsqueeze_at(g, scope, &root, &[2]);
        let root_h = ops::mul(g, scope, &root_u, &self.horizontal);
        let nonroot = ops::sub(g, scope, &nonroot_all, &root_h);
        let nonroot = reshape_to(g, scope, &nonroot, &[Dim::Batch, Dim::Static(CONTEXT_AND_TARGET), Dim::Static(RIC_DIM)]);
        let col0 = mat_col(g, scope, &r, 0);
        let col1 = mat_col(g, scope, &r, 1);
        let six = ops::concat(g, scope, &[&col0, &col1], last + 1); // [B,8,34,6]
        let six = reshape_to(g, scope, &six, &[Dim::Batch, Dim::Static(CONTEXT_AND_TARGET), Dim::Static(ROT6_DIM)]);
        let local_poses = ops::concat(g, scope, &[&nonroot, &six], last); // [B,8,303]
        Ok((global_root, local_root, local_poses))
    }

    /// Global poses `[B, 64, 414]` -> `pred_qpos [B, 64, 36]` (root xyz, root quaternion
    /// back in XYZW, 29 hinge angles). `scope` is the node-name prefix (`/post`). The
    /// caller slices the foot contacts off the 414-dim features itself.
    pub fn emit_postprocess(&self, g: &mut Graph, scope: &str, global_poses: &str) -> Result<String> {
        debug_assert_eq!(g.shape(global_poses).len(), 3);
        // decode: 6D rotations -> per-joint matrices, parent-relative local rotations.
        let rot6 = gather_rows(g, scope, global_poses, &self.rotation_indices, 2); // [B,64,204]
        let rot6 = reshape_to(g, scope, &rot6, &[Dim::Batch, Dim::Static(64), Dim::Static(NUM_MODEL as i64), Dim::Static(6)]);
        let r = six_matrix(g, scope, &rot6); // [B,64,34,3,3], columns (x, y, z)
        let id4 = unsqueeze_at(g, scope, &self.identity, &[0, 1]);
        let idb = expand_batch(g, scope, &id4, global_poses, &[64, 1, 3, 3]); // identity for the root "parent"
        let parents_r = gather_rows(g, scope, &r, &self.parent_indices, 2); // [B,64,33,3,3]
        let parents = ops::concat(g, scope, &[&idb, &parents_r], 2); // [B,64,34,3,3]
        let parents_t = ops::transpose(g, scope, &parents, &[0, 1, 2, 4, 3]);
        let pr = ops::matmul(g, scope, &parents_t, &r);
        let local = ops::matmul(g, scope, &self.f2q, &pr); // [B,64,34,3,3]

        let root_feat = gather_rows(g, scope, global_poses, &self.root_indices, 2); // [B,64,3]
        let root_p = rotate(g, scope, &self.basis_t, &root_feat); // [B,64,3]
        let local0_s = slice_at(g, scope, &local, 2, 0, 1);
        let local0 = squeeze_at(g, scope, &local0_s, &[2]); // [B,64,3,3]
        let bt_l = ops::matmul(g, scope, &self.basis_t, &local0);
        let root_r = ops::matmul(g, scope, &bt_l, &self.basis); // [B,64,3,3]

        // Hinge angles: three atan2 decompositions of each hinge's local matrix, dotted
        // with the motion-space axes.
        let hinge = gather_rows(g, scope, &local, &self.mapping, 2); // [B,64,29,3,3]
        let a0_y = mat_entry(g, scope, &hinge, 2, 1);
        let a0_x = mat_entry(g, scope, &hinge, 2, 2);
        let a0 = ops::atan2(g, scope, &a0_y, &a0_x);
        let a1_y = mat_entry(g, scope, &hinge, 0, 2);
        let a1_x = mat_entry(g, scope, &hinge, 0, 0);
        let a1 = ops::atan2(g, scope, &a1_y, &a1_x);
        let a2_y = mat_entry(g, scope, &hinge, 1, 0);
        let a2_x = mat_entry(g, scope, &hinge, 1, 1);
        let a2 = ops::atan2(g, scope, &a2_y, &a2_x);
        let angles = stack(g, scope, &[a0, a1, a2], 3); // [B,64,29,3]
        let scaled = ops::mul(g, scope, &angles, &self.model_axes);
        let hinges = reduce_sum_at(g, scope, &scaled, &[3], false); // [B,64,29]

        let q = matrix_quat(g, scope, &root_r, global_poses); // [B,64,4] WXYZ
        let q_xyz = slice_at(g, scope, &q, 2, 1, 4);
        let q_w = slice_at(g, scope, &q, 2, 0, 1);
        let quat_xyzw = ops::concat(g, scope, &[&q_xyz, &q_w], 2);
        Ok(ops::concat(g, scope, &[&root_p, &quat_xyzw, &hinges], 2)) // [B,64,36]
    }

    /// Forward kinematics of `poses` (`[B,8,36]`, WXYZ root quaternion): joint positions
    /// `[B,8,34,3]` and rotations `[B,8,34,3,3]` in model space. The 30-step chain over
    /// the MuJoCo parent list is unrolled as batched 3x3 matrix math; the four virtual
    /// joints get identity rotations and positions from the neutral offsets.
    fn emit_fk(&self, g: &mut Graph, scope: &str, poses: &str) -> (String, String) {
        let last = g.shape(poses).len() as i64 - 1;
        let two = c_scalar(g, scope, 2.0);
        let dofs = slice_at(g, scope, poses, last, 7, G1_QPOS_DIM); // [B,8,29]
        let half = ops::div(g, scope, &dofs, &two);
        let cos_h = ops::cos(g, scope, &half);
        let cos_h = unsqueeze_at(g, scope, &cos_h, &[last + 1]);
        let sin_h = ops::sin(g, scope, &half);
        let sin_h = unsqueeze_at(g, scope, &sin_h, &[last + 1]);
        let vec = ops::mul(g, scope, &sin_h, &self.axes);
        let quats = ops::concat(g, scope, &[&cos_h, &vec], last + 1); // [B,8,29,4]
        let qm = quat_matrix(g, scope, &quats); // [B,8,29,3,3]
        let local = ops::matmul(g, scope, &self.q2t_mapped, &qm); // [B,8,29,3,3]

        let root_q = slice_at(g, scope, poses, last, 3, 7);
        let mut rotations = vec![quat_matrix(g, scope, &root_q)];
        let root_p = slice_at(g, scope, poses, last, 0, 3);
        let mut positions = vec![ops::add(g, scope, &root_p, &self.rest0)];
        for joint in 1..NUM_MUJOCO {
            let parent = self.parents[joint] as usize;
            let step = rotate(g, scope, &rotations[parent], &self.fk_offset[joint - 1]);
            positions.push(ops::add(g, scope, &positions[parent], &step));
            let local_j_s = slice_at(g, scope, &local, 2, joint as i64 - 1, joint as i64);
            let local_j = squeeze_at(g, scope, &local_j_s, &[2]);
            rotations.push(ops::matmul(g, scope, &rotations[parent], &local_j));
        }
        // Change of basis (MuJoCo -> motion space) for all 30 robot joints at once.
        let p_stack = stack(g, scope, &positions, 2); // [B,8,30,3]
        let p_all = rotate(g, scope, &self.basis, &p_stack);
        let r_stack = stack(g, scope, &rotations, 2); // [B,8,30,3,3]
        let r_b = ops::matmul(g, scope, &self.basis, &r_stack);
        let r_all = ops::matmul(g, scope, &r_b, &self.basis_t);

        // Model joints: robot joints gather from the FK result; virtual joints extend
        // their parent with a constant offset and the identity rotation.
        let id4 = unsqueeze_at(g, scope, &self.identity, &[0, 1]);
        let idb = expand_batch(g, scope, &id4, poses, &[CONTEXT_AND_TARGET, 3, 3]);
        let mut model_p: Vec<String> = Vec::with_capacity(NUM_MODEL);
        let mut model_r: Vec<String> = Vec::with_capacity(NUM_MODEL);
        for joint in 0..NUM_MODEL {
            let robot = self.reverse[joint];
            if robot >= 0 {
                let pj = slice_at(g, scope, &p_all, 2, robot, robot + 1);
                model_p.push(squeeze_at(g, scope, &pj, &[2]));
                let rj = slice_at(g, scope, &r_all, 2, robot, robot + 1);
                model_r.push(squeeze_at(g, scope, &rj, &[2]));
            } else {
                let parent = self.model_parents[joint] as usize;
                let offset = self.voffset[joint].as_ref().expect("virtual joint offset");
                let step = rotate(g, scope, &model_r[parent], offset);
                model_p.push(ops::add(g, scope, &model_p[parent], &step));
                model_r.push(idb.clone());
            }
        }
        (stack(g, scope, &model_p, 2), stack(g, scope, &model_r, 2))
    }
}

// ---- local emit helpers ----
//
// The shape/index constants these create are named with `Graph::fresh` so repeated use
// under one scope cannot collide (the fixed `ops::c_vec` hint names would).

/// A uniquely-named rank-1 int64 constant (Slice bounds, axes, index lists).
fn ivec(g: &mut Graph, scope: &str, values: &[i64]) -> String {
    let name = g.fresh(&format!("{scope}/iv"));
    ops::c_i64(g, &name, &[values.len() as i64], values)
}

/// `Slice` with static bounds on one axis (`ops::slice_axis` with unique constants).
fn slice_at(g: &mut Graph, scope: &str, x: &str, axis: i64, start: i64, end: i64) -> String {
    let starts = ivec(g, scope, &[start]);
    let ends = ivec(g, scope, &[end]);
    let axes = ivec(g, scope, &[axis]);
    let steps = ivec(g, scope, &[1]);
    let mut out = g.shape(x).to_vec();
    out[axis as usize] = Dim::Static(end - start);
    let dtype = g.dtype(x);
    g.node(scope, "Slice", &[x, &starts, &ends, &axes, &steps], &[], dtype, &out)
}

/// `Unsqueeze` at `axes` (sorted).
fn unsqueeze_at(g: &mut Graph, scope: &str, x: &str, axes: &[i64]) -> String {
    let mut out = g.shape(x).to_vec();
    for &ax in axes {
        out.insert(ax as usize, Dim::Static(1));
    }
    let axes_t = ivec(g, scope, axes);
    let dtype = g.dtype(x);
    g.node(scope, "Unsqueeze", &[x, &axes_t], &[], dtype, &out)
}

/// `Squeeze` at `axes` (the dims must be statically 1).
fn squeeze_at(g: &mut Graph, scope: &str, x: &str, axes: &[i64]) -> String {
    let removed: std::collections::HashSet<i64> = axes.iter().copied().collect();
    let out: Vec<Dim> = g
        .shape(x)
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(&(*i as i64)))
        .map(|(_, d)| d.clone())
        .collect();
    let axes_t = ivec(g, scope, axes);
    let dtype = g.dtype(x);
    g.node(scope, "Squeeze", &[x, &axes_t], &[], dtype, &out)
}

/// `Reshape` with a static target (`Dim::Batch` keeps the input batch dim).
fn reshape_to(g: &mut Graph, scope: &str, x: &str, dims: &[Dim]) -> String {
    let mut seen_infer = false;
    let values: Vec<i64> = dims
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v,
            Dim::Batch => 0,
            Dim::Param(_) => {
                assert!(!seen_infer, "one inferred dimension per reshape");
                seen_infer = true;
                -1
            }
        })
        .collect();
    let shape = ivec(g, scope, &values);
    let dtype = g.dtype(x);
    g.node(scope, "Reshape", &[x, &shape], &[("allowzero", Attr::Int(0))], dtype, dims)
}

/// `Gather` with a static index vector.
fn gather_rows(g: &mut Graph, scope: &str, x: &str, indices: &[i64], axis: i64) -> String {
    let idx = ivec(g, scope, indices);
    ops::gather(g, scope, x, &idx, axis)
}

/// `ReduceSum` over `axes`.
fn reduce_sum_at(g: &mut Graph, scope: &str, x: &str, axes: &[i64], keepdims: bool) -> String {
    let s = g.shape(x);
    let out: Vec<Dim> = if keepdims {
        s.iter().enumerate().map(|(i, d)| if axes.contains(&(i as i64)) { Dim::Static(1) } else { d.clone() }).collect()
    } else {
        s.iter().enumerate().filter(|(i, _)| !axes.contains(&(*i as i64))).map(|(_, d)| d.clone()).collect()
    };
    let axes_t = ivec(g, scope, axes);
    let dtype = g.dtype(x);
    g.node(scope, "ReduceSum", &[x, &axes_t], &[("keepdims", Attr::Int(keepdims as i64))], dtype, &out)
}

/// Floor-mod (`torch.remainder` with a positive divisor), as `a - b * floor(a / b)`:
/// ONNX Runtime has no floor-mod kernel for floats at opset 17 (`Mod` with `fmod=0`),
/// so it is decomposed instead.
fn modulo(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let ratio = ops::div(g, scope, a, b);
    let q = ops::floor(g, scope, &ratio);
    let q = ops::mul(g, scope, &q, b);
    ops::sub(g, scope, a, &q)
}

/// `Expand` to a leading runtime batch dim (gathered from `batch_src`) + static `tail`.
/// This is the one genuinely data-dependent shape: the traced graphs emit the same
/// Shape -> Gather -> Concat -> Expand chain.
fn expand_batch(g: &mut Graph, scope: &str, x: &str, batch_src: &str, tail: &[i64]) -> String {
    let shape = ops::shape_of(g, scope, batch_src);
    let batch = gather_rows(g, scope, &shape, &[0], 0); // [1]
    let tail_t = ivec(g, scope, tail);
    let target = ops::concat(g, scope, &[&batch, &tail_t], 0);
    let mut dims = vec![Dim::Batch];
    dims.extend(tail.iter().map(|&t| Dim::Static(t)));
    let dtype = g.dtype(x);
    g.node(scope, "Expand", &[x, &target], &[], dtype, &dims)
}

/// Stack tensors along a new axis: unsqueeze each, then concat.
fn stack(g: &mut Graph, scope: &str, xs: &[String], axis: i64) -> String {
    let us: Vec<String> = xs.iter().map(|x| unsqueeze_at(g, scope, x, &[axis])).collect();
    let refs: Vec<&str> = us.iter().map(String::as_str).collect();
    ops::concat(g, scope, &refs, axis)
}

/// `m[..., r, c]` of a batch of 3x3 matrices (the last two dims are the matrix).
fn mat_entry(g: &mut Graph, scope: &str, m: &str, r: i64, c: i64) -> String {
    let rank = g.shape(m).len() as i64;
    let row_s = slice_at(g, scope, m, rank - 2, r, r + 1);
    let row = squeeze_at(g, scope, &row_s, &[rank - 2]);
    let e = slice_at(g, scope, &row, rank - 2, c, c + 1);
    squeeze_at(g, scope, &e, &[rank - 2])
}

/// Column `c` of a batch of 3x3 matrices: `m[..., c]`.
fn mat_col(g: &mut Graph, scope: &str, m: &str, c: i64) -> String {
    let last = g.shape(m).len() as i64 - 1;
    let col = slice_at(g, scope, m, last, c, c + 1);
    squeeze_at(g, scope, &col, &[last])
}

/// Rotate points by matrices: `(m @ p[..., None])[..., 0]`.
fn rotate(g: &mut Graph, scope: &str, m: &str, p: &str) -> String {
    let rank = g.shape(p).len() as i64;
    let pu = unsqueeze_at(g, scope, p, &[rank]);
    let mp = ops::matmul(g, scope, m, &pu);
    let out_axis = g.shape(&mp).len() as i64 - 1;
    squeeze_at(g, scope, &mp, &[out_axis])
}

/// Wrap radians to [-pi, pi): `(x + pi) mod 2pi - pi`.
fn wrap_angle(g: &mut Graph, scope: &str, x: &str) -> String {
    let pi = c_scalar(g, scope, std::f32::consts::PI);
    let two_pi = c_scalar(g, scope, 2.0 * std::f32::consts::PI);
    let shifted = ops::add(g, scope, x, &pi);
    let wrapped = modulo(g, scope, &shifted, &two_pi);
    ops::sub(g, scope, &wrapped, &pi)
}

/// `v / |v|` along the last axis.
fn normalize3(g: &mut Graph, scope: &str, v: &str) -> String {
    let last = g.shape(v).len() as i64 - 1;
    let sq = ops::mul(g, scope, v, v);
    let summed = reduce_sum_at(g, scope, &sq, &[last], true);
    let norm = ops::sqrt(g, scope, &summed);
    ops::div(g, scope, v, &norm)
}

/// The 3D cross product along the last axis.
fn cross3(g: &mut Graph, scope: &str, a: &str, b: &str) -> String {
    let last = g.shape(a).len() as i64 - 1;
    let a0 = slice_at(g, scope, a, last, 0, 1);
    let a1 = slice_at(g, scope, a, last, 1, 2);
    let a2 = slice_at(g, scope, a, last, 2, 3);
    let b0 = slice_at(g, scope, b, last, 0, 1);
    let b1 = slice_at(g, scope, b, last, 1, 2);
    let b2 = slice_at(g, scope, b, last, 2, 3);
    let a1b2 = ops::mul(g, scope, &a1, &b2);
    let a2b1 = ops::mul(g, scope, &a2, &b1);
    let c0 = ops::sub(g, scope, &a1b2, &a2b1);
    let a2b0 = ops::mul(g, scope, &a2, &b0);
    let a0b2 = ops::mul(g, scope, &a0, &b2);
    let c1 = ops::sub(g, scope, &a2b0, &a0b2);
    let a0b1 = ops::mul(g, scope, &a0, &b1);
    let a1b0 = ops::mul(g, scope, &a1, &b0);
    let c2 = ops::sub(g, scope, &a0b1, &a1b0);
    ops::concat(g, scope, &[&c0, &c1, &c2], last)
}

/// 6D rotation (first two columns) -> 3x3 matrix with columns (x, y, z),
/// orthonormalized (upstream `six_matrix`).
fn six_matrix(g: &mut Graph, scope: &str, v: &str) -> String {
    let last = g.shape(v).len() as i64 - 1;
    let v0 = slice_at(g, scope, v, last, 0, 3);
    let x = normalize3(g, scope, &v0);
    let v1 = slice_at(g, scope, v, last, 3, 6);
    let zx = cross3(g, scope, &x, &v1);
    let z = normalize3(g, scope, &zx);
    let y = cross3(g, scope, &z, &x);
    stack(g, scope, &[x, y, z], last + 1)
}

/// WXYZ quaternions `[..., 4]` -> rotation matrices `[..., 3, 3]` (row-major, with
/// upstream's `2 / |q|^2` normalization).
fn quat_matrix(g: &mut Graph, scope: &str, q: &str) -> String {
    let last = g.shape(q).len() as i64 - 1;
    let r = slice_at(g, scope, q, last, 0, 1);
    let x = slice_at(g, scope, q, last, 1, 2);
    let y = slice_at(g, scope, q, last, 2, 3);
    let z = slice_at(g, scope, q, last, 3, 4);
    let one = c_scalar(g, scope, 1.0);
    let two = c_scalar(g, scope, 2.0);
    let qq = ops::mul(g, scope, q, q);
    let qq_sum = reduce_sum_at(g, scope, &qq, &[last], true);
    let s = ops::div(g, scope, &two, &qq_sum);
    let xx = ops::mul(g, scope, &x, &x);
    let yy = ops::mul(g, scope, &y, &y);
    let zz = ops::mul(g, scope, &z, &z);
    let xy = ops::mul(g, scope, &x, &y);
    let xz = ops::mul(g, scope, &x, &z);
    let yz = ops::mul(g, scope, &y, &z);
    let xr = ops::mul(g, scope, &x, &r);
    let yr = ops::mul(g, scope, &y, &r);
    let zr = ops::mul(g, scope, &z, &r);
    let yy_zz = ops::add(g, scope, &yy, &zz);
    let s_yy_zz = ops::mul(g, scope, &s, &yy_zz);
    let xx_zz = ops::add(g, scope, &xx, &zz);
    let s_xx_zz = ops::mul(g, scope, &s, &xx_zz);
    let xx_yy = ops::add(g, scope, &xx, &yy);
    let s_xx_yy = ops::mul(g, scope, &s, &xx_yy);
    let xy_zr = ops::sub(g, scope, &xy, &zr);
    let xz_yr = ops::add(g, scope, &xz, &yr);
    let xy_zr_s = ops::add(g, scope, &xy, &zr);
    let yz_xr = ops::sub(g, scope, &yz, &xr);
    let xz_yr_s = ops::sub(g, scope, &xz, &yr);
    let yz_xr_s = ops::add(g, scope, &yz, &xr);
    let rows = [
        ops::sub(g, scope, &one, &s_yy_zz),
        ops::mul(g, scope, &s, &xy_zr),
        ops::mul(g, scope, &s, &xz_yr),
        ops::mul(g, scope, &s, &xy_zr_s),
        ops::sub(g, scope, &one, &s_xx_zz),
        ops::mul(g, scope, &s, &yz_xr),
        ops::mul(g, scope, &s, &xz_yr_s),
        ops::mul(g, scope, &s, &yz_xr_s),
        ops::sub(g, scope, &one, &s_xx_yy),
    ];
    let flat = ops::concat(g, scope, &rows.iter().map(String::as_str).collect::<Vec<_>>(), last);
    let mut dims: Vec<Dim> = g.shape(q)[..last as usize].to_vec();
    dims.extend([Dim::Static(3), Dim::Static(3)]);
    reshape_to(g, scope, &flat, &dims)
}

/// Rotation matrices -> WXYZ quaternions, Shepperd's method: four candidates,
/// clamp_min(0).sqrt(), rows divided by `2 * clamp_min(a, 0.1)`, and the best candidate
/// picked by argmax + GatherElements. Branch-free — the traced original's single `If`
/// is a statically-resolvable squeeze, so this emits a plain `Squeeze`. `batch_src` is
/// any value with the runtime batch dim (for the argmax-index expand).
fn matrix_quat(g: &mut Graph, scope: &str, m: &str, batch_src: &str) -> String {
    let b = g.shape(m).len() as i64 - 2; // batch rank
    let m00 = mat_entry(g, scope, m, 0, 0);
    let m01 = mat_entry(g, scope, m, 0, 1);
    let m02 = mat_entry(g, scope, m, 0, 2);
    let m10 = mat_entry(g, scope, m, 1, 0);
    let m11 = mat_entry(g, scope, m, 1, 1);
    let m12 = mat_entry(g, scope, m, 1, 2);
    let m20 = mat_entry(g, scope, m, 2, 0);
    let m21 = mat_entry(g, scope, m, 2, 1);
    let m22 = mat_entry(g, scope, m, 2, 2);
    let one = c_scalar(g, scope, 1.0);
    let one_m00 = ops::add(g, scope, &one, &m00);
    let one_m00_m11 = ops::add(g, scope, &one_m00, &m11);
    let c0 = ops::add(g, scope, &one_m00_m11, &m22); // 1 + m00 + m11 + m22
    let one_m00_m_m11 = ops::sub(g, scope, &one_m00, &m11);
    let c1 = ops::sub(g, scope, &one_m00_m_m11, &m22); // 1 + m00 - m11 - m22
    let one_m_m00 = ops::sub(g, scope, &one, &m00);
    let one_m_m00_m11 = ops::add(g, scope, &one_m_m00, &m11);
    let c2 = ops::sub(g, scope, &one_m_m00_m11, &m22); // 1 - m00 + m11 - m22
    let one_m_m00_m_m11 = ops::sub(g, scope, &one_m_m00, &m11);
    let c3 = ops::add(g, scope, &one_m_m00_m_m11, &m22); // 1 - m00 - m11 + m22
    let cands = stack(g, scope, &[c0, c1, c2, c3], b); // [..., 4]
    let zero = c_scalar(g, scope, 0.0);
    let clamped0 = ops::maximum(g, scope, &cands, &zero);
    let a = ops::sqrt(g, scope, &clamped0); // [..., 4]
    let d21_12 = ops::sub(g, scope, &m21, &m12);
    let d02_20 = ops::sub(g, scope, &m02, &m20);
    let d10_01 = ops::sub(g, scope, &m10, &m01);
    let s10_01 = ops::add(g, scope, &m10, &m01);
    let s02_20 = ops::add(g, scope, &m02, &m20);
    let s20_02 = ops::add(g, scope, &m20, &m02);
    let s12_21 = ops::add(g, scope, &m12, &m21);
    let s21_12 = ops::add(g, scope, &m21, &m12);
    let a0 = slice_at(g, scope, &a, b, 0, 1);
    let a0_sq = ops::mul(g, scope, &a0, &a0);
    let a1 = slice_at(g, scope, &a, b, 1, 2);
    let a1_sq = ops::mul(g, scope, &a1, &a1);
    let a2 = slice_at(g, scope, &a, b, 2, 3);
    let a2_sq = ops::mul(g, scope, &a2, &a2);
    let a3 = slice_at(g, scope, &a, b, 3, 4);
    let a3_sq = ops::mul(g, scope, &a3, &a3);
    let d21_12u = unsqueeze_at(g, scope, &d21_12, &[b]);
    let d02_20u = unsqueeze_at(g, scope, &d02_20, &[b]);
    let d10_01u = unsqueeze_at(g, scope, &d10_01, &[b]);
    let s10_01u = unsqueeze_at(g, scope, &s10_01, &[b]);
    let s02_20u = unsqueeze_at(g, scope, &s02_20, &[b]);
    let s20_02u = unsqueeze_at(g, scope, &s20_02, &[b]);
    let s12_21u = unsqueeze_at(g, scope, &s12_21, &[b]);
    let s21_12u = unsqueeze_at(g, scope, &s21_12, &[b]);
    let row0 = ops::concat(g, scope, &[&a0_sq, &d21_12u, &d02_20u, &d10_01u], b);
    let row1 = ops::concat(g, scope, &[&d21_12u, &a1_sq, &s10_01u, &s02_20u], b);
    let row2 = ops::concat(g, scope, &[&d02_20u, &s10_01u, &a2_sq, &s12_21u], b);
    let row3 = ops::concat(g, scope, &[&d10_01u, &s20_02u, &s21_12u, &a3_sq], b);
    let rows = stack(g, scope, &[row0, row1, row2, row3], b); // [..., 4, 4]
    let tenth = c_scalar(g, scope, 0.1);
    let two_d = c_scalar(g, scope, 2.0);
    let clamped = ops::maximum(g, scope, &a, &tenth);
    let denom2 = ops::mul(g, scope, &clamped, &two_d);
    let denom = unsqueeze_at(g, scope, &denom2, &[b + 1]);
    let rows = ops::div(g, scope, &rows, &denom);
    // Pick the best-conditioned candidate per matrix.
    let best = ops::argmax(g, scope, &a, b, false); // [...] int64
    let best = unsqueeze_at(g, scope, &best, &[b]); // [..., 1]
    let best = unsqueeze_at(g, scope, &best, &[b]); // [..., 1, 1]
    let rows_shape = g.shape(&rows).to_vec();
    let mut tail: Vec<i64> = rows_shape[1..rows_shape.len() - 2].iter().map(|d| d.get().expect("static rows batch")).collect();
    tail.extend([1, 4]);
    let best = expand_batch(g, scope, &best, batch_src, &tail); // [..., 1, 4]
    let picked = ops::gather_elements(g, scope, &rows, &best, b); // [..., 1, 4]
    squeeze_at(g, scope, &picked, &[b]) // [..., 4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;
    use crate::graph::DType;

    #[test]
    fn emits_pre_and_postprocess() {
        let mut g = Graph::new();
        let rig = G1Rig::load(&mut g, &config::G1).unwrap();
        let qpos = g.input("qpos", DType::F32, &[Dim::Batch, Dim::Static(8), Dim::Static(G1_QPOS_DIM)]);
        let (global_root, local_root, local_poses) = rig.emit_preprocess(&mut g, "/pre", &qpos).unwrap();
        assert_eq!(g.shape(&global_root), &[Dim::Batch, Dim::Static(8), Dim::Static(5)]);
        assert_eq!(g.shape(&local_root), &[Dim::Batch, Dim::Static(8), Dim::Static(4)]);
        assert_eq!(g.shape(&local_poses), &[Dim::Batch, Dim::Static(8), Dim::Static(303)]);
        let feats = g.input("global_poses", DType::F32, &[Dim::Batch, Dim::Static(64), Dim::Static(414)]);
        let pred = rig.emit_postprocess(&mut g, "/post", &feats).unwrap();
        assert_eq!(g.shape(&pred), &[Dim::Batch, Dim::Static(64), Dim::Static(G1_QPOS_DIM)]);
        g.output(&global_root);
        g.output(&local_root);
        g.output(&local_poses);
        g.output(&pred);
        // Every value name (initializers and node outputs) must be unique.
        let mut names = std::collections::HashSet::new();
        for t in &g.initializers {
            assert!(names.insert(t.name.clone()), "duplicate initializer {}", t.name);
        }
        for n in &g.nodes {
            for o in &n.outputs {
                assert!(names.insert(o.clone()), "duplicate value {o}");
            }
        }
        let model = g.to_model("main_graph");
        assert!(model.graph.unwrap().node.len() > 100);
    }
}
