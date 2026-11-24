// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The SOMA human rig: the embedded skinned mesh, the runtime skeleton, pose -> geometry
//! evaluation (forward kinematics + LBS bone matrices) and foot-contact sole outlines.
//!
//! Ports upstream `app/runtime/common/rig.py` (SOMA half), `motionbricks/io/usd.py`
//! (`load_skeletal_mesh_with_skeleton_from_usd`, here pre-baked into `assets/soma_mesh.bin`)
//! and `motionbricks/renderers/skeletal_mesh_renderer.py` / `foot_contact_renderer.py`.
//!
//! Simplifications vs upstream:
//! - The `RelaxedHands`/`Fingers` override pose is not ported; fingers stay at the mesh's rest
//!   pose, which is what a clip remapped onto the mesh skeleton produces anyway.
//! - [`sole_outlines`] draws one convex-hull loop per foot-contact joint around the vertices
//!   that joint mainly moves, instead of upstream's three expanding rings split at the toe
//!   joint. The heel/toe split falls out of the dominant-influence assignment for free.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use glam::{Mat4, Vec3};

use motionbricks_rs::runtime as rt;
use rt::{ModelRig, Skeleton, Transform};

use super::clips::list_clips;
use super::mesh::{self, SkinnedMesh};
use crate::viewer::Skin;

/// The embedded SOMA skinned mesh (converted from upstream's `soma_base_skel_minimal_no_root.usd`).
pub const MESH_BYTES: &[u8] = include_bytes!("../../assets/soma_mesh.bin");

/// Lift of a sole outline above the lowest foot vertex (upstream `_OUTLINE_LIFT`).
const OUTLINE_LIFT: f32 = 0.004;
/// Distance the outline runs outside the foot's convex hull (upstream `margin`, one ring only).
const OUTLINE_MARGIN: f32 = 0.015;

/// The SOMA rig: skinned mesh, skeletons and the model->runtime space conversion.
pub struct SomaRig {
    /// The embedded skinned mesh (77 joints of skinning data).
    pub mesh: SkinnedMesh,
    /// Skeleton of the mesh hierarchy. Clips are remapped onto it by joint name, exactly like
    /// upstream (`ClipRegistry(rig, skeleton=skeletal_mesh.skeleton)`); joints a clip does not
    /// animate (fingers, eyes, jaw) keep their rest pose.
    pub skeleton: Arc<Skeleton>,
    /// The skeleton `model-rs` builds from a SOMA BVH clip (`Hips` first, the identity `Root`
    /// dropped): the hierarchy the engine and clip loaders use without reading the USD. The
    /// shipped clips carry the full 77-joint hierarchy, so its names match the mesh skeleton's.
    pub runtime_skeleton: Arc<Skeleton>,
    /// SOMA model space (Y up) -> Z-up world, facing like the G1 (upstream `model_to_runtime`).
    pub model_to_runtime: Transform,
}

impl SomaRig {
    /// Load the rig: parse the embedded mesh and build the runtime skeleton from the first SOMA
    /// BVH clip found under `clip_dir` (searched recursively), as `model-rs` does.
    pub fn load(clip_dir: impl AsRef<Path>) -> Result<SomaRig> {
        let mesh = mesh::parse(MESH_BYTES)?;
        let skeleton = Arc::new(mesh_skeleton(&mesh)?);
        let clips = list_clips(clip_dir.as_ref(), ".bvh")?;
        let first = clips.first().with_context(|| format!("no SOMA BVH clips under {}", clip_dir.as_ref().display()))?;
        let runtime_skeleton = Arc::new(
            rt::load_skeleton_from_bvh(first, true, true).with_context(|| format!("could not build the SOMA skeleton from {}", first.display()))?,
        );
        Ok(SomaRig { mesh, skeleton, runtime_skeleton, model_to_runtime: rt::model_to_runtime(ModelRig::Soma) })
    }

    /// Forward kinematics over the mesh hierarchy: local joint transforms -> world (model-space)
    /// joint transforms. `root` is applied on top of the pose's own root (usually
    /// `Transform::IDENTITY` or the space conversion).
    pub fn fk(&self, local: &[Transform], root: Transform) -> Result<Vec<Transform>> {
        fk(&self.skeleton, local, root)
    }

    /// World joint matrices for [`crate::viewer::Frame::log_skinned_mesh`]'s `bone_transforms`.
    pub fn bone_matrices(&self, world: &[Transform]) -> Vec<Mat4> {
        bone_matrices(world)
    }

    /// [`SomaRig::fk`] + [`SomaRig::bone_matrices`] in one call.
    pub fn evaluate(&self, local: &[Transform], root: Transform) -> Result<Vec<Mat4>> {
        Ok(bone_matrices(&self.fk(local, root)?))
    }

    /// Expand a pose on another skeleton (typically the 30-joint `runtime_skeleton`) onto the
    /// mesh hierarchy by joint name; joints the source lacks keep the mesh's rest pose.
    pub fn expand_pose(&self, source: &Skeleton, local: &[Transform]) -> Result<Vec<Transform>> {
        if local.len() != source.num_joints() {
            bail!("pose has {} transforms but the source skeleton has {} joints", local.len(), source.num_joints());
        }
        let mut out = self.skeleton.reference_local_transforms().to_vec();
        for (j, name) in self.skeleton.joint_names().iter().enumerate() {
            if let Some(s) = source.joint_index(name) {
                out[j] = local[s];
            }
        }
        Ok(out)
    }

    /// The skin for [`crate::viewer::Frame::log_skinned_mesh`], with the geometry bind
    /// transform folded into the inverse-bind matrices: at bind pose each per-joint product
    /// `world_j * inverse_bind_j` then equals `geom_bind`, mapping geometry space to skeleton
    /// space, as USD's skinning specifies.
    pub fn viewer_skin(&self) -> Skin {
        let mut skin = self.mesh.skin.clone();
        for inv_bind in &mut skin.inverse_bind {
            *inv_bind *= self.mesh.geom_bind;
        }
        skin
    }

    /// Sole outlines of the four foot-contact joints (see [`sole_outlines`]).
    pub fn sole_outlines(&self) -> Result<Vec<SoleOutline>> {
        sole_outlines(&self.mesh)
    }
}

/// A closed loop of points, in one joint's bind-local space, outlining where the model
/// predicts foot contact (upstream `ContactOutline`; the loop closes implicitly).
#[derive(Clone, Debug)]
pub struct SoleOutline {
    /// Foot contact index: 0 left heel, 1 left toe, 2 right heel, 3 right toe.
    pub contact: usize,
    /// Joint in the mesh hierarchy whose world transform moves this outline.
    pub joint: usize,
    /// Loop points in the joint's bind-local space.
    pub points: Vec<Vec3>,
}

/// Runtime transform -> world matrix (positions in meters, XYZW quaternion).
pub fn transform_to_mat4(t: Transform) -> Mat4 {
    Mat4::from_rotation_translation(glam::Quat::from_array(t.q), Vec3::from_array(t.p.0))
}

/// A rigid 4x4 matrix -> runtime transform (any scale is dropped).
pub fn mat4_to_transform(m: Mat4) -> Transform {
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    Transform::new(rt::Vec3(translation.to_array()), rotation.to_array())
}

/// Build the runtime skeleton of a parsed mesh (names, parents, rest-local transforms).
pub fn mesh_skeleton(mesh: &SkinnedMesh) -> Result<Skeleton> {
    Skeleton::new(mesh.joint_names.clone(), mesh.skin.joint_parents.clone(), mesh.rest_local.iter().map(|&m| mat4_to_transform(m)).collect())
}

/// FK: compose local joint transforms through the hierarchy under `root`.
pub fn fk(skeleton: &Skeleton, local: &[Transform], root: Transform) -> Result<Vec<Transform>> {
    if local.len() != skeleton.num_joints() {
        bail!("pose has {} transforms but the skeleton has {} joints", local.len(), skeleton.num_joints());
    }
    Ok(skeleton.compute_world_transforms(local, root))
}

/// World joint transforms -> joint matrices for the viewer's LBS.
pub fn bone_matrices(world: &[Transform]) -> Vec<Mat4> {
    world.iter().map(|&t| transform_to_mat4(t)).collect()
}

/// One convex-hull loop per foot-contact joint, in the joint's bind-local space.
///
/// Simplification of upstream `skeletal_mesh_sole_outlines`: instead of three expanding rings
/// around the whole foot split at the toe joint, each contact joint outlines the vertices it
/// is the dominant influence of (single ring, [`OUTLINE_MARGIN`] offset). A joint with too few
/// dominant vertices falls back to looser weight thresholds; if none work its outline is left
/// out, so the result can hold fewer than four outlines.
pub fn sole_outlines(mesh: &SkinnedMesh) -> Result<Vec<SoleOutline>> {
    let mut outlines = Vec::with_capacity(rt::SOMA_FOOT_CONTACT_JOINTS.len());
    for (contact, &name) in rt::SOMA_FOOT_CONTACT_JOINTS.iter().enumerate() {
        let joint = mesh
            .joint_names
            .iter()
            .position(|n| n == name)
            .with_context(|| format!("the SOMA mesh has no foot-contact joint {name:?}"))?;
        // Vertex -> joint-local bind position, for the vertices this joint mainly moves.
        let from_geom = mesh.skin.inverse_bind[joint] * mesh.geom_bind;
        let local = |v: usize| from_geom.transform_point3(mesh.mesh.positions[v]);
        let joint_weight = |v: usize| -> f32 {
            mesh.skin.weights[v].iter().zip(mesh.skin.weight_values[v]).filter(|&(&j, _)| j as usize == joint).map(|(_, w)| w).sum()
        };
        let dominant = |v: usize| -> bool {
            let best = (0..4).map(|k| mesh.skin.weight_values[v][k]).fold(0.0f32, f32::max);
            joint_weight(v) >= best && best > 0.0
        };
        let mut points: Vec<Vec3> = Vec::new();
        for threshold in [f32::INFINITY, 0.5, 0.25] {
            points = (0..mesh.mesh.positions.len())
                .filter(|&v| if threshold == f32::INFINITY { dominant(v) } else { joint_weight(v) >= threshold })
                .map(local)
                .collect();
            if points.len() >= 3 {
                break;
            }
        }
        // 2D convex hull over (x, z), lifted just above the lowest vertex.
        let floor = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) + OUTLINE_LIFT;
        let hull = convex_hull(&points.iter().map(|p| [p.x, p.z]).collect::<Vec<_>>());
        if hull.len() < 3 {
            continue;
        }
        let center = hull.iter().fold([0.0; 2], |c, p| [c[0] + p[0] / hull.len() as f32, c[1] + p[1] / hull.len() as f32]);
        let loop_points = hull
            .iter()
            .map(|p| {
                let d = [p[0] - center[0], p[1] - center[1]];
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let (mx, mz) = if len > 1e-9 { (d[0] / len * OUTLINE_MARGIN, d[1] / len * OUTLINE_MARGIN) } else { (0.0, 0.0) };
                Vec3::new(p[0] + mx, floor, p[1] + mz)
            })
            .collect();
        outlines.push(SoleOutline { contact, joint, points: loop_points });
    }
    Ok(outlines)
}

/// World-space line segments of the sole outlines under the given world joint transforms
/// (model space), ready for [`crate::viewer::Frame::log_lines`].
pub fn outline_segments(outlines: &[SoleOutline], world: &[Transform]) -> Vec<[Vec3; 2]> {
    let mut segments = Vec::new();
    for outline in outlines {
        let Some(&t) = world.get(outline.joint) else { continue };
        let m = transform_to_mat4(t);
        let n = outline.points.len();
        for i in 0..n {
            segments.push([m.transform_point3(outline.points[i]), m.transform_point3(outline.points[(i + 1) % n])]);
        }
    }
    segments
}

/// Andrew's monotone chain: the convex hull of 2D points, counter-clockwise, no repeated or
/// collinear boundary points. Fewer than 3 distinct non-collinear points yield what there is.
fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal).then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(pts.len() + 1);
    // Lower hull over the sorted points, then the upper hull over them reversed; the start
    // point ends up repeated at the end.
    for &p in pts.iter().chain(pts.iter().rev()) {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}
