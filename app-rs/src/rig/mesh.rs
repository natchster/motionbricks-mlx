// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Reader for the `.mbmesh` binary format (`app-rs/assets/soma_mesh.bin`), produced from the
//! upstream SOMA USD rig by `app-rs/scripts/convert_soma_mesh.py`. Layout (all little-endian,
//! strings u8-length-prefixed UTF-8):
//!
//! ```text
//! magic "MBM1" | u32 num_vertices | u32 num_triangles
//! f32 positions[3V] | f32 normals[3V] | u32 indices[3T]
//! u32 num_joints | str joint_names[J] | i32 parents[J] (-1 = root)
//! f32 inverse_bind_matrices[16J] (row-major 4x4)
//! f32 rest_local_transforms[16J] (row-major 4x4)
//! u16 joint_indices[4V] | f32 joint_weights[4V]
//! f32 geom_bind_transform[16] (row-major 4x4)
//! ```
//!
//! The file stores USD's row-vector matrices in row-major order. Reading the 16 floats with
//! [`Mat4::from_cols_array`] yields the transpose, which is exactly the same linear map in
//! column-vector (glam) convention, so the matrices are usable as-is.
//!
//! The parser never panics and never trusts a count from the file: every read is bounds-checked
//! before any allocation happens.

use anyhow::{Context as _, Result, bail};

use glam::{Mat4, Vec3};

use crate::viewer::{Mesh, Skin};

/// File magic at offset 0.
pub const MAGIC: &[u8; 4] = b"MBM1";

/// The SOMA skinned mesh: geometry, skin weights, joint hierarchy and bind transforms.
#[derive(Clone, Debug)]
pub struct SkinnedMesh {
    /// Triangle mesh in geometry space (Y up, meters).
    pub mesh: Mesh,
    /// Skin weights and bind data as stored in the file (see [`SkinnedMesh::geom_bind`]).
    pub skin: Skin,
    /// Joint names, in skinning order.
    pub joint_names: Vec<String>,
    /// Rest pose: one local (parent-relative) transform per joint.
    pub rest_local: Vec<Mat4>,
    /// USD `geomBindTransform`: maps the mesh's geometry space to the skeleton's bind space.
    /// Fold it into the inverse-bind matrices as `inverse_bind[j] * geom_bind` before skinning.
    pub geom_bind: Mat4,
}

/// Bounds-checked little-endian cursor over the file bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Take `n` bytes, failing instead of overrunning. All allocation happens only after this
    /// check, so corrupt counts in the file cannot trigger huge allocations.
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .with_context(|| format!("unexpected end of mesh data: {} bytes left at offset {}, need {n}", self.bytes.len() - self.pos, self.pos))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        let bytes = self.take(n.checked_mul(2).context("u16 count overflows")?)?;
        Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    fn u32(&mut self) -> Result<u32> {
        let c = self.take(4)?;
        Ok(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        let bytes = self.take(n.checked_mul(4).context("u32 count overflows")?)?;
        Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn i32s(&mut self, n: usize) -> Result<Vec<i32>> {
        let bytes = self.take(n.checked_mul(4).context("i32 count overflows")?)?;
        Ok(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let bytes = self.take(n.checked_mul(4).context("f32 count overflows")?)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn vec3s(&mut self, n: usize) -> Result<Vec<Vec3>> {
        Ok(self.f32s(n.checked_mul(3).context("vec3 count overflows")?)?.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect())
    }

    /// Read `n` row-major 4x4 matrices. See the module docs: `from_cols_array` on row-major
    /// bytes gives the transpose, i.e. the same transform in column-vector convention.
    fn mat4s(&mut self, n: usize) -> Result<Vec<Mat4>> {
        Ok(self
            .f32s(n.checked_mul(16).context("mat4 count overflows")?)?
            .chunks_exact(16)
            .map(|c| Mat4::from_cols_array(c.try_into().expect("chunks_exact(16)")))
            .collect())
    }
}

/// Parse a `.mbmesh` file. Every length in the file is checked before use; malformed input
/// returns an error instead of panicking.
pub fn parse(bytes: &[u8]) -> Result<SkinnedMesh> {
    let mut r = Reader { bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        bail!("not an MBM1 mesh file (bad magic)");
    }
    let num_vertices = r.u32()? as usize;
    let num_triangles = r.u32()? as usize;
    let positions = r.vec3s(num_vertices)?;
    let normals = r.vec3s(num_vertices)?;
    let indices = r.u32s(num_triangles.checked_mul(3).context("triangle count overflows")?)?;
    for &i in &indices {
        if i as usize >= num_vertices {
            bail!("triangle index {i} is out of range ({num_vertices} vertices)");
        }
    }

    let num_joints = r.u32()? as usize;
    let mut joint_names = Vec::with_capacity(num_joints.min(4096));
    for _ in 0..num_joints {
        let len = r.u8()? as usize;
        let name = std::str::from_utf8(r.take(len)?).context("joint name is not valid UTF-8")?;
        joint_names.push(name.to_string());
    }
    let parents = r.i32s(num_joints)?;
    for (j, &p) in parents.iter().enumerate() {
        let ok = if j == 0 { p < 0 } else { p >= 0 && (p as usize) < j };
        if !ok {
            bail!("joint {j} ({:?}) has invalid parent index {p}", joint_names.get(j).map_or("?", String::as_str));
        }
    }
    let inverse_bind = r.mat4s(num_joints)?;
    let rest_local = r.mat4s(num_joints)?;

    let joint_indices = r.u16s(num_vertices.checked_mul(4).context("joint index count overflows")?)?;
    let weights: Vec<[u16; 4]> = joint_indices
        .chunks_exact(4)
        .map(|c| {
            for &j in c {
                if j as usize >= num_joints {
                    bail!("skin joint index {j} is out of range ({num_joints} joints)");
                }
            }
            Ok([c[0], c[1], c[2], c[3]])
        })
        .collect::<Result<_>>()?;
    let weight_values: Vec<[f32; 4]> = r
        .f32s(num_vertices.checked_mul(4).context("joint weight count overflows")?)?
        .chunks_exact(4)
        .map(|c| {
            for &w in c {
                if !w.is_finite() {
                    bail!("skin weights must be finite, got {w}");
                }
            }
            Ok([c[0], c[1], c[2], c[3]])
        })
        .collect::<Result<_>>()?;

    let geom_bind = r.mat4s(1)?[0];
    if r.pos != bytes.len() {
        bail!("{} trailing bytes after the mesh data", bytes.len() - r.pos);
    }

    Ok(SkinnedMesh {
        mesh: Mesh { positions, normals, indices },
        skin: Skin { joint_parents: parents, inverse_bind, weights, weight_values },
        joint_names,
        rest_local,
        geom_bind,
    })
}
