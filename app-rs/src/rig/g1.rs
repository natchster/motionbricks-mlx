// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The Unitree G1 robot rig: the MJCF body tree, the STL link meshes merged into one
//! [`Mesh`], and forward kinematics over joint coordinates (rigid links, no skinning).
//!
//! Ports upstream `motionbricks/io/utils.py::get_unitree_g1_model` (a Newton MJCF import)
//! and the drawing half of `motionbricks/renderers/dof_mesh_renderer.py`.
//!
//! Simplifications vs upstream (Newton's full MJCF importer):
//! - Only `<geom>` entries referencing a mesh are loaded; primitive geoms (the four collision
//!   spheres per ankle, the shoulder cylinders, the floor plane) are skipped.
//! - Visual and collision copies of the same mesh are both kept, as upstream draws both.
//! - Inertials, actuators, sensors, sites, joint ranges and the scene/lighting section are
//!   ignored: this is a rig for drawing, not simulation.
//! - Every actuated joint is a hinge at the body origin, which holds for
//!   `g1_29dof_rev_1_0.xml`; anything else (ball/slide joints, off-origin anchors) is an error.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

use motionbricks_rs::runtime as rt;
use rt::{ModelRig, Transform};

use crate::viewer::Mesh;

/// Path of the G1 MJCF below an upstream checkout.
pub const MJCF_RELATIVE_PATH: &str = "motionbricks/assets/robots/unitree_g1/g1_29dof_rev_1_0.xml";

/// One revolute joint of a body. `dof` indexes the rig's joint coordinate order
/// ([`G1Rig::joint_names`], the runtime G1 contract order, not the MJCF document order).
#[derive(Clone, Debug)]
pub struct G1Joint {
    /// MJCF joint name (also the CSV column name).
    pub name: String,
    /// Rotation axis in the body's local frame (normalized).
    pub axis: [f32; 3],
    /// Index of this joint's angle within the joint coordinates (`q[7 + dof]`).
    pub dof: usize,
}

/// One rigid body of the MJCF tree, in document (depth-first, parents-first) order.
#[derive(Clone, Debug)]
pub struct G1Body {
    /// MJCF body name.
    pub name: String,
    /// Parent body index, -1 for the root (pelvis).
    pub parent: i32,
    /// Body frame relative to its parent (MJCF `pos`/`quat`).
    pub offset: Transform,
    /// The body's revolute joint; the root's free joint and fixed bodies have none.
    pub joint: Option<G1Joint>,
}

/// A subrange of the merged [`G1Rig::mesh`] drawn with one rigid transform.
#[derive(Clone, Debug)]
pub struct G1Link {
    /// Body this geometry belongs to.
    pub body: usize,
    /// The geom's offset in the body frame (MJCF geom `pos`/`quat`).
    pub offset: Transform,
    /// Fill color from the geom's `rgba` attribute.
    pub rgba: [f32; 4],
    /// Vertex range in the merged mesh.
    pub vertices: Range<usize>,
    /// Index range in the merged mesh.
    pub indices: Range<usize>,
}

/// The Unitree G1 rig: body tree plus all link meshes merged into one viewer mesh.
pub struct G1Rig {
    /// The body tree (30 bodies: pelvis + 29 actuated links), parents before children.
    pub bodies: Vec<G1Body>,
    /// Per-geom draw units into [`G1Rig::mesh`], in body order.
    pub links: Vec<G1Link>,
    /// All link STL geometry merged; slice per link with [`G1Rig::link_mesh`].
    pub mesh: Mesh,
    /// Joint coordinate order (the runtime G1 contract: `UNITREE_G1_JOINT_NAMES`).
    pub joint_names: Vec<String>,
    /// G1 model space is already Z-up world: the identity (upstream `model_to_runtime`).
    pub model_to_runtime: Transform,
}

impl G1Rig {
    /// Load the rig from an upstream checkout root (the directory holding `motionbricks/`).
    pub fn load(upstream_root: impl AsRef<Path>) -> Result<G1Rig> {
        Self::load_mjcf(upstream_root.as_ref().join(MJCF_RELATIVE_PATH))
    }

    /// Load the rig from the MJCF file itself; meshes resolve against `<xml dir>/<meshdir>/`.
    pub fn load_mjcf(mjcf_path: impl AsRef<Path>) -> Result<G1Rig> {
        let mjcf_path = mjcf_path.as_ref();
        let xml = std::fs::read_to_string(mjcf_path).with_context(|| format!("could not read MJCF file {}", mjcf_path.display()))?;
        let mjcf = parse_mjcf(&xml).with_context(|| format!("could not parse MJCF file {}", mjcf_path.display()))?;
        let dir = mjcf_path.parent().unwrap_or(Path::new(".")).join(&mjcf.mesh_dir);
        Self::from_mjcf(&mjcf, &dir)
    }

    /// Build the rig from a parsed MJCF and the directory holding its mesh files.
    pub fn from_mjcf(mjcf: &Mjcf, mesh_dir: &Path) -> Result<G1Rig> {
        // MJCF joint name -> index in the runtime contract's joint coordinate order.
        let dofs: HashMap<&str, usize> = rt::UNITREE_G1_JOINT_NAMES.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut bodies = Vec::with_capacity(mjcf.bodies.len());
        for b in &mjcf.bodies {
            let joint = b
                .joint
                .as_ref()
                .map(|j| {
                    let Some(&dof) = dofs.get(j.name.as_str()) else {
                        bail!("MJCF joint {:?} is not part of the runtime G1 contract", j.name);
                    };
                    Ok(G1Joint { name: j.name.clone(), axis: j.axis, dof })
                })
                .transpose()?;
            bodies.push(G1Body { name: b.name.clone(), parent: b.parent, offset: b.offset, joint });
        }

        let mut mesh = Mesh::default();
        let mut links = Vec::new();
        let mut stl_cache: HashMap<PathBuf, Mesh> = HashMap::new();
        for (body_index, b) in mjcf.bodies.iter().enumerate() {
            for g in &b.geoms {
                let file = mjcf
                    .mesh_files
                    .get(&g.mesh)
                    .with_context(|| format!("geom of body {:?} references undeclared mesh asset {:?}", b.name, g.mesh))?;
                let path = mesh_dir.join(file);
                if !stl_cache.contains_key(&path) {
                    let m = load_stl(&path)?;
                    stl_cache.insert(path.clone(), m);
                }
                let Some(stl) = stl_cache.get(&path) else { bail!("internal error: STL cache lost {}", path.display()) };
                let vertex_start = mesh.positions.len();
                let index_start = mesh.indices.len();
                mesh.positions.extend_from_slice(&stl.positions);
                mesh.normals.extend_from_slice(&stl.normals);
                if stl.indices.is_empty() {
                    // A flat triangle soup gets explicit sequential indices, so the merged
                    // mesh's index buffer stays uniform across links.
                    mesh.indices.extend(vertex_start as u32..mesh.positions.len() as u32);
                } else {
                    mesh.indices.extend(stl.indices.iter().map(|&i| i + vertex_start as u32));
                }
                links.push(G1Link {
                    body: body_index,
                    offset: g.offset,
                    rgba: g.rgba,
                    vertices: vertex_start..mesh.positions.len(),
                    indices: index_start..mesh.indices.len(),
                });
            }
        }
        Ok(G1Rig {
            bodies,
            links,
            mesh,
            joint_names: rt::UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect(),
            model_to_runtime: rt::model_to_runtime(ModelRig::UnitreeG1),
        })
    }

    /// Forward kinematics: joint coordinates (`7 + 29` floats: root position, root XYZW
    /// quaternion, then one hinge angle per [`G1Rig::joint_names`] entry) -> per-body world
    /// transforms in model space. The root body's world transform is the coordinate root
    /// directly, as with MJCF free joints (the pelvis' `pos` attribute only seeds `qpos0`).
    pub fn fk(&self, q: &[f32]) -> Result<Vec<Transform>> {
        let expected = rt::ROOT_COORDINATE_COUNT + self.joint_names.len();
        if q.len() != expected {
            bail!("a G1 pose holds {expected} coordinates, got {}", q.len());
        }
        let mut world = Vec::with_capacity(self.bodies.len());
        world.push(rt::read_root(q));
        for i in 1..self.bodies.len() {
            let b = &self.bodies[i];
            let parent = usize::try_from(b.parent).context("body parent must be a body before it")?;
            let mut w = world[parent] * b.offset;
            if let Some(j) = &b.joint {
                w = w * Transform::new(rt::Vec3::ZERO, rt::math::quat_from_axis_angle(rt::Vec3(j.axis), q[rt::ROOT_COORDINATE_COUNT + j.dof]));
            }
            world.push(w);
        }
        Ok(world)
    }

    /// Per-link world transforms for a pose: one per [`G1Rig::links`] entry, for
    /// [`crate::viewer::Frame::log_mesh`].
    pub fn link_transforms(&self, q: &[f32]) -> Result<Vec<Transform>> {
        let world = self.fk(q)?;
        Ok(self.links.iter().map(|l| world[l.body] * l.offset).collect())
    }

    /// Slice one link's geometry out of the merged mesh (indices rebased to 0).
    pub fn link_mesh(&self, link: usize) -> Result<Mesh> {
        let Some(l) = self.links.get(link) else { bail!("no link {link} ({} links)", self.links.len()) };
        let base = l.vertices.start as u32;
        Ok(Mesh {
            positions: self.mesh.positions[l.vertices.clone()].to_vec(),
            normals: self.mesh.normals[l.vertices.clone()].to_vec(),
            indices: self.mesh.indices[l.indices.clone()].iter().map(|&i| i - base).collect(),
        })
    }

    /// The standing reference pose: identity root at the pelvis' rest height, all hinges at 0.
    pub fn reference_q(&self) -> Vec<f32> {
        let mut q = vec![0.0; rt::ROOT_COORDINATE_COUNT + self.joint_names.len()];
        let root = self.bodies.first().map_or(Transform::IDENTITY, |b| b.offset);
        rt::write_root(root, &mut q);
        q
    }
}

/// One MJCF geom referencing a mesh asset (parsed form, before STLs load).
#[derive(Clone, Debug)]
pub struct MjcfGeom {
    /// Name of the referenced `<mesh>` asset.
    pub mesh: String,
    /// Geom offset in the body frame.
    pub offset: Transform,
    /// Geom color.
    pub rgba: [f32; 4],
}

/// One MJCF body (parsed form: the joint's dof is not resolved yet).
#[derive(Clone, Debug)]
pub struct MjcfBody {
    /// Body name.
    pub name: String,
    /// Parent body index, -1 for the root.
    pub parent: i32,
    /// Body frame relative to its parent.
    pub offset: Transform,
    /// Hinge joint (name, axis); the free root joint and fixed bodies have none.
    pub joint: Option<MjcfJoint>,
    /// Mesh geoms of this body.
    pub geoms: Vec<MjcfGeom>,
}

/// A parsed MJCF hinge joint.
#[derive(Clone, Debug)]
pub struct MjcfJoint {
    /// Joint name.
    pub name: String,
    /// Rotation axis in the body frame.
    pub axis: [f32; 3],
}

/// The parsed MJCF subset the rig needs.
#[derive(Clone, Debug)]
pub struct Mjcf {
    /// `<compiler meshdir="...">`, relative to the MJCF file.
    pub mesh_dir: String,
    /// Mesh asset name -> file name (relative to `mesh_dir`).
    pub mesh_files: HashMap<String, String>,
    /// Body tree in document order (parents before children).
    pub bodies: Vec<MjcfBody>,
}

/// MJCF quaternions are `w x y z`; the runtime uses `x y z w`.
fn mjcf_quat(w: f32, x: f32, y: f32, z: f32) -> rt::Quat {
    [x, y, z, w]
}

fn parse_floats(value: &str, n: usize, what: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> = value
        .split_whitespace()
        .map(|s| s.parse().with_context(|| format!("{what}: {s:?} is not a number")))
        .collect::<Result<_>>()?;
    if v.len() != n {
        bail!("{what}: expected {n} numbers, got {} in {value:?}", v.len());
    }
    Ok(v)
}

/// Parse the MJCF subset the rig draws: the body tree with hinge joints and mesh geoms, the
/// mesh assets and `meshdir`. See the module docs for what is intentionally skipped.
pub fn parse_mjcf(xml: &str) -> Result<Mjcf> {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut mesh_dir = String::new();
    let mut mesh_files = HashMap::new();
    let mut bodies: Vec<MjcfBody> = Vec::new();
    let mut stack: Vec<usize> = Vec::new(); // open `<body>` elements, innermost last

    let attr = |e: &quick_xml::events::BytesStart, name: &str| -> Result<Option<String>> {
        for a in e.attributes().with_checks(false) {
            let a = a.context("malformed XML attribute")?;
            if a.key.as_ref() == name.as_bytes() {
                return Ok(Some(String::from_utf8(a.value.into_owned()).context("XML attribute is not valid UTF-8")?));
            }
        }
        Ok(None)
    };
    let frame = |e: &quick_xml::events::BytesStart, what: &str| -> Result<Transform> {
        let pos = match attr(e, "pos")? {
            Some(v) => parse_floats(&v, 3, &format!("{what} pos"))?,
            None => vec![0.0; 3],
        };
        let q = match attr(e, "quat")? {
            Some(v) => parse_floats(&v, 4, &format!("{what} quat"))?,
            None => vec![1.0, 0.0, 0.0, 0.0],
        };
        Ok(Transform::new(rt::Vec3::new(pos[0], pos[1], pos[2]), mjcf_quat(q[0], q[1], q[2], q[3])))
    };

    loop {
        match reader.read_event().context("malformed MJCF XML")? {
            quick_xml::events::Event::Start(e) if e.name().as_ref() == b"body" => {
                let name = attr(&e, "name")?.context("a <body> needs a name")?;
                let parent = stack.last().map_or(-1, |&p| p as i32);
                stack.push(bodies.len());
                bodies.push(MjcfBody { name, parent, offset: frame(&e, "body")?, joint: None, geoms: Vec::new() });
            }
            quick_xml::events::Event::End(e) if e.name().as_ref() == b"body" => {
                stack.pop();
            }
            quick_xml::events::Event::Empty(e) | quick_xml::events::Event::Start(e) => {
                match e.name().as_ref() {
                    b"compiler" => {
                        if let Some(dir) = attr(&e, "meshdir")? {
                            mesh_dir = dir;
                        }
                    }
                    b"mesh" => {
                        if let (Some(name), Some(file)) = (attr(&e, "name")?, attr(&e, "file")?) {
                            mesh_files.insert(name, file);
                        }
                    }
                    b"joint" => {
                        let Some(&b) = stack.last() else { continue }; // free root joint sits in the root body
                        let kind = attr(&e, "type")?.unwrap_or_else(|| "hinge".to_string());
                        if kind != "hinge" {
                            continue; // the pelvis' free joint: the pose's root coordinates drive it
                        }
                        let name = attr(&e, "name")?.context("a <joint> needs a name")?;
                        let pos = match attr(&e, "pos")? {
                            Some(v) => parse_floats(&v, 3, "joint pos")?,
                            None => vec![0.0; 3],
                        };
                        if pos.iter().any(|&c| c != 0.0) {
                            bail!("joint {name:?} has a non-zero anchor, which this port does not support");
                        }
                        let axis = match attr(&e, "axis")? {
                            Some(v) => parse_floats(&v, 3, "joint axis")?,
                            None => vec![0.0, 0.0, 1.0], // MJCF default
                        };
                        let axis = rt::Vec3::new(axis[0], axis[1], axis[2]).normalize();
                        if axis.length() == 0.0 {
                            bail!("joint {name:?} has a zero axis");
                        }
                        bodies[b].joint = Some(MjcfJoint { name, axis: axis.0 });
                    }
                    b"geom" => {
                        let (Some(&b), Some(mesh_name)) = (stack.last(), attr(&e, "mesh")?) else { continue }; // primitive geoms are skipped
                        let rgba = match attr(&e, "rgba")? {
                            Some(v) => parse_floats(&v, 4, "geom rgba")?,
                            None => vec![0.5, 0.5, 0.5, 1.0],
                        };
                        bodies[b].geoms.push(MjcfGeom { mesh: mesh_name, offset: frame(&e, "geom")?, rgba: [rgba[0], rgba[1], rgba[2], rgba[3]] });
                    }
                    _ => {}
                }
            }
            quick_xml::events::Event::Eof => break,
            _ => {}
        }
    }
    if bodies.is_empty() {
        bail!("the MJCF file declares no bodies");
    }
    Ok(Mjcf { mesh_dir, mesh_files, bodies })
}

/// Load an STL file (binary or ASCII) as a flat triangle soup.
pub fn load_stl(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("could not read STL file {}", path.display()))?;
    parse_stl(&bytes).with_context(|| format!("could not parse STL file {}", path.display()))
}

/// Parse STL bytes, sniffing the encoding: if the size matches the binary layout
/// (`84 + 50 * count`) it is binary (some binary STLs start with "solid"), otherwise ASCII.
pub fn parse_stl(bytes: &[u8]) -> Result<Mesh> {
    let binary_count = |b: &[u8]| -> Option<usize> {
        if b.len() < 84 {
            return None;
        }
        let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
        (84 + 50 * n == b.len()).then_some(n)
    };
    if let Some(n) = binary_count(bytes) {
        return parse_binary_stl(&bytes[84..], n);
    }
    parse_ascii_stl(std::str::from_utf8(bytes).context("the STL is neither binary nor ASCII")?)
}

/// Binary STL: `count` 50-byte records (facet normal + 3 vertices + u16 attribute).
fn parse_binary_stl(records: &[u8], count: usize) -> Result<Mesh> {
    if records.len() != 50 * count {
        bail!("binary STL holds {} record bytes, expected {count} records", records.len());
    }
    let mut mesh = Mesh { positions: Vec::with_capacity(3 * count), normals: Vec::with_capacity(3 * count), indices: Vec::new() };
    for r in records.chunks_exact(50) {
        let f = |i: usize| f32::from_le_bytes([r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]]);
        let v = |i: usize| glam::Vec3::new(f(3 * i), f(3 * i + 1), f(3 * i + 2));
        let (p0, p1, p2) = (v(1), v(2), v(3));
        let mut n = v(0);
        if n.length_squared() < 1e-12 {
            n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        }
        if ![p0, p1, p2, n].iter().all(|p| p.is_finite()) {
            bail!("binary STL contains non-finite coordinates");
        }
        mesh.positions.extend_from_slice(&[p0, p1, p2]);
        mesh.normals.extend_from_slice(&[n, n, n]);
    }
    Ok(mesh)
}

/// ASCII STL: `facet normal` / `vertex` lines, three vertices per facet.
fn parse_ascii_stl(text: &str) -> Result<Mesh> {
    let mut mesh = Mesh::default();
    let mut normal = glam::Vec3::ZERO;
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let v = |t: &[&str], what: &str| -> Result<glam::Vec3> {
            if t.len() != 4 {
                bail!("ASCII STL: a {what} line needs three numbers");
            }
            Ok(glam::Vec3::new(t[1].parse()?, t[2].parse()?, t[3].parse()?))
        };
        match tokens.as_slice() {
            [first, rest @ ..] if *first == "facet" && rest.first() == Some(&"normal") => normal = v(&tokens[1..], "facet normal")?,
            [first, rest @ ..] if *first == "vertex" => {
                let p = v(&tokens, "vertex")?;
                mesh.positions.push(p);
                mesh.normals.push(normal);
                if mesh.positions.len() % 3 == 0 && normal.length_squared() < 1e-12 {
                    let n = mesh.positions.len();
                    let face = (mesh.positions[n - 2] - mesh.positions[n - 3]).cross(mesh.positions[n - 1] - mesh.positions[n - 3]).normalize_or_zero();
                    mesh.normals[n - 3..].fill(face);
                }
            }
            _ => {}
        }
    }
    if mesh.positions.is_empty() || mesh.positions.len() % 3 != 0 {
        bail!("ASCII STL holds {} vertices, not a positive multiple of 3", mesh.positions.len());
    }
    Ok(mesh)
}
