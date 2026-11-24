// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Tests for the rig module: the `.mbmesh` reader, STL/MJCF parsing, forward kinematics and
//! the clip registry. Tests that need the upstream checkout or the motion checkpoints skip
//! gracefully when those are absent.
//!
//! `src/lib.rs` does not declare the rig module yet (that wiring is the integrator's job), so
//! the viewer and rig modules are included here directly by path; `crate::viewer` inside the
//! rig sources resolves against the `viewer` module below.

#[path = "../src/viewer/mod.rs"]
pub mod viewer;
#[path = "../src/rig/mod.rs"]
pub mod rig;

use std::path::{Path, PathBuf};

use motionbricks_rs::runtime as rt;
use rt::math::QUAT_IDENTITY;
use rt::{Skeleton, Transform};

use rig::{clips, g1, mesh, soma};

const MESH_BYTES: &[u8] = include_bytes!("../assets/soma_mesh.bin");

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn motions_dir() -> PathBuf {
    workspace().join("checkpoints/motions")
}

fn upstream_dir() -> PathBuf {
    workspace().join("sandbox/upstream")
}

fn skip(reason: &str, path: &Path) {
    eprintln!("skipping ({reason}): {}", path.display());
}

// ---------------------------------------------------------------------------
// .mbmesh parsing
// ---------------------------------------------------------------------------

#[test]
fn mbmesh_parses_soma_mesh() {
    let m = mesh::parse(MESH_BYTES).expect("the embedded SOMA mesh parses");
    assert_eq!(m.mesh.positions.len(), 18_056, "vertex count");
    assert_eq!(m.mesh.normals.len(), 18_056, "normal count");
    assert_eq!(m.mesh.indices.len(), 3 * 36_108, "index count");
    assert_eq!(m.joint_names.len(), 77, "joint count");
    assert_eq!(m.skin.inverse_bind.len(), 77);
    assert_eq!(m.rest_local.len(), 77);
    assert_eq!(m.skin.weights.len(), 18_056);
    assert_eq!(m.skin.weight_values.len(), 18_056);

    // Every index references a real vertex.
    assert!(m.mesh.indices.iter().all(|&i| (i as usize) < m.mesh.positions.len()));
    // Skin weights are normalized (the converter divides by the weight sum).
    for (v, w) in m.skin.weight_values.iter().enumerate() {
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "vertex {v}: skin weights sum to {sum}");
        assert!(m.skin.weights[v].iter().all(|&j| (j as usize) < m.joint_names.len()));
    }
    // Parents: the root is first, every other parent comes before its child.
    assert_eq!(m.skin.joint_parents[0], -1);
    for (j, &p) in m.skin.joint_parents.iter().enumerate().skip(1) {
        assert!(p >= 0 && (p as usize) < j, "joint {j} has parent {p}");
    }
    // The bind data is finite.
    assert!(m.mesh.positions.iter().all(|p| p.is_finite()));
    assert!(m.skin.inverse_bind.iter().all(|m| m.is_finite()));
    assert!(m.geom_bind.is_finite());

    // The joint hierarchy makes a valid runtime skeleton.
    let skeleton = mesh::parse(MESH_BYTES).map(|m| soma::mesh_skeleton(&m)).expect("parse").expect("skeleton");
    assert_eq!(skeleton.num_joints(), 77);
    assert_eq!(skeleton.joint_name(0), Some("Hips"));
}

#[test]
fn mbmesh_rejects_malformed_input() {
    assert!(mesh::parse(b"").is_err());
    assert!(mesh::parse(b"XXXX....").is_err(), "bad magic");
    // Truncations at every point of a valid prefix fail without panicking.
    for end in [4, 8, 12, 100, 1024, MESH_BYTES.len() / 2, MESH_BYTES.len() - 1] {
        assert!(mesh::parse(&MESH_BYTES[..end]).is_err(), "truncated at {end}");
    }
    // Absurd counts must fail, not allocate.
    let mut absurd = b"MBM1".to_vec();
    absurd.extend_from_slice(&u32::MAX.to_le_bytes()); // vertices
    absurd.extend_from_slice(&u32::MAX.to_le_bytes()); // triangles
    absurd.extend_from_slice(&[0u8; 64]);
    assert!(mesh::parse(&absurd).is_err());
    // Trailing garbage is rejected.
    let mut trailing = MESH_BYTES.to_vec();
    trailing.push(0);
    assert!(mesh::parse(&trailing).is_err());
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// Two-bone hand check: root translated to (0, 1, 0); child offset (1, 0, 0) and rotated 90
/// degrees about Z. A point at (1, 0, 0) in the child's frame lands at (1, 2, 0).
#[test]
fn fk_two_bone_hand_check() {
    let identity = Transform::IDENTITY;
    let offset = Transform::new(rt::Vec3::new(1.0, 0.0, 0.0), QUAT_IDENTITY);
    let skeleton = Skeleton::new(vec!["root".to_string(), "tip".to_string()], vec![-1, 0], vec![identity, offset]).expect("skeleton");

    let rot_z_90 = [0.0, 0.0, (std::f32::consts::FRAC_PI_4).sin(), (std::f32::consts::FRAC_PI_4).cos()];
    let pose = vec![
        Transform::new(rt::Vec3::new(0.0, 1.0, 0.0), QUAT_IDENTITY),
        Transform::new(rt::Vec3::new(1.0, 0.0, 0.0), rot_z_90),
    ];
    let world = soma::fk(&skeleton, &pose, Transform::IDENTITY).expect("fk");
    assert_eq!(world.len(), 2);
    assert!((world[0].p.0[0] - 0.0).abs() < 1e-6 && (world[0].p.0[1] - 1.0).abs() < 1e-6, "root at {:?}", world[0].p);
    assert!((world[1].p.0[0] - 1.0).abs() < 1e-6 && (world[1].p.0[1] - 1.0).abs() < 1e-6, "tip at {:?}", world[1].p);

    let bones = soma::bone_matrices(&world);
    let p = bones[1].transform_point3(glam::Vec3::new(1.0, 0.0, 0.0));
    assert!(p.abs_diff_eq(glam::Vec3::new(1.0, 2.0, 0.0), 1e-5), "rotated point at {p:?}");

    // A pose of the wrong width is an error, not a panic.
    assert!(soma::fk(&skeleton, &pose[..1], Transform::IDENTITY).is_err());
}

/// The G1's rigid FK: a root plus one hinge child. Offset (0, 0, 1), hinge about +Y by 90
/// degrees; a point at (1, 0, 0) in the child frame lands at the origin.
#[test]
fn fk_g1_rigid_hand_check() {
    let rig = g1::G1Rig {
        bodies: vec![
            g1::G1Body { name: "pelvis".into(), parent: -1, offset: Transform::IDENTITY, joint: None },
            g1::G1Body {
                name: "link".into(),
                parent: 0,
                offset: Transform::new(rt::Vec3::new(0.0, 0.0, 1.0), QUAT_IDENTITY),
                joint: Some(g1::G1Joint { name: "j".into(), axis: [0.0, 1.0, 0.0], dof: 0 }),
            },
        ],
        links: Vec::new(),
        mesh: viewer::Mesh::default(),
        joint_names: vec!["j".into()],
        model_to_runtime: Transform::IDENTITY,
    };
    let mut q = vec![0.0; 8];
    q[6] = 1.0; // identity root rotation (XYZW)
    q[7] = std::f32::consts::FRAC_PI_2;
    let world = rig.fk(&q).expect("fk");
    assert_eq!(world.len(), 2);
    assert!(world[0].p.length() < 1e-6, "root at the origin");
    assert!((world[1].p.0[2] - 1.0).abs() < 1e-6, "child offset along z: {:?}", world[1].p);
    let p = world[1].point(rt::Vec3::new(1.0, 0.0, 0.0));
    assert!(p.length() < 1e-5, "+Y 90-degree hinge maps (1,0,0) to (0,0,-1): got {p:?}");

    // Wrong coordinate count is an error.
    assert!(rig.fk(&q[..7]).is_err());
}

// ---------------------------------------------------------------------------
// STL and MJCF (real G1 assets; skip when the upstream checkout is absent)
// ---------------------------------------------------------------------------

#[test]
fn stl_parses_real_g1_mesh() {
    let path = upstream_dir().join("motionbricks/assets/robots/unitree_g1/meshes/pelvis.STL");
    if !path.is_file() {
        return skip("no upstream checkout", &path);
    }
    let mesh = g1::load_stl(&path).expect("pelvis.STL parses");
    assert!(!mesh.positions.is_empty() && mesh.positions.len() % 3 == 0, "triangle soup");
    assert_eq!(mesh.normals.len(), mesh.positions.len());
    assert!(mesh.positions.iter().all(|p| p.is_finite()));
    assert!(mesh.normals.iter().all(|n| n.is_finite()));
    // The pelvis is tens of centimeters across, not millimeters or meters.
    let (lo, hi) = mesh.positions.iter().fold((glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    let extent = (hi - lo).max_element();
    assert!((0.05..2.0).contains(&extent), "pelvis extent {extent} m");

    // The same bytes parse identically through the raw parser, and garbage fails.
    let bytes = std::fs::read(&path).expect("read");
    assert_eq!(g1::parse_stl(&bytes).expect("parse").positions.len(), mesh.positions.len());
    assert!(g1::parse_stl(b"not an stl").is_err());
}

#[test]
fn mjcf_parses_g1_and_stls_exist() {
    let xml_path = upstream_dir().join(g1::MJCF_RELATIVE_PATH);
    if !xml_path.is_file() {
        return skip("no upstream checkout", &xml_path);
    }
    let xml = std::fs::read_to_string(&xml_path).expect("read MJCF");
    let mjcf = g1::parse_mjcf(&xml).expect("MJCF parses");
    assert_eq!(mjcf.bodies.len(), 30, "30 bodies including the pelvis root");
    assert_eq!(mjcf.bodies[0].name, "pelvis");
    assert_eq!(mjcf.bodies[0].parent, -1);
    for (i, b) in mjcf.bodies.iter().enumerate().skip(1) {
        assert!(b.parent >= 0 && (b.parent as usize) < i, "body {} has parent {}", b.name, b.parent);
        assert!(b.joint.is_some(), "body {} has a hinge joint", b.name);
    }
    // Every referenced STL exists next to the XML under meshdir.
    let mesh_dir = xml_path.parent().expect("dir").join(&mjcf.mesh_dir);
    assert!(!mjcf.mesh_files.is_empty());
    for (name, file) in &mjcf.mesh_files {
        assert!(mesh_dir.join(file).is_file(), "mesh asset {name:?}: {} is missing", mesh_dir.join(file).display());
    }

    // The full rig loads: all 29 contract dofs are covered, geometry merged per link.
    let rig = g1::G1Rig::load(upstream_dir()).expect("G1 rig loads");
    assert_eq!(rig.bodies.len(), 30);
    let mut dofs: Vec<usize> = rig.bodies.iter().filter_map(|b| b.joint.as_ref().map(|j| j.dof)).collect();
    dofs.sort_unstable();
    assert_eq!(dofs, (0..29).collect::<Vec<_>>(), "one dof per contract joint");
    assert_eq!(rig.joint_names.len(), 29);
    assert!(!rig.links.is_empty());
    assert!(!rig.mesh.positions.is_empty());
    assert!(rig.links.iter().all(|l| l.vertices.end <= rig.mesh.positions.len() && l.indices.end <= rig.mesh.indices.len()));

    // FK of the reference pose: pelvis at its rest height, links above the floor-ish region.
    let q = rig.reference_q();
    let world = rig.fk(&q).expect("fk");
    assert_eq!(world.len(), 30);
    assert!(world.iter().all(|t| t.p.0.iter().all(|v| v.is_finite())));
    assert!((world[0].p.0[2] - 0.793).abs() < 1e-5, "pelvis rest height: {:?}", world[0].p);
    let link_transforms = rig.link_transforms(&q).expect("link transforms");
    assert_eq!(link_transforms.len(), rig.links.len());

    // Slicing a link yields a self-contained mesh.
    let link0 = rig.link_mesh(0).expect("link 0");
    assert!(!link0.positions.is_empty());
    assert!(link0.indices.iter().all(|&i| (i as usize) < link0.positions.len()));
}

// ---------------------------------------------------------------------------
// Clip registry (real clips; skip when the checkpoints are absent)
// ---------------------------------------------------------------------------

#[test]
fn clip_registry_loads_soma_bvh() {
    let soma_dir = motions_dir().join("soma");
    if !soma_dir.is_dir() {
        return skip("no motion checkpoints", &soma_dir);
    }
    let rig = soma::SomaRig::load(&soma_dir).expect("SOMA rig loads");
    // The runtime skeleton comes from a BVH clip: Hips first, the identity Root dropped.
    // These clips carry the full 77-joint hierarchy, matching the mesh skeleton by name.
    assert_eq!(rig.runtime_skeleton.joint_name(0), Some("Hips"));
    assert_eq!(rig.runtime_skeleton.num_joints(), 77);
    assert_eq!(rig.runtime_skeleton.joint_names(), rig.skeleton.joint_names());
    assert!(rt::SOMA_JOINT_NAMES.iter().all(|n| rig.runtime_skeleton.joint_index(n).is_some()), "the 30 contract joints are present");
    assert_eq!(rig.skeleton.num_joints(), 77, "the mesh skeleton");
    assert_eq!(rig.viewer_skin().inverse_bind.len(), 77);

    // Sole outlines: one loop per foot contact, in the contact order.
    let outlines = rig.sole_outlines().expect("sole outlines");
    assert_eq!(outlines.len(), 4, "heel and toe per foot");
    for (c, o) in outlines.iter().enumerate() {
        assert_eq!(o.contact, c);
        assert!(o.points.len() >= 3, "outline {c} has {} points", o.points.len());
        assert!(o.points.iter().all(|p| p.is_finite()));
    }

    let mut registry = clips::ClipRegistry::soma(rig.skeleton.clone());
    let clips_found = registry.list(&soma_dir).expect("list");
    assert!(!clips_found.is_empty(), "BVH clips under {}", soma_dir.display());
    assert!(clips_found.iter().all(|p| p.extension().is_some_and(|e| e == "bvh")));
    let path = clips_found[0].clone();

    let asset = registry.load(&path).expect("load");
    assert!(!asset.name().is_empty());
    assert!(asset.clip.num_frames() > 0);
    assert!(asset.duration() > 0.0);
    assert_eq!(asset.last_frame(), asset.clip.num_frames() - 1);
    let buffer = asset.clip.as_soma().expect("a SOMA clip").clone();
    assert_eq!(buffer.skeleton().num_joints(), 77, "clips are remapped onto the mesh skeleton");

    // Loading again hits the cache.
    registry.load(&path).expect("cached");
    assert_eq!(registry.len(), 1);

    // The clip drives FK: a sampled pose yields finite bone matrices for the viewer.
    let pose = buffer.sample(0.0, Transform::IDENTITY).expect("sample");
    let bones = rig.evaluate(&pose, Transform::IDENTITY).expect("evaluate");
    assert_eq!(bones.len(), 77);
    assert!(bones.iter().all(|m| m.is_finite()));
    let world = rig.fk(&pose, Transform::IDENTITY).expect("fk");
    assert!(!soma::outline_segments(&outlines, &world).is_empty());

    // A looping copy joins its ends while the shared clip is untouched.
    let looping = registry.load_looping(&path).expect("load_looping");
    let looping = looping.as_soma().expect("SOMA").clone();
    assert_eq!(looping.num_frames(), buffer.num_frames());
    let (first, last) = (looping.frame(0), looping.frame(looping.num_frames() - 1));
    let dp = (first[0].p - last[0].p).length();
    assert!(dp < 1e-4, "looping clip: first/last root positions differ by {dp}");
    for j in 1..first.len() {
        let dot: f32 = first[j].q.iter().zip(last[j].q).map(|(a, b)| a * b).sum();
        assert!(dot.abs() > 1.0 - 1e-3, "looping clip: joint {j} rotations differ");
    }
    let shared = registry.load(&path).expect("shared").clip.as_soma().expect("SOMA").clone();
    assert_eq!(shared.frame(0)[0].p.0, buffer.frame(0)[0].p.0, "the shared clip is unchanged");

    // Errors name the file.
    let err = registry.load(soma_dir.join("no_such_clip.bvh")).unwrap_err();
    assert!(format!("{err:#}").contains("no_such_clip.bvh"), "{err:#}");
    let csv = motions_dir().join("unitree_g1/locomotion");
    if csv.is_dir() {
        let csv = clips::list_clips(&csv, ".csv").expect("list")[0].clone();
        let err = registry.load(&csv).unwrap_err();
        assert!(format!("{err:#}").contains(".bvh"), "{err:#}");
    }
}

#[test]
fn clip_registry_loads_g1_csv() {
    let g1_dir = motions_dir().join("unitree_g1");
    if !g1_dir.is_dir() {
        return skip("no motion checkpoints", &g1_dir);
    }
    let mut registry = clips::ClipRegistry::unitree_g1_contract();
    let found = registry.list(&g1_dir).expect("list");
    assert!(!found.is_empty(), "CSV clips under {}", g1_dir.display());
    let asset = registry.load(&found[0]).expect("load");
    let buffer = asset.clip.as_g1().expect("a G1 clip");
    assert_eq!(buffer.num_dofs(), 29);
    assert_eq!(buffer.sample_rate(), rt::CSV_SAMPLE_RATE);
    assert!(buffer.num_frames() > 0);
    assert_eq!(buffer.joint_names(), rt::UNITREE_G1_JOINT_NAMES);

    // A looping copy joins its ends (joint coordinates included).
    let looping = registry.load_looping(&found[0]).expect("load_looping");
    let looping = looping.as_g1().expect("G1").clone();
    let (first, last) = (looping.frame(0).to_vec(), looping.frame(looping.num_frames() - 1).to_vec());
    for (c, (a, b)) in first.iter().zip(&last).enumerate().skip(7) {
        assert!((a - b).abs() < 1e-3, "looping clip: coordinate {c} differs by {}", (a - b).abs());
    }

    // A BVH is rejected by a G1 registry.
    let soma_dir = motions_dir().join("soma");
    if soma_dir.is_dir() {
        let bvh = clips::list_clips(&soma_dir, ".bvh").expect("list")[0].clone();
        let err = registry.load(&bvh).unwrap_err();
        assert!(format!("{err:#}").contains(".csv"), "{err:#}");
    }

    // A registry can also be built from an explicit joint order.
    let names: Vec<String> = rt::UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
    assert!(clips::ClipRegistry::unitree_g1(names).is_ok());
    assert!(clips::ClipRegistry::unitree_g1(Vec::new()).is_err());
}
