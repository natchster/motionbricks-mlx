// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Integration tests for the document IO module (`src/documents/`).

use std::fs;
use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use motionbricks_app::documents::smart_primitive::{
    Axis, ConeSide, ConstraintType, OperationOrder, PlaybackMode,
};
use motionbricks_app::documents::{
    ModelRig, SceneDocument, SmartPrimitiveDocument, Transform,
};

const EULER_TOLERANCE_DEGREES: f32 = 1.0e-3;

fn assets_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn documents_with_suffix(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("readable assets directory") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().to_lowercase().ends_with(suffix))
            {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
}

fn scene_files() -> Vec<PathBuf> {
    let paths = documents_with_suffix(&assets_dir().join("scenes"), ".scene.json");
    assert!(!paths.is_empty(), "no scene documents shipped");
    paths
}

fn smart_primitive_files() -> Vec<PathBuf> {
    let paths =
        documents_with_suffix(&assets_dir().join("smart_primitives"), ".smart_primitive.json");
    assert!(!paths.is_empty(), "no Smart Primitive documents shipped");
    paths
}

/// A fresh temp directory per test case, cleaned up on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "motionbricks-documents-test-{}-{tag}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Top-level object keys in file order. `serde_json::Map` does not preserve
/// order (the crate does not enable `preserve_order`), so scan the text.
fn top_level_keys(json: &str) -> Vec<String> {
    let bytes = json.as_bytes();
    let mut index = 0;
    let mut keys = Vec::new();

    fn skip_ws(bytes: &[u8], index: &mut usize) {
        while *index < bytes.len() && matches!(bytes[*index], b' ' | b'\t' | b'\n' | b'\r') {
            *index += 1;
        }
    }
    fn read_string<'a>(bytes: &'a [u8], index: &mut usize) -> &'a str {
        assert_eq!(bytes[*index], b'"', "expected a string");
        let start = *index + 1;
        *index += 1;
        while bytes[*index] != b'"' {
            *index += if bytes[*index] == b'\\' { 2 } else { 1 };
        }
        let end = *index;
        *index += 1;
        std::str::from_utf8(&bytes[start..end]).expect("key is UTF-8")
    }
    fn skip_value(bytes: &[u8], index: &mut usize) {
        let mut depth = 0usize;
        loop {
            match bytes[*index] {
                b'"' => {
                    read_string(bytes, index);
                    if depth == 0 {
                        return;
                    }
                }
                b'{' | b'[' => {
                    depth += 1;
                    *index += 1;
                }
                b'}' | b']' => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    *index += 1;
                    if depth == 0 {
                        return;
                    }
                }
                b',' if depth == 0 => return,
                _ => *index += 1,
            }
        }
    }

    skip_ws(bytes, &mut index);
    assert_eq!(bytes[index], b'{', "document is an object");
    index += 1;
    loop {
        skip_ws(bytes, &mut index);
        if bytes[index] == b'}' {
            return keys;
        }
        if bytes[index] == b',' {
            index += 1;
            continue;
        }
        keys.push(read_string(bytes, &mut index).to_string());
        skip_ws(bytes, &mut index);
        assert_eq!(bytes[index], b':', "expected ':'");
        index += 1;
        skip_ws(bytes, &mut index);
        skip_value(bytes, &mut index);
    }
}

fn assert_vec3_close(actual: Vec3, expected: Vec3, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs().max_element() <= tolerance,
        "{what}: expected {expected:?}, got {actual:?}"
    );
}

fn assert_transform_eq(actual: &Transform, expected: &Transform, what: &str) {
    assert_eq!(
        actual, expected,
        "{what}: transforms must round-trip exactly"
    );
}

fn assert_euler_close(actual: Vec3, expected: Vec3, what: &str) {
    assert_vec3_close(actual, expected, EULER_TOLERANCE_DEGREES, what);
}

fn assert_scene_eq(actual: &SceneDocument, expected: &SceneDocument) {
    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.rig, expected.rig);
    assert_eq!(actual.selected_item_id, expected.selected_item_id);
    assert_eq!(actual.camera, expected.camera);
    assert_eq!(actual.character_start.id, expected.character_start.id);
    assert_transform_eq(
        &actual.character_start.transform,
        &expected.character_start.transform,
        "character_start.transform",
    );
    assert_euler_close(
        actual.character_start.euler_degrees,
        expected.character_start.euler_degrees,
        "character_start.euler_degrees",
    );
    assert_eq!(
        actual.smart_primitives.len(),
        expected.smart_primitives.len()
    );
    for (actual, expected) in actual
        .smart_primitives
        .iter()
        .zip(&expected.smart_primitives)
    {
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.parent_id, expected.parent_id);
        assert_transform_eq(&actual.transform, &expected.transform, "smart_primitive.transform");
        assert_euler_close(
            actual.euler_degrees,
            expected.euler_degrees,
            "smart_primitive.euler_degrees",
        );
        assert_eq!(actual.reference, expected.reference);
        assert_eq!(actual.visibility, expected.visibility);
        assert_eq!(actual.target_pose_color, expected.target_pose_color);
        assert_eq!(actual.behavior, expected.behavior);
    }
    assert_eq!(actual.solid_cubes.len(), expected.solid_cubes.len());
    for (actual, expected) in actual.solid_cubes.iter().zip(&expected.solid_cubes) {
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.parent_id, expected.parent_id);
        assert_transform_eq(&actual.transform, &expected.transform, "solid_cube.transform");
        assert_euler_close(
            actual.euler_degrees,
            expected.euler_degrees,
            "solid_cube.euler_degrees",
        );
        assert_eq!(actual.dimensions, expected.dimensions);
        assert_eq!(actual.color, expected.color);
    }
}

fn assert_smart_primitive_eq(actual: &SmartPrimitiveDocument, expected: &SmartPrimitiveDocument) {
    fn assert_node_eq(
        actual: &motionbricks_app::documents::SceneNode,
        expected: &motionbricks_app::documents::SceneNode,
    ) {
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.name, expected.name);
        assert_transform_eq(
            &actual.local_transform,
            &expected.local_transform,
            "node.local_transform",
        );
        match (actual.euler_degrees, expected.euler_degrees) {
            (Some(actual), Some(expected)) => {
                assert_euler_close(actual, expected, "node.euler_degrees")
            }
            (None, None) => {}
            _ => panic!("node.euler_degrees null-ness changed"),
        }
        assert_eq!(actual.visible, expected.visible);
        assert_eq!(actual.translation_locks, expected.translation_locks);
        assert_eq!(actual.rotation_locks, expected.rotation_locks);
        assert_eq!(actual.components, expected.components);
        assert_eq!(actual.children.len(), expected.children.len());
        for (actual, expected) in actual.children.iter().zip(&expected.children) {
            assert_node_eq(actual, expected);
        }
    }

    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.rig, expected.rig);
    assert_eq!(actual.selected_node_id, expected.selected_node_id);
    assert_eq!(actual.camera, expected.camera);
    assert_eq!(actual.sequences, expected.sequences);
    assert_node_eq(&actual.root, &expected.root);
}

#[test]
fn scene_documents_round_trip() {
    let temp = TempDir::new("scene-round-trip");
    for source_path in scene_files() {
        let mut document = SceneDocument::load(&source_path)
            .unwrap_or_else(|error| panic!("load {}: {error}", source_path.display()));

        // Save under the same file name in a temp dir and load it back.
        let saved_path = temp.0.join(source_path.file_name().unwrap());
        let returned_path = document.save(&saved_path).expect("save scene");
        assert_eq!(returned_path, saved_path);
        let reloaded = SceneDocument::load(&saved_path).expect("reload scene");
        assert_scene_eq(&reloaded, &document);

        // The re-serialized document keeps the source's top-level keys and order.
        let source_json = fs::read_to_string(&source_path).unwrap();
        let saved_json = fs::read_to_string(&saved_path).unwrap();
        assert_eq!(
            top_level_keys(&saved_json),
            top_level_keys(&source_json),
            "top-level keys of {}",
            source_path.display()
        );

        // A second load/save cycle is byte-stable (same file name, so the
        // document name derived from the path stays the same).
        let mut reloaded = reloaded;
        let second_dir = TempDir::new("scene-round-trip-second");
        let second_path = second_dir.0.join(source_path.file_name().unwrap());
        reloaded.save(&second_path).expect("re-save scene");
        assert_eq!(
            fs::read(&saved_path).unwrap(),
            fs::read(&second_path).unwrap(),
            "save must be byte-stable for {}",
            source_path.display()
        );
    }
}

#[test]
fn smart_primitive_documents_round_trip() {
    let temp = TempDir::new("smart-primitive-round-trip");
    for source_path in smart_primitive_files() {
        let mut document = SmartPrimitiveDocument::load(&source_path)
            .unwrap_or_else(|error| panic!("load {}: {error}", source_path.display()));

        let saved_path = temp.0.join(source_path.file_name().unwrap());
        let returned_path = document.save(&saved_path).expect("save Smart Primitive");
        assert_eq!(returned_path, saved_path);
        let reloaded = SmartPrimitiveDocument::load(&saved_path).expect("reload Smart Primitive");
        assert_smart_primitive_eq(&reloaded, &document);

        let source_json = fs::read_to_string(&source_path).unwrap();
        let saved_json = fs::read_to_string(&saved_path).unwrap();
        assert_eq!(
            top_level_keys(&saved_json),
            top_level_keys(&source_json),
            "top-level keys of {}",
            source_path.display()
        );

        let mut reloaded = reloaded;
        let second_dir = TempDir::new("smart-primitive-round-trip-second");
        let second_path = second_dir.0.join(source_path.file_name().unwrap());
        reloaded.save(&second_path).expect("re-save Smart Primitive");
        assert_eq!(
            fs::read(&saved_path).unwrap(),
            fs::read(&second_path).unwrap(),
            "save must be byte-stable for {}",
            source_path.display()
        );
    }
}

#[test]
fn saved_documents_use_two_space_json_and_atomic_writes() {
    let temp = TempDir::new("layout");
    let source_path = scene_files().pop().unwrap();
    let mut document = SceneDocument::load(&source_path).unwrap();
    let saved_path = temp.0.join(source_path.file_name().unwrap());
    document.save(&saved_path).unwrap();

    let saved_json = fs::read_to_string(&saved_path).unwrap();
    assert!(saved_json.starts_with("{\n  \"format\":"), "2-space indent");
    assert!(saved_json.ends_with("}\n"), "trailing newline");
    let leftovers: Vec<_> = fs::read_dir(&temp.0)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "no temporary files left behind");
}

#[test]
fn scene_hierarchy_world_transforms() {
    // A cube under a Smart Primitive rotated 90 degrees about z.
    let mut document = SceneDocument::load(scene_files().pop().unwrap()).unwrap();
    let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    document.smart_primitives[0].transform =
        Transform::new(Vec3::new(1.0, 2.0, 3.0), rotation);
    document.smart_primitives[0].parent_id = None;
    let parent_id = document.smart_primitives[0].id;
    document.solid_cubes[0].parent_id = Some(parent_id);
    document.solid_cubes[0].transform = Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY);
    let cube_id = document.solid_cubes[0].id;

    let world = document.world_transform(cube_id).expect("cube exists");
    assert_vec3_close(
        world.translation,
        Vec3::new(1.0, 3.0, 3.0),
        1.0e-6,
        "world translation",
    );
    // World rotation is the parent's 90 degrees about z: x maps to y.
    assert_vec3_close(
        world.rotation_xyzw * Vec3::X,
        Vec3::Y,
        1.0e-6,
        "world rotation",
    );

    // The parent's own world transform equals its local one (top level).
    let parent_world = document.world_transform(parent_id).expect("parent exists");
    assert_vec3_close(
        parent_world.translation,
        Vec3::new(1.0, 2.0, 3.0),
        1.0e-6,
        "parent world translation",
    );

    // flatten() moves every item to the top level with its world transform.
    let flattened = document.flatten();
    assert!(
        flattened
            .smart_primitives
            .iter()
            .all(|item| item.parent_id.is_none())
            && flattened
                .solid_cubes
                .iter()
                .all(|item| item.parent_id.is_none())
    );
    let flat_cube = flattened
        .solid_cubes
        .iter()
        .find(|cube| cube.id == cube_id)
        .unwrap();
    assert_vec3_close(
        flat_cube.transform.translation,
        Vec3::new(1.0, 3.0, 3.0),
        1.0e-6,
        "flattened world translation",
    );

    // children() returns direct children in ID order: the reparented cube (6)
    // and "Box" (9), which shipped under SmallJumpSP (2).
    let children = document.children(Some(parent_id));
    assert_eq!(
        children.iter().map(|item| item.id()).collect::<Vec<_>>(),
        vec![6, 9]
    );
    assert!(document.next_free_id() > cube_id);
}

#[test]
fn scene_hierarchy_matches_shipped_example() {
    // soma_gym: cube "Wall" (id 6) sits under "MidJumpSP" (id 5, rotated -90
    // degrees about z); the cube's local +90 about z cancels out in world space.
    let path = assets_dir().join("scenes/soma_gym.scene.json");
    let document = SceneDocument::load(path).unwrap();
    let world = document.world_transform(6).expect("Wall exists");
    assert_vec3_close(
        world.translation,
        Vec3::new(4.0, 8.0, 0.39313892),
        1.0e-5,
        "Wall world translation",
    );
    assert_vec3_close(
        world.rotation_xyzw * Vec3::X,
        Vec3::X,
        1.0e-5,
        "Wall world rotation",
    );
}

#[test]
fn smart_primitive_world_transform_and_ids() {
    let path = assets_dir().join("smart_primitives/soma/sit.smart_primitive.json");
    let document = SmartPrimitiveDocument::load(&path).unwrap();

    // Node 12 ("ToSit") sits under node 5 ("Sit"), rotated 90 degrees about z
    // at the origin, so its world translation is the local one rotated.
    let world = document.world_transform(12).expect("node 12 exists");
    assert_vec3_close(
        world.translation,
        Vec3::new(0.40323576, 0.020531408, 0.0),
        1.0e-5,
        "node 12 world translation",
    );
    assert_vec3_close(
        world.rotation_xyzw * Vec3::X,
        Vec3::Y,
        1.0e-5,
        "node 12 world rotation",
    );

    // Root's world transform is its local one; IDs allocate above the maximum.
    let root_world = document.world_transform(document.root.id).unwrap();
    assert_eq!(root_world, document.root.local_transform);
    assert_eq!(document.find_parent(document.root.id), None);
    assert_eq!(document.find_parent(12).map(|node| node.id), Some(5));
    assert_eq!(document.next_free_id(), 14);
    assert!(document.is_visible(12).unwrap());
}

#[test]
fn smart_primitive_runtime_accessors_tree_order() {
    let path = assets_dir().join("smart_primitives/soma/sit.smart_primitive.json");
    let document = SmartPrimitiveDocument::load(&path).unwrap();

    let target_poses = document.target_poses();
    assert_eq!(
        target_poses.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![5, 6, 12],
        "target poses in tree order"
    );
    assert_eq!(
        target_poses
            .iter()
            .map(|(_, component)| component.id)
            .collect::<Vec<_>>(),
        vec![7, 8, 13]
    );
    assert_eq!(
        document
            .interaction_boxes()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(
        document.pivots().iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![3]
    );
}

#[test]
fn sequence_resolution_soma_sit() {
    let path = assets_dir().join("smart_primitives/soma/sit.smart_primitive.json");
    let document = SmartPrimitiveDocument::load(&path).unwrap();

    assert_eq!(document.sequences.len(), 2);

    let sit = document.resolve_sequence(9).expect("Sit sequence");
    assert_eq!(sit.name, "Sit");
    assert_eq!(
        sit.target_poses
            .iter()
            .map(|(node_id, _)| *node_id)
            .collect::<Vec<_>>(),
        vec![12, 5],
        "Sit steps in authored order"
    );
    assert_eq!(
        sit.target_poses
            .iter()
            .map(|(_, component)| component.id)
            .collect::<Vec<_>>(),
        vec![13, 7]
    );
    let (pivot_node_id, pivot) = sit.pivot.expect("Sit has a pivot");
    assert_eq!(pivot_node_id, 3);
    assert_eq!(pivot.id, 11);
    assert_eq!(pivot.rotation_axis, Axis::Z);
    assert_eq!(pivot.rotation_constraint, ConstraintType::Unconstrained);
    assert_eq!(pivot.translation_axis, Axis::None);
    assert_eq!(pivot.cone_side, ConeSide::Forward);
    assert_eq!(pivot.operation_order, OperationOrder::TranslationThenRotation);

    let stand = document.resolve_sequence(10).expect("Stand sequence");
    assert_eq!(stand.name, "Stand");
    assert_eq!(
        stand
            .target_poses
            .iter()
            .map(|(node_id, _)| *node_id)
            .collect::<Vec<_>>(),
        vec![6]
    );
    assert_eq!(stand.pivot, None);

    assert_eq!(document.resolve_sequences().len(), 2);

    // Spot-check a target pose's fields, including the playback mode enum.
    let to_stand = document.find_node(6).unwrap().target_pose().unwrap();
    assert_eq!(to_stand.playback_mode, PlaybackMode::Time);
    assert_eq!(to_stand.num_poses_to_use, 4);
    assert!(to_stand.show_skeletons);
    assert_eq!(
        to_stand.clip.as_deref(),
        Some("../../motions/soma/sitting/sit_on_chair_stop_R_001__A047.bvh")
    );
}

#[test]
fn clip_paths_resolve_relative_to_the_document() {
    let path = assets_dir().join("smart_primitives/soma/sit.smart_primitive.json");
    let document = SmartPrimitiveDocument::load(&path).unwrap();
    let sit = document.find_node(5).unwrap().target_pose().unwrap();
    let clip_path = sit.clip_path(&path).expect("node 5 has a clip");
    assert_eq!(
        clip_path,
        assets_dir().join("motions/soma/sitting/sit_on_chair_loop.bvh"),
        "the clip resolves against the document's directory"
    );
}

#[test]
fn scene_references_resolve_and_rigs_match() {
    let path = assets_dir().join("scenes/soma_gym.scene.json");
    let document = SceneDocument::load(&path).unwrap();
    document.check_reference_rigs(&path).expect("rigs match");
    let reference = document.smart_primitives[0].reference.as_ref().unwrap();
    assert_eq!(
        SceneDocument::resolve_reference(reference, &path),
        assets_dir().join("smart_primitives/soma/jump_0_5m.smart_primitive.json")
    );
}

#[test]
fn validation_rejects_wrong_format() {
    let source = fs::read_to_string(scene_files().pop().unwrap()).unwrap();
    let wrong = source.replacen(
        "\"format\": \"motionbricks.scene\"",
        "\"format\": \"motionbricks.not_a_scene\"",
        1,
    );
    assert_ne!(wrong, source);
    let temp = TempDir::new("wrong-format");
    let path = temp.0.join("bad.scene.json");
    fs::write(&path, wrong).unwrap();
    let error = SceneDocument::load(&path).unwrap_err();
    assert!(
        error
            .message()
            .contains("Unsupported Scene format 'motionbricks.not_a_scene'"),
        "{error}"
    );
}

#[test]
fn validation_rejects_bad_schema_version() {
    let source = fs::read_to_string(scene_files().pop().unwrap()).unwrap();
    let wrong = source.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
    assert_ne!(wrong, source);
    let temp = TempDir::new("bad-schema");
    let path = temp.0.join("bad.scene.json");
    fs::write(&path, wrong).unwrap();
    let error = SceneDocument::load(&path).unwrap_err();
    assert!(
        error
            .message()
            .contains("Unsupported Scene schema version 2"),
        "{error}"
    );
}

#[test]
fn validation_rejects_dangling_parent_id() {
    let source = fs::read_to_string(assets_dir().join("scenes/soma_gym.scene.json")).unwrap();
    let wrong = source.replacen("\"parent_id\": 5", "\"parent_id\": 999", 1);
    assert_ne!(wrong, source);
    let temp = TempDir::new("dangling-parent");
    let path = temp.0.join("bad.scene.json");
    fs::write(&path, wrong).unwrap();
    let error = SceneDocument::load(&path).unwrap_err();
    assert!(
        error
            .message()
            .contains("must have a Smart Primitive or Solid Cube as its parent"),
        "{error}"
    );
}

#[test]
fn validation_rejects_parent_cycle() {
    let source = fs::read_to_string(assets_dir().join("scenes/soma_gym.scene.json")).unwrap();
    // Point MidJumpSP (id 5, currently top level) under its own child Wall (id 6).
    let wrong = source.replacen(
        "\"name\": \"MidJumpSP\",\n      \"parent_id\": null",
        "\"name\": \"MidJumpSP\",\n      \"parent_id\": 6",
        1,
    );
    assert_ne!(wrong, source);
    let temp = TempDir::new("parent-cycle");
    let path = temp.0.join("bad.scene.json");
    fs::write(&path, wrong).unwrap();
    let error = SceneDocument::load(&path).unwrap_err();
    assert!(error.message().contains("cannot sit under itself"), "{error}");
}

#[test]
fn validation_rejects_rig_mismatch() {
    let path = assets_dir().join("smart_primitives/soma/sit.smart_primitive.json");
    let error = SmartPrimitiveDocument::load_for_rig(&path, ModelRig::UnitreeG1).unwrap_err();
    assert!(
        error
            .message()
            .contains("was saved for the 'soma' rig, but this app uses 'unitree_g1'"),
        "{error}"
    );
    SmartPrimitiveDocument::load_for_rig(&path, ModelRig::Soma).expect("matching rig loads");
    assert_eq!(SmartPrimitiveDocument::load_rig(&path).unwrap(), ModelRig::Soma);
}

#[test]
fn validation_rejects_dangling_sequence_node() {
    let source =
        fs::read_to_string(assets_dir().join("smart_primitives/soma/sit.smart_primitive.json"))
            .unwrap();
    let wrong = source.replacen("        12,\n        5", "        99,\n        5", 1);
    assert_ne!(wrong, source);
    let temp = TempDir::new("dangling-sequence-node");
    let path = temp.0.join("bad.smart_primitive.json");
    fs::write(&path, wrong).unwrap();
    let error = SmartPrimitiveDocument::load(&path).unwrap_err();
    assert!(
        error.message().contains("does not identify a hierarchy node"),
        "{error}"
    );
}

#[test]
fn validation_rejects_pivot_node_without_pivot() {
    let source =
        fs::read_to_string(assets_dir().join("smart_primitives/soma/sit.smart_primitive.json"))
            .unwrap();
    // Node 2 holds the Interaction Box, not a Pivot.
    let wrong = source.replacen("\"pivot_node_id\": 3", "\"pivot_node_id\": 2", 1);
    assert_ne!(wrong, source);
    let temp = TempDir::new("pivot-without-pivot");
    let path = temp.0.join("bad.smart_primitive.json");
    fs::write(&path, wrong).unwrap();
    let error = SmartPrimitiveDocument::load(&path).unwrap_err();
    assert!(
        error
            .message()
            .contains("must identify a node with a Pivot component"),
        "{error}"
    );
}

#[test]
fn validation_rejects_missing_required_field() {
    let source = fs::read_to_string(scene_files().pop().unwrap()).unwrap();
    let wrong = source.replacen("      \"far\": 1000.0\n", "", 1);
    assert_ne!(wrong, source);
    let temp = TempDir::new("missing-field");
    let path = temp.0.join("bad.scene.json");
    fs::write(&path, wrong).unwrap();
    assert!(SceneDocument::load(&path).is_err());
}

#[test]
fn unknown_fields_are_tolerated() {
    let source = fs::read_to_string(scene_files().pop().unwrap()).unwrap();
    let extended = source.replacen(
        "\"schema_version\": 1,",
        "\"schema_version\": 1,\n  \"future_field\": {\"anything\": [1, 2, 3]},",
        1,
    );
    assert_ne!(extended, source);
    let temp = TempDir::new("unknown-fields");
    let path = temp.0.join("extended.scene.json");
    fs::write(&path, &extended).unwrap();
    let mut document = SceneDocument::load(&path).expect("unknown fields are ignored");
    // ...and are dropped on save.
    document.save(&path).unwrap();
    assert!(!fs::read_to_string(&path).unwrap().contains("future_field"));
}

#[test]
fn load_requires_the_document_suffix() {
    let temp = TempDir::new("suffix");
    let path = temp.0.join("not_a_scene.json");
    fs::copy(scene_files().pop().unwrap(), &path).unwrap();
    let error = SceneDocument::load(&path).unwrap_err();
    assert!(
        error.message().contains("Scene documents must end in .scene.json"),
        "{error}"
    );
}

#[test]
fn euler_degrees_follow_rotation_xyzw_on_save() {
    let temp = TempDir::new("euler-sync");
    let mut document = SceneDocument::load(scene_files().pop().unwrap()).unwrap();
    // Rotate the first Smart Primitive 90 degrees about z and save: the written
    // euler_degrees must be recomputed from the quaternion.
    document.smart_primitives[0].transform.rotation_xyzw =
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let path = temp.0.join("synced.scene.json");
    document.save(&path).unwrap();
    let saved = fs::read_to_string(&path).unwrap();
    assert!(
        saved.contains("90.0") || saved.contains("89.99"),
        "euler_degrees recomputed on save"
    );
    let reloaded = SceneDocument::load(&path).unwrap();
    assert_euler_close(
        reloaded.smart_primitives[0].euler_degrees,
        Vec3::new(0.0, 0.0, 90.0),
        "synced euler",
    );
}
