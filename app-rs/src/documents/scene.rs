// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Scene documents (`.scene.json`), ported from upstream
//! `app/runtime/common/io/scene_document.py`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::smart_primitive::{SMART_PRIMITIVE_SUFFIX, SmartPrimitiveDocument};
use super::transform::Transform;
use super::{
    CameraState, DocumentError, ModelRig, Result, atomic_write, check_document_header,
    document_name_from_path, normalize_document_path, parse_json, read_document_text,
    resolve_reference_path, serialize_pretty, validate_color,
};

pub const SCENE_FORMAT: &str = "motionbricks.scene";
pub const SCENE_SCHEMA_VERSION: i64 = 1;
pub const SCENE_SUFFIX: &str = ".scene.json";

/// Where the character starts in the scene. It always stays at the top level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharacterStart {
    pub id: u32,
    pub transform: Transform,
    pub euler_degrees: Vec3,
}

impl CharacterStart {
    /// The scene item ID upstream reserves for the CharacterStart.
    pub const ID: u32 = 1;
}

/// Display toggles of a placed Smart Primitive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Visibility {
    pub target_poses: bool,
    pub interaction_boxes: bool,
}

/// A placed Smart Primitive, including its behavior class and display settings.
///
/// `transform` is relative to the parent item, or to world space without a
/// parent. `reference` is stored exactly as written in the file (relative to the
/// document); use [`SceneDocument::resolve_reference`] to resolve it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmartPrimitiveInstance {
    pub id: u32,
    pub name: String,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub parent_id: Option<u32>,
    pub transform: Transform,
    pub euler_degrees: Vec3,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub reference: Option<String>,
    pub visibility: Visibility,
    pub target_pose_color: [f32; 3],
    pub behavior: String,
}

/// Size of a Solid Cube in meters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CubeDimensions {
    pub width: f32,
    pub depth: f32,
    pub height: f32,
}

/// A Solid Cube placed by its center, with its size in meters and its color.
///
/// `transform` is relative to the parent item, or to world space without a parent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidCube {
    pub id: u32,
    pub name: String,
    #[serde(deserialize_with = "super::deserialize_required_nullable")]
    pub parent_id: Option<u32>,
    pub transform: Transform,
    pub euler_degrees: Vec3,
    pub dimensions: CubeDimensions,
    pub color: [f32; 3],
}

/// A borrowed scene item of any kind, upstream's `SceneItem` union.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneItemRef<'a> {
    CharacterStart(&'a CharacterStart),
    SmartPrimitive(&'a SmartPrimitiveInstance),
    SolidCube(&'a SolidCube),
}

impl SceneItemRef<'_> {
    pub fn id(self) -> u32 {
        match self {
            Self::CharacterStart(item) => item.id,
            Self::SmartPrimitive(item) => item.id,
            Self::SolidCube(item) => item.id,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CharacterStart(_) => "CharacterStart",
            Self::SmartPrimitive(_) => "SmartPrimitive",
            Self::SolidCube(_) => "SolidCube",
        }
    }

    pub fn transform(self) -> Transform {
        match self {
            Self::CharacterStart(item) => item.transform,
            Self::SmartPrimitive(item) => item.transform,
            Self::SolidCube(item) => item.transform,
        }
    }

    /// The CharacterStart always stays at the top level (upstream `parent_id` is
    /// a `ClassVar[None]`).
    pub fn parent_id(self) -> Option<u32> {
        match self {
            Self::CharacterStart(_) => None,
            Self::SmartPrimitive(item) => item.parent_id,
            Self::SolidCube(item) => item.parent_id,
        }
    }
}

/// A scene: its CharacterStart, Smart Primitives, and Solid Cubes, plus the
/// editor selection and camera stored in the document header.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneDocument {
    pub name: String,
    pub rig: ModelRig,
    pub selected_item_id: u32,
    pub camera: CameraState,
    pub character_start: CharacterStart,
    pub smart_primitives: Vec<SmartPrimitiveInstance>,
    pub solid_cubes: Vec<SolidCube>,
}

#[derive(Deserialize)]
struct SceneFile {
    format: String,
    schema_version: i64,
    name: String,
    rig: String,
    editor: SceneEditorIn,
    character_start: CharacterStart,
    smart_primitives: Vec<SmartPrimitiveInstance>,
    solid_cubes: Vec<SolidCube>,
}

#[derive(Deserialize)]
struct SceneEditorIn {
    selected_item_id: i64,
    camera: CameraState,
}

#[derive(Serialize)]
struct SceneFileOut<'a> {
    format: &'static str,
    schema_version: i64,
    name: &'a str,
    rig: &'static str,
    editor: SceneEditorOut<'a>,
    character_start: &'a CharacterStart,
    smart_primitives: &'a [SmartPrimitiveInstance],
    solid_cubes: &'a [SolidCube],
}

#[derive(Serialize)]
struct SceneEditorOut<'a> {
    selected_item_id: u32,
    camera: &'a CameraState,
}

impl SceneDocument {
    /// Load a scene plus its editor selection and camera state, upstream's
    /// `load_scene_document()`. The file name must end in `.scene.json`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        document_name_from_path(path, SCENE_SUFFIX, "Scene")?;
        let file: SceneFile = parse_json(&read_document_text(path)?, path)?;
        let rig = check_document_header(
            "Scene",
            SCENE_FORMAT,
            SCENE_SCHEMA_VERSION,
            &path.file_name().unwrap_or_default().to_string_lossy(),
            &file.format,
            file.schema_version,
            &file.rig,
        )?;
        if file.editor.selected_item_id < 1 {
            return Err(DocumentError::new(
                "document.editor.selected_item_id must be at least 1.".to_string(),
            ));
        }
        file.editor.camera.validate("document.editor.camera")?;
        let mut scene = Self {
            name: file.name,
            rig,
            selected_item_id: file.editor.selected_item_id as u32,
            camera: file.editor.camera,
            character_start: file.character_start,
            smart_primitives: file.smart_primitives,
            solid_cubes: file.solid_cubes,
        };
        scene.validate()?;
        if scene.find_item(scene.selected_item_id).is_none() {
            scene.selected_item_id = scene.character_start.id;
        }
        Ok(scene)
    }

    /// Save the scene and return the path it was saved to, upstream's
    /// `save_scene_document()`.
    ///
    /// The path gets the `.scene.json` suffix when it lacks it, the document
    /// name is taken from the file name, and the previous file is replaced only
    /// after the write succeeds. `rotation_xyzw` is authoritative: every item's
    /// `euler_degrees` is recomputed from it before writing, upstream's sync rule.
    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<PathBuf> {
        self.sync_euler_degrees();
        self.validate()?;
        self.camera.validate("document.editor.camera")?;
        let document_path = normalize_document_path(path.as_ref(), SCENE_SUFFIX);
        let name = document_name_from_path(&document_path, SCENE_SUFFIX, "Scene")?;
        let document = SceneFileOut {
            format: SCENE_FORMAT,
            schema_version: SCENE_SCHEMA_VERSION,
            name: &name,
            rig: self.rig.as_str(),
            editor: SceneEditorOut {
                selected_item_id: self.selected_item_id,
                camera: &self.camera,
            },
            character_start: &self.character_start,
            smart_primitives: &self.smart_primitives,
            solid_cubes: &self.solid_cubes,
        };
        let json = serialize_pretty(&document, &document_path)?;
        atomic_write(&document_path, json.as_bytes())?;
        Ok(document_path)
    }

    /// Recompute every item's `euler_degrees` from its `rotation_xyzw`.
    pub fn sync_euler_degrees(&mut self) {
        self.character_start.euler_degrees = self.character_start.transform.euler_degrees();
        for instance in &mut self.smart_primitives {
            instance.euler_degrees = instance.transform.euler_degrees();
        }
        for cube in &mut self.solid_cubes {
            cube.euler_degrees = cube.transform.euler_degrees();
        }
    }

    /// Upstream's `_validate_scene()` and `_validate_hierarchy()`: the
    /// CharacterStart uses ID 1, item IDs are unique and positive, names are
    /// non-empty, cube dimensions are positive, and every parent is an existing
    /// Smart Primitive or Solid Cube with no cycles.
    pub fn validate(&self) -> Result<()> {
        if self.character_start.id != CharacterStart::ID {
            return Err(DocumentError::new(
                "CharacterStart must use scene item ID 1.".to_string(),
            ));
        }
        self.character_start
            .transform
            .validate("document.character_start.transform")?;
        validate_euler(self.character_start.euler_degrees, "document.character_start.euler_degrees")?;

        let mut used_ids = HashSet::from([self.character_start.id]);
        for (index, instance) in self.smart_primitives.iter().enumerate() {
            let location = format!("smart_primitives[{index}]");
            if instance.id < 1 || !used_ids.insert(instance.id) {
                return Err(DocumentError::new(format!(
                    "{location}.id must be unique and positive."
                )));
            }
            if instance.name.trim().is_empty() {
                return Err(DocumentError::new(format!(
                    "{location}.name cannot be empty."
                )));
            }
            instance
                .transform
                .validate(&format!("Smart Primitive '{}'.transform", instance.name))?;
            validate_euler(instance.euler_degrees, &format!("{location}.euler_degrees"))?;
            validate_color(
                &instance.target_pose_color,
                &format!("document.{location}.target_pose_color"),
            )?;
        }
        for (index, cube) in self.solid_cubes.iter().enumerate() {
            let location = format!("solid_cubes[{index}]");
            if cube.id < 1 || !used_ids.insert(cube.id) {
                return Err(DocumentError::new(format!(
                    "{location}.id must be unique and positive."
                )));
            }
            if cube.name.trim().is_empty() {
                return Err(DocumentError::new(format!(
                    "{location}.name cannot be empty."
                )));
            }
            cube.transform
                .validate(&format!("Solid Cube '{}'.transform", cube.name))?;
            validate_euler(cube.euler_degrees, &format!("{location}.euler_degrees"))?;
            for (attribute, value) in [
                ("width", cube.dimensions.width),
                ("depth", cube.dimensions.depth),
                ("height", cube.dimensions.height),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(DocumentError::new(format!(
                        "{location}.{attribute} must be greater than zero."
                    )));
                }
            }
            validate_color(&cube.color, &format!("document.{location}.color"))?;
        }

        for item in self
            .smart_primitives
            .iter()
            .map(SceneItemRef::SmartPrimitive)
            .chain(self.solid_cubes.iter().map(SceneItemRef::SolidCube))
        {
            let mut ancestors = HashSet::from([item.id()]);
            let mut parent_id = item.parent_id();
            while let Some(id) = parent_id {
                let parent = self.find_item(id).filter(|parent| {
                    !matches!(parent, SceneItemRef::CharacterStart(_))
                });
                let Some(parent) = parent else {
                    return Err(DocumentError::new(format!(
                        "'{}' must have a Smart Primitive or Solid Cube as its parent.",
                        self.item_name(item)
                    )));
                };
                if !ancestors.insert(id) {
                    return Err(DocumentError::new(format!(
                        "'{}' cannot sit under itself.",
                        self.item_name(item)
                    )));
                }
                parent_id = parent.parent_id();
            }
        }
        Ok(())
    }

    /// Check that every referenced Smart Primitive was saved for the scene's rig,
    /// upstream's `check_reference_rigs()`. References resolve relative to
    /// `document_path`, the file this scene was loaded from.
    pub fn check_reference_rigs(&self, document_path: impl AsRef<Path>) -> Result<()> {
        for (index, instance) in self.smart_primitives.iter().enumerate() {
            let Some(reference) = &instance.reference else {
                continue;
            };
            let location = format!("smart_primitives[{index}].reference");
            if !reference.to_lowercase().ends_with(SMART_PRIMITIVE_SUFFIX) {
                return Err(DocumentError::new(format!(
                    "{location} must reference a {SMART_PRIMITIVE_SUFFIX} file."
                )));
            }
            let reference_path = resolve_reference_path(reference, document_path.as_ref());
            if !reference_path.is_file() {
                return Err(DocumentError::new(format!(
                    "Could not find the file {location} references: {}.",
                    reference_path.display()
                )));
            }
            let rig = SmartPrimitiveDocument::load_rig(&reference_path)?;
            if rig != self.rig {
                return Err(DocumentError::new(format!(
                    "{} was saved for the '{rig}' rig, but this scene uses '{}'.",
                    reference_path.file_name().unwrap_or_default().to_string_lossy(),
                    self.rig
                )));
            }
        }
        Ok(())
    }

    /// Resolve an instance's `reference` against the scene's location, upstream's
    /// `resolve_reference()`.
    pub fn resolve_reference(reference: &str, document_path: impl AsRef<Path>) -> PathBuf {
        resolve_reference_path(reference, document_path.as_ref())
    }

    /// Return the scene item with `item_id`, or `None`. Upstream's `find_item()`.
    pub fn find_item(&self, item_id: u32) -> Option<SceneItemRef<'_>> {
        if item_id == self.character_start.id {
            return Some(SceneItemRef::CharacterStart(&self.character_start));
        }
        for item in &self.smart_primitives {
            if item.id == item_id {
                return Some(SceneItemRef::SmartPrimitive(item));
            }
        }
        for item in &self.solid_cubes {
            if item.id == item_id {
                return Some(SceneItemRef::SolidCube(item));
            }
        }
        None
    }

    /// Direct children in creation order (ascending ID); `None` selects the
    /// top-level items. Upstream's `children()`.
    pub fn children(&self, parent_id: Option<u32>) -> Vec<SceneItemRef<'_>> {
        let mut items: Vec<SceneItemRef<'_>> = self
            .smart_primitives
            .iter()
            .map(SceneItemRef::SmartPrimitive)
            .chain(self.solid_cubes.iter().map(SceneItemRef::SolidCube))
            .filter(|item| item.parent_id() == parent_id)
            .collect();
        items.sort_by_key(|item| item.id());
        items
    }

    /// The item's position and rotation in world space, upstream's
    /// `world_transform()`: the local transform composed through all parents.
    pub fn world_transform(&self, item_id: u32) -> Option<Transform> {
        let item = self.find_item(item_id)?;
        let mut transform = item.transform();
        let mut visited = HashSet::from([item.id()]);
        let mut parent_id = item.parent_id();
        while let Some(id) = parent_id {
            if !visited.insert(id) {
                return None;
            }
            let parent = self.find_item(id)?;
            transform = parent.transform().mul(&transform);
            parent_id = parent.parent_id();
        }
        Some(transform)
    }

    /// A copy where every item is at the top level, placed by its world
    /// transform. Upstream's `flatten()` (which, like upstream, leaves the
    /// redundant `euler_degrees` fields untouched).
    pub fn flatten(&self) -> SceneDocument {
        let mut flattened = self.clone();
        for instance in &mut flattened.smart_primitives {
            instance.transform = self
                .world_transform(instance.id)
                .expect("validated scene hierarchy");
            instance.parent_id = None;
        }
        for cube in &mut flattened.solid_cubes {
            cube.transform = self
                .world_transform(cube.id)
                .expect("validated scene hierarchy");
            cube.parent_id = None;
        }
        flattened
    }

    /// The next free scene item ID: one above the highest ID in use.
    pub fn next_free_id(&self) -> u32 {
        self.smart_primitives
            .iter()
            .map(|item| item.id)
            .chain(self.solid_cubes.iter().map(|item| item.id))
            .chain([self.character_start.id])
            .max()
            .unwrap_or(CharacterStart::ID)
            + 1
    }

    fn item_name<'a>(&self, item: SceneItemRef<'a>) -> &'a str {
        match item {
            SceneItemRef::CharacterStart(_) => "CharacterStart",
            SceneItemRef::SmartPrimitive(item) => &item.name,
            SceneItemRef::SolidCube(item) => &item.name,
        }
    }
}

fn validate_euler(euler_degrees: Vec3, location: &str) -> Result<()> {
    if !euler_degrees.is_finite() {
        return Err(DocumentError::new(format!(
            "{location} must be an array of 3 numbers."
        )));
    }
    Ok(())
}
