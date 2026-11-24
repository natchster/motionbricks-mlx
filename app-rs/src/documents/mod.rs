// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! JSON document IO for the MotionBricks applications.
//!
//! Ports the shared document machinery of upstream
//! `app/runtime/common/io/document_io.py` (header and camera handling, transform
//! validation, relative references, atomic writes) and the two document models:
//! [`scene`] (`.scene.json`, upstream `scene_document.py`) and
//! [`smart_primitive`] (`.smart_primitive.json`, upstream
//! `smart_primitive_document.py`). Pure data and math only: no viewer, no engine.

pub mod scene;
pub mod smart_primitive;
pub mod transform;

pub use scene::{
    CharacterStart, CubeDimensions, SCENE_FORMAT, SCENE_SCHEMA_VERSION, SCENE_SUFFIX, SceneDocument,
    SceneItemRef, SmartPrimitiveInstance, SolidCube, Visibility,
};
pub use smart_primitive::{
    Axis, ConeSide, ConstraintType, IK_PIN_ALL_MASK, InteractionBoxComponent,
    MAX_TRANSITION_EARLY_BY_FRAMES, MAX_TRAJECTORY_SCALE, MIN_TRAJECTORY_SCALE,
    MODEL_MAX_TOKEN_COUNT, MODEL_MIN_TOKEN_COUNT, MODEL_POSES_PER_TOKEN, NodeComponent,
    OperationOrder, PivotComponent, PlaybackMode, ResolvedSequence, SMART_PRIMITIVE_FORMAT,
    SMART_PRIMITIVE_SCHEMA_VERSION, SMART_PRIMITIVE_SUFFIX, SceneNode, SmartPrimitiveDocument,
    TargetPoseComponent, TargetPoseSequence,
};
pub use transform::Transform;

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Raised for invalid document data or failed file operations.
///
/// Upstream has a `DocumentError` base class with `SceneDocumentError` and
/// `SmartPrimitiveDocumentError` subclasses; this port flattens them into one
/// type and keeps the upstream message texts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentError {
    message: String,
}

impl DocumentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for DocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DocumentError {}

pub type Result<T> = std::result::Result<T, DocumentError>;

/// MotionBricks rig types, upstream's `ModelRig` in `motionbricks/model/contract.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelRig {
    Soma,
    UnitreeG1,
}

impl ModelRig {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Soma => "soma",
            Self::UnitreeG1 => "unitree_g1",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "soma" => Some(Self::Soma),
            "unitree_g1" => Some(Self::UnitreeG1),
            _ => None,
        }
    }
}

impl fmt::Display for ModelRig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The editor camera saved in every document's `editor.camera` object.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    pub pos: Vec3,
    pub pivot: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraState {
    /// Upstream's `_load_camera()`: finite values, 0 < fov < 180, near > 0,
    /// far > near. Runs on load and on save, so a bad camera never reaches a file.
    pub fn validate(&self, location: &str) -> Result<()> {
        for (field, values) in [
            ("pos", self.pos.to_array()),
            ("pivot", self.pivot.to_array()),
            ("pitch", [self.pitch, 0.0, 0.0]),
            ("yaw", [self.yaw, 0.0, 0.0]),
            ("fov", [self.fov, 0.0, 0.0]),
            ("near", [self.near, 0.0, 0.0]),
            ("far", [self.far, 0.0, 0.0]),
        ] {
            if !values.iter().all(|value| value.is_finite()) {
                return Err(DocumentError::new(format!(
                    "{location}.{field} must be a finite number."
                )));
            }
        }
        if !(0.0 < self.fov && self.fov < 180.0) {
            return Err(DocumentError::new(format!(
                "{location}.fov must be above 0 and below 180 degrees."
            )));
        }
        if self.near <= 0.0 {
            return Err(DocumentError::new(format!(
                "{location}.near must be greater than zero."
            )));
        }
        if self.far <= self.near {
            return Err(DocumentError::new(format!(
                "{location}.far must be greater than the near plane."
            )));
        }
        Ok(())
    }
}

/// Check the shared document header, upstream's `DocumentType.read()`:
/// format string, schema version, and rig. Returns the parsed rig.
pub(crate) fn check_document_header(
    kind: &str,
    expected_format: &str,
    expected_schema_version: i64,
    file_name: &str,
    format: &str,
    schema_version: i64,
    rig: &str,
) -> Result<ModelRig> {
    if format != expected_format {
        return Err(DocumentError::new(format!(
            "Unsupported {kind} format '{format}'."
        )));
    }
    if schema_version < 1 {
        return Err(DocumentError::new(
            "document.schema_version must be at least 1.".to_string(),
        ));
    }
    if schema_version != expected_schema_version {
        return Err(DocumentError::new(format!(
            "Unsupported {kind} schema version {schema_version}."
        )));
    }
    ModelRig::from_name(rig)
        .ok_or_else(|| DocumentError::new(format!("{file_name} uses unknown rig '{rig}'.")))
}

/// Upstream's `_color()`: RGB channels from 0 to 1.
pub(crate) fn validate_color(color: &[f32; 3], location: &str) -> Result<()> {
    if color.iter().any(|channel| !channel.is_finite()) {
        return Err(DocumentError::new(format!(
            "{location} must be an array of 3 numbers."
        )));
    }
    if color.iter().any(|channel| !(0.0..=1.0).contains(channel)) {
        return Err(DocumentError::new(format!(
            "{location} values must be 0 to 1."
        )));
    }
    Ok(())
}

/// Deserialize a required field that may still be JSON `null`.
///
/// serde defaults a missing `Option<T>` field to `None`; upstream instead treats
/// a missing field as an error (`Missing {location}.{field}.`) while `null` is a
/// legal value. Routing the field through this function disables serde's Option
/// shortcut so a missing field errors and an explicit `null` becomes `None`.
pub(crate) fn deserialize_required_nullable<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    <Option<T> as serde::Deserialize>::deserialize(deserializer)
}

/// Read a document file as text, upstream's `load_json()`.
pub(crate) fn read_document_text(path: &Path) -> Result<String> {
    if !path.is_file() {
        return Err(DocumentError::new(format!(
            "Could not find document: {}.",
            path.display()
        )));
    }
    fs::read_to_string(path)
        .map_err(|error| DocumentError::new(format!("Could not read {}: {error}", path.display())))
}

/// Parse JSON into a document structure, upstream's JSON error mapping.
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(text: &str, path: &Path) -> Result<T> {
    serde_json::from_str(text)
        .map_err(|error| DocumentError::new(format!("Invalid JSON in {}: {error}", path.display())))
}

/// Pretty-print with a 2-space indent and a trailing newline, like upstream's
/// `json.dumps(document, indent=2, allow_nan=False) + "\n"`.
///
/// serde_json would silently write non-finite floats as `null`; callers run the
/// document's `validate()` first, which rejects them as upstream does.
pub(crate) fn serialize_pretty<T: Serialize>(value: &T, path: &Path) -> Result<String> {
    let mut text = serde_json::to_string_pretty(value).map_err(|error| {
        DocumentError::new(format!(
            "Could not serialize document for {}: {error}",
            path.display()
        ))
    })?;
    text.push('\n');
    Ok(text)
}

/// Replace the destination only after a temporary file in the same directory is
/// fully written and fsynced, upstream's `atomic_write()`.
pub fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let context = |error: std::io::Error| {
        DocumentError::new(format!(
            "Could not save document to {}: {error}",
            path.display()
        ))
    };
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(context)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| DocumentError::new(format!("Could not save document to {}.", path.display())))?;
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = format!(
        "{:x}{:x}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let temporary_path = parent.join(format!(".{}.{}.tmp", file_name.to_string_lossy(), unique));
    let result = (|| -> std::result::Result<(), std::io::Error> {
        let mut stream = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        stream
            .write_all(data)
            .and_then(|()| stream.sync_all())
            .and_then(|()| {
                drop(stream);
                fs::rename(&temporary_path, path)
            })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result.map_err(context)
}

/// Complete a partial document suffix, upstream's `DocumentType.normalize_path()`:
/// `name` → `name.scene.json`, `name.scene` → `name.scene.json`,
/// `name.json` → `name.scene.json`, case-insensitively.
pub(crate) fn normalize_document_path(path: &Path, suffix: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lower_name = name.to_lowercase();
    let stem_suffix = suffix.strip_suffix(".json").unwrap_or(suffix);
    let new_name = if lower_name.ends_with(suffix) {
        name
    } else if lower_name.ends_with(stem_suffix) {
        format!("{name}.json")
    } else if lower_name.ends_with(".json") {
        format!("{}{}", &name[..name.len() - 5], suffix)
    } else {
        format!("{name}{suffix}")
    };
    path.with_file_name(new_name)
}

/// The document name from a file path, upstream's `DocumentType.name_from_path()`:
/// the file name without the document suffix. Errors when the suffix is wrong or
/// the name would be empty.
pub(crate) fn document_name_from_path(path: &Path, suffix: &str, kind: &str) -> Result<String> {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !file_name.to_lowercase().ends_with(suffix) {
        return Err(DocumentError::new(format!(
            "{kind} documents must end in {suffix}."
        )));
    }
    let name = &file_name[..file_name.len() - suffix.len()];
    if name.is_empty() {
        return Err(DocumentError::new(format!(
            "A {kind} document must have a name."
        )));
    }
    Ok(name.to_string())
}

/// Resolve a stored reference against the document that holds it, upstream's
/// `resolve_reference()`: relative paths join the document's directory, absolute
/// paths stay unchanged. The result is normalized lexically (`.` and `..`
/// resolved) without touching the filesystem, so documents can be loaded and
/// round-tripped even where the referenced file does not exist.
pub fn resolve_reference_path(reference: &str, document_path: &Path) -> PathBuf {
    let reference_path = Path::new(reference);
    let joined = if reference_path.is_absolute() {
        reference_path.to_path_buf()
    } else {
        document_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(reference_path)
    };
    normalize_lexically(&joined)
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                let can_pop = normalized
                    .file_name()
                    .is_some_and(|name| name != "..");
                if can_pop {
                    normalized.pop();
                } else if !normalized.is_absolute() {
                    normalized.push("..");
                }
            }
        }
    }
    normalized
}
