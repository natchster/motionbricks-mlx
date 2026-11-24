// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Per-rig clip registry: each clip file loads once and is shared (ports upstream
//! `app/runtime/common/clip_registry.py`). BVH clips (SOMA) are remapped onto the registry's
//! skeleton by joint name; CSV clips (Unitree G1) must match the registry's joint order.
//! Looping variants are separate, corrected copies — the shared clip's poses never change.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use motionbricks_rs::runtime as rt;
use rt::{AnimationBuffer, DofAnimationBuffer, ModelRig, Skeleton};

/// A loaded clip: local joint transforms (SOMA) or joint coordinates (G1).
#[derive(Clone, Debug)]
pub enum Clip {
    /// SOMA clip (from BVH).
    Soma(AnimationBuffer),
    /// Unitree G1 clip (from CSV).
    G1(DofAnimationBuffer),
}

impl Clip {
    /// Frames in use.
    pub fn num_frames(&self) -> usize {
        match self {
            Clip::Soma(b) => b.num_frames(),
            Clip::G1(b) => b.num_frames(),
        }
    }

    /// Clip duration in seconds.
    pub fn duration(&self) -> f32 {
        match self {
            Clip::Soma(b) => b.duration(),
            Clip::G1(b) => b.duration(),
        }
    }

    /// The clip's sample rate.
    pub fn sample_rate(&self) -> f32 {
        match self {
            Clip::Soma(b) => b.sample_rate(),
            Clip::G1(b) => b.sample_rate(),
        }
    }

    /// The SOMA buffer, if this is one.
    pub fn as_soma(&self) -> Option<&AnimationBuffer> {
        match self {
            Clip::Soma(b) => Some(b),
            Clip::G1(_) => None,
        }
    }

    /// The G1 buffer, if this is one.
    pub fn as_g1(&self) -> Option<&DofAnimationBuffer> {
        match self {
            Clip::Soma(_) => None,
            Clip::G1(b) => Some(b),
        }
    }

    /// A copy of the frames in use (upstream `animation.copy()`).
    fn copy(&self) -> Clip {
        match self {
            Clip::Soma(b) => Clip::Soma(b.copy()),
            Clip::G1(b) => Clip::G1(b.copy()),
        }
    }

    /// Spread the first/last pose difference over the clip so its ends join
    /// (upstream `make_loopable`).
    fn make_loopable(&mut self) {
        match self {
            Clip::Soma(b) => b.make_loopable(),
            Clip::G1(b) => b.make_loopable(),
        }
    }
}

/// A source clip shared by target poses and playback. Its poses must not change.
#[derive(Debug)]
pub struct ClipAsset {
    /// Resolved path of the clip file.
    pub path: PathBuf,
    /// The loaded animation.
    pub clip: Clip,
}

impl ClipAsset {
    /// Clip name, from its file name without the suffix.
    pub fn name(&self) -> &str {
        self.path.file_stem().and_then(|s| s.to_str()).unwrap_or("")
    }

    /// Index of the clip's last frame.
    pub fn last_frame(&self) -> usize {
        self.clip.num_frames().saturating_sub(1)
    }

    /// Clip duration in seconds.
    pub fn duration(&self) -> f32 {
        self.clip.duration()
    }
}

/// Load each clip file once for one rig (upstream `ClipRegistry`).
///
/// SOMA registries remap BVH clips onto their skeleton (missing joints, e.g. fingers, keep
/// the skeleton's reference pose). G1 registries check that each CSV's joint order matches
/// the runtime G1 contract. Errors name the offending file.
pub struct ClipRegistry {
    rig: ModelRig,
    skeleton: Option<Arc<Skeleton>>,
    joint_names: Vec<String>,
    clips: HashMap<PathBuf, ClipAsset>,
}

impl ClipRegistry {
    /// A registry for SOMA `.bvh` clips, remapped onto `skeleton`.
    pub fn soma(skeleton: Arc<Skeleton>) -> ClipRegistry {
        ClipRegistry { rig: ModelRig::Soma, skeleton: Some(skeleton), joint_names: Vec::new(), clips: HashMap::new() }
    }

    /// A registry for Unitree G1 `.csv` clips; each clip's joint columns must match
    /// `joint_names` (use [`ClipRegistry::unitree_g1_contract`] for the runtime contract).
    pub fn unitree_g1(joint_names: Vec<String>) -> Result<ClipRegistry> {
        if joint_names.is_empty() {
            bail!("Unitree G1 clips need the joint order they are checked against");
        }
        Ok(ClipRegistry { rig: ModelRig::UnitreeG1, skeleton: None, joint_names, clips: HashMap::new() })
    }

    /// A G1 registry checked against the runtime contract's joint order
    /// (`UNITREE_G1_JOINT_NAMES`).
    pub fn unitree_g1_contract() -> ClipRegistry {
        Self::unitree_g1(rt::UNITREE_G1_JOINT_NAMES.iter().map(|s| s.to_string()).collect()).expect("the contract joint order is not empty")
    }

    /// The rig this registry serves.
    pub fn rig(&self) -> ModelRig {
        self.rig
    }

    /// File suffix of this registry's clips (`.bvh` or `.csv`).
    pub fn suffix(&self) -> &'static str {
        match self.rig {
            ModelRig::Soma => ".bvh",
            ModelRig::UnitreeG1 => ".csv",
        }
    }

    /// Number of clips loaded so far.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Whether no clip has been loaded yet.
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// All clip files of this registry's kind under `root`, recursively, sorted.
    pub fn list(&self, root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        list_clips(root.as_ref(), self.suffix())
    }

    /// Return the clip at `path`, loading it on first use. Failures name the file.
    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<&ClipAsset> {
        let path = path.as_ref();
        let suffix_matches = path.extension().and_then(|e| e.to_str()).is_some_and(|e| format!(".{}", e.to_lowercase()) == self.suffix());
        if !suffix_matches {
            bail!("this registry loads {} clips: {}", self.suffix(), path.display());
        }
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if !self.clips.contains_key(&key) {
            if !path.is_file() {
                bail!("could not find clip: {}", path.display());
            }
            let clip = self
                .load_file(path)
                .with_context(|| format!("could not load clip {}", path.display()))?;
            self.clips.insert(key.clone(), ClipAsset { path: key.clone(), clip });
        }
        let asset = self.clips.get(&key).with_context(|| format!("internal error: clip cache lost {}", path.display()))?;
        Ok(asset)
    }

    /// A copy of the clip at `path` whose last frame joins its first (upstream `load_looping`:
    /// the shared clip keeps its poses; the copy is corrected by `make_loopable`, which spreads
    /// the end/start difference over the whole clip). Each call returns a new copy.
    pub fn load_looping(&mut self, path: impl AsRef<Path>) -> Result<Clip> {
        let mut looping = self.load(path)?.clip.copy();
        looping.make_loopable();
        Ok(looping)
    }

    fn load_file(&self, path: &Path) -> Result<Clip> {
        match self.rig {
            ModelRig::Soma => {
                let skeleton = self.skeleton.as_ref().expect("a SOMA registry holds a skeleton").clone();
                Ok(Clip::Soma(rt::load_bvh(path, Some(skeleton), true)?))
            }
            ModelRig::UnitreeG1 => {
                let clip = rt::load_csv(path, rt::CSV_SAMPLE_RATE)?;
                if clip.joint_names() != self.joint_names {
                    bail!("the CSV joint order must match the G1 model's joint order");
                }
                Ok(Clip::G1(clip))
            }
        }
    }
}

/// All files under `root` with the given extension (case-insensitive, leading `.` optional),
/// recursively, sorted by path.
pub fn list_clips(root: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
    let suffix = suffix.trim_start_matches('.');
    if !root.is_dir() {
        bail!("clip root is not a directory: {}", root.display());
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.with_context(|| format!("could not list {}", root.display()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(suffix)) {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}
