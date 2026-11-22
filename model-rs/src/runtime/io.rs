// SPDX-License-Identifier: Apache-2.0
// Runtime API of motionbricks-rs, ported from NVlabs/motionbricks (io/bvh.py, io/csv.py,
// utils/space_conversion_utils.py).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Clips (BVH for SOMA, CSV for the G1) and the rig-space conventions.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use super::animation::{AnimationBuffer, DofAnimationBuffer, Skeleton, create_animation_buffer_for_skeleton};
use super::contract::{ModelRig, ROOT_COORDINATE_COUNT};
use super::math::*;

/// Sample rate of the robot motion CSV clips (the files do not store it).
pub const CSV_SAMPLE_RATE: f32 = 120.0;
const CSV_ROOT_TITLE: [&str; 7] = ["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"];

struct Hierarchy {
    skeleton: Skeleton,
    channels: Vec<Vec<String>>,
}

/// What a BVH channel animates.
#[derive(Clone, Copy)]
enum Channel {
    Position(usize),
    Rotation(usize),
    Other,
}

impl Channel {
    /// Only position and rotation channels need an axis (`Xposition`, `Zrotation`, ...).
    fn parse(name: &str) -> Result<Self> {
        let c = name.to_lowercase();
        let axis = || match c.as_bytes().first() {
            Some(b'x') => Ok(0),
            Some(b'y') => Ok(1),
            Some(b'z') => Ok(2),
            _ => bail!("unknown BVH channel {name:?}"),
        };
        Ok(if c.ends_with("position") {
            Channel::Position(axis()?)
        } else if c.ends_with("rotation") {
            Channel::Rotation(axis()?)
        } else {
            Channel::Other
        })
    }
}

fn read_hierarchy<'a>(lines: &mut impl Iterator<Item = &'a str>, remove_namespace: bool) -> Result<Hierarchy> {
    let mut names: Vec<String> = Vec::new();
    let mut parents: Vec<i32> = Vec::new();
    let mut offsets: Vec<[f32; 3]> = Vec::new();
    let mut channels: Vec<Vec<String>> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut in_end_site = false;
    for line in lines.by_ref() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&first) = tokens.first() else { continue };
        match first {
            "MOTION" => break,
            "ROOT" | "JOINT" => {
                let name = tokens.get(1).context("joint without a name")?;
                names.push(if remove_namespace { name.rsplit(':').next().unwrap_or(name).to_string() } else { name.to_string() });
                parents.push(stack.last().map_or(-1, |&p| p as i32));
                stack.push(names.len() - 1);
                offsets.push([0.0; 3]);
                channels.push(Vec::new());
            }
            "End" if tokens.get(1) == Some(&"Site") => in_end_site = true,
            "OFFSET" if !in_end_site => {
                let j = *stack.last().context("OFFSET outside a joint")?;
                let values = tokens.get(1..4).with_context(|| format!("BVH joint {:?}: OFFSET needs three values", names[j]))?;
                for (k, v) in values.iter().enumerate() {
                    offsets[j][k] = v.parse().with_context(|| format!("BVH joint {:?}: OFFSET value {v:?} is not a number", names[j]))?;
                }
            }
            "CHANNELS" => {
                let j = *stack.last().context("CHANNELS outside a joint")?;
                let count = tokens.get(1).with_context(|| format!("BVH joint {:?}: CHANNELS needs a count", names[j]))?;
                let count: usize = count.parse().with_context(|| format!("BVH joint {:?}: channel count {count:?} is not a number", names[j]))?;
                if count != tokens.len() - 2 {
                    bail!("BVH joint {:?} declares {count} channels but lists {}", names[j], tokens.len() - 2);
                }
                channels[j] = tokens[2..].iter().map(|s| s.to_string()).collect();
            }
            "}" => {
                if in_end_site {
                    in_end_site = false;
                } else {
                    stack.pop();
                }
            }
            _ => {}
        }
    }
    let reference = offsets.iter().map(|o| Transform::new(Vec3::new(o[0] * 0.01, o[1] * 0.01, o[2] * 0.01), QUAT_IDENTITY)).collect();
    Ok(Hierarchy { skeleton: Skeleton::new(names, parents, reference)?, channels })
}

fn read_bvh_text(path: &Path) -> Result<String> {
    if path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() != Some("bvh") {
        bail!("expected a BVH file: {}", path.display());
    }
    std::fs::read_to_string(path).with_context(|| format!("could not read BVH file {}", path.display()))
}

fn without_root(s: &Skeleton) -> Result<Skeleton> {
    if s.num_joints() < 2 || s.joint_parent(1) != 0 || (1..s.num_joints()).any(|j| s.joint_parent(j) == -1) {
        bail!("remove_root needs a single root joint with a child at index 1");
    }
    let parents = s.parent_indices()[1..].iter().map(|&p| if p > 0 { p - 1 } else { -1 }).collect();
    let mut reference = s.reference_local_transforms()[1..].to_vec();
    reference[0] = s.reference_local_transforms()[0] * reference[0];
    Skeleton::new(s.joint_names()[1..].to_vec(), parents, reference)
}

/// The joint hierarchy and reference pose of a BVH file (offsets in cm become meters).
///
/// `remove_root` drops a leading `Root` joint that holds the identity, giving the `Hips`-rooted
/// skeleton the SOMA engine expects (upstream reads it from `soma_base_skel_minimal_no_root.usd`).
pub fn load_skeleton_from_bvh(path: impl AsRef<Path>, remove_namespace: bool, remove_root: bool) -> Result<Skeleton> {
    let text = read_bvh_text(path.as_ref())?;
    let skeleton = read_hierarchy(&mut text.lines(), remove_namespace)?.skeleton;
    if remove_root { without_root(&skeleton) } else { Ok(skeleton) }
}

/// Compose per-axis rotations in the file's channel order (`q = q0 * q1 * q2`), angles in degrees.
fn euler_to_quaternion(angles: &[f32], order: &[usize]) -> Quat {
    let mut q = QUAT_IDENTITY;
    for (angle, &axis) in angles.iter().zip(order) {
        let mut a = [0.0; 3];
        a[axis] = 1.0;
        q = quat_mul(q, quat_normalize(quat_from_axis_angle(Vec3(a), angle.to_radians())));
    }
    quat_normalize(q)
}

/// Load a BVH clip: positions cm -> m, rotation channels in degrees, in the order listed.
/// `skeleton` remaps the clip by joint name (missing joints keep its reference pose).
pub fn load_bvh(path: impl AsRef<Path>, skeleton: Option<Arc<Skeleton>>, remove_namespace: bool) -> Result<AnimationBuffer> {
    let path = path.as_ref();
    let text = read_bvh_text(path)?;
    let mut lines = text.lines();
    let Hierarchy { skeleton: file_skeleton, channels } = read_hierarchy(&mut lines, remove_namespace)?;
    let (mut frames, mut sample_rate) = (0usize, None);
    for line in lines.by_ref() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["Frames:", n, ..] => frames = n.parse().with_context(|| format!("BVH frame count {n:?} is not a number"))?,
            ["Frame", "Time:", t, ..] => {
                let dt: f64 = t.parse().with_context(|| format!("BVH frame time {t:?} is not a number"))?;
                if !(dt > 0.0 && dt.is_finite()) {
                    bail!("BVH frame time must be positive, got {dt}");
                }
                sample_rate = Some(((1.0 / dt) * 100.0).round() / 100.0);
                break;
            }
            _ => {}
        }
    }
    let sample_rate = sample_rate.with_context(|| format!("BVH frame time is missing: {}", path.display()))? as f32;
    let kinds: Vec<Vec<Channel>> = channels.iter().map(|c| c.iter().map(|n| Channel::parse(n)).collect::<Result<_>>()).collect::<Result<_>>()?;
    let width: usize = kinds.iter().map(Vec::len).sum();
    if width == 0 && frames > 0 {
        bail!("BVH declares {frames} frames but no channels: {}", path.display());
    }
    let samples: Vec<f32> = lines
        .flat_map(str::split_whitespace)
        .map(|v| v.parse::<f32>().with_context(|| format!("BVH motion value {v:?} is not a number")))
        .collect::<Result<_>>()?;
    if Some(samples.len()) != frames.checked_mul(width) {
        bail!("BVH motion has {} values; expected {frames} frames x {width} channels", samples.len());
    }
    let j = file_skeleton.num_joints();
    let mut poses = Vec::with_capacity(frames * j);
    let (mut angles, mut order) = (Vec::new(), Vec::new());
    for f in 0..frames {
        let row = &samples[f * width..(f + 1) * width];
        let mut column = 0;
        for (joint, joint_channels) in kinds.iter().enumerate() {
            let mut t = file_skeleton.reference_local_transforms()[joint];
            angles.clear();
            order.clear();
            for channel in joint_channels {
                match *channel {
                    Channel::Position(axis) => t.p.0[axis] = row[column] * 0.01,
                    Channel::Rotation(axis) => {
                        angles.push(row[column]);
                        order.push(axis);
                    }
                    Channel::Other => {}
                }
                column += 1;
            }
            if !angles.is_empty() {
                t.q = euler_to_quaternion(&angles, &order);
            }
            poses.push(t);
        }
    }
    let clip = AnimationBuffer::new(Arc::new(file_skeleton), sample_rate, poses, None, None)?;
    Ok(match skeleton {
        Some(s) => create_animation_buffer_for_skeleton(&clip, s),
        None => clip,
    })
}

/// `scipy.spatial.transform.Rotation.from_euler("xyz", ...)`: extrinsic X, then Y, then Z.
fn quat_from_euler_xyz_extrinsic(r: [f32; 3]) -> Quat {
    let axis = |i: usize| {
        let mut a = [0.0; 3];
        a[i] = 1.0;
        Vec3(a)
    };
    let (qx, qy, qz) = (quat_from_axis_angle(axis(0), r[0]), quat_from_axis_angle(axis(1), r[1]), quat_from_axis_angle(axis(2), r[2]));
    quat_normalize(quat_mul(qz, quat_mul(qy, qx)))
}

fn euler_xyz_extrinsic_from_quat(q: Quat) -> [f64; 3] {
    let (x, y, z, w) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

/// Load a robot motion CSV (cm, extrinsic XYZ degrees, joint degrees) as joint coordinates
/// (m, XYZW quaternion, radians), keeping the file's joint column order. Fields are trimmed;
/// blank lines and a UTF-8 BOM are skipped.
pub fn load_csv(path: impl AsRef<Path>, sample_rate: f32) -> Result<DofAnimationBuffer> {
    let path = path.as_ref();
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let header: Vec<String> = reader.headers().with_context(|| format!("reading {}", path.display()))?.iter().map(str::to_string).collect();
    if header.len() <= CSV_ROOT_TITLE.len() || header[..CSV_ROOT_TITLE.len()] != CSV_ROOT_TITLE {
        bail!("{}: expected the columns Frame and root_translateX to root_rotateZ, then one column per joint angle", path.display());
    }
    let names: Vec<String> = header[CSV_ROOT_TITLE.len()..].to_vec();
    let nc = ROOT_COORDINATE_COUNT + names.len();
    let mut poses = Vec::new();
    let mut record = csv::StringRecord::new();
    let mut row_index = 0;
    while reader.read_record(&mut record).with_context(|| format!("reading {}", path.display()))? {
        if record.len() == 1 && record[0].is_empty() {
            continue; // a line of whitespace
        }
        row_index += 1;
        let row: Vec<f64> = record
            .iter()
            .map(|v| v.parse::<f64>().with_context(|| format!("{}: motion row {row_index}: {v:?} is not a number", path.display())))
            .collect::<Result<_>>()?;
        if row.len() != header.len() {
            bail!("{}: row width does not match the CSV header", path.display());
        }
        let mut q = vec![0.0f32; nc];
        for k in 0..3 {
            q[k] = (row[1 + k] * 0.01) as f32;
        }
        let r = [row[4].to_radians() as f32, row[5].to_radians() as f32, row[6].to_radians() as f32];
        q[3..7].copy_from_slice(&quat_from_euler_xyz_extrinsic(r));
        for (k, v) in row[CSV_ROOT_TITLE.len()..].iter().enumerate() {
            q[ROOT_COORDINATE_COUNT + k] = v.to_radians() as f32;
        }
        if q.iter().any(|v| !v.is_finite()) {
            bail!("{}: motion coordinates must be finite", path.display());
        }
        poses.extend(q);
    }
    if poses.is_empty() {
        bail!("{}: CSV contains no motion frames", path.display());
    }
    DofAnimationBuffer::new(names, sample_rate, poses, None, None)
}

/// Write the frames in use of `buffer` as a robot motion CSV (fields quoted only when needed).
pub fn save_csv(path: impl AsRef<Path>, buffer: &DofAnimationBuffer) -> Result<()> {
    if buffer.num_frames() == 0 {
        bail!("buffer must contain at least one frame in use");
    }
    let path = path.as_ref();
    let mut out = csv::Writer::from_path(path).with_context(|| format!("writing {}", path.display()))?;
    out.write_record(CSV_ROOT_TITLE.iter().copied().chain(buffer.joint_names().iter().map(String::as_str)))?;
    let mut row = Vec::new();
    for f in 0..buffer.num_frames() {
        let q = buffer.frame(f);
        let e = euler_xyz_extrinsic_from_quat([q[3], q[4], q[5], q[6]]);
        row.clear();
        row.push(f.to_string());
        row.extend(q[..3].iter().map(|v| (v * 100.0).to_string()));
        row.extend(e.iter().map(|v| v.to_degrees().to_string()));
        row.extend(q[ROOT_COORDINATE_COUNT..].iter().map(|v| v.to_degrees().to_string()));
        out.write_record(&row)?;
    }
    out.flush().with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// How converted characters face, following Maya or MuJoCo conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacingDirectionType {
    Maya,
    Mujoco,
}

/// Convert transforms from Y-up source data to a Z-up world.
pub struct SpaceConverter {
    pub converter: Quat,
}

impl SpaceConverter {
    pub fn new(facing: FacingDirectionType) -> Self {
        let converter = match facing {
            FacingDirectionType::Mujoco => quat_from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 90f32.to_radians()),
            FacingDirectionType::Maya => {
                quat_mul(quat_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 90f32.to_radians()), quat_from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 90f32.to_radians()))
            }
        };
        SpaceConverter { converter }
    }
    pub fn convert_transform(&self, t: Transform) -> Transform {
        Transform::new(Vec3::ZERO, self.converter) * t
    }
}

/// Model space -> Z-up world: SOMA (Y up) turns into Z up facing like the G1 (MuJoCo convention);
/// the G1 model space is already Z up, so its transform is the identity.
pub fn model_to_runtime(rig: ModelRig) -> Transform {
    match rig {
        ModelRig::Soma => SpaceConverter::new(FacingDirectionType::Mujoco).convert_transform(Transform::IDENTITY),
        ModelRig::UnitreeG1 => Transform::IDENTITY,
    }
}
