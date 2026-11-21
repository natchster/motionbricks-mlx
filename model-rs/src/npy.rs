// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Minimal `.npy` / `.npz` reader (little-endian numeric arrays, C order), enough for the
//! normalization statistics, the rig assets and the golden test fixtures.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;

#[derive(Debug, Clone)]
pub enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<i32>,
    pub data: NpyData,
}

impl Npy {
    pub fn parse(bytes: &[u8]) -> Result<Npy> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            bail!("not a .npy file");
        }
        let (header_len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12),
            v => bail!("unsupported .npy version {v}"),
        };
        let header = std::str::from_utf8(&bytes[start..start + header_len])?;
        let field = |key: &str| -> Result<&str> {
            let at = header.find(key).with_context(|| format!("npy header lacks {key}"))? + key.len();
            Ok(header[at..].trim_start_matches([':', ' ']))
        };
        let descr = field("'descr'")?;
        let descr = &descr[1..descr[1..].find('\'').unwrap() + 1];
        if field("'fortran_order'")?.starts_with("True") {
            bail!("fortran-order arrays are not supported");
        }
        let shape_text = field("'shape'")?;
        let shape_text = &shape_text[1..shape_text.find(')').unwrap()];
        let shape: Vec<i32> = shape_text
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>())
            .collect::<std::result::Result<_, _>>()?;
        let body = &bytes[start + header_len..];
        let count: usize = shape.iter().map(|&d| d as usize).product();
        fn read<T, const N: usize>(body: &[u8], count: usize, f: fn([u8; N]) -> T) -> Vec<T> {
            body.chunks_exact(N).take(count).map(|c| f(c.try_into().unwrap())).collect()
        }
        let data = match descr {
            "<f4" => NpyData::F32(read(body, count, f32::from_le_bytes)),
            "<f8" => NpyData::F64(read(body, count, f64::from_le_bytes)),
            "<i4" => NpyData::I32(read(body, count, i32::from_le_bytes)),
            "<i8" => NpyData::I64(read(body, count, i64::from_le_bytes)),
            "|b1" | "|u1" => NpyData::Bool(body[..count].iter().map(|&b| b != 0).collect()),
            other => bail!("unsupported npy dtype {other}"),
        };
        Ok(Npy { shape, data })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Npy> {
        let path = path.as_ref();
        Npy::parse(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            NpyData::F32(v) => v.clone(),
            NpyData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| x as i32 as f32).collect(),
        }
    }

    pub fn to_i32(&self) -> Vec<i32> {
        match &self.data {
            NpyData::F32(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::F64(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::I32(v) => v.clone(),
            NpyData::I64(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| x as i32).collect(),
        }
    }

    /// float32 MLX array (float64 data is narrowed, like the Python port's `np.asarray(.., float32)`).
    pub fn f32_array(&self) -> Array {
        Array::from_slice(&self.to_f32(), &self.shape)
    }

    pub fn i32_array(&self) -> Array {
        Array::from_slice(&self.to_i32(), &self.shape)
    }
}

/// All arrays of an `.npz` archive (stored or deflated members).
pub fn load_npz(path: impl AsRef<Path>) -> Result<HashMap<String, Npy>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_npz(zip::ZipArchive::new(file)?)
}

/// An `.npz` archive held in memory (e.g. `include_bytes!`).
pub fn load_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, Npy>> {
    read_npz(zip::ZipArchive::new(std::io::Cursor::new(bytes))?)
}

fn read_npz<R: std::io::Read + std::io::Seek>(mut archive: zip::ZipArchive<R>) -> Result<HashMap<String, Npy>> {
    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i)?;
        let name = member.name().trim_end_matches(".npy").to_string();
        let mut bytes = Vec::with_capacity(member.size() as usize);
        member.read_to_end(&mut bytes)?;
        out.insert(name, Npy::parse(&bytes)?);
    }
    Ok(out)
}
