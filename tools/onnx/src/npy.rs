// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Minimal `.npy` / `.npz` reader (little-endian numeric arrays, C order), enough for the
//! normalization statistics and the rig assets. Same parser as `model-rs/src/npy.rs`,
//! without the MLX array conversions.
//!
//! Parsing never panics: a malformed header, an unsupported dtype or a body shorter than the
//! shape needs is an error.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

#[derive(Debug, Clone)]
pub enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

impl NpyData {
    pub fn len(&self) -> usize {
        match self {
            NpyData::F32(v) => v.len(),
            NpyData::F64(v) => v.len(),
            NpyData::I32(v) => v.len(),
            NpyData::I64(v) => v.len(),
            NpyData::U8(v) => v.len(),
            NpyData::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One array. `shape` and `data` are public for building arrays by hand; [`Npy::new`] (or
/// [`Npy::check`]) verifies that they agree.
#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<i32>,
    pub data: NpyData,
}

/// Number of elements of `shape` (non-negative dims, no overflow).
fn element_count(shape: &[i32]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        ensure!(d >= 0, "negative dimension in shape {shape:?}");
        n.checked_mul(d as usize).with_context(|| format!("shape {shape:?} overflows"))
    })
}

/// `'value'...` -> `value`.
fn quoted(s: &str) -> Option<&str> {
    s.strip_prefix('\'')?.split_once('\'').map(|(v, _)| v)
}

impl Npy {
    /// An array whose data length matches its shape.
    pub fn new(shape: Vec<i32>, data: NpyData) -> Result<Npy> {
        let n = Npy { shape, data };
        n.check()?;
        Ok(n)
    }

    /// Fails if `data` does not hold exactly `product(shape)` elements.
    pub fn check(&self) -> Result<()> {
        let want = element_count(&self.shape)?;
        ensure!(self.data.len() == want, "array of shape {:?} needs {want} values, has {}", self.shape, self.data.len());
        Ok(())
    }

    pub fn parse(bytes: &[u8]) -> Result<Npy> {
        ensure!(bytes.len() >= 10 && bytes.starts_with(b"\x93NUMPY"), "not a .npy file");
        let (header_len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 => {
                let len = bytes.get(8..12).context("truncated .npy header")?;
                (u32::from_le_bytes(len.try_into()?) as usize, 12)
            }
            v => bail!("unsupported .npy version {v}"),
        };
        let header = bytes.get(start..start + header_len).context("truncated .npy header")?;
        let header = std::str::from_utf8(header).context(".npy header is not UTF-8")?;
        let field = |key: &str| -> Result<&str> {
            let at = header.find(key).with_context(|| format!("npy header lacks {key}"))? + key.len();
            Ok(header[at..].trim_start_matches([':', ' ']))
        };
        let descr = quoted(field("'descr'")?).context("malformed npy descr")?;
        let shape_text = field("'shape'")?.strip_prefix('(').and_then(|s| s.split_once(')')).context("malformed npy shape")?.0;
        let shape: Vec<i32> = shape_text
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>().with_context(|| format!("bad npy dimension {s:?}")))
            .collect::<Result<_>>()?;
        let count = element_count(&shape)?;
        // Fortran order only differs from C order when two or more dimensions exceed 1.
        if field("'fortran_order'")?.starts_with("True") && shape.iter().filter(|&&d| d > 1).count() > 1 {
            bail!("fortran-order arrays are not supported");
        }
        let elem = match descr {
            "<f4" | "<i4" => 4,
            "<f8" | "<i8" => 8,
            "|b1" | "|u1" => 1,
            other => bail!("unsupported npy dtype {other} (little-endian f4, f8, i4, i8, u1 and b1 are)"),
        };
        let need = count.checked_mul(elem).context("npy array too large")?;
        let body = &bytes[start + header_len..];
        ensure!(body.len() >= need, "npy body has {} bytes; shape {shape:?} of {descr} needs {need}", body.len());
        let body = &body[..need];
        fn read<T, const N: usize>(body: &[u8], f: fn([u8; N]) -> T) -> Vec<T> {
            body.chunks_exact(N).map(|c| f(c.try_into().expect("chunks_exact yields N bytes"))).collect()
        }
        let data = match descr {
            "<f4" => NpyData::F32(read(body, f32::from_le_bytes)),
            "<f8" => NpyData::F64(read(body, f64::from_le_bytes)),
            "<i4" => NpyData::I32(read(body, i32::from_le_bytes)),
            "<i8" => NpyData::I64(read(body, i64::from_le_bytes)),
            "|u1" => NpyData::U8(body.to_vec()),
            _ => NpyData::Bool(body.iter().map(|&b| b != 0).collect()),
        };
        Ok(Npy { shape, data })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Npy> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Npy::parse(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            NpyData::F32(v) => v.clone(),
            NpyData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as i32 as f32).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| x as i32 as f32).collect(),
        }
    }

    pub fn to_i32(&self) -> Vec<i32> {
        match &self.data {
            NpyData::F32(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::F64(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::I32(v) => v.clone(),
            NpyData::I64(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| x as i32).collect(),
        }
    }

    /// `x != 0` in the array's own dtype (NumPy semantics: 0.5 and NaN are true).
    pub fn to_bool(&self) -> Vec<bool> {
        match &self.data {
            NpyData::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
            NpyData::F64(v) => v.iter().map(|&x| x != 0.0).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x != 0).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x != 0).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x != 0).collect(),
            NpyData::Bool(v) => v.clone(),
        }
    }
}

/// All arrays of an `.npz` archive (stored or deflated members).
pub fn load_npz(path: impl AsRef<Path>) -> Result<HashMap<String, Npy>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_npz(zip::ZipArchive::new(file).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("reading {}", path.display()))
}

/// An `.npz` archive held in memory (e.g. `include_bytes!`).
pub fn load_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, Npy>> {
    read_npz(zip::ZipArchive::new(std::io::Cursor::new(bytes))?)
}

fn read_npz<R: std::io::Read + std::io::Seek>(mut archive: zip::ZipArchive<R>) -> Result<HashMap<String, Npy>> {
    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i)?;
        let name = member.name().to_string();
        // The size is the archive's claim; cap the up-front reservation.
        let mut bytes = Vec::with_capacity(member.size().min(1 << 26) as usize);
        member.read_to_end(&mut bytes).with_context(|| format!("reading member {name}"))?;
        let array = Npy::parse(&bytes).with_context(|| format!("parsing member {name}"))?;
        out.insert(name.strip_suffix(".npy").unwrap_or(&name).to_string(), array);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `.npy` file with a padded header (`header_len` overrides the stored length).
    fn npy(version: u8, descr: &str, shape: &str, body: &[u8], header_len: Option<u32>) -> Vec<u8> {
        let h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut out = b"\x93NUMPY".to_vec();
        out.extend([version, 0]);
        let len = header_len.unwrap_or(h.len() as u32);
        if version == 1 { out.extend((len as u16).to_le_bytes()) } else { out.extend(len.to_le_bytes()) }
        out.extend(h.as_bytes());
        out.extend(body);
        out
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn parses_supported_layouts() {
        let a = Npy::parse(&npy(1, "<f4", "(2, 3)", &f32_bytes(&[1., 2., 3., 4., 5., 6.]), None)).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
        let v2 = Npy::parse(&npy(2, "<i4", "(2,)", &[7, 0, 0, 0, 255, 255, 255, 255], None)).unwrap();
        assert_eq!(v2.to_i32(), vec![7, -1]);
        let scalar = Npy::parse(&npy(1, "<f8", "()", &2.5f64.to_le_bytes(), None)).unwrap();
        assert_eq!((scalar.shape.len(), scalar.to_f32()), (0, vec![2.5]));
        let wide = Npy::parse(&npy(1, "<i8", "(1,)", &(-3i64).to_le_bytes(), None)).unwrap();
        assert_eq!(wide.to_i32(), vec![-3]);
        let u8s = Npy::parse(&npy(1, "|u1", "(3,)", &[0, 2, 255], None)).unwrap();
        assert_eq!((u8s.to_i32(), u8s.to_bool()), (vec![0, 2, 255], vec![false, true, true]));
        let flags = Npy::parse(&npy(1, "|b1", "(2,)", &[1, 0], None)).unwrap();
        assert_eq!(flags.to_bool(), vec![true, false]);
        let empty = Npy::parse(&npy(1, "<f4", "(0, 4)", &[], None)).unwrap();
        assert!(empty.data.is_empty());
        // Trailing bytes are ignored, like NumPy.
        assert_eq!(Npy::parse(&npy(1, "<f4", "(1,)", &f32_bytes(&[1., 9.]), None)).unwrap().to_f32(), vec![1.]);
    }

    #[test]
    fn malformed_files_are_errors() {
        let ok = npy(1, "<f4", "(1, 8, 5)", &f32_bytes(&[0.0; 40]), None);
        assert!(Npy::parse(&ok).is_ok());
        let bad: Vec<(&str, Vec<u8>)> = vec![
            ("too short", b"\x93NUMPY\x01".to_vec()),
            ("bad magic", b"\x93NUMPX\x01\x00\x00\x00".to_vec()),
            ("v2 length cut", b"\x93NUMPY\x02\x00\x10\x00".to_vec()),
            ("header past end", npy(1, "<f4", "(1, 8, 5)", &[], Some(60_000))),
            ("truncated body", npy(1, "<f4", "(1, 8, 5)", &[0; 12], None)),
            ("truncated bools", npy(1, "|b1", "(4,)", &[1], None)),
            ("negative dim", npy(1, "<f4", "(-1,)", &[0; 4], None)),
            ("overflowing shape", npy(1, "<f4", "(2147483647, 2147483647, 8)", &[], None)),
            ("big-endian", npy(1, ">f4", "(1,)", &[0; 4], None)),
            ("unquoted descr", b"\x93NUMPY\x01\x00\x30\x00{'descr': f4, 'fortran_order': False, 'shape': (1,), }".to_vec()),
            ("shape without parens", b"\x93NUMPY\x01\x00\x30\x00{'descr': '<f4', 'fortran_order': False, 'shape': 1, }".to_vec()),
        ];
        for (what, bytes) in bad {
            assert!(Npy::parse(&bytes).is_err(), "{what} must be an error");
        }
        let fortran = |shape: &str, n: usize| {
            let h = format!("{{'descr': '<f4', 'fortran_order': True, 'shape': {shape}, }}");
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend((h.len() as u16).to_le_bytes());
            b.extend(h.as_bytes());
            b.extend(f32_bytes(&vec![0.0; n]));
            b
        };
        assert!(Npy::parse(&fortran("(2, 3)", 6)).is_err());
        assert!(Npy::parse(&fortran("(1, 3)", 3)).is_ok(), "one non-unit dim: same layout as C order");
    }

    #[test]
    fn masks_follow_numpy_nonzero() {
        let m = Npy::new(vec![4], NpyData::F32(vec![0.0, 0.5, f32::NAN, -0.0])).unwrap();
        assert_eq!(m.to_bool(), vec![false, true, true, false]);
        assert!(Npy::new(vec![2, 2], NpyData::I32(vec![1, 2, 3])).is_err());
        assert!(Npy::new(vec![-1], NpyData::I32(vec![])).is_err());
    }

    #[test]
    fn npz_member_names() {
        let arrays = load_npz_bytes(include_bytes!("../../../model/motionbricks_mlx/assets/soma_model_rig.npz")).unwrap();
        assert!(arrays.contains_key("capture_offsets") && arrays.contains_key("standard_offsets"));
        assert!(arrays.values().all(|a| a.check().is_ok()));
    }
}
