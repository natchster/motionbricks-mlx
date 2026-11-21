// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Helpers shared by the benchmark and the development binaries: order statistics and argument
//! parsing that report bad input as errors instead of panicking.
#![allow(dead_code)]

use std::fmt::Display;
use std::str::FromStr;

use anyhow::{Context, Result, bail};

/// Median (the mean of the two middle values for even lengths; NaN when empty). Sorts `v`.
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Nearest-rank percentile, `p` in [0, 1] (NaN when empty). Sorts `v`.
pub fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// `value` parsed as `T`, with the argument's name in the error.
pub fn parse<T: FromStr>(value: &str, what: &str) -> Result<T>
where
    T::Err: Display,
{
    value.parse().map_err(|e| anyhow::anyhow!("invalid {what} {value:?}: {e}"))
}

/// Command-line arguments after the program name.
pub struct Argv {
    args: std::iter::Peekable<std::vec::IntoIter<String>>,
}

impl Argv {
    pub fn from_env() -> Self {
        Argv { args: std::env::args().skip(1).collect::<Vec<_>>().into_iter().peekable() }
    }

    pub fn next(&mut self) -> Option<String> {
        self.args.next()
    }

    /// The value after `flag`.
    pub fn value(&mut self, flag: &str) -> Result<String> {
        self.args.next().with_context(|| format!("{flag} needs a value"))
    }

    pub fn parsed<T: FromStr>(&mut self, flag: &str) -> Result<T>
    where
        T::Err: Display,
    {
        let v = self.value(flag)?;
        parse(&v, flag)
    }

    /// Every value after `flag` up to the next `--option` (at least one).
    pub fn values<T: FromStr>(&mut self, flag: &str) -> Result<Vec<T>>
    where
        T::Err: Display,
    {
        let mut out = Vec::new();
        while let Some(v) = self.args.next_if(|a| !a.starts_with("--")) {
            out.push(parse(&v, flag)?);
        }
        if out.is_empty() {
            bail!("{flag} needs at least one value");
        }
        Ok(out)
    }

    /// The remaining positional arguments parsed as `T` (`default` when there are none).
    pub fn rest<T: FromStr + Clone>(&mut self, what: &str, default: &[T]) -> Result<Vec<T>>
    where
        T::Err: Display,
    {
        let out = self.args.by_ref().map(|v| parse(&v, what)).collect::<Result<Vec<T>>>()?;
        Ok(if out.is_empty() { default.to_vec() } else { out })
    }
}
