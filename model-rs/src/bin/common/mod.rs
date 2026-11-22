// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Helpers shared by the benchmark and the development binaries: order statistics and argument
//! handling that report bad input as errors instead of panicking.
#![allow(dead_code)]

use std::fmt::Display;
use std::str::FromStr;

use anyhow::Result;
use lexopt::ValueExt;

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

/// The positional arguments (an option is an error), for the tools that take no options.
pub fn positionals() -> Result<Vec<String>> {
    let mut parser = lexopt::Parser::from_env();
    let mut out = Vec::new();
    while let Some(arg) = parser.next()? {
        match arg {
            lexopt::Arg::Value(v) => out.push(v.string()?),
            other => return Err(other.unexpected().into()),
        }
    }
    Ok(out)
}
