// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Export SOMA or Unitree G1 checkpoints to batched ONNX (the `motionbricks-onnx` CLI,
//! a drop-in replacement for upstream `uv run --project tools/onnx motionbricks-onnx`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use motionbricks_onnx::config::get_spec;
use motionbricks_onnx::export::{ExportPrecision, export_model};
use prost::Message;

const USAGE: &str = "Export SOMA or Unitree G1 checkpoints to batched ONNX.

Usage: motionbricks-onnx --model {soma|unitree_g1} --checkpoints DIR [options]

  --model NAME       Model to export (soma or unitree_g1)
  --checkpoints DIR  Downloaded checkpoint folder, such as checkpoints/MotionBricks/soma
  --precision P      fp32 (default), or fp16 for mixed FP16
  --output PATH      Output path (default: mmm_v6_{soma|g1}_{fp32|fp16}.onnx)
  --force            Replace an existing output after all checks pass
  --verbose          Show each export step
  -h, --help         Show this help
";

struct Args {
    model: String,
    checkpoints: PathBuf,
    precision: ExportPrecision,
    output: Option<PathBuf>,
    force: bool,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: String::new(),
        checkpoints: PathBuf::new(),
        precision: ExportPrecision::Fp32,
        output: None,
        force: false,
        verbose: false,
    };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            lexopt::Arg::Long("model") => args.model = parser.value().map_err(|e| e.to_string())?.into_string().map_err(|_| "--model is not UTF-8")?,
            lexopt::Arg::Long("checkpoints") => args.checkpoints = PathBuf::from(parser.value().map_err(|e| e.to_string())?),
            lexopt::Arg::Long("precision") => {
                let v = parser.value().map_err(|e| e.to_string())?;
                args.precision = match v.to_str() {
                    Some("fp32") => ExportPrecision::Fp32,
                    Some("fp16") => ExportPrecision::Fp16,
                    _ => return Err("--precision must be fp32 or fp16".to_string()),
                };
            }
            lexopt::Arg::Long("output") => args.output = Some(PathBuf::from(parser.value().map_err(|e| e.to_string())?)),
            lexopt::Arg::Long("force") => args.force = true,
            lexopt::Arg::Long("verbose") => args.verbose = true,
            lexopt::Arg::Short('h') | lexopt::Arg::Long("help") => return Err(USAGE.to_string()),
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    if args.model.is_empty() || args.checkpoints.as_os_str().is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(args)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let spec = get_spec(&args.model).map_err(|e| e.to_string())?;
    let output = args.output.unwrap_or_else(|| PathBuf::from(spec.default_output(args.precision.name())));
    if output.exists() && !args.force {
        return Err(format!("Output already exists: {}. Use --force to replace it.", output.display()));
    }
    let start = Instant::now();
    let model = export_model(&args.checkpoints, spec, args.precision).map_err(|e| format!("Export failed: {e:#}"))?;
    if args.verbose {
        let graph = model.graph.as_ref().expect("graph");
        eprintln!(
            "graph: {} nodes, {} initializers ({:.0} MB of weights)",
            graph.node.len(),
            graph.initializer.len(),
            graph.initializer.iter().map(|t| t.raw_data.len()).sum::<usize>() as f64 / 1e6
        );
    }
    let parent = output.parent().map(PathBuf::from).unwrap_or_default();
    std::fs::create_dir_all(&parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    // Stage in a temporary folder in the output directory, like upstream: a failed export
    // leaves no half-written graph, and a hard link fails if another process won the race.
    let staged = parent.join(format!(".{}.tmp", output.file_name().unwrap_or_default().to_string_lossy()));
    let result = (|| -> Result<(), String> {
        let bytes = model.encode_to_vec();
        std::fs::write(&staged, bytes).map_err(|e| format!("writing {}: {e}", staged.display()))?;
        if args.force {
            std::fs::rename(&staged, &output).map_err(|e| format!("replacing {}: {e}", output.display()))?;
        } else if let Err(e) = std::fs::hard_link(&staged, &output) {
            std::fs::remove_file(&staged).ok();
            return Err(format!("Output appeared meanwhile: {} ({e})", output.display()));
        } else {
            std::fs::remove_file(&staged).ok();
        }
        Ok(())
    })();
    if result.is_err() {
        std::fs::remove_file(&staged).ok();
    }
    result?;
    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "{}: {} ({:.0} MB) in {:.0}s",
        spec.name,
        output.display(),
        size as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            if e.contains("Usage:") {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
