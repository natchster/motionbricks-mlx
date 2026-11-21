"""Raw CUDA C++ compiled to Metal with the sandboxed CuMetal toolchain (optional layer)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[1] / "native"
CUMETALC = shutil.which("cumetalc") or next((str(p) for p in NATIVE.glob("toolchains/cumetal-*/bin/cumetalc")), None)
XCODE_DEVELOPER = Path("/Applications/Xcode.app/Contents/Developer")

pytestmark = pytest.mark.skipif(CUMETALC is None, reason="CuMetal not installed (sandbox/native/install_cumetal.sh; needs full Xcode)")


def toolchain_env() -> dict[str, str]:
    """cumetalc and the CuMetal runtime JIT both shell out to `xcrun metal`. If `xcode-select`
    still points at the Command Line Tools, aim xcrun at Xcode for these subprocesses only."""
    env = {**os.environ, "CUMETAL_TRACE_GPU": "1"}
    finds_metal = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, env=env).returncode == 0
    if not finds_metal and "DEVELOPER_DIR" not in env and XCODE_DEVELOPER.is_dir():
        env["DEVELOPER_DIR"] = str(XCODE_DEVELOPER)
    return env


@pytest.mark.parametrize("sample", ["vector_add", "reduce_shared", "sgemm_cublas"])
def test_sample_runs_on_apple_gpu(sample, tmp_path):
    env, binary = toolchain_env(), tmp_path / sample
    subprocess.run([CUMETALC, str(NATIVE / "samples" / f"{sample}.cu"), "-o", str(binary)], check=True, env=env)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "device=apple_gpu" in result.stdout + result.stderr  # launched on the GPU, not a host fallback
