"""The model package's download manifests match upstream: checkpoints against checksums.py, sample
clips against the Git LFS pointers at the pinned commit."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "sandbox" / "upstream"


def load_fetch_assets():
    spec = importlib.util.spec_from_file_location("fetch_assets", ROOT / "sandbox/scripts/fetch_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (UPSTREAM / "tools/onnx/checksums.py").is_file(), reason="needs sandbox/upstream (run sandbox/setup.sh)")
def test_checkpoint_manifest_matches_upstream():
    from motionbricks_mlx import hub

    ours = {(e.sha256, e.size, e.path) for e in hub.manifest()}
    assert ours == load_fetch_assets().upstream_checksums()
    assert {path.split("/")[0] for _, _, path in ours} == {"soma", "g1"}


@pytest.mark.skipif(not (UPSTREAM / ".git").exists(), reason="needs sandbox/upstream (run sandbox/setup.sh)")
def test_clip_manifest_matches_upstream_lfs_pointers():
    from motionbricks_mlx import hub

    for entry in hub.manifest(hub.CLIP_MANIFEST):
        pointer = subprocess.run(["git", "-C", str(UPSTREAM), "show", f"{hub.UPSTREAM_COMMIT}:{entry.path}"],
                                 capture_output=True, text=True, check=True).stdout
        fields = dict(line.split(" ", 1) for line in pointer.strip().splitlines())
        assert fields["oid"] == f"sha256:{entry.sha256}" and int(fields["size"]) == entry.size, entry.path
