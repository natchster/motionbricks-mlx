"""The model package's download manifests match upstream: checkpoints against checksums.py, sample
clips against the Git LFS pointers at the pinned commit. Plus the sandbox's own fetch and
reference-oracle plumbing, offline."""

import hashlib
import importlib.util
import json
import subprocess

import pytest
from cudashim import reference
from cudashim.paths import ROOT, UPSTREAM


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


def test_fetch_lfs_replaces_a_pointer_with_the_verified_file(tmp_path, monkeypatch):
    """fetch_assets.fetch_lfs goes through motionbricks_mlx.hub.fetch (file:// instead of GitHub)."""
    fetch_assets = load_fetch_assets()
    payload = b"rig" * 5000
    source = tmp_path / "remote.npz"
    source.write_bytes(payload)
    upstream = tmp_path / "upstream"
    target = upstream / "tools/onnx/assets/rig.npz"
    target.parent.mkdir(parents=True)

    def pointer(oid, size):
        target.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n")

    monkeypatch.setattr(fetch_assets, "UPSTREAM", upstream)
    monkeypatch.setattr(fetch_assets.hub, "clip_url", lambda entry: source.as_uri())
    assert fetch_assets.UPSTREAM_COMMIT == fetch_assets.hub.UPSTREAM_COMMIT

    pointer("0" * 64, len(payload))  # wrong digest: refused, pointer left in place, no .part
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch_assets.fetch_lfs("tools/onnx/assets/rig.npz")
    assert fetch_assets.parse_pointer(target) is not None
    assert not list(upstream.rglob("*.part"))

    pointer(hashlib.sha256(payload).hexdigest(), len(payload))
    fetch_assets.fetch_lfs("tools/onnx/assets/rig.npz")
    assert target.read_bytes() == payload
    fetch_assets.fetch_lfs("tools/onnx/assets/rig.npz")  # now a real file: left alone


def test_reference_model_names():
    assert reference.kind("unitree_g1") == reference.kind("g1") == "g1"
    assert reference.kind("soma") == "soma"
    assert reference.checkpoint_dir("unitree_g1") == reference.CHECKPOINTS / "g1"
    with pytest.raises(ValueError):
        reference.kind("smpl")  # used to fall through to SOMA


def test_missing_checkpoints_say_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "CHECKPOINTS", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"5 soma checkpoint file\(s\) missing .*motionbricks-mlx download"):
        reference.require_checkpoints("soma")


def test_fixture_meta_records_the_oracle(tmp_path):
    pytest.importorskip("torch")
    path = reference.write_meta(tmp_path, reference.fixture_meta("test", "cpu", cases=["trace"]))
    meta = json.loads(path.read_text())
    assert meta["device"] == "cpu" and meta["generator"] == "test" and meta["cases"] == ["trace"]
    assert {"torch", "numpy", "python", "upstream_commit"} <= set(meta)
