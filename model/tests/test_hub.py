"""The checkpoint/clip downloader's verify-then-rename flow, offline (``file://`` URLs, no network)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from motionbricks_mlx import hub

ROOT = Path(__file__).resolve().parents[2]
SHELL_DOWNLOADER = ROOT / "model-rs" / "scripts" / "download_weights.sh"
PAYLOAD = b"motionbricks" * 1000


def entry_for(data: bytes, path: str = "g1/out/model.safetensors") -> hub.Entry:
    return hub.Entry(hashlib.sha256(data).hexdigest(), len(data), path)


@pytest.fixture
def source(tmp_path) -> Path:
    path = tmp_path / "remote" / "model.safetensors"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    return path


def leftovers(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.part"))


def test_fetch_downloads_verifies_and_renames(source, tmp_path):
    dest = tmp_path / "dest"
    target = hub.fetch(entry_for(PAYLOAD), dest, quiet=True, url=source.as_uri())
    assert target == dest / "g1/out/model.safetensors"
    assert target.read_bytes() == PAYLOAD
    assert leftovers(dest) == []


def test_fetch_skips_a_verified_file(source, tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    hub.fetch(entry_for(PAYLOAD), dest, quiet=True, url=source.as_uri())

    def no_network(*_):
        raise AssertionError("a verified file must not be downloaded again")

    monkeypatch.setattr(hub, "_stream", no_network)
    assert hub.fetch(entry_for(PAYLOAD), dest, quiet=True, url=source.as_uri()).read_bytes() == PAYLOAD


def test_fetch_replaces_a_corrupt_file(source, tmp_path):
    dest = tmp_path / "dest"
    target = dest / "g1/out/model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * len(PAYLOAD))  # right size, wrong bytes
    hub.fetch(entry_for(PAYLOAD), dest, quiet=True, url=source.as_uri())
    assert target.read_bytes() == PAYLOAD


def test_size_mismatch_is_reported_and_cleaned_up(source, tmp_path):
    dest = tmp_path / "dest"
    entry = hub.Entry(hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD) + 1, "g1/out/model.safetensors")
    with pytest.raises(RuntimeError, match=r"size mismatch for g1/out/model.safetensors: got 12000 bytes, expected 12001"):
        hub.fetch(entry, dest, quiet=True, url=source.as_uri())
    assert not (dest / entry.path).exists()
    assert leftovers(dest) == []


def test_checksum_mismatch_is_reported_and_cleaned_up(source, tmp_path):
    dest = tmp_path / "dest"
    entry = hub.Entry("0" * 64, len(PAYLOAD), "g1/out/model.safetensors")
    with pytest.raises(RuntimeError, match="checksum mismatch for g1/out/model.safetensors"):
        hub.fetch(entry, dest, quiet=True, url=source.as_uri())
    assert not (dest / entry.path).exists()
    assert leftovers(dest) == []


def test_network_error_names_the_file_and_cleans_up(tmp_path):
    dest = tmp_path / "dest"
    missing = (tmp_path / "remote" / "absent.safetensors").as_uri()
    with pytest.raises(RuntimeError, match=r"downloading g1/out/model.safetensors from .* failed: .*rerun to retry"):
        hub.fetch(entry_for(PAYLOAD), dest, quiet=True, url=missing)
    assert leftovers(dest) == []


def test_model_names():
    assert hub.checkpoint_dir("unitree_g1", "ckpt") == Path("ckpt/g1")
    assert hub.checkpoint_dir("g1", "ckpt") == Path("ckpt/g1")
    assert hub.checkpoint_dir("soma", "ckpt") == Path("ckpt/soma")
    assert {e.path.split("/")[0] for e in hub.files_for("unitree_g1")} == {"g1"}
    assert len(hub.files_for("soma")) == len(hub.files_for("g1")) == 5
    with pytest.raises(ValueError):
        hub.files_for("smpl")


@pytest.mark.skipif(not SHELL_DOWNLOADER.is_file(), reason="repository checkout only")
def test_shell_downloader_pins_the_same_revisions():
    """model-rs/scripts/download_weights.sh duplicates the pins for Python-free installs."""
    script = SHELL_DOWNLOADER.read_text()
    assert re.search(r'^REVISION="([0-9a-f]{40})"$', script, re.M).group(1) == hub.REVISION
    assert re.search(r'^UPSTREAM_COMMIT="([0-9a-f]{40})"$', script, re.M).group(1) == hub.UPSTREAM_COMMIT
    assert "checkpoints.sha256" in script and "clips.sha256" in script
