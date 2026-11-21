# SPDX-License-Identifier: Apache-2.0
"""Download and verify the released checkpoints, without the sandbox or extra dependencies.

The files come from ``huggingface.co/nvidia/MotionBricks`` at the pinned revision this port was
tested with, over plain HTTPS (``urllib``). Every file is checked against the size and SHA-256 in
``assets/checkpoints.sha256`` (upstream's own checksum table). Files already present and correct
are not downloaded again. A download goes to a unique ``.part`` file next to its target, is
hashed while it streams, and is renamed into place only once its size and SHA-256 match, so an
interrupted or corrupt download never leaves a file that looks complete.

    motionbricks-mlx download                       # both models into checkpoints/MotionBricks
    motionbricks-mlx download --model unitree_g1 --dir /data/MotionBricks
"""

from __future__ import annotations

import hashlib
import os
import secrets
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import canonical_name

REPO = "nvidia/MotionBricks"
REVISION = "3cb362d9b95aa40462abcb4d8cfae9ffb6d97410"
URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
MANIFEST = Path(__file__).resolve().parent / "assets" / "checkpoints.sha256"
# Sample clips for the Runtime API examples, from upstream's Git LFS at the pinned commit.
CLIP_MANIFEST = Path(__file__).resolve().parent / "assets" / "clips.sha256"
UPSTREAM_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"  # NVlabs/motionbricks v0.2.0
CLIP_URL = "https://media.githubusercontent.com/media/NVlabs/motionbricks/{commit}/{path}"
FOLDERS = {"soma": "soma", "unitree_g1": "g1"}  # model name -> folder in the repository
TIMEOUT_S = 60  # a connection that delivers no data for this long fails instead of hanging
CHUNK = 1 << 20


@dataclass(frozen=True)
class Entry:
    sha256: str
    size: int
    path: str  # relative to the repository root, e.g. "g1/out/.../model-step=2000000.safetensors"


def manifest(path: Path = MANIFEST) -> list[Entry]:
    entries = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            sha, size, relative = line.split()
            entries.append(Entry(sha, int(size), relative))
    return entries


def files_for(model: str) -> list[Entry]:
    folder = FOLDERS[canonical_name(model)]  # ValueError for an unknown model
    return [e for e in manifest() if e.path.startswith(folder + "/")]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, entry: Entry) -> bool:
    return path.is_file() and path.stat().st_size == entry.size and sha256(path) == entry.sha256


def _stream(url: str, staged: Path) -> tuple[int, str]:
    """Download ``url`` into the new file ``staged``; returns (bytes, sha256 hex digest)."""
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "motionbricks-mlx"})
    with staged.open("xb") as out, urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def fetch(entry: Entry, dest: Path, quiet: bool = False, url: str | None = None, relative: str | None = None) -> Path:
    """Download one file into ``dest / entry.path`` (via a unique ``.part`` file) and verify it.

    Raises ``RuntimeError`` naming the file on a network error, a size mismatch or a SHA-256
    mismatch; the partial file is removed, so rerunning retries cleanly."""
    target = Path(dest) / (relative or entry.path)
    if verified(target, entry):
        if not quiet:
            print(f"{entry.path} (verified)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Unique per attempt, so two concurrent downloads of the same file cannot interleave.
    staged = target.with_name(f"{target.name}.{os.getpid()}-{secrets.token_hex(4)}.part")
    if not quiet:
        print(f"{entry.path} ({entry.size / 1e6:.1f} MB)")
    url = url or URL.format(repo=REPO, revision=REVISION, path=urllib.request.quote(entry.path))
    try:
        size, digest = _stream(url, staged)
    except BaseException as exc:
        staged.unlink(missing_ok=True)
        if isinstance(exc, Exception):
            raise RuntimeError(f"downloading {entry.path} from {url} failed: {exc}; rerun to retry") from exc
        raise
    if size != entry.size:
        staged.unlink()
        raise RuntimeError(f"size mismatch for {entry.path}: got {size} bytes, expected {entry.size} "
                           "(truncated or wrong file); rerun to retry")
    if digest != entry.sha256:
        staged.unlink()
        raise RuntimeError(f"checksum mismatch for {entry.path}: sha256 {digest}, expected {entry.sha256}")
    staged.replace(target)
    return target


def download(models=("soma", "unitree_g1"), dest: str | Path = "checkpoints/MotionBricks", quiet: bool = False) -> Path:
    """Fetch and verify the checkpoints of ``models``; returns ``dest``. The folder for a model is
    ``dest / "soma"`` or ``dest / "g1"``, which is what :func:`motionbricks_mlx.load` takes."""
    dest = Path(dest)
    for model in models:
        for entry in files_for(model):
            fetch(entry, dest, quiet)
    return dest


def checkpoint_dir(model: str, dest: str | Path = "checkpoints/MotionBricks") -> Path:
    return Path(dest) / FOLDERS[canonical_name(model)]


def download_clips(dest: str | Path = "checkpoints/motions", quiet: bool = False) -> Path:
    """Fetch and verify the sample clips the Runtime API examples use into ``dest`` (laid out like
    upstream's ``assets/motions``). They are NVIDIA sample data under the NVIDIA Sample Data
    Evaluation License, downloaded from upstream's repository."""
    dest = Path(dest)
    for entry in manifest(CLIP_MANIFEST):
        fetch(entry, dest, quiet, url=clip_url(entry), relative=entry.path.removeprefix("assets/motions/"))
    if not quiet:
        print("sample clips: NVIDIA Sample Data Evaluation License, see "
              f"https://github.com/NVlabs/motionbricks/blob/{UPSTREAM_COMMIT}/assets/motions/LICENSE.txt")
    return dest


def clip_url(entry: Entry) -> str:
    """Upstream's Git LFS media URL of a file at the pinned commit (``entry.path`` is repository-relative)."""
    return CLIP_URL.format(commit=UPSTREAM_COMMIT, path=urllib.request.quote(entry.path))
