# SPDX-License-Identifier: Apache-2.0
"""Download and verify the released checkpoints, without the sandbox or extra dependencies.

The files come from ``huggingface.co/nvidia/MotionBricks`` at the pinned revision this port was
tested with, over plain HTTPS (``urllib``). Every file is checked against the size and SHA-256 in
``assets/checkpoints.sha256`` (upstream's own checksum table). Files already present and correct
are not downloaded again.

    motionbricks-mlx download                       # both models into checkpoints/MotionBricks
    motionbricks-mlx download --model unitree_g1 --dir /data/MotionBricks
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "nvidia/MotionBricks"
REVISION = "3cb362d9b95aa40462abcb4d8cfae9ffb6d97410"
URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
MANIFEST = Path(__file__).resolve().parent / "assets" / "checkpoints.sha256"
FOLDERS = {"soma": "soma", "unitree_g1": "g1"}  # model name -> folder in the repository


@dataclass(frozen=True)
class Entry:
    sha256: str
    size: int
    path: str  # relative to the repository root, e.g. "g1/out/.../model-step=2000000.safetensors"


def manifest() -> list[Entry]:
    entries = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            sha, size, path = line.split()
            entries.append(Entry(sha, int(size), path))
    return entries


def files_for(model: str) -> list[Entry]:
    if model not in FOLDERS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(FOLDERS)}")
    return [e for e in manifest() if e.path.startswith(FOLDERS[model] + "/")]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, entry: Entry) -> bool:
    return path.is_file() and path.stat().st_size == entry.size and sha256(path) == entry.sha256


def fetch(entry: Entry, dest: Path, quiet: bool = False) -> Path:
    """Download one file into ``dest / entry.path`` (via a ``.part`` file) and verify it."""
    target = dest / entry.path
    if verified(target, entry):
        if not quiet:
            print(f"{entry.path} (verified)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".part")
    if not quiet:
        print(f"{entry.path} ({entry.size / 1e6:.1f} MB)")
    url = URL.format(repo=REPO, revision=REVISION, path=urllib.request.quote(entry.path))
    request = urllib.request.Request(url, headers={"User-Agent": "motionbricks-mlx"})
    with urllib.request.urlopen(request) as response, staged.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if staged.stat().st_size != entry.size or sha256(staged) != entry.sha256:
        staged.unlink()
        raise RuntimeError(f"checksum mismatch for {entry.path}")
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
    return Path(dest) / FOLDERS[model]
