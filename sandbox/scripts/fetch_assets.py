"""Fetch everything the sandbox needs, verifying every byte against a published digest.

* Upstream source: NVlabs/motionbricks cloned at a pinned commit into sandbox/upstream/.
* Git-LFS assets (rig .npz, sample clips, ONNX graphs): downloaded from GitHub's media
  endpoint and checked against the SHA-256 ``oid`` stored in each LFS pointer, so
  git-lfs itself is not required.
* Checkpoints: the SafeTensors + stats files from huggingface.co/nvidia/MotionBricks,
  checked against the size/SHA-256 table in upstream ``tools/onnx/checksums.py``.

Usage (from the repo root):  uv run python sandbox/scripts/fetch_assets.py [--onnx] [--motions]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "https://github.com/NVlabs/motionbricks.git"
UPSTREAM_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"  # v0.2.0
MEDIA_URL = "https://media.githubusercontent.com/media/NVlabs/motionbricks/{commit}/{path}"
HF_REPO = "nvidia/MotionBricks"
HF_REVISION = "3cb362d9b95aa40462abcb4d8cfae9ffb6d97410"  # the weights this port was tested with

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "sandbox" / "upstream"
CHECKPOINTS = ROOT / "checkpoints" / "MotionBricks"

# LFS files the reference oracle and tests need. The rig archives are tiny and required.
RIG_FILES = ["tools/onnx/assets/g1_model_rig.npz", "tools/onnx/assets/soma_model_rig.npz"]
ONNX_FILES = [
    "motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx",
    "motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx",
]
MOTION_GLOBS = ["assets/motions/unitree_g1/**/*.csv", "assets/motions/soma/**/*.bvh"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clone_upstream() -> None:
    if (UPSTREAM / ".git").is_dir():
        head = subprocess.run(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if head == UPSTREAM_COMMIT:
            print(f"upstream: already at {UPSTREAM_COMMIT[:12]}")
            return
        raise SystemExit(f"{UPSTREAM} is at {head}, expected {UPSTREAM_COMMIT}. Remove it and rerun.")
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", UPSTREAM_REPO, str(UPSTREAM)], check=True, env=env)
    subprocess.run(["git", "-C", str(UPSTREAM), "checkout", "--quiet", UPSTREAM_COMMIT], check=True, env=env)
    print(f"upstream: cloned at {UPSTREAM_COMMIT[:12]}")


def parse_pointer(path: Path) -> tuple[str, int] | None:
    """Return (sha256, size) if path is an LFS pointer, else None."""
    if path.stat().st_size > 1024:
        return None
    text = path.read_text(errors="replace")
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines())
    return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def fetch_lfs(relpath: str) -> None:
    target = UPSTREAM / relpath
    pointer = parse_pointer(target)
    if pointer is None:
        print(f"lfs: {relpath} (present)")
        return
    oid, size = pointer
    url = MEDIA_URL.format(commit=UPSTREAM_COMMIT, path=relpath)
    staged = target.with_suffix(target.suffix + ".part")
    print(f"lfs: {relpath} ({size / 1e6:.1f} MB)")
    with urllib.request.urlopen(url) as response, staged.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if staged.stat().st_size != size or sha256(staged) != oid:
        staged.unlink()
        raise SystemExit(f"checksum mismatch for {relpath}")
    staged.replace(target)


def fetch_checkpoints() -> None:
    from huggingface_hub import hf_hub_download

    spec = importlib.util.spec_from_file_location("checksums", UPSTREAM / "tools/onnx/checksums.py")
    checksums = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checksums)
    hf_dirs = {"soma": "soma", "g1": "g1"}
    for kind, release in checksums.CHECKPOINTS.items():
        for entry in release["files"]:
            rel = f"{hf_dirs[kind]}/{release['root_directory']}/{entry['path']}"
            local = CHECKPOINTS / rel
            if local.is_file() and local.stat().st_size == entry["bytes"] and sha256(local) == entry["sha256"]:
                print(f"hf: {rel} (verified)")
                continue
            print(f"hf: {rel} ({entry['bytes'] / 1e6:.1f} MB)")
            hf_hub_download(HF_REPO, rel, revision=HF_REVISION, local_dir=CHECKPOINTS)
            if local.stat().st_size != entry["bytes"] or sha256(local) != entry["sha256"]:
                raise SystemExit(f"checksum mismatch for {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", action="store_true", help="also fetch the FP32 ONNX graphs (~1.5 GB)")
    parser.add_argument("--motions", action="store_true", help="also fetch the sample BVH/CSV clips")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip the Hugging Face checkpoints")
    args = parser.parse_args()

    clone_upstream()
    for rel in RIG_FILES:
        fetch_lfs(rel)
    if args.motions:
        for pattern in MOTION_GLOBS:
            for path in sorted(UPSTREAM.glob(pattern)):
                fetch_lfs(str(path.relative_to(UPSTREAM)))
    if args.onnx:
        for rel in ONNX_FILES:
            fetch_lfs(rel)
    if not args.no_checkpoints:
        fetch_checkpoints()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
