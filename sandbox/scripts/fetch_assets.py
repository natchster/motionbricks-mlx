"""Fetch everything the sandbox needs, verifying every byte against a published digest.

* Upstream source: NVlabs/motionbricks cloned at a pinned commit into sandbox/upstream/.
* Git-LFS assets (rig .npz, sample clips, ONNX graphs): downloaded from GitHub's media
  endpoint and checked against the SHA-256 ``oid`` stored in each LFS pointer, so
  git-lfs itself is not required.
* Checkpoints: the SafeTensors + stats files from huggingface.co/nvidia/MotionBricks, fetched
  by the model package's own downloader (``motionbricks_mlx.hub``, also usable without the
  sandbox). Its size/SHA-256 manifest is first checked against upstream ``tools/onnx/checksums.py``.

Downloads use ``motionbricks_mlx.hub.fetch``: a unique ``.part`` file, hashed while it streams,
renamed into place only when size and SHA-256 match.

Usage (from the repo root):  uv run --all-packages python sandbox/scripts/fetch_assets.py [--onnx] [--motions]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from cudashim.paths import CHECKPOINTS, UPSTREAM
from motionbricks_mlx import hub

UPSTREAM_REPO = "https://github.com/NVlabs/motionbricks.git"
UPSTREAM_COMMIT = hub.UPSTREAM_COMMIT  # v0.2.0, the commit the package's manifests come from

# LFS files the reference oracle and tests need. The rig archives are tiny and required.
RIG_FILES = ["tools/onnx/assets/g1_model_rig.npz", "tools/onnx/assets/soma_model_rig.npz"]
ONNX_FILES = [
    "motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx",
    "motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx",
]
MOTION_GLOBS = ["assets/motions/unitree_g1/**/*.csv", "assets/motions/soma/**/*.bvh"]


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
    """Replace the Git LFS pointer at ``relpath`` with the file it points to, verified against its oid."""
    target = UPSTREAM / relpath
    pointer = parse_pointer(target)
    if pointer is None:
        print(f"lfs: {relpath} (present)")
        return
    oid, size = pointer
    entry = hub.Entry(oid, size, relpath)
    print(f"lfs: {relpath} ({size / 1e6:.1f} MB)")
    hub.fetch(entry, UPSTREAM, quiet=True, url=hub.clip_url(entry))


def upstream_checksums() -> set[tuple[str, int, str]]:
    """(sha256, bytes, repository path) of every checkpoint file in upstream's checksums.py."""
    spec = importlib.util.spec_from_file_location("checksums", UPSTREAM / "tools/onnx/checksums.py")
    checksums = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checksums)
    folders = {"soma": "soma", "g1": "g1"}
    return {
        (e["sha256"], e["bytes"], f"{folders[kind]}/{release['root_directory']}/{e['path']}")
        for kind, release in checksums.CHECKPOINTS.items()
        for e in release["files"]
    }


def fetch_checkpoints() -> None:
    from motionbricks_mlx import hub

    ours = {(e.sha256, e.size, e.path) for e in hub.manifest()}
    if ours != upstream_checksums():
        raise SystemExit("motionbricks_mlx/assets/checkpoints.sha256 differs from upstream checksums.py")
    hub.download(dest=CHECKPOINTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", action="store_true", help="also fetch the FP32 ONNX graphs (~1.5 GB)")
    parser.add_argument("--motions", action="store_true", help="also fetch the sample BVH/CSV clips")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip the Hugging Face checkpoints")
    args = parser.parse_args()

    try:
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
    except RuntimeError as exc:  # download / verification failures from motionbricks_mlx.hub
        raise SystemExit(f"error: {exc}") from exc
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
