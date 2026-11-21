#!/usr/bin/env bash
# Download and verify the MotionBricks checkpoints without Python (for the Rust ports).
#
#   model-rs/scripts/download_weights.sh [all|soma|unitree_g1] [destination]
#   model-rs/scripts/download_weights.sh clips [destination]   # sample clips for the Runtime API examples
#
# Files come from huggingface.co/nvidia/MotionBricks at the revision this port was tested with,
# and are checked against model/motionbricks_mlx/assets/checkpoints.sha256 (size + SHA-256,
# from upstream's checksums.py). Verified files are skipped. Default destination:
# <repo>/checkpoints/MotionBricks, which holds soma/ and g1/.
#
# A file downloads to <name>.part and is renamed only after its size and SHA-256 match. An
# interrupted download resumes from its .part on the next run; a .part that then fails
# verification is deleted, so the run after that starts clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# The pins equal motionbricks_mlx.hub.REVISION / UPSTREAM_COMMIT (checked by model/tests/test_hub.py).
REVISION="3cb362d9b95aa40462abcb4d8cfae9ffb6d97410"
UPSTREAM_COMMIT="54fc563aeaaf819a753d72cc9fdc6713989cdd43"
MODEL="${1:-all}"
DEST="${2:-$ROOT/checkpoints/MotionBricks}"
MANIFEST="$ROOT/model/motionbricks_mlx/assets/checkpoints.sha256"
URL="https://huggingface.co/nvidia/MotionBricks/resolve/$REVISION"
LICENSE_URL="https://github.com/NVlabs/motionbricks/blob/$UPSTREAM_COMMIT/assets/motions/LICENSE.txt"
strip=""
# Retry transient failures; fail a connection that stalls (under 1 KB/s for 60 s) instead of hanging.
CURL=(curl --fail --location --silent --show-error -A motionbricks-rs
      --retry 3 --retry-delay 2 --connect-timeout 30 --speed-limit 1024 --speed-time 60)

if [[ "$MODEL" == "clips" ]]; then
  # NVIDIA sample data (NVIDIA Sample Data Evaluation License), from upstream's Git LFS at the
  # pinned commit, laid out like upstream's assets/motions.
  DEST="${2:-$ROOT/checkpoints/motions}"
  MANIFEST="$ROOT/model/motionbricks_mlx/assets/clips.sha256"
  URL="https://media.githubusercontent.com/media/NVlabs/motionbricks/$UPSTREAM_COMMIT"
  strip="assets/motions/"
  MODEL=all
fi

case "$MODEL" in
  all) prefix="" ;;
  soma) prefix="soma/" ;;
  unitree_g1|g1) prefix="g1/" ;;
  *) echo "usage: $0 [all|soma|unitree_g1|clips] [destination]" >&2; exit 2 ;;
esac

size_of() { wc -c < "$1" | tr -d '[:space:]'; }

ok() {  # file size sha256
  [[ -f "$1" ]] && [[ "$(size_of "$1")" == "$2" ]] && [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$3" ]]
}

download() {  # url part expected_size: resume a partial .part where the server allows it
  local rc=0
  # A .part at least as large as the file cannot be a prefix of it: start over.
  if [[ -f "$2" ]] && (( $(size_of "$2") >= $3 )); then rm -f "$2"; fi
  "${CURL[@]}" --continue-at - --output "$2" "$1" || rc=$?
  if (( rc == 33 )); then  # the server refused the byte range: download the whole file
    rm -f "$2"
    rc=0
    "${CURL[@]}" --output "$2" "$1" || rc=$?
  fi
  return "$rc"
}

grep -v '^#' "$MANIFEST" | while read -r sha size path; do
  [[ -z "$sha" || "$path" != "$prefix"* ]] && continue
  file="$DEST/${path#"$strip"}"
  if ok "$file" "$size" "$sha"; then echo "$path (verified)"; continue; fi
  mkdir -p "$(dirname "$file")"
  echo "$path ($((size / 1000000)) MB)"
  if ! download "$URL/$path" "$file.part" "$size"; then
    echo "download of $path failed; rerun to resume" >&2
    exit 1
  fi
  if ! ok "$file.part" "$size" "$sha"; then
    got="$(size_of "$file.part")"
    rm -f "$file.part"
    if [[ "$got" != "$size" ]]; then
      echo "size mismatch for $path: got $got bytes, expected $size; rerun to retry" >&2
    else
      echo "checksum mismatch for $path; rerun to retry" >&2
    fi
    exit 1
  fi
  mv "$file.part" "$file"
done
if [[ -n "$strip" ]]; then
  echo "clips: $DEST (NVIDIA Sample Data Evaluation License: $LICENSE_URL)"
else
  echo "checkpoints: $DEST/soma, $DEST/g1"
fi
