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
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:-all}"
DEST="${2:-$ROOT/checkpoints/MotionBricks}"
MANIFEST="$ROOT/model/motionbricks_mlx/assets/checkpoints.sha256"
REVISION="3cb362d9b95aa40462abcb4d8cfae9ffb6d97410"
URL="https://huggingface.co/nvidia/MotionBricks/resolve/$REVISION"
strip=""

if [[ "$MODEL" == "clips" ]]; then
  # NVIDIA sample data (NVIDIA Sample Data Evaluation License), from upstream's Git LFS at the
  # pinned commit, laid out like upstream's assets/motions.
  DEST="${2:-$ROOT/checkpoints/motions}"
  MANIFEST="$ROOT/model/motionbricks_mlx/assets/clips.sha256"
  URL="https://media.githubusercontent.com/media/NVlabs/motionbricks/54fc563aeaaf819a753d72cc9fdc6713989cdd43"
  strip="assets/motions/"
  MODEL=all
fi

case "$MODEL" in
  all) prefix="" ;;
  soma) prefix="soma/" ;;
  unitree_g1|g1) prefix="g1/" ;;
  *) echo "usage: $0 [all|soma|unitree_g1|clips] [destination]" >&2; exit 2 ;;
esac

ok() {  # file size sha256
  [[ -f "$1" ]] && [[ "$(stat -f%z "$1")" == "$2" ]] && [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$3" ]]
}

grep -v '^#' "$MANIFEST" | while read -r sha size path; do
  [[ -z "$sha" || "$path" != "$prefix"* ]] && continue
  file="$DEST/${path#"$strip"}"
  if ok "$file" "$size" "$sha"; then echo "$path (verified)"; continue; fi
  mkdir -p "$(dirname "$file")"
  echo "$path ($((size / 1000000)) MB)"
  curl --fail --location --silent --show-error -A motionbricks-rs -o "$file.part" "$URL/$path"
  if ! ok "$file.part" "$size" "$sha"; then rm -f "$file.part"; echo "checksum mismatch for $path" >&2; exit 1; fi
  mv "$file.part" "$file"
done
if [[ -n "$strip" ]]; then
  echo "clips: $DEST (NVIDIA Sample Data Evaluation License: https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/assets/motions/LICENSE.txt)"
else
  echo "checkpoints: $DEST/soma, $DEST/g1"
fi
