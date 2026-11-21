#!/usr/bin/env bash
# Download and verify the MotionBricks checkpoints without Python (for the Rust ports).
#
#   model-rs/scripts/download_weights.sh [all|soma|unitree_g1] [destination]
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

case "$MODEL" in
  all) prefix="" ;;
  soma) prefix="soma/" ;;
  unitree_g1|g1) prefix="g1/" ;;
  *) echo "usage: $0 [all|soma|unitree_g1] [destination]" >&2; exit 2 ;;
esac

ok() {  # file size sha256
  [[ -f "$1" ]] && [[ "$(stat -f%z "$1")" == "$2" ]] && [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$3" ]]
}

grep -v '^#' "$MANIFEST" | while read -r sha size path; do
  [[ -z "$sha" || "$path" != "$prefix"* ]] && continue
  file="$DEST/$path"
  if ok "$file" "$size" "$sha"; then echo "$path (verified)"; continue; fi
  mkdir -p "$(dirname "$file")"
  echo "$path ($((size / 1000000)) MB)"
  curl --fail --location --silent --show-error -A motionbricks-rs -o "$file.part" "$URL/$path"
  if ! ok "$file.part" "$size" "$sha"; then rm -f "$file.part"; echo "checksum mismatch for $path" >&2; exit 1; fi
  mv "$file.part" "$file"
done
echo "checkpoints: $DEST/soma, $DEST/g1"
