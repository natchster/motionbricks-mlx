#!/usr/bin/env bash
# Build CuMetal (CUDA C++ -> Metal compiler + CUDA runtime/cuBLAS subset) into an isolated
# prefix inside this repo. Nothing is installed system-wide and PATH is not modified;
# `source sandbox/native/env.sh` activates it for one shell.
#
# Pinned to the reviewed v0.8.0 source release (Apache-2.0, github.com/Lulzx/cuda-metal);
# both tarballs are verified against the SHA-256 values of the upstream Homebrew formula.
# The optional DYLD-injection binary shim stays disabled (source-first mode).
#
# Requirements: Apple silicon, macOS 14+, full Xcode with the Metal toolchain
# (`xcodebuild -downloadComponent MetalToolchain`), and `brew install cmake llvm lz4 zstd`.
set -euo pipefail

VERSION=0.8.0
SRC_URL="https://github.com/Lulzx/cuda-metal/archive/refs/tags/v${VERSION}.tar.gz"
SRC_SHA=19e92bfe802e8fbb202a67aba0adc9e14d1ff5eb4e409233f8c64541e39a23c5
VF64_URL="https://github.com/Lulzx/VF64-metal/archive/feeb6aefd5b72c6c02224a412c512677689b530b.tar.gz"
VF64_SHA=cb0058d3b22386409f1440acc690bed81c575edc7696d59f0c05c01b6e996670

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
PREFIX="$HERE/toolchains/cumetal-$VERSION"

fail() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -m)" == "arm64" ]] || fail "CuMetal needs Apple silicon (arm64)."
major="$(sw_vers -productVersion | cut -d. -f1)"
(( major >= 14 )) || fail "macOS 14+ required (found $(sw_vers -productVersion))."
# Use an installed Xcode even when `xcode-select` still points at the Command Line Tools,
# without changing the system-wide selection (DEVELOPER_DIR is per process).
XCODE_DEV=/Applications/Xcode.app/Contents/Developer
if ! xcrun --find metal >/dev/null 2>&1 && [[ -z "${DEVELOPER_DIR:-}" && -d "$XCODE_DEV" ]]; then
  export DEVELOPER_DIR="$XCODE_DEV"
  if ! xcrun --find metal >/dev/null 2>&1; then
    echo "Downloading Xcode's Metal toolchain component..."
    # The first catalog fetch right after installing Xcode can fail transiently; retry once.
    xcodebuild -downloadComponent MetalToolchain || xcodebuild -downloadComponent MetalToolchain
  fi
fi
for tool in metal metallib; do
  xcrun --find "$tool" >/dev/null 2>&1 || fail "Apple '$tool' compiler missing. Install Xcode (not just the Command Line Tools), then run: xcodebuild -downloadComponent MetalToolchain"
done
command -v cmake >/dev/null || fail "cmake missing: brew install cmake"
LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null || true)"
[[ -d "$LLVM_PREFIX/lib/cmake/llvm" ]] || fail "Homebrew LLVM missing: brew install llvm lz4 zstd"

fetch() {  # url sha256 dest
  curl -sSfL "$1" -o "$3.part"
  echo "$2  $3.part" | shasum -a 256 -c - >/dev/null || { rm -f "$3.part"; fail "checksum mismatch for $1"; }
  mv "$3.part" "$3"
}

mkdir -p "$BUILD"
fetch "$SRC_URL" "$SRC_SHA" "$BUILD/cumetal.tar.gz"
fetch "$VF64_URL" "$VF64_SHA" "$BUILD/vf64.tar.gz"
rm -rf "$BUILD/src" && mkdir -p "$BUILD/src"
tar -xzf "$BUILD/cumetal.tar.gz" -C "$BUILD/src" --strip-components 1
mkdir -p "$BUILD/src/third_party/VF64-metal"
tar -xzf "$BUILD/vf64.tar.gz" -C "$BUILD/src/third_party/VF64-metal" --strip-components 1

cmake -S "$BUILD/src" -B "$BUILD/cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCUMETAL_BUILD_TESTS=OFF \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
cmake --build "$BUILD/cmake" --parallel
cmake --install "$BUILD/cmake"

"$PREFIX/bin/cumetal" version
"$PREFIX/bin/cumetal" doctor || true
echo
echo "Installed to $PREFIX. Activate with:  source sandbox/native/env.sh"
