# Source me: put the sandboxed CuMetal toolchain (cumetalc, cumetal) first on PATH for this
# shell only. Keeps fake nvcc/libcudart away from unrelated builds.
_cumetal_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/toolchains"
_cumetal_bin="$(ls -d "$_cumetal_root"/cumetal-*/bin 2>/dev/null | tail -1)"
if [ -n "$_cumetal_bin" ]; then
  # Point xcrun at Xcode for this shell if the system default is still the Command Line Tools.
  if ! xcrun --find metal >/dev/null 2>&1 && [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  fi
  export PATH="$_cumetal_bin:$PATH"
  export CUMETAL_HOME="$(dirname "$_cumetal_bin")"
  echo "cumetal toolchain: $_cumetal_bin"
else
  echo "cumetal not installed; run sandbox/native/install_cumetal.sh" >&2
fi
unset _cumetal_root _cumetal_bin
