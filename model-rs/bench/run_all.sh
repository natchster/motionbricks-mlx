#!/usr/bin/env bash
# Interleaved Python-vs-Rust benchmark rounds. The order alternates every round so slow drifts
# in machine load hit both runtimes equally; bench/report.py pairs runs within each round.
#
#   model-rs/bench/run_all.sh [rounds=5] [dtypes="float32 bfloat16"]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ROUNDS="${1:-5}"
DTYPES="${2:-float32 bfloat16}"
OUT="${OUT:-$HERE/results/raw}"
PY_FLAGS="${PY_FLAGS:-}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
(cd "$ROOT/model-rs" && cargo build --release --quiet)
[[ -f "$HERE/requests/soma_N1.npz" ]] || (cd "$ROOT" && uv run --all-packages python model-rs/bench/make_requests.py)

run_rust()   { "$ROOT/model-rs/target/release/mbrs-bench" --model "$1" --dtype "$2" --iters 30 --json "$3"; }
run_python() { (cd "$ROOT" && uv run --all-packages python model-rs/bench/bench_python.py --model "$1" --dtype "$2" --iters 30 --json "$3" $PY_FLAGS); }

for dtype in $DTYPES; do
  for round in $(seq 1 "$ROUNDS"); do
    for model in soma unitree_g1; do
      order="python rust"; (( round % 2 == 0 )) && order="rust python"
      for rt in $order; do
        load="$(sysctl -n vm.loadavg | awk '{print $2}')"
        echo "== $dtype round $round $model $rt (load $load)"
        "run_$rt" "$model" "$dtype" "$OUT/${dtype}_r${round}_${model}_${rt}.json" | grep -E "N=1 |N=32|stream"
        echo "$load" > "$OUT/${dtype}_r${round}_${model}_${rt}.load"
      done
    done
  done
done
echo "done: $OUT"
