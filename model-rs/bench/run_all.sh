#!/usr/bin/env bash
# Interleaved Python-vs-Rust benchmark rounds. The order alternates every round so slow drifts
# in machine load hit both runtimes equally; bench/report.py pairs runs within each round.
#
#   model-rs/bench/run_all.sh [rounds=5] [dtypes="float32 bfloat16"]
#   python3 model-rs/bench/report.py model-rs/bench/results/raw --table metal   # or --table mlx-rs
#
# The Python side needs only the model package (no torch/warp/onnxruntime). Its environment is
# synced once, before the first round; the rounds then run with `uv run --no-sync`, so no round
# re-resolves or installs anything right before it is timed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ROUNDS="${1:-5}"
DTYPES="${2:-float32 bfloat16}"
OUT="${OUT:-$HERE/results/raw}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
(cd "$ROOT/model-rs" && cargo build --release --quiet)
# --inexact keeps packages other workspace members installed (e.g. the sandbox's) in the shared .venv.
(cd "$ROOT" && uv sync --inexact --package motionbricks-mlx --quiet)
for model in soma unitree_g1; do
  for batch in 1 2 4 8 16 32; do
    [[ -f "$HERE/requests/${model}_N${batch}.npz" ]] && continue
    (cd "$ROOT" && uv run --no-sync python model-rs/bench/make_requests.py)
    break 2
  done
done

run_rust()   { "$ROOT/model-rs/target/release/mbrs-bench" --model "$1" --dtype "$2" --iters 30 --json "$3"; }
run_python() { (cd "$ROOT" && uv run --no-sync python model-rs/bench/bench_python.py --model "$1" --dtype "$2" --iters 30 --json "$3"); }

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
