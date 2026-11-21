"""Write the benchmark requests (motionbricks_mlx.examples, seed 0 like model/benchmarks/bench.py)
to .npz files, so the Python and Rust harnesses time byte-identical inputs.

    uv run --all-packages python model-rs/bench/make_requests.py
"""

from pathlib import Path

import numpy as np

from motionbricks_mlx import examples

OUT = Path(__file__).resolve().parent / "requests"


def main():
    OUT.mkdir(exist_ok=True)
    for name in ("soma", "unitree_g1"):
        for batch in (1, 2, 4, 8, 16, 32):
            np.savez(OUT / f"{name}_N{batch}.npz", **examples.request(name, batch=batch))
    print(f"wrote requests to {OUT}")


if __name__ == "__main__":
    main()
