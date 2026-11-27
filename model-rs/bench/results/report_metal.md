3 interleaved rounds; 1-min load average median 2.3 (range 1.4-5.8).

#### soma float32: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 10.61 | 10.52 | **5.00** | 2.12x (2.12-2.13) | 2.11x (2.10-2.11) | 4.79 |
| 2 | 11.02 | 10.94 | **5.70** | 1.93x (1.91-1.94) | 1.91x (1.90-1.93) | 5.49 |
| 4 | 11.92 | 11.89 | **8.08** | 1.48x (1.46-1.48) | 1.47x (1.47-1.48) | 7.86 |
| 8 | 15.95 | 15.88 | **12.95** | 1.23x (1.19-1.23) | 1.23x (1.18-1.24) | 12.73 |
| 16 | 25.30 | 25.06 | **22.56** | 1.12x (1.08-1.14) | 1.09x (1.08-1.16) | 22.30 |
| 32 | 46.02 | 44.58 | **42.88** | 1.07x (1.05-1.07) | 1.05x (1.04-1.07) | 42.57 |

Stream of single requests (req/s): Python sequential 95, Python pipelined 151, mlx-rs pipelined 151, **Metal runtime sequential 200**

#### unitree_g1 float32: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 10.50 | 10.40 | **4.88** | 2.15x (2.15-2.15) | 2.13x (2.12-2.14) | 4.67 |
| 2 | 10.88 | 10.78 | **5.59** | 1.95x (1.94-1.96) | 1.93x (1.93-1.94) | 5.37 |
| 4 | 11.84 | 11.75 | **7.96** | 1.49x (1.48-1.49) | 1.47x (1.47-1.48) | 7.73 |
| 8 | 15.82 | 15.64 | **13.08** | 1.21x (1.18-1.24) | 1.19x (1.17-1.23) | 12.86 |
| 16 | 25.38 | 25.24 | **23.31** | 1.09x (1.08-1.15) | 1.08x (1.08-1.14) | 23.08 |
| 32 | 46.03 | 45.93 | **43.67** | 1.05x (1.05-1.06) | 1.05x (1.05-1.05) | 43.45 |

Stream of single requests (req/s): Python sequential 95, Python pipelined 154, mlx-rs pipelined 155, **Metal runtime sequential 205**

#### soma bfloat16: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 8.29 | 8.16 | **3.76** | 2.25x (2.14-2.29) | 2.17x (2.12-2.23) | 3.56 |
| 2 | 8.98 | 8.83 | **4.42** | 2.05x (1.95-2.09) | 1.99x (1.94-2.03) | 4.23 |
| 4 | 10.12 | 9.92 | **6.76** | 1.48x (1.48-1.50) | 1.47x (1.42-1.48) | 6.58 |
| 8 | 14.09 | 13.96 | **11.17** | 1.28x (1.26-1.29) | 1.28x (1.24-1.28) | 10.97 |
| 16 | 22.19 | 22.16 | **19.06** | 1.18x (1.08-1.20) | 1.18x (1.08-1.18) | 18.81 |
| 32 | 38.80 | 37.82 | **37.35** | 1.04x (1.01-1.07) | 1.04x (1.01-1.07) | 37.08 |

Stream of single requests (req/s): Python sequential 121, Python pipelined 188, mlx-rs pipelined 185, **Metal runtime sequential 266**

#### unitree_g1 bfloat16: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 8.25 | 8.21 | **3.64** | 2.26x (2.25-2.45) | 2.31x (2.24-2.33) | 3.46 |
| 2 | 8.78 | 8.73 | **4.36** | 2.01x (2.00-2.05) | 2.01x (1.98-2.02) | 4.16 |
| 4 | 9.93 | 9.83 | **6.68** | 1.49x (1.47-1.50) | 1.47x (1.46-1.49) | 6.47 |
| 8 | 13.74 | 13.71 | **10.76** | 1.28x (1.26-1.32) | 1.27x (1.27-1.28) | 10.54 |
| 16 | 21.90 | 21.62 | **19.00** | 1.15x (1.11-1.18) | 1.14x (1.12-1.14) | 18.79 |
| 32 | 38.17 | 38.11 | **36.08** | 1.05x (0.98-1.06) | 1.05x (1.01-1.06) | 35.85 |

Stream of single requests (req/s): Python sequential 120, Python pipelined 189, mlx-rs pipelined 200, **Metal runtime sequential 276**

#### soma float16: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 8.14 | 8.17 | **3.78** | 2.15x (2.14-2.20) | 2.16x (2.15-2.20) | 3.58 |
| 2 | 8.81 | 8.85 | **4.50** | 1.97x (1.96-1.99) | 1.97x (1.96-2.01) | 4.29 |
| 4 | 9.93 | 9.91 | **6.78** | 1.47x (1.46-1.47) | 1.46x (1.45-1.49) | 6.57 |
| 8 | 13.86 | 13.89 | **10.95** | 1.26x (1.25-1.29) | 1.27x (1.25-1.29) | 10.71 |
| 16 | 21.77 | 21.69 | **18.96** | 1.15x (1.10-1.16) | 1.14x (1.11-1.16) | 18.70 |
| 32 | 37.86 | 37.88 | **36.61** | 1.03x (1.01-1.03) | 1.04x (1.03-1.04) | 36.34 |

Stream of single requests (req/s): Python sequential 123, Python pipelined 195, mlx-rs pipelined 197, **Metal runtime sequential 265**

#### unitree_g1 float16: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 8.22 | 8.33 | **3.57** | 2.31x (2.26-2.32) | 2.35x (2.25-2.37) | 3.38 |
| 2 | 8.73 | 8.78 | **4.39** | 2.00x (1.98-2.03) | 2.00x (1.98-2.05) | 4.20 |
| 4 | 9.88 | 9.90 | **6.71** | 1.47x (1.47-1.47) | 1.48x (1.47-1.48) | 6.49 |
| 8 | 13.72 | 13.93 | **10.85** | 1.26x (1.26-1.29) | 1.28x (1.26-1.30) | 10.63 |
| 16 | 21.64 | 21.90 | **19.64** | 1.11x (1.10-1.12) | 1.12x (1.10-1.13) | 19.38 |
| 32 | 38.44 | 38.81 | **37.83** | 1.02x (1.00-1.02) | 1.03x (1.02-1.03) | 37.61 |

Stream of single requests (req/s): Python sequential 122, Python pipelined 195, mlx-rs pipelined 191, **Metal runtime sequential 280**

#### soma int8: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 7.96 | — | **3.81** | 2.26x (2.08-2.86) | — | 3.61 |
| 2 | 8.47 | — | **5.03** | 1.72x (1.65-1.78) | — | 4.85 |
| 4 | 10.66 | — | **7.74** | 1.38x (1.37-1.44) | — | 7.53 |
| 8 | 15.27 | — | **12.70** | 1.21x (1.20-1.23) | — | 12.46 |
| 16 | 25.60 | — | **23.19** | 1.11x (1.10-1.13) | — | 22.95 |
| 32 | 46.75 | — | **43.48** | 1.08x (1.06-1.09) | — | 43.20 |

#### unitree_g1 int8: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 7.85 | — | **3.71** | 2.12x (2.07-2.16) | — | 3.51 |
| 2 | 8.32 | — | **4.88** | 1.70x (1.70-1.72) | — | 4.68 |
| 4 | 10.60 | — | **7.63** | 1.39x (1.39-1.40) | — | 7.41 |
| 8 | 15.23 | — | **12.55** | 1.22x (1.20-1.28) | — | 12.34 |
| 16 | 24.79 | — | **22.49** | 1.10x (1.10-1.15) | — | 22.28 |
| 32 | 45.60 | — | **42.57** | 1.08x (1.06-1.11) | — | 42.33 |

