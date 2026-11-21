4 interleaved rounds; 1-min load average median 1.9 (range 1.2-3.7).

#### soma FP32: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 10.70 | 10.62 | **7.14** | 1.50x (1.50-1.53) | 1.49x (1.48-1.52) | 6.77 |
| 2 | 11.08 | 11.01 | **9.23** | 1.20x (1.19-1.20) | 1.19x (1.19-1.19) | 8.85 |
| 4 | 12.01 | 11.95 | **10.79** | 1.11x (1.11-1.12) | 1.11x (1.10-1.12) | 10.40 |
| 8 | 16.00 | 15.93 | **15.06** | 1.06x (1.06-1.07) | 1.06x (1.06-1.06) | 14.65 |
| 16 | 25.09 | 25.00 | **25.06** | 1.00x (0.99-1.01) | 1.00x (1.00-1.01) | 24.62 |
| 32 | 45.48 | 45.57 | **46.66** | 0.97x (0.97-0.98) | 0.98x (0.97-0.98) | 46.18 |

Stream of single requests (req/s): Python sequential 94, Python pipelined 150, mlx-rs pipelined 150, **Metal runtime sequential 140**

#### unitree_g1 FP32: median ms per batched call, end to end (inputs in, outputs on the host)

| N | Python MLX (compiled) | Rust mlx-rs (compiled) | **Rust Metal runtime** | Metal speedup vs Python | vs mlx-rs | Metal GPU time |
|---|---|---|---|---|---|---|
| 1 | 10.55 | 10.50 | **7.09** | 1.49x (1.48-1.51) | 1.48x (1.48-1.49) | 6.72 |
| 2 | 10.90 | 10.86 | **9.10** | 1.20x (1.18-1.20) | 1.19x (1.19-1.19) | 8.73 |
| 4 | 11.87 | 11.82 | **10.72** | 1.11x (1.10-1.12) | 1.10x (1.10-1.11) | 10.33 |
| 8 | 15.83 | 15.74 | **14.93** | 1.06x (1.06-1.06) | 1.05x (1.05-1.06) | 14.53 |
| 16 | 24.80 | 24.84 | **24.88** | 1.00x (0.99-1.02) | 1.00x (0.99-1.02) | 24.48 |
| 32 | 44.85 | 45.00 | **46.01** | 0.98x (0.97-0.98) | 0.98x (0.97-0.98) | 45.60 |

Stream of single requests (req/s): Python sequential 95, Python pipelined 154, mlx-rs pipelined 154, **Metal runtime sequential 141**

