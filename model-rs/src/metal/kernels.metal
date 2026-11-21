// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//
// Model-specific Metal kernels for MotionBricks (static shapes: <= 32 tokens, <= 64 frames).
// `T` (float, half or bfloat) is the network element type, set by the host before compiling
// (`typedef ... T;`). Accumulation, masks, split-K partials and normalization math are float.
// Weights keep the PyTorch/MLX layout: Linear W[N][K], Conv1d W[N][k][Cin] (channels-last), so a
// conv is a GEMM over K' = k * Cin. INT8 weights use MLX's affine format (bits 8, group 64):
// packed uint32 [N][K/4] (first element in the low byte), float scales/biases [N][K/64].
#include <metal_stdlib>
using namespace metal;

typedef vec<T, 4> T4;

// ---------------------------------------------------------------------------------------------
// GEMM: Y[m, n] = epilogue( sum_k A(m, k) * W[n, k] + bias[n] )
// ---------------------------------------------------------------------------------------------
struct GemmParams {
  int M, N, K;
  int lda, ldy;          // row strides of A (segment 1) and Y
  int act;               // 0 none, 1 relu, 2 leaky relu 0.01
  int conv_taps;         // 0 = plain GEMM, else Conv1d kernel size (zero padding = dilation * (taps / 2))
  int conv_T;            // frames per sequence at the output resolution
  int conv_cin;          // input channels
  int conv_dil;          // dilation
  int has_mask;          // multiply A rows by amask[row] (conv: the source frame at output resolution)
  int has_res;           // add R[rrow, n] after the activation
  int ldr;
  int k_split;           // K elements per split (multiple of BK); K when not split
  int splits;
  int a_relu;            // relu(A) before the mask
  int conv_up;           // 2: nearest 2x upsampling of the source frames (source T = conv_T / 2)
  int k1, k2;            // concat: k < k1 from A, k1 <= k < k1 + k2 from A2, else A3 (0 = single source)
  int lda2, lda3;
  int y_mask;            // multiply the result by ymask[m]
  int y_pred;            // write only rows with ymask[m] != 0
  int blend;             // v = v * ymask[m] + missing[(m % blend_mod) * N + n] * (1 - ymask[m])
  int blend_mod;
  int r_mod;             // residual row = m % r_mod (0: m)
};

inline float activate(float v, int act) {
  return act == 1 ? max(v, 0.0f) : (act == 2 ? max(v * 0.01f, v) : v);
}

// Loader specialization (function constants; undefined = false).
constant bool fc_plain [[function_constant(0)]];   // M/N/K tile-aligned, conv taps / concat segments constant per K tile
constant bool fc_conv [[function_constant(1)]];    // implicit Conv1d im2col
constant bool fc_concat [[function_constant(2)]];  // A = [A | A2 | A3] along K
constant bool fc_amask [[function_constant(3)]];   // A rows multiplied by amask
constant bool fc_arelu [[function_constant(4)]];   // relu(A)
constant bool fc_up [[function_constant(5)]];      // conv over 2x nearest-upsampled frames
constant bool fc_q8 [[function_constant(6)]];      // W is INT8 (affine, group 64)
constant bool fc_vec [[function_constant(7)]];     // all K extents / strides multiples of 4
#define FC(x) (is_function_constant_defined(x) && x)
constant bool PLAIN = FC(fc_plain);
constant bool CONV = FC(fc_conv);
constant bool CONCAT = FC(fc_concat);
constant bool AMASK = FC(fc_amask);
constant bool ARELU = FC(fc_arelu);
constant bool UP = FC(fc_up);
constant bool Q8 = FC(fc_q8);
constant bool VEC = FC(fc_vec);

struct Epi {
  device const T* bias;
  device const T* R;
  device const float* ymask;
  device const T* missing;
};

// Shared epilogue (direct GEMM output and split-K reduce). Returns false if the row is skipped.
inline bool epilogue(thread const Epi& e, constant GemmParams& p, int m, int n, thread float& v) {
  if (p.y_pred && e.ymask[m] == 0.0f) return false;
  v = activate(v + float(e.bias[n]), p.act);
  if (p.has_res) v += float(e.R[(p.r_mod ? m % p.r_mod : m) * p.ldr + n]);
  if (p.y_mask) v *= e.ymask[m];
  if (p.blend) {
    float h = e.ymask[m];
    v = v * h + float(e.missing[(m % p.blend_mod) * p.N + n]) * (1.0f - h);
  }
  return true;
}

// Per-thread A-row context, computed once per kernel.
struct ARow {
  bool ok;        // row < M
  int row;        // m (clamped)
  int b, t;       // conv: sequence and frame
  float mask;     // non-conv row mask
};

inline ARow a_row(constant GemmParams& p, device const float* amask, int m) {
  ARow r;
  r.ok = m < p.M;
  r.row = min(m, p.M - 1);
  r.b = CONV ? r.row / p.conv_T : 0;
  r.t = CONV ? r.row - r.b * p.conv_T : 0;
  r.mask = (AMASK && !CONV) ? amask[r.row] : 1.0f;
  return r;
}

inline float4 a_post(float4 v, float mask) {
  if (ARELU) v = max(v, float4(0.0f));
  if (AMASK) v *= mask;
  return v;
}

// 4 consecutive k of A(m, k) (never straddling a tap or concat segment: boundaries are multiples of 4).
inline float4 load_a4(device const T* A, device const T* A2, device const T* A3, device const float* amask,
                      constant GemmParams& p, thread const ARow& r, int k, int kend) {
  if (!r.ok || k >= kend) return float4(0.0f);
  if (CONV) {
    int tap = k / p.conv_cin, c = k - tap * p.conv_cin;
    int ts = r.t + (tap - p.conv_taps / 2) * p.conv_dil;
    if (ts < 0 || ts >= p.conv_T) return float4(0.0f);
    int src_row = UP ? r.b * (p.conv_T / 2) + ts / 2 : r.b * p.conv_T + ts;
    float m = AMASK ? amask[r.b * p.conv_T + ts] : 1.0f;
    return a_post(float4(*(device const T4*)(A + src_row * p.lda + c)), m);
  }
  if (CONCAT) {
    if (k >= p.k1 + p.k2) return a_post(float4(*(device const T4*)(A3 + r.row * p.lda3 + (k - p.k1 - p.k2))), r.mask);
    if (k >= p.k1) return a_post(float4(*(device const T4*)(A2 + r.row * p.lda2 + (k - p.k1))), r.mask);
  }
  return a_post(float4(*(device const T4*)(A + r.row * p.lda + k)), r.mask);
}

inline float load_a1(device const T* A, device const T* A2, device const T* A3, device const float* amask,
                     constant GemmParams& p, thread const ARow& r, int k, int kend) {
  if (!r.ok || k >= kend) return 0.0f;
  float v, m = r.mask;
  if (CONV) {
    int tap = k / p.conv_cin, c = k - tap * p.conv_cin;
    int ts = r.t + (tap - p.conv_taps / 2) * p.conv_dil;
    if (ts < 0 || ts >= p.conv_T) return 0.0f;
    int src_row = UP ? r.b * (p.conv_T / 2) + ts / 2 : r.b * p.conv_T + ts;
    m = AMASK ? amask[r.b * p.conv_T + ts] : 1.0f;
    v = float(A[src_row * p.lda + c]);
  } else if (CONCAT && k >= p.k1 + p.k2) {
    v = float(A3[r.row * p.lda3 + (k - p.k1 - p.k2)]);
  } else if (CONCAT && k >= p.k1) {
    v = float(A2[r.row * p.lda2 + (k - p.k1)]);
  } else {
    v = float(A[r.row * p.lda + k]);
  }
  if (ARELU) v = max(v, 0.0f);
  if (AMASK) v *= m;
  return v;
}

// 4 consecutive k of W row n (T or dequantized INT8: w = scale * q + bias).
inline float4 load_w4(device const T* W, device const uint* Wq, device const float* qs, device const float* qb,
                      constant GemmParams& p, int n, int k) {
  if (Q8) {
    uint q = Wq[n * (p.K / 4) + k / 4];
    int g = n * (p.K / 64) + k / 64;
    float s = qs[g], bb = qb[g];
    return float4(float(q & 0xff), float((q >> 8) & 0xff), float((q >> 16) & 0xff), float(q >> 24)) * s + bb;
  }
  return float4(*(device const T4*)(W + n * p.K + k));
}

// GEMM on simdgroup matrices, structured like MLX's steel GEMM: each simdgroup owns TM x TN 8x8
// fragments interleaved across the threadgroup tile; fragments are loaded lane-wise from
// threadgroup memory (stored as T, accumulated as float); serpentine multiply-accumulate order.
// Per-thread load slots are fixed for the whole K loop. Split-K writes float partials to P.
template <int BM, int BN, int BK, int WM, int WN>
[[kernel]] void gemm_mma(
    device const T* A [[buffer(0)]], device const T* W [[buffer(1)]], device const T* bias [[buffer(2)]],
    device T* Y [[buffer(3)]], constant GemmParams& p [[buffer(4)]], device const float* amask [[buffer(5)]],
    device const T* R [[buffer(6)]], device const T* A2 [[buffer(7)]], device const T* A3 [[buffer(8)]],
    device const float* ymask [[buffer(9)]], device const T* missing [[buffer(10)]],
    device const uint* Wq [[buffer(11)]], device const float* qs [[buffer(12)]], device const float* qb [[buffer(13)]],
    device float* P [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const Epi epi{bias, R, ymask, missing};
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int LDS = BK + 16 / sizeof(T);
  constexpr int SA = (BM * BK / 4 + NT - 1) / NT, SB = (BN * BK / 4 + NT - 1) / NT;
  threadgroup T As[BM * LDS];
  threadgroup T Bs[BN * LDS];
  const int m0 = int(tgid.y) * BM, n0 = int(tgid.x) * BN;
  const int kbeg = int(tgid.z) * p.k_split, kend = min(p.K, kbeg + p.k_split);
  const short qid = short(lane) / 4;
  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
  const short tm = 8 * (short(sg) / WN), tn = 8 * (short(sg) % WN);
  simdgroup_float8x8 acc[TM][TN];
  _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; i++)
    _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // Load slots (vector paths): tile row / column per slot, fixed across the K loop.
  int a_r[SA], a_c[SA], b_r[SB], b_c[SB];
  ARow arow[SA];
  device const T* a_src[SA];
  device const T* b_src[SB];
  _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++) {
    int i = min(int(tid) + s * NT, BM * BK / 4 - 1);
    a_r[s] = i / (BK / 4);
    a_c[s] = (i % (BK / 4)) * 4;
    arow[s] = a_row(p, amask, m0 + a_r[s]);
    a_src[s] = A + arow[s].row * p.lda + a_c[s];
  }
  _Pragma("clang loop unroll(full)") for (int s = 0; s < SB; s++) {
    int i = min(int(tid) + s * NT, BN * BK / 4 - 1);
    b_r[s] = i / (BK / 4);
    b_c[s] = (i % (BK / 4)) * 4;
    b_src[s] = W + min(n0 + b_r[s], p.N - 1) * p.K + b_c[s];
  }
  // INT8 plain path: packed row pointers and the row's group scales/biases (BK divides the
  // quantization group of 64, so one tile column block lies in a single group).
  device const uint* bq_src[SB];
  device const float* bs_src[SB];
  device const float* bb_src[SB];
  if (PLAIN && Q8) {
    _Pragma("clang loop unroll(full)") for (int s = 0; s < SB; s++) {
      const int row = min(n0 + b_r[s], p.N - 1);
      bq_src[s] = Wq + row * (p.K / 4) + b_c[s] / 4;
      bs_src[s] = qs + row * (p.K / 64);
      bb_src[s] = qb + row * (p.K / 64);
    }
  }
  const threadgroup T* a_base = As + (tm + fm) * LDS + fn;
  const threadgroup T* b_base = Bs + (tn + fn) * LDS + fm;

  for (int k0 = kbeg; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (PLAIN) {
      if (CONV) {
        // Tile-aligned implicit conv: the tap (and so the frame offset) is constant per K tile.
        const int tap = k0 / p.conv_cin, c0 = k0 - tap * p.conv_cin;
        const int off = (tap - p.conv_taps / 2) * p.conv_dil;
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++) {
          const int ts = arow[s].t + off;
          T4 v = T4(0);
          if (ts >= 0 && ts < p.conv_T) {
            const int src = UP ? arow[s].b * (p.conv_T / 2) + ts / 2 : arow[s].b * p.conv_T + ts;
            float4 x = float4(*(device const T4*)(A + src * p.lda + c0 + a_c[s]));
            if (ARELU) x = max(x, float4(0.0f));
            if (AMASK) x *= amask[arow[s].b * p.conv_T + ts];
            v = T4(x);
          }
          *(threadgroup T4*)(As + a_r[s] * LDS + a_c[s]) = v;
        }
      } else if (CONCAT) {
        // Segment boundaries are tile-aligned: one source per K tile.
        device const T* src = A;
        int ld = p.lda, kk = k0;
        if (k0 >= p.k1 + p.k2) { src = A3; ld = p.lda3; kk = k0 - p.k1 - p.k2; }
        else if (k0 >= p.k1) { src = A2; ld = p.lda2; kk = k0 - p.k1; }
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++) {
          const float4 x = a_post(float4(*(device const T4*)(src + arow[s].row * ld + kk + a_c[s])), arow[s].mask);
          *(threadgroup T4*)(As + a_r[s] * LDS + a_c[s]) = T4(x);
        }
      } else if (ARELU || AMASK) {
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++)
          *(threadgroup T4*)(As + a_r[s] * LDS + a_c[s]) = T4(a_post(float4(*(device const T4*)(a_src[s] + k0)), arow[s].mask));
      } else {
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++)
          *(threadgroup T4*)(As + a_r[s] * LDS + a_c[s]) = *(device const T4*)(a_src[s] + k0);
      }
      if (Q8) {
        const int g = k0 / 64;
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SB; s++) {
          const uint q = bq_src[s][k0 / 4];
          const float sc = bs_src[s][g], bi = bb_src[s][g];
          const float4 w = float4(float(q & 0xff), float((q >> 8) & 0xff), float((q >> 16) & 0xff), float(q >> 24)) * sc + bi;
          *(threadgroup T4*)(Bs + b_r[s] * LDS + b_c[s]) = T4(w);
        }
      } else {
        _Pragma("clang loop unroll(full)") for (int s = 0; s < SB; s++)
          *(threadgroup T4*)(Bs + b_r[s] * LDS + b_c[s]) = *(device const T4*)(b_src[s] + k0);
      }
    } else if (VEC) {
      _Pragma("clang loop unroll(full)") for (int s = 0; s < SA; s++)
        *(threadgroup T4*)(As + a_r[s] * LDS + a_c[s]) = T4(load_a4(A, A2, A3, amask, p, arow[s], k0 + a_c[s], kend));
      _Pragma("clang loop unroll(full)") for (int s = 0; s < SB; s++) {
        int gn = n0 + b_r[s], gk = k0 + b_c[s];
        float4 w = (gn < p.N && gk < kend) ? load_w4(W, Wq, qs, qb, p, gn, gk) : float4(0.0f);
        *(threadgroup T4*)(Bs + b_r[s] * LDS + b_c[s]) = T4(w);
      }
    } else {
      for (int i = int(tid); i < BM * BK; i += NT) {
        int r = i / BK, c = i % BK;
        ARow ar = a_row(p, amask, m0 + r);
        As[r * LDS + c] = T(load_a1(A, A2, A3, amask, p, ar, k0 + c, kend));
      }
      for (int i = int(tid); i < BN * BK; i += NT) {
        int r = i / BK, c = i % BK, gn = n0 + r, gk = k0 + c;
        Bs[r * LDS + c] = (gn < p.N && gk < kend) ? W[gn * p.K + gk] : T(0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[TM], b[TN];
      simdgroup_barrier(mem_flags::mem_none);
      _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; i++) {
        const threadgroup T* ap = a_base + (i * 8 * WM) * LDS + kk;
        reinterpret_cast<thread float2&>(a[i].thread_elements()) = float2(float(ap[0]), float(ap[1]));
      }
      simdgroup_barrier(mem_flags::mem_none);
      _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; j++) {
        const threadgroup T* bp = b_base + (j * 8 * WN) * LDS + kk;
        reinterpret_cast<thread float2&>(b[j].thread_elements()) = float2(float(bp[0]), float(bp[LDS]));
      }
      simdgroup_barrier(mem_flags::mem_none);
      _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; i++) {
        _Pragma("clang loop unroll(full)") for (int jj = 0; jj < TN; jj++) {
          const int j = (i % 2) ? (TN - 1 - jj) : jj;  // serpentine order
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
  }
  const bool split = p.splits > 1;
  _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; i++) {
    _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; j++) {
      auto e = acc[i][j].thread_elements();
      const int gm = m0 + tm + i * 8 * WM + fm;
      const int gn = n0 + tn + j * 8 * WN + fn;
      if (gm >= p.M) continue;
      _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; c++) {
        const int n = gn + c;
        if (n >= p.N) continue;
        if (split) {
          P[(int(tgid.z) * p.M + gm) * p.N + n] = e[c];
        } else {
          float v = e[c];
          if (epilogue(epi, p, gm, n, v)) Y[gm * p.ldy + n] = T(v);
        }
      }
    }
  }
}

#define MMA(BM, BN, BK, WM, WN) \
  template [[host_name("gemm_mma_" #BM "_" #BN "_" #BK "_" #WM "_" #WN)]] [[kernel]] void gemm_mma<BM, BN, BK, WM, WN>( \
      device const T*, device const T*, device const T*, device T*, constant GemmParams&, device const float*, \
      device const T*, device const T*, device const T*, device const float*, device const T*, \
      device const uint*, device const float*, device const float*, device float*, uint3, uint, uint, uint);
MMA(16, 32, 32, 1, 2)
MMA(16, 64, 32, 1, 4)
MMA(16, 64, 64, 1, 4)
MMA(32, 32, 16, 2, 2)
MMA(32, 64, 16, 2, 2)
MMA(32, 64, 32, 2, 2)
MMA(64, 32, 16, 2, 2)
MMA(64, 64, 16, 2, 2)
MMA(64, 32, 32, 2, 2)
MMA(64, 64, 16, 2, 4)
MMA(64, 64, 32, 2, 4)

// Sum split-K partials [splits][M][N], then the shared epilogue.
[[kernel]] void gemm_splitk_reduce(
    device const float* P [[buffer(0)]], device const T* bias [[buffer(1)]],
    device T* Y [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
    device const T* R [[buffer(4)]], device const float* ymask [[buffer(5)]],
    device const T* missing [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const int total = p.M * p.N;
  if (int(gid) >= total) return;
  const int m = int(gid) / p.N, n = int(gid) % p.N;
  float s = 0.0f;
  for (int z = 0; z < p.splits; z++) s += P[z * total + int(gid)];
  const Epi epi{bias, R, ymask, missing};
  if (epilogue(epi, p, m, n, s)) Y[m * p.ldy + n] = T(s);
}

// ---------------------------------------------------------------------------------------------
// Multi-head self-attention for short sequences (L <= 32) straight from the packed in_proj
// output qkv (B, L, 3D); key padding mask pad (B, L), 1 = padded. One threadgroup per (b, head),
// Q/K/V of the head staged in threadgroup memory; one simdgroup per query, one lane per key.
// ---------------------------------------------------------------------------------------------
struct AttnParams { int B, L, D, H, hd, has_pad; float scale; };

[[kernel]] void attention(
    device const T* qkv [[buffer(0)]], device const float* pad [[buffer(1)]],
    device T* out [[buffer(2)]], constant AttnParams& p [[buffer(3)]],
    threadgroup float* shm [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]]) {
  const int LD = p.hd + 1;
  threadgroup float* Qs = shm;
  threadgroup float* Ks = shm + p.L * LD;
  threadgroup float* Vs = shm + 2 * p.L * LD;
  const int b = int(gid) / p.H, h = int(gid) % p.H;
  const int row3 = 3 * p.D;
  device const T* base = qkv + b * p.L * row3 + h * p.hd;
  for (int i = int(tid); i < p.L * p.hd; i += int(nt)) {
    int j = i / p.hd, c = i % p.hd;
    device const T* r = base + j * row3 + c;
    Qs[j * LD + c] = float(r[0]);
    Ks[j * LD + c] = float(r[p.D]);
    Vs[j * LD + c] = float(r[2 * p.D]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int q = int(sg), j = int(lane);
  if (q >= p.L) return;
  float s = -INFINITY;
  if (j < p.L && !(p.has_pad && pad[b * p.L + j] != 0.0f)) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int i = 0; i < p.hd; i += 4) {
      a0 = fma(Qs[q * LD + i], Ks[j * LD + i], a0);
      a1 = fma(Qs[q * LD + i + 1], Ks[j * LD + i + 1], a1);
      a2 = fma(Qs[q * LD + i + 2], Ks[j * LD + i + 2], a2);
      a3 = fma(Qs[q * LD + i + 3], Ks[j * LD + i + 3], a3);
    }
    s = ((a0 + a1) + (a2 + a3)) * p.scale;
  }
  const float mx = simd_max(s);
  const float e = (s == -INFINITY) ? 0.0f : exp(s - mx);
  const float pj = e / simd_sum(e);
  for (int i = int(lane); i < p.hd; i += 32) {
    float o0 = 0.0f, o1 = 0.0f;
    int jj = 0;
    for (; jj + 1 < p.L; jj += 2) {
      o0 = fma(simd_shuffle(pj, ushort(jj)), Vs[jj * LD + i], o0);
      o1 = fma(simd_shuffle(pj, ushort(jj + 1)), Vs[(jj + 1) * LD + i], o1);
    }
    if (jj < p.L) o0 = fma(simd_shuffle(pj, ushort(jj)), Vs[jj * LD + i], o0);
    out[(b * p.L + q) * p.D + h * p.hd + i] = T(o0 + o1);
  }
}

// Attention on simdgroup matrices for L <= LP (LP = 16 or 32) and head dim HD: one threadgroup
// per (b, head), one simdgroup per 8 queries. Q/K/V of the head are staged in threadgroup memory
// (rows >= L zeroed); S = Q K^T in float fragments, masked softmax per row (4 lanes per row),
// O = P V written straight to the output rows.
template <int LP, int HD>
[[kernel]] void attention_mma(
    device const T* qkv [[buffer(0)]], device const float* pad [[buffer(1)]],
    device T* out [[buffer(2)]], constant AttnParams& p [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr int LD = HD + 16 / sizeof(T), LS = LP + 4, NT = LP * 4;
  threadgroup T Qs[LP * LD], Ks[LP * LD], Vs[LP * LD];
  threadgroup float Ss[LP * LS];
  const int b = int(gid) / p.H, h = int(gid) % p.H;
  const int row3 = 3 * p.D;
  device const T* base = qkv + b * p.L * row3 + h * HD;
  _Pragma("clang loop unroll(full)") for (int i = int(tid); i < LP * HD / 4; i += NT) {
    const int j = i / (HD / 4), c = (i % (HD / 4)) * 4;
    T4 q = T4(0), k = T4(0), v = T4(0);
    if (j < p.L) {
      device const T* r = base + j * row3 + c;
      q = *(device const T4*)r;
      k = *(device const T4*)(r + p.D);
      v = *(device const T4*)(r + 2 * p.D);
    }
    *(threadgroup T4*)(Qs + j * LD + c) = q;
    *(threadgroup T4*)(Ks + j * LD + c) = k;
    *(threadgroup T4*)(Vs + j * LD + c) = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short qid = short(lane) / 4;
  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
  const int r0 = int(sg) * 8;

  simdgroup_float8x8 S[LP / 8];
  _Pragma("clang loop unroll(full)") for (int j = 0; j < LP / 8; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  _Pragma("clang loop unroll(full)") for (int kk = 0; kk < HD; kk += 8) {
    simdgroup_float8x8 a;
    const threadgroup T* ap = Qs + (r0 + fm) * LD + kk + fn;
    reinterpret_cast<thread float2&>(a.thread_elements()) = float2(float(ap[0]), float(ap[1]));
    _Pragma("clang loop unroll(full)") for (int j = 0; j < LP / 8; j++) {
      simdgroup_float8x8 bm;  // K^T: element (k, key) = K[key][k]
      const threadgroup T* bp = Ks + (j * 8 + fn) * LD + kk + fm;
      reinterpret_cast<thread float2&>(bm.thread_elements()) = float2(float(bp[0]), float(bp[LD]));
      simdgroup_multiply_accumulate(S[j], a, bm, S[j]);
    }
  }
  _Pragma("clang loop unroll(full)") for (int j = 0; j < LP / 8; j++) {
    auto e = S[j].thread_elements();
    _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; c++) {
      const int key = j * 8 + fn + c;
      const bool ok = key < p.L && !(p.has_pad && pad[b * p.L + key] != 0.0f);
      Ss[(r0 + fm) * LS + key] = ok ? e[c] * p.scale : -INFINITY;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  {
    threadgroup float* sr = Ss + (r0 + int(lane) / 4) * LS;
    float mx = -INFINITY;
    _Pragma("clang loop unroll(full)") for (int c = int(lane) % 4; c < LP; c += 4) mx = max(mx, sr[c]);
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    float sum = 0.0f;
    _Pragma("clang loop unroll(full)") for (int c = int(lane) % 4; c < LP; c += 4) {
      const float e = sr[c] == -INFINITY ? 0.0f : exp(sr[c] - mx);
      sr[c] = e;
      sum += e;
    }
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    _Pragma("clang loop unroll(full)") for (int c = int(lane) % 4; c < LP; c += 4) sr[c] = sr[c] / sum;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 O[HD / 8];
  _Pragma("clang loop unroll(full)") for (int c = 0; c < HD / 8; c++) O[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  _Pragma("clang loop unroll(full)") for (int kk = 0; kk < LP; kk += 8) {
    simdgroup_float8x8 a;
    reinterpret_cast<thread float2&>(a.thread_elements()) = *(const threadgroup float2*)(Ss + (r0 + fm) * LS + kk + fn);
    _Pragma("clang loop unroll(full)") for (int c = 0; c < HD / 8; c++) {
      simdgroup_float8x8 bm;  // V: element (key, dim)
      const threadgroup T* bp = Vs + (kk + fm) * LD + c * 8 + fn;
      reinterpret_cast<thread float2&>(bm.thread_elements()) = float2(float(bp[0]), float(bp[1]));
      simdgroup_multiply_accumulate(O[c], a, bm, O[c]);
    }
  }
  const int q = r0 + fm;
  if (q < p.L) {
    device T* o = out + (b * p.L + q) * p.D + h * HD + fn;
    _Pragma("clang loop unroll(full)") for (int c = 0; c < HD / 8; c++) {
      auto e = O[c].thread_elements();
      o[c * 8] = T(e[0]);
      o[c * 8 + 1] = T(e[1]);
    }
  }
}

#define ATTN(LP, HD) \
  template [[host_name("attention_mma_" #LP "_" #HD)]] [[kernel]] void attention_mma<LP, HD>( \
      device const T*, device const float*, device T*, constant AttnParams&, uint, uint, uint, uint);
ATTN(16, 32)
ATTN(16, 64)
ATTN(32, 32)
ATTN(32, 64)

// ---------------------------------------------------------------------------------------------
// y = LayerNorm(x + r) * w + b over the last axis (float math); one threadgroup per row.
// In half precision x + r is rounded to T first, as in MLX where the add is a T op.
// ---------------------------------------------------------------------------------------------
struct LnParams { int D; float eps; };

inline float tg_sum(float v, threadgroup float* red, uint sg, uint lane, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? red[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

[[kernel]] void add_layernorm(
    device const T* x [[buffer(0)]], device const T* r [[buffer(1)]],
    device const T* w [[buffer(2)]], device const T* bvec [[buffer(3)]],
    device T* y [[buffer(4)]], constant LnParams& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float red[32];
  device const T* xr = x + row * p.D;
  device const T* rr = r + row * p.D;
  float sum = 0.0f;
  for (int i = int(tid); i < p.D; i += int(nt)) sum += float(T(xr[i] + rr[i]));
  const float mean = tg_sum(sum, red, sg, lane, nsg) / p.D;
  float var = 0.0f;
  for (int i = int(tid); i < p.D; i += int(nt)) { float d = float(T(xr[i] + rr[i])) - mean; var += d * d; }
  const float inv = rsqrt(tg_sum(var, red, sg, lane, nsg) / p.D + p.eps);
  for (int i = int(tid); i < p.D; i += int(nt))
    y[row * p.D + i] = T((float(T(xr[i] + rr[i])) - mean) * inv * float(w[i]) + float(bvec[i]));
}
