// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
// Model-specific Metal kernels for MotionBricks (static shapes: <= 32 tokens, <= 64 frames).
// FP32 storage and accumulation. Weights keep the PyTorch/MLX layout: Linear W[N][K],
// Conv1d W[N][k][Cin] (channels-last), so a conv is a GEMM over K' = k * Cin.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------------------
// GEMM: Y[m, n] = act( sum_k A(m, k) * W[n, k] + bias[n] )  (+ optional residual R[m, n])
//
// A is either a plain row-major matrix (conv_taps == 0) or the implicit im2col of a channels-last
// sequence for a k-tap dilated Conv1d (conv_taps > 0): A(m, tap * Cin + c) = X[b, t + (tap - c0) * d, c]
// with zero padding and an optional per-frame input mask (multiplied in, like `x * mask`).
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

struct ASources {
  device const float* A;
  device const float* A2;
  device const float* A3;
  device const float* mask;
};

// Source address of A(m, k); returns -1 for implicit zero padding. `mrow` gets the mask index.
inline long a_index(constant GemmParams& p, int m, int k, thread int& mrow, thread int& seg) {
  if (p.conv_taps == 0) {
    mrow = m;
    if (p.k1 == 0 || k < p.k1) { seg = 0; return (long)m * p.lda + k; }
    if (k < p.k1 + p.k2) { seg = 1; return (long)m * p.lda2 + (k - p.k1); }
    seg = 2;
    return (long)m * p.lda3 + (k - p.k1 - p.k2);
  }
  seg = 0;
  int tap = k / p.conv_cin, c = k - tap * p.conv_cin;
  int b = m / p.conv_T, t = m - b * p.conv_T;
  int ts = t + (tap - p.conv_taps / 2) * p.conv_dil;
  if (ts < 0 || ts >= p.conv_T) return -1;
  mrow = b * p.conv_T + ts;
  long src_row = p.conv_up == 2 ? (long)b * (p.conv_T / 2) + ts / 2 : (long)mrow;
  return src_row * p.lda + c;
}

inline float a_finish(float v, constant GemmParams& p, int mrow, device const float* mask) {
  if (p.a_relu) v = max(v, 0.0f);
  return p.has_mask ? v * mask[mrow] : v;
}

inline float4 a_finish4(float4 v, constant GemmParams& p, int mrow, device const float* mask) {
  if (p.a_relu) v = max(v, float4(0.0f));
  return p.has_mask ? v * mask[mrow] : v;
}

inline float load_a(thread const ASources& s, constant GemmParams& p, int m, int k) {
  int mrow, seg;
  long idx = a_index(p, m, k, mrow, seg);
  if (idx < 0) return 0.0f;
  device const float* base = seg == 0 ? s.A : (seg == 1 ? s.A2 : s.A3);
  return a_finish(base[idx], p, mrow, s.mask);
}

// 4 consecutive k (never straddling a segment or tap: all boundaries are multiples of 4).
inline float4 load_a4(thread const ASources& s, constant GemmParams& p, int m, int k) {
  if (m >= p.M || k >= p.K) return float4(0.0f);
  int mrow, seg;
  long idx = a_index(p, m, k, mrow, seg);
  if (idx < 0) return float4(0.0f);
  device const float* base = seg == 0 ? s.A : (seg == 1 ? s.A2 : s.A3);
  return a_finish4(*(device const float4*)(base + idx), p, mrow, s.mask);
}

struct Epi {
  device const float* bias;
  device const float* R;
  device const float* ymask;
  device const float* missing;
};

// Shared epilogue (direct GEMM output and split-K reduce). Returns false if the row is skipped.
inline bool epilogue(thread const Epi& e, constant GemmParams& p, int m, int n, thread float& v) {
  if (p.y_pred && e.ymask[m] == 0.0f) return false;
  v = activate(v + e.bias[n], p.act);
  if (p.has_res) v += e.R[(long)(p.r_mod ? m % p.r_mod : m) * p.ldr + n];
  if (p.y_mask) v *= e.ymask[m];
  if (p.blend) {
    float h = e.ymask[m];
    v = v * h + e.missing[(long)(m % p.blend_mod) * p.N + n] * (1.0f - h);
  }
  return true;
}

inline bool vec_ok(constant GemmParams& p) {
  return (p.K & 3) == 0 && (p.lda & 3) == 0 && (p.conv_taps == 0 || (p.conv_cin & 3) == 0) &&
         (p.k1 == 0 || ((p.k1 & 3) == 0 && (p.k2 & 3) == 0 && (p.lda2 & 3) == 0 && (p.lda3 & 3) == 0));
}

// GEMM on simdgroup matrices. Threadgroup tile BM x BN, WM x WN simdgroups, each owning a
// (BM/WM) x (BN/WN) block of 8x8 accumulators. A and W tiles are staged in threadgroup memory
// with float4 loads (W rows are contiguous along K, i.e. the "NT" layout); the epilogue writes
// each lane's accumulator elements straight to memory (MLX's fragment coordinate mapping).
// Split-K: grid.z partitions K; with splits > 1 the raw partial sums go to Y as [z][M][N] and
// gemm_splitk_reduce applies bias / activation / residual.
template <int BM, int BN, int BK, int WM, int WN>
[[kernel]] void gemm_mma(
    device const float* A [[buffer(0)]], device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* Y [[buffer(3)]],
    constant GemmParams& p [[buffer(4)]], device const float* mask [[buffer(5)]],
    device const float* R [[buffer(6)]], device const float* A2 [[buffer(7)]],
    device const float* A3 [[buffer(8)]], device const float* ymask [[buffer(9)]],
    device const float* missing [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const ASources src{A, A2, A3, mask};
  const Epi epi{bias, R, ymask, missing};
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN, FM = TM / 8, FN = TN / 8;
  constexpr int LDS = BK + 4;
  threadgroup float As[BM * LDS];
  threadgroup float Bs[BN * LDS];
  const int m0 = int(tgid.y) * BM, n0 = int(tgid.x) * BN;
  const int kbeg = int(tgid.z) * p.k_split, kend = min(p.K, kbeg + p.k_split);
  const int sm = (int(sg) / WN) * TM, sn = (int(sg) % WN) * TN;
  simdgroup_float8x8 acc[FM][FN];
  _Pragma("clang loop unroll(full)") for (int i = 0; i < FM; i++)
    _Pragma("clang loop unroll(full)") for (int j = 0; j < FN; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const bool vec = vec_ok(p);
  for (int k0 = kbeg; k0 < kend; k0 += BK) {
    if (vec) {
      for (int i = int(tid); i < BM * BK / 4; i += NT) {
        int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
        float4 v = (k0 + c < kend) ? load_a4(src, p, m0 + r, k0 + c) : float4(0.0f);
        *(threadgroup float4*)(As + r * LDS + c) = v;
      }
      for (int i = int(tid); i < BN * BK / 4; i += NT) {
        int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
        int gn = n0 + r, gk = k0 + c;
        float4 v = (gn < p.N && gk < kend) ? *(device const float4*)(W + (long)gn * p.K + gk) : float4(0.0f);
        *(threadgroup float4*)(Bs + r * LDS + c) = v;
      }
    } else {
      for (int i = int(tid); i < BM * BK; i += NT) {
        int r = i / BK, c = i % BK, gm = m0 + r, gk = k0 + c;
        As[r * LDS + c] = (gm < p.M && gk < kend) ? load_a(src, p, gm, gk) : 0.0f;
      }
      for (int i = int(tid); i < BN * BK; i += NT) {
        int r = i / BK, c = i % BK, gn = n0 + r, gk = k0 + c;
        Bs[r * LDS + c] = (gn < p.N && gk < kend) ? W[(long)gn * p.K + gk] : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[FM], b[FN];
      _Pragma("clang loop unroll(full)") for (int i = 0; i < FM; i++) simdgroup_load(a[i], As + (sm + 8 * i) * LDS + kk, LDS);
      _Pragma("clang loop unroll(full)") for (int j = 0; j < FN; j++) simdgroup_load(b[j], Bs + (sn + 8 * j) * LDS + kk, LDS, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)") for (int i = 0; i < FM; i++)
        _Pragma("clang loop unroll(full)") for (int j = 0; j < FN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const short qid = short(lane) / 4;
  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
  const bool split = p.splits > 1;
  _Pragma("clang loop unroll(full)") for (int i = 0; i < FM; i++) {
    _Pragma("clang loop unroll(full)") for (int j = 0; j < FN; j++) {
      auto e = acc[i][j].thread_elements();
      const int gm = m0 + sm + 8 * i + fm;
      const int gn = n0 + sn + 8 * j + fn;
      if (gm >= p.M) continue;
      _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; c++) {
        const int n = gn + c;
        if (n >= p.N) continue;
        if (split) {
          Y[((long)tgid.z * p.M + gm) * p.N + n] = e[c];
        } else {
          float v = e[c];
          if (epilogue(epi, p, gm, n, v)) Y[(long)gm * p.ldy + n] = v;
        }
      }
    }
  }
}

#define MMA(BM, BN, BK, WM, WN) \
  template [[host_name("gemm_mma_" #BM "_" #BN "_" #BK "_" #WM "_" #WN)]] [[kernel]] void gemm_mma<BM, BN, BK, WM, WN>( \
      device const float*, device const float*, device const float*, device float*, \
      constant GemmParams&, device const float*, device const float*, device const float*, \
      device const float*, device const float*, device const float*, uint3, uint, uint, uint);
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
MMA(128, 32, 16, 4, 1)
MMA(64, 64, 16, 4, 2)

// Epilogue on a finished steel GEMM output YS (M x N, bias already added: pass zeros as bias),
// written to Y with the full set of options (activation, residual, output mask, predicate, blend).
[[kernel]] void epilogue_apply(
    device const float* YS [[buffer(0)]], device float* Y [[buffer(1)]], constant GemmParams& p [[buffer(2)]],
    device const float* zeros [[buffer(3)]], device const float* R [[buffer(4)]],
    device const float* ymask [[buffer(5)]], device const float* missing [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.M * p.N) return;
  const int m = int(gid) / p.N, n = int(gid) % p.N;
  const Epi epi{zeros, R, ymask, missing};
  float v = YS[gid];
  if (epilogue(epi, p, m, n, v)) Y[(long)m * p.ldy + n] = v;
}

// Materialize the effective A operand (conv im2col with dilation/upsampling, concatenation,
// relu, row mask) as a plain M x K matrix, so large GEMMs can use the steel kernel.
[[kernel]] void materialize_a(
    device const float* A [[buffer(0)]], device const float* A2 [[buffer(1)]], device const float* A3 [[buffer(2)]],
    device const float* mask [[buffer(3)]], device float* out [[buffer(4)]], constant GemmParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const ASources src{A, A2, A3, mask};
  if (vec_ok(p)) {
    const int kq = p.K / 4;
    if (int(gid) >= p.M * kq) return;
    const int m = int(gid) / kq, k = (int(gid) % kq) * 4;
    *(device float4*)(out + (long)m * p.K + k) = load_a4(src, p, m, k);
  } else {
    if (int(gid) >= p.M * p.K) return;
    const int m = int(gid) / p.K, k = int(gid) % p.K;
    out[gid] = load_a(src, p, m, k);
  }
}

// Sum split-K partials [splits][M][N], then the shared epilogue.
[[kernel]] void gemm_splitk_reduce(
    device const float* P [[buffer(0)]], device const float* bias [[buffer(1)]],
    device float* Y [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
    device const float* R [[buffer(4)]], device const float* ymask [[buffer(5)]],
    device const float* missing [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const int total = p.M * p.N;
  if (int(gid) >= total) return;
  const int m = int(gid) / p.N, n = int(gid) % p.N;
  float s = 0.0f;
  for (int z = 0; z < p.splits; z++) s += P[(long)z * total + gid];
  const Epi epi{bias, R, ymask, missing};
  if (epilogue(epi, p, m, n, s)) Y[(long)m * p.ldy + n] = s;
}

// ---------------------------------------------------------------------------------------------
// Multi-head self-attention for short sequences (L <= 32) straight from the packed in_proj
// output qkv (B, L, 3D); key padding mask pad (B, L), 1 = padded. One threadgroup per (b, head),
// one simdgroup per query, one lane per key. Output (B, L, D) in the layout out_proj expects.
// ---------------------------------------------------------------------------------------------
struct AttnParams { int B, L, D, H, hd, has_pad; float scale; };

[[kernel]] void attention(
    device const float* qkv [[buffer(0)]], device const float* pad [[buffer(1)]],
    device float* out [[buffer(2)]], constant AttnParams& p [[buffer(3)]],
    threadgroup float* shm [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]]) {
  // Q, K, V of one (batch, head) staged once in threadgroup memory (L x (hd + 1) each, sized by
  // the host); then one simdgroup per query and one lane per key.
  const int LD = p.hd + 1;
  threadgroup float* Qs = shm;
  threadgroup float* Ks = shm + p.L * LD;
  threadgroup float* Vs = shm + 2 * p.L * LD;
  const int b = int(gid) / p.H, h = int(gid) % p.H;
  const long row3 = 3L * p.D;
  device const float* base = qkv + (long)b * p.L * row3 + h * p.hd;
  for (int i = int(tid); i < p.L * p.hd; i += int(nt)) {
    int j = i / p.hd, c = i % p.hd;
    device const float* r = base + j * row3 + c;
    Qs[j * LD + c] = r[0];
    Ks[j * LD + c] = r[p.D];
    Vs[j * LD + c] = r[2 * p.D];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int q = int(sg), j = int(lane);
  if (q >= p.L) return;
  float s = -INFINITY;
  if (j < p.L && !(p.has_pad && pad[b * p.L + j] != 0.0f)) {
    // 4 independent accumulators: the dot product is latency-bound, not throughput-bound.
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
    out[((long)b * p.L + q) * p.D + h * p.hd + i] = o0 + o1;
  }
}

// ---------------------------------------------------------------------------------------------
// y = LayerNorm(x + r) * w + b over the last axis; one threadgroup per row.
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
    device const float* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device const float* w [[buffer(2)]], device const float* bvec [[buffer(3)]],
    device float* y [[buffer(4)]], constant LnParams& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float red[32];
  device const float* xr = x + (long)row * p.D;
  device const float* rr = r + (long)row * p.D;
  float sum = 0.0f;
  for (int i = int(tid); i < p.D; i += int(nt)) sum += xr[i] + rr[i];
  const float mean = tg_sum(sum, red, sg, lane, nsg) / p.D;
  float var = 0.0f;
  for (int i = int(tid); i < p.D; i += int(nt)) { float d = xr[i] + rr[i] - mean; var += d * d; }
  const float inv = rsqrt(tg_sum(var, red, sg, lane, nsg) / p.D + p.eps);
  for (int i = int(tid); i < p.D; i += int(nt)) y[(long)row * p.D + i] = (xr[i] + rr[i] - mean) * inv * w[i] + bvec[i];
}
