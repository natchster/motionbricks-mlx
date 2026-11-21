// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
// MotionBricks glue and rig kernels (one thread per item; transcriptions of inference.py,
// representation.py and rig.py). Masks are float 0/1. FPS = 30, frames per token = 4.

constant constexpr float FPS = 30.0f;

struct Dims {
  int B;        // batch
  int PD;       // local pose features incl. hip height (SOMA 268, G1 304)
  int LD;       // local layout dim (SOMA 365, G1 413)
  int GD;       // global layout dim (SOMA 366, G1 414)
  int E;        // root embedding width (512)
  int H;        // pose codebook heads
  int codes;    // codes per head
  int cw;       // code width per head
  int len_w;    // length embedding width (128)
  int forced;   // 1: num_tokens input forces the duration
};

// --- input normalization (inference.py: recentring + normalize) --------------------------------
// gr (B,8,5), lr (B,8,4), lp (B,8,PD-1) -> gn, ln, pn; info[b] = (heading0, off_x, off_z, 0)
[[kernel]] void prep(
    device const float* gr [[buffer(0)]], device const float* lr [[buffer(1)]], device const float* lp [[buffer(2)]],
    device const float* stats [[buffer(3)]],  // g_mean5 g_scale5 l_mean4 l_scale4 p_mean[PD] p_scale[PD]
    device float* gn [[buffer(4)]], device float* ln [[buffer(5)]], device float* pn [[buffer(6)]],
    device float* info [[buffer(7)]], constant Dims& d [[buffer(8)]],
    device T* gn_t [[buffer(9)]], device T* ln_t [[buffer(10)]], device T* pn_t [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 8) return;
  const int b = int(gid) / 8;
  device const float* g0 = gr + b * 40;
  const float ox = g0[0], oz = g0[2];
  device const float* g = gr + gid * 5;
  const float rec[5] = {g[0] - ox, g[1] - 0.0f, g[2] - oz, g[3] - 0.0f, g[4] - 0.0f};
  device const float* gm = stats;
  device const float* gs = stats + 5;
  device const float* lm = stats + 10;
  device const float* ls = stats + 14;
  device const float* pm = stats + 18;
  device const float* ps = stats + 18 + d.PD;
  for (int i = 0; i < 5; i++) { float v = (rec[i] - gm[i]) / gs[i]; gn[gid * 5 + i] = v; gn_t[gid * 5 + i] = T(v); }
  for (int i = 0; i < 4; i++) { float v = (lr[gid * 4 + i] - lm[i]) / ls[i]; ln[gid * 4 + i] = v; ln_t[gid * 4 + i] = T(v); }
  { float v = (rec[1] - pm[0]) / ps[0]; pn[gid * d.PD] = v; pn_t[gid * d.PD] = T(v); }
  for (int i = 1; i < d.PD; i++) { float v = (lp[gid * (d.PD - 1) + i - 1] - pm[i]) / ps[i]; pn[gid * d.PD + i] = v; pn_t[gid * d.PD + i] = T(v); }
  if (gid % 8 == 0) {
    info[b * 4 + 0] = atan2(g0[4], g0[3]);
    info[b * 4 + 1] = ox;
    info[b * 4 + 2] = oz;
    info[b * 4 + 3] = 0.0f;
  }
}

// --- rotations (rig.py) --------------------------------------------------------------------------
inline float3 nrm(float3 v) { return v / sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
inline float3 crs(float3 a, float3 b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
// m[r][c] row-major 3x3 helpers
struct M3 { float m[3][3]; };
inline M3 mul(M3 a, M3 b) {
  M3 o;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) o.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c];
  return o;
}
inline M3 tr(M3 a) {
  M3 o;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) o.m[r][c] = a.m[c][r];
  return o;
}
inline float3 mv(M3 a, float3 v) {
  return float3(a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z, a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
                a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}
inline M3 load3(device const float* p) {
  M3 o;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) o.m[r][c] = p[r * 3 + c];
  return o;
}
inline M3 eye3() {
  M3 o;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) o.m[r][c] = r == c ? 1.0f : 0.0f;
  return o;
}
// cont6d_to_matrix: columns (x, y, z)
inline M3 six_to_m(thread const float* s) {
  float3 x = nrm(float3(s[0], s[1], s[2]));
  float3 z = nrm(crs(x, float3(s[3], s[4], s[5])));
  float3 y = crs(z, x);
  M3 o;
  for (int r = 0; r < 3; r++) { o.m[r][0] = x[r]; o.m[r][1] = y[r]; o.m[r][2] = z[r]; }
  return o;
}
// WXYZ quaternion -> matrix (2 / |q|^2 normalization)
inline M3 quat_m(float r, float i, float j, float k) {
  float s = 2.0f / (r * r + i * i + j * j + k * k);
  M3 o;
  o.m[0][0] = 1.0f - s * (j * j + k * k); o.m[0][1] = s * (i * j - k * r); o.m[0][2] = s * (i * k + j * r);
  o.m[1][0] = s * (i * j + k * r); o.m[1][1] = 1.0f - s * (i * i + k * k); o.m[1][2] = s * (j * k - i * r);
  o.m[2][0] = s * (i * k - j * r); o.m[2][1] = s * (j * k + i * r); o.m[2][2] = 1.0f - s * (i * i + j * j);
  return o;
}

// SOMA capture <-> standard T-pose retarget of 6D rotations, in place:
// rows of `x` (row stride `ld`), joint j's 6D at col0 + 6j: m <- (m @ from[j]) @ to[j]^T.
[[kernel]] void soma_retarget(
    device float* x [[buffer(0)]], device const float* from [[buffer(1)]], device const float* to [[buffer(2)]],
    constant int4& p [[buffer(3)]],  // rows, ld, col0, joints
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.x * p.w) return;
  const int row = int(gid) / p.w, j = int(gid) % p.w;
  device float* s = x + (long)row * p.y + p.z + 6 * j;
  float six[6] = {s[0], s[1], s[2], s[3], s[4], s[5]};
  M3 m = mul(mul(six_to_m(six), load3(from + 9 * j)), tr(load3(to + 9 * j)));
  s[0] = m.m[0][0]; s[1] = m.m[1][0]; s[2] = m.m[2][0];
  s[3] = m.m[0][1]; s[4] = m.m[1][1]; s[5] = m.m[2][1];
}

// MLX remainder semantics (result takes the divisor's sign).
inline float rem(float a, float b) {
  float r = fmod(a, b);
  return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r;
}
inline float wrap_pi(float a) { return rem(a + M_PI_F, 2.0f * M_PI_F) - M_PI_F; }

// --- Unitree G1 forward kinematics + feature extraction (G1Rig.fk / preprocess) ------------------
// rig floats: basis 9 | axes 29*3 | q2t_mapped 29*9 | rest 30*3 | neutral 34*3 | f2q 34*9 | model_axes 29*3
// rig ints:   parents 30 | model_parents 34 | reverse 34 | mapping 29
struct G1Off { int basis, axes, q2t, rest, neutral, f2q, model_axes; int parents, model_parents, reverse, mapping; };
constant constexpr G1Off GO = {0, 9, 9 + 87, 9 + 87 + 261, 9 + 87 + 261 + 90, 9 + 87 + 261 + 90 + 102,
                               9 + 87 + 261 + 90 + 102 + 306, 0, 30, 64, 98};

// One thread per batch item: 8 poses (4 context + 4 target), XYZW root quaternion inputs.
[[kernel]] void g1_preprocess(
    device const float* ctx [[buffer(0)]], device const float* tgt [[buffer(1)]],
    device const float* rf [[buffer(2)]], device const int* ri [[buffer(3)]],
    device float* gr [[buffer(4)]], device float* lr [[buffer(5)]], device float* lp [[buffer(6)]],
    constant Dims& d [[buffer(7)]], uint b [[thread_position_in_grid]]) {
  if (int(b) >= d.B) return;
  const M3 basis = load3(rf + GO.basis);
  float3 root[8];
  float heading[8];
  for (int f = 0; f < 8; f++) {
    device const float* q = (f < 4 ? ctx + (b * 4 + f) * 36 : tgt + (b * 4 + f - 4) * 36);
    // XYZW -> WXYZ
    M3 rot[30];
    float3 pos[30];
    rot[0] = quat_m(q[6], q[3], q[4], q[5]);
    float3 rest0 = float3(rf[GO.rest], rf[GO.rest + 1], rf[GO.rest + 2]);
    pos[0] = float3(q[0], q[1], q[2]) + rest0;
    for (int jt = 1; jt < 30; jt++) {
      int par = ri[GO.parents + jt];
      float d2 = q[7 + jt - 1] / 2.0f;
      float c = cos(d2), s = sin(d2);
      device const float* ax = rf + GO.axes + (jt - 1) * 3;
      M3 local = mul(load3(rf + GO.q2t + (jt - 1) * 9), quat_m(c, s * ax[0], s * ax[1], s * ax[2]));
      float3 off = float3(rf[GO.rest + jt * 3] - rf[GO.rest + par * 3], rf[GO.rest + jt * 3 + 1] - rf[GO.rest + par * 3 + 1],
                          rf[GO.rest + jt * 3 + 2] - rf[GO.rest + par * 3 + 2]);
      pos[jt] = pos[par] + mv(rot[par], off);
      rot[jt] = mul(rot[par], local);
    }
    // to model space (34 joints)
    float3 mp[34];
    M3 mr[34];
    M3 bt = tr(basis);
    for (int jt = 0; jt < 34; jt++) {
      int robot = ri[GO.reverse + jt];
      if (robot >= 0) {
        mp[jt] = mv(basis, pos[robot]);
        mr[jt] = mul(mul(basis, rot[robot]), bt);
      } else {
        int par = ri[GO.model_parents + jt];
        float3 off = float3(rf[GO.neutral + jt * 3] - rf[GO.neutral + par * 3], rf[GO.neutral + jt * 3 + 1] - rf[GO.neutral + par * 3 + 1],
                            rf[GO.neutral + jt * 3 + 2] - rf[GO.neutral + par * 3 + 2]);
        mp[jt] = mp[par] + mv(mr[par], off);
        mr[jt] = eye3();
      }
    }
    root[f] = mp[0];
    heading[f] = atan2(mr[0].m[0][2], mr[0].m[2][2]);
    device float* g = gr + (b * 8 + f) * 5;
    g[0] = mp[0].x; g[1] = mp[0].y; g[2] = mp[0].z; g[3] = cos(heading[f]); g[4] = sin(heading[f]);
    device float* l = lp + (long)(b * 8 + f) * (d.PD - 1);
    for (int jt = 1; jt < 34; jt++) {
      float3 nr = mp[jt] - mp[0] * float3(1.0f, 0.0f, 1.0f);
      l[(jt - 1) * 3 + 0] = nr.x; l[(jt - 1) * 3 + 1] = nr.y; l[(jt - 1) * 3 + 2] = nr.z;
    }
    for (int jt = 0; jt < 34; jt++) {
      device float* s = l + 99 + jt * 6;
      s[0] = mr[jt].m[0][0]; s[1] = mr[jt].m[1][0]; s[2] = mr[jt].m[2][0];
      s[3] = mr[jt].m[0][1]; s[4] = mr[jt].m[1][1]; s[5] = mr[jt].m[2][1];
    }
  }
  for (int start = 0; start < 8; start += 4) {
    for (int i = 0; i < 3; i++) {
      int f = start + i;
      device float* o = lr + (b * 8 + f) * 4;
      o[0] = wrap_pi(heading[f + 1] - heading[f]) * 30.0f;
      o[1] = (root[f + 1].x - root[f].x) * 30.0f;
      o[2] = (root[f + 1].z - root[f].z) * 30.0f;
      o[3] = root[f].y;
    }
    device float* o = lr + (b * 8 + start + 3) * 4;
    o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = root[start + 3].y;
  }
}

// --- root network glue ---------------------------------------------------------------------------
// X1 (B, 9, E): row 0 = num-token query embedding, rows 1..8 = positioned frame embeddings.
// FE (B, 8, E) = frame embeddings (start block for frames 0..3, end block for 4..7).
[[kernel]] void root_assemble(
    device const T* S [[buffer(0)]], device const T* Eb [[buffer(1)]],
    device const T* pos_emb [[buffer(2)]], device const T* query_emb [[buffer(3)]],
    device const int* nt_in [[buffer(4)]], device T* X1 [[buffer(5)]], device T* FE [[buffer(6)]],
    constant Dims& d [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 9 * d.E) return;
  const int c = int(gid) % d.E, r = (int(gid) / d.E) % 9, b = int(gid) / (9 * d.E);
  if (r == 0) {
    int q = (d.forced ? nt_in[b] : 18) - 6;
    X1[gid] = query_emb[q * d.E + c];
    return;
  }
  const int f = r - 1;
  T fe = (f < 4 ? S : Eb)[(long)(b * 8 + f) * d.E + c];
  FE[(long)(b * 8 + f) * d.E + c] = fe;
  X1[gid] = T(float(fe) + float(pos_emb[f * d.E + c]));
}

// Duration choice (shared by the kernels below): masked logits from the first token's head.
inline int choose(device const T* logits, device const float* allowed, device const int* nt_in, constant Dims& d, int b) {
  if (d.forced && nt_in[b] != 18) return nt_in[b] - 6;
  device const T* l = logits + (long)b * 9 * 12;  // row b*9 of (B*9, 12)
  int best = 0;
  float bv = -INFINITY;
  for (int i = 0; i < 11; i++) {
    float v = allowed[b * 11 + i] != 0.0f ? float(l[i]) : -INFINITY;
    if (v > bv) { bv = v; best = i; }
  }
  return best;
}

// X2 (B, 25, E): middle token, positioned frames, position table; pad2 (B, 25);
// nt (B), token masks m16 (B,16), m32 (B,32), m64 (B,64), pose padding (B,16), masked logits (B,12).
[[kernel]] void root_second_input(
    device const T* logits [[buffer(0)]], device const float* allowed [[buffer(1)]],
    device const int* nt_in [[buffer(2)]], device const T* FE [[buffer(3)]],
    device const T* middle_emb [[buffer(4)]], device const T* pos_table [[buffer(5)]],
    device T* X2 [[buffer(6)]], device float* pad2 [[buffer(7)]], device int* nt [[buffer(8)]],
    device float* m16 [[buffer(9)]], device float* logits_out [[buffer(10)]], constant Dims& d [[buffer(11)]],
    device float* m32 [[buffer(12)]], device float* m64 [[buffer(13)]], device float* ppad [[buffer(14)]],
    device const T* in_pos [[buffer(15)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 25 * d.E) return;
  const int c = int(gid) % d.E, r = (int(gid) / d.E) % 25, b = int(gid) / (25 * d.E);
  const int chosen = choose(logits, allowed, nt_in, d, b);
  const int pred = chosen + 6;
  T v;
  if (r == 0) v = middle_emb[chosen * d.E + c];
  else if (r < 9) v = T(float(FE[(long)(b * 8 + r - 1) * d.E + c]) + float(in_pos[(r - 1) * d.E + c]));  // positioned (pre-transformer)
  else v = pos_table[(r - 9) * d.E + c];
  X2[gid] = v;
  if (c == 0) pad2[b * 25 + r] = (r >= 9 && (r - 9) >= pred) ? 1.0f : 0.0f;
  if (r == 0 && c < 64) {
    m64[b * 64 + c] = c < 4 * pred ? 1.0f : 0.0f;
    if (c < 32) m32[b * 32 + c] = c < 2 * pred ? 1.0f : 0.0f;
    if (c < 16) { m16[b * 16 + c] = c < pred ? 1.0f : 0.0f; ppad[b * 16 + c] = c >= pred ? 1.0f : 0.0f; }
    if (c == 0) nt[b] = pred;
    if (c < 12) {
      device const T* l = logits + (long)b * 9 * 12;
      logits_out[b * 12 + c] = (c < 11 && allowed[b * 11 + c] != 0.0f) ? float(l[c]) : -INFINITY;
    }
  }
}

// Dense 64-frame decoder conditions: dense_frame (B,64,E), dense_root (B,64,5), dense_has (B,64).
[[kernel]] void root_dense_cond(
    device const T* FE [[buffer(0)]], device const float* gn [[buffer(1)]],
    device const float* hg [[buffer(2)]], device const float* hl [[buffer(3)]], device const float* hp [[buffer(4)]],
    device const T* no_frame [[buffer(5)]], device const int* nt [[buffer(6)]],
    device T* dframe [[buffer(7)]], device T* droot [[buffer(8)]], device float* dhas [[buffer(9)]],
    constant Dims& d [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 64 * d.E) return;
  const int c = int(gid) % d.E, t = (int(gid) / d.E) % 64, b = int(gid) / (64 * d.E);
  const int pred = nt[b];
  int src = -1;  // frame index 0..7 supplying this dense frame
  if (t < 4) src = t;
  for (int i = 0; i < 4; i++)
    if (t == pred * 4 - 4 + i) src = 4 + i;
  T v = no_frame[c];
  if (src >= 0) {
    int fi = b * 8 + src;
    bool has_frame = hp[fi] != 0.0f || hg[fi] != 0.0f || hl[fi] != 0.0f;
    v = has_frame ? FE[(long)fi * d.E + c] : no_frame[c];
  }
  dframe[gid] = v;
  if (c < 5) droot[(b * 64 + t) * 5 + c] = T(src >= 0 ? gn[(b * 8 + src) * 5 + c] : 0.0f);
  if (c == 0) dhas[b * 64 + t] = src >= 0 ? hg[b * 8 + src] : 0.0f;
}

// dst[b, r, :] = src[b, r + off, :] for r < rows_out (rows_in per batch item in src).
[[kernel]] void copy_rows(
    device const T* src [[buffer(0)]], device T* dst [[buffer(1)]],
    constant int4& p [[buffer(2)]],  // B, rows_in, rows_out, off
    constant int& width [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.x * p.z * width) return;
  const int c = int(gid) % width, r = (int(gid) / width) % p.z, b = int(gid) / (width * p.z);
  dst[gid] = src[((long)b * p.y + r + p.w) * width + c];
}

// --- root -> pose (global_root_to_local, last-frame patch, dense pose conditions) ----------------
// One thread per (b, t). PR: normalized pred root (B,64,5).
[[kernel]] void root_to_pose(
    device const T* PR [[buffer(0)]], device const int* nt [[buffer(1)]],
    device const float* stats [[buffer(2)]],  // as in prep
    device const float* ln [[buffer(3)]], device const float* hl [[buffer(4)]],
    device const float* pn [[buffer(5)]], device const float* hp [[buffer(6)]],
    device const T* len_emb [[buffer(7)]],
    device T* ext [[buffer(8)]], device T* rcond [[buffer(9)]], device T* pcond [[buffer(10)]],
    device float* hpc [[buffer(11)]], device T* LE [[buffer(12)]],
    constant Dims& d [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 64) return;
  const int b = int(gid) / 64, t = int(gid) % 64;
  device const float* gm = stats;
  device const float* gs = stats + 5;
  device const float* lm = stats + 10;
  device const float* ls = stats + 14;
  const int pred = nt[b];
  const int last = pred * 4 - 1;
  auto root = [&](int f, int i) { return float(PR[(b * 64 + f) * 5 + i]) * gs[i] + gm[i]; };
  auto angle = [&](int f) {
    float h = atan2(root(f, 4), root(f, 3));
    return atan2(sin(h / 2.0f), cos(h / 2.0f)) * 2.0f;
  };
  // velocities at frame f (the last valid frame repeats the previous one)
  auto vel = [&](int f, thread float& ang, thread float& vx, thread float& vz) {
    if (f >= 63) { ang = 0.0f; vx = 0.0f; vz = 0.0f; return; }
    float a0 = angle(f), a1 = angle(f + 1);
    float c0 = cos(a0), s0 = sin(a0), c1 = cos(a1), s1 = sin(a1);
    float cd = c1 * c0 + s1 * s0, sd = s1 * c0 - c1 * s0;
    ang = FPS * atan2(sd, cd);
    vx = FPS * (root(f + 1, 0) - root(f, 0));
    vz = FPS * (root(f + 1, 2) - root(f, 2));
  };
  float ang, vx, vz;
  vel(t == last ? t - 1 : t, ang, vx, vz);
  float lrn[4] = {(ang - lm[0]) / ls[0], (vx - lm[1]) / ls[1], (vz - lm[2]) / ls[2], (root(t, 1) - lm[3]) / ls[3]};
  if (t == last) {
    float h = hl[b * 8 + 7];
    for (int i = 0; i < 4; i++) lrn[i] = h * ln[(b * 8 + 7) * 4 + i] + (1.0f - h) * lrn[i];
  }
  ext[gid * 2 + 0] = T(lrn[1]);
  ext[gid * 2 + 1] = T(lrn[2]);
  rcond[gid * 4 + 0] = PR[gid * 5 + 0];
  rcond[gid * 4 + 1] = PR[gid * 5 + 2];
  rcond[gid * 4 + 2] = PR[gid * 5 + 3];
  rcond[gid * 4 + 3] = PR[gid * 5 + 4];
  int src = -1;
  if (t < 4) src = t;
  for (int i = 0; i < 4; i++)
    if (t == pred * 4 - 4 + i) src = 4 + i;
  for (int i = 0; i < d.PD; i++) pcond[(long)gid * d.PD + i] = T(src >= 0 ? pn[(long)(b * 8 + src) * d.PD + i] : 0.0f);
  hpc[gid] = src >= 0 ? hp[b * 8 + src] : 0.0f;
  if (t < 16)
    for (int i = 0; i < d.len_w; i++) LE[(b * 16 + t) * d.len_w + i] = len_emb[(pred - 6) * d.len_w + i];
}

// --- pose tokens: argmax per head + codebook lookup -----------------------------------------------
[[kernel]] void argmax_dequant(
    device const T* logits [[buffer(0)]], device const T* codebook [[buffer(1)]],
    device int* tokens [[buffer(2)]], device T* lat [[buffer(3)]],
    constant Dims& d [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 16 * d.H) return;
  const int h = int(gid) % d.H, row = int(gid) / d.H;
  device const T* l = logits + (long)row * d.H * d.codes + h * d.codes;
  int best = 0;
  float bv = float(l[0]);
  for (int i = 1; i < d.codes; i++)
    if (float(l[i]) > bv) { bv = float(l[i]); best = i; }
  tokens[gid] = best;
  for (int i = 0; i < d.cw; i++) lat[(long)row * d.H * d.cw + h * d.cw + i] = codebook[(h * d.codes + best) * d.cw + i];
}

// --- decode: local -> global, final root, re-anchoring --------------------------------------------
// One thread per batch item. recon (B,64,LD) normalized local; out (B,64,GD) gets the root (5).
// mode 1 = SOMA "from_height_constraint_violation", 0 = G1 "from_pose_module".
[[kernel]] void finish_root(
    device const T* recon [[buffer(0)]], device const T* PR [[buffer(1)]],
    device const float* stats [[buffer(2)]], device const float* gn [[buffer(3)]],
    device const float* hg [[buffer(4)]], device const int* nt [[buffer(5)]], device const float* info [[buffer(6)]],
    device float* out [[buffer(7)]], constant Dims& d [[buffer(8)]], constant int& mode [[buffer(9)]],
    uint b [[thread_position_in_grid]]) {
  if (int(b) >= d.B) return;
  device const float* gm = stats;
  device const float* gs = stats + 5;
  device const float* lm = stats + 10;
  device const float* ls = stats + 14;
  float GR[64][5];
  float yaw = 0.0f, x = 0.0f, z = 0.0f;
  for (int t = 0; t < 64; t++) {
    device const T* r = recon + (long)(b * 64 + t) * d.LD;
    float r0 = float(r[0]) * ls[0] + lm[0], r1 = float(r[1]) * ls[1] + lm[1], r2 = float(r[2]) * ls[2] + lm[2], r3 = float(r[3]) * ls[3] + lm[3];
    float hd = atan2(sin(yaw / 2.0f), cos(yaw / 2.0f)) * 2.0f;
    GR[t][0] = x; GR[t][1] = r3; GR[t][2] = z; GR[t][3] = cos(hd); GR[t][4] = sin(hd);
    yaw += r0 / FPS;
    x += r1 / FPS;
    z += r2 / FPS;
  }
  bool use_root = false;
  if (mode == 1) {
    const int pred = nt[b];
    float ep = 0.0f, er = 0.0f, ws = 0.0f;
    for (int i = 0; i < 4; i++) {
      int tf = (pred - 1) * 4 + i;
      float h_root = float(PR[(b * 64 + tf) * 5 + 1]) * gs[1] + gm[1];
      float h_pose = GR[tf][1];
      float h_tgt = gn[(b * 8 + 4 + i) * 5 + 1] * gs[1] + gm[1];
      float w = hg[b * 8 + 4 + i];
      ep += abs(h_pose - h_tgt) * w;
      er += abs(h_root - h_tgt) * w;
      ws += w;
    }
    float err_pose = ep / ws, err_root = er / ws;
    use_root = (err_root < err_pose) && (err_pose > 0.05f);
  }
  const float h0 = info[b * 4], ox = info[b * 4 + 1], oz = info[b * 4 + 2];
  for (int t = 0; t < 64; t++) {
    float rt[5];
    for (int i = 0; i < 5; i++) rt[i] = use_root ? float(PR[(b * 64 + t) * 5 + i]) * gs[i] + gm[i] : GR[t][i];
    float a = atan2(rt[4], rt[3]) + h0;
    device float* o = out + (long)(b * 64 + t) * d.GD;
    o[0] = rt[0] + ox; o[1] = rt[1]; o[2] = rt[2] + oz; o[3] = cos(a); o[4] = sin(a);
  }
}

// out[b, t, 5 + i] = unnormalized body feature i (thread per element).
[[kernel]] void body_out(
    device const T* recon [[buffer(0)]], device const float* body_stats [[buffer(1)]],
    device float* out [[buffer(2)]], constant Dims& d [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const int nb = d.LD - 4;
  if (int(gid) >= d.B * 64 * nb) return;
  const int i = int(gid) % nb, row = int(gid) / nb;
  out[(long)row * d.GD + 5 + i] = float(recon[(long)row * d.LD + 4 + i]) * body_stats[nb + i] + body_stats[i];
}

// --- Unitree G1 joint-angle decoding (G1Rig.decode / postprocess), one thread per frame -----------
inline float4 matrix_quat(M3 m) {
  float m00 = m.m[0][0], m01 = m.m[0][1], m02 = m.m[0][2], m10 = m.m[1][0], m11 = m.m[1][1], m12 = m.m[1][2];
  float m20 = m.m[2][0], m21 = m.m[2][1], m22 = m.m[2][2];
  float a[4] = {sqrt(max(1.0f + m00 + m11 + m22, 0.0f)), sqrt(max(1.0f + m00 - m11 - m22, 0.0f)),
                sqrt(max(1.0f - m00 + m11 - m22, 0.0f)), sqrt(max(1.0f - m00 - m11 + m22, 0.0f))};
  float rows[4][4] = {{a[0] * a[0], m21 - m12, m02 - m20, m10 - m01},
                      {m21 - m12, a[1] * a[1], m10 + m01, m02 + m20},
                      {m02 - m20, m10 + m01, a[2] * a[2], m12 + m21},
                      {m10 - m01, m20 + m02, m21 + m12, a[3] * a[3]}};
  int best = 0;
  for (int i = 1; i < 4; i++)
    if (a[i] > a[best]) best = i;
  float den = 2.0f * max(a[best], 0.1f);
  return float4(rows[best][0] / den, rows[best][1] / den, rows[best][2] / den, rows[best][3] / den);
}

[[kernel]] void g1_postprocess(
    device const float* G [[buffer(0)]], device const float* rf [[buffer(1)]], device const int* ri [[buffer(2)]],
    device float* qpos [[buffer(3)]], device float* contacts [[buffer(4)]],
    constant Dims& d [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= d.B * 64) return;
  device const float* g = G + (long)gid * d.GD;
  const int rot0 = 5 + 99;
  M3 r[34];
  for (int j = 0; j < 34; j++) {
    float six[6];
    for (int i = 0; i < 6; i++) six[i] = g[rot0 + 6 * j + i];
    r[j] = six_to_m(six);
  }
  const M3 basis = load3(rf + GO.basis);
  const M3 bt = tr(basis);
  device float* q = qpos + (long)gid * 36;
  float3 rp = mv(bt, float3(g[0], g[1], g[2]));
  q[0] = rp.x; q[1] = rp.y; q[2] = rp.z;
  M3 local0 = mul(load3(rf + GO.f2q), mul(tr(eye3()), r[0]));
  float4 wxyz = matrix_quat(mul(mul(bt, local0), basis));
  q[3] = wxyz.y; q[4] = wxyz.z; q[5] = wxyz.w; q[6] = wxyz.x;
  for (int i = 0; i < 29; i++) {
    int j = ri[GO.mapping + i];
    M3 par = j == 0 ? eye3() : r[ri[GO.model_parents + j]];
    M3 h = mul(load3(rf + GO.f2q + 9 * j), mul(tr(par), r[j]));
    float3 ang = float3(atan2(h.m[2][1], h.m[2][2]), atan2(h.m[0][2], h.m[0][0]), atan2(h.m[1][0], h.m[1][1]));
    device const float* ax = rf + GO.model_axes + 3 * i;
    q[7 + i] = ang.x * ax[0] + ang.y * ax[1] + ang.z * ax[2];
  }
  for (int i = 0; i < 4; i++) contacts[gid * 4 + i] = g[d.GD - 4 + i];
}
