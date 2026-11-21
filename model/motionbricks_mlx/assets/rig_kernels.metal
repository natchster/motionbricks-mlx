// SPDX-License-Identifier: Apache-2.0
// Rig kernels shared by motionbricks_mlx (mx.fast.metal_kernel) and model-rs (mlx-rs MetalKernel).
// Sections are split on the `// @@ NAME` markers; each body after HEADER is the body of a kernel
// whose signature MLX generates from the input/output names.

// @@ HEADER
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
inline M3 load3(const device float* p) {
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
inline float3 nrm(float3 v) { return v / sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
inline float3 crs(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
// rig.cont6d_to_matrix: columns (x, y, z)
inline M3 six_to_m(thread const float* s) {
  float3 x = nrm(float3(s[0], s[1], s[2]));
  float3 z = nrm(crs(x, float3(s[3], s[4], s[5])));
  float3 y = crs(z, x);
  M3 o;
  for (int r = 0; r < 3; r++) { o.m[r][0] = x[r]; o.m[r][1] = y[r]; o.m[r][2] = z[r]; }
  return o;
}
// rig.quat_matrix (WXYZ, 2 / |q|^2 normalization)
inline M3 quat_m(float r, float i, float j, float k) {
  float s = 2.0f / (r * r + i * i + j * j + k * k);
  M3 o;
  o.m[0][0] = 1.0f - s * (j * j + k * k); o.m[0][1] = s * (i * j - k * r); o.m[0][2] = s * (i * k + j * r);
  o.m[1][0] = s * (i * j + k * r); o.m[1][1] = 1.0f - s * (i * i + k * k); o.m[1][2] = s * (j * k - i * r);
  o.m[2][0] = s * (i * k - j * r); o.m[2][1] = s * (j * k + i * r); o.m[2][2] = 1.0f - s * (i * i + j * j);
  return o;
}
// rig.matrix_quat
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
// MLX remainder semantics (sign of the divisor), rig.wrap
inline float rem(float a, float b) {
  float r = fmod(a, b);
  return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r;
}
inline float wrap_pi(float a) { return rem(a + M_PI_F, 2.0f * M_PI_F) - M_PI_F; }
// Packed G1 rig (G1Rig.rf / G1Rig.ri; sizes in kernels.G1_RF_LAYOUT / G1_RI_LAYOUT, checked by test_layers):
// floats basis 9 | axes 29*3 | q2t_mapped 29*9 | rest 30*3 |
// neutral 34*3 | f2q 34*9 | model_axes 29*3; ints parents 30 | model_parents 34 | reverse 34 | mapping 29
constant constexpr int F_BASIS = 0, F_AXES = 9, F_Q2T = 96, F_REST = 357, F_NEUTRAL = 447, F_F2Q = 549, F_MAXES = 855;
constant constexpr int I_PARENTS = 0, I_MPARENTS = 30, I_REVERSE = 64, I_MAPPING = 98;

// @@ G1_PREPROCESS
  const uint b = thread_position_in_grid.x;
  if (b >= B) return;
  const M3 basis = load3(rf + F_BASIS);
  const M3 bt = tr(basis);
  float3 root[8];
  float heading[8];
  for (int f = 0; f < 8; f++) {
    const device float* q = f < 4 ? context + (b * 4 + f) * 36 : target + (b * 4 + f - 4) * 36;
    M3 rot[30];
    float3 pos[30];
    rot[0] = quat_m(q[6], q[3], q[4], q[5]);  // XYZW -> WXYZ
    pos[0] = float3(q[0], q[1], q[2]) + float3(rf[F_REST], rf[F_REST + 1], rf[F_REST + 2]);
    for (int jt = 1; jt < 30; jt++) {
      int par = ri[I_PARENTS + jt];
      float h = q[7 + jt - 1] / 2.0f;
      float c = cos(h), s = sin(h);
      const device float* ax = rf + F_AXES + (jt - 1) * 3;
      M3 local = mul(load3(rf + F_Q2T + (jt - 1) * 9), quat_m(c, s * ax[0], s * ax[1], s * ax[2]));
      float3 off = float3(rf[F_REST + jt * 3] - rf[F_REST + par * 3], rf[F_REST + jt * 3 + 1] - rf[F_REST + par * 3 + 1],
                          rf[F_REST + jt * 3 + 2] - rf[F_REST + par * 3 + 2]);
      pos[jt] = pos[par] + mv(rot[par], off);
      rot[jt] = mul(rot[par], local);
    }
    float3 mp[34];
    M3 mr[34];
    for (int jt = 0; jt < 34; jt++) {
      int robot = ri[I_REVERSE + jt];
      if (robot >= 0) {
        mp[jt] = mv(basis, pos[robot]);
        mr[jt] = mul(mul(basis, rot[robot]), bt);
      } else {
        int par = ri[I_MPARENTS + jt];
        float3 off = float3(rf[F_NEUTRAL + jt * 3] - rf[F_NEUTRAL + par * 3], rf[F_NEUTRAL + jt * 3 + 1] - rf[F_NEUTRAL + par * 3 + 1],
                            rf[F_NEUTRAL + jt * 3 + 2] - rf[F_NEUTRAL + par * 3 + 2]);
        mp[jt] = mp[par] + mv(mr[par], off);
        mr[jt] = eye3();
      }
    }
    root[f] = mp[0];
    heading[f] = atan2(mr[0].m[0][2], mr[0].m[2][2]);
    device float* g = global_root + (b * 8 + f) * 5;
    g[0] = mp[0].x; g[1] = mp[0].y; g[2] = mp[0].z; g[3] = cos(heading[f]); g[4] = sin(heading[f]);
    device float* l = local_poses + (b * 8 + f) * 303;
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
      device float* o = local_root + (b * 8 + f) * 4;
      o[0] = wrap_pi(heading[f + 1] - heading[f]) * 30.0f;
      o[1] = (root[f + 1].x - root[f].x) * 30.0f;
      o[2] = (root[f + 1].z - root[f].z) * 30.0f;
      o[3] = root[f].y;
    }
    device float* o = local_root + (b * 8 + start + 3) * 4;
    o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = root[start + 3].y;
  }

// @@ G1_POSTPROCESS
  const uint row = thread_position_in_grid.x;
  if (row >= ROWS) return;
  const device float* g = features + row * 414;
  M3 r[34];
  for (int j = 0; j < 34; j++) {
    float six[6];
    for (int i = 0; i < 6; i++) six[i] = g[104 + 6 * j + i];
    r[j] = six_to_m(six);
  }
  const M3 basis = load3(rf + F_BASIS);
  const M3 bt = tr(basis);
  device float* q = qpos + row * 36;
  float3 rp = mv(bt, float3(g[0], g[1], g[2]));
  q[0] = rp.x; q[1] = rp.y; q[2] = rp.z;
  M3 local0 = mul(load3(rf + F_F2Q), mul(tr(eye3()), r[0]));
  float4 wxyz = matrix_quat(mul(mul(bt, local0), basis));
  q[3] = wxyz.y; q[4] = wxyz.z; q[5] = wxyz.w; q[6] = wxyz.x;
  for (int i = 0; i < 29; i++) {
    int j = ri[I_MAPPING + i];
    M3 par = j == 0 ? eye3() : r[ri[I_MPARENTS + j]];
    M3 h = mul(load3(rf + F_F2Q + 9 * j), mul(tr(par), r[j]));
    float3 ang = float3(atan2(h.m[2][1], h.m[2][2]), atan2(h.m[0][2], h.m[0][0]), atan2(h.m[1][0], h.m[1][1]));
    const device float* ax = rf + F_MAXES + 3 * i;
    q[7 + i] = ang.x * ax[0] + ang.y * ax[1] + ang.z * ax[2];
  }
  for (int i = 0; i < 4; i++) contacts[row * 4 + i] = g[410 + i];

// @@ SOMA_RETARGET
  const uint gid = thread_position_in_grid.x;
  if (gid >= ROWS * J) return;
  const uint row = gid / J, j = gid % J;
  const device float* s = six + row * J * 6 + 6 * j;
  float v[6] = {s[0], s[1], s[2], s[3], s[4], s[5]};
  M3 m = mul(mul(six_to_m(v), load3(src + 9 * j)), tr(load3(dst + 9 * j)));
  device float* o = out + row * J * 6 + 6 * j;
  o[0] = m.m[0][0]; o[1] = m.m[1][0]; o[2] = m.m[2][0];
  o[3] = m.m[0][1]; o[4] = m.m[1][1]; o[5] = m.m[2][1];
