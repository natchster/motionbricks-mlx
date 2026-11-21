// A dense layer the way CUDA code usually writes it: cuBLAS SGEMM (column-major) followed by a
// custom bias + ReLU epilogue kernel on the same stream, checked against a host reference.
// Under CuMetal the GEMM runs as an MPS matrix multiply and the epilogue as a compiled Metal kernel.
#include <cmath>
#include <cstdio>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void bias_relu(float* c, const float* bias, int m, int n) {  // c is m x n, column-major
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m * n) c[i] = fmaxf(c[i] + bias[i % m], 0.f);
}

int main() {
  const int m = 128, n = 96, k = 64;
  std::vector<float> a(m * k), b(k * n), bias(m), c(m * n, 0.f), ref(m * n, 0.f);
  for (int i = 0; i < m * k; ++i) a[i] = std::sin(0.1f * i);
  for (int i = 0; i < k * n; ++i) b[i] = std::cos(0.07f * i);
  for (int i = 0; i < m; ++i) bias[i] = 0.05f * (i % 7) - 0.15f;
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < m; ++row) {
      double acc = bias[row];
      for (int p = 0; p < k; ++p) acc += (double)a[row + p * m] * b[p + col * k];
      ref[row + col * m] = (float)std::fmax(acc, 0.0);
    }
  float *da, *db, *dbias, *dc;
  cudaMalloc(&da, a.size() * sizeof(float));
  cudaMalloc(&db, b.size() * sizeof(float));
  cudaMalloc(&dbias, bias.size() * sizeof(float));
  cudaMalloc(&dc, c.size() * sizeof(float));
  cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); return 1; }
  const float alpha = 1.f, beta = 0.f;
  if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da, m, db, k, &beta, dc, m) !=
      CUBLAS_STATUS_SUCCESS) { std::printf("cublasSgemm failed\n"); return 1; }
  bias_relu<<<(m * n + 255) / 256, 256>>>(dc, dbias, m, n);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("bias_relu launch failed\n"); return 1;
  }
  cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost);
  double max_err = 0;
  for (int i = 0; i < m * n; ++i) max_err = std::fmax(max_err, std::fabs(c[i] - ref[i]));
  std::printf("sgemm+bias+relu %dx%dx%d max_err=%g %s\n", m, n, k, max_err, max_err < 1e-3 ? "PASS" : "FAIL");
  cublasDestroy(handle);
  cudaFree(da); cudaFree(db); cudaFree(dbias); cudaFree(dc);
  return max_err < 1e-3 ? 0 : 2;
}
