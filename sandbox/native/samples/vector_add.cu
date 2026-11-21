// Classic CUDA vector add: grid-stride loop, cudaMalloc/cudaMemcpy, error checks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                             \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float* ha = (float*)std::malloc(bytes);
  float* hb = (float*)std::malloc(bytes);
  float* hc = (float*)std::malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = std::sin(i) ; hb[i] = std::cos(i); }
  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  vector_add<<<256, 256>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  double max_err = 0;
  for (int i = 0; i < n; ++i) max_err = std::fmax(max_err, std::fabs(hc[i] - (ha[i] + hb[i])));
  std::printf("vector_add n=%d max_err=%g %s\n", n, max_err, max_err < 1e-6 ? "PASS" : "FAIL");
  cudaFree(da); cudaFree(db); cudaFree(dc);
  std::free(ha); std::free(hb); std::free(hc);
  return max_err < 1e-6 ? 0 : 2;
}
