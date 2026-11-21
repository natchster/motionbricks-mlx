// Block reduction with __shared__ memory, __syncthreads and warp shuffles (width 32).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void block_sum(const float* in, float* out, int n) {
  __shared__ float partial[32];
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) v += in[i];
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    if (threadIdx.x == 0) atomicAdd(out, v);
  }
}

int main() {
  const int n = 1 << 22;
  float* h = new float[n];
  for (int i = 0; i < n; ++i) h[i] = 1.0f;
  float *d_in, *d_out, result = 0.f;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  cudaMemcpy(d_in, h, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(float));
  block_sum<<<128, 256>>>(d_in, d_out, n);
  cudaDeviceSynchronize();
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  bool ok = result == (float)n;
  std::printf("reduce_shared n=%d sum=%.0f %s\n", n, result, ok ? "PASS" : "FAIL");
  cudaFree(d_in); cudaFree(d_out); delete[] h;
  return ok ? 0 : 2;
}
