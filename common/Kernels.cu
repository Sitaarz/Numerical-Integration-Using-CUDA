//
// Created by Krystian on 1.06.2025.
//

#include "Kernels.cuh"
#include "../src/Constants.cuh"

__global__ void reduceSumKernel(const double* input, double* output, int n) {
    __shared__ double sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? input[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}