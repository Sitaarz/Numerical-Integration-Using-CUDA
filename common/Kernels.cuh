//
// Created by Krystian on 1.06.2025.
//

#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void reduceSumKernel(const double* input, double* output, int n);

#endif
// KERNELS_CUH