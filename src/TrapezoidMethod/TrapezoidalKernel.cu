//
// Created by Krystian on 12.05.2025.
//

#include "TrapezoidalKernel.cuh"
#include "../Constants.cuh"
#include "../../common/FunctionStrategy.cuh"

__global__ void trapezoidKernel(FunctionType functionType, double delta, double a, int n, double *results) {
    __shared__  double sharedData[BLOCK_SIZE];

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    DoubleFunctionPtr functionToCalculate = FunctionStrategy::getFunctionReference(functionType);

    sharedData[threadIdx.x] = functionToCalculate(a + static_cast<double>(idx) * delta);

    __syncthreads();

    if (idx < n && threadIdx.x < BLOCK_SIZE - 1) {
        results[idx] = (sharedData[threadIdx.x] + sharedData[threadIdx.x + 1]) / 2.0 * delta;
    }
}
