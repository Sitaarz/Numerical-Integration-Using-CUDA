#include "MonteCarloKernel.cuh"
#include "../../common/FunctionStrategy.cuh"
#include "../../common/Types.h"
#include <curand_kernel.h>

__global__ void monteCarloKernel(FunctionType functionType, double a, double b, int n, double* results) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);
    if (idx < n) {
        curandState state;
        unsigned long seed = 1234ULL + idx;
        // unsigned long seed = clock64();
        curand_init(seed, idx, 0, &state);
        results[idx] = (*function)(a + (b - a) * curand_uniform(&state));
    }
} 