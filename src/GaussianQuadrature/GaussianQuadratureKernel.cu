#include "GaussianQuadratureKernel.cuh"
#include "../../common/FunctionStrategy.cuh"
#include "../../common/Types.h"
#include <curand_kernel.h>

__global__ void gaussianQuadratureKernel(FunctionType functionType, double a, double b, int n, double* results, double* nodes, double* weights) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);
    if (idx < n) {
        double x_i = nodes[idx];
        double w_i = weights[idx];
        results[idx] = w_i * (*function)(0.5 * ((b - a) * x_i + (b + a)));
    }
} 