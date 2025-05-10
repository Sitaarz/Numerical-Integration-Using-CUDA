//
// Created by Krystian on 10.05.2025.
//

#include "RectangleKernel.cuh"
#include "../../common/FunctionFactory.cuh"
#include "../../common/Types.h"
__global__ void rectangleKernel(FunctionType functionType, double delta, double a, int n, double* results) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    DoubleFunctionPtr function = FunctionFactory::getFunctionReference(functionType);
    if (idx < n) {
        results[idx] = (*function)(a + static_cast<double>(idx) * delta) * delta;
    }
} 