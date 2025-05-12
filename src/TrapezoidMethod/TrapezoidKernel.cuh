//
// Created by Krystian on 12.05.2025.
//

#ifndef TRAPEZOIDKERNEL_CUH
#define TRAPEZOIDKERNEL_CUH

#include "TrapezoidKernel.cuh"
#include "../../common/Types.h"

__global__ void trapezoidKernel(FunctionType functionType, double delta, double a, int n, double* results);

#endif //TRAPEZOIDKERNEL_CUH
