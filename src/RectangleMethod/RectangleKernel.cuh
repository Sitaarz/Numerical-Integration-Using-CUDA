//
// Created by Krystian on 10.05.2025.
//

#ifndef RECTANGLEKERNEL_CUH
#define RECTANGLEKERNEL_CUH

#include "../../common/Types.h"

__global__ void rectangleKernel(FunctionType functionType, double delta, double a, int n, double* results);

#endif //RECTANGLEKERNEL_CUH
