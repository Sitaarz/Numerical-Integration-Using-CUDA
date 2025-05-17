#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

#include "../../common/Types.h"

__global__ void monteCarloKernel(FunctionType functionType, double a, double b, int n, double* results);

#endif //MONTE_CARLO_CUH
