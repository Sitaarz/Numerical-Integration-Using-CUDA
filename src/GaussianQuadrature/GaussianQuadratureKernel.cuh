#ifndef GAUSSIAN_QUADRATURE_CUH
#define GAUSSIAN_QUADRATURE_CUH

#include "../../common/Types.h"

__global__ void gaussianQuadratureKernel(FunctionType functionType, double a, double b, int n, double* results, double* nodes, double* weights);

#endif //GAUSSIAN_QUADRATURE_CUH
