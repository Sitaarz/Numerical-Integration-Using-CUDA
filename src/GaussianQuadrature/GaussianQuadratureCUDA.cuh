#ifndef GAUSSIAN_QUADRATURE_CUDA_CUH
#define GAUSSIAN_QUADRATURE_CUDA_CUH

#include "../AbstractIntegralCalculator.h"

class GaussianQuadratureCUDA final: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n) override;
};

#endif // GAUSSIAN_QUADRATURE_CUDA_CUH