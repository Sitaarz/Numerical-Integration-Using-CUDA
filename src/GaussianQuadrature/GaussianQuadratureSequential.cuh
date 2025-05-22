#ifndef GAUSSIAN_QUADRATURE_SEQUENTIAL_CUH
#define GAUSSIAN_QUADRATURE_SEQUENTIAL_CUH

#include "../AbstractIntegralCalculator.h"

class GaussianQuadratureSequential final: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n) override;
};

#endif // GAUSSIAN_QUADRATURE_SEQUENTIAL_CUH