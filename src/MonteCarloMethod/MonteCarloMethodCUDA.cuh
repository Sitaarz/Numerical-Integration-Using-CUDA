#ifndef MONTE_CARLO_METHOD_CUDA_CUH
#define MONTE_CARLO_METHOD_CUDA_CUH

#include "../AbstractIntegralCalculator.h"

class MonteCarloMethodCUDA final : public AbstractIntegralCalculator {
    double calculate(FunctionType functionType, double a, double b, int n) override;
};

#endif // MONTE_CARLO_METHOD_CUDA_CUH