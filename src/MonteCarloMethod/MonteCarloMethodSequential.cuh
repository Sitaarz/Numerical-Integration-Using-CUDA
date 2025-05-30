#ifndef MONTE_CARLO_METHOD_SEQUENTIAL_CUH
#define MONTE_CARLO_METHOD_SEQUENTIAL_CUH

#include "../AbstractIntegralCalculator.h"

class MonteCarloMethodSequential final: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
};

#endif // MONTE_CARLO_METHOD_SEQUENTIAL_CUH