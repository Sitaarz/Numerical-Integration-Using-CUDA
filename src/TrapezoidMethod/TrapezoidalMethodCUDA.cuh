//
// Created by Krystian on 12.05.2025.
//

#ifndef TRAPEZOIDMETHODCUDA_CUH
#define TRAPEZOIDMETHODCUDA_CUH
#include "../AbstractIntegralCalculator.h"

class TrapezoidalMethodCUDA final: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
};

#endif //TRAPEZOIDMETHODCUDA_CUH