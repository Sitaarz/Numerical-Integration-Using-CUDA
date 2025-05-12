//
// Created by Krystian on 12.05.2025.
//

#ifndef TRAPEZOIDMETHODCUDA_CUH
#define TRAPEZOIDMETHODCUDA_CUH
#include "../AbstractIntegralCalculator.h"

class TrapezoidMethodCUDA: public AbstractIntegralCalculator {
    double calculate(FunctionType functionType, double a, double b, int n) override;
};

#endif //TRAPEZOIDMETHODCUDA_CUH