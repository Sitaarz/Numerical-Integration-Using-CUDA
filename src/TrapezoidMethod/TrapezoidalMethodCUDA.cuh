//
// Created by Krystian on 12.05.2025.
//

#ifndef TRAPEZOIDMETHODCUDA_CUH
#define TRAPEZOIDMETHODCUDA_CUH
#include "../AbstractIntegralCalculator.h"

class TrapezoidalMethodCUDA final: public AbstractIntegralCalculator {
    cudaEvent_t start, stop;
    float timeElapsed;
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
    TrapezoidalMethodCUDA() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
};

#endif //TRAPEZOIDMETHODCUDA_CUH