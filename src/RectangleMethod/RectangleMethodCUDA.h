//
// Created by Krystian on 10.05.2025.
//

#ifndef RECTANGLE_METHOD_CUDA_H
#define RECTANGLE_METHOD_CUDA_H

#include "../AbstractIntegralCalculator.h"
#include "../../common/Types.h"

typedef double (*DoubleFunctionPtr)(double);

class RectangleMethodCUDA final : public AbstractIntegralCalculator {
    cudaEvent_t start, stop;
    float timeElapsed;
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
    RectangleMethodCUDA() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
};

#endif // RECTANGLE_METHOD_CUDA_H 