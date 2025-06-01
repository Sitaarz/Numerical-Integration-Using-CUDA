//
// Created by Krystian on 31.05.2025.
//

#ifndef SIMPSONINTEGRATION_H
#define SIMPSONINTEGRATION_H

#include "../AbstractIntegralCalculator.h"

class SimpsonIntegration final: public AbstractIntegralCalculator{
    double sumArrayDouble(const double* h_data, int n);
    cudaEvent_t start, stop;
    float timeElapsed;
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
    SimpsonIntegration() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
};



#endif //SIMPSONINTEGRATION_H
