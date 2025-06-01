//
// Created by Krystian on 1.06.2025.
//

#include "TrapezoidalMethodSequential.cuh"

#include <chrono>
#include <iostream>

#include "../../common/FunctionStrategy.cuh"
#include "../../common/types.h"

double TrapezoidalMethodSequential::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    auto start = std::chrono::high_resolution_clock::now();

    double h = (b - a) / n;
    double sum = 0.0;

    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);

    for (int i = 0; i < n; ++i) {
        double x_i = a + i * h;
        double x_next = a + (i + 1) * h;
        double f_x_i = function(x_i);
        double f_x_next = function(x_next);
        sum += (f_x_i + f_x_next) * h / 2.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count()*1000 << " ms" << std::endl;

    return sum;
}