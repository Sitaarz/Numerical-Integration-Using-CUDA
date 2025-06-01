//
// Created by HP on 1.06.2025.
//

#include "RectangleMethodSequential.cuh"

#include <chrono>
#include <iostream>

#include "../../common/FunctionStrategy.cuh"
#include <stdexcept>

double RectangleMethodSequential::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be greater than zero.");
    }
    auto start = std::chrono::high_resolution_clock::now();
    double delta = (b - a) / n;
    double integral = 0.0;

    DoubleFunctionPtr f = FunctionStrategy::getFunctionReference(functionType);

    for (int i = 0; i < n; ++i) {
        double x = a + i * delta;
        integral += f(x) * delta;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count()*1000 << " ms" << std::endl;

    return integral;
}
