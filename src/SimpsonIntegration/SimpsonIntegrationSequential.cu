//
// Created by HP on 1.06.2025.
//

#include "SimpsonIntegrationSequential.cuh"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include "../../common/FunctionStrategy.cuh"

double SimpsonIntegrationSequential::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) throw std::invalid_argument("n must be positive");
    if (b <= a) throw std::invalid_argument("b must be greater than a");
    if (n % 2 != 0) throw std::invalid_argument("n must be even for Simpson's rule");

    auto start = std::chrono::high_resolution_clock::now();

    DoubleFunctionPtr f = FunctionStrategy::getFunctionReference(functionType);
    double h = (b - a) / n;
    double sum_odd = 0.0, sum_even = 0.0;

    for (int i = 1; i < n; i += 2) sum_odd += (*f)(a + i * h);
    for (int i = 2; i < n; i += 2) sum_even += (*f)(a + i * h);

    double result = (h / 3.0) * ((*f)(a) + 4.0 * sum_odd + 2.0 * sum_even + (*f)(b));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (test) std::cout << "Time: " << duration.count() << " ms" << std::endl;

    return result;
}


