#include <stdexcept>
#include "GaussianQuadratureSequential.cuh"

#include <string>
#include <chrono>
#include <iostream>

#include "../../common/FunctionStrategy.cuh"
#include "../Constants.cuh"

double GaussianQuadratureSequential::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }
    if (n < MIN_NODES) {
        throw std::invalid_argument("n must be at least " + std::to_string(MIN_NODES));
    }
    if (n > MAX_NODES) {
        throw std::invalid_argument("n must be at most " + std::to_string(MAX_NODES));
    }

    auto start = std::chrono::high_resolution_clock::now();

    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);
    double sum = 0.0;
    double h = 0.5 * (b - a);
    double c = 0.5 * (a + b);
    const double* nodes = nodes_data[n];
    const double* weights = weights_data[n];
    for (int i = 0; i < n; i++) {
        double x_i = nodes[i];
        double w_i = weights[i];
        sum += w_i * (*function)(h * x_i + c);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (test) std::cout << "Time: " << duration.count() << " microSeconds" << std::endl;

    return h * sum;
}