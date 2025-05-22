#include <stdexcept>
#include <cstdlib>
#include "MonteCarloMethodSequential.cuh"
#include "../../common/FunctionStrategy.cuh"
#include <chrono>

double MonteCarloMethodSequential::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }

    auto start = std::chrono::high_resolution_clock::now();

    srand(time(nullptr));
    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (*function)(a + (b - a) * (double(rand()) / RAND_MAX));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (test) std::cout << "Time: " << duration.count() << " ms" << std::endl;

    return (b - a) * sum / n;
}