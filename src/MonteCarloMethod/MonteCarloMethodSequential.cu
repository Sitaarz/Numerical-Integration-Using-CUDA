#include <stdexcept>
#include <cstdlib>
#include "MonteCarloMethodSequential.cuh"
#include "../../common/FunctionStrategy.cuh"

double MonteCarloMethodSequential::calculate(FunctionType functionType, double a, double b, int n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }

    srand(time(nullptr));
    DoubleFunctionPtr function = FunctionStrategy::getFunctionReference(functionType);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (*function)(a + (b - a) * (double(rand()) / RAND_MAX));
    }

    return (b - a) * sum / n;
}