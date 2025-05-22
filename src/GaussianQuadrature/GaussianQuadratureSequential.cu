#include <stdexcept>
#include "GaussianQuadratureSequential.cuh"
#include "../../common/FunctionStrategy.cuh"
#include "../Constants.cuh"

double GaussianQuadratureSequential::calculate(FunctionType functionType, double a, double b, int n) {
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }
    if (n < MIN_NODES) {
        throw std::invalid_argument("n must be at least " + std::to_string(MIN_NODES));
    }
    if (n > MAX_NODES) {
        throw std::invalid_argument("n must be at most " + std::to_string(MAX_NODES));
    }

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

    return h * sum;
}