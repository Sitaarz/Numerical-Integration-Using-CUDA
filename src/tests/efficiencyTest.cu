#include <iostream>
#include "common/IntegrationMethodFactory.h"
#include "src/AbstractIntegralCalculator.h"
#include "common/Types.h"

void testGaussianQuadrature() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("GaussianQuadrature");

    // tests
}

void testMonteCarloMethod() {
    std::unique_ptr<AbstractIntegralCalculator> parallelCalculator =
        IntegrationMethodFactory::createIntegralCalculator("MonteCarlo");

    std::unique_ptr<AbstractIntegralCalculator> sequentialCalculator =
        std::make_unique<MonteCarloMethodSequential>();

    std::cout << "Monte Carlo method" << std::endl;
    std::cout << "[PARALLEL] ";
    parallelCalculator->calculate(FunctionType::square, 0, 2, 100000, true);
    std::cout << "[SEQUENTIAL] ";
    sequentialCalculator->calculate(FunctionType::square, 0, 2, 100000, true);
}

void testTrapezoidalMethod() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("Trapezoidal");

    // tests
}

void testRectangleMethod() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("Rectangle");

    // tests
}

void testSimpsonMethod() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("Simpson");

    // tests
}

int main() {
    testGaussianQuadrature();
    testMonteCarloMethod();
    testTrapezoidalMethod();
    testRectangleMethod();
    // testSimpsonMethod();
    return 0;
}
