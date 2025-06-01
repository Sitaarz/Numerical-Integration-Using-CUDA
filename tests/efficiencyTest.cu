#include <iostream>

#include "../src/MonteCarloMethod/MonteCarloMethodSequential.cuh"
#include "../src/SimpsonIntegration/SimpsonIntegrationSequential.cuh"

#include "../common/IntegrationMethodFactory.h"
#include "../common/Types.h"
#include "../src/RectangleMethod/RectangleMethodSequential.cuh"
#include "../src/TrapezoidMethod/TrapezoidalMethodSequential.cuh"

namespace efficiencyTest {
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
        parallelCalculator->calculate(FunctionType::square, 0, 2, 100000000, true);
        std::cout << "[SEQUENTIAL] ";
        sequentialCalculator->calculate(FunctionType::square, 0, 2, 100000000, true);
    }

    void testTrapezoidalMethod() {
        std::unique_ptr<AbstractIntegralCalculator> parallelCalculator =
              IntegrationMethodFactory::createIntegralCalculator("trapezoidal");

        std::unique_ptr<AbstractIntegralCalculator> sequentialCalculator = std::make_unique<TrapezoidalMethodSequential>();

        std::cout << "Trapezoidal method" << std::endl;
        std::cout << "[PARALLEL] ";
        parallelCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
        std::cout << "[SEQUENTIAL] ";
        sequentialCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
    }

    void testRectangleMethod() {
        std::unique_ptr<AbstractIntegralCalculator> parallelCalculator =
              IntegrationMethodFactory::createIntegralCalculator("rectangle");

        std::unique_ptr<AbstractIntegralCalculator> sequentialCalculator = std::make_unique<RectangleMethodSequential>();

        std::cout << "Rectangle method" << std::endl;
        std::cout << "[PARALLEL] ";
        parallelCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
        std::cout << "[SEQUENTIAL] ";
        sequentialCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
    }

    void testSimpsonMethod() {
        std::unique_ptr<AbstractIntegralCalculator> parallelCalculator =
            IntegrationMethodFactory::createIntegralCalculator("Simpson");

        std::unique_ptr<AbstractIntegralCalculator> sequentialCalculator = std::make_unique<SimpsonIntegrationSequential>();

        std::cout << "Simpson method" << std::endl;
        std::cout << "[PARALLEL] ";
        parallelCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
        std::cout << "[SEQUENTIAL] ";
        sequentialCalculator->calculate(FunctionType::square, 0, 2, 100000000000, true);
    }
}

// int main() {
//     using namespace efficiencyTest;
//
//     testGaussianQuadrature();
//     testMonteCarloMethod();
//     testTrapezoidalMethod();
//     testRectangleMethod();
//     testSimpsonMethod();
//     return 0;
// }