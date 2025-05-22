#include <iostream>
#include "unitTest.cuh"
#include "common/IntegrationMethodFactory.h"
#include "src/AbstractIntegralCalculator.h"
#include "common/Types.h"

void testGaussianQuadrature() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("GaussianQuadrature");

    UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 1, 2), 0.333, 0.001, "Estimated integral of x^2 from 0 to 1");
    UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, -1, 1, 2), 0, 0, "Estimated integral of x^3 from -1 to 1");
    UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi, 3), 2, 0.0001, "Estimated integral of sin(x) from 0 to pi");
    UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 3), 1.718, 0.001, "Estimated integral of e^x from 0 to 1");
}

void testMonteCarloMethod() {
    std::unique_ptr<AbstractIntegralCalculator> calculator =
        IntegrationMethodFactory::createIntegralCalculator("MonteCarlo");

    UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 2, 10000), 2.6667, 0.0001, "Estimated integral of x^2 from 0 to 2");
    UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, 0, 1, 10000), 0.25, 0.001, "Estimated integral of x^3 from 0 to 1");
    UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi / 2, 10000), 1, 0.001, "Estimated integral of sin(x) from 0 to pi/2");
    UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 10000), 1.718, 0.002, "Estimated integral of e^x from 0 to 1");
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
