#include "unitTests.cuh"
#include "../common/IntegrationMethodFactory.h"
#include "../common/Types.h"
#include "../src/AbstractIntegralCalculator.h"
#include <numbers>

namespace unitTest {
    void testGaussianQuadrature() {
        std::cout << "Tensting Gaussian Quadrature Method" << std::endl;
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator("GaussianQuadrature");

        UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 1, 8), 0.333, 0.001, "Estimated integral of x^2 from 0 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, -1, 1, 8), 0, 0, "Estimated integral of x^3 from -1 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi, 8), 2, 0.001, "Estimated integral of sin(x) from 0 to pi");
        UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 8), 1.718, 0.001, "Estimated integral of e^x from 0 to 1");
    }

    void testMonteCarloMethod() {
        std::cout << "Testing Monte Carlo Method" << std::endl;
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator("MonteCarlo");

        UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 2, 1000000), 2.6667, 0.001, "Estimated integral of x^2 from 0 to 2");
        UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, 0, 1, 1000000), 0.25, 0.001, "Estimated integral of x^3 from 0 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi / 2.0, 1000000), 1, 0.001, "Estimated integral of sin(x) from 0 to pi/2");
        UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 1000000), 1.718, 0.002, "Estimated integral of e^x from 0 to 1");
    }

    void testTrapezoidalMethod() {
        std::cout << "Testing Trapezoidal Method" << std::endl;
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator("Trapezoidal");

        UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 1, 1000000000), 1.0/3.0, 0.01, "Trapezoidal integral of x^2 from 0 to 2");
        UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, 0, 1, 1000000000), 0.25, 0.01, "Trapezoidal integral of x^3 from 0 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi / 2.0, 1000000000), 1, 0.01, "Trapezoidal integral of sin(x) from 0 to pi/2");
        UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 1000000000), 1.718, 0.01, "Trapezoidal integral of e^x from 0 to 1");
    }

    void testRectangleMethod() {
        std::cout << "Testing Rectangle Method" << std::endl;

        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator("Rectangle");

        UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 1, 1000000000), 1.0/3.0, 0.001, "Rectangle: integral of x^2 from 0 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, -1, 1, 1000000000), 0.0, 0.001, "Rectangle: integral of x^3 from -1 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi, 1000000000), 2.0, 0.001, "Rectangle: integral of sin(x) from 0 to pi");
        UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 1000000000), 1.718, 0.001, "Rectangle: integral of e^x from 0 to 1");
    }

    void testSimpsonMethod() {
        std::cout << "Testing Simpson Method" << std::endl;
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator("Simpson");

        UnitTest::assertEqual(calculator->calculate(FunctionType::square, 0, 1, 1000000), 1.0/3.0, 0.0001, "Simpson: integral of x^2 from 0 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::cubic, -1, 1, 1000000), 0.0, 0.001, "Simpson: integral of x^3 from -1 to 1");
        UnitTest::assertEqual(calculator->calculate(FunctionType::sinus, 0, std::numbers::pi, 1000000), 2.0, 0.001, "Simpson: integral of sin(x) from 0 to pi");
        UnitTest::assertEqual(calculator->calculate(FunctionType::exponential, 0, 1, 1000000), 1.718, 0.001, "Simpson: integral of e^x from 0 to 1");
    }

}

    // int main() {
    //     using namespace unitTest;
    //
    //     testGaussianQuadrature();
    //     testMonteCarloMethod();
    //     testTrapezoidalMethod();
    //     testRectangleMethod();
    //     testSimpsonMethod();
    //     return 0;
    // }
