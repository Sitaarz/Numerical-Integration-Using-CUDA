#include <iostream>
#include <memory>
#include "common/IntegrationMethodFactory.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "src/AbstractIntegralCalculator.h"

int main() {
    std::string method, functionName;
    double a = 0.0f;
    double b = 0.0f;
    int n = 0;

    std::cout << "Set integration method (Rectangle, Trapezoidal, MonteCarlo, GaussianQuadrature, Simpson): ";
    std::cin >> method;

    std::cout << "Set function (square, cubic, sinus, cosinus, exponential, hyperbolic, logarithm, squareRoot): ";
    std::cin >> functionName;

    std::cout << "Set lower bound (a): ";
    std::cin >> a;

    std::cout << "Set upper bound (b): ";
    std::cin >> b;

    std::cout << "Set number of intervals/random points/nodes/inputs (n): ";
    std::cin >> n;

    try {
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator(method);

        FunctionType functionType = getFunctionType(functionName);

        double result = calculator->calculate(functionType, a, b, n);


        std::cout << "Result: " << result << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
