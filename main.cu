#include <iostream>
#include <memory>
#include "common/IntegrationMethodFactory.h"
#include "common/Types.h"
#include "src/AbstractIntegralCalculator.h"

int main() {
    std::string method;
    double a = 0.0f;
    double b = 0.0f;
    int n = 0;

    std::cout << "Set integration method (rectangle, trapezoidal, Monte Carlo): ";
    std::cin >> method;

    std::cout << "Set lower bound (a): ";
    std::cin >> a;

    std::cout << "Set upper bound (b): ";
    std::cin >> b;

    // TODO in case of Monte Carlo its more like a number of random points
    std::cout << "Set number of intervals (n): ";
    std::cin >> n;

    try {
        std::unique_ptr<AbstractIntegralCalculator> calculator =
            IntegrationMethodFactory::createIntegralCalculator(method);

        double result = calculator->calculate(FunctionType::square, a, b, n);


        std::cout << "Result: " << result << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
