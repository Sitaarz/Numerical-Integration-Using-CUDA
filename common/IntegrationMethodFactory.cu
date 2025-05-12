//
// Created by Krystian on 10.05.2025.
//

#include <stdexcept>
#include <memory>
#include "IntegrationMethodFactory.h"
#include "Types.h"
#include "../src/RectangleMethod/RectangleMethodCUDA.h"
#include "../src/TrapezoidMethod/TrapezoidalMethodCUDA.cuh"

// IntegrationMethodFactory.cu
std::unique_ptr<AbstractIntegralCalculator> IntegrationMethodFactory::createIntegralCalculator(const std::string& input) {
    IntegrationMethod method = parseMethodFromInput(input);

    switch (method) {
        case IntegrationMethod::rectangle:
            return std::make_unique<RectangleMethodCUDA>();
        case IntegrationMethod::trapezoidal:
            return std::make_unique<TrapezoidalMethodCUDA>();
        default:
            throw std::invalid_argument("Invalid integral calculator method");
    }
}

IntegrationMethod IntegrationMethodFactory::parseMethodFromInput(const std::string& input) {
    if (input == "rectangle")
        return IntegrationMethod::rectangle;
    if (input == "trapezoidal")
        return IntegrationMethod::trapezoidal;

    throw std::invalid_argument("Unknown method: " + input);
}
