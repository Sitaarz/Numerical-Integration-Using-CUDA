//
// Created by Krystian on 10.05.2025.
//

#include "IntegrationMethodFactory.h"
#include "Types.h"
#include <stdexcept>
#include <memory>
#include "../src/RectangleMethod/RectangleMethodCUDA.h"

// IntegrationMethodFactory.cu
std::unique_ptr<AbstractIntegralCalculator> IntegrationMethodFactory::createIntegralCalculator(const std::string& input) {
    IntegrationMethod method = parseMethodFromInput(input);

    switch (method) {
        case IntegrationMethod::rectangle:
            return std::make_unique<RectangleMethodCUDA>();
        default:
            throw std::invalid_argument("Invalid integral calculator method");
    }
}

IntegrationMethod IntegrationMethodFactory::parseMethodFromInput(const std::string& input) {
    if (input == "rectangle") return IntegrationMethod::rectangle;
    throw std::invalid_argument("Invalid method: " + input);
}
