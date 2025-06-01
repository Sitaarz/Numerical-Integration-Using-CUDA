//
// Created by Krystian on 10.05.2025.
//

#include <stdexcept>
#include <memory>
#include "IntegrationMethodFactory.h"
#include "Types.h"
#include "../src/RectangleMethod/RectangleMethodCUDA.h"
#include "../src/TrapezoidMethod/TrapezoidalMethodCUDA.cuh"
#include "../src/MonteCarloMethod/MonteCarloMethodCUDA.cuh"
#include "../src/GaussianQuadrature/GaussianQuadratureCUDA.cuh"
#include "../src/SimpsonIntegration/SimpsonIntegration.cuh"
#include "Utils.h"

// IntegrationMethodFactory.cu
std::unique_ptr<AbstractIntegralCalculator> IntegrationMethodFactory::createIntegralCalculator(const std::string& input) {
    IntegrationMethod method = parseMethodFromInput(input);

    switch (method) {
        case IntegrationMethod::rectangle:
            return std::make_unique<RectangleMethodCUDA>();
        case IntegrationMethod::trapezoidal:
            return std::make_unique<TrapezoidalMethodCUDA>();
        case IntegrationMethod::monteCarlo:
            return std::make_unique<MonteCarloMethodCUDA>();
        case IntegrationMethod::gaussianQuadrature:
            return std::make_unique<GaussianQuadratureCUDA>();
        case IntegrationMethod::simpson:
            return std::make_unique<SimpsonIntegration>();
        default:
            throw std::invalid_argument("Invalid integral calculator method");
    }
}

IntegrationMethod IntegrationMethodFactory::parseMethodFromInput(const std::string& input) {
    std::string trimmedInput = trimAndToLowerCase(input);

    if (trimmedInput == "rectangle")
        return IntegrationMethod::rectangle;
    if (trimmedInput == "trapezoidal")
        return IntegrationMethod::trapezoidal;
    if (trimmedInput == "montecarlo")
        return IntegrationMethod::monteCarlo;
    if (trimmedInput == "gaussianquadrature")
        return IntegrationMethod::gaussianQuadrature;
    if (trimmedInput == "simpson")
        return IntegrationMethod::simpson;

    throw std::invalid_argument("Unknown method: " + input);
}
