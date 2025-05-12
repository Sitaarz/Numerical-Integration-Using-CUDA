//
// Created by Krystian on 10.05.2025.
//

#ifndef INTEGRATIONMETHODFACTORY_H
#define INTEGRATIONMETHODFACTORY_H
#include <memory>
#include <string>

#include "../src/AbstractIntegralCalculator.h"
#include "Types.h"


class IntegrationMethodFactory {
    static IntegrationMethod parseMethodFromInput(const std::string& input);
public:
    static std::unique_ptr<AbstractIntegralCalculator> createIntegralCalculator(const std::string& input);
};



#endif //INTEGRATIONMETHODFACTORY_H
