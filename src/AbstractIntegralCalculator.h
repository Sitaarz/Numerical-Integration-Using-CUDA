//
// Created by Krystian on 10.05.2025.
//

#ifndef ABSTRACT_INTEGRAL_CALCULATOR_H
#define ABSTRACT_INTEGRAL_CALCULATOR_H

#include "../common/Types.h"

class AbstractIntegralCalculator {
public:
    virtual double calculate(FunctionType functionType, double a, double b, int n, bool test = false) = 0;
    virtual ~AbstractIntegralCalculator() = default;
};

#endif //ABSTRACT_INTEGRAL_CALCULATOR_H
