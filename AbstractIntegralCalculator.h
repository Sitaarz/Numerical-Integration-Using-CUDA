//
// Created by Krystian on 10.05.2025.
//

#ifndef ABSTRACTINTEGRALCALCULATOR_H
#define ABSTRACTINTEGRALCALCULATOR_H
#include <functional>

class AbstractIntegralCalculator {
public:
    virtual double calculate(std::function<double(double)> func, double a, double b, int n) = 0;
    virtual ~AbstractIntegralCalculator() = default;
};
#endif //ABSTRACTINTEGRALCALCULATOR_H
