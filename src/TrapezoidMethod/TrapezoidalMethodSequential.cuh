//
// Created by HP on 1.06.2025.
//

#ifndef TRAPEZOIDALMETHODSEQUENTIAL_H
#define TRAPEZOIDALMETHODSEQUENTIAL_H
#include "../AbstractIntegralCalculator.h"


class TrapezoidalMethodSequential final: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
};


#endif //TRAPEZOIDALMETHODSEQUENTIAL_H
