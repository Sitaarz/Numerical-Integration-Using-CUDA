//
// Created by Krystian on 1.06.2025.
//

#ifndef SIMPSONINTEGRATIONSEQUENTIAL_H
#define SIMPSONINTEGRATIONSEQUENTIAL_H
#include"../../common/Types.h"
#include "../AbstractIntegralCalculator.h"

class SimpsonIntegrationSequential: public AbstractIntegralCalculator {
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test = false);
};

#endif //SIMPSONINTEGRATIONSEQUENTIAL_H
