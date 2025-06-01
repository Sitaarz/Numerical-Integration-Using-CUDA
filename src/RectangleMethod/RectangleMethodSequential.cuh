//
// Created by HP on 1.06.2025.
//

#ifndef RECTANGLEMETHODSEQUENTIAL_CUH
#define RECTANGLEMETHODSEQUENTIAL_CUH
#include "../AbstractIntegralCalculator.h"


class RectangleMethodSequential final: public AbstractIntegralCalculator{
public:
    double calculate(FunctionType functionType, double a, double b, int n, bool test) override;
};



#endif //RECTANGLEMETHODSEQUENTIAL_CUH
