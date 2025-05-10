//
// Created by Krystian on 10.05.2025.
//

#ifndef FUNCTION_FACTORY_H
#define FUNCTION_FACTORY_H

#include "Types.h"

using DoubleFunctionPtr = double (*)(double);

__device__ double squareFunction(double x);
__device__ double cubicFunction(double x);

class FunctionFactory {
public:
    __device__ static DoubleFunctionPtr getFunctionReference(FunctionType functionType);
};

#endif // FUNCTION_FACTORY_H 