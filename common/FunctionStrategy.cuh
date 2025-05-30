//
// Created by Krystian on 10.05.2025.
//

#ifndef FUNCTION_FACTORY_H
#define FUNCTION_FACTORY_H

#include "Types.h"

using DoubleFunctionPtr = double (*)(double);

__host__ __device__ double squareFunction(double x);
__host__ __device__ double cubicFunction(double x);
__host__ __device__ double sinusFunction(double x);
__host__ __device__ double cosinusFunction(double x);
__host__ __device__ double exponentialFunction(double x);
__host__ __device__ double hyperbolicFunction(double x);
__host__ __device__ double logarithmFunction(double x);
__host__ __device__ double squareRootFunction(double x);

class FunctionStrategy {
public:
    __host__ __device__ static DoubleFunctionPtr getFunctionReference(FunctionType functionType);
};

#endif // FUNCTION_FACTORY_H 