//
// Created by Krystian on 10.05.2025.
//

#include "FunctionStrategy.cuh"

__device__ double squareFunction(double x) {
    return x*x;
}

__device__ double cubicFunction(double x) {
    return x*x*x;
}

__device__ DoubleFunctionPtr FunctionStrategy::getFunctionReference(FunctionType functionType) {
    switch (functionType) {
        case FunctionType::square:
            return squareFunction;
        case FunctionType::cubic:
            return cubicFunction;
        default:
            return nullptr;
    }
}