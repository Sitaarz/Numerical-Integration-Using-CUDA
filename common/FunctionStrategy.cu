//
// Created by Krystian on 10.05.2025.
//

#include "FunctionStrategy.cuh"

__host__ __device__ double squareFunction(double x) {
    return x*x;
}

__host__ __device__ double cubicFunction(double x) {
    return x*x*x;
}

__host__ __device__ double sinusFunction(double x) {
    return sin(x);
}

__host__ __device__ double cosinusFunction(double x) {
    return cos(x);
}

__host__ __device__ double exponentialFunction(double x) {
    return exp(x);
}

__host__ __device__ double hyperbolicFunction(double x) {
    return 1 / x;
}

__host__ __device__ double logarithmFunction(double x) {
    return log(x);
}

__host__ __device__ double squareRootFunction(double x) {
    return sqrt(x);
}

__host__ __device__ DoubleFunctionPtr FunctionStrategy::getFunctionReference(FunctionType functionType) {
    switch (functionType) {
        case FunctionType::square:
            return squareFunction;
        case FunctionType::cubic:
            return cubicFunction;
        case FunctionType::sinus:
            return sinusFunction;
        case FunctionType::cosinus:
            return cosinusFunction;
        case FunctionType::exponential:
            return exponentialFunction;
        case FunctionType::hyperbolic:
            return hyperbolicFunction;
        case FunctionType::logarithm:
            return logarithmFunction;
        case FunctionType::squareRoot:
            return squareRootFunction;
        default:
            return nullptr;
    }
}