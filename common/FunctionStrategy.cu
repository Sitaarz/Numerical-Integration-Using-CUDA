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

__device__ double sinus(double x) {
    return sin(x);
}

__device__ double cosinus(double x) {
    return cos(x);
}

__device__ double exponential(double x) {
    return exp(x);
}

__device__ double hyperbolic(double x) {
    return 1 / x;
}

__device__ double logarithm(double x) {
    return log(x);
}

__device__ double squareRoot(double x) {
    return sqrt(x);
}

__device__ DoubleFunctionPtr FunctionStrategy::getFunctionReference(FunctionType functionType) {
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