//
// Created by Krystian on 10.05.2025.
//

#ifndef TYPES_H
#define TYPES_H

using DoubleFunctionPtr = double (*)(double);

enum class FunctionType {
    square,
    cubic,
    sinus,
    cosinus,
    exponential,
    hyperbolic,
    logarithm,
    squareRoot,
};

enum class IntegrationMethod {
    rectangle,
    trapezoidal,
    monteCarlo,
	gaussianQuadrature,
    simpson
};

#endif // TYPES_H