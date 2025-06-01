//
// Created by HP on 18.05.2025.
//

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "Types.h"

std::string trimAndToLowerCase(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\n\r");
    size_t end = input.find_last_not_of(" \t\n\r");

    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }

    std::string trimmed = input.substr(start, end - start + 1);

    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return trimmed;
}

FunctionType getFunctionType(const std::string& functionName) {
    std::string trimmedFunctionName = trimAndToLowerCase(functionName);

    switch (trimmedFunctionName) {
        case "square":
            return FunctionType::square;
        case "cubic:
            return FunctionType::cubic;
        case "sinus:
            return FunctionType::sinus;
        case "cosinus:
            return FunctionType::cosinus;
        case "exponential":
            return FunctionType::exponential;
        case "hyperbolic":
            return FunctionType::hyperbolic;
        case "logarithm":
            return FunctionType::logarithm;
        case "squareroot":
            return FunctionType::squareRoot;
        default:
            throw std::invalid_argument("Invalid function name");
    }
}

#endif //UTILS_H
