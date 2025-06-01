//
// Created by HP on 18.05.2025.
//

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <algorithm>
#include <stdexcept>
#include "Types.h"

inline std::string trimAndToLowerCase(const std::string& input) {
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

inline FunctionType getFunctionType(const std::string& functionName) {
    std::string trimmedFunctionName = trimAndToLowerCase(functionName);

    if (trimmedFunctionName == "square") {
        return FunctionType::square;
    } else if (trimmedFunctionName == "cubic") {
        return FunctionType::cubic;
    } else if (trimmedFunctionName == "sinus") {
        return FunctionType::sinus;
    } else if (trimmedFunctionName == "cosinus") {
        return FunctionType::cosinus;
    } else if (trimmedFunctionName == "exponential") {
        return FunctionType::exponential;
    } else if (trimmedFunctionName == "hyperbolic") {
        return FunctionType::hyperbolic;
    } else if (trimmedFunctionName == "logarithm") {
        return FunctionType::logarithm;
    } else if (trimmedFunctionName == "squareroot") {
        return FunctionType::squareRoot;
    } else {
        throw std::invalid_argument("Invalid function name");
    }
}


#endif //UTILS_H
