//
// Created by HP on 18.05.2025.
//

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <algorithm>
#include <cctype>

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
#endif //UTILS_H
