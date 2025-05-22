#ifndef UNIT_TEST_CUH
#define UNIT_TEST_CUH

#include <iostream>
#include <string>

class UnitTest {
public:
    static void assertEqual(int actual, int expected, double delta, const std::string& testName) {
        if (actual <= expected + delta && actual >= expected - delta) {
            std::cout << "[PASS] " << testName << std::endl;
        } else {
            std::cout << "[FAIL] " << testName
                      << " | Expected: " << expected << ", Got: " << actual << std::endl;
        }
    }
};

#endif // UNIT_TEST_CUH
