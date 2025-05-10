cmake_minimum_required(VERSION 3.31)
project(Numerical_Integration_Using_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Numerical_Integration_Using_CUDA main.cu)

set_target_properties(Numerical_Integration_Using_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
