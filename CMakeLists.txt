cmake_minimum_required(VERSION 3.31)
project(Numerical_Integration_Using_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Numerical_Integration_Using_CUDA main.cu
        common/FunctionStrategy.cu
        common/Types.h
        src/Constants.cuh
        src/AbstractIntegralCalculator.h
        common/IntegrationMethodFactory.cu
        common/IntegrationMethodFactory.h
        src/RectangleMethod/RectangleMethodCUDA.cu
        src/RectangleMethod/RectangleKernel.cu
        src/RectangleMethod/RectangleKernel.cuh
        src/TrapezoidMethod/TrapezoidalMethodCUDA.cu
        src/TrapezoidMethod/TrapezoidalMethodCUDA.cuh
        src/TrapezoidMethod/TrapezoidalKernel.cu
        src/TrapezoidMethod/TrapezoidalKernel.cuh
        src/MonteCarloMethod/MonteCarloMethodCUDA.cu
        src/MonteCarloMethod/MonteCarloMethodCUDA.cuh
        src/MonteCarloMethod/MonteCarloKernel.cu
        src/MonteCarloMethod/MonteCarloKernel.cuh
        src/MonteCarloMethod/MonteCarloMethodSequential.cu
        src/MonteCarloMethod/MonteCarloMethodSequential.cuh
        src/GaussianQuadrature/GaussianQuadratureCUDA.cu
        src/GaussianQuadrature/GaussianQuadratureCUDA.cuh
        src/GaussianQuadrature/GaussianQuadratureKernel.cu
        src/GaussianQuadrature/GaussianQuadratureKernel.cuh
        src/GaussianQuadrature/GaussianQuadratureSequential.cu
        src/GaussianQuadrature/GaussianQuadratureSequential.cuh
        common/Utils.h
)
set_target_properties(Numerical_Integration_Using_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
