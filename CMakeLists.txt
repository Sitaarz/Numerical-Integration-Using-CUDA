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
        tests/unitTests.cu
        tests/unitTests.cuh
        tests/efficiencyTest.cu
        common/Utils.h
        src/SimpsonIntegration/SimpsonIntegration.cu
        src/SimpsonIntegration/SimpsonIntegration.cuh
        src/SimpsonIntegration/SimpsonIntegrationSequential.cu
        src/SimpsonIntegration/SimpsonIntegrationSequential.cuh
        src/RectangleMethod/RectangleMethodSequential.cu
        src/RectangleMethod/RectangleMethodSequential.cuh
        src/TrapezoidMethod/TrapezoidalMethodSequential.cu
        src/TrapezoidMethod/TrapezoidalMethodSequential.cuh
        common/Kernels.cuh
        common/Kernels.cu
)
set_target_properties(Numerical_Integration_Using_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
