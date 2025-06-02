# Numerical Integration Using CUDA

This project demonstrates how to perform **numerical integration** using CUDA for parallel computation. Several integration methods are implemented and accelerated using GPU to showcase performance gains on CUDA-enabled devices.

## Implemented Integration Methods

- **Rectangle Method**
- **Trapezoidal Method**
- **Simpson's Rule**
- **Monte Carlo Method**
- **Gaussian Quadrature**

## Requirements
To compile this CUDA project, you need:
1. A CUDA Toolkit version compatible with your GPU driver,
2. CMake version >= 3.18

## Compilation
To compile the project, follow these steps:
- Open a terminal and navigate to the project directory
- Run the following commands:
  ```bash
  mkdir build
  cd build
  cmake ..
  cmake --build .
  ```

Make sure that:
- The `nvcc` compiler is available in your system's `PATH` (e.g., `/usr/local/cuda/bin`)
- NVIDIA drivers and a compatible version of the CUDA Toolkit are installed
- You are using CMake version ≥ 3.18 (it supports CUDA as a language from that version onward)

## Execution
To run the program:
- Navigate to the `build` folder
- Run:
  ```bash
  ./Numerical_Integration_Using_CUDA
  ```
