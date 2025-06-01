//
// Created by Krystian on 10.05.2025.
//

#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>

#include "RectangleMethodCUDA.h"
#include "RectangleKernel.cuh"
#include "../Constants.cuh"
#include "../../common/Kernels.cuh"

double RectangleMethodCUDA::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) throw std::invalid_argument("n must be positive");
    if (b <= a) throw std::invalid_argument("b must be greater than a");

    cudaEventRecord(start, 0);

    double* d_results;
    double delta = (b - a) / n;
    cudaError_t error = cudaMalloc(&d_results, n * sizeof(double));
    if (error != cudaSuccess) throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));

    int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rectangleKernel<<<blocksPerGrid, BLOCK_SIZE>>>(functionType, delta, a, n, d_results);
    error = cudaGetLastError();
    if (error != cudaSuccess) { cudaFree(d_results); throw std::runtime_error("Failed to launch kernel: " + std::string(cudaGetErrorString(error))); }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) { cudaFree(d_results); throw std::runtime_error("Failed to synchronize device: " + std::string(cudaGetErrorString(error))); }

    // Redukcja sumy na GPU
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    int reduce_n = n;
    double* d_in = d_results;
    double* d_out = nullptr;
    while (reduce_n > 1) {
        int reduce_blocks = (reduce_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        d_out = (reduce_blocks == 1) ? d_sum : d_results; // ostatnia iteracja do d_sum
        reduceSumKernel<<<reduce_blocks, BLOCK_SIZE>>>(d_in, d_out, reduce_n);
        cudaDeviceSynchronize();
        d_in = d_out;
        reduce_n = reduce_blocks;
    }

    double integral = 0.0;
    cudaMemcpy(&integral, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_sum);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    if (test) {
        std::cout << "Time " << timeElapsed << " ms" << std::endl;
    }

    return integral;
}