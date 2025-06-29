//
// Created by Krystian on 12.05.2025.
//

#include "TrapezoidalMethodCUDA.cuh"

#include <iostream>
#include <stdexcept>

#include "TrapezoidalKernel.cuh"
#include "../Constants.cuh"

double TrapezoidalMethodCUDA::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }

    double* d_results;
    double delta = (b - a) / n;

    cudaEventRecord(start, 0);


    cudaError_t error = cudaMalloc(&d_results, n * sizeof(double));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
    }

    int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    trapezoidKernel<<<blocksPerGrid, BLOCK_SIZE>>>(functionType, delta, a, n, d_results);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_results);
        throw std::runtime_error("Failed to launch kernel: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(d_results);
        throw std::runtime_error("Failed to synchronize device: " + std::string(cudaGetErrorString(error)));
    }

    auto* h_results = new double[n];
    error = cudaMemcpy(h_results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        delete[] h_results;
        cudaFree(d_results);
        throw std::runtime_error("Failed to copy results from device: " + std::string(cudaGetErrorString(error)));
    }

    double integral = 0.0;
    for (int i = 0; i < n; i++) {
        integral += h_results[i];
    }

    delete[] h_results;
    cudaFree(d_results);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    if (test) {
        std::cout << "Time " << timeElapsed << " ms" << std::endl;
    }

    return integral;
}