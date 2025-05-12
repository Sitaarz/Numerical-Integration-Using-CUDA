//
// Created by Krystian on 10.05.2025.
//

#include <stdexcept>
#include <cuda_runtime.h>
#include "RectangleMethodCUDA.h"
#include "RectangleKernel.cuh"

double RectangleMethodCUDA::calculate(FunctionType functionType, double a, double b, int n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }

    double* d_results;
    double delta = (b - a) / n;

        cudaError_t error = cudaMalloc(&d_results, n * sizeof(double));
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        rectangleKernel<<<blocksPerGrid, threadsPerBlock>>>(functionType, delta, a, n, d_results);
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

        double integral = 0;
        for (int i = 0; i < n; i++) {
            integral += h_results[i];
        }

        delete[] h_results;
        cudaFree(d_results);

    return integral;
}