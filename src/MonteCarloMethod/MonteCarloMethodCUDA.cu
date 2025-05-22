#include <stdexcept>
#include <cuda_runtime.h>
#include "MonteCarloMethodCUDA.cuh"
#include "MonteCarloKernel.cuh"
#include "../Constants.cuh"

double MonteCarloMethodCUDA::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }

    double* d_results;
    cudaError_t error = cudaMalloc(&d_results, n * sizeof(double));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
    }

    int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    monteCarloKernel<<<blocksPerGrid, BLOCK_SIZE>>>(functionType, a, b, n, d_results);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms , start , stop);

    if (test) std::cout << "Time: " << ms << " ms" << std::endl;

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

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_results[i];
    }

    double integral = (b - a) * sum / n;

    delete[] h_results;
    cudaFree(d_results);

    return integral;
}