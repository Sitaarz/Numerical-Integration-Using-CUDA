#include <stdexcept>
#include <cuda_runtime.h>
#include "GaussianQuadratureCUDA.cuh"
#include "GaussianQuadratureKernel.cuh"
#include "../Constants.cuh"

double GaussianQuadratureCUDA::calculate(FunctionType functionType, double a, double b, int n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("b must be greater than a");
    }
    if (n < MIN_NODES) {
        throw std::invalid_argument("n must be at least " + std::to_string(MIN_NODES));
    if (n > MAX_NODES) {
        throw std::invalid_argument("n must be at most " + std::to_string(MAX_NODES));

    double* d_results;
    cudaError_t error = cudaMalloc(&d_results, n * sizeof(double));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
    }

    double* d_nodes;
    error = cudaMalloc(&d_nodes, n * sizeof(double));
    if (error != cudaSuccess) {
        cudaFree(d_results);
        throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
    }

    double* d_weights;
    error = cudaMalloc(&d_weights, n * sizeof(double));
    if (error != cudaSuccess) {
        cudaFree(d_results);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaMemcpy(d_nodes, nodes_data[n], n * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_results);
        cudaFree(d_nodes);
        cudaFree(d_weights);
        throw std::runtime_error("Failed to copy nodes data to device: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaMemcpy(d_weights, weights_data[n], n * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_results);
        cudaFree(d_nodes);
        cudaFree(d_weights);
        throw std::runtime_error("Failed to copy weights data to device: " + std::string(cudaGetErrorString(error)));
    }

    int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gaussianQuadratureKernel<<<blocksPerGrid, BLOCK_SIZE>>>(functionType, a, b, n, d_results, d_nodes, d_weights);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_results);
        cudaFree(d_nodes);
        cudaFree(d_weights);
        throw std::runtime_error("Failed to launch kernel: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(d_results);
        cudaFree(d_nodes);
        cudaFree(d_weights);
        throw std::runtime_error("Failed to synchronize device: " + std::string(cudaGetErrorString(error)));
    }

    auto* h_results = new double[n];
    error = cudaMemcpy(h_results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        delete[] h_results;
        cudaFree(d_results);
        cudaFree(d_nodes);
        cudaFree(d_weights);
        throw std::runtime_error("Failed to copy results from device: " + std::string(cudaGetErrorString(error)));
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_results[i];
    }

    double h = 0.5f * (b - a);
    double integral = sum * h;

    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_nodes);
    cudaFree(d_weights);

    return integral;
}