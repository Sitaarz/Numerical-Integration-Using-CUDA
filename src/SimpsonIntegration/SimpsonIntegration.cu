// src/SimpsonIntegration/SimpsonIntegration.cu

#include "SimpsonIntegration.cuh"

#include <iostream>
#include <stdexcept>
#include "../../common/FunctionStrategy.cuh"
#include "../../common/Types.h"
#include "../Constants.cuh"
#include "../../common/Kernels.cuh"

__global__ void simpsonKernel(FunctionType functionType, double a, double delta, int n, double* values, double* oddBlockSums, double* evenBlockSums) {
    __shared__ double odd_sum[BLOCK_SIZE];
    __shared__ double even_sum[BLOCK_SIZE];

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    double val = 0.0;
    if (idx < n + 1) {
        DoubleFunctionPtr f = FunctionStrategy::getFunctionReference(functionType);
        double x = a + idx * delta;
        val = f(x);
        values[idx] = val;
    }

    // Sumowanie nieparzystych/parzystych indeksów w bloku
    odd_sum[tid] = (idx < n && idx % 2 == 1) ? val : 0.0;
    even_sum[tid] = (idx < n && idx % 2 == 0 && idx != 0 && idx != n) ? val : 0.0;
    __syncthreads();

    // Redukcja w bloku
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            odd_sum[tid] += odd_sum[tid + s];
            even_sum[tid] += even_sum[tid + s];
        }
        __syncthreads();
    }

    // Zapisz sumy blokowe do globalnej pamięci
    if (tid == 0) {
        oddBlockSums[blockIdx.x] = odd_sum[0];
        evenBlockSums[blockIdx.x] = even_sum[0];
    }
}



double SimpsonIntegration::calculate(FunctionType functionType, double a, double b, int n, bool test) {
    if (n <= 0) throw std::invalid_argument("n must be positive");
    if (b <= a) throw std::invalid_argument("b must be greater than a");
    if (n % 2 != 0) throw std::invalid_argument("n must be even for Simpson's rule");

    cudaEventRecord(start, 0);

    double delta = (b - a) / n;
    int points = n + 1;
    int blocks = (points + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double* d_values;
    double* d_oddBlockSums;
    double* d_evenBlockSums;
    cudaMalloc(&d_values, points * sizeof(double));
    cudaMalloc(&d_oddBlockSums, blocks * sizeof(double));
    cudaMalloc(&d_evenBlockSums, blocks * sizeof(double));


    simpsonKernel<<<blocks, BLOCK_SIZE>>>(functionType, a, delta, n, d_values, d_oddBlockSums, d_evenBlockSums);
    cudaDeviceSynchronize();

    // Redukcja rekurencyjna na GPU
    double* d_sum_odd;
    double* d_sum_even;
    cudaMalloc(&d_sum_odd, sizeof(double));
    cudaMalloc(&d_sum_even, sizeof(double));

    int reduce_blocks = blocks;
    double* d_oddReduce = d_oddBlockSums;
    double* d_evenReduce = d_evenBlockSums;

    while (reduce_blocks > 1) {
        int next_blocks = (reduce_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduceSumKernel<<<next_blocks, BLOCK_SIZE>>>(d_oddReduce, d_oddBlockSums, reduce_blocks);
        reduceSumKernel<<<next_blocks, BLOCK_SIZE>>>(d_evenReduce, d_evenBlockSums, reduce_blocks);
        cudaDeviceSynchronize();
        d_oddReduce = d_oddBlockSums;
        d_evenReduce = d_evenBlockSums;
        reduce_blocks = next_blocks;
    }

    reduceSumKernel<<<1, BLOCK_SIZE>>>(d_oddReduce, d_sum_odd, reduce_blocks);
    reduceSumKernel<<<1, BLOCK_SIZE>>>(d_evenReduce, d_sum_even, reduce_blocks);
    cudaDeviceSynchronize();

    double sum_odd = 0.0, sum_even = 0.0;
    cudaMemcpy(&sum_odd, d_sum_odd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_even, d_sum_even, sizeof(double), cudaMemcpyDeviceToHost);

    double* h_values = new double[points];
    cudaMemcpy(h_values, d_values, points * sizeof(double), cudaMemcpyDeviceToHost);

    double integral = (delta / 3.0) * (h_values[0] + 4.0 * sum_odd + 2.0 * sum_even + h_values[n]);

    delete[] h_values;
    cudaFree(d_values);
    cudaFree(d_oddBlockSums);
    cudaFree(d_evenBlockSums);
    cudaFree(d_sum_odd);
    cudaFree(d_sum_even);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    if (test) {
     std::cout << "Time " << timeElapsed << " ms" << std::endl;
    }

    return integral;
}