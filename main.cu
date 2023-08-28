#include "utils.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#define M 4096
#define N 4096
#define K 4096

__host__ void init_matrix(float* m, size_t n) {
    for (size_t i = 0; i < n; i ++) {
        m[i] = float(rand() % 110) / 55 - 1.0f;
    }
}

__host__ int test_validation(float* rhc, float* thc, size_t n) {
    for (size_t i = 0; i < n; i ++) {
        if (std::abs(rhc[i] - thc[i]) > 0.0001) {
            std::cerr << "rhc[" << i << "]: " << rhc[i] << " vs. thc[" << i << "]: " << thc[i] << std::endl;
            return 1;
        }
    }
    return 0;
}

int capture_cuda_error() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    return 0;
}

#define VALUE_CHECK(RHC, THC, N, NAME)                                  \
    if (test_validation(RHC, THC, N) != 0) {                            \
        std::cerr << #NAME << ": validation test failed." << std::endl; \
        return 1;                                                       \
    }                                                                   \
    std::cout << #NAME << ": validation test success." << std::endl;

#define CUDA_CHECK_STATUS()             \
    if (capture_cuda_error() != 0) {    \
        return 1;                       \
    }

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define TEST_SGEMM_MODULE(module, gdim, bdim)                                                       \
    {                                                                                               \
        std::cout << #module << ": " << std::endl;                                                  \
        float* tmpc;                                                                                \
        cudaMalloc((void**)&tmpc, M * N * sizeof(float));                                           \
        {                                                                                           \
            TIME_GUARD(TEST_SGEMM_MODULE);                                                          \
            module<<<gdim, bdim>>>(M, N, K, alpha, a, b, beta, tmpc);                               \
            CUDA_CHECK_STATUS();                                                                    \
            cudaDeviceSynchronize();                                                                \
        }                                                                                           \
        CUDA_CHECK_STATUS();                                                                        \
        cudaMemcpy(thc, tmpc, M * N * sizeof(float), cudaMemcpyDeviceToHost);                       \
        VALUE_CHECK(rhc, thc, (M * N), module);                                                     \
        cudaFree(tmpc);                                                                             \
    }

int main() {
    srand(time(NULL));
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *a, *b, *c;
    float *ha = (float*)malloc(M * K * sizeof(float));
    float *hb = (float*)malloc(K * N * sizeof(float));
    float *rhc = (float*)malloc(M * N * sizeof(float));    // results of cublasSgemm
    float *thc = (float*)malloc(M * N * sizeof(float));    // results of my kernels
    
    cudaMalloc((void**)&a, M * K * sizeof(float));
    cudaMalloc((void**)&b, K * N * sizeof(float));
    cudaMalloc((void**)&c, M * N * sizeof(float));

    {
        TIME_GUARD(INIT);
        init_matrix(ha, M * K);
        init_matrix(hb, K * N);
        cudaMemcpy(a, ha, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b, hb, K * N * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK_STATUS();
    }

    cudaDeviceSynchronize();
    // warmup the machine
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, c, N);
    cudaDeviceSynchronize();
    {
        TIME_GUARD(CUBLAS);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, c, N);
        cudaDeviceSynchronize();
        CUDA_CHECK_STATUS();
    }
    cudaMemcpy(rhc, c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(ha);
    free(hb);
    free(rhc);
    free(thc);

    return 0;
}
