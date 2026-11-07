/*
 * cuBLAS FP32 baseline —— 用于和手写 GEMM (v1~v4) 对比
 *
 * 编译: nvcc -O3 -lcublas cublas_baseline_fp32.cu -o cublas_baseline_fp32
 * 运行: ./cublas_baseline_fp32
 *
 * 注意: cuBLAS 默认列主序。行主序 C = A * B 等价于列主序 C^T = B^T * A^T，
 *       所以 cublasSgemm 的维度传 (N, M, K)，矩阵参数顺序是 B, A，ldc = N。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(expr)                                           \
    do                                                             \
    {                                                              \
        cudaError_t _st = (expr);                                  \
        if (_st != cudaSuccess)                                    \
        {                                                          \
            fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_st));                      \
            exit(1);                                               \
        }                                                          \
    } while (0)

#define CHECK_CUBLAS(expr)                                                 \
    do                                                                     \
    {                                                                      \
        cublasStatus_t _st = (expr);                                       \
        if (_st != CUBLAS_STATUS_SUCCESS)                                  \
        {                                                                  \
            fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, \
                    (int)_st);                                             \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

constexpr int WARMUP = 10;
constexpr int REPS = 100;

static const float kAlpha = 1.0f;
static const float kBeta = 0.0f;

void gemm_cublas(cublasHandle_t handle, const float *dA, const float *dB,
                 float *dC, int M, int K, int N)
{
    // 行主序 C(MxN) = A(MxK) * B(KxN)
    // 列主序等价: C^T(NxM) = B^T(NxK) * A^T(KxM)
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &kAlpha,
                             dB, N,
                             dA, K,
                             &kBeta,
                             dC, N));
}

float bench(cublasHandle_t handle, const float *dA, const float *dB,
            float *dC, int M, int K, int N)
{
    for (int i = 0; i < WARMUP; i++)
        gemm_cublas(handle, dA, dB, dC, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < REPS; i++)
        gemm_cublas(handle, dA, dB, dC, M, K, N);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / REPS;
}

void run_size(int M, int K, int N)
{
    float *hA = (float *)malloc(M * K * sizeof(float));
    float *hB = (float *)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
        hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    float ms = bench(handle, dA, dB, dC, M, K, N);
    double flops = 2.0 * M * N * K;
    double tflops = flops / ms / 1e9;
    printf("M=N=K=%-5d  %8.2f us  %6.1f TFLOPS\n", M, ms * 1000.0, tflops);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
}

int main()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("cuBLAS SGEMM (FP32), warmup=%d reps=%d\n\n", WARMUP, REPS);
    printf("%-14s %10s %12s\n", "size", "time", "TFLOPS");
    printf("----------------------------------------\n");

    srand(42);
    run_size(1024, 1024, 1024);
    run_size(4096, 4096, 4096);
    run_size(8192, 8192, 8192);

    return 0;
}