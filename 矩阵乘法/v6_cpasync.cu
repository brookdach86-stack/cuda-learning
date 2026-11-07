#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__device__ __forceinline__ void cp_async_ca(void *smem, const void *gmem)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"l"(smem), "l"(gmem));
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;");
}

__global__ void matmul_v6(float *C, float *A, float *B, int M, int K, int N)
{
    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BK + 1][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row = threadIdx.y * TM;
    int thread_col = threadIdx.x * TN;

    float Csub[TM][TN] = {{0.0f}};

    {
        int bk = 0;
        for (int r = threadIdx.y; r < BM; r += blockDim.y)
            for (int c = threadIdx.x; c < BK; c += blockDim.x)
            {
                int gr = block_row + r, gc = bk + c;
                As[0][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
        for (int r = threadIdx.y; r < BK; r += blockDim.y)
            for (int c = threadIdx.x; c < BN; c += blockDim.x)
            {
                int gr = bk + r, gc = block_col + c;
                Bs[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
    }
    __syncthreads();

    int comp = 0, read = 1;

    for (int bk = BK; bk < K; bk += BK)
    {
        for (int r = threadIdx.y; r < BM; r += blockDim.y)
            for (int c = threadIdx.x; c < BK; c += blockDim.x)
            {
                int gr = block_row + r, gc = bk + c;
                if (gr < M && gc < K)
                    cp_async_ca(&As[read][r][c], &A[gr * K + gc]);
                else
                    As[read][r][c] = 0.0f;
            }
        for (int r = threadIdx.y; r < BK; r += blockDim.y)
            for (int c = threadIdx.x; c < BN; c += blockDim.x)
            {
                int gr = bk + r, gc = block_col + c;
                if (gr < K && gc < N)
                    cp_async_ca(&Bs[read][r][c], &B[gr * N + gc]);
                else
                    Bs[read][r][c] = 0.0f;
            }

        for (int k = 0; k < BK; k++)
        {
            float Areg[TM];
            float Breg[TN];
            for (int ti = 0; ti < TM; ti++)
                Areg[ti] = As[comp][thread_row + ti][k];
            for (int tj = 0; tj < TN; tj++)
                Breg[tj] = Bs[comp][k][thread_col + tj];
            for (int ti = 0; ti < TM; ti++)
                for (int tj = 0; tj < TN; tj++)
                    Csub[ti][tj] += Areg[ti] * Breg[tj];
        }

        cp_async_wait_all();
        __syncthreads();

        comp ^= 1;
        read ^= 1;
    }

    for (int k = 0; k < BK; k++)
    {
        float Areg[TM];
        float Breg[TN];
        for (int ti = 0; ti < TM; ti++)
            Areg[ti] = As[comp][thread_row + ti][k];
        for (int tj = 0; tj < TN; tj++)
            Breg[tj] = Bs[comp][k][thread_col + tj];
        for (int ti = 0; ti < TM; ti++)
            for (int tj = 0; tj < TN; tj++)
                Csub[ti][tj] += Areg[ti] * Breg[tj];
    }

    int gr = block_row + thread_row;
    int gc = block_col + thread_col;
    for (int ti = 0; ti < TM; ti++)
        for (int tj = 0; tj < TN; tj++)
        {
            int r = gr + ti, c = gc + tj;
            if (r < M && c < N)
                C[r * N + c] = Csub[ti][tj];
        }
}

float *cpu_mmul(float *A, float *B, int M, int K, int N)
{
    float *C = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    return C;
}

int check(float *g, float *c, int n, float eps = 1e-3f)
{
    for (int i = 0; i < n; i++)
        if (fabsf(g[i] - c[i]) > eps)
        {
            printf("FAIL at %d: %f vs %f\n", i, g[i], c[i]);
            return 0;
        }
    return 1;
}

int main()
{
    int M = 8192, K = 8192, N = 8192;
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
        A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        B[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int w = 0; w < 10; w++)
        matmul_v6<<<grid, block>>>(dC, dA, dB, M, K, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("KERNEL ERR: %s\n", cudaGetErrorString(err));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int w = 0; w < 100; w++)
        matmul_v6<<<grid, block>>>(dC, dA, dB, M, K, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float t;
    cudaEventElapsedTime(&t, s, e);
    printf("GPU: %.3f ms (100 reps, avg)\n", t / 100.0);

    float *gpuC = (float *)malloc(M * N * sizeof(float));
    cudaMemcpy(gpuC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float *cpuC = cpu_mmul(A, B, M, K, N);
    printf(check(gpuC, cpuC, M * N) ? "PASS\n" : "FAIL\n");
    free(gpuC);
    free(cpuC);

    double flops = 2.0 * M * N * K;
    double tflops = flops / (t / 100.0) / 1e9;
    printf("TFLOPS: %.2f\n", tflops);

    free(A);
    free(B);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}