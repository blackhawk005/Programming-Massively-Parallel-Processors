#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

void vecAddH(const float *A_h, const float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

__global__ void vecAddKernel(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAddD(const float *A, const float *B, float *C, int n) {
    int size = n * static_cast<int>(sizeof(float));
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C_d), size));

    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main() {
    const int n = 10000;
    std::vector<float> A(n), B(n), C_h(n), C_d(n);

    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i) * 0.5f;
        B[i] = static_cast<float>(i) * 1.5f;
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    vecAddH(A.data(), B.data(), C_h.data(), n);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    auto gpuStart = std::chrono::high_resolution_clock::now();
    vecAddD(A.data(), B.data(), C_d.data(), n);
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs =
        std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

    float maxAbsErr = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = std::fabs(C_h[i] - C_d[i]);
        if (err > maxAbsErr) {
            maxAbsErr = err;
        }
    }

    std::printf("Vector size: %d\n", n);
    std::printf("CPU vecAddH time: %.6f ms\n", cpuMs);
    std::printf("GPU vecAddD time (H2D + kernel + D2H): %.6f ms\n", gpuMs);
    std::printf("Max absolute error: %.6f\n", maxAbsErr);
    if (gpuMs > 0.0) {
        std::printf("Speedup (CPU/GPU): %.2fx\n", cpuMs / gpuMs);
    }

    return 0;
}
