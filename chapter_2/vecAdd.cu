#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

// Wrap CUDA runtime calls so failures are reported with file/line context.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// vecAddH: plain CPU add loop
// O(n), single-threaded host implementation.  
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

// vecAddD: full GPU pipeline for one run
// 1) cudaMalloc: allocate A_d, B_d, C_d on device
// 2) cudaMemcpy H2D: copy A/B from host RAM to device VRAM
// 3) launch vecAddKernel<<<blocks,threads>>>: GPU does C=A+B
// 4) cudaGetLastError: catch launch/config errors immediately
// 5) cudaMemcpy D2H: copy C back to host
// 6) cudaFree: release device memory

void vecAddD(const float *A, const float *B, float *C, int n) {
    int size = n * static_cast<int>(sizeof(float));
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;

    // Allocate device buffers.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C_d), size));

    // Transfer inputs from host RAM to GPU memory.
    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Launch asynchronous kernel work on the GPU.
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    // Catch launch configuration/runtime errors from the previous launch.
    CUDA_CHECK(cudaGetLastError());

    // Copy output back to host RAM.
    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    // Release device memory.
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

// timeVecAddH: run vecAddH many times and return average ms
// (reduces timer noise for tiny workloads)
double timeVecAddH(const float *A, const float *B, float *C, int n, int runs) {
    // Average over many runs to reduce timer noise for short CPU work.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        vecAddH(A, B, C, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double totalMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    return totalMs / static_cast<double>(runs);
}

// timeVecAddDEndToEnd: run vecAddD many times and return average ms
// includes allocation + transfers + kernel + free
double timeVecAddDEndToEnd(const float *A, const float *B, float *C, int n,
                           int runs) {
    // Includes allocation + copies + kernel + free in each run.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        vecAddD(A, B, C, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double totalMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    return totalMs / static_cast<double>(runs);
}


// timeVecAddDKernelOnly:
// - allocate/copy once
// - warm up one kernel launch
// - for each run: record start event, launch kernel, record stop event
// - synchronize stop event, get elapsed ms, accumulate
// - return average kernel-only time
double timeVecAddDKernelOnly(const float *A, const float *B, float *C, int n,
                             int runs) {
    int size = n * static_cast<int>(sizeof(float));
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    // One-time setup outside timed section.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B_d), size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C_d), size));
    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Warm up the kernel once before collecting timing.
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());
    // Ensure warm-up completes before entering measured loop.
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalKernelMs = 0.0f;
    for (int i = 0; i < runs; i++) {
        // CUDA events measure elapsed time on the GPU timeline.
        CUDA_CHECK(cudaEventRecord(startEvent));
        vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stopEvent));
        // Wait until stop event is reached so elapsed time is valid.
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float iterMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iterMs, startEvent, stopEvent));
        totalKernelMs += iterMs;
    }

    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return static_cast<double>(totalKernelMs) / static_cast<double>(runs);
}

// main:
// - create n=10000 input vectors
// - CUDA warm-up (cudaFree(0)) to avoid first-use init cost in benchmark
// - warm-up one vecAddD run
// - measure CPU avg, GPU end-to-end avg, GPU kernel-only avg
// - validate correctness via max |C_h - C_d|
// - print both speedups (end-to-end and kernel-only)
int main() {
    const int n = 10000;
    const int cpuRuns = 5000;
    const int gpuEndToEndRuns = 100;
    const int gpuKernelRuns = 1000;
    std::vector<float> A(n), B(n), C_h(n), C_d(n);

    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i) * 0.5f;
        B[i] = static_cast<float>(i) * 1.5f;
    }

    // Warm up CUDA context and a single end-to-end run before benchmarking.
    // cudaFree(0) is a common trick to force runtime/context initialization.
    CUDA_CHECK(cudaFree(0));
    vecAddD(A.data(), B.data(), C_d.data(), n);

    double cpuMs = timeVecAddH(A.data(), B.data(), C_h.data(), n, cpuRuns);
    double gpuEndToEndMs = timeVecAddDEndToEnd(A.data(), B.data(), C_d.data(),
                                               n, gpuEndToEndRuns);
    double gpuKernelMs =
        timeVecAddDKernelOnly(A.data(), B.data(), C_d.data(), n, gpuKernelRuns);

    float maxAbsErr = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = std::fabs(C_h[i] - C_d[i]);
        if (err > maxAbsErr) {
            maxAbsErr = err;
        }
    }

    std::printf("Vector size: %d\n", n);
    std::printf("CPU vecAddH avg time (%d runs): %.6f ms\n", cpuRuns, cpuMs);
    std::printf(
        "GPU vecAddD end-to-end avg time (%d runs, malloc+H2D+kernel+D2H+free): "
        "%.6f ms\n",
        gpuEndToEndRuns, gpuEndToEndMs);
    std::printf("GPU kernel-only avg time (%d runs): %.6f ms\n", gpuKernelRuns,
                gpuKernelMs);
    std::printf("Max absolute error: %.6f\n", maxAbsErr);
    if (gpuEndToEndMs > 0.0) {
        std::printf("Speedup (CPU/GPU end-to-end): %.4fx\n",
                    cpuMs / gpuEndToEndMs);
    }
    if (gpuKernelMs > 0.0) {
        std::printf("Speedup (CPU/GPU kernel-only): %.4fx\n",
                    cpuMs / gpuKernelMs);
    }

    return 0;
}
