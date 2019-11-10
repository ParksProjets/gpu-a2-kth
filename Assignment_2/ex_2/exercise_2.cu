#include <stdio.h>
#include <stdint.h>
#include <chrono>


// Constant values.
#define TB_SIZE 256
uint64_t ARRAY_SIZE = 300'000'000;

// Check if the given pointer is NULL.
#define PTR_CHECK(cmd) if ((x) == NULL) { \
    printf("ERROR: null pointer at line %d\n", __LINE__); abort(); }

// Check if the given command has returned an error.
#define CUDA_CHECK(cmd) if ((cmd) != cudaSuccess) { \
    printf("ERROR: cuda error at line %d\n", __LINE__); abort(); }

// Kind of benchmark to do.
enum BenchKind { kBoth, kOnlyCPU, kOnlyGPU };


// Run SAXPY on GPU.
__global__ void SAXPY_gpu(float *x, float *y, float a, uint64_t ARRAY_SIZE)
{
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < ARRAY_SIZE)
        y[index] += a * x[index];
}


// Run SAXPY on CPU.
void SAXPY_cpu(float *x, float *y, float a)
{
    for (uint64_t index = 0; index < ARRAY_SIZE; index++)
        y[index] += a * x[index];
}


// Initialize input array.
void initArray(float *array)
{
    for (uint64_t index = 0; index < ARRAY_SIZE; index++)
        array[index] = (float)index / 1.42;
}


// Entry point of the program.
int main(int argc, const char **argv)
{
    // When the program is ran with -h, show usage.
    if (argc == 2 && !strcmp(argv[1], "-h")) {
        printf("Usage: ./exercise_2 [array size] [kind]\n");
        exit(0);
    }

    // Read data from CLI.
    if (argc >= 2) ARRAY_SIZE = atoll(argv[1]);
    printf("Using array size of %llu\n", ARRAY_SIZE);

    BenchKind kind = kBoth;
    if (argc >= 3 && !strcmp(argv[2], "cpu")) kind = kOnlyCPU;
    if (argc >= 3 && !strcmp(argv[2], "gpu")) kind = kOnlyGPU;

    // Allocate data elements.
    float a = 12.f;

    float *x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    PTR_CHECK(x);

    float *y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    PTR_CHECK(x);
    

    // Run SAXPY on GPU.
    if (kind != kOnlyCPU) {
        printf("Starting GPU test ...\n");
        initArray(x); initArray(y);

        float *gpux, *gpuy;
        CUDA_CHECK(cudaMalloc(&gpux, sizeof(float) * ARRAY_SIZE));
        CUDA_CHECK(cudaMalloc(&gpuy, sizeof(float) * ARRAY_SIZE));

        auto start = std::chrono::system_clock::now();

        CUDA_CHECK(cudaMemcpy(gpux, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuy, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice));

        SAXPY_gpu<<<((ARRAY_SIZE + TB_SIZE - 1) / TB_SIZE), TB_SIZE>>>(gpux, gpuy, a, ARRAY_SIZE);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(y, gpuy, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost));

        auto end = std::chrono::system_clock::now();
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("GPU time: %d ms\n", ms);
    }

    // Run SAXPY on CPU.
    if (kind != kOnlyGPU) {
        printf("Starting CPU test ...\n");
        initArray(x); initArray(y);
        
        auto start2 = std::chrono::system_clock::now();
        SAXPY_cpu(x, y, a);

        auto end2 = std::chrono::system_clock::now();
        int ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
        printf("CPU time: %d ms\n", ms2);
    }

    printf("Done\n", y[13]);
}
