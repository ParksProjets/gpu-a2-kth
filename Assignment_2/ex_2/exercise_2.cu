#include <cstddef>
#include <cuda.h>
#include <omp.h>

#include <iostream>
#include <chrono>


#define ARRAY_SIZE 300'000'000
#define TB_SIZE 256

#define cudaCheck(err) if (err != cudaSuccess) { \
    printf("ERROR: %d\n", __LINE__); abort(); }



__global__ void SAXPY_gpu(float *x, float *y, float a)
{
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < ARRAY_SIZE)
        y[index] += a * x[index];    
}


void SAXPY_cpu(float *x, float *y, float a)
{
    for (int index = 0; index < ARRAY_SIZE; index++)
        y[index] += a * x[index];
}


void initArray(float *array)
{
    for (int index = 0; index < ARRAY_SIZE; index++)
        array[index] += array[index];
}


int main()
{
    float a = 12.f;

    float *x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    initArray(x);

    float *y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    initArray(y);


    // Run SAXPY on GPU.
    printf("Starting GPU test ...\n");
    auto start = std::chrono::system_clock::now();

    float *gpux, *gpuy;
    cudaCheck(cudaMalloc(&gpux, sizeof(float) * ARRAY_SIZE));
    cudaCheck(cudaMalloc(&gpuy, sizeof(float) * ARRAY_SIZE));

    cudaCheck(cudaMemcpy(gpux, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpuy, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice));

    SAXPY_gpu<<<((ARRAY_SIZE + TB_SIZE - 1) / TB_SIZE), TB_SIZE>>>(gpux, gpuy, a);
    // cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(y, gpuy, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost));

    auto end = std::chrono::system_clock::now();
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("GPU time: %d ms\n", ms);


    // Run SAXPY on CPU.
    printf("Starting CPU test ...\n");
    initArray(y);
    auto start2 = std::chrono::system_clock::now();

    SAXPY_cpu(x, y, a);

    auto end2 = std::chrono::system_clock::now();
    int ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    printf("CPU time: %d ms\n", ms2);
}
