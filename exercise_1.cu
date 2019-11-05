#include <stdio.h>
#include <cuda.h>


__global__ void kernel()
{
    printf("Hello World! My threadId is %d\n", blockDim.x * blockIdx.x + threadIdx.x);
}


int main()
{
    kernel<<<1, 256>>>();

    cudaDeviceSynchronize();
}
