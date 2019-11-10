#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>


// Constant values.
int ARRAY_SIZE = 300'000'000;

// Check if the given pointer is NULL.
#define PTR_CHECK(cmd) if ((x) == NULL) { \
    printf("ERROR: null pointer at line %d\n", __LINE__); abort(); }


// Run SAXPY on CPU.
void SAXPY_cpu(float *x, float *y, float a)
{
    #pragma omp parallel for
    for (int index = 0; index < ARRAY_SIZE; index++)
        y[index] += a * x[index];
}


// Initialize input array.
void initArray(float *array)
{
    for (int index = 0; index < ARRAY_SIZE; index++)
        array[index] = (float)index / 1.42;
}


// Entry point of the program.
int main(int argc, const char **argv)
{
    // When the program is ran with -h, show usage.
    if (argc == 2 && !strcmp(argv[1], "-h")) {
        printf("Usage: ./exercise_2_openmp [array size]\n");
        exit(0);
    }

    // Read data from CLI.
    if (argc >= 2) ARRAY_SIZE = atoi(argv[1]);

    // Allocate data elements.
    float a = 12.f;

    float *x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    PTR_CHECK(x);

    float *y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    PTR_CHECK(y);

    // Run SAXPY on CPU.
    printf("Starting CPU test ...\n");
    initArray(x); initArray(y);

    auto start2 = std::chrono::system_clock::now();
    SAXPY_cpu(x, y, a);

    auto end2 = std::chrono::system_clock::now();
    int ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    printf("CPU time: %d ms\n", ms2);

    printf("Done\n", y[13]);
}
