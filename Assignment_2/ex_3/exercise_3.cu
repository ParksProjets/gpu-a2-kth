#include <stdio.h>
#include <chrono>


// Constant values.
int NUM_PARTICLES  = 10000;
int NUM_ITERATIONS = 1000;
int BLOCK_SIZE     = 32;


// Data of a single particule.
struct Particle {
    float3 position;
    float3 velocity;
};

// Kind of benchmark to do.
enum BenchKind { kBoth, kOnlyCPU, kOnlyGPU };


// Generate a random number between in range [a, b].
#define RAND_FLOAT(a,b) (a + (float)rand() / RAND_MAX * (b-a))

// Check if the given command has returned an error.
#define CUDA_CHECK(cmd) if ((cmd) != cudaSuccess) { \
    printf("ERROR: cuda error at line %d\n", __LINE__); abort(); }


// Initialize the array of particules.
Particle *CreateParticuleArray()
{
    srand(42);
    Particle *array = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);

    for (int index = 0; index < NUM_PARTICLES; index++) {
        array[index].velocity.x = RAND_FLOAT(1, 10);
        array[index].velocity.y = RAND_FLOAT(1, 20);
        array[index].velocity.z = RAND_FLOAT(5, 30);

        array[index].position.x = RAND_FLOAT(-100, 100);
        array[index].position.y = RAND_FLOAT(-100, 100);
        array[index].position.z = RAND_FLOAT(-100, 100);
    }

    return array;
}


// Update a particule by one single step.
__host__ __device__ void UpdateParticule(Particle &particule, const float3 &dvel)
{
    particule.velocity.x += dvel.x;
    particule.velocity.y += dvel.y;
    particule.velocity.z += dvel.z;

    particule.position.x += particule.velocity.x;
    particule.position.y += particule.velocity.y;
    particule.position.z += particule.velocity.z;
}


// GPU kernel for updating a particule by one single step.
__global__ void GpuUpdate(Particle *particules, int NUM_PARTICLES, float3 dvel)
{
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < NUM_PARTICLES)
        UpdateParticule(particules[index], dvel);
}


// Make all iterations on GPU.
void GpuInterations(Particle *array, float3 dvel)
{
    printf("\nStarting GPU test ...\n");
    auto start = std::chrono::system_clock::now();

    int num_blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Particle *gpuarray;
    CUDA_CHECK(cudaMalloc(&gpuarray, sizeof(Particle) * NUM_PARTICLES));
    CUDA_CHECK(cudaMemcpy(gpuarray, array, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyHostToDevice));

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        GpuUpdate<<<num_blocks, BLOCK_SIZE>>>(gpuarray, NUM_PARTICLES, dvel);
        cudaDeviceSynchronize();  // Make sure the particules were updated.
    }

    CUDA_CHECK(cudaMemcpy(array, gpuarray,  sizeof(Particle) * NUM_PARTICLES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gpuarray));

    auto end = std::chrono::system_clock::now();
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("GPU time: %d ms\n\n", ms);
}


// Make all iterations on CPU.
void CpuInterations(Particle *particules, float3 dvel)
{
    printf("\nStarting CPU test ...\n");
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int index = 0; index < NUM_PARTICLES; index++)
            UpdateParticule(particules[index], dvel);
    }

    auto end = std::chrono::system_clock::now();
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CPU time: %d ms\n", ms);
}


// Entry point of this program.
int main(int argc, const char **argv)
{
    // When the program is ran with -h, show usage.
    if (argc == 2 && !strcmp(argv[1], "-h")) {
        printf("Usage: ./exercise_3 [num particules] [num iterations] [block size] [kind]\n");
        exit(0);
    }

    // Read number of particules, number of iterations and block size.
    if (argc >= 2) NUM_PARTICLES  = atoi(argv[1]);
    if (argc >= 3) NUM_ITERATIONS = atoi(argv[2]);
    if (argc >= 4) BLOCK_SIZE     = atoi(argv[3]);

    BenchKind kind = kBoth;
    if (argc >= 5 && !strcmp(argv[4], "cpu")) kind = kOnlyCPU;
    if (argc >= 5 && !strcmp(argv[4], "gpu")) kind = kOnlyGPU;


    // Velocity increment on each step.
    float3 dvel = make_float3(-1.f, 3.45f, 7.3f);

    // Run iterations on CPU.
    Particle *cpuarray = CreateParticuleArray();
    if (kind != kOnlyGPU)
        CpuInterations(cpuarray, dvel);

    // Run iterations on GPU.
    Particle *gpuarray = CreateParticuleArray();
    if (kind != kOnlyCPU)
        GpuInterations(gpuarray, dvel);


    // Make sure that the two particule arrays are the same.
    if (kind == kBoth && memcmp(cpuarray, gpuarray, sizeof(Particle) * NUM_PARTICLES)) {
        printf("ERROR: particule arrays are differents!\n\n");
        exit(1);
    }
}
