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


// Generate a random number between in range [a, b].
#define RAND_FLOAT(a,b) (a + (float)rand() / RAND_MAX * (b-a))

// Check if the given command has returned an error.
#define CUDA_CHECK(cmd) if ((cmd) != cudaSuccess) { \
    printf("ERROR: cuda error at line %d\n", __LINE__); abort(); }


// Initialize the array of particules.
Particle *CreateParticuleArray()
{
    srand(42);

    Particle *array;
    CUDA_CHECK(cudaMallocManaged(&array, sizeof(Particle) * NUM_PARTICLES));

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
__device__ void UpdateParticule(Particle &particule, const float3 &dvel)
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
    int num_blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        GpuUpdate<<<num_blocks, BLOCK_SIZE>>>(array, NUM_PARTICLES, dvel);
        cudaDeviceSynchronize();  // Make sure the particules were updated.
    }
}


// Entry point of this program.
int main(int argc, const char **argv)
{
    // When the program is ran with -h, show usage.
    if (argc == 2 && !strcmp(argv[1], "-h")) {
        printf("Usage: ./exercise_2a [num particules] [num iterations] [block size]\n");
        exit(0);
    }

    // Read number of particules, number of iterations and block size.
    if (argc >= 2) NUM_PARTICLES  = atoi(argv[1]);
    if (argc >= 3) NUM_ITERATIONS = atoi(argv[2]);
    if (argc >= 4) BLOCK_SIZE     = atoi(argv[3]);

    // Velocity increment on each step.
    float3 dvel = make_float3(-1.f, 3.45f, 7.3f);


    // Run iterations on GPU.
    Particle *array = CreateParticuleArray();

    printf("\nStarting GPU test ...\n");
    auto start = std::chrono::system_clock::now();

    GpuInterations(array, dvel);

    auto end = std::chrono::system_clock::now();
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("GPU time: %d ms\n\n", ms);
}
