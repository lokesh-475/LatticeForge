#include "cuda/cuda_manager.cuh"
#include <device_launch_parameters.h>
#include <iostream>

namespace potts {
namespace cuda {

__global__ void curand_setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long long seed, int L) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= L || y >= L) return;

    int idx = y * L + x;
    
    // Initialize Philox states.
    // Each thread gets the same seed, but a unique sequence number (idx),
    // and offset 0 to ensure independent streams of random numbers.
    curand_init(seed, idx, 0, &state[idx]);
}

void CudaManager::init_curand_states(curandStatePhilox4_32_10_t *d_states, int L, unsigned long long seed) {
    dim3 threads(16, 16);
    dim3 blocks((L + threads.x - 1) / threads.x, (L + threads.y - 1) / threads.y);

    curand_setup_kernel<<<blocks, threads>>>(d_states, seed, L);
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace potts
