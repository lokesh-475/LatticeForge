#include "cuda/cuda_manager.cuh"
#include <device_launch_parameters.h>
#include <iostream>

namespace potts {
namespace cuda {

__global__ void curand_setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long long seed, int total_sites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_sites) return;

    // Initialize Philox states with contiguous 1D index mapping.
    // Consecutive threads access consecutive memory addresses, ensuring coalesced global memory stores.
    curand_init(seed, idx, 0, &state[idx]);
}

void CudaManager::init_curand_states(curandStatePhilox4_32_10_t *d_states, int L, unsigned long long seed) {
    int total_sites = L * L;
    int block_size = 256; // 8 warps for 128-byte warp coalescing
    int num_blocks = (total_sites + block_size - 1) / block_size;

    curand_setup_kernel<<<num_blocks, block_size>>>(d_states, seed, total_sites);
    cudaDeviceSynchronize();
}

void* CudaManager::allocate_pinned_host(size_t bytes, unsigned int flags) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, flags);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void CudaManager::free_pinned_host(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

} // namespace cuda
} // namespace potts
