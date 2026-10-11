#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <cstddef>

namespace potts {
namespace cuda {

__global__ void curand_setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long long seed, int total_sites);

class CudaManager {
public:
    static void init_curand_states(curandStatePhilox4_32_10_t *d_states, int L, unsigned long long seed);

    // Pinned host memory buffers (cudaHostAlloc) maximizing PCIe bus throughput
    static void* allocate_pinned_host(size_t bytes, unsigned int flags = cudaHostAllocDefault);
    static void free_pinned_host(void* ptr);

    template <typename T>
    static T* allocate_pinned(size_t count, unsigned int flags = cudaHostAllocDefault) {
        return static_cast<T*>(allocate_pinned_host(count * sizeof(T), flags));
    }
};

} // namespace cuda
} // namespace potts

#endif // CUDA_MANAGER_CUH
