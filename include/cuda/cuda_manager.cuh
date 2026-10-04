#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include <curand_kernel.h>

namespace potts {
namespace cuda {

__global__ void curand_setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long long seed, int L);

class CudaManager {
public:
    static void init_curand_states(curandStatePhilox4_32_10_t *d_states, int L, unsigned long long seed);
};

} // namespace cuda
} // namespace potts

#endif // CUDA_MANAGER_CUH
