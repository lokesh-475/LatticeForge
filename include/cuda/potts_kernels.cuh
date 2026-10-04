#ifndef POTTS_KERNELS_CUH
#define POTTS_KERNELS_CUH

#include <curand_kernel.h>
#include <cstdint>

namespace potts {
namespace cuda {

void launch_potts_checkerboard(
    uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    int q_states,
    double j_gb,
    bool is_red_phase
);

} // namespace cuda
} // namespace potts

#endif // POTTS_KERNELS_CUH
