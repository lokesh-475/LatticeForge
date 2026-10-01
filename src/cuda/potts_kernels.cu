#include "cuda/potts_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace potts {
namespace cuda {

__global__
void potts_checkerboard_kernel(
    uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    int mask,
    int q_states,
    double j_gb,
    int phase_parity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= L || y >= L) return;

    // Red-Black checkerboard: phase_parity is 0 for Red, 1 for Black
    if (((x + y) & 1) != phase_parity) return;

    int idx = y * L + x;
    curandStatePhilox4_32_10_t local_state = d_states[idx];

    uint16_t curr = d_orientations[idx];
    
    // Generate a proposed state uniformly in [1, q_states]
    uint16_t prop = (curand(&local_state) % q_states) + 1;

    if (prop != curr) {
        // Branchless periodic boundary via bitwise mask
        int up = ((y - 1) & mask) * L + x;
        int down = ((y + 1) & mask) * L + x;
        int left = y * L + ((x - 1) & mask);
        int right = y * L + ((x + 1) & mask);

        double e_curr = 0.0;
        double e_prop = 0.0;

        uint16_t n_up = d_orientations[up];
        uint16_t n_down = d_orientations[down];
        uint16_t n_left = d_orientations[left];
        uint16_t n_right = d_orientations[right];

        if (n_up != curr) e_curr += j_gb;
        if (n_down != curr) e_curr += j_gb;
        if (n_left != curr) e_curr += j_gb;
        if (n_right != curr) e_curr += j_gb;

        if (n_up != prop) e_prop += j_gb;
        if (n_down != prop) e_prop += j_gb;
        if (n_left != prop) e_prop += j_gb;
        if (n_right != prop) e_prop += j_gb;

        if (e_prop - e_curr <= 0.0) {
            d_orientations[idx] = prop;
        }
    }

    d_states[idx] = local_state;
}

void launch_potts_checkerboard(
    uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    int q_states,
    double j_gb,
    bool is_red_phase
) {
    dim3 threads(16, 16);
    dim3 blocks((L + threads.x - 1) / threads.x, (L + threads.y - 1) / threads.y);
    int mask = L - 1;
    int phase_parity = is_red_phase ? 0 : 1;

    potts_checkerboard_kernel<<<blocks, threads>>>(
        d_orientations,
        d_states,
        L,
        mask,
        q_states,
        j_gb,
        phase_parity
    );
}

} // namespace cuda
} // namespace potts
