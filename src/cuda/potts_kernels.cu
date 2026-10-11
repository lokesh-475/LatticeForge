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
    int shift,
    int mask,
    int q_states,
    double j_gb,
    int phase_parity
) {
    // Contiguous 1D index mapping: consecutive threads in a warp access consecutive addresses
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sites = L * L;
    if (idx >= total_sites) return;

    // Bitwise mapping from contiguous 1D index to (x, y) coordinates
    int y = (idx >> shift) & mask;
    int x = idx & mask;

    // Red-Black checkerboard: phase_parity is 0 for Red, 1 for Black
    if (((x + y) & 1) != phase_parity) return;

    // Coalesced 128-byte aligned global memory access for Philox RNG state
    curandStatePhilox4_32_10_t local_state = d_states[idx];

    // Coalesced global memory load for current site orientation
    uint16_t curr = d_orientations[idx];
    
    // Generate a proposed state uniformly in [1, q_states]
    uint16_t prop = (curand(&local_state) % q_states) + 1;

    if (prop != curr) {
        // Continuous 1D flattened index mapping with bitwise periodic boundary conditions
        int up = (((y - 1) & mask) << shift) | x;
        int down = (((y + 1) & mask) << shift) | x;
        int left = (y << shift) | ((x - 1) & mask);
        int right = (y << shift) | ((x + 1) & mask);

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
    int total_sites = L * L;
    // 256 threads per block (8 full warps) aligning global memory transactions for 128-byte warp coalescing
    int block_size = 256;
    int num_blocks = (total_sites + block_size - 1) / block_size;
    int mask = L - 1;
    int phase_parity = is_red_phase ? 0 : 1;

    int shift = 0;
    while ((1 << shift) < L) {
        shift++;
    }

    potts_checkerboard_kernel<<<num_blocks, block_size>>>(
        d_orientations,
        d_states,
        L,
        shift,
        mask,
        q_states,
        j_gb,
        phase_parity
    );
}

} // namespace cuda
} // namespace potts
