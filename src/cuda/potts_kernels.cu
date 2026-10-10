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
    __shared__ uint16_t tile[18][18];

    // Load 18x18 tile into shared memory, including 1-cell halo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    for (int i = tid; i < 324; i += blockSize) {
        int r = i / 18 - 1;
        int c = i % 18 - 1;
        int gx = (blockIdx.x * blockDim.x + c) & mask;
        int gy = (blockIdx.y * blockDim.y + r) & mask;
        tile[r + 1][c + 1] = d_orientations[gy * L + gx];
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= L || y >= L) return;

    // Red-Black checkerboard: phase_parity is 0 for Red, 1 for Black
    if (((x + y) & 1) != phase_parity) return;

    int idx = y * L + x;
    curandStatePhilox4_32_10_t local_state = d_states[idx];

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    uint16_t curr = tile[ty][tx];
    
    // Generate a proposed state uniformly in [1, q_states]
    uint16_t prop = (curand(&local_state) % q_states) + 1;

    if (prop != curr) {
        double e_curr = 0.0;
        double e_prop = 0.0;

        uint16_t n_up = tile[ty - 1][tx];
        uint16_t n_down = tile[ty + 1][tx];
        uint16_t n_left = tile[ty][tx - 1];
        uint16_t n_right = tile[ty][tx + 1];

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
