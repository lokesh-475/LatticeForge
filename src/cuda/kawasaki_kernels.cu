#include "cuda/kawasaki_kernels.cuh"
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

namespace potts {
namespace cuda {

__global__
void kawasaki_4color_kernel(
    uint8_t* d_elements,
    const uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    int shift,
    int mask,
    double h_seg,
    double j_chem,
    double temp,
    int active_color,
    int direction
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sites = L * L;
    if (idx1 >= total_sites) return;

    int y1 = (idx1 >> shift) & mask;
    int x1 = idx1 & mask;

    // 4-color partition: active_color in {0, 1, 2, 3}
    // Color 0: (even, even), Color 1: (even, odd), Color 2: (odd, even), Color 3: (odd, odd)
    int color = ((y1 & 1) << 1) | (x1 & 1);
    if (color != active_color) return;

    // Directional neighbor partner: 0: Up, 1: Down, 2: Left, 3: Right
    int y2 = y1;
    int x2 = x1;
    if (direction == 0) {
        y2 = (y1 - 1) & mask;
    } else if (direction == 1) {
        y2 = (y1 + 1) & mask;
    } else if (direction == 2) {
        x2 = (x1 - 1) & mask;
    } else if (direction == 3) {
        x2 = (x1 + 1) & mask;
    }
    int idx2 = (y2 << shift) | x2;

    uint8_t c1 = d_elements[idx1];
    uint8_t c2 = d_elements[idx2];

    // Swapping identical chemical species produces zero net change
    if (c1 == c2) return;

    // Coalesced Philox RNG state load
    curandStatePhilox4_32_10_t local_state = d_states[idx1];

    // 4 neighbors of site 1
    int up1    = (((y1 - 1) & mask) << shift) | x1;
    int down1  = (((y1 + 1) & mask) << shift) | x1;
    int left1  = (y1 << shift) | ((x1 - 1) & mask);
    int right1 = (y1 << shift) | ((x1 + 1) & mask);

    uint16_t g1 = d_orientations[idx1];
    int phi1 = (d_orientations[up1] != g1) +
               (d_orientations[down1] != g1) +
               (d_orientations[left1] != g1) +
               (d_orientations[right1] != g1);

    // 4 neighbors of site 2
    int up2    = (((y2 - 1) & mask) << shift) | x2;
    int down2  = (((y2 + 1) & mask) << shift) | x2;
    int left2  = (y2 << shift) | ((x2 - 1) & mask);
    int right2 = (y2 << shift) | ((x2 + 1) & mask);

    uint16_t g2 = d_orientations[idx2];
    int phi2 = (d_orientations[up2] != g2) +
               (d_orientations[down2] != g2) +
               (d_orientations[left2] != g2) +
               (d_orientations[right2] != g2);

    // Delta E segregation
    double delta_e_seg = 0.0;
    if (c1 == 1 && c2 == 0) {
        delta_e_seg = -h_seg * (phi2 - phi1);
    } else {
        delta_e_seg = -h_seg * (phi1 - phi2);
    }

    double delta_e = delta_e_seg;

    // Delta E chemical mixing
    if (j_chem != 0.0) {
        int nbrs1_c = d_elements[up1] + d_elements[down1] + d_elements[left1] + d_elements[right1];
        int nbrs2_c = d_elements[up2] + d_elements[down2] + d_elements[left2] + d_elements[right2];

        int n1_ones = nbrs1_c - c2;
        int n2_ones = nbrs2_c - c1;

        double delta_e_chem = 0.0;
        if (c1 == 1 && c2 == 0) {
            delta_e_chem = j_chem * 2.0 * (static_cast<double>(n1_ones) - static_cast<double>(n2_ones));
        } else {
            delta_e_chem = j_chem * 2.0 * (static_cast<double>(n2_ones) - static_cast<double>(n1_ones));
        }
        delta_e += delta_e_chem;
    }

    // Metropolis-Hastings acceptance
    float rand_val = curand_uniform(&local_state);
    if (delta_e <= 0.0 || (temp > 0.0 && rand_val < exp(-delta_e / temp))) {
        d_elements[idx1] = c2;
        d_elements[idx2] = c1;
    }

    d_states[idx1] = local_state;
}

void launch_kawasaki_4color(
    uint8_t* d_elements,
    const uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    double h_seg,
    double j_chem,
    double temp,
    int active_color,
    int direction
) {
    int total_sites = L * L;
    int block_size = 256;
    int num_blocks = (total_sites + block_size - 1) / block_size;
    int mask = L - 1;

    int shift = 0;
    while ((1 << shift) < L) {
        shift++;
    }

    kawasaki_4color_kernel<<<num_blocks, block_size>>>(
        d_elements,
        d_orientations,
        d_states,
        L,
        shift,
        mask,
        h_seg,
        j_chem,
        temp,
        active_color,
        direction
    );
}

void launch_kawasaki_sweep(
    uint8_t* d_elements,
    const uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    double h_seg,
    double j_chem,
    double temp
) {
    // 4 directions x 4 colors: directional, concurrent sub-lattice partitioning
    for (int dir = 0; dir < 4; ++dir) {
        for (int color = 0; color < 4; ++color) {
            launch_kawasaki_4color(
                d_elements,
                d_orientations,
                d_states,
                L,
                h_seg,
                j_chem,
                temp,
                color,
                dir
            );
        }
    }
}

__inline__ __device__ double warp_reduce_sum_double(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ unsigned long long warp_reduce_sum_ull(unsigned long long val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__
void hamiltonian_reduction_kernel(
    GpuMetricsRaw* d_metrics,
    const uint16_t* d_orientations,
    const uint8_t* d_elements,
    int L,
    int shift,
    int mask,
    double j_gb,
    double h_seg,
    double j_chem,
    int total_sites
) {
    double my_ham = 0.0;
    unsigned long long my_gb = 0;
    unsigned long long my_ce_gb = 0;
    unsigned long long my_ce_bulk = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_sites; idx += blockDim.x * gridDim.x) {
        int y = (idx >> shift) & mask;
        int x = idx & mask;

        int up    = (((y - 1) & mask) << shift) | x;
        int down  = (((y + 1) & mask) << shift) | x;
        int left  = (y << shift) | ((x - 1) & mask);
        int right = (y << shift) | ((x + 1) & mask);

        uint16_t g = d_orientations[idx];
        int phi = (d_orientations[up] != g) +
                  (d_orientations[down] != g) +
                  (d_orientations[left] != g) +
                  (d_orientations[right] != g);

        uint8_t c = d_elements[idx];
        int unlike = (d_elements[up] != c) +
                     (d_elements[down] != c) +
                     (d_elements[left] != c) +
                     (d_elements[right] != c);

        double h_i = 0.5 * j_gb * phi - h_seg * c * phi + 0.5 * j_chem * unlike;
        my_ham += h_i;

        int is_gb = (phi > 0);
        if (is_gb) my_gb++;
        if (c == 1) {
            if (is_gb) my_ce_gb++;
            else my_ce_bulk++;
        }
    }

    // Warp-level reduction
    my_ham = warp_reduce_sum_double(my_ham);
    my_gb = warp_reduce_sum_ull(my_gb);
    my_ce_gb = warp_reduce_sum_ull(my_ce_gb);
    my_ce_bulk = warp_reduce_sum_ull(my_ce_bulk);

    __shared__ double s_ham[8];
    __shared__ unsigned long long s_gb[8];
    __shared__ unsigned long long s_ce_gb[8];
    __shared__ unsigned long long s_ce_bulk[8];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_ham[warp_id] = my_ham;
        s_gb[warp_id] = my_gb;
        s_ce_gb[warp_id] = my_ce_gb;
        s_ce_bulk[warp_id] = my_ce_bulk;
    }

    __syncthreads();

    if (warp_id == 0) {
        double b_ham = (lane < 8) ? s_ham[lane] : 0.0;
        unsigned long long b_gb = (lane < 8) ? s_gb[lane] : 0;
        unsigned long long b_ce_gb = (lane < 8) ? s_ce_gb[lane] : 0;
        unsigned long long b_ce_bulk = (lane < 8) ? s_ce_bulk[lane] : 0;

        b_ham = warp_reduce_sum_double(b_ham);
        b_gb = warp_reduce_sum_ull(b_gb);
        b_ce_gb = warp_reduce_sum_ull(b_ce_gb);
        b_ce_bulk = warp_reduce_sum_ull(b_ce_bulk);

        if (lane == 0) {
            atomicAdd(&d_metrics->hamiltonian, b_ham);
            atomicAdd(&d_metrics->total_gb_sites, b_gb);
            atomicAdd(&d_metrics->ce_at_gb, b_ce_gb);
            atomicAdd(&d_metrics->ce_at_bulk, b_ce_bulk);
        }
    }
}

GpuMetrics compute_metrics_cuda(
    GpuMetricsRaw* d_metrics,
    const uint16_t* d_orientations,
    const uint8_t* d_elements,
    int L,
    double j_gb,
    double h_seg,
    double j_chem
) {
    int total_sites = L * L;
    int block_size = 256;
    int num_blocks = (total_sites + block_size - 1) / block_size;
    int mask = L - 1;

    int shift = 0;
    while ((1 << shift) < L) {
        shift++;
    }

    cudaMemset(d_metrics, 0, sizeof(GpuMetricsRaw));

    hamiltonian_reduction_kernel<<<num_blocks, block_size>>>(
        d_metrics,
        d_orientations,
        d_elements,
        L,
        shift,
        mask,
        j_gb,
        h_seg,
        j_chem,
        total_sites
    );

    GpuMetricsRaw h_raw;
    cudaMemcpy(&h_raw, d_metrics, sizeof(GpuMetricsRaw), cudaMemcpyDeviceToHost);

    GpuMetrics out;
    out.hamiltonian = h_raw.hamiltonian;
    out.total_gb_sites = h_raw.total_gb_sites;
    unsigned long long total_sites_ull = static_cast<unsigned long long>(total_sites);
    out.total_bulk_sites = total_sites_ull - h_raw.total_gb_sites;
    out.c_boundary = static_cast<double>(h_raw.ce_at_gb) / std::max(1ULL, h_raw.total_gb_sites);
    out.c_bulk = static_cast<double>(h_raw.ce_at_bulk) / std::max(1ULL, out.total_bulk_sites);
    out.total_ce = h_raw.ce_at_gb + h_raw.ce_at_bulk;

    return out;
}

double compute_hamiltonian_cuda(
    GpuMetricsRaw* d_metrics,
    const uint16_t* d_orientations,
    const uint8_t* d_elements,
    int L,
    double j_gb,
    double h_seg,
    double j_chem
) {
    GpuMetrics m = compute_metrics_cuda(d_metrics, d_orientations, d_elements, L, j_gb, h_seg, j_chem);
    return m.hamiltonian;
}

} // namespace cuda
} // namespace potts
