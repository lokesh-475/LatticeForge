#ifndef KAWASAKI_KERNELS_CUH
#define KAWASAKI_KERNELS_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace potts {
namespace cuda {

struct GpuMetricsRaw {
    double hamiltonian;
    unsigned long long total_gb_sites;
    unsigned long long ce_at_gb;
    unsigned long long ce_at_bulk;
};

struct GpuMetrics {
    double hamiltonian;
    double c_boundary;
    double c_bulk;
    unsigned long long total_ce;
    unsigned long long total_gb_sites;
    unsigned long long total_bulk_sites;
};

// 4-color sub-lattice partition kernel launcher for directional concurrent Kawasaki atomic swaps
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
);

// Full directional sweep across all 4 directions and all 4 colors
void launch_kawasaki_sweep(
    uint8_t* d_elements,
    const uint16_t* d_orientations,
    curandStatePhilox4_32_10_t* d_states,
    int L,
    double h_seg,
    double j_chem,
    double temp
);

// GPU-side reduction to calculate the total Hamiltonian H and segregation metrics
GpuMetrics compute_metrics_cuda(
    GpuMetricsRaw* d_metrics,
    const uint16_t* d_orientations,
    const uint8_t* d_elements,
    int L,
    double j_gb,
    double h_seg,
    double j_chem
);

double compute_hamiltonian_cuda(
    GpuMetricsRaw* d_metrics,
    const uint16_t* d_orientations,
    const uint8_t* d_elements,
    int L,
    double j_gb,
    double h_seg,
    double j_chem
);

} // namespace cuda
} // namespace potts

#endif // KAWASAKI_KERNELS_CUH
