#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include "cuda/potts_kernels.cuh"
#include "cuda/kawasaki_kernels.cuh"
#include "cuda/cuda_manager.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

double compute_energy(const potts::Grid& grid, double j_gb) {
    double energy = 0.0;
    for (int y = 0; y < grid.size(); ++y) {
        for (int x = 0; x < grid.size(); ++x) {
            uint16_t curr = grid.get_orientation(y, x);
            if (grid.get_orientation(grid.up(y), x) != curr) energy += j_gb;
            if (grid.get_orientation(grid.down(y), x) != curr) energy += j_gb;
            if (grid.get_orientation(y, grid.left(x)) != curr) energy += j_gb;
            if (grid.get_orientation(y, grid.right(x)) != curr) energy += j_gb;
        }
    }
    return energy / 2.0;
}

int main() {
    potts::Config config;
    config.size = 128; // must be power of 2
    config.q_states = 32;
    config.j_gb = 1.0;
    config.seed = 42;

    int L = config.size;
    int num_elements = L * L;

    potts::Grid grid_cpu(L);
    potts::Grid grid_gpu(L, true); // Pinned host memory buffer (cudaHostAlloc)

    // Verify pinned host memory allocation
    if (!grid_gpu.is_pinned()) {
        std::cerr << "FAIL: grid_gpu is not marked as pinned.\n";
        return 1;
    }
    cudaPointerAttributes attr;
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, grid_gpu.get_orientations_ptr());
    if (attr_err != cudaSuccess || attr.type != cudaMemoryTypeHost) {
        std::cerr << "FAIL: grid_gpu buffer is not pinned host memory (type: " << attr.type << ").\n";
        return 1;
    }
    std::cout << "Verified pinned host memory buffer (cudaHostAlloc) for grid_gpu.\n";

    // Initialize with random states
    std::mt19937 rng(config.seed);
    std::uniform_int_distribution<uint16_t> dist(1, config.q_states);
    for (int i = 0; i < num_elements; ++i) {
        uint16_t state = dist(rng);
        grid_cpu.set_orientation(i, state);
        grid_gpu.set_orientation(i, state);
    }

    double initial_energy = compute_energy(grid_cpu, config.j_gb);
    std::cout << "Initial Energy: " << initial_energy << "\n";

    // Run CPU simulation
    int mcs = 50;
    for (int i = 0; i < mcs; ++i) {
        potts::potts_mc_step_cpu(grid_cpu, config, rng, num_elements);
    }

    double cpu_energy = compute_energy(grid_cpu, config.j_gb);
    std::cout << "CPU Energy after " << mcs << " MCS: " << cpu_energy << "\n";

    // Run GPU simulation
    uint16_t* d_orientations = nullptr;
    curandStatePhilox4_32_10_t* d_states = nullptr;

    cudaMalloc(&d_orientations, num_elements * sizeof(uint16_t));
    cudaMalloc(&d_states, num_elements * sizeof(curandStatePhilox4_32_10_t));

    // Profile pinned Host-to-Device transfer
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    cudaMemcpy(d_orientations, grid_gpu.get_orientations_ptr(), num_elements * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start_event, stop_event);
    double transfer_gb = (num_elements * sizeof(uint16_t)) / (1024.0 * 1024.0 * 1024.0);
    double h2d_gbs = transfer_gb / (h2d_ms / 1000.0);
    std::cout << "Pinned H2D Throughput: " << h2d_gbs << " GB/s (" << h2d_ms << " ms)\n";
    
    potts::cuda::CudaManager::init_curand_states(d_states, L, config.seed);

    for (int i = 0; i < mcs; ++i) {
        // Red phase
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, true);
        cudaDeviceSynchronize();
        // Black phase
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, false);
        cudaDeviceSynchronize();
    }

    // Profile pinned Device-to-Host transfer
    cudaEventRecord(start_event);
    cudaMemcpy(grid_gpu.get_orientations_ptr(), d_orientations, num_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, start_event, stop_event);
    double d2h_gbs = transfer_gb / (d2h_ms / 1000.0);
    std::cout << "Pinned D2H Throughput: " << d2h_gbs << " GB/s (" << d2h_ms << " ms)\n";

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double gpu_energy = compute_energy(grid_gpu, config.j_gb);
    std::cout << "GPU Energy after " << mcs << " MCS: " << gpu_energy << "\n";

    cudaFree(d_orientations);
    cudaFree(d_states);

    // Statistical equivalence check (within 5% difference)
    double diff = std::abs(cpu_energy - gpu_energy) / cpu_energy;
    std::cout << "Potts Energy difference: " << diff * 100.0 << "%\n";

    if (diff >= 0.05) {
        std::cout << "FAIL: CPU and GPU Potts results diverge significantly.\n";
        return 1;
    }
    std::cout << "PASS: CPU and GPU Potts results are statistically equivalent.\n";

    // =========================================================================
    // Part 2: GPU Hamiltonian Reduction Verification vs CPU Hamiltonian
    // =========================================================================
    std::cout << "\n--- Testing GPU Hamiltonian Reduction vs CPU Calculation ---\n";
    potts::Grid grid_ham(L, true);
    potts::initialize_voronoi(grid_ham, 36, 0.30, 999);

    double cpu_ham = potts::compute_total_energy_cpu(grid_ham, config);
    std::cout << "CPU Computed Hamiltonian: " << cpu_ham << "\n";

    uint8_t* d_elements = nullptr;
    potts::cuda::GpuMetricsRaw* d_metrics = nullptr;
    cudaMalloc(&d_orientations, num_elements * sizeof(uint16_t));
    cudaMalloc(&d_elements, num_elements * sizeof(uint8_t));
    cudaMalloc(&d_states, num_elements * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&d_metrics, sizeof(potts::cuda::GpuMetricsRaw));

    cudaMemcpy(d_orientations, grid_ham.get_orientations_ptr(), num_elements * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements, grid_ham.get_elements_ptr(), num_elements * sizeof(uint8_t), cudaMemcpyHostToDevice);

    potts::cuda::GpuMetrics gpu_metrics = potts::cuda::compute_metrics_cuda(
        d_metrics, d_orientations, d_elements, L, config.j_gb, config.h_seg, config.j_chem
    );
    std::cout << "GPU Reduced Hamiltonian: " << gpu_metrics.hamiltonian << "\n";
    std::cout << "Initial c_boundary: " << gpu_metrics.c_boundary << " | c_bulk: " << gpu_metrics.c_bulk << "\n";

    double ham_err = std::abs(cpu_ham - gpu_metrics.hamiltonian);
    std::cout << "Hamiltonian Absolute Difference: " << ham_err << "\n";
    if (ham_err > 1e-3) {
        std::cerr << "FAIL: GPU Hamiltonian reduction diverges from CPU calculation!\n";
        return 1;
    }
    std::cout << "PASS: GPU Hamiltonian reduction matches CPU calculation.\n";

    // =========================================================================
    // Part 3: 4-Color Partition Kawasaki Kernel Atomic Conservation (Delta N_Ce = 0)
    // =========================================================================
    std::cout << "\n--- Testing 4-Color Partition Kawasaki Atomic Conservation (Delta N_Ce = 0) ---\n";
    unsigned long long ce_initial = gpu_metrics.total_ce;
    std::cout << "Initial Ce Atoms: " << ce_initial << "\n";

    potts::cuda::CudaManager::init_curand_states(d_states, L, 777);

    int kawasaki_steps = 50;
    for (int s = 0; s < kawasaki_steps; ++s) {
        // Potts step
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, true);
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, false);
        // Kawasaki directional 4-color partition sweep
        potts::cuda::launch_kawasaki_sweep(d_elements, d_orientations, d_states, L, config.h_seg, config.j_chem, config.temp);
    }

    potts::cuda::GpuMetrics final_metrics = potts::cuda::compute_metrics_cuda(
        d_metrics, d_orientations, d_elements, L, config.j_gb, config.h_seg, config.j_chem
    );

    std::cout << "Final Ce Atoms: " << final_metrics.total_ce << "\n";
    std::cout << "Final Hamiltonian: " << final_metrics.hamiltonian << "\n";
    std::cout << "Final c_boundary: " << final_metrics.c_boundary << " | c_bulk: " << final_metrics.c_bulk << "\n";

    if (final_metrics.total_ce != ce_initial) {
        std::cerr << "FAIL: Atomic conservation violation! Expected " << ce_initial 
                  << " Ce atoms, got " << final_metrics.total_ce << " (Delta N_Ce != 0)\n";
        return 1;
    }
    std::cout << "PASS: Strict atomic conservation verified (Delta N_Ce = 0).\n";

    if (final_metrics.c_boundary <= final_metrics.c_bulk) {
        std::cerr << "FAIL: Expected c_boundary > c_bulk under segregation regime!\n";
        return 1;
    }
    std::cout << "PASS: Solute boundary segregation verified (c_boundary > c_bulk).\n";

    cudaFree(d_orientations);
    cudaFree(d_elements);
    cudaFree(d_states);
    cudaFree(d_metrics);

    std::cout << "\nALL TESTS PASSED SUCCESSFULLY.\n";
    return 0;
}
