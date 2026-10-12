#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include "cuda/potts_kernels.cuh"
#include "cuda/kawasaki_kernels.cuh"
#include "cuda/cuda_manager.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <cassert>
#include <cuda_runtime.h>

std::string resolve_output_path() {
    if (std::filesystem::exists("outputs")) {
        return "outputs/energy_log.csv";
    }
    if (std::filesystem::exists("../outputs")) {
        return "../outputs/energy_log.csv";
    }
    std::filesystem::create_directories("outputs");
    return "outputs/energy_log.csv";
}

int main(int argc, char** argv) {
    potts::Config config;
    config.size = 128;
    config.q_states = 32;
    config.temp = 0.4; // Low Ts for segregation
    config.j_gb = 1.0;
    config.h_seg = 1.5;
    config.j_chem = 0.5;
    config.seed = 42;

    int total_mcs = 50;
    if (argc > 1) {
        total_mcs = std::atoi(argv[1]);
    }

    int L = config.size;
    int total_sites = L * L;

    std::cout << "========================================================\n";
    std::cout << " LatticeForge CUDA Engine: Potts + Kawasaki Simulation \n";
    std::cout << " Lattice Size: " << L << "x" << L << " | Steps: " << total_mcs << " | Temp: " << config.temp << "\n";
    std::cout << "========================================================\n";

    // Host pinned grid buffer
    potts::Grid grid(L, true);
    potts::initialize_voronoi(grid, 36, 0.30, config.seed);

    int initial_ce_count = 0;
    for (int i = 0; i < total_sites; ++i) {
        if (grid.get_element(i) == 1) initial_ce_count++;
    }
    std::cout << "Initial Ce atoms: " << initial_ce_count << " (" 
              << (100.0 * initial_ce_count / total_sites) << "%)\n";

    // Allocate Device memory
    uint16_t* d_orientations = nullptr;
    uint8_t* d_elements = nullptr;
    curandStatePhilox4_32_10_t* d_states = nullptr;
    potts::cuda::GpuMetricsRaw* d_metrics = nullptr;

    cudaMalloc(&d_orientations, total_sites * sizeof(uint16_t));
    cudaMalloc(&d_elements, total_sites * sizeof(uint8_t));
    cudaMalloc(&d_states, total_sites * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&d_metrics, sizeof(potts::cuda::GpuMetricsRaw));

    // Copy Host -> Device
    cudaMemcpy(d_orientations, grid.get_orientations_ptr(), total_sites * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements, grid.get_elements_ptr(), total_sites * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Initialize Philox RNG
    potts::cuda::CudaManager::init_curand_states(d_states, L, config.seed);

    // Prepare CSV output
    std::string csv_path = resolve_output_path();
    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open " << csv_path << " for writing!\n";
        return 1;
    }

    // Write header
    csv_file << "mcs,hamiltonian,c_boundary,c_bulk,temp\n";
    csv_file.flush();
    std::cout << "Streaming metrics to: " << csv_path << "\n\n";

    // Step 0: Initial state evaluation
    potts::cuda::GpuMetrics m0 = potts::cuda::compute_metrics_cuda(
        d_metrics, d_orientations, d_elements, L, config.j_gb, config.h_seg, config.j_chem
    );

    csv_file << 0 << ","
             << std::fixed << std::setprecision(4) << m0.hamiltonian << ","
             << std::setprecision(6) << m0.c_boundary << ","
             << m0.c_bulk << ","
             << std::setprecision(4) << config.temp << "\n";
    csv_file.flush();

    std::cout << std::setw(6) << "MCS" 
              << std::setw(16) << "Hamiltonian" 
              << std::setw(14) << "c_boundary" 
              << std::setw(14) << "c_bulk" 
              << std::setw(14) << "Enrichment" << "\n";
    std::cout << std::string(64, '-') << "\n";
    std::cout << std::setw(6) << 0 
              << std::setw(16) << std::fixed << std::setprecision(2) << m0.hamiltonian
              << std::setw(14) << std::setprecision(4) << m0.c_boundary
              << std::setw(14) << m0.c_bulk
              << std::setw(14) << (m0.c_boundary / std::max(1e-6, m0.c_bulk)) << "\n";

    // Main Simulation Loop
    for (int step = 1; step <= total_mcs; ++step) {
        // 1. Potts Grain Growth: Red-Black checkerboard
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, true);
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, config.q_states, config.j_gb, false);

        // 2. Kawasaki Solute Diffusion: 4-color sub-lattice partition directional kernel
        potts::cuda::launch_kawasaki_sweep(d_elements, d_orientations, d_states, L, config.h_seg, config.j_chem, config.temp);

        // 3. GPU-side Reduction: Compute Hamiltonian and segregation metrics
        potts::cuda::GpuMetrics m = potts::cuda::compute_metrics_cuda(
            d_metrics, d_orientations, d_elements, L, config.j_gb, config.h_seg, config.j_chem
        );

        // Strict Atomic Conservation Assertion: Delta N_Ce == 0
        assert(m.total_ce == static_cast<unsigned long long>(initial_ce_count) && "Atomic conservation violation: Delta N_Ce != 0!");

        // Stream row directly to CSV
        csv_file << step << ","
                 << std::fixed << std::setprecision(4) << m.hamiltonian << ","
                 << std::setprecision(6) << m.c_boundary << ","
                 << m.c_bulk << ","
                 << std::setprecision(4) << config.temp << "\n";
        csv_file.flush();

        if (step % 5 == 0 || step == total_mcs) {
            std::cout << std::setw(6) << step 
                      << std::setw(16) << std::fixed << std::setprecision(2) << m.hamiltonian
                      << std::setw(14) << std::setprecision(4) << m.c_boundary
                      << std::setw(14) << m.c_bulk
                      << std::setw(14) << (m.c_boundary / std::max(1e-6, m.c_bulk)) << "\n";
        }
    }

    std::cout << std::string(64, '-') << "\n";
    std::cout << "Simulation completed successfully. Energy log written to " << csv_path << "\n";

    csv_file.close();

    // If written to ../outputs/energy_log.csv, mirror to outputs/ if needed
    if (csv_path == "../outputs/energy_log.csv" && std::filesystem::exists("outputs")) {
        try {
            std::filesystem::copy_file(csv_path, "outputs/energy_log.csv", std::filesystem::copy_options::overwrite_existing);
        } catch (...) {}
    }

    // Cleanup
    cudaFree(d_orientations);
    cudaFree(d_elements);
    cudaFree(d_states);
    cudaFree(d_metrics);

    return 0;
}
