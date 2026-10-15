#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include "cuda/potts_kernels.cuh"
#include "cuda/kawasaki_kernels.cuh"
#include "cuda/cuda_manager.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace fs = std::filesystem;

struct BenchmarkResult {
    int grid_dim;
    long long total_sites;
    int mcs_steps;

    // CUDA Timings (ms) & Throughput
    float potts_time_ms;
    float kawasaki_time_ms;
    float total_time_ms;
    double potts_gsups;
    double kawasaki_gsups;
    double sustained_gsups;

    // Single-Threaded CPU Baseline Timings (ms) & Throughput
    float cpu_potts_time_ms;
    float cpu_kawasaki_time_ms;
    float cpu_total_time_ms;
    double cpu_sustained_gsups;

    // Relative Acceleration / Speedup Factors (x-fold)
    double potts_speedup;
    double kawasaki_speedup;
    double sustained_speedup;
};

struct GpuVoronoiSeed {
    float y, x;
    uint16_t q;
    uint8_t is_ce;
};

// GPU Voronoi initialization kernel for arbitrary grid dimension L
__global__ void init_voronoi_gpu_kernel(
    uint16_t* d_orientations,
    uint8_t* d_elements,
    const GpuVoronoiSeed* d_seeds,
    int num_seeds,
    int L,
    int shift,
    int mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sites = L * L;
    if (idx >= total_sites) return;

    int y = (idx >> shift) & mask;
    int x = idx & mask;

    float min_d2 = 1e30f;
    int best = 0;
    float fL = static_cast<float>(L);

    for (int i = 0; i < num_seeds; ++i) {
        float dy = fabsf(static_cast<float>(y) - d_seeds[i].y);
        if (dy > fL - dy) dy = fL - dy;
        float dx = fabsf(static_cast<float>(x) - d_seeds[i].x);
        if (dx > fL - dx) dx = fL - dx;
        float d2 = dy * dy + dx * dx;
        if (d2 < min_d2) {
            min_d2 = d2;
            best = i;
        }
    }

    d_orientations[idx] = d_seeds[best].q;
    d_elements[idx] = d_seeds[best].is_ce;
}

std::string resolve_benchmark_csv_path() {
    if (fs::exists("outputs") && fs::is_directory("outputs")) {
        return "outputs/benchmark_results.csv";
    }
    if (fs::exists("../outputs") && fs::is_directory("../outputs")) {
        return "../outputs/benchmark_results.csv";
    }
    fs::create_directories("outputs");
    return "outputs/benchmark_results.csv";
}

std::string resolve_performance_logs_path() {
    if (fs::exists("docs") && fs::is_directory("docs")) {
        fs::create_directories("docs/benchmarks");
        return "docs/benchmarks/performance_logs.csv";
    }
    if (fs::exists("../docs") && fs::is_directory("../docs")) {
        fs::create_directories("../docs/benchmarks");
        return "../docs/benchmarks/performance_logs.csv";
    }
    fs::create_directories("docs/benchmarks");
    return "docs/benchmarks/performance_logs.csv";
}

void print_gpu_info() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "================================================================================\n";
    std::cout << " LatticeForge High-Resolution CUDA Benchmark Engine\n";
    std::cout << " GPU Device:           " << prop.name << "\n";
    std::cout << " Compute Capability:  " << prop.major << "." << prop.minor << "\n";
    std::cout << " Multiprocessors (SM): " << prop.multiProcessorCount << "\n";
    std::cout << " Total Global Memory:  " << std::fixed << std::setprecision(2)
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << " Warp Size:            " << prop.warpSize << "\n";
    std::cout << "================================================================================\n\n";
}

BenchmarkResult benchmark_grid(int L, int mcs_steps, int warmup_steps) {
    long long total_sites = static_cast<long long>(L) * L;
    int shift = 0;
    while ((1 << shift) < L) {
        shift++;
    }
    int mask = L - 1;

    std::cout << ">>> Running benchmark for Grid: " << L << "x" << L 
              << " (" << total_sites << " sites) | Steps: " << mcs_steps 
              << " (Warmup: " << warmup_steps << ")...\n";

    // Physics parameters matching standard simulation regime
    const int q_states = 32;
    const double j_gb = 1.0;
    const double h_seg = 1.5;
    const double j_chem = 0.5;
    const double temp = 0.4;
    const unsigned long long seed = 42 + L;

    // Allocate GPU Device buffers
    uint16_t* d_orientations = nullptr;
    uint8_t* d_elements = nullptr;
    curandStatePhilox4_32_10_t* d_states = nullptr;

    CUDA_CHECK(cudaMalloc(&d_orientations, total_sites * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_elements, total_sites * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_states, total_sites * sizeof(curandStatePhilox4_32_10_t)));

    // Initialize Philox RNG states
    potts::cuda::CudaManager::init_curand_states(d_states, L, seed);

    // Initialize Voronoi grains on GPU
    const int num_grains = 36;
    const double fraction_ce = 0.30;
    std::mt19937 host_rng(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> pos_dist(0.0f, static_cast<float>(L));
    std::uniform_int_distribution<int> q_dist(1, q_states);

    std::vector<GpuVoronoiSeed> host_seeds(num_grains);
    int num_ce = static_cast<int>(std::round(num_grains * fraction_ce));
    for (int i = 0; i < num_grains; ++i) {
        host_seeds[i] = {
            pos_dist(host_rng),
            pos_dist(host_rng),
            static_cast<uint16_t>(q_dist(host_rng)),
            static_cast<uint8_t>(i < num_ce ? 1 : 0)
        };
    }
    std::shuffle(host_seeds.begin(), host_seeds.end(), host_rng);

    GpuVoronoiSeed* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, num_grains * sizeof(GpuVoronoiSeed)));
    CUDA_CHECK(cudaMemcpy(d_seeds, host_seeds.data(), num_grains * sizeof(GpuVoronoiSeed), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = static_cast<int>((total_sites + block_size - 1) / block_size);
    init_voronoi_gpu_kernel<<<num_blocks, block_size>>>(
        d_orientations, d_elements, d_seeds, num_grains, L, shift, mask
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_seeds));

    // Warm-up Phase: prime instruction caches, L2 cache, and GPU frequency boosts
    for (int w = 0; w < warmup_steps; ++w) {
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, true);
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, false);
        potts::cuda::launch_kawasaki_sweep(d_elements, d_orientations, d_states, L, h_seg, j_chem, temp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // High-Resolution cudaEvent_t Timers
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // 1. Potts Checkerboard Benchmark Run
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    for (int s = 0; s < mcs_steps; ++s) {
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, true);
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, false);
    }
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float potts_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&potts_time_ms, start_event, stop_event));

    // 2. Kawasaki 4-Color Sweep Benchmark Run
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    for (int s = 0; s < mcs_steps; ++s) {
        potts::cuda::launch_kawasaki_sweep(d_elements, d_orientations, d_states, L, h_seg, j_chem, temp);
    }
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float kawasaki_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kawasaki_time_ms, start_event, stop_event));

    // 3. Combined Sustained Multi-Physics Pipeline Benchmark Run
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    for (int s = 0; s < mcs_steps; ++s) {
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, true);
        potts::cuda::launch_potts_checkerboard(d_orientations, d_states, L, q_states, j_gb, false);
        potts::cuda::launch_kawasaki_sweep(d_elements, d_orientations, d_states, L, h_seg, j_chem, temp);
    }
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float total_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_event, stop_event));

    // Cleanup cudaEvent_t objects and Device memory
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_orientations));
    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_states));

    // Giga-SUPS Calculation (GPU):
    // 1 Site Update = 1 lattice site evaluation
    // Site Updates across M steps = total_sites * mcs_steps
    // Giga-SUPS = (total_sites * mcs_steps) / (time_ms * 1e6)
    double total_evaluations = static_cast<double>(total_sites) * mcs_steps;
    double potts_gsups = total_evaluations / (potts_time_ms * 1.0e6);
    double kawasaki_gsups = total_evaluations / (kawasaki_time_ms * 1.0e6);
    double sustained_gsups = total_evaluations / (total_time_ms * 1.0e6);

    // -------------------------------------------------------------------------
    // Single-Threaded C++ Baseline Benchmark Phase
    // -------------------------------------------------------------------------
    std::cout << "    [CPU Baseline] Profiling single-threaded C++ engine for Grid: " << L << "x" << L << "...\n";
    int cpu_sample_steps = (L >= 2048) ? 5 : ((L >= 1024) ? 10 : ((L >= 512) ? 20 : 50));
    
    potts::Grid cpu_grid(L);
    potts::initialize_voronoi(cpu_grid, num_grains, fraction_ce, static_cast<int>(seed));
    std::mt19937 cpu_rng(static_cast<unsigned int>(seed));
    potts::Config cpu_config;
    cpu_config.size = L;
    cpu_config.q_states = q_states;
    cpu_config.temp = temp;
    cpu_config.j_gb = j_gb;
    cpu_config.h_seg = h_seg;
    cpu_config.j_chem = j_chem;
    cpu_config.seed = static_cast<int>(seed);

    // Warm-up CPU caches
    for (int w = 0; w < std::min(2, warmup_steps); ++w) {
        potts::potts_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
        potts::kawasaki_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
    }

    // 1. CPU Potts Sweep
    auto c_p_start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < cpu_sample_steps; ++s) {
        potts::potts_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
    }
    auto c_p_stop = std::chrono::high_resolution_clock::now();
    double cpu_p_sample_ms = std::chrono::duration<double, std::milli>(c_p_stop - c_p_start).count();
    float cpu_potts_time_ms = static_cast<float>((cpu_p_sample_ms / cpu_sample_steps) * mcs_steps);

    // 2. CPU Kawasaki Sweep
    auto c_k_start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < cpu_sample_steps; ++s) {
        potts::kawasaki_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
    }
    auto c_k_stop = std::chrono::high_resolution_clock::now();
    double cpu_k_sample_ms = std::chrono::duration<double, std::milli>(c_k_stop - c_k_start).count();
    float cpu_kawasaki_time_ms = static_cast<float>((cpu_k_sample_ms / cpu_sample_steps) * mcs_steps);

    // 3. CPU Sustained Multi-Physics Pipeline
    auto c_tot_start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < cpu_sample_steps; ++s) {
        potts::potts_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
        potts::kawasaki_mc_step_cpu(cpu_grid, cpu_config, cpu_rng);
    }
    auto c_tot_stop = std::chrono::high_resolution_clock::now();
    double cpu_tot_sample_ms = std::chrono::duration<double, std::milli>(c_tot_stop - c_tot_start).count();
    float cpu_total_time_ms = static_cast<float>((cpu_tot_sample_ms / cpu_sample_steps) * mcs_steps);

    double cpu_sustained_gsups = total_evaluations / (cpu_total_time_ms * 1.0e6);

    // Relative Speedup Calculation
    double potts_speedup = (potts_time_ms > 0.0f) ? (static_cast<double>(cpu_potts_time_ms) / potts_time_ms) : 1.0;
    double kawasaki_speedup = (kawasaki_time_ms > 0.0f) ? (static_cast<double>(cpu_kawasaki_time_ms) / kawasaki_time_ms) : 1.0;
    double sustained_speedup = (total_time_ms > 0.0f) ? (static_cast<double>(cpu_total_time_ms) / total_time_ms) : 1.0;

    std::cout << "    [CUDA GPU] Potts:     " << std::setw(9) << std::fixed << std::setprecision(3) << potts_time_ms 
              << " ms | " << std::setw(6) << std::setprecision(2) << potts_gsups << " G-SUPS\n";
    std::cout << "    [CUDA GPU] Kawasaki:  " << std::setw(9) << std::fixed << std::setprecision(3) << kawasaki_time_ms 
              << " ms | " << std::setw(6) << std::setprecision(2) << kawasaki_gsups << " G-SUPS\n";
    std::cout << "    [CUDA GPU] Sustained: " << std::setw(9) << std::fixed << std::setprecision(3) << total_time_ms 
              << " ms | " << std::setw(6) << std::setprecision(2) << sustained_gsups << " G-SUPS\n";
    std::cout << "    [CPU Base] Sustained: " << std::setw(9) << std::fixed << std::setprecision(3) << cpu_total_time_ms 
              << " ms | " << std::setw(6) << std::setprecision(3) << cpu_sustained_gsups << " G-SUPS\n";
    std::cout << "    >>> Relative Acceleration: " << std::setw(6) << std::setprecision(2) << sustained_speedup 
              << "x (Potts: " << potts_speedup << "x, Kawasaki: " << kawasaki_speedup << "x)\n\n";

    return BenchmarkResult{
        L,
        total_sites,
        mcs_steps,
        potts_time_ms,
        kawasaki_time_ms,
        total_time_ms,
        potts_gsups,
        kawasaki_gsups,
        sustained_gsups,
        cpu_potts_time_ms,
        cpu_kawasaki_time_ms,
        cpu_total_time_ms,
        cpu_sustained_gsups,
        potts_speedup,
        kawasaki_speedup,
        sustained_speedup
    };
}

int main(int argc, char** argv) {
    int mcs_steps = 100;
    int warmup_steps = 10;
    if (argc > 1) {
        mcs_steps = std::atoi(argv[1]);
    }
    if (argc > 2) {
        warmup_steps = std::atoi(argv[2]);
    }

    print_gpu_info();

    // Required benchmark grids: 128^2, 512^2, 1024^2, 2048^2
    const std::vector<int> grid_sizes = {128, 512, 1024, 2048};
    std::vector<BenchmarkResult> results;
    results.reserve(grid_sizes.size());

    for (int L : grid_sizes) {
        BenchmarkResult res = benchmark_grid(L, mcs_steps, warmup_steps);
        results.push_back(res);
    }

    // Write performance logs to outputs/benchmark_results.csv and docs/benchmarks/performance_logs.csv
    std::string csv_path = resolve_benchmark_csv_path();
    std::string docs_csv_path = resolve_performance_logs_path();

    auto write_csv = [&](const std::string& path) {
        std::ofstream csv_file(path);
        if (!csv_file.is_open()) {
            std::cerr << "Failed to open CSV file for writing at: " << path << "\n";
            return false;
        }

        csv_file << "grid_dim,total_sites,mcs_steps,"
                 << "potts_time_ms,kawasaki_time_ms,total_time_ms,"
                 << "potts_gsups,kawasaki_gsups,sustained_gsups,"
                 << "cpu_potts_time_ms,cpu_kawasaki_time_ms,cpu_total_time_ms,"
                 << "cpu_sustained_gsups,potts_speedup,kawasaki_speedup,sustained_speedup\n";

        for (const auto& r : results) {
            csv_file << r.grid_dim << ","
                     << r.total_sites << ","
                     << r.mcs_steps << ","
                     << std::fixed << std::setprecision(4)
                     << r.potts_time_ms << ","
                     << r.kawasaki_time_ms << ","
                     << r.total_time_ms << ","
                     << std::setprecision(4)
                     << r.potts_gsups << ","
                     << r.kawasaki_gsups << ","
                     << r.sustained_gsups << ","
                     << r.cpu_potts_time_ms << ","
                     << r.cpu_kawasaki_time_ms << ","
                     << r.cpu_total_time_ms << ","
                     << r.cpu_sustained_gsups << ","
                     << std::setprecision(2)
                     << r.potts_speedup << ","
                     << r.kawasaki_speedup << ","
                     << r.sustained_speedup << "\n";
        }
        csv_file.close();
        return true;
    };

    write_csv(csv_path);
    write_csv(docs_csv_path);

    // Mirror to alternative directory structures if running from build/
    if (csv_path == "outputs/benchmark_results.csv" && fs::exists("../outputs")) {
        try { fs::copy_file(csv_path, "../outputs/benchmark_results.csv", fs::copy_options::overwrite_existing); } catch (...) {}
    } else if (csv_path == "../outputs/benchmark_results.csv" && fs::exists("outputs")) {
        try { fs::copy_file(csv_path, "outputs/benchmark_results.csv", fs::copy_options::overwrite_existing); } catch (...) {}
    }

    if (docs_csv_path == "docs/benchmarks/performance_logs.csv" && fs::exists("../docs")) {
        try {
            fs::create_directories("../docs/benchmarks");
            fs::copy_file(docs_csv_path, "../docs/benchmarks/performance_logs.csv", fs::copy_options::overwrite_existing);
        } catch (...) {}
    } else if (docs_csv_path == "../docs/benchmarks/performance_logs.csv" && fs::exists("docs")) {
        try {
            fs::create_directories("docs/benchmarks");
            fs::copy_file(docs_csv_path, "docs/benchmarks/performance_logs.csv", fs::copy_options::overwrite_existing);
        } catch (...) {}
    }

    // Print Formatted Summary Table
    std::cout << "\n===================================================================================================================================================\n";
    std::cout << "                                               LATTICEFORGE BENCHMARK SUMMARY & ACCELERATION RESULTS                                               \n";
    std::cout << "===================================================================================================================================================\n";
    std::cout << std::setw(11) << "Grid Dim" 
              << std::setw(14) << "Total Sites" 
              << std::setw(11) << "MCS Steps" 
              << std::setw(15) << "GPU Time(ms)" 
              << std::setw(15) << "CPU Time(ms)" 
              << std::setw(13) << "GPU G-SUPS" 
              << std::setw(13) << "CPU G-SUPS" 
              << std::setw(15) << "Potts Speedup"
              << std::setw(18) << "Kawasaki Speedup" 
              << std::setw(19) << "Sustained Speedup" << "\n";
    std::cout << std::string(147, '-') << "\n";

    for (const auto& r : results) {
        std::string dim_str = std::to_string(r.grid_dim) + "x" + std::to_string(r.grid_dim);
        std::cout << std::setw(11) << dim_str
                  << std::setw(14) << r.total_sites
                  << std::setw(11) << r.mcs_steps
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.total_time_ms
                  << std::setw(15) << r.cpu_total_time_ms
                  << std::setw(13) << std::setprecision(2) << r.sustained_gsups
                  << std::setw(13) << std::setprecision(4) << r.cpu_sustained_gsups
                  << std::setw(14) << std::setprecision(2) << r.potts_speedup << "x"
                  << std::setw(17) << r.kawasaki_speedup << "x"
                  << std::setw(18) << r.sustained_speedup << "x" << "\n";
    }
    std::cout << std::string(147, '=') << "\n\n";

    std::cout << "[Speedup Verification Target: >40x Acceleration on Production Lattice]\n";
    for (const auto& r : results) {
        std::cout << "  - Grid " << std::setw(4) << r.grid_dim << "^2 (" 
                  << std::setw(7) << r.total_sites << " sites): "
                  << std::fixed << std::setprecision(2) << r.sustained_speedup << "x sustained speedup "
                  << "(GPU: " << r.total_time_ms << " ms vs CPU: " << r.cpu_total_time_ms << " ms)";
        if (r.sustained_speedup >= 40.0) {
            std::cout << "  [PASSED >40x TARGET!]\n";
        } else {
            std::cout << "\n";
        }
    }
    std::cout << "\nDetailed performance logs written to:\n";
    std::cout << "  - " << csv_path << "\n";
    std::cout << "  - " << docs_csv_path << "\n\n";

    return 0;
}
