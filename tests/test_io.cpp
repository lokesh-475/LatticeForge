#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include "potts/lattice_io.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

void test_header_spec() {
    std::cout << "[Test 1] Validating 128-byte header binary layout...\n";
    assert(sizeof(potts::LatticeSnapshotHeader) == 128);
    potts::LatticeSnapshotHeader h;
    assert(sizeof(h.magic) == 8);
    assert(sizeof(h.header_size) == 4);
    assert(sizeof(h.total_sites) == 8);
    assert(sizeof(h.temp) == 8);
    std::cout << "  -> Header structure is strictly 128-byte packed.\n";
}

void test_roundtrip_128() {
    std::cout << "[Test 2] Testing roundtrip serialization for 128x128 grid...\n";
    const int L = 128;
    potts::Grid grid_orig(L);
    potts::initialize_voronoi(grid_orig, 24, 0.35, 12345);

    potts::Config cfg;
    cfg.size = L;
    cfg.temp = 0.45;
    cfg.j_gb = 1.0;
    cfg.h_seg = 1.5;
    cfg.j_chem = 0.5;
    const int step = 42;

    std::string test_path = "outputs/test_snapshot_128.bin";
    auto metrics = potts::dump_lattice_binary(test_path, grid_orig, step, cfg);

    assert(metrics.success);
    assert(metrics.total_sites == 16384);
    assert(metrics.payload_bytes == 16384 * 3);
    assert(metrics.total_file_bytes == 128 + 16384 * 3);
    std::cout << "  -> Dumped 16,384 sites in " << metrics.time_ms << " ms (" 
              << metrics.throughput_mb_s << " MB/s)\n";

    // Validate header reading
    potts::LatticeSnapshotHeader header;
    bool read_ok = potts::read_snapshot_header(test_path, header);
    assert(read_ok);
    assert(std::string(header.magic, 8) == "LFORGE01");
    assert(header.version == 1);
    assert(header.header_size == 128);
    assert(header.dim_x == 128);
    assert(header.dim_y == 128);
    assert(header.total_sites == 16384);
    assert(header.bytes_per_sigma == 2);
    assert(header.bytes_per_c == 1);
    assert(header.mcs_step == 42);
    assert(std::abs(header.temp - 0.45) < 1e-6);
    assert(std::abs(header.j_gb - 1.0) < 1e-6);
    assert(std::abs(header.h_seg - 1.5) < 1e-6);
    assert(std::abs(header.j_chem - 0.5) < 1e-6);

    // Load into fresh grid
    potts::Grid grid_loaded(L);
    potts::LatticeSnapshotHeader loaded_header;
    std::string err;
    bool load_ok = potts::load_lattice_binary(test_path, grid_loaded, &loaded_header, &err);
    assert(load_ok);

    // Verify 100% bitwise exact parity for orientations and elements
    for (int i = 0; i < grid_orig.total_sites(); ++i) {
        assert(grid_orig.get_orientation(i) == grid_loaded.get_orientation(i));
        assert(grid_orig.get_element(i) == grid_loaded.get_element(i));
    }
    std::cout << "  -> Bitwise exact fidelity verified across all 16,384 sites.\n";

    // Clean up
    fs::remove(test_path);
}

void test_multimillion_sites_performance() {
    std::cout << "[Test 3] Multi-million site raw binary dump stress test (2048x2048 = 4,194,304 sites)...\n";
    const int L = 2048;
    const size_t total_sites = static_cast<size_t>(L) * L;
    
    // Allocate 2048x2048 grid (~12 MB payload)
    potts::Grid grid(L);

    // Synthetic initialization with deterministic pattern
    uint16_t* sigma = grid.get_orientations_ptr();
    uint8_t* c = grid.get_elements_ptr();
    for (size_t i = 0; i < total_sites; ++i) {
        sigma[i] = static_cast<uint16_t>((i % 32) + 1);
        c[i] = static_cast<uint8_t>((i & 3) == 0 ? 1 : 0);
    }

    potts::Config cfg;
    cfg.size = L;
    cfg.temp = 0.4;
    cfg.j_gb = 1.0;
    cfg.h_seg = 1.5;
    cfg.j_chem = 0.5;

    std::string test_path = "outputs/test_snapshot_2048.bin";
    auto metrics = potts::dump_lattice_binary(test_path, grid, 100, cfg);

    assert(metrics.success);
    assert(metrics.total_sites == 4194304);
    assert(metrics.total_file_bytes == 128 + 4194304 * 3); // 12,583,040 bytes (~12 MB)

    std::cout << "  -> Multi-million site dump completed!\n";
    std::cout << "     Sites:      " << metrics.total_sites << "\n";
    std::cout << "     File Size:  " << std::fixed << std::setprecision(2) 
              << (metrics.total_file_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "     Write Time: " << std::setprecision(3) << metrics.time_ms << " ms\n";
    std::cout << "     Throughput: " << std::setprecision(1) << metrics.throughput_mb_s << " MB/s\n";

    // Design requirement: raw binary dump must execute in milliseconds without text formatting overhead
    assert(metrics.time_ms < 1000.0 && "Binary dump took longer than 1 second!");

    // Verify loaded data integrity on large grid
    potts::Grid grid_verify(L);
    std::string err;
    bool load_ok = potts::load_lattice_binary(test_path, grid_verify, nullptr, &err);
    assert(load_ok);

    const uint16_t* sigma_v = grid_verify.get_orientations_ptr();
    const uint8_t* c_v = grid_verify.get_elements_ptr();
    for (size_t i = 0; i < total_sites; i += 1024) { // Stride check
        assert(sigma[i] == sigma_v[i]);
        assert(c[i] == c_v[i]);
    }
    std::cout << "  -> High-speed reload and array integrity check passed.\n";

    // Clean up
    fs::remove(test_path);
}

void test_error_handling() {
    std::cout << "[Test 4] Robust error handling & integrity validation...\n";
    
    // 1. Non-existent file
    potts::LatticeSnapshotHeader h;
    std::string err;
    bool ok = potts::read_snapshot_header("outputs/non_existent_file.bin", h, &err);
    assert(!ok);
    assert(!err.empty());
    std::cout << "  -> Non-existent file correctly rejected: " << err << "\n";

    // 2. Corrupt magic
    std::string corrupt_path = "outputs/test_corrupt.bin";
    {
        std::ofstream f(corrupt_path, std::ios::binary);
        char bad_magic[8] = {'B', 'A', 'D', 'M', 'A', 'G', 'I', 'C'};
        f.write(bad_magic, 8);
        std::vector<char> dummy(120, 0);
        f.write(dummy.data(), dummy.size());
    }
    ok = potts::read_snapshot_header(corrupt_path, h, &err);
    assert(!ok);
    assert(!err.empty());
    std::cout << "  -> Corrupt magic correctly rejected: " << err << "\n";
    fs::remove(corrupt_path);

    // 3. Dimension mismatch
    potts::Grid grid_small(128);
    potts::initialize_voronoi(grid_small, 10, 0.2, 42);
    std::string small_path = "outputs/test_small.bin";
    potts::dump_lattice_binary(small_path, grid_small);

    potts::Grid grid_large(256);
    ok = potts::load_lattice_binary(small_path, grid_large, nullptr, &err);
    assert(!ok);
    assert(!err.empty());
    std::cout << "  -> Dimension mismatch correctly caught: " << err << "\n";
    fs::remove(small_path);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " LatticeForge High-Throughput Binary Snapshot Unit Tests\n";
    std::cout << "========================================================\n";

    test_header_spec();
    test_roundtrip_128();
    test_multimillion_sites_performance();
    test_error_handling();

    std::cout << "\n>>> ALL BINARY I/O TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
