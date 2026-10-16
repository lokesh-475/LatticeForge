#ifndef LATTICE_FORGE_LATTICE_IO_HPP
#define LATTICE_FORGE_LATTICE_IO_HPP

#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace potts {

// 8-byte file magic identifier: "LFORGE01"
constexpr char SNAPSHOT_MAGIC[8] = {'L', 'F', 'O', 'R', 'G', 'E', '0', '1'};
constexpr uint32_t SNAPSHOT_VERSION = 1;

#pragma pack(push, 1)
/**
 * @brief High-throughput binary snapshot header (128 bytes).
 * 
 * Provides self-describing metadata for uncompressed binary dumps of
 * multi-million site arrays (\sigma and c) without ASCII parsing overhead.
 */
struct LatticeSnapshotHeader {
    char     magic[8];          // 8 bytes: 'L', 'F', 'O', 'R', 'G', 'E', '0', '1'
    uint32_t header_size;       // 4 bytes: sizeof(LatticeSnapshotHeader) = 128
    uint32_t version;           // 4 bytes: format version (1)
    uint32_t dim_x;             // 4 bytes: lattice width L
    uint32_t dim_y;             // 4 bytes: lattice height L
    uint64_t total_sites;       // 8 bytes: N = L * L
    uint32_t bytes_per_sigma;   // 4 bytes: sizeof(uint16_t) = 2
    uint32_t bytes_per_c;       // 4 bytes: sizeof(uint8_t) = 1
    int32_t  mcs_step;          // 4 bytes: Monte Carlo step at time of dump
    uint32_t flags;             // 4 bytes: format flags (0 = uncompressed raw)
    double   temp;              // 8 bytes: reduced temperature Ts
    double   j_gb;              // 8 bytes: Potts grain boundary energy coefficient
    double   h_seg;             // 8 bytes: solute segregation potential
    double   j_chem;            // 8 bytes: chemical unlike-bond interaction coefficient
    uint8_t  reserved[48];      // 48 bytes: zeroed padding ensuring exact 128-byte alignment
};
#pragma pack(pop)

static_assert(sizeof(LatticeSnapshotHeader) == 128, "LatticeSnapshotHeader must be exactly 128 bytes");

/**
 * @brief I/O performance metrics and diagnostics.
 */
struct IoMetrics {
    std::string filepath;
    size_t      total_sites = 0;
    size_t      payload_bytes = 0;
    size_t      total_file_bytes = 0;
    double      time_ms = 0.0;
    double      throughput_mb_s = 0.0;
    bool        success = false;
    std::string error_message;
};

/**
 * @brief Dumps continuous uncompressed binary arrays (\sigma and c) with header.
 * 
 * @param filepath Target file path (e.g. outputs/lattice_state.bin)
 * @param grid Reference to simulation Grid containing host buffers
 * @param mcs_step Current Monte Carlo step index
 * @param config Simulation physics configuration
 * @return IoMetrics with timing and throughput
 */
IoMetrics dump_lattice_binary(
    const std::string& filepath,
    const Grid& grid,
    int mcs_step = 0,
    const Config& config = Config()
);

/**
 * @brief Raw pointer dump for CUDA device copies or custom memory buffers.
 */
IoMetrics dump_lattice_binary(
    const std::string& filepath,
    const uint16_t* orientations,
    const uint8_t* elements,
    int L,
    int mcs_step = 0,
    const Config& config = Config()
);

/**
 * @brief High-level snapshot dumper resolving path in outputs/ directory.
 */
IoMetrics dump_lattice_snapshot(
    const Grid& grid,
    int mcs_step = 0,
    const Config& config = Config(),
    const std::string& filename = "lattice_state.bin"
);

/**
 * @brief Reads binary snapshot header metadata without reading full payloads.
 */
bool read_snapshot_header(
    const std::string& filepath,
    LatticeSnapshotHeader& header_out,
    std::string* error_out = nullptr
);

/**
 * @brief Loads binary snapshot directly into an existing Grid instance.
 */
bool load_lattice_binary(
    const std::string& filepath,
    Grid& grid,
    LatticeSnapshotHeader* header_out = nullptr,
    std::string* error_out = nullptr
);

/**
 * @brief Loads binary snapshot into dynamically sized std::vector buffers.
 */
bool load_lattice_binary(
    const std::string& filepath,
    std::vector<uint16_t>& orientations_out,
    std::vector<uint8_t>& elements_out,
    LatticeSnapshotHeader& header_out,
    std::string* error_out = nullptr
);

} // namespace potts

#endif // LATTICE_FORGE_LATTICE_IO_HPP
