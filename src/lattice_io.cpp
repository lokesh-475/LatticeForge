#include "potts/lattice_io.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <cmath>

namespace potts {

namespace fs = std::filesystem;

static std::string resolve_output_path(const std::string& filename) {
    if (fs::exists("outputs") && fs::is_directory("outputs")) {
        return "outputs/" + filename;
    }
    if (fs::exists("../outputs") && fs::is_directory("../outputs")) {
        return "../outputs/" + filename;
    }
    fs::create_directories("outputs");
    return "outputs/" + filename;
}

static void mirror_output_file(const std::string& path, const std::string& filename) {
    if (filename.rfind("test_", 0) == 0) return; // Do not mirror temporary test files
    try {
        if (path == "../outputs/" + filename && fs::exists("outputs")) {
            fs::copy_file(path, "outputs/" + filename, fs::copy_options::overwrite_existing);
        } else if (path == "outputs/" + filename && fs::exists("../outputs")) {
            fs::copy_file(path, "../outputs/" + filename, fs::copy_options::overwrite_existing);
        }
    } catch (...) {
        // Non-critical mirror failure ignored
    }
}

IoMetrics dump_lattice_binary(
    const std::string& filepath,
    const uint16_t* orientations,
    const uint8_t* elements,
    int L,
    int mcs_step,
    const Config& config
) {
    IoMetrics metrics;
    metrics.filepath = filepath;
    metrics.total_sites = static_cast<size_t>(L) * L;
    metrics.payload_bytes = metrics.total_sites * sizeof(uint16_t) + metrics.total_sites * sizeof(uint8_t);
    metrics.total_file_bytes = sizeof(LatticeSnapshotHeader) + metrics.payload_bytes;

    if (!orientations || !elements || L <= 0) {
        metrics.success = false;
        metrics.error_message = "Invalid null pointer or non-positive grid dimension.";
        return metrics;
    }

    try {
        fs::path p(filepath);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        // Construct packed 128-byte header
        LatticeSnapshotHeader header;
        std::memset(&header, 0, sizeof(header));
        std::memcpy(header.magic, SNAPSHOT_MAGIC, 8);
        header.header_size = static_cast<uint32_t>(sizeof(LatticeSnapshotHeader));
        header.version = SNAPSHOT_VERSION;
        header.dim_x = static_cast<uint32_t>(L);
        header.dim_y = static_cast<uint32_t>(L);
        header.total_sites = static_cast<uint64_t>(metrics.total_sites);
        header.bytes_per_sigma = static_cast<uint32_t>(sizeof(uint16_t));
        header.bytes_per_c = static_cast<uint32_t>(sizeof(uint8_t));
        header.mcs_step = mcs_step;
        header.flags = 0; // Uncompressed raw binary
        header.temp = config.temp;
        header.j_gb = config.j_gb;
        header.h_seg = config.h_seg;
        header.j_chem = config.j_chem;

        // High-resolution timer for pure unformatted binary dump
        auto start = std::chrono::high_resolution_clock::now();

        std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            metrics.success = false;
            metrics.error_message = "Failed to open file for binary write: " + filepath;
            return metrics;
        }

        // 1. Write fixed 128-byte header
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 2. Stream unformatted multi-million site sigma array in a single direct write
        out.write(
            reinterpret_cast<const char*>(orientations),
            metrics.total_sites * sizeof(uint16_t)
        );

        // 3. Stream unformatted multi-million site c array in a single direct write
        out.write(
            reinterpret_cast<const char*>(elements),
            metrics.total_sites * sizeof(uint8_t)
        );

        out.flush();
        auto end = std::chrono::high_resolution_clock::now();

        if (out.fail()) {
            metrics.success = false;
            metrics.error_message = "Stream write failure occurred during binary dump.";
            return metrics;
        }
        out.close();

        metrics.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_mb = static_cast<double>(metrics.total_file_bytes) / (1024.0 * 1024.0);
        metrics.throughput_mb_s = (metrics.time_ms > 0.0) ? (total_mb / (metrics.time_ms / 1000.0)) : 0.0;
        metrics.success = true;

        // Auto-mirror to opposite directory if applicable
        fs::path file_p(filepath);
        mirror_output_file(filepath, file_p.filename().string());

    } catch (const std::exception& ex) {
        metrics.success = false;
        metrics.error_message = ex.what();
    }

    return metrics;
}

IoMetrics dump_lattice_binary(
    const std::string& filepath,
    const Grid& grid,
    int mcs_step,
    const Config& config
) {
    return dump_lattice_binary(
        filepath,
        grid.get_orientations_ptr(),
        grid.get_elements_ptr(),
        grid.size(),
        mcs_step,
        config
    );
}

IoMetrics dump_lattice_snapshot(
    const Grid& grid,
    int mcs_step,
    const Config& config,
    const std::string& filename
) {
    std::string path = resolve_output_path(filename);
    return dump_lattice_binary(path, grid, mcs_step, config);
}

bool read_snapshot_header(
    const std::string& filepath,
    LatticeSnapshotHeader& header_out,
    std::string* error_out
) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        if (error_out) *error_out = "Cannot open file: " + filepath;
        return false;
    }

    in.read(reinterpret_cast<char*>(&header_out), sizeof(LatticeSnapshotHeader));
    if (in.gcount() != sizeof(LatticeSnapshotHeader)) {
        if (error_out) *error_out = "File too small to contain valid snapshot header.";
        return false;
    }

    if (std::memcmp(header_out.magic, SNAPSHOT_MAGIC, 8) != 0) {
        if (error_out) *error_out = "Invalid magic bytes in header (not LFORGE01).";
        return false;
    }

    if (header_out.version != SNAPSHOT_VERSION) {
        if (error_out) *error_out = "Unsupported snapshot format version: " + std::to_string(header_out.version);
        return false;
    }

    return true;
}

bool load_lattice_binary(
    const std::string& filepath,
    Grid& grid,
    LatticeSnapshotHeader* header_out,
    std::string* error_out
) {
    LatticeSnapshotHeader header;
    if (!read_snapshot_header(filepath, header, error_out)) {
        return false;
    }

    if (header.dim_x != static_cast<uint32_t>(grid.size()) || 
        header.dim_y != static_cast<uint32_t>(grid.size())) {
        if (error_out) {
            *error_out = "Grid dimension mismatch: file is " + 
                         std::to_string(header.dim_x) + "x" + std::to_string(header.dim_y) +
                         " but Grid is " + std::to_string(grid.size()) + "x" + std::to_string(grid.size());
        }
        return false;
    }

    if (header.bytes_per_sigma != sizeof(uint16_t) || header.bytes_per_c != sizeof(uint8_t)) {
        if (error_out) *error_out = "Incompatible site array element byte sizes.";
        return false;
    }

    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        if (error_out) *error_out = "Cannot reopen file for reading payload.";
        return false;
    }

    // Seek past variable header size for future forward compatibility
    in.seekg(header.header_size, std::ios::beg);

    size_t total_sites = static_cast<size_t>(header.total_sites);
    size_t sigma_bytes = total_sites * sizeof(uint16_t);
    size_t c_bytes = total_sites * sizeof(uint8_t);

    in.read(reinterpret_cast<char*>(grid.get_orientations_ptr()), sigma_bytes);
    if (in.gcount() != static_cast<std::streamsize>(sigma_bytes)) {
        if (error_out) *error_out = "Corrupted or truncated orientations array in snapshot.";
        return false;
    }

    in.read(reinterpret_cast<char*>(grid.get_elements_ptr()), c_bytes);
    if (in.gcount() != static_cast<std::streamsize>(c_bytes)) {
        if (error_out) *error_out = "Corrupted or truncated elements array in snapshot.";
        return false;
    }

    if (header_out) {
        *header_out = header;
    }

    return true;
}

bool load_lattice_binary(
    const std::string& filepath,
    std::vector<uint16_t>& orientations_out,
    std::vector<uint8_t>& elements_out,
    LatticeSnapshotHeader& header_out,
    std::string* error_out
) {
    if (!read_snapshot_header(filepath, header_out, error_out)) {
        return false;
    }

    if (header_out.bytes_per_sigma != sizeof(uint16_t) || header_out.bytes_per_c != sizeof(uint8_t)) {
        if (error_out) *error_out = "Incompatible site array element byte sizes.";
        return false;
    }

    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        if (error_out) *error_out = "Cannot reopen file for reading payload.";
        return false;
    }

    in.seekg(header_out.header_size, std::ios::beg);

    size_t total_sites = static_cast<size_t>(header_out.total_sites);
    orientations_out.resize(total_sites);
    elements_out.resize(total_sites);

    size_t sigma_bytes = total_sites * sizeof(uint16_t);
    size_t c_bytes = total_sites * sizeof(uint8_t);

    in.read(reinterpret_cast<char*>(orientations_out.data()), sigma_bytes);
    if (in.gcount() != static_cast<std::streamsize>(sigma_bytes)) {
        if (error_out) *error_out = "Corrupted or truncated orientations array in snapshot.";
        return false;
    }

    in.read(reinterpret_cast<char*>(elements_out.data()), c_bytes);
    if (in.gcount() != static_cast<std::streamsize>(c_bytes)) {
        if (error_out) *error_out = "Corrupted or truncated elements array in snapshot.";
        return false;
    }

    return true;
}

} // namespace potts
