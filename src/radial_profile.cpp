#include "potts/radial_profile.hpp"
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace potts {

static constexpr double PI = 3.14159265358979323846;
static constexpr double TWO_PI = 2.0 * PI;

std::vector<GrainCentroid> compute_grain_centroids(const Grid& grid, int min_grain_size) {
    int L = grid.size();
    int total_sites = L * L;
    DSU dsu(total_sites);

    // 4-neighbor connected components on periodic 2D lattice
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int idx = grid.idx(y, x);
            uint16_t ori = grid.get_orientation(idx);

            // Right neighbor
            int r_idx = grid.idx(y, grid.right(x));
            if (grid.get_orientation(r_idx) == ori) {
                dsu.unite(idx, r_idx);
            }

            // Down neighbor
            int d_idx = grid.idx(grid.down(y), x);
            if (grid.get_orientation(d_idx) == ori) {
                dsu.unite(idx, d_idx);
            }
        }
    }

    // Group sites by root component
    std::unordered_map<int, std::vector<int>> component_sites;
    component_sites.reserve(128);
    for (int i = 0; i < total_sites; ++i) {
        component_sites[dsu.find(i)].push_back(i);
    }

    std::vector<GrainCentroid> grains;
    grains.reserve(component_sites.size());

    double scale = TWO_PI / static_cast<double>(L);
    int grain_id = 0;

    for (const auto& [root_id, sites] : component_sites) {
        if (static_cast<int>(sites.size()) < min_grain_size) {
            continue;
        }

        // Circular mean for toroidal coordinates to handle PBC wrapping seamlessly
        double sum_cos_x = 0.0, sum_sin_x = 0.0;
        double sum_cos_y = 0.0, sum_sin_y = 0.0;
        int ce_count = 0;

        for (int idx : sites) {
            int y = idx / L;
            int x = idx % L;
            if (grid.get_element(idx) == 1) {
                ce_count++;
            }

            double theta_x = x * scale;
            sum_cos_x += std::cos(theta_x);
            sum_sin_x += std::sin(theta_x);

            double theta_y = y * scale;
            sum_cos_y += std::cos(theta_y);
            sum_sin_y += std::sin(theta_y);
        }

        double mean_theta_x = std::atan2(sum_sin_x, sum_cos_x);
        if (mean_theta_x < 0.0) mean_theta_x += TWO_PI;
        double cx = (mean_theta_x / TWO_PI) * L;
        if (cx >= L) cx -= L;

        double mean_theta_y = std::atan2(sum_sin_y, sum_cos_y);
        if (mean_theta_y < 0.0) mean_theta_y += TWO_PI;
        double cy = (mean_theta_y / TWO_PI) * L;
        if (cy >= L) cy -= L;

        // Compute maximum radius from centroid under minimum image convention
        double r_max = 0.0;
        for (int idx : sites) {
            int y = idx / L;
            int x = idx % L;

            double dx = std::abs(x - cx);
            if (dx > L * 0.5) dx = L - dx;

            double dy = std::abs(y - cy);
            if (dy > L * 0.5) dy = L - dy;

            double r = std::sqrt(dx * dx + dy * dy);
            if (r > r_max) r_max = r;
        }

        double r_eff = std::sqrt(static_cast<double>(sites.size()) / PI);

        GrainCentroid gc;
        gc.id = grain_id++;
        gc.orientation = grid.get_orientation(sites[0]);
        gc.cy = cy;
        gc.cx = cx;
        gc.num_sites = static_cast<int>(sites.size());
        gc.r_max = r_max;
        gc.r_eff = r_eff;
        gc.ce_count = ce_count;
        gc.ce_fraction = static_cast<double>(ce_count) / sites.size();

        grains.push_back(gc);
    }

    // Sort grains by size descending
    std::sort(grains.begin(), grains.end(), [](const GrainCentroid& a, const GrainCentroid& b) {
        return a.num_sites > b.num_sites;
    });

    return grains;
}

RadialProfileResult compute_radial_profile(
    const Grid& grid,
    double bin_width,
    int min_grain_size
) {
    int L = grid.size();
    int total_sites = L * L;
    DSU dsu(total_sites);

    // 4-neighbor connected components on periodic 2D lattice
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int idx = grid.idx(y, x);
            uint16_t ori = grid.get_orientation(idx);

            int r_idx = grid.idx(y, grid.right(x));
            if (grid.get_orientation(r_idx) == ori) {
                dsu.unite(idx, r_idx);
            }

            int d_idx = grid.idx(grid.down(y), x);
            if (grid.get_orientation(d_idx) == ori) {
                dsu.unite(idx, d_idx);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> component_sites;
    component_sites.reserve(128);
    for (int i = 0; i < total_sites; ++i) {
        component_sites[dsu.find(i)].push_back(i);
    }

    double scale = TWO_PI / static_cast<double>(L);
    std::vector<GrainCentroid> grains;
    grains.reserve(component_sites.size());

    struct SiteSample {
        double r;
        uint8_t is_ce;
    };
    std::vector<SiteSample> samples;
    samples.reserve(total_sites);

    double total_r_eff = 0.0;
    int grain_id = 0;

    for (const auto& [root_id, sites] : component_sites) {
        if (static_cast<int>(sites.size()) < min_grain_size) {
            continue;
        }

        double sum_cos_x = 0.0, sum_sin_x = 0.0;
        double sum_cos_y = 0.0, sum_sin_y = 0.0;
        int ce_count = 0;

        for (int idx : sites) {
            int y = idx / L;
            int x = idx % L;
            if (grid.get_element(idx) == 1) {
                ce_count++;
            }

            double theta_x = x * scale;
            sum_cos_x += std::cos(theta_x);
            sum_sin_x += std::sin(theta_x);

            double theta_y = y * scale;
            sum_cos_y += std::cos(theta_y);
            sum_sin_y += std::sin(theta_y);
        }

        double mean_theta_x = std::atan2(sum_sin_x, sum_cos_x);
        if (mean_theta_x < 0.0) mean_theta_x += TWO_PI;
        double cx = (mean_theta_x / TWO_PI) * L;
        if (cx >= L) cx -= L;

        double mean_theta_y = std::atan2(sum_sin_y, sum_cos_y);
        if (mean_theta_y < 0.0) mean_theta_y += TWO_PI;
        double cy = (mean_theta_y / TWO_PI) * L;
        if (cy >= L) cy -= L;

        double r_max = 0.0;
        for (int idx : sites) {
            int y = idx / L;
            int x = idx % L;

            double dx = std::abs(x - cx);
            if (dx > L * 0.5) dx = L - dx;

            double dy = std::abs(y - cy);
            if (dy > L * 0.5) dy = L - dy;

            double r = std::sqrt(dx * dx + dy * dy);
            if (r > r_max) r_max = r;

            samples.push_back({r, grid.get_element(idx)});
        }

        double r_eff = std::sqrt(static_cast<double>(sites.size()) / PI);
        total_r_eff += r_eff;

        GrainCentroid gc;
        gc.id = grain_id++;
        gc.orientation = grid.get_orientation(sites[0]);
        gc.cy = cy;
        gc.cx = cx;
        gc.num_sites = static_cast<int>(sites.size());
        gc.r_max = r_max;
        gc.r_eff = r_eff;
        gc.ce_count = ce_count;
        gc.ce_fraction = static_cast<double>(ce_count) / sites.size();

        grains.push_back(gc);
    }

    RadialProfileResult result;
    result.grains = grains;
    result.avg_grain_radius = grains.empty() ? 10.0 : (total_r_eff / grains.size());

    if (samples.empty()) {
        result.c_core = 0.0;
        result.c_peak = 0.0;
        result.shell_threshold = 0.0;
        result.r_inner = 0.0;
        result.r_outer = 0.0;
        result.shell_thickness = 0.0;
        result.enrichment_ratio = 1.0;
        return result;
    }

    // Determine maximum radial extent
    double max_r = 0.0;
    for (const auto& s : samples) {
        if (s.r > max_r) max_r = s.r;
    }

    int num_bins = static_cast<int>(std::ceil(max_r / bin_width)) + 1;
    std::vector<int> site_counts(num_bins, 0);
    std::vector<int> ce_counts(num_bins, 0);

    for (const auto& s : samples) {
        int b = static_cast<int>(s.r / bin_width);
        if (b >= 0 && b < num_bins) {
            site_counts[b]++;
            if (s.is_ce == 1) {
                ce_counts[b]++;
            }
        }
    }

    // Construct discrete radial bins
    result.profile.reserve(num_bins);
    for (int b = 0; b < num_bins; ++b) {
        if (site_counts[b] < 4) {
            continue; // Ignore sparsely sampled edge bins
        }

        RadialBin rb;
        rb.radial_distance = (b + 0.5) * bin_width;
        rb.sample_count = site_counts[b];
        rb.ce_concentration = static_cast<double>(ce_counts[b]) / site_counts[b];
        rb.std_error = std::sqrt(rb.ce_concentration * (1.0 - rb.ce_concentration) / rb.sample_count);
        rb.normalized_distance = rb.radial_distance / result.avg_grain_radius;

        result.profile.push_back(rb);
    }

    // Quantitative Shell Thickness Analysis
    if (result.profile.size() >= 2) {
        int max_sites = 0;
        for (const auto& rb : result.profile) {
            if (rb.sample_count > max_sites) max_sites = rb.sample_count;
        }
        int min_rep_sites = std::max(25, static_cast<int>(0.08 * max_sites));

        // Core concentration: minimum/average in the bulk region (r < 0.4 * R_avg)
        double c_min = 1.0;
        double c_max = 0.0;
        double r_peak = 0.0;
        int peak_bin = 0;

        for (size_t i = 0; i < result.profile.size(); ++i) {
            const auto& rb = result.profile[i];
            if (rb.radial_distance <= 0.4 * result.avg_grain_radius) {
                if (rb.ce_concentration < c_min) {
                    c_min = rb.ce_concentration;
                }
            }
            // Search for peak within representative grain boundary domain
            if (rb.sample_count >= min_rep_sites && rb.radial_distance <= 1.8 * result.avg_grain_radius) {
                if (rb.ce_concentration > c_max) {
                    c_max = rb.ce_concentration;
                    r_peak = rb.radial_distance;
                    peak_bin = static_cast<int>(i);
                }
            }
        }

        if (c_min > c_max) c_min = result.profile[0].ce_concentration;

        result.c_core = c_min;
        result.c_peak = c_max;
        result.enrichment_ratio = (c_min > 1e-6) ? (c_max / c_min) : 1.0;
        result.shell_threshold = c_min + 0.5 * (c_max - c_min);

        // Find inner shell radius by linear interpolation of threshold crossing
        double r_inner = 0.0;
        bool found_cross = false;
        for (int i = 0; i < peak_bin; ++i) {
            double c1 = result.profile[i].ce_concentration;
            double c2 = result.profile[i + 1].ce_concentration;
            if (c1 <= result.shell_threshold && c2 >= result.shell_threshold) {
                double r1 = result.profile[i].radial_distance;
                double r2 = result.profile[i + 1].radial_distance;
                double t = (result.shell_threshold - c1) / (c2 - c1 + 1e-12);
                r_inner = r1 + t * (r2 - r1);
                found_cross = true;
                break;
            }
        }

        if (!found_cross) {
            // Fallback: estimate from half of peak radius
            r_inner = std::max(0.0, r_peak * 0.7);
        }

        result.r_inner = r_inner;
        result.r_outer = r_peak;
        result.shell_thickness = std::max(0.0, r_peak - r_inner);
    } else {
        result.c_core = 0.0;
        result.c_peak = 0.0;
        result.shell_threshold = 0.0;
        result.r_inner = 0.0;
        result.r_outer = 0.0;
        result.shell_thickness = 0.0;
        result.enrichment_ratio = 1.0;
    }

    return result;
}

std::string resolve_output_file(const std::string& filename) {
    if (std::filesystem::exists("outputs")) {
        return "outputs/" + filename;
    }
    if (std::filesystem::exists("../outputs")) {
        return "../outputs/" + filename;
    }
    std::filesystem::create_directories("outputs");
    return "outputs/" + filename;
}

bool export_radial_profile_csv(
    const RadialProfileResult& result,
    const std::string& profile_csv_path,
    const std::string& centroids_csv_path
) {
    // 1. Export radial profile
    std::ofstream prof_file(profile_csv_path);
    if (!prof_file.is_open()) {
        std::cerr << "Failed to open " << profile_csv_path << " for writing!\n";
        return false;
    }

    prof_file << "radial_distance,ce_concentration,sample_count,std_error,normalized_distance\n";
    for (const auto& rb : result.profile) {
        prof_file << std::fixed << std::setprecision(4) << rb.radial_distance << ","
                  << std::setprecision(6) << rb.ce_concentration << ","
                  << rb.sample_count << ","
                  << std::setprecision(6) << rb.std_error << ","
                  << std::setprecision(4) << rb.normalized_distance << "\n";
    }
    prof_file.close();

    // Mirror to local outputs directory if written to ../outputs
    if (profile_csv_path.rfind("../outputs/", 0) == 0 && std::filesystem::exists("outputs")) {
        try {
            std::filesystem::copy_file(profile_csv_path, "outputs/radial_profile.csv",
                                       std::filesystem::copy_options::overwrite_existing);
        } catch (...) {}
    }

    // 2. Export grain centroids if requested
    if (!centroids_csv_path.empty()) {
        std::ofstream cent_file(centroids_csv_path);
        if (cent_file.is_open()) {
            cent_file << "grain_id,orientation,centroid_y,centroid_x,num_sites,r_max,r_eff,ce_count,ce_fraction\n";
            for (const auto& gc : result.grains) {
                cent_file << gc.id << ","
                          << gc.orientation << ","
                          << std::fixed << std::setprecision(3) << gc.cy << ","
                          << gc.cx << ","
                          << gc.num_sites << ","
                          << std::setprecision(2) << gc.r_max << ","
                          << gc.r_eff << ","
                          << gc.ce_count << ","
                          << std::setprecision(4) << gc.ce_fraction << "\n";
            }
            cent_file.close();

            if (centroids_csv_path.rfind("../outputs/", 0) == 0 && std::filesystem::exists("outputs")) {
                try {
                    std::filesystem::copy_file(centroids_csv_path, "outputs/grain_centroids.csv",
                                               std::filesystem::copy_options::overwrite_existing);
                } catch (...) {}
            }
        }
    }

    return true;
}

} // namespace potts
