#ifndef LATTICE_FORGE_RADIAL_PROFILE_HPP
#define LATTICE_FORGE_RADIAL_PROFILE_HPP

#include "potts/grid.hpp"
#include <vector>
#include <string>

namespace potts {

// Disjoint Set Union (DSU) with path compression and rank union for 2D toroidal CCL
class DSU {
public:
    std::vector<int> parent;
    std::vector<int> rank;

    explicit DSU(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int i) {
        int root = i;
        while (root != parent[root]) {
            root = parent[root];
        }
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (rank[root_i] < rank[root_j]) {
                parent[root_i] = root_j;
            } else if (rank[root_i] > rank[root_j]) {
                parent[root_j] = root_i;
            } else {
                parent[root_j] = root_i;
                rank[root_i]++;
            }
        }
    }
};

// Represents an identified grain and its topological centroid on the 2D periodic torus
struct GrainCentroid {
    int id;
    uint16_t orientation;
    double cy;            // Centroid y-coordinate [0, L)
    double cx;            // Centroid x-coordinate [0, L)
    int num_sites;        // Total sites in grain (area)
    double r_max;         // Maximum radial distance to grain boundary
    double r_eff;         // Equivalent circular radius sqrt(Area / pi)
    int ce_count;         // Total Ce atoms in grain
    double ce_fraction;   // Average Ce atomic fraction in grain
};

// Discrete radial bin data for concentration profile
struct RadialBin {
    double radial_distance;      // Radial distance from centroid (lattice units)
    double ce_concentration;     // Average Ce atomic fraction in bin
    int sample_count;            // Number of lattice sites in bin
    double std_error;            // Statistical standard error
    double normalized_distance;  // r / R_boundary (normalized radial distance)
};

// Comprehensive radial profile and quantitative shell thickness analysis result
struct RadialProfileResult {
    std::vector<GrainCentroid> grains;
    std::vector<RadialBin> profile;
    double avg_grain_radius;     // Mean boundary radius across grains
    double c_core;               // Core/bulk Ce concentration
    double c_peak;               // Peak Ce concentration at boundary shell
    double shell_threshold;      // C50 threshold: C_core + 0.5 * (C_peak - C_core)
    double r_inner;              // Inner shell boundary radius (crossing of C50)
    double r_outer;              // Outer shell boundary radius (peak position)
    double shell_thickness;      // Quantitative shell thickness delta = r_outer - r_inner
    double enrichment_ratio;     // C_peak / C_core
};

// Computes grain centroids using circular mean under periodic boundary conditions
std::vector<GrainCentroid> compute_grain_centroids(const Grid& grid, int min_grain_size = 10);

// Computes the radial Ce concentration profile and quantitative shell thickness across all grains
RadialProfileResult compute_radial_profile(
    const Grid& grid,
    double bin_width = 0.5,
    int min_grain_size = 10
);

// Resolves file output path in 'outputs/' or '../outputs/'
std::string resolve_output_file(const std::string& filename);

// Exports the radial profile and optionally grain centroids to CSV
bool export_radial_profile_csv(
    const RadialProfileResult& result,
    const std::string& profile_csv_path,
    const std::string& centroids_csv_path = ""
);

} // namespace potts

#endif // LATTICE_FORGE_RADIAL_PROFILE_HPP
