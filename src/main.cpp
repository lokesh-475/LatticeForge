#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include "potts/radial_profile.hpp"

class LatticeForgeEngine {
private:
    potts::Config config;
    potts::Grid grid;
    std::mt19937 rng;

public:
    LatticeForgeEngine(potts::Config c) : config(c), grid(c.size), rng(c.seed) {
        // Initialize with realistic polycrystalline Voronoi grains
        potts::initialize_voronoi(grid, 36, 0.30, c.seed);
    }

    void run(int steps) {
        std::cout << "Starting simulation for " << steps << " steps...\n";
        for (int s = 0; s < steps; ++s) {
            potts::potts_mc_step_cpu(grid, config, rng);
            potts::kawasaki_mc_step_cpu(grid, config, rng);
            if (s % 10 == 0) std::cout << "Step " << s << " completed.\n";
        }
        std::cout << "Simulation finished.\n";

        // Compute grain centroids and radial profile analytics
        std::string radial_path = potts::resolve_output_file("radial_profile.csv");
        std::string centroids_path = potts::resolve_output_file("grain_centroids.csv");
        auto radial_res = potts::compute_radial_profile(grid, 0.5, 10);
        potts::export_radial_profile_csv(radial_res, radial_path, centroids_path);

        std::cout << "\n========================================================\n";
        std::cout << " Radial Concentration & Shell Thickness Analytics (CPU) \n";
        std::cout << "========================================================\n";
        std::cout << " Identified Grains     : " << radial_res.grains.size() << "\n";
        std::cout << " Mean Boundary Radius  : " << std::fixed << std::setprecision(2) << radial_res.avg_grain_radius << " lattice units\n";
        std::cout << " Bulk Core Conc. (C0)  : " << std::setprecision(4) << radial_res.c_core << "\n";
        std::cout << " Boundary Peak Conc.   : " << radial_res.c_peak << "\n";
        std::cout << " Enrichment (Peak/Core): " << std::setprecision(2) << radial_res.enrichment_ratio << "x\n";
        std::cout << " Shell Threshold (C50) : " << std::setprecision(4) << radial_res.shell_threshold << "\n";
        std::cout << " Inner Shell Radius    : " << std::setprecision(2) << radial_res.r_inner << " lattice units\n";
        std::cout << " Outer Shell Radius    : " << radial_res.r_outer << " lattice units\n";
        std::cout << " Quantitative Thickness: " << radial_res.shell_thickness << " lattice units\n";
        std::cout << " Exported Radial Profile to " << radial_path << "\n";
        std::cout << "========================================================\n";
    }
};

int main() {
    potts::Config cfg;
    cfg.size = 128; // Must be power of 2 for Grid class
    cfg.temp = 0.4; // Low Ts
    
    LatticeForgeEngine engine(cfg);
    engine.run(50);
    
    return 0;
}
