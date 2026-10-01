#ifndef LATTICE_FORGE_MC_CPU_HPP
#define LATTICE_FORGE_MC_CPU_HPP

#include "potts/grid.hpp"
#include <random>

namespace potts {

    struct Config {
        int size = 128;
        int q_states = 32;
        double temp = 1.0;
        double j_gb = 1.0;
        double h_seg = 1.5;
        double j_chem = 0.5;
        int seed = 42;
    };

    int get_gb_character(const Grid& grid, int y, int x);
    
    // Returns number of accepted flips
    int potts_mc_step_cpu(Grid& grid, const Config& config, std::mt19937& rng, int steps = -1);
    
    // Returns number of accepted swaps
    int kawasaki_mc_step_cpu(Grid& grid, const Config& config, std::mt19937& rng, int steps = -1);

}

#endif // LATTICE_FORGE_MC_CPU_HPP
