#include "potts/mc_cpu.hpp"
#include <cmath>
#include <cassert>

namespace potts {

int kawasaki_mc_step_cpu(Grid& grid, const Config& config, std::mt19937& rng, int steps) {
    int L = grid.size();
    int total_attempts = (steps == -1) ? (L * L) : steps;
    int accepted_swaps = 0;

    // Hard assertion for atomic conservation (Pre-check)
    int initial_ce_count = 0;
    for (int i = 0; i < grid.total_sites(); ++i) {
        if (grid.get_element(i) == 1) initial_ce_count++;
    }

    std::uniform_int_distribution<int> xy_dist(0, L - 1);
    std::uniform_int_distribution<int> d_dist(0, 3);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);

    for (int k = 0; k < total_attempts; ++k) {
        int y1 = xy_dist(rng);
        int x1 = xy_dist(rng);
        int d = d_dist(rng);
        
        int y2 = y1;
        int x2 = x1;
        
        if (d == 0) y2 = grid.up(y1);
        else if (d == 1) y2 = grid.down(y1);
        else if (d == 2) x2 = grid.left(x1);
        else if (d == 3) x2 = grid.right(x1);

        uint8_t c1 = grid.get_element(y1, x1);
        uint8_t c2 = grid.get_element(y2, x2);

        if (c1 == c2) continue;

        int phi1 = get_gb_character(grid, y1, x1);
        int phi2 = get_gb_character(grid, y2, x2);

        double delta_e_seg = 0.0;
        if (c1 == 1 && c2 == 0) {
            delta_e_seg = -config.h_seg * (phi2 - phi1);
        } else {
            delta_e_seg = -config.h_seg * (phi1 - phi2);
        }

        double delta_e = delta_e_seg;

        if (config.j_chem != 0.0) {
            int nbrs1_c = grid.get_element(grid.up(y1), x1) + grid.get_element(grid.down(y1), x1) +
                          grid.get_element(y1, grid.left(x1)) + grid.get_element(y1, grid.right(x1));
            int nbrs2_c = grid.get_element(grid.up(y2), x2) + grid.get_element(grid.down(y2), x2) +
                          grid.get_element(y2, grid.left(x2)) + grid.get_element(y2, grid.right(x2));

            int n1_ones = nbrs1_c - c2;
            int n2_ones = nbrs2_c - c1;

            double delta_e_chem = 0.0;
            if (c1 == 1 && c2 == 0) {
                delta_e_chem = config.j_chem * 2.0 * (static_cast<double>(n1_ones) - static_cast<double>(n2_ones));
            } else {
                delta_e_chem = config.j_chem * 2.0 * (static_cast<double>(n2_ones) - static_cast<double>(n1_ones));
            }
            delta_e += delta_e_chem;
        }

        if (delta_e <= 0.0 || (config.temp > 0.0 && u_dist(rng) < std::exp(-delta_e / config.temp))) {
            grid.set_element(y1, x1, c2);
            grid.set_element(y2, x2, c1);
            accepted_swaps++;
        }
    }

    // Hard assertion for atomic conservation (Post-check)
    int final_ce_count = 0;
    for (int i = 0; i < grid.total_sites(); ++i) {
        if (grid.get_element(i) == 1) final_ce_count++;
    }
    assert(initial_ce_count == final_ce_count && "Atomic conservation violation! Delta N_Ce != 0");

    return accepted_swaps;
}

} // namespace potts
