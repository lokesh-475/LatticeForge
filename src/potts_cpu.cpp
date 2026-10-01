#include "potts/mc_cpu.hpp"
#include <vector>

namespace potts {

int get_gb_character(const Grid& grid, int y, int x) {
    uint16_t g = grid.get_orientation(y, x);
    int diffs = 0;
    if (grid.get_orientation(grid.up(y), x) != g) diffs++;
    if (grid.get_orientation(grid.down(y), x) != g) diffs++;
    if (grid.get_orientation(y, grid.left(x)) != g) diffs++;
    if (grid.get_orientation(y, grid.right(x)) != g) diffs++;
    return diffs;
}

int potts_mc_step_cpu(Grid& grid, const Config& config, std::mt19937& rng, int steps) {
    int L = grid.size();
    int total_attempts = (steps == -1) ? (L * L) : steps;
    int accepted_flips = 0;

    std::uniform_int_distribution<int> xy_dist(0, L - 1);
    std::uniform_int_distribution<int> q_dist(1, config.q_states);

    for (int k = 0; k < total_attempts; ++k) {
        int y = xy_dist(rng);
        int x = xy_dist(rng);
        uint16_t curr = grid.get_orientation(y, x);
        uint16_t prop = static_cast<uint16_t>(q_dist(rng));

        if (curr == prop) continue;

        double e_curr = 0, e_prop = 0;
        
        uint16_t n_up = grid.get_orientation(grid.up(y), x);
        uint16_t n_down = grid.get_orientation(grid.down(y), x);
        uint16_t n_left = grid.get_orientation(y, grid.left(x));
        uint16_t n_right = grid.get_orientation(y, grid.right(x));

        if (n_up != curr) e_curr += config.j_gb;
        if (n_down != curr) e_curr += config.j_gb;
        if (n_left != curr) e_curr += config.j_gb;
        if (n_right != curr) e_curr += config.j_gb;

        if (n_up != prop) e_prop += config.j_gb;
        if (n_down != prop) e_prop += config.j_gb;
        if (n_left != prop) e_prop += config.j_gb;
        if (n_right != prop) e_prop += config.j_gb;

        if (e_prop - e_curr <= 0) {
            grid.set_orientation(y, x, prop);
            accepted_flips++;
        }
    }
    return accepted_flips;
}

} // namespace potts
