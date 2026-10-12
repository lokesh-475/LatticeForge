#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

namespace potts {

// Utility to check if a number is a power of 2
static bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

static int get_shift(int x) {
    int shift = 0;
    while (x > 1) {
        x >>= 1;
        shift++;
    }
    return shift;
}

Grid::Grid(int L, bool pinned) : L_(L), pinned_(pinned) {
    if (!is_power_of_two(L)) {
        throw std::invalid_argument("Grid size L must be a power of 2 for modulo-free bitwise boundary handling.");
    }
    mask_ = L - 1;
    shift_ = get_shift(L);
    
    int total = L * L;
    orientations_ = HostBuffer<uint16_t>(total, pinned);
    elements_ = HostBuffer<uint8_t>(total, pinned);
    neighbors_.resize(total);

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int i = idx(y, x);
            neighbors_[i].up = idx(up(y), x);
            neighbors_[i].down = idx(down(y), x);
            neighbors_[i].left = idx(y, left(x));
            neighbors_[i].right = idx(y, right(x));
        }
    }
}

void initialize_voronoi(Grid& grid, int num_grains, double fraction_ce, int seed) {
    int L = grid.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_dist(0.0, static_cast<double>(L));
    std::uniform_int_distribution<int> q_dist(1, 32);

    struct Seed {
        double y, x;
        uint16_t q;
        bool is_ce;
    };

    std::vector<Seed> seeds(num_grains);
    int num_ce = static_cast<int>(std::round(num_grains * fraction_ce));
    for (int i = 0; i < num_grains; ++i) {
        seeds[i] = { pos_dist(rng), pos_dist(rng), static_cast<uint16_t>(q_dist(rng)), (i < num_ce) };
    }
    std::shuffle(seeds.begin(), seeds.end(), rng);

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            double min_d2 = 1e18;
            int best = 0;
            for (int i = 0; i < num_grains; ++i) {
                double dy = std::abs(static_cast<double>(y) - seeds[i].y);
                if (dy > static_cast<double>(L) - dy) dy = static_cast<double>(L) - dy;
                double dx = std::abs(static_cast<double>(x) - seeds[i].x);
                if (dx > static_cast<double>(L) - dx) dx = static_cast<double>(L) - dx;
                double d2 = dy * dy + dx * dx;
                if (d2 < min_d2) {
                    min_d2 = d2;
                    best = i;
                }
            }
            grid.set_orientation(y, x, seeds[best].q);
            grid.set_element(y, x, seeds[best].is_ce ? 1 : 0);
        }
    }
}

double compute_total_energy_cpu(const Grid& grid, const Config& config) {
    int L = grid.size();
    double h_potts = 0.0;
    double h_seg = 0.0;
    double h_chem = 0.0;

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int i = grid.idx(y, x);
            uint16_t g = grid.get_orientation(i);
            uint8_t c = grid.get_element(i);

            int up = grid.idx(grid.up(y), x);
            int down = grid.idx(grid.down(y), x);
            int left = grid.idx(y, grid.left(x));
            int right = grid.idx(y, grid.right(x));

            int phi = (grid.get_orientation(up) != g) +
                      (grid.get_orientation(down) != g) +
                      (grid.get_orientation(left) != g) +
                      (grid.get_orientation(right) != g);

            int unlike = (grid.get_element(up) != c) +
                         (grid.get_element(down) != c) +
                         (grid.get_element(left) != c) +
                         (grid.get_element(right) != c);

            h_potts += 0.5 * config.j_gb * phi;
            h_seg += -config.h_seg * c * phi;
            h_chem += 0.5 * config.j_chem * unlike;
        }
    }
    return h_potts + h_seg + h_chem;
}

} // namespace potts

