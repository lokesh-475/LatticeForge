#include "potts/grid.hpp"

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

Grid::Grid(int L) : L_(L) {
    if (!is_power_of_two(L)) {
        throw std::invalid_argument("Grid size L must be a power of 2 for modulo-free bitwise boundary handling.");
    }
    mask_ = L - 1;
    shift_ = get_shift(L);
    
    int total = L * L;
    orientations_.resize(total, 0);
    elements_.resize(total, 0);
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

} // namespace potts
