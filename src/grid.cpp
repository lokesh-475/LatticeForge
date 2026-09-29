#include "potts/grid.hpp"

namespace potts {

// Utility to check if a number is a power of 2
static bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

Grid::Grid(int L) : L_(L) {
    if (!is_power_of_two(L)) {
        throw std::invalid_argument("Grid size L must be a power of 2 for modulo-free bitwise boundary handling.");
    }
    mask_ = L - 1;
    
    int total = L * L;
    orientations_.resize(total, 0);
    elements_.resize(total, 0);
}

} // namespace potts
