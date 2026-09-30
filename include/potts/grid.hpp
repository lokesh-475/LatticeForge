#ifndef LATTICE_FORGE_GRID_HPP
#define LATTICE_FORGE_GRID_HPP

#include <cstdint>
#include <vector>
#include <stdexcept>

namespace potts {

struct Neighbors {
    int up, down, left, right;
};

class Grid {
private:
    int L_;
    int mask_; // For modulo-free periodic boundary conditions
    int shift_; // For bitwise multiplication replacement
    std::vector<uint16_t> orientations_;
    std::vector<uint8_t> elements_;
    std::vector<Neighbors> neighbors_;

    // Disable default construction if we want to enforce size
    Grid() = delete;

public:
    // Constructor requires L to be a power of 2
    explicit Grid(int L);

    // Continuous 1D flattened array access
    inline int idx(int y, int x) const {
        return ((y & mask_) << shift_) | (x & mask_);
    }
    
    // Efficient neighbor lookups
    inline int up(int y) const { return (y - 1) & mask_; }
    inline int down(int y) const { return (y + 1) & mask_; }
    inline int left(int x) const { return (x - 1) & mask_; }
    inline int right(int x) const { return (x + 1) & mask_; }

    inline const Neighbors& get_neighbors(int i) const {
        return neighbors_[i];
    }

    // Accessors
    uint16_t get_orientation(int y, int x) const { return orientations_[idx(y, x)]; }
    void set_orientation(int y, int x, uint16_t val) { orientations_[idx(y, x)] = val; }
    
    uint16_t get_orientation(int i) const { return orientations_[i]; }
    void set_orientation(int i, uint16_t val) { orientations_[i] = val; }
    
    uint8_t get_element(int y, int x) const { return elements_[idx(y, x)]; }
    void set_element(int y, int x, uint8_t val) { elements_[idx(y, x)] = val; }
    
    uint8_t get_element(int i) const { return elements_[i]; }
    void set_element(int i, uint8_t val) { elements_[i] = val; }
    
    // Raw pointer access for CUDA or fast iterators
    uint16_t* get_orientations_ptr() { return orientations_.data(); }
    const uint16_t* get_orientations_ptr() const { return orientations_.data(); }
    
    uint8_t* get_elements_ptr() { return elements_.data(); }
    const uint8_t* get_elements_ptr() const { return elements_.data(); }
    
    int size() const { return L_; }
    int total_sites() const { return L_ * L_; }
};

} // namespace potts

#endif // LATTICE_FORGE_GRID_HPP
