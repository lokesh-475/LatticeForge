#ifndef LATTICE_FORGE_GRID_HPP
#define LATTICE_FORGE_GRID_HPP

#include <cstdint>
#include <vector>
#include <stdexcept>

#include <cstring>
#include <string>
#include <cuda_runtime.h>

namespace potts {

struct Neighbors {
    int up, down, left, right;
};

// RAII pinned/pageable host buffer for maximizing bus throughput
template <typename T>
class HostBuffer {
private:
    T* data_ = nullptr;
    size_t size_ = 0;
    bool pinned_ = false;

public:
    HostBuffer() = default;

    HostBuffer(size_t size, bool pinned = false) : size_(size), pinned_(pinned) {
        if (size_ == 0) return;
        if (pinned_) {
            cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaHostAlloc failed: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            data_ = new T[size_]();
        }
        std::memset(data_, 0, size_ * sizeof(T));
    }

    ~HostBuffer() {
        reset();
    }

    void reset() {
        if (data_) {
            if (pinned_) {
                cudaFreeHost(data_);
            } else {
                delete[] data_;
            }
            data_ = nullptr;
            size_ = 0;
            pinned_ = false;
        }
    }

    HostBuffer(HostBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_), pinned_(other.pinned_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.pinned_ = false;
    }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            pinned_ = other.pinned_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.pinned_ = false;
        }
        return *this;
    }

    HostBuffer(const HostBuffer& other) : size_(other.size_), pinned_(other.pinned_) {
        if (size_ > 0) {
            if (pinned_) {
                cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T), cudaHostAllocDefault);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaHostAlloc failed: " + std::string(cudaGetErrorString(err)));
                }
            } else {
                data_ = new T[size_]();
            }
            std::memcpy(data_, other.data_, size_ * sizeof(T));
        }
    }

    HostBuffer& operator=(const HostBuffer& other) {
        if (this != &other) {
            HostBuffer temp(other);
            *this = std::move(temp);
        }
        return *this;
    }

    inline T* data() noexcept { return data_; }
    inline const T* data() const noexcept { return data_; }
    inline size_t size() const noexcept { return size_; }
    inline bool is_pinned() const noexcept { return pinned_; }

    inline T& operator[](size_t idx) noexcept { return data_[idx]; }
    inline const T& operator[](size_t idx) const noexcept { return data_[idx]; }
};

class Grid {
private:
    int L_;
    int mask_; // For modulo-free periodic boundary conditions
    int shift_; // For bitwise multiplication replacement
    bool pinned_ = false;
    HostBuffer<uint16_t> orientations_;
    HostBuffer<uint8_t> elements_;
    std::vector<Neighbors> neighbors_;

    // Disable default construction if we want to enforce size
    Grid() = delete;

public:
    // Constructor requires L to be a power of 2; pinned enables cudaHostAlloc buffers
    explicit Grid(int L, bool pinned = false);

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
    bool is_pinned() const { return pinned_; }
};

} // namespace potts

#endif // LATTICE_FORGE_GRID_HPP
