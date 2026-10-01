#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "potts/grid.hpp"
#include "potts/mc_cpu.hpp"

class LatticeForgeEngine {
private:
    potts::Config config;
    potts::Grid grid;
    std::mt19937 rng;

public:
    LatticeForgeEngine(potts::Config c) : config(c), grid(c.size), rng(c.seed) {
        std::uniform_int_distribution<int> q_dist(1, c.q_states);
        std::uniform_real_distribution<double> e_dist(0.0, 1.0);
        
        for (int y = 0; y < c.size; ++y) {
            for (int x = 0; x < c.size; ++x) {
                grid.set_orientation(y, x, static_cast<uint16_t>(q_dist(rng)));
                grid.set_element(y, x, e_dist(rng) < 0.3 ? 1 : 0); // 30% Ce
            }
        }
    }

    void run(int steps) {
        std::cout << "Starting simulation for " << steps << " steps...\n";
        for (int s = 0; s < steps; ++s) {
            potts::potts_mc_step_cpu(grid, config, rng);
            potts::kawasaki_mc_step_cpu(grid, config, rng);
            if (s % 10 == 0) std::cout << "Step " << s << " completed.\n";
        }
        std::cout << "Simulation finished.\n";
    }
};

int main() {
    potts::Config cfg;
    cfg.size = 128; // Must be power of 2 for new Grid class
    cfg.temp = 0.4; // Low Ts
    
    LatticeForgeEngine engine(cfg);
    engine.run(50);
    
    return 0;
}
