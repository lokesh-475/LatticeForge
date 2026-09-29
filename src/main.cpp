#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "potts/grid.hpp"

struct Config {
    int size = 128;
    int q_states = 32;
    double temp = 1.0;
    double j_gb = 1.0;
    double h_seg = 1.5;
    double j_chem = 0.5;
    int seed = 42;
};

class LatticeForgeEngine {
private:
    Config config;
    potts::Grid grid;
    std::mt19937 rng;

    int get_gb_character(int y, int x) const {
        uint16_t g = grid.get_orientation(y, x);
        int diffs = 0;
        int n[4][2] = {{grid.up(y), x}, {grid.down(y), x}, {y, grid.left(x)}, {y, grid.right(x)}};
        for (auto& p : n) {
            if (grid.get_orientation(p[0], p[1]) != g) diffs++;
        }
        return diffs;
    }

public:
    LatticeForgeEngine(Config c) : config(c), grid(c.size), rng(c.seed) {
        std::uniform_int_distribution<int> q_dist(1, c.q_states);
        std::uniform_real_distribution<double> e_dist(0.0, 1.0);
        
        for (int y = 0; y < c.size; ++y) {
            for (int x = 0; x < c.size; ++x) {
                grid.set_orientation(y, x, static_cast<uint16_t>(q_dist(rng)));
                grid.set_element(y, x, e_dist(rng) < 0.3 ? 1 : 0); // 30% Ce
            }
        }
    }

    void potts_mc_step() {
        int L = config.size;
        std::uniform_int_distribution<int> xy_dist(0, L - 1);
        std::uniform_int_distribution<int> q_dist(1, config.q_states);
        
        for (int k = 0; k < L * L; ++k) {
            int y = xy_dist(rng), x = xy_dist(rng);
            uint16_t prop = static_cast<uint16_t>(q_dist(rng));
            uint16_t curr = grid.get_orientation(y, x);
            
            if (prop == curr) continue;
            
            int n[4][2] = {{grid.up(y), x}, {grid.down(y), x}, {y, grid.left(x)}, {y, grid.right(x)}};
            double e_curr = 0, e_prop = 0;
            
            for (auto& p : n) {
                uint16_t nid = grid.get_orientation(p[0], p[1]);
                if (nid != curr) e_curr += config.j_gb;
                if (nid != prop) e_prop += config.j_gb;
            }
            
            if (e_prop - e_curr <= 0) {
                grid.set_orientation(y, x, prop);
            }
        }
    }

    void kawasaki_mc_step() {
        int L = config.size;
        std::uniform_int_distribution<int> xy_dist(0, L - 1);
        std::uniform_int_distribution<int> d_dist(0, 3);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        
        for (int k = 0; k < L * L; ++k) {
            int y1 = xy_dist(rng), x1 = xy_dist(rng);
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
            
            int phi1 = get_gb_character(y1, x1);
            int phi2 = get_gb_character(y2, x2);
            
            double dE_seg = (c1 == 1 && c2 == 0) ? -config.h_seg * (phi2 - phi1) : -config.h_seg * (phi1 - phi2);
            
            // Simplified chemical energy for brevity
            double dE = dE_seg; 
            
            if (dE <= 0 || (config.temp > 0 && u_dist(rng) < std::exp(-dE / config.temp))) {
                grid.set_element(y1, x1, c2);
                grid.set_element(y2, x2, c1);
            }
        }
    }

    void run(int steps) {
        std::cout << "Starting simulation for " << steps << " steps...\n";
        for (int s = 0; s < steps; ++s) {
            potts_mc_step();
            kawasaki_mc_step();
            if (s % 10 == 0) std::cout << "Step " << s << " completed.\n";
        }
        std::cout << "Simulation finished.\n";
    }
};

int main() {
    Config cfg;
    cfg.size = 128; // Must be power of 2 for new Grid class
    cfg.temp = 0.4; // Low Ts
    
    LatticeForgeEngine engine(cfg);
    engine.run(50);
    
    return 0;
}
