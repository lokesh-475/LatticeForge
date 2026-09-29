#include <iostream>
#include <vector>
#include <random>
#include <cmath>

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
    std::vector<int> grain_id;
    std::vector<uint8_t> element_type;
    std::mt19937 rng;

    int idx(int y, int x) const { return y * config.size + x; }

    int get_gb_character(int y, int x) const {
        int g = grain_id[idx(y, x)];
        int L = config.size;
        int diffs = 0;
        int n[4][2] = {{(y-1+L)%L, x}, {(y+1)%L, x}, {y, (x-1+L)%L}, {y, (x+1)%L}};
        for (auto& p : n) {
            if (grain_id[idx(p[0], p[1])] != g) diffs++;
        }
        return diffs;
    }

public:
    LatticeForgeEngine(Config c) : config(c), rng(c.seed) {
        int total = c.size * c.size;
        grain_id.resize(total);
        element_type.resize(total);
        
        std::uniform_int_distribution<int> q_dist(1, c.q_states);
        std::uniform_real_distribution<double> e_dist(0.0, 1.0);
        
        for (int i = 0; i < total; ++i) {
            grain_id[i] = q_dist(rng);
            element_type[i] = e_dist(rng) < 0.3 ? 1 : 0; // 30% Ce
        }
    }

    void potts_mc_step() {
        int L = config.size;
        std::uniform_int_distribution<int> xy_dist(0, L - 1);
        std::uniform_int_distribution<int> q_dist(1, config.q_states);
        
        for (int k = 0; k < L * L; ++k) {
            int y = xy_dist(rng), x = xy_dist(rng);
            int prop = q_dist(rng);
            int curr = grain_id[idx(y, x)];
            
            if (prop == curr) continue;
            
            int n[4][2] = {{(y-1+L)%L, x}, {(y+1)%L, x}, {y, (x-1+L)%L}, {y, (x+1)%L}};
            double e_curr = 0, e_prop = 0;
            
            for (auto& p : n) {
                int nid = grain_id[idx(p[0], p[1])];
                if (nid != curr) e_curr += config.j_gb;
                if (nid != prop) e_prop += config.j_gb;
            }
            
            if (e_prop - e_curr <= 0) {
                grain_id[idx(y, x)] = prop;
            }
        }
    }

    void kawasaki_mc_step() {
        int L = config.size;
        std::uniform_int_distribution<int> xy_dist(0, L - 1);
        std::uniform_int_distribution<int> d_dist(0, 3);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        
        int deltas[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        
        for (int k = 0; k < L * L; ++k) {
            int y1 = xy_dist(rng), x1 = xy_dist(rng);
            int d = d_dist(rng);
            int y2 = (y1 + deltas[d][0] + L) % L;
            int x2 = (x1 + deltas[d][1] + L) % L;
            
            uint8_t c1 = element_type[idx(y1, x1)];
            uint8_t c2 = element_type[idx(y2, x2)];
            
            if (c1 == c2) continue;
            
            int phi1 = get_gb_character(y1, x1);
            int phi2 = get_gb_character(y2, x2);
            
            double dE_seg = (c1 == 1 && c2 == 0) ? -config.h_seg * (phi2 - phi1) : -config.h_seg * (phi1 - phi2);
            
            // Simplified chemical energy for brevity
            double dE = dE_seg; 
            
            if (dE <= 0 || (config.temp > 0 && u_dist(rng) < std::exp(-dE / config.temp))) {
                element_type[idx(y1, x1)] = c2;
                element_type[idx(y2, x2)] = c1;
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
    cfg.size = 128;
    cfg.temp = 0.4; // Low Ts
    
    LatticeForgeEngine engine(cfg);
    engine.run(50);
    
    return 0;
}
