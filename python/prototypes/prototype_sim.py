"""
LatticeForge - High-Throughput Microstructure Simulation Engine
Week 1 Prototype: Coupled 2D Q-State Potts & Kawasaki Atomic Diffusion Dynamics

Physics Reference:
    Ma et al., "Microstructure evolution and magnetic properties of Multi-Main-Phase
    Nd-Ce-Fe-B permanent magnets," Journal of Alloys and Compounds (2025).

Core Formulation:
    1. Potts Model (Curvature-driven Grain Coarsening):
       H_potts = (J_gb / 2) * sum_{<i, j>} (1 - delta_{sigma_i, sigma_j})
    2. Kawasaki Dynamics (Conserved Solute Segregation):
       H_chem = -H_seg * sum_i [ c_i * phi_gb(i) ]
       where:
         sigma_i in {1, ..., Q}  (Crystallographic grain orientation)
         c_i in {0, 1}           (0 = Nd-rich core, 1 = Ce/La-rich solute)
         phi_gb(i)               (Local grain boundary character / coordination disorder)
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    size: int = 128
    q_states: int = 32
    temp: float = 1.0
    j_gb: float = 1.0
    h_seg: float = 1.5
    j_chem: float = 0.5  # Chemical mixing energy parameter
    seed: int = 42


class LatticeForgePrototype:
    """
    2D Coupled Potts-Kawasaki Simulation Prototype on a discrete square lattice
    with Periodic Boundary Conditions (PBC).
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.size = self.config.size
        self.q_states = self.config.q_states
        self.temp = self.config.temp
        self.j_gb = self.config.j_gb
        self.h_seg = self.config.h_seg
        self.j_chem = getattr(self.config, 'j_chem', 0.0)
        
        self.rng = np.random.default_rng(self.config.seed)
        
        # Dual-grid arrays
        # Orientation field: sigma in [1, Q]
        self.grain_id = np.zeros((self.size, self.size), dtype=np.int32)
        # Chemical element field: c in {0, 1} (0: Nd-rich core, 1: Ce-rich shell/solute)
        self.element_type = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Tracking metrics
        self.initial_ce_count: int = 0
        self.step_count: int = 0

    def initialize_voronoi(self, num_grains: int = 36, fraction_ce: float = 0.30) -> None:
        """
        Initializes grain morphology using a 2D Voronoi tessellation under
        Periodic Boundary Conditions (Minimum Image Convention).
        Assigns initial Ce/La concentration to a subset of grains / boundary phases.
        """
        L = self.size
        seed_coords = self.rng.uniform(0, L, size=(num_grains, 2))
        seed_orientations = self.rng.integers(1, self.q_states + 1, size=num_grains, dtype=np.int32)
        
        # Create coordinate grid
        y_grid, x_grid = np.indices((L, L))
        
        # Compute nearest Voronoi seed using Periodic Boundary Conditions (Minimum Image Convention)
        # Vectorized over all seeds for high performance
        min_dist_sq = np.full((L, L), np.inf, dtype=np.float64)
        closest_seed_idx = np.zeros((L, L), dtype=np.int32)
        
        for idx, (sy, sx) in enumerate(seed_coords):
            dy = np.abs(y_grid - sy)
            dy = np.minimum(dy, L - dy)
            dx = np.abs(x_grid - sx)
            dx = np.minimum(dx, L - dx)
            dist_sq = dy * dy + dx * dx
            
            mask = dist_sq < min_dist_sq
            min_dist_sq[mask] = dist_sq[mask]
            closest_seed_idx[mask] = idx
            
        self.grain_id = seed_orientations[closest_seed_idx]
        
        # Multi-Main-Phase (MMP) solute seeding:
        # A subset of original powder grains are Ce-rich (as in blended powder sintering)
        unique_seeds = np.unique(closest_seed_idx)
        num_ce_seeds = int(np.round(len(unique_seeds) * fraction_ce))
        ce_seeds = self.rng.choice(unique_seeds, size=num_ce_seeds, replace=False)
        
        self.element_type.fill(0)
        for s_idx in ce_seeds:
            self.element_type[closest_seed_idx == s_idx] = 1
            
        self.initial_ce_count = int(np.sum(self.element_type))
        self.step_count = 0

    def get_neighbors_pbc(self, y: int, x: int) -> List[Tuple[int, int]]:
        """Returns 4 nearest neighbors (Von Neumann stencil) under PBC."""
        L = self.size
        return [
            ((y - 1) % L, x),
            ((y + 1) % L, x),
            (y, (x - 1) % L),
            (y, (x + 1) % L),
        ]

    def local_gb_character(self, y: int, x: int) -> int:
        """
        Quantifies local grain boundary character / coordination disorder at (y, x):
        phi_gb(i) = sum_{j in N(i)} (1 - delta_{sigma_i, sigma_j})
        """
        gid = self.grain_id[y, x]
        L = self.size
        nbrs = [
            self.grain_id[(y - 1) % L, x],
            self.grain_id[(y + 1) % L, x],
            self.grain_id[y, (x - 1) % L],
            self.grain_id[y, (x + 1) % L],
        ]
        return sum(gid != nid for nid in nbrs)

    def potts_mc_step(self, steps: Optional[int] = None) -> int:
        """
        Performs Monte Carlo attempts for grain boundary curvature minimization
        via Q-state Potts transitions.
        
        Returns:
            Number of accepted orientation flips.
        """
        L = self.size
        total_attempts = steps if steps is not None else (L * L)
        accepted_flips = 0
        
        ys = self.rng.integers(0, L, size=total_attempts)
        xs = self.rng.integers(0, L, size=total_attempts)
        proposed_states = self.rng.integers(1, self.q_states + 1, size=total_attempts)
        random_uniforms = self.rng.random(size=total_attempts)
        
        for k in range(total_attempts):
            y, x = ys[k], xs[k]
            current_sigma = self.grain_id[y, x]
            proposed_sigma = proposed_states[k]
            
            if current_sigma == proposed_sigma:
                continue
                
            nbrs = [
                self.grain_id[(y - 1) % L, x],
                self.grain_id[(y + 1) % L, x],
                self.grain_id[y, (x - 1) % L],
                self.grain_id[y, (x + 1) % L],
            ]
            
            e_current = sum(self.j_gb * (current_sigma != nid) for nid in nbrs)
            e_proposed = sum(self.j_gb * (proposed_sigma != nid) for nid in nbrs)
            delta_e = e_proposed - e_current
            
            # Metropolis-Hastings acceptance for grain growth (typically T=0 limit for curvature-driven)
            if delta_e <= 0.0:
                self.grain_id[y, x] = proposed_sigma
                accepted_flips += 1
                
        return accepted_flips

    def kawasaki_mc_step(self, steps: Optional[int] = None) -> int:
        """
        Performs Kawasaki pairwise atomic exchange attempts.
        Conserves exact atomic counts while modeling solute segregation to GBs.
        
        Energy formulation:
            H_chem = -H_seg * sum_i [ c_i * phi_gb(i) ]
        When swapping site 1 (y1, x1) and site 2 (y2, x2):
            Delta E_chem = -H_seg * [ (c1_new - c1_old)*phi_gb(1) + (c2_new - c2_old)*phi_gb(2) ]
        
        Returns:
            Number of accepted atomic swaps.
        """
        L = self.size
        total_attempts = steps if steps is not None else (L * L)
        accepted_swaps = 0
        
        y1s = self.rng.integers(0, L, size=total_attempts)
        x1s = self.rng.integers(0, L, size=total_attempts)
        dir_choices = self.rng.integers(0, 4, size=total_attempts)
        random_uniforms = self.rng.random(size=total_attempts)
        
        # Direction offsets: (dy, dx)
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for k in range(total_attempts):
            y1, x1 = y1s[k], x1s[k]
            dy, dx = deltas[dir_choices[k]]
            y2 = (y1 + dy) % L
            x2 = (x1 + dx) % L
            
            c1 = self.element_type[y1, x1]
            c2 = self.element_type[y2, x2]
            
            # Swapping identical chemical species produces zero net change
            if c1 == c2:
                continue
                
            phi1 = self.local_gb_character(y1, x1)
            phi2 = self.local_gb_character(y2, x2)
            
            # Segregation energy change
            if c1 == 1 and c2 == 0:
                delta_e_seg = -self.h_seg * (phi2 - phi1)
            else:
                # c1=0 and c2=1: c1 becomes 1, c2 becomes 0
                delta_e_seg = -self.h_seg * (phi1 - phi2)
                
            delta_e = delta_e_seg
            
            # Add chemical mixing energy if supported
            if self.j_chem != 0.0:
                # Count '1' neighbors (excluding the swapped sites themselves)
                nbrs1_c = self.element_type[(y1 - 1) % L, x1] + self.element_type[(y1 + 1) % L, x1] + \
                          self.element_type[y1, (x1 - 1) % L] + self.element_type[y1, (x1 + 1) % L]
                nbrs2_c = self.element_type[(y2 - 1) % L, x2] + self.element_type[(y2 + 1) % L, x2] + \
                          self.element_type[y2, (x2 - 1) % L] + self.element_type[y2, (x2 + 1) % L]
                
                # They are guaranteed to be neighbors due to dy, dx selection
                # Subtract their contribution to each other's neighbor count
                n1_ones = nbrs1_c - c2
                n2_ones = nbrs2_c - c1
                
                # Mixing energy: E = -J_chem * sum (c_i == c_j) -> positive J favors clustering
                # Wait, usually mixing energy is J_chem * sum (c_i != c_j)
                # Let's use E = - J_chem * sum(c_i == c_j). 
                # If site i becomes 1 instead of 0, the change in number of '1' neighbors is + (n_ones) and '0' neighbors is - (n_zeros)
                # Let's simply count the change in like-neighbors.
                # If c1 changes from 1 to 0, we lose `n1_ones` like-bonds, and gain `4 - 1 - n1_ones = 3 - n1_ones` like-bonds (0-0).
                # To make it simple: E = J_chem * (number of unlike bonds)
                # Currently: unlike bonds = (c1=1, so 3-n1_ones are 0) + (c2=0, so n2_ones are 1)
                # After swap: c1=0 (so n1_ones are 1), c2=1 (so 3-n2_ones are 0)
                # Delta unlike bonds = (n1_ones + 3 - n2_ones) - (3 - n1_ones + n2_ones) 
                #                    = 2 * n1_ones - 2 * n2_ones
                # So if c1=1, c2=0: Delta E_chem = J_chem * 2 * (n1_ones - n2_ones)
                
                if c1 == 1 and c2 == 0:
                    delta_e_chem = self.j_chem * 2.0 * (float(n1_ones) - float(n2_ones))
                else:
                    delta_e_chem = self.j_chem * 2.0 * (float(n2_ones) - float(n1_ones))
                
                delta_e += delta_e_chem

            # Metropolis-Hastings acceptance
            if delta_e <= 0.0 or (self.temp > 0.0 and random_uniforms[k] < np.exp(-delta_e / self.temp)):
                self.element_type[y1, x1] = c2
                self.element_type[y2, x2] = c1
                accepted_swaps += 1
                
        # Verification of strict atom conservation
        current_ce_count = int(np.sum(self.element_type))
        assert current_ce_count == self.initial_ce_count, (
            f"Atomic conservation violation! Expected {self.initial_ce_count}, "
            f"got {current_ce_count} (Delta={current_ce_count - self.initial_ce_count})"
        )
        
        return accepted_swaps

    def step(self, potts_mc_sweeps: int = 1, kawasaki_mc_sweeps: int = 1) -> Dict[str, int]:
        """Executes coupled Monte Carlo steps (Potts + Kawasaki)."""
        flips = 0
        for _ in range(potts_mc_sweeps):
            flips += self.potts_mc_step()
            
        swaps = 0
        for _ in range(kawasaki_mc_sweeps):
            swaps += self.kawasaki_mc_step()
            
        self.step_count += 1
        return {"flips": flips, "swaps": swaps}

    def compute_metrics(self) -> Dict[str, float]:
        """
        Computes physical metrics:
            - gb_fraction: Fraction of lattice sites exhibiting boundary character
            - ce_at_gb_ratio: Fraction of total Ce atoms residing at grain boundaries
            - segregation_enrichment: Concentration of Ce at GB vs bulk interior
        """
        L = self.size
        # Vectorized GB character computation
        north = np.roll(self.grain_id, -1, axis=0)
        south = np.roll(self.grain_id, 1, axis=0)
        east = np.roll(self.grain_id, -1, axis=1)
        west = np.roll(self.grain_id, 1, axis=1)
        
        gb_map = (
            (self.grain_id != north).astype(np.int32)
            + (self.grain_id != south).astype(np.int32)
            + (self.grain_id != east).astype(np.int32)
            + (self.grain_id != west).astype(np.int32)
        )
        
        is_gb = (gb_map > 0)
        total_sites = L * L
        total_gb_sites = np.sum(is_gb)
        total_bulk_sites = total_sites - total_gb_sites
        
        is_ce = (self.element_type == 1)
        ce_at_gb = np.sum(is_ce & is_gb)
        ce_at_bulk = np.sum(is_ce & (~is_gb))
        total_ce = np.sum(is_ce)
        
        c_gb = ce_at_gb / max(1, total_gb_sites)
        c_bulk = ce_at_bulk / max(1, total_bulk_sites)
        enrichment = c_gb / max(1e-6, c_bulk)
        
        return {
            "gb_fraction": float(total_gb_sites / total_sites),
            "ce_at_gb_ratio": float(ce_at_gb / max(1, total_ce)),
            "c_gb": float(c_gb),
            "c_bulk": float(c_bulk),
            "enrichment_factor": float(enrichment),
        }


def run_prototype_simulation(
    temp: float,
    steps: int = 35,
    size: int = 128,
    seed: int = 42
) -> Tuple[LatticeForgePrototype, List[Dict[str, float]]]:
    """Runs a complete prototype simulation and collects metrics over time."""
    config = SimulationConfig(size=size, temp=temp, seed=seed)
    sim = LatticeForgePrototype(config)
    sim.initialize_voronoi(num_grains=36, fraction_ce=0.30)
    
    history = [sim.compute_metrics()]
    for _ in range(steps):
        sim.step(potts_mc_sweeps=1, kawasaki_mc_sweeps=1)
        history.append(sim.compute_metrics())
        
    return sim, history


if __name__ == "__main__":
    print("=================================================================")
    print(" LatticeForge Week 1 Prototype: Potts + Kawasaki Dual-Grid Engine")
    print("=================================================================")
    
    # 1. Low Temperature (Controlled Segregation)
    print("\n[1/2] Simulating Low Temperature Sintering (Ts = 0.4)...")
    sim_low, hist_low = run_prototype_simulation(temp=0.4, steps=30, size=128, seed=101)
    m_low = sim_low.compute_metrics()
    print(f" -> Ce Partitioned to GB: {m_low['ce_at_gb_ratio']*100:.2f}%")
    print(f" -> GB/Bulk Enrichment Factor: {m_low['enrichment_factor']:.2f}x")
    
    # 2. High Temperature (Excessive Interdiffusion / Homogenization)
    print("\n[2/2] Simulating High Temperature Sintering (Ts = 2.8)...")
    sim_high, hist_high = run_prototype_simulation(temp=2.8, steps=30, size=128, seed=101)
    m_high = sim_high.compute_metrics()
    print(f" -> Ce Partitioned to GB: {m_high['ce_at_gb_ratio']*100:.2f}%")
    print(f" -> GB/Bulk Enrichment Factor: {m_high['enrichment_factor']:.2f}x")
    
    print("\nVerification check:")
    print(f" Low Ts Enrichment ({m_low['enrichment_factor']:.2f}x) > High Ts Enrichment ({m_high['enrichment_factor']:.2f}x): "
          f"{m_low['enrichment_factor'] > m_high['enrichment_factor']}")
    print("All atomic conservation assertions verified successfully.")
    
    # Ensure verification strictly evaluates to True
    assert m_low['enrichment_factor'] > m_high['enrichment_factor'], "Verification failed!"
    
    # Generate Output Visuals
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Low Ts
    ax = axes[0]
    ax.imshow(sim_low.grain_id, cmap='tab20b', alpha=0.5)
    ce_mask_low = np.ma.masked_where(sim_low.element_type == 0, sim_low.element_type)
    ax.imshow(ce_mask_low, cmap='autumn', interpolation='nearest', alpha=1.0)
    ax.set_title("Low Ts = 0.4\nCore-Shell Segregation")
    ax.axis('off')
    
    # Panel 2: High Ts
    ax = axes[1]
    ax.imshow(sim_high.grain_id, cmap='tab20b', alpha=0.5)
    ce_mask_high = np.ma.masked_where(sim_high.element_type == 0, sim_high.element_type)
    ax.imshow(ce_mask_high, cmap='autumn', interpolation='nearest', alpha=1.0)
    ax.set_title("High Ts = 2.8\nThermally Mixed Solid Solution")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/validation_coreshell.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nRendered outputs/validation_coreshell.png")
