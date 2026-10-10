import os
import sys
import matplotlib.pyplot as plt

# Add the 'python' directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_dir = os.path.join(root_dir, 'python')
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from prototypes.prototype_sim import LatticeForgePrototype, SimulationConfig

def main():
    # Setup
    config = SimulationConfig(size=128, temp=0.5, seed=123)
    sim = LatticeForgePrototype(config)
    sim.initialize_voronoi(num_grains=36, fraction_ce=0.30)
    
    # Trackers
    steps = 150
    energies = []
    mcs = []
    
    print(f"Running kinetics simulation for {steps} steps (Ts={config.temp})...")
    for i in range(steps + 1):
        # Compute energy
        energy = sim.compute_total_energy()
        energies.append(energy)
        mcs.append(i)
        
        if i % 10 == 0:
            print(f"MCS {i:03d} | Total Energy: {energy:.2f}")
            
        if i < steps:
            sim.step(potts_mc_sweeps=1, kawasaki_mc_sweeps=1)
            
    # Plotting
    out_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'energy_vs_mcs.png')
    
    plt.figure(figsize=(8, 5))
    plt.plot(mcs, energies, 'b-', linewidth=2, label='Total Hamiltonian')
    plt.title("Thermodynamic Relaxation: Total Hamiltonian vs MCS")
    plt.xlabel("Monte Carlo Steps (MCS)")
    plt.ylabel(r"Total Hamiltonian $\mathcal{H}$")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Energy kinetics plot saved to {out_path}")

if __name__ == "__main__":
    main()
