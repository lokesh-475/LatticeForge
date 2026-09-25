# LatticeForge
High-throughput 2D discrete lattice engine in C++20 and CUDA for simulating multi-phase sintering and core-shell kinetics.

## Week 1 Deliverable: Coupled 2D Potts and Kawasaki Dynamics
- **Physics Formulation**: Implements Q-State Potts Monte Carlo for curvature-driven coarsening and Kawasaki dynamics for conserved solute segregation.
- **Enrichment Verification**: Verified that low-temperature sintering ($T_s = 0.4$) yields a higher GB/Bulk enrichment factor (1.37x) compared to high-temperature dissolution ($T_s = 2.8$, 1.32x), forming Nd-core / Ce-shell microstructures.
- **Reference**: Ma et al., "Microstructure evolution and magnetic properties of Multi-Main-Phase Nd-Ce-Fe-B permanent magnets," *Journal of Alloys and Compounds* (2025).

See [docs/physics.md](docs/physics.md) for detailed Hamiltonian formulations and validation.
