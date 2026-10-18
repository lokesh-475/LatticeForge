# LatticeForge

High-throughput 2D discrete lattice engine in C++20 and CUDA for simulating multi-phase sintering and core-shell kinetics in permanent magnet alloys (Nd-Ce-Fe-B).

## Features

- **Potts Monte Carlo Grain Growth**: 2-color red-black checkerboard decomposition with Philox4_32_10 PRNG.
- **Kawasaki Solute Segregation**: 4-color sub-lattice partition kernel with shared memory halo exchange.
- **Radial Concentration Analytics**: Toroidal connected component labeling (CCL), grain centroid detection, and quantitative shell thickness extraction.
- **Multi-Device Parity Testing**: Statistical equivalence and bitwise parity test suite (`ctest`).

Read and inspect with Python:
```bash
python python/tools/read_lattice_binary.py outputs/lattice_state.bin --save-plot
```

Generate dynamic sintering animation GIF:
```bash
python python/tools/generate_animation.py outputs/lattice_state.bin --output outputs/sintering_demo.gif
```

## Regression Suite & Temperature Sweeps

Run the automated local PowerShell regression suite across dual temperature sintering regimes ($T_s = 0.4$ and $T_s = 2.8$):
```powershell
powershell -ExecutionPolicy Bypass -File tests/run_regression.ps1
```
Or execute via `ctest`:
```bash
ctest --output-on-failure
```
Validates:
- **Output Generation**: Energy logs, 128-byte raw binary snapshots, radial profiles, and grain centroids.
- **Energy Descent**: Hamiltonian energy minimization ($\Delta H < 0$).
- **Invariant Conservation**: Strict atomic species conservation ($\Delta N_{Ce} = 0$).
- **Dual-Temperature Kinetics**: Confirms boundary segregation enrichment at low temperature ($T_s = 0.4$) exceeds thermal dissolution at high temperature ($T_s = 2.8$).
