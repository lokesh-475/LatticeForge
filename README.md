# LatticeForge

High-throughput 2D discrete lattice engine in C++20 and CUDA for simulating multi-phase sintering and core-shell kinetics in permanent magnet alloys (Nd-Ce-Fe-B).

## Features

- **Potts Monte Carlo Grain Growth**: 2-color red-black checkerboard decomposition with Philox4_32_10 PRNG.
- **Kawasaki Solute Segregation**: 4-color sub-lattice partition kernel with shared memory halo exchange.
- **High-Throughput Binary Snapshot Dumping**: Zero-overhead uncompressed binary snapshots (`outputs/lattice_state.bin`) writing multi-million site arrays ($\sigma$ and $c$) in milliseconds (up to ~2,000 MB/s streaming bandwidth).
- **Radial Concentration Analytics**: Toroidal connected component labeling (CCL), grain centroid detection, and quantitative shell thickness extraction.
- **Multi-Device Parity Testing**: Statistical equivalence and bitwise parity test suite (`ctest`).

## Binary Snapshot Format (`outputs/lattice_state.bin`)

- Packed 128-byte header (`LFORGE01` magic identifier, grid dimension, simulation parameters).
- Sequential contiguous arrays:
  - $\sigma$ (orientations): `uint16_t` ($N$ sites)
  - $c$ (chemical elements): `uint8_t` ($N$ sites)
- Total size: $128 + 3 \times N$ bytes (e.g., 12.0 MB for $2048 \times 2048$, dumped in ~6 ms).

Read and inspect with Python:
```bash
python python/tools/read_lattice_binary.py outputs/lattice_state.bin --save-plot
```

Generate dynamic sintering animation GIF:
```bash
python python/tools/generate_animation.py outputs/lattice_state.bin --output outputs/sintering_demo.gif
```