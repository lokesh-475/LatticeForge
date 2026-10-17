#!/usr/bin/env python3
"""
LatticeForge Dynamic Sintering Animation Exporter & Binary Frame Loader

Parses binary lattice snapshots (outputs/lattice_state.bin) and compiles them
into high-quality dynamic sintering animations (outputs/sintering_demo.gif).

Features:
- High-throughput binary loader supporting both single-frame snapshots and
  multi-frame trajectory dumps with 128-byte packed headers.
- Dual-panel coupled kinetics visualization:
    1. Potts Grain Growth: Curvature-driven crystallographic grain coarsening
    2. Kawasaki Solute Segregation: Conserved atomic exchange yielding core-shell morphology
- Dynamic HUD tracking Monte Carlo Step (MCS), active grains Q, GB Ce concentration,
  and quantitative enrichment factor.
- Supports compiling existing binary frames or evolving from a binary snapshot state.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure paths to import LatticeForge python modules
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TOOL_DIR)
ROOT_DIR = os.path.dirname(PYTHON_DIR)

if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tools.read_lattice_binary import (
    load_lattice_frames,
    load_lattice_snapshot,
    dump_lattice_frames,
    resolve_snapshot_path,
)
from prototypes.prototype_sim import (
    LatticeForgePrototype,
    SimulationConfig,
)


def compute_frame_metrics(grain_id: np.ndarray, element_type: np.ndarray, j_gb: float = 1.0, h_seg: float = 1.5, j_chem: float = 0.5):
    """
    Computes spatial segregation metrics and grain statistics for a frame.
    """
    L = grain_id.shape[0]
    total_sites = L * L

    # Von Neumann boundary character (periodic)
    north = np.roll(grain_id, -1, axis=0)
    south = np.roll(grain_id, 1, axis=0)
    east = np.roll(grain_id, -1, axis=1)
    west = np.roll(grain_id, 1, axis=1)

    gb_map = (
        (grain_id != north).astype(np.int32)
        + (grain_id != south).astype(np.int32)
        + (grain_id != east).astype(np.int32)
        + (grain_id != west).astype(np.int32)
    )

    is_gb = (gb_map > 0)
    total_gb_sites = int(np.sum(is_gb))
    total_bulk_sites = total_sites - total_gb_sites

    is_ce = (element_type == 1)
    total_ce = int(np.sum(is_ce))
    ce_at_gb = int(np.sum(is_ce & is_gb))
    ce_at_bulk = total_ce - ce_at_gb

    c_gb = ce_at_gb / max(1, total_gb_sites)
    c_bulk = ce_at_bulk / max(1, total_bulk_sites)
    enrichment = c_gb / max(1e-6, c_bulk)
    unique_grains = len(np.unique(grain_id))

    # Total Hamiltonian calculation
    h_potts = (j_gb / 2.0) * float(np.sum(gb_map))
    h_seg_val = -h_seg * float(np.sum(element_type * gb_map))
    h_chem_val = 0.0
    if j_chem != 0.0:
        c_north = np.roll(element_type, -1, axis=0)
        c_south = np.roll(element_type, 1, axis=0)
        c_east = np.roll(element_type, -1, axis=1)
        c_west = np.roll(element_type, 1, axis=1)
        unlike_bonds = (
            (element_type != c_north).astype(np.int32)
            + (element_type != c_south).astype(np.int32)
            + (element_type != c_east).astype(np.int32)
            + (element_type != c_west).astype(np.int32)
        )
        h_chem_val = (j_chem / 2.0) * float(np.sum(unlike_bonds))

    hamiltonian = h_potts + h_seg_val + h_chem_val

    return {
        "num_grains": unique_grains,
        "total_ce": total_ce,
        "ce_at_gb": ce_at_gb,
        "ce_at_gb_pct": (ce_at_gb / max(1, total_ce)) * 100.0,
        "c_gb": c_gb,
        "c_bulk": c_bulk,
        "enrichment": enrichment,
        "hamiltonian": hamiltonian,
    }


def prepare_animation_frames(snapshot_path, requested_frames=40, potts_sweeps=1, kawasaki_sweeps=2, force_simulate=False):
    """
    Loads binary frames from snapshot file, or evolves dynamic simulation starting
    from the binary snapshot initial state if only 1 frame is provided.
    """
    try:
        resolved = resolve_snapshot_path(snapshot_path)
        binary_frames = load_lattice_frames(resolved)
        print(f" Loaded {len(binary_frames)} frame(s) from binary snapshot: {resolved}")
    except (FileNotFoundError, ValueError) as err:
        print(f" Notice: Binary snapshot load ({snapshot_path}): {err}")
        print(" Fallback: Initializing Voronoi simulation from scratch...")
        binary_frames = []

    # If multi-frame binary file has enough frames and not forced to simulate, use them directly
    if len(binary_frames) > 1 and not force_simulate:
        frames_data = []
        for fr in binary_frames[:requested_frames]:
            gid = fr["sigma"]
            elem = fr["c"]
            h = fr["header"]
            m = compute_frame_metrics(gid, elem, h["j_gb"], h["h_seg"], h["j_chem"])
            frames_data.append({
                "grain_id": gid,
                "element_type": elem,
                "mcs_step": h["mcs_step"],
                "temp": h["temp"],
                "metrics": m,
                "header": h,
            })
        print(f" Extracted {len(frames_data)} pre-computed frames directly from binary stream.")
        return frames_data

    # Evolve forward using Python simulation engine
    if binary_frames:
        # Seed simulation with the binary snapshot's microstructure and physics
        base_frame = binary_frames[0]
        h = base_frame["header"]
        cfg = SimulationConfig(
            size=base_frame["L"],
            temp=h["temp"],
            j_gb=h["j_gb"],
            h_seg=h["h_seg"],
            j_chem=h["j_chem"],
            seed=42
        )
        sim = LatticeForgePrototype(cfg)
        sim.grain_id = base_frame["sigma"].astype(np.int32).copy()
        sim.element_type = base_frame["c"].astype(np.uint8).copy()
        sim.initial_ce_count = int(np.sum(sim.element_type))
        sim.step_count = h["mcs_step"]
        print(f" Initialized simulation from binary frame (Step={sim.step_count}, L={sim.size}, Grains={len(np.unique(sim.grain_id))})")
    else:
        # From-scratch fallback
        cfg = SimulationConfig(size=128, temp=0.4, seed=42)
        sim = LatticeForgePrototype(cfg)
        sim.initialize_voronoi(num_grains=28, fraction_ce=0.25)
        print(" Initialized synthetic Voronoi microstructure (28 grains, 25% Ce).")

    frames_data = []
    print(f" Simulating {requested_frames} dynamic sintering frames (Potts={potts_sweeps}/fr, Kawasaki={kawasaki_sweeps}/fr)...")
    
    t_start = time.perf_counter()
    for f in range(requested_frames):
        if f > 0:
            sim.step(potts_mc_sweeps=potts_sweeps, kawasaki_mc_sweeps=kawasaki_sweeps)

        gid = sim.grain_id.copy()
        elem = sim.element_type.copy()
        m = compute_frame_metrics(gid, elem, sim.j_gb, sim.h_seg, sim.j_chem)

        frames_data.append({
            "grain_id": gid,
            "element_type": elem,
            "mcs_step": sim.step_count,
            "temp": sim.temp,
            "metrics": m,
            "header": {
                "dim_x": sim.size,
                "dim_y": sim.size,
                "temp": sim.temp,
                "j_gb": sim.j_gb,
                "h_seg": sim.h_seg,
                "j_chem": sim.j_chem,
                "mcs_step": sim.step_count,
            }
        })
        if (f + 1) % 10 == 0 or (f + 1) == requested_frames:
            print(f"   [Frame {f + 1:02d}/{requested_frames}] MCS {sim.step_count:03d} | Grains: {m['num_grains']:02d} | Ce at GB: {m['ce_at_gb_pct']:.1f}% | Enrichment: {m['enrichment']:.2f}x")

    t_elapsed = time.perf_counter() - t_start
    print(f" Kinetics evolution completed in {t_elapsed:.2f} s ({requested_frames / t_elapsed:.1f} fps).")

    return frames_data


def export_sintering_animation(frames_data, output_path="outputs/sintering_demo.gif", fps=10, dpi=120):
    """
    Renders frames into a high-resolution, publication-quality animated GIF.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    first_frame = frames_data[0]
    initial_gid = first_frame["grain_id"]
    initial_elem = first_frame["element_type"]
    L = initial_gid.shape[0]

    # Create figure with high aesthetic styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=dpi)
    fig.patch.set_facecolor("#12161f")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0d13")
        ax.axis("off")

    # Panel 1: Grain Morphology (Potts Model)
    im_grain = ax1.imshow(initial_gid, cmap="tab20b", animated=True)
    title_grain = ax1.set_title(
        f"Grain Growth (Potts MC)\nActive Grains: Q={first_frame['metrics']['num_grains']}",
        fontsize=12, fontweight="bold", color="#e2e8f0", pad=10
    )

    # Panel 2: Solute Segregation (Kawasaki Dynamics)
    im_bg = ax2.imshow(initial_gid, cmap="binary", alpha=0.25)
    ce_mask = np.ma.masked_where(initial_elem == 0, initial_elem)
    im_chem = ax2.imshow(ce_mask, cmap="autumn", interpolation="nearest", animated=True)
    title_chem = ax2.set_title(
        f"Solute Segregation (Kawasaki)\nCe at GB: {first_frame['metrics']['ce_at_gb_pct']:.1f}% (Enrichment: {first_frame['metrics']['enrichment']:.2f}x)",
        fontsize=12, fontweight="bold", color="#fca5a5", pad=10
    )

    # HUD Super Title
    m0 = first_frame["metrics"]
    hud_title = fig.suptitle(
        f"LatticeForge: Dynamic Core-Shell Sintering Kinetics\n"
        f"Step: {first_frame['mcs_step']} MCS | Ts: {first_frame['temp']:.2f} | Dimension: {L}x{L} | Sites: {L*L:,}",
        fontsize=13, fontweight="bold", color="#38bdf8", y=0.98
    )

    # Footnote/Metadata text
    info_text = fig.text(
        0.5, 0.03,
        "Red-Black Potts Checkerboard + 4-Color Partition Kawasaki Solute Diffusion | Delta N_Ce = 0",
        ha="center", fontsize=9, color="#94a3b8", style="italic"
    )

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])

    def update(frame_idx):
        fr = frames_data[frame_idx]
        gid = fr["grain_id"]
        elem = fr["element_type"]
        m = fr["metrics"]
        step = fr["mcs_step"]
        temp = fr["temp"]

        # Update grain orientation map
        im_grain.set_array(gid)
        title_grain.set_text(f"Grain Growth (Potts MC)\nActive Grains: Q={m['num_grains']}")

        # Update solute segregation overlay
        new_ce_mask = np.ma.masked_where(elem == 0, elem)
        im_chem.set_array(new_ce_mask)
        title_chem.set_text(f"Solute Segregation (Kawasaki)\nCe at GB: {m['ce_at_gb_pct']:.1f}% (Enrichment: {m['enrichment']:.2f}x)")

        # Update HUD text
        hud_title.set_text(
            f"LatticeForge: Dynamic Core-Shell Sintering Kinetics\n"
            f"Step: {step} MCS | Ts: {temp:.2f} | Dimension: {L}x{L} | Frame: {frame_idx + 1}/{len(frames_data)}"
        )

        return [im_grain, im_chem, title_grain, title_chem, hud_title]

    print(f" Compiling {len(frames_data)} frames into animated GIF at {fps} FPS...")
    interval_ms = int(1000.0 / fps)
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=interval_ms, blit=False
    )

    t0 = time.perf_counter()
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    t1 = time.perf_counter()

    gif_size_bytes = os.path.getsize(output_path)
    print(f" Successfully exported animation to: {output_path}")
    print(f" GIF Size:     {gif_size_bytes / (1024.0 * 1024.0):.2f} MB ({gif_size_bytes:,} bytes)")
    print(f" Render Time:  {t1 - t0:.2f} s")
    print(f" Frames:       {len(frames_data)} frames @ {fps} fps ({len(frames_data)/fps:.1f} s duration)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="LatticeForge Dynamic Sintering Animation Exporter")
    parser.add_argument("snapshot", nargs="?", default="outputs/lattice_state.bin",
                        help="Path to binary snapshot or trajectory file (default: outputs/lattice_state.bin)")
    parser.add_argument("--output", "-o", default="outputs/sintering_demo.gif",
                        help="Path for compiled GIF output (default: outputs/sintering_demo.gif)")
    parser.add_argument("--frames", "-n", type=int, default=40,
                        help="Number of animation frames to render (default: 40)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Animation playback frame rate (default: 10)")
    parser.add_argument("--potts-sweeps", type=int, default=1,
                        help="Potts MC sweeps per dynamic frame (default: 1)")
    parser.add_argument("--kawasaki-sweeps", type=int, default=2,
                        help="Kawasaki MC sweeps per dynamic frame (default: 2)")
    parser.add_argument("--simulate", action="store_true",
                        help="Force forward kinetic simulation starting from snapshot state")
    parser.add_argument("--dump-binary", default=None,
                        help="Optional output path to save multi-frame binary trajectory (.bin)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Rendering DPI (default: 120)")

    args = parser.parse_args()

    print("================================================================================")
    print(" LatticeForge Dynamic Sintering Animation Exporter")
    print("================================================================================")
    print(f" Source Snapshot:    {args.snapshot}")
    print(f" Target Output:      {args.output}")
    print(f" Frame Count:        {args.frames} frames @ {args.fps} FPS")
    print(f" Coupled Sweeps:     {args.potts_sweeps} Potts + {args.kawasaki_sweeps} Kawasaki per frame")
    print("================================================================================")

    # Prepare animation frames
    frames_data = prepare_animation_frames(
        snapshot_path=args.snapshot,
        requested_frames=args.frames,
        potts_sweeps=args.potts_sweeps,
        kawasaki_sweeps=args.kawasaki_sweeps,
        force_simulate=args.simulate
    )

    # Optional multi-frame binary dump
    if args.dump_binary:
        binary_save_path = args.dump_binary
        bin_frames = []
        for fr in frames_data:
            bin_frames.append({
                "sigma": fr["grain_id"].astype(np.uint16),
                "c": fr["element_type"].astype(np.uint8),
                "header": fr["header"],
                "mcs_step": fr["mcs_step"],
                "temp": fr["temp"],
            })
        dump_lattice_frames(binary_save_path, bin_frames)
        print(f" Exported multi-frame binary trajectory to: {binary_save_path}")

    # Compile into animated GIF
    export_sintering_animation(
        frames_data=frames_data,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi
    )
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
