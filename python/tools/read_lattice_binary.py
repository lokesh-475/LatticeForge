#!/usr/bin/env python3
"""
LatticeForge High-Throughput Binary Snapshot Reader & Validator

Reads uncompressed binary lattice snapshots (outputs/lattice_state.bin)
dumped by the C++ engine. Parses the 128-byte header and loads full
multi-million site arrays (sigma and c) in milliseconds via NumPy memory streaming.

Format:
- 128-byte packed header ("LFORGE01", dimensions, physical parameters)
- Orientations array (sigma): uint16 [total_sites]
- Chemical elements array (c): uint8 [total_sites]
"""

import os
import sys
import time
import struct
import argparse
import numpy as np

HEADER_STRUCT_FORMAT = "<8sIIIIQIIiIdddd48s"
HEADER_SIZE = 128
EXPECTED_MAGIC = b"LFORGE01"

def load_lattice_snapshot(filepath="outputs/lattice_state.bin"):
    """
    Loads an uncompressed binary lattice snapshot into NumPy arrays.

    Parameters:
        filepath (str): Path to .bin snapshot file.

    Returns:
        dict containing:
            'header': metadata dictionary,
            'sigma': uint16 2D array of grain orientations [L, L],
            'c': uint8 2D array of chemical identities [L, L],
            'L': grid dimension,
            'total_sites': total lattice sites,
            'mcs_step': simulation step,
            'read_time_ms': elapsed load time in milliseconds
    """
    if not os.path.exists(filepath):
        # Check fallback to ../outputs/ or outputs/
        alt_paths = [
            os.path.join("outputs", os.path.basename(filepath)),
            os.path.join("..", "outputs", os.path.basename(filepath)),
            os.path.join(os.path.dirname(__file__), "..", "..", "outputs", os.path.basename(filepath))
        ]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                filepath = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")

    t0 = time.perf_counter()

    with open(filepath, "rb") as f:
        header_data = f.read(HEADER_SIZE)
        if len(header_data) < HEADER_SIZE:
            raise ValueError(f"File too small ({len(header_data)} bytes) to contain 128-byte header.")

        unpacked = struct.unpack(HEADER_STRUCT_FORMAT, header_data)
        magic = unpacked[0]
        header_size = unpacked[1]
        version = unpacked[2]
        dim_x = unpacked[3]
        dim_y = unpacked[4]
        total_sites = unpacked[5]
        bytes_sigma = unpacked[6]
        bytes_c = unpacked[7]
        mcs_step = unpacked[8]
        flags = unpacked[9]
        temp = unpacked[10]
        j_gb = unpacked[11]
        h_seg = unpacked[12]
        j_chem = unpacked[13]

        if magic != EXPECTED_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}, expected {EXPECTED_MAGIC!r}")

        if bytes_sigma != 2 or bytes_c != 1:
            raise ValueError(f"Unexpected element byte size: sigma={bytes_sigma}, c={bytes_c}")

        # Seek to designated header size for forward compatibility
        f.seek(header_size, os.SEEK_SET)

        # Stream contiguous arrays directly into NumPy memory buffers
        sigma = np.fromfile(f, dtype=np.uint16, count=total_sites)
        c = np.fromfile(f, dtype=np.uint8, count=total_sites)

    t1 = time.perf_counter()
    read_time_ms = (t1 - t0) * 1000.0

    if sigma.size != total_sites or c.size != total_sites:
        raise ValueError(f"Incomplete payload: expected {total_sites} sites, got sigma={sigma.size}, c={c.size}")

    sigma = sigma.reshape((dim_y, dim_x))
    c = c.reshape((dim_y, dim_x))

    header_info = {
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "header_size": header_size,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "total_sites": total_sites,
        "bytes_per_sigma": bytes_sigma,
        "bytes_per_c": bytes_c,
        "mcs_step": mcs_step,
        "flags": flags,
        "temp": temp,
        "j_gb": j_gb,
        "h_seg": h_seg,
        "j_chem": j_chem,
    }

    return {
        "header": header_info,
        "sigma": sigma,
        "c": c,
        "L": dim_x,
        "total_sites": total_sites,
        "mcs_step": mcs_step,
        "read_time_ms": read_time_ms,
        "filepath": filepath,
    }

def main():
    parser = argparse.ArgumentParser(description="LatticeForge Binary Snapshot Reader & Inspector")
    parser.add_argument("snapshot", nargs="?", default="outputs/lattice_state.bin", help="Path to binary snapshot")
    parser.add_argument("--save-plot", action="store_true", help="Generate PNG visual preview of state")
    args = parser.parse_args()

    print("================================================================================")
    print(" LatticeForge Binary Snapshot Reader (outputs/lattice_state.bin)")
    print("================================================================================")
    
    try:
        data = load_lattice_snapshot(args.snapshot)
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return 1

    h = data["header"]
    sigma = data["sigma"]
    c = data["c"]
    L = data["L"]
    total_sites = data["total_sites"]
    file_bytes = 128 + total_sites * 3

    ce_count = int(np.sum(c == 1))
    ce_pct = (ce_count / total_sites) * 100.0
    num_grains = len(np.unique(sigma))

    print(f" File Path:           {data['filepath']}")
    print(f" File Size:           {file_bytes / (1024.0 * 1024.0):.2f} MB ({file_bytes:,} bytes)")
    print(f" Lattice Dimension:   {L} x {L} ({total_sites:,} sites)")
    print(f" Monte Carlo Step:    {h['mcs_step']}")
    print(f" Physics Regime:      Temp={h['temp']:.2f}, J_gb={h['j_gb']:.2f}, H_seg={h['h_seg']:.2f}, J_chem={h['j_chem']:.2f}")
    print(f" Active Grains (q):   {num_grains} unique orientations")
    print(f" Solute Concentration: {ce_count:,} Ce sites ({ce_pct:.2f}% of lattice)")
    print(f" High-Throughput Load: Read & unpacked in {data['read_time_ms']:.3f} ms")
    throughput_mbs = (file_bytes / (1024.0 * 1024.0)) / (data['read_time_ms'] / 1000.0)
    print(f" Streaming Bandwidth:  {throughput_mbs:.1f} MB/s")
    print("================================================================================")

    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Lattice Snapshot: Step {h['mcs_step']} ({L}x{L})", fontsize=14, fontweight="bold")

            ax1.imshow(sigma, cmap="tab20b")
            ax1.set_title(f"Grain Orientations (sigma, Q={num_grains})")
            ax1.axis("off")

            ce_mask = np.ma.masked_where(c == 0, c)
            ax2.imshow(sigma, cmap="binary", alpha=0.25)
            ax2.imshow(ce_mask, cmap="autumn", interpolation="nearest")
            ax2.set_title(f"Solute Segregation (Ce={ce_pct:.1f}%)")
            ax2.axis("off")

            plt.tight_layout()
            out_img = os.path.splitext(data["filepath"])[0] + "_preview.png"
            plt.savefig(out_img, dpi=150)
            plt.close()
            print(f"Saved snapshot preview to: {out_img}")
        except ImportError:
            print("matplotlib not available for --save-plot.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
