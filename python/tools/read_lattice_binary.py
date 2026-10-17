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

def resolve_snapshot_path(filepath="outputs/lattice_state.bin"):
    """Resolves snapshot path with fallbacks across parent and outputs directories."""
    if os.path.exists(filepath):
        return filepath
    alt_paths = [
        os.path.join("outputs", os.path.basename(filepath)),
        os.path.join("..", "outputs", os.path.basename(filepath)),
        os.path.join(os.path.dirname(__file__), "..", "..", "outputs", os.path.basename(filepath)),
        os.path.join(os.path.dirname(__file__), "..", "..", filepath),
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"Snapshot file not found: {filepath}")

def load_lattice_frames(filepath="outputs/lattice_state.bin"):
    """
    Loads all uncompressed binary lattice snapshots (frames) from a file.
    Supports single-frame snapshot dumps as well as multi-frame trajectory files.

    Parameters:
        filepath (str): Path to .bin snapshot file containing one or more frames.

    Returns:
        list of dicts, each containing:
            'header': metadata dictionary,
            'sigma': uint16 2D array of grain orientations [L, L],
            'c': uint8 2D array of chemical identities [L, L],
            'L': grid dimension,
            'total_sites': total lattice sites,
            'mcs_step': simulation step,
            'temp': temperature Ts,
            'read_time_ms': total elapsed read time in milliseconds,
            'filepath': resolved file path
    """
    resolved_path = resolve_snapshot_path(filepath)
    t0 = time.perf_counter()
    frames = []

    with open(resolved_path, "rb") as f:
        frame_idx = 0
        while True:
            header_data = f.read(HEADER_SIZE)
            if not header_data or len(header_data) == 0:
                break
            if len(header_data) < HEADER_SIZE:
                # File ended mid-header
                break

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
                if frame_idx == 0:
                    raise ValueError(f"Invalid magic bytes in {resolved_path}: {magic!r}, expected {EXPECTED_MAGIC!r}")
                else:
                    break

            if bytes_sigma != 2 or bytes_c != 1:
                raise ValueError(f"Frame {frame_idx}: Unexpected element byte size: sigma={bytes_sigma}, c={bytes_c}")

            # Advance past extra header bytes if header_size > HEADER_SIZE
            if header_size > HEADER_SIZE:
                f.seek(header_size - HEADER_SIZE, os.SEEK_CUR)

            # Stream contiguous arrays directly into NumPy memory buffers
            sigma = np.fromfile(f, dtype=np.uint16, count=total_sites)
            c = np.fromfile(f, dtype=np.uint8, count=total_sites)

            if sigma.size != total_sites or c.size != total_sites:
                if frame_idx == 0:
                    raise ValueError(f"Incomplete payload: expected {total_sites} sites, got sigma={sigma.size}, c={c.size}")
                else:
                    break

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

            frames.append({
                "header": header_info,
                "sigma": sigma,
                "c": c,
                "L": dim_x,
                "total_sites": total_sites,
                "mcs_step": mcs_step,
                "temp": temp,
                "filepath": resolved_path,
                "frame_index": frame_idx
            })
            frame_idx += 1

    t1 = time.perf_counter()
    read_time_ms = (t1 - t0) * 1000.0

    if not frames:
        raise ValueError(f"No valid frames found in {resolved_path}")

    for fr in frames:
        fr["read_time_ms"] = read_time_ms

    return frames

def load_lattice_snapshot(filepath="outputs/lattice_state.bin"):
    """
    Loads an uncompressed binary lattice snapshot into NumPy arrays.
    Returns the first frame from the snapshot file.
    """
    frames = load_lattice_frames(filepath)
    return frames[0]

def dump_lattice_snapshot(filepath, sigma, c, mcs_step=0, temp=0.4, j_gb=1.0, h_seg=1.5, j_chem=0.5, append=False):
    """
    Dumps or appends an uncompressed binary lattice snapshot with 128-byte packed header.

    Parameters:
        filepath (str): Target output file path.
        sigma (np.ndarray): 2D uint16 grain orientation field [L, L].
        c (np.ndarray): 2D uint8 chemical element field [L, L].
        mcs_step (int): Monte Carlo step index.
        temp (float): Reduced temperature Ts.
        j_gb (float): Potts grain boundary energy coefficient.
        h_seg (float): Solute segregation potential.
        j_chem (float): Solute-matrix interaction coefficient.
        append (bool): If True, append to existing file; otherwise overwrite.
    """
    dim_y, dim_x = sigma.shape
    total_sites = dim_x * dim_y
    magic = EXPECTED_MAGIC
    header_size = HEADER_SIZE
    version = 1
    bytes_sigma = 2
    bytes_c = 1
    flags = 0
    reserved = b"\x00" * 48

    header_bytes = struct.pack(
        HEADER_STRUCT_FORMAT,
        magic, header_size, version, dim_x, dim_y, total_sites,
        bytes_sigma, bytes_c, int(mcs_step), flags,
        float(temp), float(j_gb), float(h_seg), float(j_chem),
        reserved
    )

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mode = "ab" if append else "wb"
    with open(filepath, mode) as f:
        f.write(header_bytes)
        sigma.astype(np.uint16).tofile(f)
        c.astype(np.uint8).tofile(f)

def dump_lattice_frames(filepath, frames):
    """
    Dumps a list of frames sequentially into a single multi-frame trajectory binary file.

    Parameters:
        filepath (str): Target output file path.
        frames (list): List of dicts with 'sigma', 'c', and optional 'header' or physics keys.
    """
    for i, fr in enumerate(frames):
        h = fr.get("header", {})
        dump_lattice_snapshot(
            filepath=filepath,
            sigma=fr["sigma"],
            c=fr["c"],
            mcs_step=h.get("mcs_step", fr.get("mcs_step", i)),
            temp=h.get("temp", fr.get("temp", 0.4)),
            j_gb=h.get("j_gb", fr.get("j_gb", 1.0)),
            h_seg=h.get("h_seg", fr.get("h_seg", 1.5)),
            j_chem=h.get("j_chem", fr.get("j_chem", 0.5)),
            append=(i > 0)
        )

def main():
    parser = argparse.ArgumentParser(description="LatticeForge Binary Snapshot Reader & Inspector")
    parser.add_argument("snapshot", nargs="?", default="outputs/lattice_state.bin", help="Path to binary snapshot")
    parser.add_argument("--save-plot", action="store_true", help="Generate PNG visual preview of state")
    args = parser.parse_args()

    print("================================================================================")
    print(" LatticeForge Binary Snapshot Reader (outputs/lattice_state.bin)")
    print("================================================================================")
    
    try:
        frames = load_lattice_frames(args.snapshot)
        data = frames[0]
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return 1

    h = data["header"]
    sigma = data["sigma"]
    c = data["c"]
    L = data["L"]
    total_sites = data["total_sites"]
    single_frame_bytes = 128 + total_sites * 3
    file_bytes = os.path.getsize(data["filepath"])

    ce_count = int(np.sum(c == 1))
    ce_pct = (ce_count / total_sites) * 100.0
    num_grains = len(np.unique(sigma))

    print(f" File Path:           {data['filepath']}")
    print(f" File Size:           {file_bytes / (1024.0 * 1024.0):.2f} MB ({file_bytes:,} bytes)")
    print(f" Frames in File:      {len(frames)} snapshot(s)")
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
