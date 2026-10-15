#!/usr/bin/env python3
"""
LatticeForge High-Resolution Speedup & Scaling Analytics Visualizer

Quantitative acceleration and throughput scaling benchmarking of LatticeForge
CUDA kernels (Potts Monte Carlo checkerboard and Kawasaki 4-color exchange)
relative to the single-threaded C++20 baseline.

Confirms and visualizes >40x acceleration target across production lattice scales.
Generates publication-quality dual-panel scaling figures for documentation:
  docs/benchmarks/speedup_curve.png
"""

import os
import sys
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def get_project_paths():
    """Resolves project paths relative to this script."""
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(tool_dir)
    root_dir = os.path.dirname(python_dir)
    docs_bench_dir = os.path.join(root_dir, 'docs', 'benchmarks')
    outputs_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(docs_bench_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return root_dir, docs_bench_dir, outputs_dir


def load_benchmark_data(root_dir, docs_bench_dir, outputs_dir):
    """
    Finds and loads benchmark results from docs/benchmarks/performance_logs.csv
    or outputs/benchmark_results.csv.
    """
    csv_candidates = [
        os.path.join(docs_bench_dir, 'performance_logs.csv'),
        os.path.join(outputs_dir, 'benchmark_results.csv'),
        os.path.join(root_dir, 'docs', 'benchmarks', 'performance_logs.csv'),
        os.path.join(root_dir, 'outputs', 'benchmark_results.csv'),
        os.path.join(root_dir, 'build', 'docs', 'benchmarks', 'performance_logs.csv'),
        os.path.join(root_dir, 'build', 'outputs', 'benchmark_results.csv'),
    ]

    chosen_csv = None
    for path in csv_candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            chosen_csv = path
            break

    if not chosen_csv:
        print("Benchmark CSV not found. Attempting to execute benchmark_engine.exe...")
        exe_candidates = [
            os.path.join(root_dir, 'benchmark_engine.exe'),
            os.path.join(root_dir, 'build', 'benchmark_engine.exe'),
        ]
        executed = False
        for exe in exe_candidates:
            if os.path.exists(exe):
                print(f"Executing: {exe} 50 5...")
                res = subprocess.run([exe, "50", "5"], cwd=root_dir, capture_output=True, text=True)
                if res.returncode == 0:
                    executed = True
                    break

        for path in csv_candidates:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                chosen_csv = path
                break

    if not chosen_csv:
        raise FileNotFoundError("Could not find or generate benchmark results CSV.")

    print(f"Loading performance logs from: {chosen_csv}")
    rows = []
    with open(chosen_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'grid_dim': int(r['grid_dim']),
                'total_sites': int(r['total_sites']),
                'mcs_steps': int(r['mcs_steps']),
                'potts_time_ms': float(r['potts_time_ms']),
                'kawasaki_time_ms': float(r['kawasaki_time_ms']),
                'total_time_ms': float(r['total_time_ms']),
                'potts_gsups': float(r['potts_gsups']),
                'kawasaki_gsups': float(r['kawasaki_gsups']),
                'sustained_gsups': float(r['sustained_gsups']),
                'cpu_potts_time_ms': float(r.get('cpu_potts_time_ms', 0.0)),
                'cpu_kawasaki_time_ms': float(r.get('cpu_kawasaki_time_ms', 0.0)),
                'cpu_total_time_ms': float(r.get('cpu_total_time_ms', 0.0)),
                'cpu_sustained_gsups': float(r.get('cpu_sustained_gsups', 0.0)),
                'potts_speedup': float(r.get('potts_speedup', 1.0)),
                'kawasaki_speedup': float(r.get('kawasaki_speedup', 1.0)),
                'sustained_speedup': float(r.get('sustained_speedup', 1.0)),
            })

    # Sort rows by total_sites ascending
    rows.sort(key=lambda x: x['total_sites'])
    return rows, chosen_csv


def print_performance_report(data):
    """Outputs a formatted console report verifying acceleration thresholds."""
    print("\n" + "=" * 90)
    print(" LATTICEFORGE GPU vs SINGLE-THREADED CPU PERFORMANCE VALIDATION")
    print("=" * 90)
    print(f"{'Grid':<12} {'Lattice Sites':<16} {'GPU Time':<12} {'CPU Time':<12} {'GPU G-SUPS':<14} {'Sustained Speedup':<18} {'Status'}")
    print("-" * 90)

    target_met = False
    max_speedup = 0.0
    for r in data:
        dim_str = f"{r['grid_dim']}x{r['grid_dim']}"
        sites_str = f"{r['total_sites']:,}"
        gpu_t_str = f"{r['total_time_ms']:.2f} ms"
        cpu_t_str = f"{r['cpu_total_time_ms']:.2f} ms"
        gpu_gsups_str = f"{r['sustained_gsups']:.3f}"
        speedup_str = f"{r['sustained_speedup']:.2f}x"
        
        status = "PASSED (>40x)" if r['sustained_speedup'] >= 40.0 else "Sub-target"
        if r['sustained_speedup'] >= 40.0:
            target_met = True
        if r['sustained_speedup'] > max_speedup:
            max_speedup = r['sustained_speedup']

        print(f"{dim_str:<12} {sites_str:<16} {gpu_t_str:<12} {cpu_t_str:<12} {gpu_gsups_str:<14} {speedup_str:<18} {status}")

    print("=" * 90)
    if target_met:
        print(f"[SUCCESS] Verified >40x CUDA acceleration! Maximum sustained speedup: {max_speedup:.2f}x.")
    else:
        print(f"[WARNING] Target 40x speedup not reached yet (Peak: {max_speedup:.2f}x).")
    print("=" * 90 + "\n")
    return target_met


def plot_speedup_curves(data, docs_bench_dir, outputs_dir):
    """
    Plots high-aesthetic dual-panel speedup and throughput scaling figure:
      Panel 1: Relative Acceleration / Speedup factor vs Lattice Size (L and Sites)
               with 40x target line and annotations.
      Panel 2: Sustained Throughput (Giga-SUPS) scaling comparing CUDA GPU vs CPU.
    """
    grid_labels = [f"{r['grid_dim']}" + r"$\times$" + f"{r['grid_dim']}\n({r['total_sites']:,})" for r in data]
    dims = np.array([r['grid_dim'] for r in data])
    sites = np.array([r['total_sites'] for r in data])
    sustained_speedup = np.array([r['sustained_speedup'] for r in data])
    potts_speedup = np.array([r['potts_speedup'] for r in data])
    kawasaki_speedup = np.array([r['kawasaki_speedup'] for r in data])
    
    gpu_gsups = np.array([r['sustained_gsups'] for r in data])
    cpu_gsups = np.array([r['cpu_sustained_gsups'] for r in data])

    x_indices = np.arange(len(data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=200)

    # Styling settings
    fig.patch.set_facecolor('#ffffff')
    plt.rcParams['font.family'] = 'sans-serif'

    # ================= PANEL 1: Acceleration / Speedup =================
    ax1.set_facecolor('#fafbfc')

    # Target threshold region (>40x)
    max_y = max(np.max(sustained_speedup), np.max(potts_speedup), np.max(kawasaki_speedup), 50.0) * 1.12
    ax1.axhspan(40.0, max_y * 1.05, color='#e8f5e9', alpha=0.55, zorder=1, label=r'Target Speedup Zone ($\geq 40\times$)')
    ax1.axhline(40.0, color='#2e7d32', linestyle='--', linewidth=2.0, zorder=2, label=r'$40\times$ Acceleration Target')

    # Line plots for speedups
    ax1.plot(x_indices, sustained_speedup, marker='o', markersize=8, linewidth=2.8,
             color='#1565c0', label=r'Combined Sustained Speedup ($S_{\mathrm{total}}$)', zorder=4)
    ax1.plot(x_indices, potts_speedup, marker='s', markersize=7, linewidth=2.2, linestyle='-.',
             color='#d32f2f', label=r'Potts Kernel Speedup ($S_{\mathrm{Potts}}$)', zorder=4)
    ax1.plot(x_indices, kawasaki_speedup, marker='^', markersize=7, linewidth=2.2, linestyle=':',
             color='#f57c00', label=r'Kawasaki Kernel Speedup ($S_{\mathrm{Kawasaki}}$)', zorder=4)

    # Annotate peak sustained speedup point in the spacious corridor between Kawasaki and Sustained curves
    peak_idx = int(np.argmax(sustained_speedup))
    peak_val = sustained_speedup[peak_idx]
    peak_x = x_indices[peak_idx]
    peak_dim = dims[peak_idx]

    ax1.annotate(
        f'Sustained Peak: {peak_val:.1f}x\n({peak_dim}x{peak_dim} Lattice)',
        xy=(peak_x, peak_val),
        xytext=(2.55, 335),
        arrowprops=dict(facecolor='#1565c0', edgecolor='#0d47a1', shrink=0.08, width=1.6, headwidth=6),
        fontsize=10.0, fontweight='bold', color='#0d47a1',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffff', edgecolor='#1565c0', alpha=0.95)
    )

    # Annotate peak Kawasaki speedup point
    kaw_peak_idx = int(np.argmax(kawasaki_speedup))
    kaw_peak_val = kawasaki_speedup[kaw_peak_idx]
    ax1.annotate(
        f'Peak Kawasaki: {kaw_peak_val:.1f}x',
        xy=(x_indices[kaw_peak_idx], kaw_peak_val),
        xytext=(2.15, 620),
        arrowprops=dict(facecolor='#f57c00', edgecolor='#e65100', shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5, fontweight='bold', color='#e65100',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff', edgecolor='#f57c00', alpha=0.95)
    )

    # Point annotations on sustained curve
    for i, (xi, sp) in enumerate(zip(x_indices, sustained_speedup)):
        if i != peak_idx:
            # For small points near the bottom, place closely above marker to avoid hitting 40x target line
            y_offset = 6.0 if sp < 20.0 else 14.0
            ax1.text(xi, sp + y_offset, f'{sp:.1f}x', ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color='#1565c0')

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(grid_labels, fontsize=9.5)
    ax1.set_xlabel('Lattice Dimension $L$ (Total Discrete Sites $N = L^2$)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel(r'Speedup Factor ($T_{\mathrm{CPU}} / T_{\mathrm{CUDA}}$)', fontsize=11, fontweight='bold')
    ax1.set_title('CUDA Acceleration Scaling vs Single-Threaded C++ Baseline', fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_ylim(0, max_y)
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.legend(loc='upper left', fontsize=9.0, framealpha=0.95)

    # ================= PANEL 2: Throughput (Giga-SUPS) =================
    ax2.set_facecolor('#fafbfc')

    bar_width = 0.36
    b1 = ax2.bar(x_indices - bar_width/2, gpu_gsups, width=bar_width, color='#1976d2',
                 label='CUDA GPU Sustained Throughput', zorder=3, edgecolor='#0d47a1', linewidth=1.2)
    b2 = ax2.bar(x_indices + bar_width/2, cpu_gsups, width=bar_width, color='#90a4ae',
                 label='Single-Threaded C++ Baseline', zorder=3, edgecolor='#546e7a', linewidth=1.2)

    # Bar value labels
    for rect in b1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h * 1.02 + 0.01, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#0d47a1')

    for rect in b2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h * 1.02 + 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=8.5, color='#37474f')

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(grid_labels, fontsize=9.5)
    ax2.set_xlabel('Lattice Dimension $L$ (Total Discrete Sites $N = L^2$)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Sustained Throughput (Giga-SUPS: $10^9$ Site Updates/s)', fontsize=11, fontweight='bold')
    ax2.set_title('Throughput Scaling & GPU Compute Engine Saturation', fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_ylim(0, max(np.max(gpu_gsups) * 1.18, 1.0))
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(loc='upper left', fontsize=9.5, framealpha=0.95)

    plt.tight_layout()

    # Destination paths
    out_docs = os.path.join(docs_bench_dir, 'speedup_curve.png')
    out_outputs = os.path.join(outputs_dir, 'speedup_curve.png')

    plt.savefig(out_docs, dpi=200, bbox_inches='tight')
    plt.savefig(out_outputs, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[+] Scaling figure generated and saved to:")
    print(f"    - {out_docs}")
    print(f"    - {out_outputs}")
    return out_docs


def main():
    root_dir, docs_bench_dir, outputs_dir = get_project_paths()
    data, chosen_csv = load_benchmark_data(root_dir, docs_bench_dir, outputs_dir)
    print_performance_report(data)
    out_path = plot_speedup_curves(data, docs_bench_dir, outputs_dir)


if __name__ == '__main__':
    main()
