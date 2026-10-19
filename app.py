import os
import sys
import time
import subprocess
import shutil
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

# Configure sys.path so LatticeForge python tools are accessible
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(APP_DIR, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

try:
    from tools.read_lattice_binary import load_lattice_snapshot, resolve_snapshot_path
except ImportError:
    # Fallback if running from a different root
    alt_python = os.path.join(APP_DIR, "LatticeForge", "python")
    if alt_python not in sys.path:
        sys.path.insert(0, alt_python)
    try:
        from tools.read_lattice_binary import load_lattice_snapshot, resolve_snapshot_path
    except Exception:
        load_lattice_snapshot = None
        resolve_snapshot_path = None

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LatticeForge | CUDA Potts & Kawasaki Sintering Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
    }
    
    /* Header Card */
    .lf-header-card {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .lf-title {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .lf-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .lf-metric-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(6px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .lf-metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.5);
    }
    .lf-metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .lf-metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }
    .lf-metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
    }
    
    /* Badges */
    .lf-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .lf-badge-cuda {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .lf-badge-cpu {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .lf-badge-conserved {
        background-color: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Binary & Executable Resolution Utilities
# -----------------------------------------------------------------------------
def locate_engine_binary(preferred_name="lattice_forge_cuda.exe") -> str:
    """Locates the C++ / CUDA engine executable across project directories."""
    candidates = [
        os.path.join(APP_DIR, preferred_name),
        os.path.join(APP_DIR, "build", preferred_name),
        os.path.join(APP_DIR, "..", preferred_name),
        os.path.join(APP_DIR, "..", "build", preferred_name),
        os.path.join(APP_DIR, "LatticeForge", preferred_name),
        os.path.join(APP_DIR, "LatticeForge", "build", preferred_name),
        shutil.which(preferred_name) or "",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK | os.R_OK):
            return os.path.abspath(path)

    # Fallback to CPU binary if CUDA binary not found
    if preferred_name != "lattice_forge.exe":
        return locate_engine_binary("lattice_forge.exe")
    return ""


def get_resolved_outputs_dir(base_dir="outputs") -> str:
    """Finds or creates outputs directory."""
    candidates = [
        os.path.join(APP_DIR, base_dir),
        os.path.join(APP_DIR, "..", base_dir),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    # Default to APP_DIR/outputs
    target = os.path.join(APP_DIR, base_dir)
    os.makedirs(target, exist_ok=True)
    return os.path.abspath(target)


# -----------------------------------------------------------------------------
# Subprocess Execution Hook
# -----------------------------------------------------------------------------
def run_simulation_subprocess(
    exe_path: str,
    steps: int,
    temp: float,
    h_seg: float,
    size: int,
    seed: int,
    output_dir: str,
):
    """
    Executes lattice_forge_cuda.exe as a local subprocess, capturing and streaming
    stdout/stderr logs while tracking simulation execution progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "energy_log.csv")
    bin_path = os.path.join(output_dir, "lattice_state.bin")
    radial_path = os.path.join(output_dir, "radial_profile.csv")
    centroids_path = os.path.join(output_dir, "grain_centroids.csv")

    cmd = [
        exe_path,
        "-s", str(int(steps)),
        "-t", f"{float(temp):.2f}",
        "-H", f"{float(h_seg):.2f}",
        "--size", str(int(size)),
        "--seed", str(int(seed)),
        "-o", csv_path,
        "-b", bin_path,
        "--radial", radial_path,
        "--centroids", centroids_path,
    ]

    working_dir = os.path.dirname(exe_path) if os.path.dirname(exe_path) else APP_DIR

    return cmd, working_dir, csv_path, bin_path, radial_path, centroids_path


# -----------------------------------------------------------------------------
# Microstructure Visualizer Helper
# -----------------------------------------------------------------------------
def plot_microstructure(sigma: np.ndarray, c: np.ndarray, max_display_dim: int = 512):
    """Generates side-by-side plots for grain orientation and solute distribution."""
    L = sigma.shape[0]

    # Downsample if lattice dimension is extremely large (e.g. 2048) for snappy UI responsiveness
    step = max(1, L // max_display_dim)
    sigma_disp = sigma[::step, ::step]
    c_disp = c[::step, ::step]

    # Generate Grain Boundary Map
    north = np.roll(sigma_disp, -1, axis=0)
    south = np.roll(sigma_disp, 1, axis=0)
    east = np.roll(sigma_disp, -1, axis=1)
    west = np.roll(sigma_disp, 1, axis=1)
    gb_map = (
        (sigma_disp != north) |
        (sigma_disp != south) |
        (sigma_disp != east) |
        (sigma_disp != west)
    ).astype(np.uint8)

    # Composite: Grains with Solute Highlighting
    # Build a 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0e1117")

    # Panel 1: Potts Grain Orientations
    ax1 = axes[0]
    ax1.set_facecolor("#0e1117")
    im1 = ax1.imshow(sigma_disp, cmap="turbo", origin="upper", interpolation="nearest")
    ax1.set_title(rf"Grain Orientations $\sigma$ ({L}×{L})", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar1.set_label("Orientation ID", color="#94a3b8", fontsize=9)

    # Panel 2: Kawasaki Solute Element Distribution
    ax2 = axes[1]
    ax2.set_facecolor("#0e1117")
    im2 = ax2.imshow(c_disp, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title("Solute Distribution (Ce Core-Shell)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.axis("off")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar2.ax.set_yticklabels(["Matrix", "Solute (Ce)"])
    cbar2.ax.tick_params(colors="#94a3b8", labelsize=8)

    # Panel 3: Boundary Segregation Overlay
    ax3 = axes[2]
    ax3.set_facecolor("#0e1117")
    # Base grayscale of grains + bright cyan grain boundaries + gold solute atoms
    overlay = np.zeros((sigma_disp.shape[0], sigma_disp.shape[1], 3), dtype=np.float32)
    norm_sigma = (sigma_disp % 32) / 32.0 * 0.45 + 0.1
    overlay[:, :, 0] = norm_sigma
    overlay[:, :, 1] = norm_sigma
    overlay[:, :, 2] = norm_sigma

    # Highlight GBs in Dark Slate
    overlay[gb_map == 1] = [0.25, 0.45, 0.7]
    # Highlight Solute Ce in Bright Amber / Gold
    overlay[c_disp == 1] = [1.0, 0.78, 0.18]

    ax3.imshow(overlay, origin="upper", interpolation="nearest")
    ax3.set_title("Microstructure Composite Overlay", color="#f8fafc", fontsize=12, fontweight="bold")
    ax3.axis("off")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS (User Inputs as requested)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Simulation Controls")

# 1. Lattice Dimensions
dim_options = [32, 64, 128, 256, 512, 1024, 2048]
lattice_dim = st.sidebar.select_slider(
    "Lattice Dimension (L × L)",
    options=dim_options,
    value=128,
    help="Grid dimension L (periodic boundary conditions, total sites = L²). Supports large-scale lattices up to 2048×2048 (4.19M sites) executed on CUDA hardware.",
)

# 2. Reduced Sintering Temperature (T_s)
temp_ts = st.sidebar.slider(
    "Temperature ($T_s$)",
    min_value=0.10,
    max_value=3.50,
    value=0.40,
    step=0.05,
    help="Reduced sintering temperature. Low Ts (< 0.8) yields strong grain boundary solute segregation and core-shell kinetics; high Ts (> 2.0) causes thermal dissolution and homogeneous mixing.",
)

# Temperature Regime Guidance Indicator
if temp_ts <= 0.8:
    st.sidebar.caption("🟢 **Low $T_s$**: Solute Segregation & Core-Shell Regime")
elif temp_ts <= 1.8:
    st.sidebar.caption("🟡 **Intermediate $T_s$**: Dynamic Grain Coarsening")
else:
    st.sidebar.caption("🔴 **High $T_s$**: Thermal Dissolution & Homogenization")

# 3. Segregation Strength (H_seg)
h_seg = st.sidebar.slider(
    "Segregation Strength ($H_{\\text{seg}}$)",
    min_value=0.00,
    max_value=4.00,
    value=1.50,
    step=0.10,
    help="Solute thermodynamic boundary affinity parameter. Higher values strongly drive solute atoms into grain boundary interfaces.",
)

# 4. Simulation Steps (MCS)
steps = st.sidebar.number_input(
    "Simulation Steps (MCS)",
    min_value=10,
    max_value=2000,
    value=50,
    step=10,
    help="Total Monte Carlo steps (each MCS consists of Potts grain growth sweep + Kawasaki directional solute exchange sweep).",
)

st.sidebar.markdown("---")

# Advanced Simulation Settings (Accordion)
with st.sidebar.expander("🛠️ Advanced Engine Parameters"):
    seed = st.number_input("RNG Seed", min_value=1, max_value=999999, value=42, step=1)
    
    # Auto-detect binary executable
    default_exe = locate_engine_binary("lattice_forge_cuda.exe")
    custom_exe = st.text_input("Engine Binary Path", value=default_exe)
    
    output_dir_input = st.text_input("Outputs Directory", value=get_resolved_outputs_dir())
    live_stream_toggle = st.checkbox("Live Console Streaming", value=True)

st.sidebar.markdown("---")

# 5. "Run Simulation" Button
run_button = st.sidebar.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True,
    help="Spawns lattice_forge_cuda.exe as a local subprocess with configured parameters.",
)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="lf-header-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="lf-title">LatticeForge Microstructure Dashboard</div>
                <div class="lf-subtitle">High-Performance CUDA Potts-Kawasaki Coupled Kinetics Simulator</div>
            </div>
            <div>
                <span class="lf-badge lf-badge-cuda">CUDA Acceleration</span>
                <span class="lf-badge lf-badge-conserved">Strict Invariant ΔN=0</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Executable Verification
resolved_exe = custom_exe if custom_exe and os.path.isfile(custom_exe) else locate_engine_binary()
if not resolved_exe:
    st.error(
        "⚠️ **CUDA Engine Binary Not Found**: `lattice_forge_cuda.exe` was not detected. "
        "Please build the project using `force_build.bat` or verify the path in the sidebar."
    )

# -----------------------------------------------------------------------------
# SUBPROCESS EXECUTION TRIGGER
# -----------------------------------------------------------------------------
if run_button:
    if not resolved_exe or not os.path.isfile(resolved_exe):
        st.error(f"Cannot run simulation: Executable not found at `{resolved_exe}`")
    else:
        cmd, cwd, csv_path, bin_path, radial_path, centroids_path = run_simulation_subprocess(
            exe_path=resolved_exe,
            steps=steps,
            temp=temp_ts,
            h_seg=h_seg,
            size=lattice_dim,
            seed=seed,
            output_dir=output_dir_input,
        )

        st.info(f"⚡ **Invoking Subprocess**: `{os.path.basename(resolved_exe)}` with L={lattice_dim}, Ts={temp_ts}, Hseg={h_seg}, Steps={steps}")

        status_container = st.status("🚀 Running CUDA Simulation Engine...", expanded=True)
        log_placeholder = st.empty()
        progress_bar = st.progress(0.0)

        accumulated_logs = []
        t_start = time.perf_counter()

        try:
            # Spawn local subprocess
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            current_mcs = 0
            for line in proc.stdout:
                accumulated_logs.append(line)
                # Parse step progress from stdout lines (e.g. "     5          187.00 ...")
                tokens = line.strip().split()
                if len(tokens) >= 5 and tokens[0].isdigit():
                    step_val = int(tokens[0])
                    current_mcs = max(current_mcs, step_val)
                    frac = min(1.0, current_mcs / max(1, steps))
                    progress_bar.progress(frac, text=f"Step {current_mcs}/{steps} MCS ({int(frac * 100)}%)")

                if live_stream_toggle and len(accumulated_logs) % 3 == 0:
                    tail = "".join(accumulated_logs[-15:])
                    log_placeholder.code(tail, language="bash")

            proc.wait()
            t_elapsed = time.perf_counter() - t_start

            if proc.returncode == 0:
                progress_bar.progress(1.0, text=f"Completed {steps}/{steps} MCS in {t_elapsed:.2f}s")
                status_container.update(
                    label=f"✅ Simulation Succeeded in {t_elapsed:.2f}s (Exit Code 0)",
                    state="complete",
                    expanded=False,
                )
                log_placeholder.code("".join(accumulated_logs[-25:]), language="bash")
                st.session_state["last_run_logs"] = "".join(accumulated_logs)
                st.session_state["last_run_time"] = t_elapsed
                st.session_state["last_run_success"] = True
                st.success(f"Simulation completed in {t_elapsed:.2f} seconds! Artifacts saved to `{output_dir_input}`.")
            else:
                status_container.update(
                    label=f"❌ Simulation Failed with Exit Code {proc.returncode}",
                    state="error",
                    expanded=True,
                )
                log_placeholder.code("".join(accumulated_logs), language="bash")
                st.error(f"Process terminated with exit code {proc.returncode}.")

        except Exception as e:
            status_container.update(label="❌ Subprocess Invocation Error", state="error")
            st.error(f"Failed to execute subprocess: {str(e)}")

# -----------------------------------------------------------------------------
# DATA LOADING & ARTIFACTS RESOLUTION
# -----------------------------------------------------------------------------
resolved_out_dir = output_dir_input if os.path.isdir(output_dir_input) else get_resolved_outputs_dir()
csv_file = os.path.join(resolved_out_dir, "energy_log.csv")
bin_file = os.path.join(resolved_out_dir, "lattice_state.bin")
radial_file = os.path.join(resolved_out_dir, "radial_profile.csv")
centroids_file = os.path.join(resolved_out_dir, "grain_centroids.csv")

has_energy = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
has_bin = os.path.isfile(bin_file) and os.path.getsize(bin_file) > 128
has_radial = os.path.isfile(radial_file) and os.path.getsize(radial_file) > 0
has_centroids = os.path.isfile(centroids_file) and os.path.getsize(centroids_file) > 0

# Load Energy Log
df_energy = pd.DataFrame()
if has_energy:
    try:
        df_energy = pd.read_csv(csv_file)
    except Exception as e:
        st.warning(f"Could not read energy log: {e}")

# Load Binary Lattice State
lattice_data = None
if has_bin and load_lattice_snapshot:
    try:
        lattice_data = load_lattice_snapshot(bin_file)
    except Exception as e:
        st.warning(f"Could not read binary lattice snapshot: {e}")

# Load Radial Profile
df_radial = pd.DataFrame()
if has_radial:
    try:
        df_radial = pd.read_csv(radial_file)
    except Exception as e:
        st.warning(f"Could not read radial profile: {e}")

# Load Centroids
df_centroids = pd.DataFrame()
if has_centroids:
    try:
        df_centroids = pd.read_csv(centroids_file)
    except Exception as e:
        st.warning(f"Could not read grain centroids: {e}")

# -----------------------------------------------------------------------------
# HERO KPI METRICS BANNER
# -----------------------------------------------------------------------------
if not df_energy.empty:
    h0 = df_energy["hamiltonian"].iloc[0]
    hf = df_energy["hamiltonian"].iloc[-1]
    delta_h = hf - h0
    pct_h = ((h0 - hf) / max(1e-6, abs(h0))) * 100.0

    cb_final = df_energy["c_boundary"].iloc[-1]
    cbulk_final = df_energy["c_bulk"].iloc[-1]
    enrichment_final = cb_final / max(1e-6, cbulk_final)

    total_ce_count = int(np.sum(lattice_data["c"] == 1)) if lattice_data is not None else 0
    total_sites_count = lattice_data["total_sites"] if lattice_data is not None else (lattice_dim * lattice_dim)
    ce_pct = (total_ce_count / max(1, total_sites_count)) * 100.0

    num_grains = len(df_centroids) if not df_centroids.empty else 0
    mean_radius = df_centroids["r_eff"].mean() if not df_centroids.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="lf-metric-card">
                <div class="lf-metric-label">Energy Descent (ΔH)</div>
                <div class="lf-metric-value">{"-" if delta_h < 0 else "+"}{abs(pct_h):.1f}%</div>
                <div class="lf-metric-sub">{h0:.0f} → {hf:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="lf-metric-card">
                <div class="lf-metric-label">GB Enrichment</div>
                <div class="lf-metric-value">{enrichment_final:.2f}×</div>
                <div class="lf-metric-sub">Peak / Bulk Ratio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="lf-metric-card">
                <div class="lf-metric-label">Atomic Invariant (ΔN_Ce)</div>
                <div class="lf-metric-value">0</div>
                <div class="lf-metric-sub">{total_ce_count:,} atoms ({ce_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="lf-metric-card">
                <div class="lf-metric-label">Active Grains</div>
                <div class="lf-metric-value">{num_grains}</div>
                <div class="lf-metric-sub">R_mean = {mean_radius:.1f} lu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        runtime_sec = st.session_state.get("last_run_time", 0.0)
        st.markdown(
            f"""
            <div class="lf-metric-card">
                <div class="lf-metric-label">Execution Time</div>
                <div class="lf-metric-value">{runtime_sec:.2f}s</div>
                <div class="lf-metric-sub">C++ / CUDA Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INTERACTIVE TABS VIEW
# -----------------------------------------------------------------------------
tab_micro, tab_kinetics, tab_radial, tab_binary, tab_console = st.tabs(
    [
        "🔬 Microstructure Analytics",
        "📈 Thermodynamic Kinetics",
        "🎯 Radial Core-Shell Profile",
        "💾 Binary State Payload",
        "🖥️ Subprocess Console Logs",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: MICROSTRUCTURE ANALYTICS
# -----------------------------------------------------------------------------
with tab_micro:
    if lattice_data is not None:
        L_val = lattice_data["L"]
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.markdown(
                f"**Microstructure Snapshot**: Lattice Dimension = **{L_val}×{L_val}** | "
                f"Total Sites = **{lattice_data['total_sites']:,}** | "
                f"Binary Loader Latency = **{lattice_data.get('read_time_ms', 0):.2f} ms**"
            )
        with col_m2:
            disp_res = 512
            if L_val > 512:
                res_opts = [512, 1024] if L_val == 1024 else [512, 1024, 2048]
                disp_res = st.selectbox(
                    "Display Resolution",
                    options=res_opts,
                    index=0,
                    help="Rendering resolution for large lattices up to 2048×2048.",
                )

        fig_micro = plot_microstructure(lattice_data["sigma"], lattice_data["c"], max_display_dim=disp_res)
        st.pyplot(fig_micro, use_container_width=True)
        plt.close(fig_micro)
    else:
        st.info("Run a simulation or place `lattice_state.bin` in outputs to visualize the 2D microstructure.")

# -----------------------------------------------------------------------------
# TAB 2: THERMODYNAMIC KINETICS
# -----------------------------------------------------------------------------
with tab_kinetics:
    if not df_energy.empty:
        col_k1, col_k2 = st.columns(2)

        with col_k1:
            st.markdown(r"#### Hamiltonian Energy Descent ($\mathcal{H}$ vs MCS)")
            fig_h, ax_h = plt.subplots(figsize=(7, 4.2), facecolor="#0e1117")
            ax_h.set_facecolor("#111827")
            ax_h.plot(df_energy["mcs"], df_energy["hamiltonian"], color="#38bdf8", linewidth=2.2, label=r"$\mathcal{H}$ Total")
            ax_h.set_xlabel("Monte Carlo Steps (MCS)", color="#94a3b8")
            ax_h.set_ylabel(r"Hamiltonian $\mathcal{H}$", color="#94a3b8")
            ax_h.tick_params(colors="#94a3b8")
            ax_h.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax_h.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)

        with col_k2:
            st.markdown("#### Boundary vs Bulk Solute Partitioning")
            fig_c, ax_c = plt.subplots(figsize=(7, 4.2), facecolor="#0e1117")
            ax_c.set_facecolor("#111827")
            ax_c.plot(df_energy["mcs"], df_energy["c_boundary"], color="#f59e0b", linewidth=2.2, label=r"$c_{\text{boundary}}$ (GB)")
            ax_c.plot(df_energy["mcs"], df_energy["c_bulk"], color="#a855f7", linewidth=2.2, linestyle="--", label=r"$c_{\text{bulk}}$ (Grains)")
            ax_c.set_xlabel("Monte Carlo Steps (MCS)", color="#94a3b8")
            ax_c.set_ylabel("Atomic Solute Fraction", color="#94a3b8")
            ax_c.tick_params(colors="#94a3b8")
            ax_c.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax_c.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

        # Enrichment Curve
        st.markdown("#### Quantitative Grain Boundary Enrichment Factor ($c_{\\text{boundary}} / c_{\\text{bulk}}$)")
        df_energy["enrichment"] = df_energy["c_boundary"] / df_energy["c_bulk"].clip(lower=1e-6)
        st.line_chart(df_energy, x="mcs", y="enrichment", color="#34d399")
    else:
        st.info("Energy log (`energy_log.csv`) not found. Run a simulation to plot kinetics curves.")

# -----------------------------------------------------------------------------
# TAB 3: RADIAL CORE-SHELL PROFILE
# -----------------------------------------------------------------------------
with tab_radial:
    if not df_radial.empty:
        col_r1, col_r2 = st.columns([1.2, 0.8])

        with col_r1:
            st.markdown("#### Radial Solute Concentration from Grain Core to Boundary")
            fig_r, ax_r = plt.subplots(figsize=(7.5, 4.5), facecolor="#0e1117")
            ax_r.set_facecolor("#111827")

            r_vals = df_radial["radial_distance"]
            c_vals = df_radial["ce_concentration"]
            err_vals = df_radial["std_error"]

            ax_r.plot(r_vals, c_vals, color="#f43f5e", linewidth=2.4, label="Solute $C(r)$ Profile")
            ax_r.fill_between(r_vals, c_vals - err_vals, c_vals + err_vals, color="#f43f5e", alpha=0.25, label="±1 Std Error")
            ax_r.set_xlabel("Radial Distance from Grain Centroid (lattice units)", color="#94a3b8")
            ax_r.set_ylabel("Solute Concentration $C(r)$", color="#94a3b8")
            ax_r.tick_params(colors="#94a3b8")
            ax_r.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax_r.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)

        with col_r2:
            st.markdown("#### Grain Centroids & Morphology Table")
            if not df_centroids.empty:
                st.dataframe(
                    df_centroids[["grain_id", "num_sites", "r_eff", "ce_count", "ce_fraction"]],
                    use_container_width=True,
                    height=320,
                )
            else:
                st.info("No centroid table available.")
    else:
        st.info("Radial concentration profile (`radial_profile.csv`) not found.")

# -----------------------------------------------------------------------------
# TAB 4: BINARY STATE PAYLOAD & DOWNLOADS
# -----------------------------------------------------------------------------
with tab_binary:
    st.markdown("#### 128-Byte Binary Snapshot Header Inspector")
    if lattice_data is not None and "header" in lattice_data:
        hdr = lattice_data["header"]
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.code(
                f"Magic Bytes: {hdr.get('magic')}\n"
                f"Format Version: {hdr.get('version')}\n"
                f"Header Size: {hdr.get('header_size')} bytes",
                language="yaml",
            )
        with h_col2:
            st.code(
                f"Dimensions: {hdr.get('dim_x')} × {hdr.get('dim_y')}\n"
                f"Total Sites: {hdr.get('total_sites'):,}\n"
                f"Simulation Step: {hdr.get('mcs_step')} MCS",
                language="yaml",
            )
        with h_col3:
            st.code(
                f"Temperature Ts: {hdr.get('temp')}\n"
                f"Potts J_gb: {hdr.get('j_gb')}\n"
                f"Segregation H_seg: {hdr.get('h_seg')}",
                language="yaml",
            )

    # Download Artifacts Section
    st.markdown("#### 📥 Export Simulation Artifacts")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    with d_col1:
        if has_bin:
            with open(bin_file, "rb") as f:
                st.download_button(
                    "📦 Download .BIN State",
                    data=f.read(),
                    file_name="lattice_state.bin",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
    with d_col2:
        if has_energy:
            with open(csv_file, "rb") as f:
                st.download_button(
                    "📊 Download Energy CSV",
                    data=f.read(),
                    file_name="energy_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with d_col3:
        if has_radial:
            with open(radial_file, "rb") as f:
                st.download_button(
                    "🎯 Download Radial CSV",
                    data=f.read(),
                    file_name="radial_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with d_col4:
        if has_centroids:
            with open(centroids_file, "rb") as f:
                st.download_button(
                    "📐 Download Centroids CSV",
                    data=f.read(),
                    file_name="grain_centroids.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# -----------------------------------------------------------------------------
# TAB 5: SUBPROCESS CONSOLE LOGS
# -----------------------------------------------------------------------------
with tab_console:
    st.markdown("#### Real-Time Engine Subprocess Execution Log")
    logs = st.session_state.get("last_run_logs", "")
    if logs:
        st.code(logs, language="bash")
    else:
        st.info("Click 'Run Simulation' in the sidebar to stream live engine stdout/stderr logs here.")
