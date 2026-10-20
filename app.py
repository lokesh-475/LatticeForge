"""
LatticeForge Dashboard Server
─────────────────────────────
Flask application that drives the C++/CUDA simulation engine and serves
a research-grade single-page dashboard for Nd-Ce-Fe-B sintering analytics.

Endpoints
  GET  /                        → dashboard SPA
  GET  /api/engine/status       → engine binary detection
  POST /api/simulation/run      → launch simulation subprocess
  GET  /api/simulation/stream   → SSE live console + progress
  GET  /api/data/energy         → energy_log.csv as JSON
  GET  /api/data/lattice        → parsed binary snapshot (downsampled)
  GET  /api/data/radial         → radial_profile.csv as JSON
  GET  /api/data/centroids      → grain_centroids.csv as JSON
  GET  /api/data/header         → 128-byte binary header metadata
  GET  /api/data/files          → list available output artifacts
  GET  /api/download/<filename> → download raw output file
"""

import os
import sys
import json
import time
import struct
import shutil
import threading
import subprocess
import csv
import base64
from io import BytesIO

import numpy as np

from flask import (
    Flask, render_template, jsonify, request, Response, send_file, abort
)

# ─── Path Setup ──────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(APP_DIR, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

try:
    from tools.read_lattice_binary import load_lattice_snapshot
except ImportError:
    load_lattice_snapshot = None

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "dashboard", "templates"),
    static_folder=os.path.join(APP_DIR, "dashboard", "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ─── Global Simulation State ────────────────────────────────────────────────
sim_state = {
    "running": False,
    "logs": [],
    "progress": 0.0,
    "current_step": 0,
    "total_steps": 0,
    "elapsed": 0.0,
    "return_code": None,
    "finished": False,
}
sim_lock = threading.Lock()


# ─── Engine Resolution ──────────────────────────────────────────────────────
def locate_engine(preferred="lattice_forge_cuda.exe"):
    candidates = [
        os.path.join(APP_DIR, preferred),
        os.path.join(APP_DIR, "build", preferred),
        os.path.join(APP_DIR, "..", preferred),
        os.path.join(APP_DIR, "..", "build", preferred),
        shutil.which(preferred) or "",
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK | os.R_OK):
            return os.path.abspath(p)
    if preferred != "lattice_forge.exe":
        return locate_engine("lattice_forge.exe")
    return ""


def resolve_outputs_dir():
    for d in [os.path.join(APP_DIR, "outputs"), os.path.join(APP_DIR, "..", "outputs")]:
        if os.path.isdir(d):
            return os.path.abspath(d)
    target = os.path.join(APP_DIR, "outputs")
    os.makedirs(target, exist_ok=True)
    return os.path.abspath(target)


# ─── Turbo Colormap (for client-side rendering in base64 PNG) ────────────
def _build_turbo_lut():
    """Pre-compute a 256-entry turbo colormap as [r,g,b] uint8 arrays."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("turbo", 256)
        lut = []
        for i in range(256):
            rgba = cmap(i / 255.0)
            lut.append([int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)])
        return lut
    except Exception:
        # Fallback: grayscale
        return [[i, i, i] for i in range(256)]

TURBO_LUT = _build_turbo_lut()


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/engine/status")
def engine_status():
    exe = locate_engine()
    return jsonify({
        "found": bool(exe),
        "path": exe,
        "name": os.path.basename(exe) if exe else "",
        "outputs_dir": resolve_outputs_dir(),
    })


@app.route("/api/simulation/run", methods=["POST"])
def run_simulation():
    global sim_state

    with sim_lock:
        if sim_state["running"]:
            return jsonify({"error": "Simulation already running"}), 409

    params = request.get_json(force=True)
    exe = locate_engine()
    if not exe:
        return jsonify({"error": "Engine binary not found"}), 404

    out_dir = resolve_outputs_dir()
    os.makedirs(out_dir, exist_ok=True)

    steps = int(params.get("steps", 40))
    temp = float(params.get("temp", 0.4))
    h_seg = float(params.get("h_seg", 2.5))
    size = int(params.get("size", 128))
    seed = int(params.get("seed", 42))

    csv_path = os.path.join(out_dir, "energy_log.csv")
    bin_path = os.path.join(out_dir, "lattice_state.bin")
    radial_path = os.path.join(out_dir, "radial_profile.csv")
    centroids_path = os.path.join(out_dir, "grain_centroids.csv")

    cmd = [
        exe,
        "-s", str(steps),
        "-t", f"{temp:.2f}",
        "-H", f"{h_seg:.2f}",
        "--size", str(size),
        "--seed", str(seed),
        "-o", csv_path,
        "-b", bin_path,
        "--radial", radial_path,
        "--centroids", centroids_path,
    ]

    with sim_lock:
        sim_state = {
            "running": True,
            "logs": [],
            "progress": 0.0,
            "current_step": 0,
            "total_steps": steps,
            "elapsed": 0.0,
            "return_code": None,
            "finished": False,
        }

    def _run():
        global sim_state
        t0 = time.perf_counter()
        cwd = os.path.dirname(exe) or APP_DIR
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                with sim_lock:
                    sim_state["logs"].append(line)
                    tokens = line.strip().split()
                    if len(tokens) >= 5 and tokens[0].isdigit():
                        step_val = int(tokens[0])
                        sim_state["current_step"] = max(sim_state["current_step"], step_val)
                        sim_state["progress"] = min(1.0, step_val / max(1, steps))
                    sim_state["elapsed"] = time.perf_counter() - t0

            proc.wait()
            with sim_lock:
                sim_state["return_code"] = proc.returncode
                sim_state["elapsed"] = time.perf_counter() - t0
                sim_state["progress"] = 1.0 if proc.returncode == 0 else sim_state["progress"]
                sim_state["running"] = False
                sim_state["finished"] = True

        except Exception as e:
            with sim_lock:
                sim_state["logs"].append(f"ERROR: {str(e)}\n")
                sim_state["running"] = False
                sim_state["finished"] = True
                sim_state["return_code"] = -1

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"status": "started", "command": " ".join(cmd)})


@app.route("/api/simulation/stream")
def simulation_stream():
    """Server-Sent Events stream for live console output and progress."""
    def generate():
        last_idx = 0
        while True:
            with sim_lock:
                logs = sim_state["logs"][last_idx:]
                last_idx += len(logs)
                data = {
                    "running": sim_state["running"],
                    "progress": sim_state["progress"],
                    "step": sim_state["current_step"],
                    "total": sim_state["total_steps"],
                    "elapsed": round(sim_state["elapsed"], 3),
                    "return_code": sim_state["return_code"],
                    "finished": sim_state["finished"],
                    "new_logs": logs,
                }
            yield f"data: {json.dumps(data)}\n\n"
            if data["finished"] and not logs:
                break
            time.sleep(0.25)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/data/energy")
def data_energy():
    out_dir = resolve_outputs_dir()
    csv_path = os.path.join(out_dir, "energy_log.csv")
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return jsonify({"error": "No energy data", "rows": []}), 200

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mcs": int(row["mcs"]),
                "hamiltonian": float(row["hamiltonian"]),
                "c_boundary": float(row["c_boundary"]),
                "c_bulk": float(row["c_bulk"]),
                "temp": float(row["temp"]),
            })
    return jsonify({"rows": rows})


@app.route("/api/data/radial")
def data_radial():
    out_dir = resolve_outputs_dir()
    path = os.path.join(out_dir, "radial_profile.csv")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return jsonify({"error": "No radial data", "rows": []}), 200

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "radial_distance": float(row["radial_distance"]),
                "ce_concentration": float(row["ce_concentration"]),
                "sample_count": int(row["sample_count"]),
                "std_error": float(row["std_error"]),
                "normalized_distance": float(row["normalized_distance"]),
            })
    return jsonify({"rows": rows})


@app.route("/api/data/centroids")
def data_centroids():
    out_dir = resolve_outputs_dir()
    path = os.path.join(out_dir, "grain_centroids.csv")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return jsonify({"error": "No centroids data", "rows": []}), 200

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "grain_id": int(row["grain_id"]),
                "orientation": int(row["orientation"]),
                "centroid_y": float(row["centroid_y"]),
                "centroid_x": float(row["centroid_x"]),
                "num_sites": int(row["num_sites"]),
                "r_max": float(row["r_max"]),
                "r_eff": float(row["r_eff"]),
                "ce_count": int(row["ce_count"]),
                "ce_fraction": float(row["ce_fraction"]),
            })
    return jsonify({"rows": rows})


@app.route("/api/data/header")
def data_header():
    out_dir = resolve_outputs_dir()
    bin_path = os.path.join(out_dir, "lattice_state.bin")
    if not os.path.isfile(bin_path) or os.path.getsize(bin_path) < 128:
        return jsonify({"error": "No binary snapshot"}), 200

    fmt = "<8sIIIIQIIiIdddd48s"
    with open(bin_path, "rb") as f:
        header_data = f.read(128)

    u = struct.unpack(fmt, header_data)
    return jsonify({
        "magic": u[0].decode("ascii", errors="replace"),
        "header_size": u[1],
        "version": u[2],
        "dim_x": u[3],
        "dim_y": u[4],
        "total_sites": u[5],
        "bytes_per_sigma": u[6],
        "bytes_per_c": u[7],
        "mcs_step": u[8],
        "flags": u[9],
        "temp": round(u[10], 4),
        "j_gb": round(u[11], 4),
        "h_seg": round(u[12], 4),
        "j_chem": round(u[13], 4),
    })


@app.route("/api/data/lattice")
def data_lattice():
    """
    Returns the lattice snapshot as base64-encoded RGBA image data for each panel,
    plus scalar statistics. This avoids sending multi-MB raw arrays to the browser.
    """
    out_dir = resolve_outputs_dir()
    bin_path = os.path.join(out_dir, "lattice_state.bin")

    if not os.path.isfile(bin_path) or os.path.getsize(bin_path) < 128:
        return jsonify({"error": "No lattice data"}), 200

    if load_lattice_snapshot is None:
        return jsonify({"error": "Binary reader not available"}), 500

    try:
        data = load_lattice_snapshot(bin_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sigma = data["sigma"]  # uint16 [L, L]
    c = data["c"]          # uint8  [L, L]
    L = data["L"]

    # Downsample for transfer if L > 512
    step = max(1, L // 512)
    sigma_d = sigma[::step, ::step]
    c_d = c[::step, ::step]
    display_dim = sigma_d.shape[0]

    # --- Panel 1: Grain orientations via turbo colormap ---
    sigma_norm = (sigma_d.astype(np.float32) / max(1, sigma_d.max()) * 255).astype(np.uint8)
    grain_rgba = np.zeros((display_dim, display_dim, 4), dtype=np.uint8)
    for i in range(256):
        mask = (sigma_norm == i)
        grain_rgba[mask, 0] = TURBO_LUT[i][0]
        grain_rgba[mask, 1] = TURBO_LUT[i][1]
        grain_rgba[mask, 2] = TURBO_LUT[i][2]
        grain_rgba[mask, 3] = 255

    # --- Panel 2: Solute distribution (magma-like) ---
    solute_rgba = np.zeros((display_dim, display_dim, 4), dtype=np.uint8)
    solute_rgba[:, :, 3] = 255
    # matrix = dark (#0A0A0A), solute = bright yellow (#FFE600)
    solute_rgba[c_d == 0, 0] = 10
    solute_rgba[c_d == 0, 1] = 10
    solute_rgba[c_d == 0, 2] = 10
    solute_rgba[c_d == 1, 0] = 255
    solute_rgba[c_d == 1, 1] = 230
    solute_rgba[c_d == 1, 2] = 0

    # --- Panel 3: Composite overlay ---
    north = np.roll(sigma_d, -1, axis=0)
    south = np.roll(sigma_d, 1, axis=0)
    east = np.roll(sigma_d, -1, axis=1)
    west = np.roll(sigma_d, 1, axis=1)
    gb = ((sigma_d != north) | (sigma_d != south) |
          (sigma_d != east) | (sigma_d != west)).astype(np.uint8)

    comp_rgba = np.zeros((display_dim, display_dim, 4), dtype=np.uint8)
    comp_rgba[:, :, 3] = 255
    base_gray = ((sigma_d % 32).astype(np.float32) / 32.0 * 80 + 15).astype(np.uint8)
    comp_rgba[:, :, 0] = base_gray
    comp_rgba[:, :, 1] = base_gray
    comp_rgba[:, :, 2] = base_gray
    # GB = dim gray
    comp_rgba[gb == 1, 0] = 56
    comp_rgba[gb == 1, 1] = 56
    comp_rgba[gb == 1, 2] = 56
    # Solute = bright accent
    comp_rgba[c_d == 1, 0] = 0
    comp_rgba[c_d == 1, 1] = 229
    comp_rgba[c_d == 1, 2] = 255

    def _to_b64_png(rgba_array):
        from PIL import Image
        img = Image.fromarray(rgba_array, "RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Statistics
    total_ce = int(np.sum(c == 1))
    total_sites = int(L * L)

    try:
        grain_png = _to_b64_png(grain_rgba)
        solute_png = _to_b64_png(solute_rgba)
        comp_png = _to_b64_png(comp_rgba)
    except ImportError:
        # Pillow not available, send raw base64 of flat RGBA bytes
        grain_png = base64.b64encode(grain_rgba.tobytes()).decode("ascii")
        solute_png = base64.b64encode(solute_rgba.tobytes()).decode("ascii")
        comp_png = base64.b64encode(comp_rgba.tobytes()).decode("ascii")

    return jsonify({
        "L": L,
        "display_dim": display_dim,
        "total_sites": total_sites,
        "total_ce": total_ce,
        "ce_pct": round(total_ce / max(1, total_sites) * 100, 2),
        "num_unique_grains": int(len(np.unique(sigma))),
        "read_time_ms": round(data.get("read_time_ms", 0), 3),
        "grain_png": grain_png,
        "solute_png": solute_png,
        "composite_png": comp_png,
    })


@app.route("/api/data/files")
def data_files():
    out_dir = resolve_outputs_dir()
    files = []
    for fname in os.listdir(out_dir):
        fpath = os.path.join(out_dir, fname)
        if os.path.isfile(fpath):
            files.append({
                "name": fname,
                "size": os.path.getsize(fpath),
            })
    return jsonify({"files": files, "dir": out_dir})


@app.route("/api/download/<filename>")
def download_file(filename):
    out_dir = resolve_outputs_dir()
    fpath = os.path.join(out_dir, filename)
    if not os.path.isfile(fpath):
        abort(404)
    return send_file(fpath, as_attachment=True)


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  LATTICEFORGE // DASHBOARD SERVER")
    print(f"  Engine: {locate_engine() or 'NOT FOUND'}")
    print(f"  Outputs: {resolve_outputs_dir()}")
    print("=" * 64)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
