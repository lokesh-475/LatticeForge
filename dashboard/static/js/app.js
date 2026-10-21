/* ════════════════════════════════════════════════════════════════════════════
   LATTICEFORGE // DASHBOARD — Client Application
   ════════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── Chart.js Global Defaults ──────────────────────────────────────────────
  if (window.Chart) {
    Chart.defaults.color = "#A1A1AA";
    Chart.defaults.borderColor = "#27272A";
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 10;
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.padding = 12;
    if (Chart.defaults.scale && Chart.defaults.scale.grid) {
      Chart.defaults.scale.grid.color = "#1C1C22";
    }
  }

  // ── DOM Refs ──────────────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // Sidebar controls
  const sizeSelect    = $("#param-size");
  const tempSlider    = $("#param-temp");
  const tempValue     = $("#param-temp-value");
  const tempRegime    = $("#temp-regime");
  const ceSlider      = $("#param-ce");
  const ceValue       = $("#param-ce-value");
  const hsegSlider    = $("#param-hseg");
  const hsegValue     = $("#param-hseg-value");
  const mcsSlider     = $("#param-mcs");
  const mcsValue      = $("#param-mcs-value");
  const mcsInput      = $("#param-mcs-input");
  const mcsPresets    = $$(".btn-preset");
  const seedInput     = $("#param-seed");
  const btnExecute    = $("#btn-execute");
  const btnText       = $("#btn-execute-text");
  const engineStatus  = $("#engine-status");

  // Telemetry
  const telemEnergyVal  = $("#telem-energy-val");
  const telemEnergySub  = $("#telem-energy-sub");
  const telemEnrichVal  = $("#telem-enrich-val");
  const telemEnrichSub  = $("#telem-enrich-sub");
  const telemInvarVal   = $("#telem-invar-val");
  const telemInvarSub   = $("#telem-invar-sub");
  const telemGrainsVal  = $("#telem-grains-val");
  const telemGrainsSub  = $("#telem-grains-sub");
  const telemTimeVal    = $("#telem-time-val");
  const telemTimeSub    = $("#telem-time-sub");

  // Tabs
  const tabBtns    = $$(".tab-btn");
  const tabPanels  = $$(".tab-panel");

  // Panels
  const microGrid       = $("#micro-grid-content");
  const consoleOutput   = $("#console-output");
  const progressWrap    = $("#progress-wrap");
  const progressFill    = $("#progress-fill");
  const progressText    = $("#progress-text");

  // Charts
  let chartOverviewHamiltonian = null;
  let chartOverviewPartition   = null;
  let chartOverviewEnrichment  = null;
  let chartOverviewRadial      = null;
  let chartHamiltonian         = null;
  let chartPartition           = null;
  let chartEnrichment          = null;
  let chartRadial              = null;

  // ── Tab Navigation & Chart Resizing ───────────────────────────────────────
  function resizeAllCharts() {
    [
      chartOverviewHamiltonian,
      chartOverviewPartition,
      chartOverviewEnrichment,
      chartOverviewRadial,
      chartHamiltonian,
      chartPartition,
      chartEnrichment,
      chartRadial,
    ].forEach((c) => {
      if (c) {
        try {
          c.resize();
          c.update();
        } catch (e) {}
      }
    });
  }

  function switchToTab(target) {
    tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === target));
    tabPanels.forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));

    setTimeout(() => {
      resizeAllCharts();
      window.dispatchEvent(new Event("resize"));
    }, 60);

    // Lazy-load or refresh data for the target tab
    if (target === "overview") loadAllData();
    else if (target === "kinetics") loadEnergyData();
    else if (target === "radial") loadRadialData();
    else if (target === "microstructure") loadLatticeData();
    else if (target === "binary") loadHeaderData();
  }

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      switchToTab(btn.dataset.tab);
    });
  });

  // ── Slider Live Updates ───────────────────────────────────────────────────
  function updateRegime(t) {
    if (t <= 0.8) {
      tempRegime.textContent = "SEGREGATION / CORE-SHELL";
      tempRegime.className = "regime-tag segregation";
    } else if (t <= 1.8) {
      tempRegime.textContent = "DYNAMIC COARSENING";
      tempRegime.className = "regime-tag coarsening";
    } else {
      tempRegime.textContent = "THERMAL DISSOLUTION";
      tempRegime.className = "regime-tag dissolution";
    }
  }

  tempSlider.addEventListener("input", () => {
    const v = parseFloat(tempSlider.value);
    tempValue.textContent = v.toFixed(2);
    updateRegime(v);
  });
  ceSlider.addEventListener("input", () => {
    ceValue.textContent = parseFloat(ceSlider.value).toFixed(2);
  });
  hsegSlider.addEventListener("input", () => {
    hsegValue.textContent = parseFloat(hsegSlider.value).toFixed(1);
  });
  function syncMcs(val) {
    const v = Math.max(10, Math.min(10000, parseInt(val) || 10));
    mcsSlider.value = v;
    if (mcsInput) mcsInput.value = v;
    if (mcsValue) mcsValue.textContent = v;
    mcsPresets.forEach((b) => {
      b.classList.toggle("active", parseInt(b.dataset.mcs) === v);
    });
  }

  mcsSlider.addEventListener("input", () => {
    syncMcs(mcsSlider.value);
  });
  if (mcsInput) {
    mcsInput.addEventListener("input", () => {
      syncMcs(mcsInput.value);
    });
  }
  mcsPresets.forEach((btn) => {
    btn.addEventListener("click", () => {
      syncMcs(btn.dataset.mcs);
    });
  });

  // ── Engine Status Check ───────────────────────────────────────────────────
  async function checkEngine() {
    try {
      const res = await fetch("/api/engine/status");
      const data = await res.json();
      if (data.found) {
        engineStatus.innerHTML = `<span class="status-dot online"></span>${data.name}`;
      } else {
        engineStatus.innerHTML = `<span class="status-dot offline"></span>ENGINE NOT FOUND`;
      }
    } catch {
      engineStatus.innerHTML = `<span class="status-dot offline"></span>SERVER ERROR`;
    }
  }

  // ── Execute Simulation ────────────────────────────────────────────────────
  btnExecute.addEventListener("click", async () => {
    btnExecute.disabled = true;
    btnText.innerHTML = '<span class="spinner"></span>RUNNING...';
    consoleOutput.textContent = "";
    progressFill.style.width = "0%";
    progressText.textContent = "";
    progressWrap.style.display = "block";

    const params = {
      size: parseInt(sizeSelect.value),
      temp: parseFloat(tempSlider.value),
      h_seg: parseFloat(hsegSlider.value),
      steps: parseInt(mcsInput ? mcsInput.value : mcsSlider.value),
      seed: parseInt(seedInput.value),
    };

    try {
      const res = await fetch("/api/simulation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      if (!res.ok) {
        const err = await res.json();
        consoleOutput.textContent = `ERROR: ${err.error || "Unknown error"}`;
        resetButton();
        return;
      }

      // Start SSE stream
      const evtSource = new EventSource("/api/simulation/stream");
      evtSource.onmessage = (event) => {
        const d = JSON.parse(event.data);

        // Append logs
        if (d.new_logs && d.new_logs.length > 0) {
          consoleOutput.textContent += d.new_logs.join("");
          consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }

        // Progress
        const pct = Math.round(d.progress * 100);
        progressFill.style.width = pct + "%";
        progressText.textContent = `STEP ${d.step}/${d.total} MCS [${pct}%] — ${d.elapsed.toFixed(2)}s`;

        // Finished
        if (d.finished) {
          evtSource.close();
          if (d.return_code === 0) {
            progressText.textContent = `COMPLETED ${d.total}/${d.total} MCS [${d.elapsed.toFixed(2)}s]`;
            progressFill.style.background = "var(--success)";
            // Reload all data panels and automatically switch to Workstation Overview to display graphs
            loadAllData().then(() => {
              setTimeout(() => {
                switchToTab("overview");
              }, 700);
            });
          } else {
            progressText.textContent = `FAILED [EXIT ${d.return_code}]`;
            progressFill.style.background = "var(--error)";
          }
          resetButton();
        }
      };
      evtSource.onerror = () => {
        evtSource.close();
        resetButton();
      };

      // Switch to live console tab while running
      switchToTab("console");

    } catch (e) {
      consoleOutput.textContent = `NETWORK ERROR: ${e.message}`;
      resetButton();
    }
  });

  function resetButton() {
    btnExecute.disabled = false;
    btnText.innerHTML = "EXECUTE SIMULATION";
    progressFill.style.background = "var(--accent)";
  }

  // ── Data Loading Functions ────────────────────────────────────────────────

  async function loadAllData() {
    await Promise.all([
      loadEnergyData(),
      loadLatticeData(),
      loadRadialData(),
      loadHeaderData(),
      loadCentroidsData(),
    ]);
  }

  // ── Helper to safely create/replace Chart.js instances ─────────────────
  function renderChartInstance(currentChart, canvasId, type, data, xTitle, yTitle, customOptions = {}) {
    const canvas = $(canvasId);
    if (!canvas || !window.Chart) return null;
    if (currentChart) {
      try { currentChart.destroy(); } catch (e) {}
    }
    return new Chart(canvas, {
      type: type,
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: true,
            labels: {
              boxWidth: 10,
              padding: 10,
              color: "#A1A1AA",
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: xTitle, color: "#71717A" },
            grid: { color: "#1C1C22" },
            ticks: { color: "#71717A", font: { size: 9 } }
          },
          y: {
            title: { display: true, text: yTitle, color: "#71717A" },
            grid: { color: "#1C1C22" },
            ticks: { color: "#71717A", font: { size: 9 } }
          }
        },
        ...customOptions
      }
    });
  }

  // ── Energy / Kinetics ──
  async function loadEnergyData() {
    try {
      const res = await fetch("/api/data/energy");
      const data = await res.json();
      if (!data.rows || data.rows.length === 0) return;

      const rows = data.rows;
      const allH = rows.map((r) => r.hamiltonian);
      const allEnrich = rows.map((r) =>
        r.c_bulk > 1e-6 ? r.c_boundary / r.c_bulk : 0
      );

      // Telemetry update (using full dataset precision)
      const h0 = allH[0], hf = allH[allH.length - 1];
      const dh = hf - h0;
      const pctH = Math.abs(((h0 - hf) / Math.max(1e-6, Math.abs(h0))) * 100);
      const sign = dh < 0 ? "\u2212" : "+";
      telemEnergyVal.textContent = `${sign}${pctH.toFixed(1)}%`;
      telemEnergySub.textContent = `${h0.toFixed(0)} → ${hf.toFixed(0)}`;

      const enrichFinal = allEnrich[allEnrich.length - 1];
      telemEnrichVal.textContent = `${enrichFinal.toFixed(2)}×`;
      telemEnrichSub.textContent = "C_boundary / C_bulk";

      // Decimate data for smooth 60fps Chart.js rendering when steps <= 10000
      let plotRows = rows;
      if (rows.length > 1500) {
        const stride = Math.ceil(rows.length / 800);
        plotRows = rows.filter((_, idx) => idx % stride === 0 || idx === rows.length - 1);
      }

      const mcs = plotRows.map((r) => r.mcs);
      const H = plotRows.map((r) => r.hamiltonian);
      const cb = plotRows.map((r) => r.c_boundary);
      const cbulk = plotRows.map((r) => r.c_bulk);
      const enrich = plotRows.map((r) =>
        r.c_bulk > 1e-6 ? r.c_boundary / r.c_bulk : 0
      );

      // 1. Hamiltonian Chart
      const dataH = {
        labels: mcs,
        datasets: [{
          label: "Hamiltonian H",
          data: H,
          borderColor: "#00E5FF",
          backgroundColor: "rgba(0,229,255,0.08)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }],
      };
      chartOverviewHamiltonian = renderChartInstance(chartOverviewHamiltonian, "#chart-overview-hamiltonian", "line", dataH, "MCS", "H");
      chartHamiltonian = renderChartInstance(chartHamiltonian, "#chart-hamiltonian", "line", dataH, "MCS", "H");

      // 2. Partition Chart
      const dataP = {
        labels: mcs,
        datasets: [
          {
            label: "c_boundary",
            data: cb,
            borderColor: "#00E5FF",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: "c_bulk",
            data: cbulk,
            borderColor: "#7C3AED",
            borderWidth: 2,
            pointRadius: 0,
            borderDash: [6, 3],
            tension: 0.3,
          },
        ],
      };
      chartOverviewPartition = renderChartInstance(chartOverviewPartition, "#chart-overview-partition", "line", dataP, "MCS", "Solute Fraction");
      chartPartition = renderChartInstance(chartPartition, "#chart-partition", "line", dataP, "MCS", "Solute Fraction");

      // 3. Enrichment Chart
      const dataE = {
        labels: mcs,
        datasets: [{
          label: "Enrichment Ratio (c_boundary / c_bulk)",
          data: enrich,
          borderColor: "#FBBF24",
          backgroundColor: "rgba(251,191,36,0.06)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }],
      };
      chartOverviewEnrichment = renderChartInstance(chartOverviewEnrichment, "#chart-overview-enrichment", "line", dataE, "MCS", "Enrichment Ratio");
      chartEnrichment = renderChartInstance(chartEnrichment, "#chart-enrichment", "line", dataE, "MCS", "Enrichment Ratio");

    } catch (e) {
      console.error("loadEnergyData:", e);
    }
  }

  // ── Lattice / Microstructure ──
  async function loadLatticeData() {
    try {
      const res = await fetch("/api/data/lattice");
      const data = await res.json();
      if (data.error) {
        if (microGrid) microGrid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">NO LATTICE DATA — Run a simulation first.</div>`;
        return;
      }

      // Populate Overview preview panel
      const gWrap = $("#overview-grain-wrap");
      const sWrap = $("#overview-solute-wrap");
      const cWrap = $("#overview-comp-wrap");
      if (gWrap) gWrap.innerHTML = `<img src="data:image/png;base64,${data.grain_png}" alt="Grain orientations" style="width:100%;height:auto;display:block;image-rendering:pixelated;" />`;
      if (sWrap) sWrap.innerHTML = `<img src="data:image/png;base64,${data.solute_png}" alt="Solute distribution" style="width:100%;height:auto;display:block;image-rendering:pixelated;" />`;
      if (cWrap) cWrap.innerHTML = `<img src="data:image/png;base64,${data.composite_png}" alt="Composite overlay" style="width:100%;height:auto;display:block;image-rendering:pixelated;" />`;

      // Populate Expanded Microstructure tab
      if (microGrid) {
        microGrid.innerHTML = `
          <div class="micro-panel">
            <div class="micro-panel-title">GRAIN ORIENTATIONS [${data.L}×${data.L}]</div>
            <img src="data:image/png;base64,${data.grain_png}" alt="Grain orientations" />
          </div>
          <div class="micro-panel">
            <div class="micro-panel-title">Ce SOLUTE DISTRIBUTION</div>
            <img src="data:image/png;base64,${data.solute_png}" alt="Solute distribution" />
          </div>
          <div class="micro-panel">
            <div class="micro-panel-title">COMPOSITE OVERLAY</div>
            <img src="data:image/png;base64,${data.composite_png}" alt="Composite overlay" />
          </div>
        `;
      }

      // Update telemetry
      telemInvarVal.textContent = "0";
      telemInvarSub.textContent = `${data.total_ce.toLocaleString()} atoms (${data.ce_pct}%)`;
      telemGrainsVal.textContent = data.num_unique_grains;
      telemGrainsSub.textContent = `${data.L}×${data.L} lattice`;

    } catch (e) {
      console.error("loadLatticeData:", e);
    }
  }

  // ── Radial Profile ──
  async function loadRadialData() {
    try {
      const [radRes, centRes] = await Promise.all([
        fetch("/api/data/radial"),
        fetch("/api/data/centroids"),
      ]);
      const radData = await radRes.json();
      const centData = await centRes.json();

      if (radData.rows && radData.rows.length > 0) {
        const r = radData.rows.map((x) => x.radial_distance.toFixed(2));
        const c = radData.rows.map((x) => x.ce_concentration);
        const err = radData.rows.map((x) => x.std_error);
        const upper = c.map((v, i) => v + err[i]);
        const lower = c.map((v, i) => v - err[i]);

        const dataR = {
          labels: r,
          datasets: [
            {
              label: "C(r)",
              data: c,
              borderColor: "#00E5FF",
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.3,
              order: 1,
            },
            {
              label: "+1 SE",
              data: upper,
              borderColor: "transparent",
              backgroundColor: "rgba(0,229,255,0.1)",
              pointRadius: 0,
              fill: "+1",
              order: 2,
            },
            {
              label: "-1 SE",
              data: lower,
              borderColor: "transparent",
              backgroundColor: "rgba(0,229,255,0.1)",
              pointRadius: 0,
              fill: false,
              order: 2,
            },
          ],
        };

        const customR = {
          plugins: {
            legend: {
              display: true,
              labels: {
                filter: (item) => item.text === "C(r)",
                color: "#A1A1AA",
                font: { family: "'JetBrains Mono', monospace", size: 10 }
              }
            }
          }
        };

        chartOverviewRadial = renderChartInstance(chartOverviewRadial, "#chart-overview-radial", "line", dataR, "Radial Distance (lu)", "C(r)", customR);
        chartRadial = renderChartInstance(chartRadial, "#chart-radial", "line", dataR, "Radial Distance (lu)", "C(r)", customR);
      }

      // Centroids table
      const tableBody = $("#centroids-table-body");
      if (tableBody && centData.rows && centData.rows.length > 0) {
        tableBody.innerHTML = centData.rows
          .map(
            (g) => `<tr>
              <td>${g.grain_id}</td>
              <td>${g.num_sites}</td>
              <td>${g.r_eff.toFixed(2)}</td>
              <td>${g.ce_count}</td>
              <td>${g.ce_fraction.toFixed(4)}</td>
            </tr>`
          )
          .join("");
      }
    } catch (e) {
      console.error("loadRadialData:", e);
    }
  }

  // ── Binary Header ──
  async function loadHeaderData() {
    try {
      const res = await fetch("/api/data/header");
      const h = await res.json();
      if (h.error) return;

      const block1 = $("#header-block-1");
      const block2 = $("#header-block-2");
      const block3 = $("#header-block-3");
      if (block1)
        block1.innerHTML = `
          <span class="hb-key">Magic Bytes:</span> <span class="hb-val">${h.magic}</span><br>
          <span class="hb-key">Format Version:</span> <span class="hb-val">${h.version}</span><br>
          <span class="hb-key">Header Size:</span> <span class="hb-val">${h.header_size} bytes</span>`;
      if (block2)
        block2.innerHTML = `
          <span class="hb-key">Dimensions:</span> <span class="hb-val">${h.dim_x} × ${h.dim_y}</span><br>
          <span class="hb-key">Total Sites:</span> <span class="hb-val">${h.total_sites.toLocaleString()}</span><br>
          <span class="hb-key">MCS Step:</span> <span class="hb-val">${h.mcs_step}</span>`;
      if (block3)
        block3.innerHTML = `
          <span class="hb-key">Temperature Ts:</span> <span class="hb-val">${h.temp}</span><br>
          <span class="hb-key">Potts J_gb:</span> <span class="hb-val">${h.j_gb}</span><br>
          <span class="hb-key">Segregation H_seg:</span> <span class="hb-val">${h.h_seg}</span>`;

      // Load file list for downloads
      const filesRes = await fetch("/api/data/files");
      const filesData = await filesRes.json();
      const downloadGrid = $("#download-grid");
      if (downloadGrid && filesData.files) {
        const targets = [
          "lattice_state.bin",
          "energy_log.csv",
          "radial_profile.csv",
          "grain_centroids.csv",
        ];
        downloadGrid.innerHTML = targets
          .filter((name) => filesData.files.some((f) => f.name === name))
          .map(
            (name) =>
              `<a class="btn-download" href="/api/download/${name}" download>${name.toUpperCase()}</a>`
          )
          .join("");
      }
    } catch (e) {
      console.error("loadHeaderData:", e);
    }
  }

  // ── Centroids (loaded with radial) ──
  async function loadCentroidsData() {
    // Already loaded in loadRadialData
  }

  // ── Initialize ────────────────────────────────────────────────────────────
  function init() {
    updateRegime(parseFloat(tempSlider.value));
    checkEngine();
    // Load data that may already exist
    loadAllData();
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
