/**
 * static/js/dataset_ui.js
 * DatasetUIController — handles the Datasets tab and associated editors.
 * Properly wired to the HTML element IDs in index.html.
 */
class DatasetUIController {
  constructor(app) {
    this._app = app;
    this._datasets = [];
    this._activeDataset = null;
    this._isPainting = false;
  }

  init() {
    this._initEventListeners();
    this.refreshDatasets();
  }

  _initEventListeners() {
    // New dataset button
    document.getElementById("newDatasetBtn")?.addEventListener("click", () => this._showCreateModal());

    // Save button
    document.getElementById("dsSaveBtn")?.addEventListener("click", () => this._saveCurrentDataset());

    // Add row
    document.getElementById("dsAddRowBtn")?.addEventListener("click", () => this._addRow());

    // Clear all
    document.getElementById("dsClearBtn")?.addEventListener("click", () => {
      if (!this._activeDataset || this._activeDataset.is_predefined) return;
      if (confirm("Clear all rows?")) {
        this._activeDataset.data = [];
        this._renderTable();
      }
    });

    // Synthetic range modal
    document.getElementById("dsSyntheticBtn")?.addEventListener("click", () => this._openSyntheticModal());
    document.getElementById("dsSyntheticCloseBtn")?.addEventListener("click", () => {
      document.getElementById("dsSyntheticModal").style.display = "none";
    });
    document.getElementById("dsSyntheticGenerateBtn")?.addEventListener("click", () => this._generateSyntheticRange());

    // Download predefined
    document.getElementById("dsDownloadBtn")?.addEventListener("click", () => this._downloadPredefined());

    // CSV import
    document.getElementById("dsImportCsvBtn")?.addEventListener("click", () => this._importCsv());
  }

  // ── REFRESH ──
  async refreshDatasets() {
    try {
      const data = await API.listDatasets();
      this._datasets = data.datasets || [];
      this._renderDatasetList();
      if (this._app._ui) {
        this._app._ui.renderDatasetSelect(this._datasets);
      }
    } catch (e) {
      console.error("Failed to fetch datasets", e);
    }
  }

  // ── LIST ──
  _renderDatasetList() {
    const container = document.getElementById("datasetList");
    if (!container) return;

    const global   = this._datasets.filter(d => d.is_predefined);
    const personal = this._datasets.filter(d => !d.is_predefined);

    let html = '';

    if (global.length) {
      html += `<div style="font-size:10px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:1px;padding:4px 0 2px;margin-bottom:4px;border-bottom:1px solid var(--border);">🌐 Global Registry</div>`;
      html += global.map(ds => this._datasetCard(ds)).join('');
    }

    if (personal.length) {
      html += `<div style="font-size:10px;font-weight:600;color:var(--green);text-transform:uppercase;letter-spacing:1px;padding:8px 0 2px;margin-bottom:4px;border-bottom:1px solid var(--border);">👤 My Datasets</div>`;
      html += personal.map(ds => this._datasetCard(ds)).join('');
    }

    if (!global.length && !personal.length) {
      html = '<div class="info-box">No datasets yet.</div>';
    }

    container.innerHTML = html;
  }

  _datasetCard(ds) {
    const isActive = this._activeDataset?.id === ds.id;
    const borderColor = ds.is_predefined ? 'var(--accent)' : 'var(--green)';
    const badge = ds.is_predefined
      ? `<span style="font-size:8px;background:var(--accent);color:#000;padding:1px 4px;border-radius:3px;font-weight:700;">GLOBAL</span>`
      : '';
    const dlBadge = (ds.is_predefined && !ds.downloaded)
      ? `<span style="font-size:8px;background:var(--yellow);color:#000;padding:1px 4px;border-radius:3px;margin-left:4px;">NOT DL</span>`
      : '';
    const samples = ds.data_length || '?';

    return `
      <div class="card dataset-card ${isActive ? 'active' : ''}"
           onclick="datasetUI.loadDataset(${ds.id})"
           style="cursor:pointer; margin-bottom:6px; border-left: 4px solid ${borderColor}; padding:6px 8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div style="font-weight:bold;font-size:12px;">${ds.name}</div>
          <div>${badge}${dlBadge}</div>
        </div>
        <div style="font-size:10px; color:var(--text3)">${ds.ds_type} • ${ds.num_inputs} in → ${ds.num_outputs || '?'} out</div>
      </div>`;
  }

  // ── LOAD ──
  async loadDataset(id) {
    try {
      const data = await API.getDataset(id);
      this._activeDataset = data.dataset;
      this._renderDatasetList();
      this._showEditor();
    } catch (e) {
      console.error("Failed to load dataset", e);
    }
  }

  // ── EDITOR ──
  _showEditor() {
    const ds = this._activeDataset;
    if (!ds) return;

    // Hide empty state, show editor
    const empty = document.getElementById("datasetEditorEmpty");
    const editor = document.getElementById("datasetEditor");
    if (empty) empty.style.display = "none";
    if (editor) editor.style.display = "flex";

    // Set name and type
    const nameEl = document.getElementById("dsName");
    const typeEl = document.getElementById("dsType");
    if (nameEl) { nameEl.value = ds.name; nameEl.disabled = ds.is_predefined; }
    if (typeEl) { typeEl.value = ds.ds_type; typeEl.disabled = ds.is_predefined; }

    // Hide all views
    const tabView = document.getElementById("dsTabularView");
    const imgView = document.getElementById("dsImageView");
    const preView = document.getElementById("dsPredefinedView");
    if (tabView) tabView.style.display = "none";
    if (imgView) imgView.style.display = "none";
    if (preView) preView.style.display = "none";

    // Disable editing controls for predefined
    const readOnly = ds.is_predefined;
    document.getElementById("dsAddRowBtn")?.toggleAttribute("disabled", readOnly);
    document.getElementById("dsSyntheticBtn")?.toggleAttribute("disabled", readOnly);
    document.getElementById("dsImportCsvBtn")?.toggleAttribute("disabled", readOnly);
    document.getElementById("dsClearBtn")?.toggleAttribute("disabled", readOnly);
    document.getElementById("dsInputOnly")?.toggleAttribute("disabled", readOnly);

    const saveBtn = document.getElementById("dsSaveBtn");
    if (saveBtn) {
      saveBtn.disabled = readOnly;
      saveBtn.textContent = readOnly ? "🔒 Read-Only (Global)" : "💾 Save Dataset";
    }

    // Show appropriate view
    if (ds.is_predefined && !ds.downloaded) {
      if (preView) preView.style.display = "block";
      const desc = document.getElementById("dsPredefinedDesc");
      if (desc) desc.textContent = ds.description || "Download this predefined dataset to use it.";
    } else if (ds.ds_type === "image") {
      if (imgView) imgView.style.display = "flex";
      // TODO: image pixel editor
    } else {
      // Tabular or predefined-downloaded
      if (tabView) tabView.style.display = "flex";
      this._renderTable();
    }
  }

  // ── TABLE ──
  _renderTable() {
    const container = document.getElementById("dsTableContainer");
    const ds = this._activeDataset;
    if (!container || !ds) return;

    const data = ds.data || [];
    const readOnly = ds.is_predefined;
    const nIn = ds.num_inputs || 0;
    const nOut = ds.num_outputs || 0;
    const inLabels = ds.input_labels || [];
    const outLabels = ds.output_labels || [];

    let html = `<table class="io-table"><thead><tr><th>#</th>`;
    for (let i = 0; i < Math.min(nIn, 20); i++) html += `<th>${inLabels[i] || 'In ' + i}</th>`;
    if (nIn > 20) html += `<th>… +${nIn - 20}</th>`;
    for (let i = 0; i < nOut; i++) html += `<th>${outLabels[i] || 'Out ' + i}</th>`;
    if (!readOnly) html += `<th></th>`;
    html += `</tr></thead><tbody>`;

    data.forEach((row, idx) => {
      html += `<tr><td style="color:var(--text3)">${idx}</td>`;
      const xArr = row.x || [];
      for (let i = 0; i < Math.min(nIn, 20); i++) {
        const val = xArr[i] ?? 0;
        if (readOnly) {
          html += `<td>${parseFloat(val).toFixed(3)}</td>`;
        } else {
          html += `<td><input type="number" step="0.01" value="${parseFloat(val).toFixed(3)}" style="width:60px;padding:2px 4px;font-size:11px;" onchange="datasetUI._updateVal(${idx},'x',${i},this.value)"></td>`;
        }
      }
      if (nIn > 20) html += `<td style="color:var(--text3)">…</td>`;

      if (row.y) {
        row.y.forEach((val, i) => {
          if (readOnly) {
            html += `<td>${parseFloat(val).toFixed(3)}</td>`;
          } else {
            html += `<td><input type="number" step="0.01" value="${parseFloat(val).toFixed(3)}" style="width:60px;padding:2px 4px;font-size:11px;" onchange="datasetUI._updateVal(${idx},'y',${i},this.value)"></td>`;
          }
        });
      } else {
        for (let i = 0; i < nOut; i++) html += `<td style="color:var(--text3)">—</td>`;
      }
      if (!readOnly) {
        html += `<td><button class="btn danger" style="padding:1px 5px;font-size:10px;" onclick="datasetUI._removeRow(${idx})">×</button></td>`;
      }
      html += `</tr>`;
    });

    html += `</tbody></table>`;
    if (data.length === 0) {
      html += `<div class="info-box" style="margin-top:10px;">No samples. ${readOnly ? '' : 'Click "+ Add Row" or use the Synthetic generator.'}</div>`;
    }
    container.innerHTML = html;
  }

  _updateVal(rowIdx, type, colIdx, val) {
    if (!this._activeDataset?.data?.[rowIdx]) return;
    this._activeDataset.data[rowIdx][type][colIdx] = parseFloat(val);
  }

  _addRow() {
    const ds = this._activeDataset;
    if (!ds || ds.is_predefined) return;
    if (!ds.data) ds.data = [];
    const newRow = {
      x: new Array(ds.num_inputs).fill(0),
      y: ds.is_input_only ? null : new Array(ds.num_outputs || 1).fill(0)
    };
    ds.data.push(newRow);
    this._renderTable();
  }

  _removeRow(idx) {
    if (!this._activeDataset?.data) return;
    this._activeDataset.data.splice(idx, 1);
    this._renderTable();
  }

  // ── SYNTHETIC RANGE ──
  _openSyntheticModal() {
    const ds = this._activeDataset;
    if (!ds || ds.is_predefined) return;

    const modal = document.getElementById("dsSyntheticModal");
    const inputs = document.getElementById("dsSyntheticInputs");
    if (!modal || !inputs) return;

    const inLabels = ds.input_labels || [];
    inputs.innerHTML = '';
    for (let i = 0; i < ds.num_inputs; i++) {
      inputs.innerHTML += `
        <div>
          <label style="font-size:11px;">${inLabels[i] || 'Input ' + i}</label>
          <div style="display:flex;gap:4px;">
            <input type="number" id="synMin${i}" value="0" step="0.1" style="flex:1;" placeholder="Min">
            <input type="number" id="synMax${i}" value="1" step="0.1" style="flex:1;" placeholder="Max">
          </div>
        </div>`;
    }
    inputs.innerHTML += `
      <div>
        <label style="font-size:11px;">Samples per dimension</label>
        <input type="number" id="synSamples" value="5" min="2" max="50" style="width:100%;">
      </div>`;

    modal.style.display = "block";
  }

  _generateSyntheticRange() {
    const ds = this._activeDataset;
    if (!ds) return;

    const samplesPerDim = parseInt(document.getElementById("synSamples")?.value) || 5;
    const ranges = [];
    for (let i = 0; i < ds.num_inputs; i++) {
      ranges.push({
        min: parseFloat(document.getElementById(`synMin${i}`)?.value) || 0,
        max: parseFloat(document.getElementById(`synMax${i}`)?.value) || 1,
      });
    }

    // Generate cartesian product for up to 3 inputs, otherwise random
    if (!ds.data) ds.data = [];
    if (ds.num_inputs <= 3) {
      const generate = (dims, current = []) => {
        if (dims.length === 0) {
          ds.data.push({ x: [...current], y: new Array(ds.num_outputs || 1).fill(0) });
          return;
        }
        const [range, ...rest] = dims;
        for (let s = 0; s < samplesPerDim; s++) {
          const val = range.min + (range.max - range.min) * s / (samplesPerDim - 1);
          generate(rest, [...current, parseFloat(val.toFixed(4))]);
        }
      };
      generate(ranges);
    } else {
      const total = samplesPerDim ** 2;
      for (let s = 0; s < total; s++) {
        const x = ranges.map(r => parseFloat((r.min + Math.random() * (r.max - r.min)).toFixed(4)));
        ds.data.push({ x, y: new Array(ds.num_outputs || 1).fill(0) });
      }
    }

    document.getElementById("dsSyntheticModal").style.display = "none";
    this._renderTable();
    alert(`Added ${ds.data.length} samples.`);
  }

  // ── DOWNLOAD PREDEFINED ──
  async _downloadPredefined() {
    const ds = this._activeDataset;
    if (!ds) return;
    try {
      const bar = document.getElementById("dsDownloadBar");
      const prog = document.getElementById("dsDownloadProgress");
      const status = document.getElementById("dsDownloadStatus");
      if (prog) prog.style.display = "block";
      if (bar) bar.style.width = "30%";
      if (status) status.textContent = "Downloading...";

      await API.downloadDataset(ds.id);

      if (bar) bar.style.width = "100%";
      if (status) status.textContent = "Complete!";

      // Reload the dataset
      setTimeout(() => {
        this.loadDataset(ds.id);
        this.refreshDatasets();
      }, 500);
    } catch (e) {
      console.error("Download failed", e);
      alert("Download failed: " + e.message);
    }
  }

  // ── CSV IMPORT ──
  _importCsv() {
    const ds = this._activeDataset;
    if (!ds || ds.is_predefined) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.tsv,.txt";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const lines = ev.target.result.trim().split("\n");
          if (!ds.data) ds.data = [];
          lines.forEach(line => {
            const vals = line.split(/[,\t]/).map(v => parseFloat(v.trim()));
            if (vals.length < ds.num_inputs) return;
            const x = vals.slice(0, ds.num_inputs);
            const y = vals.length >= ds.num_inputs + (ds.num_outputs || 0)
              ? vals.slice(ds.num_inputs, ds.num_inputs + ds.num_outputs)
              : new Array(ds.num_outputs || 1).fill(0);
            ds.data.push({ x, y });
          });
          this._renderTable();
          alert(`Imported ${lines.length} rows from CSV.`);
        } catch (err) {
          alert("CSV parse error: " + err.message);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  // ── SAVE ──
  async _saveCurrentDataset() {
    const ds = this._activeDataset;
    if (!ds) return;
    if (ds.is_predefined) {
      alert("Cannot modify a global (predefined) dataset.");
      return;
    }
    // Read name from input
    const nameEl = document.getElementById("dsName");
    if (nameEl) ds.name = nameEl.value;
    const inputOnly = document.getElementById("dsInputOnly");
    if (inputOnly) ds.is_input_only = inputOnly.checked;

    try {
      await API.updateDataset(ds.id, ds);
      alert("Dataset saved!");
      this.refreshDatasets();
    } catch (e) {
      alert("Save failed: " + e.message);
    }
  }

  // ── CREATE MODAL ──
  _showCreateModal() {
    document.getElementById("createDatasetModal").style.display = "flex";
  }

  async _confirmCreate() {
    const name = document.getElementById("newDsName").value;
    const ds_type = document.getElementById("newDsType").value;
    const num_inputs = parseInt(document.getElementById("newDsInputs").value);
    const num_outputs = parseInt(document.getElementById("newDsOutputs").value);
    const is_input_only = document.getElementById("newDsInputOnly").checked;

    try {
      const result = await API.createDataset({
        name, ds_type, num_inputs, num_outputs, is_input_only,
        data: [{ x: new Array(num_inputs).fill(0), y: is_input_only ? null : new Array(num_outputs).fill(0) }]
      });
      document.getElementById("createDatasetModal").style.display = "none";
      this.refreshDatasets();
      this.loadDataset(result.dataset.id);
    } catch (e) {
      alert("Create failed: " + e.message);
    }
  }
}

// Global instance
window.datasetUI = null;
