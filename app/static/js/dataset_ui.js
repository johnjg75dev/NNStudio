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
    
    // Close create modal when clicking outside
    document.getElementById("createDatasetModal")?.addEventListener("click", (e) => {
      if (e.target.id === "createDatasetModal") {
        this._closeCreateModal();
      }
    });

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
    
    // Close modal when clicking outside
    document.getElementById("dsSyntheticModal")?.addEventListener("click", (e) => {
      if (e.target.id === "dsSyntheticModal") {
        e.target.style.display = "none";
      }
    });

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
      this._initImageEditor();
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

    modal.style.display = "flex";
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
  
  _closeCreateModal() {
    document.getElementById("createDatasetModal").style.display = "none";
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
      this._app._ui._switchTab("datasets");
      this.refreshDatasets();
      this.loadDataset(result.dataset.id);
    } catch (e) {
      alert("Create failed: " + e.message);
    }
  }

  // ── IMAGE PIXEL EDITOR ───────────────────────────────────────────────
  _initImageEditor() {
    const ds = this._activeDataset;
    if (!ds || ds.ds_type !== "image") return;

    // Ensure dataset has at least one sample with initialized data
    if (!ds.data || ds.data.length === 0) {
      ds.data = [{
        x: new Array(ds.num_inputs).fill(0),
        y: new Array(ds.num_outputs || 1).fill(0)
      }];
    }

    // Get references to UI elements
    this._imgCanvas = document.getElementById("dsPixelCanvas");
    this._imgCtx = this._imgCanvas.getContext("2d");
    this._imgContainer = document.getElementById("dsPixelContainer");
    this._colorPicker = document.getElementById("dsColorPicker");
    this._colorInfo = document.getElementById("dsColorInfo");
    this._zoomVal = document.getElementById("dsImgZoomVal");
    this._histCanvas = document.getElementById("dsImgHist");
    this._histCtx = this._histCanvas.getContext("2d");

    // Set default values
    this._zoom = 4;
    this._isDrawing = false;
    this._currentColor = this._hexToRgb(this._colorPicker.value);
    this._brushSize = 1;

    // Update color info
    this._updateColorInfo();

    // Setup event listeners
    this._setupCanvasEvents();
    this._colorPicker.addEventListener("change", () => {
      this._currentColor = this._hexToRgb(this._colorPicker.value);
      this._updateColorInfo();
    });

    // Zoom buttons
    document.getElementById("dsImgZoomIn").addEventListener("click", () => this._zoomImage(2));
    document.getElementById("dsImgZoomOut").addEventListener("click", () => this._zoomImage(0.5));

    // Navigation buttons
    document.getElementById("dsImgPrev").addEventListener("click", () => this._navigateImage(-1));
    document.getElementById("dsImgNext").addEventListener("click", () => this._navigateImage(1));

    // Initialize with first sample
    this._currentSampleIdx = 0;
    this._renderImage();
    this._updateHistogram();
  }

  _setupCanvasEvents() {
    const canvas = this._imgCanvas;
    canvas.addEventListener("mousedown", (e) => this._startDrawing(e));
    canvas.addEventListener("mousemove", (e) => this._draw(e));
    canvas.addEventListener("mouseup", () => this._stopDrawing());
    canvas.addEventListener("mouseleave", () => this._stopDrawing());
    
    // Touch support
    canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener("touchend", (e) => {
      e.preventDefault();
      const mouseEvent = new MouseEvent("mouseup", {});
      canvas.dispatchEvent(mouseEvent);
    });
  }

  _startDrawing(e) {
    this._isDrawing = true;
    this._draw(e);
  }

  _stopDrawing() {
    if (this._isDrawing) {
      this._isDrawing = false;
      this._updateDatasetFromCanvas(); // Save changes after each stroke
    }
  }

  _draw(e) {
    if (!this._isDrawing) return;

    const rect = this._imgCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / this._zoom);
    const y = Math.floor((e.clientY - rect.top) / this._zoom);

    this._drawPixel(x, y, this._currentColor);
    this._updateHistogram(); // Update histogram in real-time
  }

  _drawPixel(x, y, color) {
    const ds = this._activeDataset;
    if (!ds || !ds.data || ds.data.length === 0) return;

    const sample = ds.data[this._currentSampleIdx];
    if (!sample.x) return;

    const width = ds.width;
    const height = ds.height;
    const channels = ds.channels || 1;

    // Bounds check
    if (x < 0 || x >= width || y < 0 || y >= height) return;

    // Update the pixel in the dataset
    const idx = (y * width + x) * channels;
    sample.x[idx] = color[0] / 255; // Red channel
    if (channels > 1) sample.x[idx + 1] = color[1] / 255; // Green
    if (channels > 2) sample.x[idx + 2] = color[2] / 255; // Blue

    // Redraw just the pixel on canvas
    const canvasX = x * this._zoom;
    const canvasY = y * this._zoom;
    const pixelSize = this._zoom;
    
    this._imgCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this._imgCtx.fillRect(canvasX, canvasY, pixelSize, pixelSize);
  }

  _navigateImage(direction) {
    const ds = this._activeDataset;
    if (!ds || !ds.data) return;

    const numSamples = ds.data.length;
    if (numSamples === 0) return;

    // Save current sample data before leaving (in case it wasn't saved)
    // The data is already updated in _drawPixel, but ensure it's committed

    // Calculate new index
    let newIndex = this._currentSampleIdx + direction;
    if (newIndex < 0) newIndex = 0;
    if (newIndex >= numSamples) newIndex = numSamples - 1;

    if (newIndex !== this._currentSampleIdx) {
      this._currentSampleIdx = newIndex;
      this._renderImage();
      this._updateHistogram();
    }
  }

  _renderImage() {
    const ds = this._activeDataset;
    if (!ds || !ds.data || ds.data.length === 0) return;

    const sample = ds.data[this._currentSampleIdx];
    if (!sample.x) return;

    const width = ds.width;
    const height = ds.height;
    const channels = ds.channels || 1;

    // Set canvas size to match image size multiplied by zoom
    this._imgCanvas.width = width * this._zoom;
    this._imgCanvas.height = height * this._zoom;
    
    // Create image data
    const imgData = this._imgCtx.createImageData(width * this._zoom, height * this._zoom);
    const data = sample.x;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * channels;
        const val = data[idx]; // Use first channel for grayscale
        const color = Math.floor(val * 255);
        
        for (let zy = 0; zy < this._zoom; zy++) {
          for (let zx = 0; zx < this._zoom; zx++) {
            const pos = ((y * this._zoom + zy) * width * this._zoom + (x * this._zoom + zx)) * 4;
            imgData.data[pos] = color;
            imgData.data[pos + 1] = color;
            imgData.data[pos + 2] = color;
            imgData.data[pos + 3] = 255;
          }
        }
      }
    }

    this._imgCtx.putImageData(imgData, 0, 0);
  }

  _updateDatasetFromCanvas() {
    // The data is already updated in the _drawPixel method
    // This method can be used for any additional processing if needed
  }

  _zoomImage(factor) {
    const ds = this._activeDataset;
    if (!ds) return;

    const newZoom = Math.max(1, Math.floor(this._zoom * factor));
    if (newZoom < 1 || newZoom > 32) return; // Limit zoom range

    this._zoom = newZoom;
    this._zoomVal.textContent = this._zoom + "x";
    this._renderImage();
  }

  _updateHistogram() {
    const ds = this._activeDataset;
    if (!ds || !ds.data || ds.data.length === 0) return;

    const sample = ds.data[this._currentSampleIdx];
    if (!sample.x) return;

    const width = ds.width;
    const height = ds.height;
    const totalPixels = width * height;
    const data = sample.x;
    
    // Calculate histogram (64 bins)
    const bins = 64;
    const histogram = new Array(bins).fill(0);
    
    for (let i = 0; i < totalPixels; i++) {
      const val = Math.floor(data[i * ds.channels] * 255); // Use first channel
      const bin = Math.min(Math.floor(val / (256 / bins)), bins - 1);
      histogram[bin]++;
    }

    // Find max count for normalization
    const maxCount = Math.max(...histogram);

    // Draw histogram
    const canvas = this._histCanvas;
    const ctx = this._histCtx;
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    
    ctx.clearRect(0, 0, w, h);
    
    // Draw bars
    const barWidth = w / bins;
    for (let i = 0; i < bins; i++) {
      const barHeight = (histogram[i] / maxCount) * (h - 20);
      ctx.fillStyle = `rgb(${i * 4}, ${i * 4}, ${i * 4})`;
      ctx.fillRect(i * barWidth, h - barHeight, barWidth - 1, barHeight);
    }
  }

  _hexToRgb(hex) {
    // Convert hex to rgb array
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ] : [255, 255, 255];
  }

  _updateColorInfo() {
    if (this._colorInfo) {
      const rgb = this._currentColor;
      this._colorInfo.textContent = `RGB: ${rgb[0]}, ${rgb[1]}, ${rgb[2]}`;
      this._colorInfo.style.color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }
  }
}

// Global instance
window.datasetUI = null;
