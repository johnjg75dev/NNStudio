/**
 * static/js/dataset_ui.js
 * DatasetUIController — handles the Datasets tab and associated editors.
 */
class DatasetUIController {
  constructor(app) {
    this._app = app;
    this._datasets = [];
    this._activeDataset = null;
    this._gridSize = 8; // Default for pixel editor
    this._isPainting = false;
  }

  init() {
    this._initEventListeners();
    this.refreshDatasets();
  }

  _initEventListeners() {
    document.getElementById("newDatasetBtn")?.addEventListener("click", () => this._showCreateModal());
    document.getElementById("saveDatasetBtn")?.addEventListener("click", () => this._saveCurrentDataset());
    document.getElementById("pixelGridSize")?.addEventListener("change", (e) => {
      this._gridSize = parseInt(e.target.value);
      this._renderPixelEditor();
    });

    // Synthetic generator triggers
    document.getElementById("genSyntheticBtn")?.addEventListener("click", () => this._generateSynthetic());
  }

  async refreshDatasets() {
    try {
      const resp = await fetch("/api/datasets");
      const data = await resp.json();
      if (data.success) {
        this._datasets = data.datasets;
        this._renderDatasetList();
        // Sync with training UI
        if (this._app._ui) {
          this._app._ui.renderDatasetSelect(this._datasets);
        }
      }
    } catch (e) {
      console.error("Failed to fetch datasets", e);
    }
  }

  _renderDatasetList() {
    const container = document.getElementById("datasetList");
    if (!container) return;

    container.innerHTML = this._datasets.map(ds => `
      <div class="card dataset-card ${this._activeDataset?.id === ds.id ? 'active' : ''}"
           onclick="datasetUI.loadDataset(${ds.id})"
           style="cursor:pointer; margin-bottom:8px; border-left: 4px solid ${ds.is_predefined ? 'var(--accent)' : 'var(--green)'}">
        <div style="font-weight:bold">${ds.name}</div>
        <div style="font-size:10px; color:var(--text3)">${ds.ds_type} • ${ds.num_inputs} in → ${ds.num_outputs} out</div>
      </div>
    `).join("");
  }

  async loadDataset(id) {
    try {
      const resp = await fetch(`/api/datasets/${id}`);
      const data = await resp.json();
      if (data.success) {
        this._activeDataset = data.dataset;
        this._renderDatasetList();
        this._renderEditor();
      }
    } catch (e) {
      console.error("Failed to load dataset", e);
    }
  }

  _renderEditor() {
    const ds = this._activeDataset;
    if (!ds) return;

    // Show appropriate editor based on type
    document.getElementById("datasetEditorHeader").innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center">
        <h4>Editing: ${ds.name}</h4>
        <div class="pill">${ds.ds_type}</div>
      </div>
    `;

    if (ds.ds_type === "image") {
      this._showPixelEditor();
    } else {
      this._showTabularEditor();
    }
  }

  _showTabularEditor() {
    document.getElementById("tabularEditor").style.display = "block";
    document.getElementById("pixelEditor").style.display = "none";
    this._renderTable();
  }

  _showPixelEditor() {
    document.getElementById("tabularEditor").style.display = "none";
    document.getElementById("pixelEditor").style.display = "block";
    this._gridSize = this._activeDataset.width || 8;
    this._renderPixelEditor();
  }

  _renderTable() {
    const container = document.getElementById("tableContainer");
    const ds = this._activeDataset;
    const data = ds.data || [];

    let html = `<table><thead><tr><th>#</th>`;
    for (let i = 0; i < ds.num_inputs; i++) html += `<th>In ${i}</th>`;
    for (let i = 0; i < ds.num_outputs; i++) html += `<th>Out ${i}</th>`;
    html += `<th>Action</th></tr></thead><tbody>`;

    data.forEach((row, idx) => {
      html += `<tr><td>${idx}</td>`;
      row.x.forEach((val, i) => {
        html += `<td><input type="number" step="0.01" value="${val}" onchange="datasetUI._updateVal(${idx}, 'x', ${i}, this.value)"></td>`;
      });
      if (row.y) {
        row.y.forEach((val, i) => {
          html += `<td><input type="number" step="0.01" value="${val}" onchange="datasetUI._updateVal(${idx}, 'y', ${i}, this.value)"></td>`;
        });
      } else {
        for (let i = 0; i < ds.num_outputs; i++) html += `<td>—</td>`;
      }
      html += `<td><button class="btn danger" style="padding:2px 6px" onclick="datasetUI._removeRow(${idx})">×</button></td></tr>`;
    });

    html += `</tbody></table>`;
    html += `<button class="btn secondary" style="margin-top:8px" onclick="datasetUI._addRow()">+ Add Row</button>`;
    container.innerHTML = html;
  }

  _updateVal(rowIdx, type, colIdx, val) {
    this._activeDataset.data[rowIdx][type][colIdx] = parseFloat(val);
  }

  _addRow() {
    const ds = this._activeDataset;
    const newRow = {
      x: new Array(ds.num_inputs).fill(0),
      y: ds.is_input_only ? null : new Array(ds.num_outputs).fill(0)
    };
    ds.data.push(newRow);
    this._renderTable();
  }

  _removeRow(idx) {
    this._activeDataset.data.splice(idx, 1);
    this._renderTable();
  }

  _renderPixelEditor() {
    const container = document.getElementById("pixelGrid");
    container.style.gridTemplateColumns = `repeat(${this._gridSize}, 1fr)`;
    container.innerHTML = "";

    // We'll use the first sample for editing in the pixel grid for now
    if (!this._activeDataset.data || this._activeDataset.data.length === 0) {
      this._activeDataset.data = [{x: new Array(this._gridSize * this._gridSize).fill(0), y: [0]}];
    }
    const pixels = this._activeDataset.data[0].x;

    for (let i = 0; i < this._gridSize * this._gridSize; i++) {
      const cell = document.createElement("div");
      cell.className = "pixel-cell";
      cell.style.background = `rgba(63, 185, 80, ${pixels[i]})`;
      cell.addEventListener("mousedown", () => {
        this._isPainting = true;
        this._togglePixel(i, cell);
      });
      cell.addEventListener("mouseenter", () => {
        if (this._isPainting) this._togglePixel(i, cell);
      });
      container.appendChild(cell);
    }
    window.addEventListener("mouseup", () => this._isPainting = false);
  }

  _togglePixel(idx, cell) {
    const val = this._activeDataset.data[0].x[idx] === 1 ? 0 : 1;
    this._activeDataset.data[0].x[idx] = val;
    cell.style.background = `rgba(63, 185, 80, ${val})`;
  }

  async _saveCurrentDataset() {
    if (!this._activeDataset) return;
    try {
      const resp = await fetch(`/api/datasets/${this._activeDataset.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this._activeDataset)
      });
      const data = await resp.json();
      if (data.success) {
        alert("Dataset saved successfully");
        this.refreshDatasets();
      }
    } catch (e) {
      console.error("Failed to save dataset", e);
    }
  }

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
      const resp = await fetch("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name, ds_type, num_inputs, num_outputs, is_input_only,
          data: [{x: new Array(num_inputs).fill(0), y: is_input_only ? null : new Array(num_outputs).fill(0)}]
        })
      });
      const data = await resp.json();
      if (data.success) {
        document.getElementById("createDatasetModal").style.display = "none";
        this.refreshDatasets();
        this.loadDataset(data.dataset.id);
      }
    } catch (e) {
      console.error("Failed to create dataset", e);
    }
  }

  async _generateSynthetic() {
    const type = document.getElementById("syntheticType").value;
    const samples = parseInt(document.getElementById("syntheticSamples").value);

    // Simple mock generation
    const newData = [];
    for (let i = 0; i < samples; i++) {
      const x = Math.random();
      let y;
      if (type === "sine") y = Math.sin(x * Math.PI * 2) * 0.5 + 0.5;
      else y = x > 0.5 ? 1 : 0;
      newData.push({ x: [x], y: [y] });
    }

    this._activeDataset.data = newData;
    this._renderTable();
    alert(`Generated ${samples} synthetic samples.`);
  }
}

// Global instance
window.datasetUI = null;
