/**
 * static/js/ui.js
 * UIController — owns all DOM interaction.
 * Reads from AppState, writes to DOM. Never calls API directly.
 *
 * Responsibilities:
 *   - Tab switching
 *   - Panel population (presets, functions, arch types)
 *   - Control bindings (sliders, selects)
 *   - Stat bar updates
 *   - Tooltip system
 *   - IO table rendering
 *   - Node inspector panel
 *   - Weight matrix panel
 *   - Test-input panel
 *   - Save / Load UI
 */
class UIController {
  constructor(registry) {
    this._registry = registry;
    this._tipEl    = document.getElementById("tip");
    this._handlers = {};   // event name → callback, for external wiring
    this._layers   = [];   // Initialize to avoid undefined map errors
    this._fnMeta   = null; // Store function metadata for test tab
    this._sweepRanges = {}; // Store sweep range for each input
  }

  on(event, fn) { this._handlers[event] = fn; }
  _emit(event, data) { this._handlers[event]?.(data); }

  // ════════════════════════════════════════════════════════
  // INIT — call once after DOM ready
  // ════════════════════════════════════════════════════════
  init() {
    this._initTabs();
    this._initTooltips();
    this._initPresets();
    this._initSavePresetBtn();
    this._initArchSelect();
    this._initFuncSelect();
    this._initLayerEditor();
    this._initSliders();
    this._initVizCheckboxes();
    this._initBuildBtn();
    this._initTrainBtns();
    this._initCanvasInteraction();
    this._initTestTab();
    this._initSaveLoad();
    this._initRightPanelTabs();
    this._initKeyboard();
  }

  // ════════════════════════════════════════════════════════
  // TABS
  // ════════════════════════════════════════════════════════
  _initTabs() {
    document.querySelectorAll("#mainTabs .tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#mainTabs .tab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const pg = "pg" + btn.dataset.pg[0].toUpperCase() + btn.dataset.pg.slice(1);
        document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
        document.getElementById(pg)?.classList.add("active");
        if (btn.dataset.pg === "test")     this._renderTestInputs(this._fnMeta);
        if (btn.dataset.pg === "saveload") this._renderModelSummary();
      });
    });
  }

  // ════════════════════════════════════════════════════════
  // TOOLTIPS (data-tip attributes)
  // ════════════════════════════════════════════════════════
  _initTooltips() {
    document.addEventListener("mouseover", e => {
      const el = e.target.closest("[data-tip]");
      if (el) { this._tipEl.innerHTML = el.dataset.tip; this._tipEl.style.display = "block"; }
    });
    document.addEventListener("mousemove", e => {
      if (this._tipEl.style.display === "block") {
        this._tipEl.style.left = Math.min(e.clientX + 14, window.innerWidth - 246) + "px";
        this._tipEl.style.top  = Math.max(4, e.clientY - 10) + "px";
      }
    });
    document.addEventListener("mouseout", e => {
      if (e.target.closest("[data-tip]")) this._tipEl.style.display = "none";
    });
  }

  // ════════════════════════════════════════════════════════
  // PRESETS — populated from registry
  // ════════════════════════════════════════════════════════
  _initPresets() {
    const grid = document.getElementById("presetGrid");
    if (!grid) return;
    grid.innerHTML = "";
    const presets = this._registry.presets || [];
    presets.forEach(p => {
      const container = document.createElement("div");
      container.className = "pbtn-wrap";
      
      const btn = document.createElement("button");
      btn.className = "pbtn";
      btn.innerHTML = `${p.label}<div class="pa">${p.func_key}</div>`;
      btn.addEventListener("click", () => this._emit("applyPreset", p));
      container.appendChild(btn);

      if (p.custom) {
        const del = document.createElement("button");
        del.className = "pdel";
        del.innerHTML = "×";
        del.dataset.tip = "Delete this preset";
        del.addEventListener("click", (e) => {
          e.stopPropagation();
          if (confirm(`Delete preset "${p.label}"?`)) {
            this._emit("deletePreset", p.id);
          }
        });
        container.appendChild(del);
      }

      grid.appendChild(container);
    });
  }

  _initSavePresetBtn() {
    document.getElementById("savePresetBtn")?.addEventListener("click", () => {
      const label = prompt("Enter a name for this preset:");
      if (!label) return;
      const desc = prompt("Enter a short description (optional):");
      const cfg = this.getConfig();
      this._emit("savePreset", { ...cfg, label, description: desc });
    });
  }

  // ════════════════════════════════════════════════════════
  // ARCHITECTURE SELECT
  // ════════════════════════════════════════════════════════
  _initArchSelect() {
    const sel = document.getElementById("archSel");
    const info = document.getElementById("archInfo");
    if (!sel) return;

    (this._registry.architectures || []).forEach(a => {
      const o = document.createElement("option");
      o.value = a.key; o.textContent = a.label;
      sel.appendChild(o);
    });

    const update = () => {
      const arch = (this._registry.architectures || []).find(a => a.key === sel.value);
      if (info && arch) info.innerHTML = arch.description || "";
      this._emit("archChanged", sel.value);
    };
    sel.addEventListener("change", update);
    update();
  }

  // ════════════════════════════════════════════════════════
  // FUNCTION SELECT
  // ════════════════════════════════════════════════════════
  _initFuncSelect() {
    const sel  = document.getElementById("funcSel");
    const desc = document.getElementById("funcDesc");
    if (!sel) return;

    (this._registry.functions || []).forEach(f => {
      const o = document.createElement("option");
      o.value = f.key; o.textContent = f.label;
      sel.appendChild(o);
    });

    const update = () => {
      const fn = (this._registry.functions || []).find(f => f.key === sel.value);
      if (desc && fn) desc.innerHTML = fn.description || "";
      this._emit("funcChanged", fn);
    };
    sel.addEventListener("change", update);
    update();
  }

  // ════════════════════════════════════════════════════════
  // SLIDERS
  // ════════════════════════════════════════════════════════
  _initSliders() {
    const bind = (sliderId, valId, transform) => {
      const sl  = document.getElementById(sliderId);
      const val = document.getElementById(valId);
      if (!sl) return;
      const update = () => {
        const v = transform ? transform(+sl.value) : +sl.value;
        if (val) val.textContent = typeof v === "number" && !Number.isInteger(v)
          ? v.toFixed(v < 0.01 ? 6 : 4).replace(/\.?0+$/, "")
          : v;
        this._emit("controlChanged", this.getConfig());
      };
      sl.addEventListener("input", update);
      update();
    };

    bind("wdSl",      "wdV",     v => v.toFixed(3));
    bind("stepsSl",   "stepsV");

    ["optSel","lossSel"].forEach(id => {
      document.getElementById(id)?.addEventListener("change",
        () => this._emit("controlChanged", this.getConfig()));
    });
  }

  // ════════════════════════════════════════════════════════
  // LAYER EDITOR
  // ════════════════════════════════════════════════════════
  _initLayerEditor() {
    this._inputNeurons = 2;
    this._outputNeurons = 1;
    this._hiddenLayers = []; // [{ type: 'dense', neurons: 4, activation: 'tanh' }]
    
    const container = document.getElementById("layersContainer");
    const addBtn = document.getElementById("addLayerBtn");

    // Set defaults from function metadata
    const updateIODefaults = (fnMeta) => {
      if (fnMeta) {
        this._inputNeurons = fnMeta.inputs ?? 2;
        this._outputNeurons = fnMeta.outputs ?? 1;
        this._renderAllLayers();
      }
    };
    this.on("funcChanged", updateIODefaults);

    // Add Dense layer button
    addBtn?.addEventListener("click", () => {
      this.addHiddenLayer({ type: "dense", neurons: 4, activation: "tanh" });
    });

    // Seed with one hidden layer by default
    if (this._hiddenLayers.length === 0) {
      this.addHiddenLayer({ type: "dense", neurons: 4, activation: "tanh" });
    }
  }

  addHiddenLayer(config) {
    const id = Math.random().toString(36).substr(2, 9);
    const layer = { ...config, id };
    this._hiddenLayers.push(layer);
    this._renderAllLayers();
    this._emit("controlChanged", this.getConfig());
  }

  removeHiddenLayer(id) {
    this._hiddenLayers = this._hiddenLayers.filter(l => l.id !== id);
    this._renderAllLayers();
    this._emit("controlChanged", this.getConfig());
  }

  _renderAllLayers() {
    const container = document.getElementById("layersContainer");
    if (!container) return;
    container.innerHTML = "";
    
    // Render Input Layer
    this._renderInputLayer(container);
    
    // Render Hidden Layers
    this._hiddenLayers.forEach((layer, index) => {
      this._renderHiddenLayer(container, layer, index + 1);
    });
    
    // Render Output Layer
    this._renderOutputLayer(container);
  }

  _renderInputLayer(container) {
    const div = document.createElement("div");
    div.className = "layer-row";
    div.style = "display: flex; gap: 4px; align-items: center; margin-bottom: 4px; background: var(--surf2); padding: 4px; border-radius: 4px; border: 1px solid var(--border);";

    // Badge
    const badge = document.createElement("span");
    badge.textContent = "IN";
    badge.style = "font-size: 9px; color: #fff; font-weight: bold; width: 28px; text-align: center; background: #3fb950; border-radius: 3px; padding: 2px 0;";

    // Label
    const lbl = document.createElement("span");
    lbl.textContent = "Input";
    lbl.style = "font-size: 10px; color: var(--text2); width: 50px;";

    // Neuron count
    const nInput = document.createElement("input");
    nInput.type = "number";
    nInput.value = this._inputNeurons;
    nInput.min = 1;
    nInput.max = 256;
    nInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
    nInput.addEventListener("input", () => {
      this._inputNeurons = parseInt(nInput.value) || 1;
      this._emit("controlChanged", this.getConfig());
    });

    const nLbl = document.createElement("span");
    nLbl.textContent = "neurons";
    nLbl.style = "font-size: 10px; color: var(--text2); flex: 1;";

    div.appendChild(badge);
    div.appendChild(lbl);
    div.appendChild(nInput);
    div.appendChild(nLbl);
    container.appendChild(div);
  }

  _renderOutputLayer(container) {
    const div = document.createElement("div");
    div.className = "layer-row";
    div.style = "display: flex; gap: 4px; align-items: center; margin-bottom: 4px; background: var(--surf2); padding: 4px; border-radius: 4px; border: 1px solid var(--border);";

    // Badge
    const badge = document.createElement("span");
    badge.textContent = "OUT";
    badge.style = "font-size: 9px; color: #fff; font-weight: bold; width: 28px; text-align: center; background: #f85149; border-radius: 3px; padding: 2px 0;";

    // Label
    const lbl = document.createElement("span");
    lbl.textContent = "Output";
    lbl.style = "font-size: 10px; color: var(--text2); width: 50px;";

    // Neuron count
    const nInput = document.createElement("input");
    nInput.type = "number";
    nInput.value = this._outputNeurons;
    nInput.min = 1;
    nInput.max = 256;
    nInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
    nInput.addEventListener("input", () => {
      this._outputNeurons = parseInt(nInput.value) || 1;
      this._emit("controlChanged", this.getConfig());
    });

    const nLbl = document.createElement("span");
    nLbl.textContent = "neurons";
    nLbl.style = "font-size: 10px; color: var(--text2); flex: 1;";

    // Activation (always sigmoid for output)
    const actLbl = document.createElement("span");
    actLbl.textContent = "sigmoid";
    actLbl.style = "font-size: 10px; color: var(--text3); width: 60px; text-align: right;";

    div.appendChild(badge);
    div.appendChild(lbl);
    div.appendChild(nInput);
    div.appendChild(nLbl);
    div.appendChild(actLbl);
    container.appendChild(div);
  }

  _renderHiddenLayer(container, layer, index) {
    const div = document.createElement("div");
    div.className = "layer-row";
    div.id = `layer-${layer.id}`;
    div.style = "display: flex; gap: 4px; align-items: center; margin-bottom: 4px; background: var(--surf2); padding: 4px; border-radius: 4px; border: 1px solid var(--border);";

    const layerType = layer.type || "dense";

    // Type badge
    const typeBadge = document.createElement("span");
    const badgeChar = layerType === "dense" ? "D" : layerType === "dropout" ? "DO" : layerType === "conv2d" ? "CV" : layerType === "maxpool2d" ? "MP" : layerType === "lstm" ? "LSTM" : "L";
    const badgeColor = layerType === "dense" ? "var(--accent)" : layerType === "dropout" ? "var(--yellow)" : layerType === "conv2d" ? "#f0883e" : layerType === "maxpool2d" ? "#f0883e" : "#39d353";
    typeBadge.textContent = badgeChar;
    typeBadge.title = layerType;
    typeBadge.style = "font-size: 9px; color: #fff; font-weight: bold; width: 24px; text-align: center; border-radius: 3px; padding: 2px 0; background: " + badgeColor;

    // Index label
    const idxLbl = document.createElement("span");
    idxLbl.textContent = `H${index}`;
    idxLbl.style = "font-size: 10px; color: var(--text2); font-weight: bold; width: 28px;";

    // Type selector
    const typeSelect = document.createElement("select");
    typeSelect.innerHTML = '<option value="dense">Dense</option><option value="dropout">Dropout</option><option value="batchnorm">BatchNorm</option><option value="conv2d">Conv2D</option><option value="maxpool2d">MaxPool</option><option value="flatten">Flatten</option><option value="lstm">LSTM</option>';
    typeSelect.value = layerType;
    typeSelect.style = "font-size: 10px; padding: 2px; height: auto; width: 80px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px;";
    typeSelect.addEventListener("change", () => {
      this._updateLayerType(layer, typeSelect.value);
      this._renderAllLayers();
      this._emit("controlChanged", this.getConfig());
    });

    // Type-specific controls
    const controlsDiv = document.createElement("div");
    controlsDiv.style = "display: flex; gap: 4px; align-items: center; flex: 1;";

    if (layerType === "dense") {
      const nInput = document.createElement("input");
      nInput.type = "number";
      nInput.value = layer.neurons || 4;
      nInput.min = 1;
      nInput.max = 256;
      nInput.style = "width: 45px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      nInput.addEventListener("input", () => {
        layer.neurons = parseInt(nInput.value) || 1;
        this._emit("controlChanged", this.getConfig());
      });

      const nLbl = document.createElement("span");
      nLbl.textContent = "neurons";
      nLbl.style = "font-size: 10px; color: var(--text2);";

      const actSel = document.createElement("select");
      actSel.innerHTML = '<option value="relu">ReLU</option><option value="tanh" selected>Tanh</option><option value="sigmoid">Sigmoid</option><option value="leakyrelu">Leaky</option><option value="gelu">GELU</option><option value="swish">Swish</option>';
      actSel.value = layer.activation || "tanh";
      actSel.style = "font-size: 10px; padding: 2px; height: auto; width: auto;";
      actSel.addEventListener("change", () => {
        layer.activation = actSel.value;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(nInput);
      controlsDiv.appendChild(nLbl);
      controlsDiv.appendChild(actSel);
    } else if (layerType === "dropout") {
      const rateInput = document.createElement("input");
      rateInput.type = "range";
      rateInput.min = 0;
      rateInput.max = 0.9;
      rateInput.step = 0.05;
      rateInput.value = layer.rate ?? 0.5;
      rateInput.style = "width: 60px;";
      rateInput.addEventListener("input", () => {
        layer.rate = parseFloat(rateInput.value);
        rateVal.textContent = layer.rate.toFixed(2);
        this._emit("controlChanged", this.getConfig());
      });

      const rateVal = document.createElement("span");
      rateVal.textContent = (layer.rate ?? 0.5).toFixed(2);
      rateVal.style = "font-size: 10px; color: var(--text2); width: 30px;";

      const rateLbl = document.createElement("span");
      rateLbl.textContent = "rate";
      rateLbl.style = "font-size: 10px; color: var(--text2);";

      controlsDiv.appendChild(rateInput);
      controlsDiv.appendChild(rateLbl);
      controlsDiv.appendChild(rateVal);
    } else if (layerType === "conv2d") {
      const ocInput = document.createElement("input");
      ocInput.type = "number";
      ocInput.value = layer.out_channels ?? 16;
      ocInput.min = 1;
      ocInput.max = 256;
      ocInput.style = "width: 40px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      ocInput.addEventListener("input", () => {
        layer.out_channels = parseInt(ocInput.value) || 16;
        this._emit("controlChanged", this.getConfig());
      });

      const ksInput = document.createElement("input");
      ksInput.type = "number";
      ksInput.value = layer.kernel_size ?? 3;
      ksInput.min = 1;
      ksInput.max = 7;
      ksInput.style = "width: 30px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      ksInput.addEventListener("input", () => {
        layer.kernel_size = parseInt(ksInput.value) || 3;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(ocInput);
      controlsDiv.appendChild(document.createTextNode("ch"));
      controlsDiv.appendChild(ksInput);
      controlsDiv.appendChild(document.createTextNode("k"));
    } else if (layerType === "maxpool2d") {
      const psInput = document.createElement("input");
      psInput.type = "number";
      psInput.value = layer.pool_size ?? 2;
      psInput.min = 2;
      psInput.max = 4;
      psInput.style = "width: 30px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      psInput.addEventListener("input", () => {
        layer.pool_size = parseInt(psInput.value) || 2;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(psInput);
      controlsDiv.appendChild(document.createTextNode("pool"));
    } else if (layerType === "lstm") {
      const hsInput = document.createElement("input");
      hsInput.type = "number";
      hsInput.value = layer.hidden_size ?? 64;
      hsInput.min = 8;
      hsInput.max = 512;
      hsInput.style = "width: 45px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      hsInput.addEventListener("input", () => {
        layer.hidden_size = parseInt(hsInput.value) || 64;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(hsInput);
      controlsDiv.appendChild(document.createTextNode("units"));
    } else {
      const otherLbl = document.createElement("span");
      otherLbl.textContent = layerType;
      otherLbl.style = "font-size: 10px; color: var(--text2);";
      controlsDiv.appendChild(otherLbl);
    }

    // Delete button
    const delBtn = document.createElement("button");
    delBtn.innerHTML = "×";
    delBtn.style = "background: none; border: none; color: var(--red); cursor: pointer; font-weight: bold; font-size: 14px; padding: 0 4px;";
    delBtn.addEventListener("click", () => {
      this.removeHiddenLayer(layer.id);
    });

    div.appendChild(typeBadge);
    div.appendChild(idxLbl);
    div.appendChild(typeSelect);
    div.appendChild(controlsDiv);
    div.appendChild(delBtn);
    container.appendChild(div);
  }

  _updateLayerType(layer, newType) {
    layer.type = newType;
    // Reset properties based on type
    if (newType === "dense") {
      layer.neurons = layer.neurons || 4;
      layer.activation = layer.activation || "tanh";
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
    } else if (newType === "dropout") {
      layer.rate = 0.5;
      delete layer.neurons;
      delete layer.activation;
      delete layer.out_channels;
    } else if (newType === "conv2d") {
      layer.out_channels = layer.out_channels || 16;
      layer.kernel_size = layer.kernel_size || 3;
      layer.activation = "relu";
      delete layer.neurons;
      delete layer.rate;
    } else if (newType === "maxpool2d") {
      layer.pool_size = 2;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
    } else if (newType === "lstm") {
      layer.hidden_size = layer.hidden_size || 64;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
    }
  }

  // ════════════════════════════════════════════════════════
  // VIZ CHECKBOXES
  // ════════════════════════════════════════════════════════
  _initVizCheckboxes() {
    ["cbLabels","cbActs","cbBias","cbGrad"].forEach(id => {
      document.getElementById(id)?.addEventListener("change",
        () => this._emit("vizChanged", this.getVizOptions()));
    });
  }

  // ════════════════════════════════════════════════════════
  // BUILD BUTTON
  // ════════════════════════════════════════════════════════
  _initBuildBtn() {
    document.getElementById("buildBtn")?.addEventListener("click",
      () => this._emit("build", this.getConfig()));
    document.getElementById("resetBtn")?.addEventListener("click",
      () => this._emit("reset"));
  }

  // ════════════════════════════════════════════════════════
  // TRAIN / STOP BUTTONS
  // ════════════════════════════════════════════════════════
  _initTrainBtns() {
    document.getElementById("trainBtn")?.addEventListener("click",
      () => this._emit("trainToggle", "start"));
    document.getElementById("stopBtn")?.addEventListener("click",
      () => this._emit("trainToggle", "stop"));
  }

  // ════════════════════════════════════════════════════════
  // CANVAS — click/hover node inspection
  // ════════════════════════════════════════════════════════
  _initCanvasInteraction() {
    const canvas = document.getElementById("netCanvas");
    if (!canvas) return;

    canvas.addEventListener("mousemove", e => {
      this._emit("canvasHover", this._canvasPoint(canvas, e));
    });
    canvas.addEventListener("mouseleave", () => {
      this._tipEl.style.display = "none";
      canvas.style.cursor = "default";
    });
    canvas.addEventListener("click", e => {
      this._emit("canvasClick", this._canvasPoint(canvas, e));
    });
  }

  _canvasPoint(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  // ════════════════════════════════════════════════════════
  // RIGHT PANEL TABS
  // ════════════════════════════════════════════════════════
  _initRightPanelTabs() {
    document.querySelectorAll(".rptab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".rptab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll(".rppane").forEach(p => p.classList.remove("active"));
        document.getElementById("rp-" + btn.dataset.rp)?.classList.add("active");
        if (btn.dataset.rp === "wmat") this._emit("showWeights");
      });
    });
  }

  // ════════════════════════════════════════════════════════
  // TEST TAB
  // ════════════════════════════════════════════════════════
  _initTestTab() {
    document.getElementById("runTestBtn")?.addEventListener("click",
      () => this._emit("runTest", this._getTestInputs()));
    document.getElementById("randTestBtn")?.addEventListener("click",
      () => this._emit("randomTest"));
    document.getElementById("sweepBtn")?.addEventListener("click",
      () => this._emit("sweep"));
  }

  _renderTestInputs(fnMeta, firstSample) {
    const c = document.getElementById("testInpContainer");
    if (!c) return;
    if (!fnMeta) {
      c.innerHTML = '<div class="info-box">Build a network in the Train tab first.</div>';
      return;
    }
    c.innerHTML = fnMeta.input_labels.map((lbl, i) => {
      const defVal = firstSample?.[i] ?? 0;
      // Initialize sweep ranges if not exist
      if (!this._sweepRanges[i]) {
        this._sweepRanges[i] = { min: 0, max: 1, steps: 5 };
      }
      return `<div class="trow">
        <label>${lbl}</label>
        <input type="number" id="ti${i}" value="${defVal.toFixed(3)}"
               step="0.01" min="0" max="1">
      </div>`;
    }).join("");
  }

  _renderSweepRangeControls(fnMeta) {
    if (!fnMeta) return;
    const c = document.getElementById("testInpContainer");
    if (!c) return;

    // Add range controls after the inputs
    let rangeHTML = '<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border)">';
    rangeHTML += '<div style="font-size:12px;color:var(--text2);margin-bottom:8px"><b>Sweep Ranges (for Sweep All)</b></div>';
    rangeHTML += fnMeta.input_labels.map((lbl, i) => `
      <div style="margin-bottom:8px;font-size:11px">
        <label style="display:block;margin-bottom:4px">${lbl}</label>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
          <label style="min-width:30px">Min:</label>
          <input type="number" id="sweep-min-${i}" value="0" step="0.01" min="0" max="1" style="width:60px;padding:4px">
          <label style="min-width:30px">Max:</label>
          <input type="number" id="sweep-max-${i}" value="1" step="0.01" min="0" max="1" style="width:60px;padding:4px">
          <label style="min-width:30px">Step:</label>
          <input type="number" id="sweep-step-${i}" value="0.2" step="0.01" min="0.01" max="1" style="width:60px;padding:4px" title="Increment between points">
        </div>
      </div>
    `).join("");
    rangeHTML += '</div>';
    
    // Insert after input container
    c.insertAdjacentHTML('afterend', rangeHTML);
  }

  _getTestInputs() {
    const inputs = [];
    let i = 0;
    while (true) {
      const el = document.getElementById("ti" + i);
      if (!el) break;
      inputs.push(parseFloat(el.value) || 0);
      i++;
    }
    return inputs;
  }

  randomiseTestInputs(n) {
    for (let i = 0; i < n; i++) {
      const el = document.getElementById("ti" + i);
      if (el) el.value = Math.random().toFixed(3);
    }
    return this._getTestInputs();
  }

  renderTestOutput(pred, fnMeta, isSeg7) {
    const c = document.getElementById("testOutContainer");
    if (!c) return;
    let html = '<div class="tog">';
    pred.forEach((v, i) => {
      const lbl = fnMeta?.output_labels?.[i] ?? ("Out" + i);
      const col = v > 0.5 ? "var(--green)" : "var(--accent)";
      html += `<div class="tout">
        <div class="tv" style="color:${col}">${v.toFixed(4)}</div>
        <div class="tk">${lbl} (${v > 0.5 ? "≈ 1" : "≈ 0"})</div>
      </div>`;
    });
    html += "</div>";
    if (isSeg7) {
      html += `<div style="margin-top:8px">
        <div class="shd" style="margin-bottom:3px">7-Seg Preview</div>
        <div class="seg-display"><div class="seg-digit">${segSVG(pred)}</div></div>
      </div>`;
    }
    c.innerHTML = html;
  }

  renderSweepResults(results, fnMeta, isSeg7) {
    const c = document.getElementById("sweepContainer");
    if (!c) return;
    if (isSeg7) {
      c.innerHTML = '<div class="seg-display">' +
        results.map(r => {
          // Convert 4-bit input to hex digit
          const digit = (r.x[0] << 3) | (r.x[1] << 2) | (r.x[2] << 1) | r.x[3];
          const hexChar = digit < 10 ? digit : String.fromCharCode(65 + digit - 10); // 0-9, A-F
          return `<div class="seg-digit" title="Hex digit ${hexChar}"><div style="font-size:9px;text-align:center;color:var(--text2);margin-bottom:2px">${hexChar}</div>${segSVG(r.pred)}</div>`;
        }).join("") +
        "</div>";
      return;
    }
    const iL = fnMeta?.input_labels  || [];
    const oL = fnMeta?.output_labels || [];
    let correct = 0;
    let html = '<table class="io-table"><thead><tr>';
    iL.forEach(l => html += `<th>${l}</th>`);
    oL.forEach(l => html += `<th>${l}✓</th><th>${l}̂</th><th>Δ</th>`);
    html += "<th>✓</th></tr></thead><tbody>";
    results.forEach(r => {
      const ok = r.y.every((yi, i) =>
        Math.abs((r.pred[i] > 0.5 ? 1 : 0) - Math.round(yi)) < 0.5);
      if (ok) correct++;
      html += "<tr>";
      r.x.forEach(v => html += `<td>${v.toFixed(2)}</td>`);
      r.y.forEach((yi, i) => {
        const e = Math.abs(yi - r.pred[i]);
        html += `<td>${yi.toFixed(2)}</td>
                 <td style="color:${e<.1?"var(--green)":"var(--red)"}">${r.pred[i].toFixed(3)}</td>
                 <td style="color:${e<.05?"var(--green)":e<.2?"var(--yellow)":"var(--red)"}">${e.toFixed(3)}</td>`;
      });
      html += `<td><span class="badge ${ok?"ok":"err"}">${ok?"✓":"✗"}</span></td></tr>`;
    });
    const pct = (correct / results.length * 100).toFixed(1);
    const col = correct === results.length ? "var(--green)" : "var(--yellow)";
    html += `</tbody></table>
      <div style="margin-top:6px;font-size:11px;color:var(--text2)">
        Accuracy: <b style="color:${col}">${correct}/${results.length} (${pct}%)</b>
      </div>`;
    c.innerHTML = html;
  }

  // ════════════════════════════════════════════════════════
  // SAVE / LOAD
  // ════════════════════════════════════════════════════════
  _initSaveLoad() {
    document.getElementById("saveBtn")?.addEventListener("click",
      () => this._emit("exportModel"));
    document.getElementById("loadBtn")?.addEventListener("click",
      () => document.getElementById("fileInp")?.click());
    document.getElementById("fileInp")?.addEventListener("change", e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        try {
          const data = JSON.parse(ev.target.result);
          this._emit("importModel", data);
        } catch (err) {
          this.showLoadInfo(`<b style="color:var(--red)">✗ Parse error:</b> ${err.message}`, false);
        }
      };
      reader.readAsText(file);
      e.target.value = "";
    });
  }

  showSaveInfo(html) {
    const el = document.getElementById("saveInfo");
    if (el) el.innerHTML = html;
  }

  showLoadInfo(html) {
    const el = document.getElementById("loadInfo");
    if (el) el.innerHTML = html;
  }

  _renderModelSummary() {
    this._emit("requestSummary");
  }

  renderModelSummary(net) {
    const el = document.getElementById("modelSummary");
    if (!el) return;
    if (!net) { el.innerHTML = '<div class="info-box">No model built yet.</div>'; return; }
    el.innerHTML = `
      <table class="summary">
        <tr><td>Task</td><td><b>${net.func_key || "—"}</b></td></tr>
        <tr><td>Architecture</td><td><b>${net.arch_key || "—"}</b></td></tr>
        <tr><td>Topology</td><td style="font-family:monospace">[${(net.topology||[]).join(" → ")}]</td></tr>
        <tr><td>Parameters</td><td><b style="color:var(--accent)">${net.param_count ?? "—"}</b></td></tr>
        <tr><td>Epoch</td><td>${net.epoch ?? 0}</td></tr>
        <tr><td>Last Loss</td><td>${net.loss_history?.length
          ? net.loss_history[net.loss_history.length-1].toFixed(5) : "—"}</td></tr>
        <tr><td>Activation</td><td>${net.activation || "—"}</td></tr>
        <tr><td>Optimizer</td><td>${net.optimizer || "—"}</td></tr>
        <tr><td>Loss Fn</td><td>${net.loss || "—"}</td></tr>
        <tr><td>Dropout</td><td>${net.dropout ?? 0}</td></tr>
        <tr><td>Weight Decay</td><td>${net.weight_decay ?? 0}</td></tr>
      </table>`;
  }

  // ════════════════════════════════════════════════════════
  // IO TABLE
  // ════════════════════════════════════════════════════════
  renderIOTable(samples, fnMeta) {
    const c = document.getElementById("ioContainer");
    if (!c || !samples || !fnMeta) return;

    if (fnMeta.key === "seg7") {
      c.innerHTML = '<div class="seg-display">' +
        samples.map(r => `<div class="seg-digit">${segSVG(r.pred)}</div>`).join("") +
        "</div>";
      return;
    }

    const iL = fnMeta.input_labels  || [];
    const oL = fnMeta.output_labels || [];
    let html = '<table class="io-table"><thead><tr>';
    iL.forEach(l => html += `<th>${l}</th>`);
    oL.forEach(l => html += `<th>${l}✓</th><th>${l}̂</th>`);
    html += "<th></th></tr></thead><tbody>";

    samples.forEach(r => {
      const ok = r.y.every((yi, i) =>
        Math.abs((r.pred[i] > 0.5 ? 1 : 0) - Math.round(yi)) < 0.5);
      html += "<tr>";
      r.x.forEach(v => html += `<td>${(+v).toFixed(2)}</td>`);
      r.y.forEach((yi, i) => {
        const e = Math.abs(yi - r.pred[i]);
        html += `<td>${(+yi).toFixed(2)}</td>
                 <td style="color:${e<.15?"var(--green)":"var(--red)"}">${r.pred[i].toFixed(2)}</td>`;
      });
      html += `<td><span class="badge ${ok?"ok":"err"}">${ok?"✓":"✗"}</span></td></tr>`;
    });
    c.innerHTML = html + "</tbody></table>";
  }

  // ════════════════════════════════════════════════════════
  // NODE INSPECTOR
  // ════════════════════════════════════════════════════════
  renderNodeInfo(node, snapshot) {
    const el = document.getElementById("nodeInfo");
    if (!el || !node || !snapshot) return;
    const { layer, idx } = node;
    const fn    = snapshot.func || {};
    const topo  = snapshot.topology || [];
    const acts  = snapshot.activations || [];
    const layers = snapshot.layers || [];

    const isIn  = layer === 0;
    const isOut = layer === topo.length - 1;
    const lbl   = isIn  ? (fn.input_labels?.[idx]  ?? `In${idx}`)
                : isOut ? (fn.output_labels?.[idx] ?? `Out${idx}`)
                        : `Hidden L${layer} N${idx + 1}`;
    const av    = acts[layer]?.[idx] ?? 0;
    const layerData = layers[layer - 1];
    const bias  = layerData?.b?.[idx];
    const wRow  = layerData?.W?.[idx] ?? [];

    el.innerHTML = `
      <b>${lbl}</b><br>
      Type: ${isIn ? "Input" : isOut ? "Output" : "Hidden"}<br>
      Activation: <b>${av.toFixed(5)}</b>
      ${bias !== undefined ? `<br>Bias: <b>${bias.toFixed(5)}</b>` : ""}
      ${wRow.length ? `<br>Fan-in: ${wRow.length}
        <br><span style="font-size:10px;color:var(--text3)">
          ${wRow.slice(0, 10).map(w => w.toFixed(3)).join(", ")}${wRow.length > 10 ? "…" : ""}
        </span>` : ""}`;

    // Switch right panel to node tab
    document.querySelectorAll(".rptab").forEach(b =>
      b.classList.toggle("active", b.dataset.rp === "node"));
    document.querySelectorAll(".rppane").forEach(p =>
      p.classList.toggle("active", p.id === "rp-node"));
  }

  // ════════════════════════════════════════════════════════
  // WEIGHT MATRIX PANEL
  // ════════════════════════════════════════════════════════
  renderWeightMatrix(layers) {
    const c = document.getElementById("wmatContainer");
    if (!c) return;
    if (!layers?.length) {
      c.innerHTML = '<div class="info-box">Build a network first.</div>';
      return;
    }
    let html = "";
    layers.forEach((l, li) => {
      html += `<div style="margin-bottom:8px">
        <div class="shd" style="margin-bottom:3px">Layer ${li+1} (${l.W.length}×${l.W[0].length})</div>
        <div style="overflow-x:auto">
        <table style="border-collapse:collapse;font-size:9px;font-family:monospace">`;
      l.W.forEach(row => {
        html += "<tr>";
        row.forEach(w => {
          const v  = Math.min(1, Math.abs(w) / 2);
          const bg = w > 0 ? `rgba(63,185,80,${v*.5})` : `rgba(248,81,73,${v*.5})`;
          html += `<td style="background:${bg};padding:1px 3px;border:1px solid #21262d;text-align:right">${w.toFixed(2)}</td>`;
        });
        html += "</tr>";
      });
      html += `</table></div>
        <div style="font-size:10px;color:var(--text2);margin-top:2px">
          b: [${l.b.map(b => b.toFixed(2)).join(", ")}]
        </div></div>`;
    });
    c.innerHTML = html;
  }

  // ════════════════════════════════════════════════════════
  // HEADER / STATUS BAR
  // ════════════════════════════════════════════════════════
  updateStats({ epoch, loss, accuracy, param_count } = {}) {
    if (epoch     !== undefined) {
      _setText("sEpoch",  epoch);
      _setText("hEpoch",  "Epoch " + epoch);
    }
    if (loss      !== undefined) {
      _setText("sLoss",   (+loss).toFixed(4));
      _setText("hLoss",   "Loss " + (+loss).toFixed(4));
    }
    if (accuracy  !== undefined) {
      _setText("sAcc",    (accuracy * 100).toFixed(1) + "%");
      _setText("hAcc",    "Acc " + (accuracy * 100).toFixed(1) + "%");
    }
    if (param_count !== undefined) _setText("sParams", param_count);
  }

  setStatus(text, cls = "") {
    const el = document.getElementById("hStatus");
    if (!el) return;
    el.textContent = text;
    el.className   = "pill" + (cls ? " " + cls : "");
  }

  setTrainButtonState(running) {
    const train = document.getElementById("trainBtn");
    const stop  = document.getElementById("stopBtn");
    if (train) train.disabled = running;
    if (stop)  stop.disabled  = !running;
  }

  // ════════════════════════════════════════════════════════
  // CONFIG READERS
  // ════════════════════════════════════════════════════════
  getConfig() {
    return {
      arch_key:      _val("archSel"),
      func_key:      _val("funcSel"),
      inputs:        this._inputNeurons || 2,
      outputs:       this._outputNeurons || 1,
      layers:        this._hiddenLayers.map(l => {
        switch (l.type) {
          case "dropout":
            return { type: "dropout", rate: l.rate ?? 0.5 };
          case "batchnorm":
            return { type: "batchnorm" };
          case "conv2d":
            return {
              type: "conv2d",
              out_channels: l.out_channels ?? 32,
              kernel_size: l.kernel_size ?? 3,
              activation: l.activation || "relu"
            };
          case "maxpool2d":
            return { type: "maxpool2d", pool_size: l.pool_size ?? 2 };
          case "flatten":
            return { type: "flatten" };
          case "lstm":
            return { type: "lstm", hidden_size: l.hidden_size ?? 64 };
          default:
            return { type: "dense", neurons: l.neurons || 4, activation: l.activation || "tanh" };
        }
      }),
      activation:    "tanh",
      optimizer:     _val("optSel"),
      loss:          _val("lossSel"),
      lr:            Math.pow(10, +_val("lrSl")),
      weight_decay:  +_val("wdSl"),
      steps:         +_val("stepsSl"),
    };
  }

  getVizOptions() {
    return {
      showLabels:      _checked("cbLabels"),
      showActivations: _checked("cbActs"),
      showBias:        _checked("cbBias"),
      showGradients:   _checked("cbGrad"),
    };
  }

  applyPreset(p) {
    _setVal("archSel",   p.arch_key);
    _setVal("funcSel",   p.func_key);

    // Clear and rebuild hidden layers from preset
    this._hiddenLayers = [];
    if (p.layers && Array.isArray(p.layers)) {
      p.layers.forEach(l => this.addHiddenLayer(l));
    }
    this._renderAllLayers();

    _setVal("optSel",    p.optimizer);
    _setVal("lossSel",   p.loss);
    _setVal("lrSl",      Math.log10(p.lr).toFixed(2));
    _setVal("wdSl",      p.weight_decay ?? 0);

    // Trigger display updates
    ["lrSl","wdSl"].forEach(id => {
      document.getElementById(id)?.dispatchEvent(new Event("input"));
    });
    ["archSel","funcSel","optSel","lossSel"].forEach(id => {
      document.getElementById(id)?.dispatchEvent(new Event("change"));
    });

    // Emit funcChanged to update IO defaults
    const fn = (this._registry.functions || []).find(f => f.key === p.func_key);
    if (fn) this._emit("funcChanged", fn);
  }

  // ════════════════════════════════════════════════════════
  // KEYBOARD
  // ════════════════════════════════════════════════════════
  _initKeyboard() {
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); this._emit("trainToggle"); }
      if (e.code === "KeyR")  this._emit("reset");
      if (e.code === "KeyB")  this._emit("build", this.getConfig());
    });
  }

  // ════════════════════════════════════════════════════════
  // PUBLIC HELPERS for App to call
  // ════════════════════════════════════════════════════════
  renderTestInputsFor(fnMeta, firstSample) {
    this._fnMeta = fnMeta; // Store for later access
    this._renderTestInputs(fnMeta, firstSample);
    this._renderSweepRangeControls(fnMeta);
  }

  getTestInputs() { return this._getTestInputs(); }

  getSweepRanges() {
    const ranges = [];
    const inputs = this._fnMeta?.input_labels || [];
    for (let i = 0; i < inputs.length; i++) {
      const minEl = document.getElementById(`sweep-min-${i}`);
      const maxEl = document.getElementById(`sweep-max-${i}`);
      const stepEl = document.getElementById(`sweep-step-${i}`);
      ranges.push({
        min: minEl ? parseFloat(minEl.value) || 0 : 0,
        max: maxEl ? parseFloat(maxEl.value) || 1 : 1,
        step: stepEl ? parseFloat(stepEl.value) || 0.2 : 0.2
      });
    }
    return ranges;
  }
}

// ── DOM tiny helpers (module-level, not on class) ──
function _val(id)          { return document.getElementById(id)?.value ?? ""; }
function _setVal(id, v)    { const el = document.getElementById(id); if (el) el.value = v; }
function _checked(id)      { return document.getElementById(id)?.checked ?? false; }
function _setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }

// ── 7-segment SVG helper (shared by IO table and test output) ──
function segSVG(pred) {
  const s = pred.map(p => p > 0.5);
  const lines = [
    {x1:3,y1:2,x2:22,y2:2},{x1:23,y1:3,x2:23,y2:20},{x1:23,y1:21,x2:23,y2:38},
    {x1:3,y1:39,x2:22,y2:39},{x1:2,y1:21,x2:2,y2:38},{x1:2,y1:3,x2:2,y2:20},
    {x1:3,y1:20,x2:22,y2:20},
  ];
  return '<svg viewBox="0 0 26 42" xmlns="http://www.w3.org/2000/svg">' +
    lines.map((l,i) =>
      `<line x1="${l.x1}" y1="${l.y1}" x2="${l.x2}" y2="${l.y2}"
             stroke="${s[i]?"#3fb950":"#1a2a1a"}" stroke-width="2.5" stroke-linecap="round"/>`
    ).join("") + "</svg>";
}
