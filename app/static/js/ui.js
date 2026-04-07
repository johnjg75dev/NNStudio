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
    this._initDatasetSelect();
    this._initTooltips();
    this._initPresets();
    this._initSavePresetBtn();
    this._initArchSelect();
    this._initFuncSelect();
    this._initLayerEditor();
    this._initSliders();
    this._initVizCheckboxes();
    this._init2DPlotControls();
    this._initBuildBtn();
    this._initTrainBtns();
    this._initCanvasInteraction();
    this._initTestTab();
    this._initSaveLoad();
    this._initRightPanelTabs();
    this._initKeyboard();
    this._initCustomFunctionsTab();
    this._initDatasetsTab();
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
        if (btn.dataset.pg === "custom")   this._onCustomTabOpen();
        if (btn.dataset.pg === "datasets") this._onDatasetsTabOpen();
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
  // PRESETS — collapsed by default, opens modal on click
  // ════════════════════════════════════════════════════════
  _initPresets() {
    const openBtn = document.getElementById("openPresetsBtn");
    const modal = document.getElementById("presetModal");
    const closeBtn = document.getElementById("closePresetModal");
    const grid = document.getElementById("presetGrid");
    
    if (!openBtn || !modal || !grid) return;

    // Populate preset grid in modal
    this._populatePresetGrid(grid);

    // Open modal when button clicked
    openBtn.addEventListener("click", () => {
      modal.style.display = "flex";
    });

    // Close modal
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });

    // Close modal when clicking overlay
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.style.display = "none";
      }
    });
  }

  _populatePresetGrid(grid) {
    grid.innerHTML = "";
    const presets = this._registry.presets || [];
    presets.forEach(p => {
      const container = document.createElement("div");
      container.className = "pbtn-wrap";
      
      const btn = document.createElement("button");
      btn.className = "pbtn";
      btn.innerHTML = `${p.label}<div class="pa">${p.func_key}</div>`;
      btn.addEventListener("click", () => {
        this._emit("applyPreset", p);
        document.getElementById("presetModal").style.display = "none";
      });
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
    const info = document.getElementById("funcInfo");
    if (!sel) return;

    sel.innerHTML = "";
    (this._registry.functions || []).forEach(f => {
      const o = document.createElement("option");
      o.value = f.key; o.textContent = f.label;
      sel.appendChild(o);
    });

    const update = () => {
      const fn = (this._registry.functions || []).find(f => f.key === sel.value);
      if (info && fn) info.innerHTML = fn.description || "";
      this._emit("funcChanged", fn);
      // Reset dataset select when function changes
      _setVal("dsSel", "");
    };
    sel.addEventListener("change", update);
    update();
  }

  _initDatasetSelect() {
    const sel = document.getElementById("dsSel");
    if (!sel) return;
    
    sel.addEventListener("change", () => {
      const dsId = sel.value;
      if (dsId) {
        this._emit("datasetSelectedForTrain", dsId);
      }
    });
  }

  renderDatasetSelect(datasets) {
    const sel = document.getElementById("dsSel");
    if (!sel) return;
    const prevVal = sel.value;
    sel.innerHTML = '<option value="">(No Dataset Selected)</option>';

    const global   = datasets.filter(d => d.is_predefined);
    const personal = datasets.filter(d => !d.is_predefined);

    if (global.length) {
      const grp = document.createElement("optgroup");
      grp.label = "🌐 Global Datasets";
      global.forEach(d => {
        const o = document.createElement("option");
        o.value = d.id;
        o.textContent = d.name + (d.downloaded === false ? " (not downloaded)" : "");
        grp.appendChild(o);
      });
      sel.appendChild(grp);
    }

    if (personal.length) {
      const grp = document.createElement("optgroup");
      grp.label = "👤 My Datasets";
      personal.forEach(d => {
        const o = document.createElement("option");
        o.value = d.id;
        o.textContent = d.name;
        grp.appendChild(o);
      });
      sel.appendChild(grp);
    }

    sel.value = prevVal;
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

    // Add Layer via Modal
    addBtn?.addEventListener("click", () => {
      this._openLayerModal();
    });
    this._initLayerModal();

    // Seed with one hidden layer by default
    if (this._hiddenLayers.length === 0) {
      this.addHiddenLayer({ type: "dense", neurons: 4, activation: "tanh" });
    }
  }

  _initLayerModal() {
    this._layerSelectedType = null;
    this._layerCatalog = [
      { type: "dense", icon: "🧠", name: "Dense", desc: "Standard fully connected layer." },
      { type: "conv2d", icon: "🖼️", name: "Conv2D", desc: "Spatial convolution for images." },
      { type: "maxpool2d", icon: "🔽", name: "MaxPool", desc: "Downsamples spatial dimensions." },
      { type: "dropout", icon: "🗑️", name: "Dropout", desc: "Randomly zeroes elements to prevent overfitting." },
      { type: "batchnorm", icon: "⚖️", name: "BatchNorm", desc: "Normalizes layer outputs." },
      { type: "flatten", icon: "▭", name: "Flatten", desc: "Reshapes n-dimensional input to 1D." },
      { type: "lstm", icon: "🔁", name: "LSTM", desc: "Process sequences with memory gates." },
      { type: "simple_rnn", icon: "🔄", name: "Simple RNN", desc: "Basic recurrent layer." },
      { type: "embedding", icon: "🔠", name: "Embedding", desc: "Maps indices to dense vectors." },
      { type: "layernorm", icon: "📏", name: "LayerNorm", desc: "Normalizes across features." },
      { type: "multihead_attention", icon: "🎯", name: "Attention", desc: "Multi-head self attention." },
      { type: "positional_encoding", icon: "📍", name: "PosEnc", desc: "Adds position info to sequence." }
    ];

    const modal = document.getElementById("layerModal");
    if (!modal) return;
    document.getElementById("closeLayerModal")?.addEventListener("click", () => modal.style.display = "none");
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.style.display = "none"; });

    const btn = document.getElementById("confirmAddLayerBtn");
    btn?.addEventListener("click", () => {
      if (!this._layerSelectedType) return;
      const cfg = this._readLayerConfigForm(this._layerSelectedType);
      this.addHiddenLayer(cfg);
      modal.style.display = "none";
    });
  }

  _openLayerModal() {
    const modal = document.getElementById("layerModal");
    if (!modal) return;
    this._layerSelectedType = null;
    
    // Populate Grid
    const grid = document.getElementById("layerCatalogGrid");
    grid.innerHTML = "";
    this._layerCatalog.forEach(l => {
      const card = document.createElement("div");
      card.className = "layer-card";
      card.style = "background: var(--surf2); border: 1px solid var(--border); border-radius: 6px; padding: 10px; cursor: pointer; transition: all 0.2s;";
      card.innerHTML = `<div style="font-size: 24px; margin-bottom: 5px; text-align: center;">${l.icon}</div>
                        <div style="font-weight: bold; font-size: 12px; color: var(--accent); margin-bottom: 4px; text-align: center;">${l.name}</div>
                        <div style="font-size: 10px; color: var(--text2); line-height: 1.4; text-align: center;">${l.desc}</div>`;
      card.addEventListener("click", () => {
        document.querySelectorAll(".layer-card").forEach(c => c.style.borderColor = "var(--border)");
        card.style.borderColor = "var(--accent)";
        this._selectLayerType(l.type);
      });
      grid.appendChild(card);
    });

    // Clear config
    document.getElementById("layerConfigArea").innerHTML = '<div class="info-box">Select a layer type from the catalog.</div>';
    document.getElementById("confirmAddLayerBtn").disabled = true;

    modal.style.display = "flex";
  }

  _selectLayerType(type) {
    this._layerSelectedType = type;
    document.getElementById("confirmAddLayerBtn").disabled = false;
    const area = document.getElementById("layerConfigArea");
    area.innerHTML = "";
    
    const addField = (id, label, inputType, val, min, max, step) => {
      const row = document.createElement("div");
      row.style = "margin-bottom: 8px;";
      row.innerHTML = `<label style="margin-bottom: 3px;">${label}</label>`;
      const inp = document.createElement(inputType === "select" ? "select" : "input");
      inp.id = `cfgM_${id}`;
      if (inputType !== "select") {
        inp.type = inputType;
        if(inputType === "checkbox") {
          inp.checked = val;
        } else {
          inp.value = val;
        }
        if(min !== undefined) inp.min = min;
        if(max !== undefined) inp.max = max;
        if(step !== undefined) inp.step = step;
        inp.style = "width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 4px; font-size: 12px;";
        if (inputType === "checkbox") inp.style.width = "auto";
      } else {
        inp.style = "width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 4px; font-size: 12px;";
      }
      row.appendChild(inp);
      area.appendChild(row);
      return inp;
    };

    if (type === "dense") {
      addField("neurons", "Neurons", "number", 4, 1, 256);
      const sel = addField("activation", "Activation", "select");
      sel.innerHTML = '<option value="relu">ReLU</option><option value="tanh" selected>Tanh</option><option value="sigmoid">Sigmoid</option><option value="leakyrelu">Leaky</option><option value="gelu">GELU</option><option value="swish">Swish</option>';
    } else if (type === "dropout") {
      addField("rate", "Rate", "number", 0.5, 0, 0.9, 0.05);
    } else if (type === "conv2d") {
      addField("out_channels", "Out Channels", "number", 16, 1, 256);
      addField("kernel_size", "Kernel Size", "number", 3, 1, 7);
      addField("stride", "Stride", "number", 1, 1, 5);
      addField("padding", "Padding", "number", 1, 0, 5);
      const sel = addField("activation", "Activation", "select");
      sel.innerHTML = '<option value="relu" selected>ReLU</option><option value="tanh">Tanh</option><option value="sigmoid">Sigmoid</option><option value="leakyrelu">Leaky</option>';
    } else if (type === "maxpool2d") {
      addField("pool_size", "Pool Size", "number", 2, 2, 4);
      addField("stride", "Stride", "number", 2, 1, 4);
      addField("padding", "Padding", "number", 0, 0, 4);
    } else if (type === "lstm" || type === "simple_rnn") {
      addField("hidden_size", "Hidden Size", "number", 64, 8, 512);
      const cb = addField("return_sequences", "Return Sequences", "checkbox", true);
    } else if (type === "embedding") {
      addField("vocab_size", "Vocab Size", "number", 1000, 10, 100000);
      addField("embed_dim", "Embed Dimension", "number", 64, 8, 512);
    } else if (type === "layernorm") {
      addField("eps", "Epsilon", "number", 1e-5, 0, 1, 1e-6);
    } else if (type === "multihead_attention") {
      addField("num_heads", "Num Heads", "number", 8, 1, 32);
      addField("d_model", "Embed Dimension", "number", 512, 8, 2048);
    } else if (type === "positional_encoding") {
      addField("d_model", "Embed Dimension", "number", 512, 8, 2048);
      addField("max_len", "Max Length", "number", 2048, 100, 10000);
    } else {
      area.innerHTML = `<div class="info-box">No configuration required for ${type}.</div>`;
    }
  }

  _readLayerConfigForm(type) {
    const cfg = { type };
    const val = (id, isNum=true, isCb=false) => {
      const el = document.getElementById(`cfgM_${id}`);
      if (!el) return undefined;
      if (isCb) return el.checked;
      return isNum ? parseFloat(el.value) : el.value;
    };

    if (type === "dense") {
      cfg.neurons = val("neurons");
      cfg.activation = val("activation", false);
    } else if (type === "dropout") {
      cfg.rate = val("rate");
    } else if (type === "conv2d") {
      cfg.out_channels = val("out_channels");
      cfg.kernel_size = val("kernel_size");
      cfg.stride = val("stride");
      cfg.padding = val("padding");
      cfg.activation = val("activation", false);
    } else if (type === "maxpool2d") {
      cfg.pool_size = val("pool_size");
      cfg.stride = val("stride");
      cfg.padding = val("padding");
    } else if (type === "lstm" || type === "simple_rnn") {
      cfg.hidden_size = val("hidden_size");
      cfg.return_sequences = val("return_sequences", false, true);
    } else if (type === "embedding") {
      cfg.vocab_size = val("vocab_size");
      cfg.embed_dim = val("embed_dim");
    } else if (type === "layernorm") {
      cfg.eps = val("eps");
    } else if (type === "multihead_attention") {
      cfg.num_heads = val("num_heads");
      cfg.d_model = val("d_model");
    } else if (type === "positional_encoding") {
      cfg.d_model = val("d_model");
      cfg.max_len = val("max_len");
    }
    return cfg;
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
    const badgeChars = {
      dense: "D",
      dropout: "DO",
      batchnorm: "BN",
      conv2d: "CV",
      maxpool2d: "MP",
      flatten: "▭",
      lstm: "LSTM",
      simple_rnn: "RNN",
      embedding: "E",
      layernorm: "LN",
      multihead_attention: "Attn",
      positional_encoding: "PE"
    };
    const badgeColors = {
      dense: "var(--accent)",
      dropout: "var(--yellow)",
      batchnorm: "#3fb950",
      conv2d: "#f0883e",
      maxpool2d: "#f0883e",
      flatten: "#bc8cff",
      lstm: "#39d353",
      simple_rnn: "#39d353",
      embedding: "#a371f7",
      layernorm: "#3fb950",
      multihead_attention: "#ff7b72",
      positional_encoding: "#a371f7"
    };
    const badgeChar = badgeChars[layerType] || "?";
    const badgeColor = badgeColors[layerType] || "var(--text2)";
    typeBadge.textContent = badgeChar;
    typeBadge.title = layerType;
    typeBadge.style = "font-size: 9px; color: #fff; font-weight: bold; width: 24px; text-align: center; border-radius: 3px; padding: 2px 0; background: " + badgeColor;

    // Index label
    const idxLbl = document.createElement("span");
    idxLbl.textContent = `H${index}`;
    idxLbl.style = "font-size: 10px; color: var(--text2); font-weight: bold; width: 28px;";

    // Type selector
    const typeSelect = document.createElement("select");
    typeSelect.innerHTML = `
      <option value="dense">Dense</option>
      <option value="dropout">Dropout</option>
      <option value="batchnorm">BatchNorm</option>
      <option value="conv2d">Conv2D</option>
      <option value="maxpool2d">MaxPool</option>
      <option value="flatten">Flatten</option>
      <option value="lstm">LSTM</option>
      <option value="simple_rnn">RNN</option>
      <option value="embedding">Embedding</option>
      <option value="layernorm">LayerNorm</option>
      <option value="multihead_attention">MultiHeadAttn</option>
      <option value="positional_encoding">PositionalEnc</option>
    `;
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
    } else if (layerType === "simple_rnn") {
      const hsInput = document.createElement("input");
      hsInput.type = "number";
      hsInput.value = layer.hidden_size ?? 32;
      hsInput.min = 8;
      hsInput.max = 256;
      hsInput.style = "width: 45px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      hsInput.addEventListener("input", () => {
        layer.hidden_size = parseInt(hsInput.value) || 32;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(hsInput);
      controlsDiv.appendChild(document.createTextNode("units"));
    } else if (layerType === "embedding") {
      const vocabInput = document.createElement("input");
      vocabInput.type = "number";
      vocabInput.value = layer.vocab_size ?? 1000;
      vocabInput.min = 10;
      vocabInput.max = 100000;
      vocabInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      vocabInput.addEventListener("input", () => {
        layer.vocab_size = parseInt(vocabInput.value) || 1000;
        this._emit("controlChanged", this.getConfig());
      });

      const dimInput = document.createElement("input");
      dimInput.type = "number";
      dimInput.value = layer.embed_dim ?? 64;
      dimInput.min = 8;
      dimInput.max = 512;
      dimInput.style = "width: 40px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      dimInput.addEventListener("input", () => {
        layer.embed_dim = parseInt(dimInput.value) || 64;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(vocabInput);
      controlsDiv.appendChild(document.createTextNode("vocab"));
      controlsDiv.appendChild(dimInput);
      controlsDiv.appendChild(document.createTextNode("dim"));
    } else if (layerType === "layernorm") {
      const epsInput = document.createElement("input");
      epsInput.type = "number";
      epsInput.value = layer.eps ?? 1e-5;
      epsInput.step = 1e-6;
      epsInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      epsInput.addEventListener("input", () => {
        layer.eps = parseFloat(epsInput.value) || 1e-5;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(epsInput);
      controlsDiv.appendChild(document.createTextNode("eps"));
    } else if (layerType === "multihead_attention") {
      const headsInput = document.createElement("input");
      headsInput.type = "number";
      headsInput.value = layer.num_heads ?? 8;
      headsInput.min = 1;
      headsInput.max = 32;
      headsInput.style = "width: 40px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      headsInput.addEventListener("input", () => {
        layer.num_heads = parseInt(headsInput.value) || 8;
        this._emit("controlChanged", this.getConfig());
      });

      const dimInput = document.createElement("input");
      dimInput.type = "number";
      dimInput.value = layer.d_model ?? 512;
      dimInput.min = 8;
      dimInput.max = 2048;
      dimInput.step = 8;
      dimInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      dimInput.addEventListener("input", () => {
        layer.d_model = parseInt(dimInput.value) || 512;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(headsInput);
      controlsDiv.appendChild(document.createTextNode("heads"));
      controlsDiv.appendChild(dimInput);
      controlsDiv.appendChild(document.createTextNode("d_model"));
    } else if (layerType === "positional_encoding") {
      const dimInput = document.createElement("input");
      dimInput.type = "number";
      dimInput.value = layer.d_model ?? 512;
      dimInput.min = 8;
      dimInput.max = 2048;
      dimInput.step = 8;
      dimInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      dimInput.addEventListener("input", () => {
        layer.d_model = parseInt(dimInput.value) || 512;
        this._emit("controlChanged", this.getConfig());
      });

      const maxlenInput = document.createElement("input");
      maxlenInput.type = "number";
      maxlenInput.value = layer.max_len ?? 2048;
      maxlenInput.min = 100;
      maxlenInput.max = 10000;
      maxlenInput.step = 100;
      maxlenInput.style = "width: 50px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 2px; font-size: 11px;";
      maxlenInput.addEventListener("input", () => {
        layer.max_len = parseInt(maxlenInput.value) || 2048;
        this._emit("controlChanged", this.getConfig());
      });

      controlsDiv.appendChild(dimInput);
      controlsDiv.appendChild(document.createTextNode("d_model"));
      controlsDiv.appendChild(maxlenInput);
      controlsDiv.appendChild(document.createTextNode("max_len"));
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
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "dropout") {
      layer.rate = 0.5;
      delete layer.neurons;
      delete layer.activation;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "batchnorm") {
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "conv2d") {
      layer.out_channels = layer.out_channels || 16;
      layer.kernel_size = layer.kernel_size || 3;
      layer.activation = "relu";
      delete layer.neurons;
      delete layer.rate;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "maxpool2d") {
      layer.pool_size = 2;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "flatten") {
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "lstm") {
      layer.hidden_size = layer.hidden_size || 64;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "simple_rnn") {
      layer.hidden_size = layer.hidden_size || 32;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "embedding") {
      layer.vocab_size = layer.vocab_size || 1000;
      layer.embed_dim = layer.embed_dim || 64;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.eps;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "layernorm") {
      layer.eps = layer.eps || 1e-5;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.num_heads;
      delete layer.d_model;
      delete layer.max_len;
    } else if (newType === "multihead_attention") {
      layer.num_heads = layer.num_heads || 8;
      layer.d_model = layer.d_model || 512;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.max_len;
    } else if (newType === "positional_encoding") {
      layer.d_model = layer.d_model || 512;
      layer.max_len = layer.max_len || 2048;
      delete layer.neurons;
      delete layer.activation;
      delete layer.rate;
      delete layer.out_channels;
      delete layer.kernel_size;
      delete layer.hidden_size;
      delete layer.pool_size;
      delete layer.vocab_size;
      delete layer.embed_dim;
      delete layer.eps;
      delete layer.num_heads;
    }
  }

  // ════════════════════════════════════════════════════════
  // VIZ CHECKBOXES
  // ════════════════════════════════════════════════════════
  _initVizCheckboxes() {
    ["cbLabels","cbActs","cbBias","cbGrad","cbDeadWeights"].forEach(id => {
      document.getElementById(id)?.addEventListener("change",
        () => this._emit("vizChanged", this.getVizOptions()));
    });
  }

  _init2DPlotControls() {
    ["cbPredBoundary", "cbDataPoints"].forEach(id => {
      document.getElementById(id)?.addEventListener("change",
        () => this._emit("plot2dChanged", this.get2DPlotOptions()));
    });
  }

  get2DPlotOptions() {
    return {
      showBoundary: document.getElementById("cbPredBoundary")?.checked ?? true,
      showPoints: document.getElementById("cbDataPoints")?.checked ?? false,
    };
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
    document.getElementById("runTestBtn")?.addEventListener("click", () => {
      const sl = document.getElementById("testStartLayer")?.value;
      const el = document.getElementById("testEndLayer")?.value;
      this._emit("runTest", {
        x: this._getTestInputs(),
        start_layer: sl ? parseInt(sl) : 0,
        end_layer: el && el !== "" ? parseInt(el) : null
      });
    });
    document.getElementById("randTestBtn")?.addEventListener("click",
      () => this._emit("randomTest"));
    document.getElementById("sweepBtn")?.addEventListener("click", () => {
      const sl = document.getElementById("testStartLayer")?.value;
      const el = document.getElementById("testEndLayer")?.value;
      this._emit("sweep", {
        start_layer: sl ? parseInt(sl) : 0,
        end_layer: el && el !== "" ? parseInt(el) : null
      });
    });
  }

  _renderTestInputs(fnMeta, firstSample) {
    const c = document.getElementById("testInpContainer");
    if (!c) return;
    if (!fnMeta) {
      c.innerHTML = '<div class="info-box">Build a network in the Train tab first.</div>';
      return;
    }
    const numInputs = fnMeta.input_labels ? fnMeta.input_labels.length : (fnMeta.inputs || 0);
    const side = Math.sqrt(numInputs);
    // detect if it's an image
    if (Number.isInteger(side) && side >= 8 && side <= 64) {
      this._testCanvasParams = { side };
      c.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:8px;">
           <div class="info-box" style="margin:0; font-size:11px;">Draw on the canvas to test custom image inputs.</div>
           <canvas id="testDrawingCanvas" width="${side}" height="${side}" style="width: 160px; height: 160px; border: 1px solid var(--border); background-color: #000; cursor: crosshair; image-rendering: pixelated; margin: 0 auto; display: block;"></canvas>
           <div class="btn-row" style="justify-content:center;">
             <button class="btn secondary" id="clearTestCanvasBtn" style="font-size:11px;">Clear Canvas</button>
           </div>
        </div>
      `;
      const canvas = document.getElementById("testDrawingCanvas");
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, side, side);
      
      let drawing = false;
      const paint = (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);
        if (x >= 0 && x < side && y >= 0 && y < side) {
           ctx.fillStyle = "white";
           ctx.fillRect(x, y, 1, 1);
           if (side > 16) {
             ctx.fillRect(x-1, y, 1, 1);
             ctx.fillRect(x+1, y, 1, 1);
             ctx.fillRect(x, y-1, 1, 1);
             ctx.fillRect(x, y+1, 1, 1);
           }
        }
      };

      canvas.addEventListener("mousedown", e => { drawing = true; paint(e); });
      canvas.addEventListener("mousemove", e => { if (drawing) paint(e); });
      canvas.addEventListener("mouseup", () => drawing = false);
      canvas.addEventListener("mouseleave", () => drawing = false);
      
      document.getElementById("clearTestCanvasBtn").addEventListener("click", () => {
         ctx.fillStyle = "black";
         ctx.fillRect(0, 0, side, side);
      });

      this._testCanvasParams.ctx = ctx;
      this._testCanvasParams.canvas = canvas;
      return;
    }

    this._testCanvasParams = null;
    c.innerHTML = (fnMeta.input_labels || []).map((lbl, i) => {
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
    if (!fnMeta || !fnMeta.input_labels || !fnMeta.input_labels.length) return;
    // Don't render sweep controls for image inputs
    if (this._testCanvasParams) return;
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
    if (this._testCanvasParams) {
        const { side, ctx } = this._testCanvasParams;
        const imgData = ctx.getImageData(0, 0, side, side).data;
        const inputs = [];
        for (let i = 0; i < imgData.length; i += 4) {
            inputs.push(imgData[i] / 255.0); // using red channel as greyscale proxy
        }
        return inputs;
    }

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
    
    if (pred && Array.isArray(pred)) {
      const side = Math.sqrt(pred.length);
      if (Number.isInteger(side) && side >= 8 && side <= 64) {
         c.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:8px;">
               <canvas id="testDrawnOut" width="${side}" height="${side}" style="width: 160px; height: 160px; border: 1px solid var(--border); background-color: #000; image-rendering: pixelated; margin: 0 auto; display: block;"></canvas>
            </div>
         `;
         const ctx = document.getElementById("testDrawnOut").getContext("2d");
         const imgData = ctx.createImageData(side, side);
         for(let i = 0; i < pred.length; i++) {
             const v = Math.max(0, Math.min(255, Math.floor(pred[i] * 255)));
             imgData.data[i*4+0] = v;
             imgData.data[i*4+1] = v;
             imgData.data[i*4+2] = v;
             imgData.data[i*4+3] = 255;
         }
         ctx.putImageData(imgData, 0, 0);
         return;
      }
    }

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
      
    document.getElementById("dbSaveBtn")?.addEventListener("click", () => {
        const name = document.getElementById("dbModelName").value;
        if (!name) return alert("Provide a model name!");
        this._emit("saveDbModel", name);
    });

    document.getElementById("dbModelList")?.addEventListener("click", e => {
       const btn = e.target.closest("button");
       if (!btn) return;
       const id = btn.dataset.id;
       if (!id) return;
       if (btn.classList.contains("btn-load")) this._emit("loadDbModel", id);
       if (btn.classList.contains("btn-del")) this._emit("deleteDbModel", id);
    });
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

  showSaveInfo(html, ok=true) {
    const el = document.getElementById("saveInfo");
    if (!el) return;
    el.innerHTML = html;
    el.className = "info-box " + (ok ? "ok" : "err");
    if (ok) setTimeout(() => { el.className = "info-box"; el.innerHTML = "Exported."; }, 4000);
  }

  showLoadInfo(html, ok=true) {
    const el = document.getElementById("loadInfo");
    if (!el) return;
    el.innerHTML = html;
    el.className = "info-box " + (ok ? "ok" : "err");
  }

  renderDbModelList(models) {
    const c = document.getElementById("dbModelList");
    if (!c) return;
    if (models.length === 0) {
       c.innerHTML = '<div class="info-box">No saved models found.</div>';
       return;
    }
    c.innerHTML = models.map(m => `
       <div style="display:flex; justify-content:space-between; align-items:center; padding: 6px; border-bottom: 1px solid var(--border)">
         <div>
            <div style="font-size:12px;font-weight:600;">${m.name}</div>
            <div style="font-size:10px;color:var(--text3);">${m.architecture_name || 'Net'} • ${m.epochs_trained} Epochs • Loss: ${m.final_loss ? m.final_loss.toFixed(3) : '?'}</div>
         </div>
         <div class="btn-row">
            <button class="btn success btn-load" data-id="${m.id}" style="font-size:10px;padding:2px 6px;">Load</button>
            <button class="btn danger btn-del" data-id="${m.id}" style="font-size:10px;padding:2px 6px;">Del</button>
         </div>
       </div>
    `).join("");
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
        samples.map((r, idx) => `<div class="seg-digit" data-sample-idx="${idx}">${segSVG(r.pred)}</div>`).join("") +
        "</div>";
      
      // Attach click handlers to 7-segment display digits
      const self = this;
      document.querySelectorAll(".seg-digit").forEach((digit) => {
        const sampleIdx = parseInt(digit.dataset.sampleIdx, 10);
        digit.addEventListener("click", () => {
          if (sampleIdx >= 0 && sampleIdx < samples.length) {
            const sample = samples[sampleIdx];
            if (sample && Array.isArray(sample.x) && sample.x.length > 0) {
              self._emit("ioRowClicked", sample.x);
            }
          }
        });
      });
      return;
    }

    const iL = fnMeta.input_labels  || [];
    const oL = fnMeta.output_labels || [];
    let html = '<table class="io-table"><thead><tr>';
    iL.forEach(l => html += `<th>${l}</th>`);
    oL.forEach(l => html += `<th>${l}✓</th><th>${l}̂</th>`);
    html += "<th></th></tr></thead><tbody>";

    samples.forEach((r, sampleIdx) => {
      const ok = r.y.every((yi, i) =>
        Math.abs((r.pred[i] > 0.5 ? 1 : 0) - Math.round(yi)) < 0.5);
      html += `<tr class="io-row" data-sample-idx="${sampleIdx}">`;
      r.x.forEach(v => html += `<td>${(+v).toFixed(2)}</td>`);
      r.y.forEach((yi, i) => {
        const e = Math.abs(yi - r.pred[i]);
        html += `<td>${(+yi).toFixed(2)}</td>
                 <td style="color:${e<.15?"var(--green)":"var(--red)"}">${r.pred[i].toFixed(2)}</td>`;
      });
      html += `<td><span class="badge ${ok?"ok":"err"}">${ok?"✓":"✗"}</span></td></tr>`;
    });
    c.innerHTML = html + "</tbody></table>";

    // Attach click handlers to rows
    const self = this;
    document.querySelectorAll(".io-row").forEach((row) => {
      const sampleIdx = parseInt(row.dataset.sampleIdx, 10);
      row.addEventListener("click", () => {
        if (sampleIdx >= 0 && sampleIdx < samples.length) {
          const sample = samples[sampleIdx];
          if (sample && Array.isArray(sample.x) && sample.x.length > 0) {
            self._emit("ioRowClicked", sample.x);
          }
        }
      });
    });
  }

  // ════════════════════════════════════════════════════════
  // NODE INSPECTOR
  // ════════════════════════════════════════════════════════
  renderNodeInfo(node, snapshot, showingInfluences = false) {
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

    let html = `<b>${lbl}</b>`;
    
    if (isIn) {
      html += `<br>Type: Input<br>Activation: <b>${av.toFixed(5)}</b>`;
    } else {
      html += `<br>Type: ${isOut ? "Output" : "Hidden"}<br>Activation: <b>${av.toFixed(5)}</b>`;
      if (bias !== undefined) {
        html += `<br>Bias: <b>${bias.toFixed(5)}</b>`;
      }
    }

    // Show influences breakdown when enabled
    if (showingInfluences && !isIn) {
      html += `<hr class="dv" style="margin: 8px 0;">
        <div style="font-size:11px;">
          <b style="color:var(--accent)">🔍 Input Influences</b><br>`;
      
      if (wRow.length > 0) {
        const prevActs = acts[layer - 1] || [];
        const contributions = wRow.map((w, i) => ({
          idx: i,
          weight: w,
          activation: prevActs[i] ?? 0,
          contribution: w * (prevActs[i] ?? 0)
        })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
        
        html += `<table style="font-size:10px; width:100%; border-collapse:collapse;">
          <tr style="color:var(--text2); border-bottom:1px solid var(--border);">
            <td style="padding:2px 0;">Input</td>
            <td style="text-align:right; padding-right:4px;">Weight</td>
            <td style="text-align:right;">Contrib</td>
          </tr>`;
        
        // Show top 8 contributors
        contributions.slice(0, 8).forEach(c => {
          const sign = c.weight > 0 ? "+" : "";
          const color = c.weight > 0 ? "var(--green)" : "var(--red)";
          html += `<tr style="border-bottom:1px solid var(--border); height:18px;">
            <td style="padding:2px 0; color:var(--text3);">N${c.idx}</td>
            <td style="text-align:right; padding-right:4px; color:${color};">${sign}${c.weight.toFixed(3)}</td>
            <td style="text-align:right; font-weight:bold; color:${c.contribution > 0 ? 'var(--green)' : 'var(--red)'};">${c.contribution.toFixed(3)}</td>
          </tr>`;
        });
        
        if (contributions.length > 8) {
          html += `<tr style="color:var(--text3); font-size:9px;">
            <td colspan="3" style="padding:2px 0;">+${contributions.length - 8} more inputs</td>
          </tr>`;
        }
        
        html += `</table>`;
      } else {
        html += `<div style="color:var(--text3); font-size:10px;">No weighted inputs</div>`;
      }
      
      html += `</div>`;
    } else if (!isIn) {
      // Show normal fan-in info when not showing influences
      if (layerData?.type === "conv2d" && Array.isArray(wRow)) {
        html += `<br><br><b style="color:var(--accent)">🔍 Filters (Out Channel ${idx})</b><br>`;
        html += `<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:5px; max-height:150px; overflow-y:auto; padding:4px;">`;
        wRow.forEach((inChannelFilter, inChIdx) => {
          if (!Array.isArray(inChannelFilter)) return;
          html += `<div><div style="font-size:9px;color:var(--text3);text-align:center;margin-bottom:2px;">In Ch ${inChIdx}</div>
            <table style="border-collapse:collapse;border:1px solid var(--border);">`;
          inChannelFilter.forEach(row => {
            if (!Array.isArray(row)) return;
            html += `<tr>`;
            row.forEach(val => {
              const v  = Math.min(1, Math.abs(val) / 2);
              const bg = val > 0 ? `rgba(63,185,80,${v*.8})` : `rgba(248,81,73,${v*.8})`;
              html += `<td style="background:${bg};width:12px;height:12px;font-size:0;" title="${val.toFixed(4)}"></td>`;
            });
            html += `</tr>`;
          });
          html += `</table></div>`;
        });
        html += `</div>`;
      } else {
        html += `${wRow.length ? `<br>Fan-in: ${wRow.length}
          <br><span style="font-size:10px;color:var(--text3)">
            ${wRow.slice(0, 10).map(w => typeof w === 'number' ? w.toFixed(3) : "[..]").join(", ")}${wRow.length > 10 ? "…" : ""}
          </span>` : ""}`;
      }
    }

    html += `<br><br>
      <button class="btn ${showingInfluences ? 'success' : 'secondary'}" id="btnShowInfluences" style="width:100%;font-size:11px;">
        ${showingInfluences ? '✓ Insights Shown' : '✨ Show Influences'}
      </button>`;

    el.innerHTML = html;

    // Wire the button event
    document.getElementById("btnShowInfluences")?.addEventListener("click", () => {
      this._emit("showInfluences", { node });
    });

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
      if (!l.W || !Array.isArray(l.W) || !l.W.length) {
        html += `<div style="margin-bottom:8px">
          <div class="shd" style="margin-bottom:3px">Layer ${li+1} (${l.type || 'unknown'})</div>
          <div class="info-box" style="font-size:10px">Weight matrix not available for this layer type.</div>
        </div>`;
        return;
      }
      const rows = l.W.length;
      const cols = Array.isArray(l.W[0]) ? l.W[0].length : 1;
      html += `<div style="margin-bottom:8px">
        <div class="shd" style="margin-bottom:3px">Layer ${li+1} (${rows}×${cols})</div>
        <div style="overflow-x:auto">
        <table style="border-collapse:collapse;font-size:9px;font-family:monospace">`;
      l.W.forEach(row => {
        html += "<tr>";
        (Array.isArray(row) ? row : []).forEach(w => {
          const wn = typeof w === 'number' ? w : parseFloat(w) || 0;
          const v  = Math.min(1, Math.abs(wn) / 2);
          const bg = wn > 0 ? `rgba(63,185,80,${v*.5})` : `rgba(248,81,73,${v*.5})`;
          html += `<td style="background:${bg};padding:1px 3px;border:1px solid #21262d;text-align:right">${wn.toFixed(2)}</td>`;
        });
        html += "</tr>";
      });
      const biases = Array.isArray(l.b) ? l.b : [];
      html += `</table></div>
        <div style="font-size:10px;color:var(--text2);margin-top:2px">
          b: [${biases.map(b => (typeof b === 'number' ? b : parseFloat(b) || 0).toFixed(2)).join(", ")}]
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
      ds_id:         _val("dsSel"),
      inputs:        this._inputNeurons || 2,
      outputs:       this._outputNeurons || 1,
      layers:        this._hiddenLayers.map(l => {
        const out = { type: l.type };
        if (l.type === "dense") { out.neurons = l.neurons; out.activation = l.activation; }
        else if (l.type === "dropout") { out.rate = l.rate; }
        else if (l.type === "conv2d") { out.out_channels = l.out_channels; out.kernel_size = l.kernel_size; out.stride = l.stride; out.padding = l.padding; out.activation = l.activation; }
        else if (l.type === "maxpool2d") { out.pool_size = l.pool_size; out.stride = l.stride; out.padding = l.padding; }
        else if (l.type === "lstm" || l.type === "simple_rnn") { out.hidden_size = l.hidden_size; out.return_sequences = l.return_sequences; }
        else if (l.type === "embedding") { out.vocab_size = l.vocab_size; out.embed_dim = l.embed_dim; }
        else if (l.type === "layernorm") { out.eps = l.eps; }
        else if (l.type === "multihead_attention") { out.num_heads = l.num_heads; out.d_model = l.d_model; }
        else if (l.type === "positional_encoding") { out.d_model = l.d_model; out.max_len = l.max_len; }
        return out;
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
      showDeadWeights: _checked("cbDeadWeights"),
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
    document.addEventListener("keydown", (e) => {
      if (
        e.target.tagName === "INPUT" ||
        e.target.tagName === "SELECT" ||
        e.target.tagName === "TEXTAREA" ||
        e.target.isContentEditable
      ) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        this._emit("trainToggle");
      }
      if (e.code === "KeyR") this._emit("reset");
      if (e.code === "KeyB") this._emit("build", this.getConfig());
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

  renderTestLayerBounds(snapshot) {
    if (!snapshot || !snapshot.layers) return;
    const sl = document.getElementById("testStartLayer");
    const el = document.getElementById("testEndLayer");
    if (!sl || !el) return;

    sl.innerHTML = `<option value="0">Input Layer</option>`;
    el.innerHTML = `<option value="">Output Layer</option>`;
    
    snapshot.layers.forEach((layer, i) => {
      const type = layer.type || "Layer";
      const lbl = `(L${i+1}) ${type}`;
      sl.innerHTML += `<option value="${i+1}">${lbl}</option>`;
      el.innerHTML += `<option value="${i+1}">${lbl}</option>`;
    });
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

  // ════════════════════════════════════════════════════════
  // CUSTOM FUNCTIONS TAB
  // ════════════════════════════════════════════════════════
  _initCustomFunctionsTab() {
    this._customTemplates = null;
    this._customExamples = null;
    this._selectedFuncId = null;

    document.getElementById("newFuncBtn")?.addEventListener("click", () => this._createNewFunc());
    document.getElementById("cfSaveBtn")?.addEventListener("click", () => this._saveCustomFunc());
    document.getElementById("cfTestBtn")?.addEventListener("click", () => this._testCustomFunc());
    document.getElementById("cfDeleteBtn")?.addEventListener("click", () => this._deleteCustomFunc());
    
    document.getElementById("cfLang")?.addEventListener("change", () => this._onLangChange());
    
    // Examples menu
    const exBtn = document.getElementById("cfExampleBtn");
    const exMenu = document.getElementById("cfExampleMenu");
    exBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      exMenu.style.display = exMenu.style.display === "block" ? "none" : "block";
    });
    document.addEventListener("click", () => { if (exMenu) exMenu.style.display = "none"; });
  }

  async _onCustomTabOpen() {
    if (!this._customTemplates) {
      try {
        const res = await API.getCustomTemplates();
        this._customTemplates = res.templates;
        this._customExamples = res.examples;
      } catch (e) { console.error("Failed to load templates:", e); }
    }
    this._refreshCustomFuncList();
  }

  _refreshCustomFuncList() {
    this._emit("refreshCustomFuncs");
  }

  renderCustomFuncList(funcs) {
    const list = document.getElementById("customFuncList");
    if (!list) return;
    list.innerHTML = "";
    if (funcs.length === 0) {
      list.innerHTML = '<div class="info-box">No custom functions yet.</div>';
      return;
    }
    funcs.forEach(f => {
      const item = document.createElement("div");
      item.className = "pbtn-wrap" + (this._selectedFuncId === f.id ? " active" : "");
      item.style = "margin-bottom: 4px;";
      
      const btn = document.createElement("button");
      btn.className = "pbtn";
      btn.style = "text-align: left; padding: 6px 8px;";
      btn.innerHTML = `<b>${f.name}</b><div class="pa">${f.language} · ${f.num_inputs}→${f.num_outputs}</div>`;
      btn.addEventListener("click", () => {
        this._selectedFuncId = f.id;
        this._emit("customFuncSelected", f.id);
        this._refreshCustomFuncList(); // to update active state
      });
      
      item.appendChild(btn);
      list.appendChild(item);
    });
  }

  _createNewFunc() {
    this._selectedFuncId = null;
    document.getElementById("funcEditorEmpty").style.display = "none";
    document.getElementById("funcEditor").style.display = "flex";
    
    document.getElementById("cfName").value = "New Function";
    document.getElementById("cfLang").value = "python";
    document.getElementById("cfInputs").value = 2;
    document.getElementById("cfOutputs").value = 1;
    document.getElementById("cfClassify").checked = false;
    document.getElementById("cfCode").value = this._customTemplates?.python?.code 
      .replace("{num_inputs}", "2").replace("{num_outputs}", "1") || "";
    document.getElementById("cfTestResult").style.display = "none";
    document.getElementById("cfDeleteBtn").style.display = "none";
    
    this._renderExamples("python");
  }

  selectCustomFunc(f) {
    this._selectedFuncId = f.id;
    document.getElementById("funcEditorEmpty").style.display = "none";
    document.getElementById("funcEditor").style.display = "flex";
    
    document.getElementById("cfName").value = f.name;
    document.getElementById("cfLang").value = f.language;
    document.getElementById("cfInputs").value = f.num_inputs;
    document.getElementById("cfOutputs").value = f.num_outputs;
    document.getElementById("cfClassify").checked = f.is_classification;
    document.getElementById("cfCode").value = f.code;
    document.getElementById("cfTestResult").style.display = "none";
    document.getElementById("cfDeleteBtn").style.display = "block";
    
    this._renderExamples(f.language);
  }

  _onLangChange() {
    const lang = document.getElementById("cfLang").value;
    const code = document.getElementById("cfCode").value;
    const nIn = document.getElementById("cfInputs").value;
    const nOut = document.getElementById("cfOutputs").value;

    // Only swap template if current code is empty or matches previous template
    if (!code || code.includes("Custom training function")) {
      document.getElementById("cfCode").value = this._customTemplates?.[lang]?.code
        .replace("{num_inputs}", nIn).replace("{num_outputs}", nOut) || "";
    }
    this._renderExamples(lang);
  }

  _renderExamples(lang) {
    const menu = document.getElementById("cfExampleMenu");
    if (!menu || !this._customExamples) return;
    const examples = this._customExamples[lang] || [];
    menu.innerHTML = "";
    examples.forEach((ex, i) => {
      const div = document.createElement("div");
      div.className = "ex-item";
      div.style = "padding:4px 8px; cursor:pointer; font-size:10px; color:var(--text2); border-radius:3px;";
      div.textContent = ex.name;
      div.onmouseover = () => { div.style.background = "var(--surf2)"; div.style.color = "var(--text)"; };
      div.onmouseout = () => { div.style.background = "transparent"; div.style.color = "var(--text2)"; };
      div.onclick = (e) => {
        e.stopPropagation();
        this.applyExample(i);
        menu.style.display = "none";
      };
      menu.appendChild(div);
    });
  }

  applyExample(idx) {
    const lang = document.getElementById("cfLang").value;
    const ex = this._customExamples[lang][idx];
    if (ex) {
      document.getElementById("cfCode").value = ex.code;
    }
  }

  _saveCustomFunc() {
    const data = {
      name: document.getElementById("cfName").value,
      language: document.getElementById("cfLang").value,
      num_inputs: parseInt(document.getElementById("cfInputs").value),
      num_outputs: parseInt(document.getElementById("cfOutputs").value),
      is_classification: document.getElementById("cfClassify").checked,
      code: document.getElementById("cfCode").value,
    };
    
    if (this._selectedFuncId) {
      this._emit("updateCustomFunc", { id: this._selectedFuncId, data });
    } else {
      this._emit("createCustomFunc", data);
    }
  }

  _testCustomFunc() {
    if (!this._selectedFuncId) {
      alert("Save the function first before testing.");
      return;
    }
    const nIn = parseInt(document.getElementById("cfInputs").value);
    const input = Array.from({ length: nIn }, () => Math.random());
    this._emit("testCustomFunc", { id: this._selectedFuncId, input });
  }

  showCustomTestResult(res) {
    const el = document.getElementById("cfTestResult");
    el.style.display = "block";
    if (res.success) {
      el.innerHTML = `<span style="color:var(--green)">✓ Success</span> (${res.exec_time.toFixed(4)}s)<br>` +
                     `In:  [${res.input.map(v => v.toFixed(2)).join(", ")}]<br>` +
                     `Out: [${res.output.map(v => v.toFixed(4)).join(", ")}]`;
    } else {
      el.innerHTML = `<span style="color:var(--red)">✗ Error:</span> ${res.error}`;
    }
  }

  _deleteCustomFunc() {
    if (this._selectedFuncId && confirm("Are you sure you want to delete this function?")) {
      this._emit("deleteCustomFunc", this._selectedFuncId);
      this._selectedFuncId = null;
      document.getElementById("funcEditor").style.display = "none";
      document.getElementById("funcEditorEmpty").style.display = "flex";
    }
  }

  // ════════════════════════════════════════════════════════
  // DATASETS TAB
  // ════════════════════════════════════════════════════════
  _initDatasetsTab() {
    this._selectedDatasetId = null;
    this._datasetData = []; // Rows for tabular editor

    document.getElementById("newDatasetBtn")?.addEventListener("click", () => this._createNewDataset());
    document.getElementById("dsSaveBtn")?.addEventListener("click", () => this._saveDataset());
    document.getElementById("dsType")?.addEventListener("change", () => this._onDsTypeChange());
    document.getElementById("dsAddRowBtn")?.addEventListener("click", () => this._addDsRow());
    document.getElementById("dsClearBtn")?.addEventListener("click", () => {
      if (confirm("Clear all rows?")) { this._datasetData = []; this._renderDsTable(); }
    });
    document.getElementById("dsSyntheticBtn")?.addEventListener("click", () => this._showSyntheticModal());
    document.getElementById("dsSyntheticCloseBtn")?.addEventListener("click", () => {
      document.getElementById("dsSyntheticModal").style.display = "none";
    });
    document.getElementById("dsSyntheticGenerateBtn")?.addEventListener("click", () => this._generateSynthetic());
    
    // CSV Import
    document.getElementById("dsImportCsvBtn")?.addEventListener("click", () => this._importCsv());

    // Image Editor
    this._initImageEditor();
  }

  async _onDatasetsTabOpen() {
    this._emit("refreshDatasets");
  }

  renderDatasetList(datasets) {
    const list = document.getElementById("datasetList");
    if (!list) return;
    list.innerHTML = "";
    if (datasets.length === 0) {
      list.innerHTML = '<div class="info-box">No datasets yet.</div>';
      return;
    }
    datasets.forEach(d => {
      const item = document.createElement("div");
      item.className = "pbtn-wrap" + (this._selectedDatasetId === d.id ? " active" : "");
      item.style = "margin-bottom: 4px;";
      
      const btn = document.createElement("button");
      btn.className = "pbtn";
      btn.style = "text-align: left; padding: 6px 8px;";
      btn.innerHTML = `<b>${d.name}</b><div class="pa">${d.ds_type} · ${d.num_inputs} in</div>`;
      btn.addEventListener("click", () => {
        this._selectedDatasetId = d.id;
        this._emit("datasetSelected", d.id);
        this.renderDatasetList(datasets); // to update active state
      });
      
      item.appendChild(btn);
      list.appendChild(item);
    });
  }

  _createNewDataset() {
    this._selectedDatasetId = null;
    this._datasetData = [];
    document.getElementById("datasetEditorEmpty").style.display = "none";
    document.getElementById("datasetEditor").style.display = "flex";
    
    document.getElementById("dsName").value = "New Dataset";
    document.getElementById("dsType").value = "tabular";
    document.getElementById("dsInputOnly").checked = false;
    
    this._onDsTypeChange();
  }

  selectDataset(ds) {
    this._selectedDatasetId = ds.id;
    this._datasetData = ds.data || [];
    document.getElementById("datasetEditorEmpty").style.display = "none";
    document.getElementById("datasetEditor").style.display = "flex";
    
    document.getElementById("dsName").value = ds.name;
    document.getElementById("dsType").value = ds.ds_type;
    document.getElementById("dsInputOnly").checked = ds.is_input_only;
    
    this._onDsTypeChange();
    if (ds.ds_type === "tabular") this._renderDsTable();
  }

  _onDsTypeChange() {
    const type = document.getElementById("dsType").value;
    document.getElementById("dsTabularView").style.display = type === "tabular" ? "flex" : "none";
    document.getElementById("dsImageView").style.display = type === "image" ? "flex" : "none";
    document.getElementById("dsPredefinedView").style.display = type === "mnist" ? "block" : "none";
    
    if (type === "tabular") this._renderDsTable();
    if (type === "image") this._renderPixelGrid();
  }

  _renderDsTable() {
    const table = document.getElementById("dsTable");
    const isInputOnly = document.getElementById("dsInputOnly").checked;
    const nIn = this._fnMeta?.inputs || 2;
    const nOut = isInputOnly ? 0 : (this._fnMeta?.outputs || 1);
    
    let html = "<thead><tr>";
    for (let i = 0; i < nIn; i++) html += `<th>In ${i}</th>`;
    if (!isInputOnly) {
      for (let i = 0; i < nOut; i++) html += `<th>Out ${i}</th>`;
    }
    html += "<th></th></tr></thead><tbody>";
    
    this._datasetData.forEach((row, ri) => {
      html += `<tr>`;
      for (let i = 0; i < nIn; i++) {
        html += `<td><input type="number" step="0.1" value="${row.x[i] || 0}" onchange="UI._updateDsVal(${ri}, 'x', ${i}, this.value)"></td>`;
      }
      if (!isInputOnly) {
        for (let i = 0; i < nOut; i++) {
          html += `<td><input type="number" step="0.1" value="${row.y[i] || 0}" onchange="UI._updateDsVal(${ri}, 'y', ${i}, this.value)"></td>`;
        }
      }
      html += `<td><button class="btn danger" style="padding:2px 5px" onclick="UI._removeDsRow(${ri})">×</button></td></tr>`;
    });
    table.innerHTML = html + "</tbody>";
  }

  _addDsRow() {
    const nIn = this._fnMeta?.inputs || 2;
    const nOut = this._fnMeta?.outputs || 1;
    this._datasetData.push({
      x: Array(nIn).fill(0),
      y: Array(nOut).fill(0)
    });
    this._renderDsTable();
  }

  _updateDsVal(ri, type, ci, val) {
    this._datasetData[ri][type][ci] = parseFloat(val) || 0;
  }

  _removeDsRow(ri) {
    this._datasetData.splice(ri, 1);
    this._renderDsTable();
  }

  _showSyntheticModal() {
    const modal = document.getElementById("dsSyntheticModal");
    const container = document.getElementById("dsSyntheticInputs");
    const nIn = this._fnMeta?.inputs || 2;
    
    modal.style.display = "block";
    container.innerHTML = "";
    
    for (let i = 0; i < nIn; i++) {
      const div = document.createElement("div");
      div.style = "border: 1px solid var(--border); padding: 6px; border-radius: 4px; background: var(--surf2);";
      div.innerHTML = `
        <div style="font-size:10px; margin-bottom:4px; font-weight:bold;">Input ${i}</div>
        <div style="display:flex; gap:5px; align-items:center;">
          <input type="number" id="dsSynMin${i}" value="0" step="0.1" style="width:60px; font-size:10px;" placeholder="Min">
          <input type="number" id="dsSynMax${i}" value="1" step="0.1" style="width:60px; font-size:10px;" placeholder="Max">
          <input type="number" id="dsSynStep${i}" value="0.5" step="0.1" style="width:60px; font-size:10px;" placeholder="Step">
        </div>
      `;
      container.appendChild(div);
    }
  }

  _generateSynthetic() {
    const nIn = this._fnMeta?.inputs || 2;
    const nOut = this._fnMeta?.outputs || 1;
    const ranges = [];
    
    for (let i = 0; i < nIn; i++) {
      const min = parseFloat(document.getElementById(`dsSynMin${i}`).value) || 0;
      const max = parseFloat(document.getElementById(`dsSynMax${i}`).value) || 1;
      const step = parseFloat(document.getElementById(`dsSynStep${i}`).value) || 1;
      
      const points = [];
      for (let v = min; v <= max + 0.0001; v += step) points.push(v);
      ranges.push(points);
    }
    
    // Cartesian product
    const cartesian = (...args) => args.reduce((a, b) => a.flatMap(d => b.map(e => [d, e].flat())));
    const results = nIn === 1 ? ranges[0].map(v => [v]) : cartesian(...ranges);
    
    results.forEach(x => {
      this._datasetData.push({
        x: Array.isArray(x) ? x : [x],
        y: Array(nOut).fill(0)
      });
    });
    
    document.getElementById("dsSyntheticModal").style.display = "none";
    this._renderDsTable();
  }

  _importCsv() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv";
    input.onchange = (e) => {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target.result;
        const rows = text.split("\n").map(r => r.split(","));
        const nIn = this._fnMeta?.inputs || 2;
        const nOut = this._fnMeta?.outputs || 1;
        
        rows.forEach(r => {
          if (r.length >= nIn) {
            const x = r.slice(0, nIn).map(v => parseFloat(v) || 0);
            const y = r.length >= nIn + nOut ? r.slice(nIn, nIn + nOut).map(v => parseFloat(v) || 0) : Array(nOut).fill(0);
            this._datasetData.push({ x, y });
          }
        });
        this._renderDsTable();
      };
      reader.readAsText(file);
    };
    input.click();
  }

  _initImageEditor() {
    this._dsZoom = 8;
    this._dsWidth = 8;
    this._dsHeight = 8;
    this._dsPixels = new Float32Array(8 * 8).fill(0);
    this._dsPainting = false;

    const canvas = document.getElementById("dsPixelCanvas");
    if (!canvas) return;

    canvas.addEventListener("mousedown", () => this._dsPainting = true);
    window.addEventListener("mouseup", () => this._dsPainting = false);
    canvas.addEventListener("mousemove", (e) => {
      if (!this._dsPainting) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / this._dsZoom);
      const y = Math.floor((e.clientY - rect.top) / this._dsZoom);
      if (x >= 0 && x < this._dsWidth && y >= 0 && y < this._dsHeight) {
        const val = document.getElementById("dsColorPicker").value === "#ffffff" ? 1.0 : 0.0;
        this._dsPixels[y * this._dsWidth + x] = val;
        this._renderPixelGrid();
      }
    });

    document.getElementById("dsImgZoomIn")?.addEventListener("click", () => {
      this._dsZoom = Math.min(32, this._dsZoom * 2);
      this._renderPixelGrid();
    });
    document.getElementById("dsImgZoomOut")?.addEventListener("click", () => {
      this._dsZoom = Math.max(2, this._dsZoom / 2);
      this._renderPixelGrid();
    });
  }

  _renderPixelGrid() {
    const canvas = document.getElementById("dsPixelCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    canvas.width = this._dsWidth * this._dsZoom;
    canvas.height = this._dsHeight * this._dsZoom;
    
    document.getElementById("dsImgZoomVal").textContent = this._dsZoom + "x";

    for (let y = 0; y < this._dsHeight; y++) {
      for (let x = 0; x < this._dsWidth; x++) {
        const val = this._dsPixels[y * this._dsWidth + x];
        ctx.fillStyle = `rgb(${val * 255}, ${val * 255}, ${val * 255})`;
        ctx.fillRect(x * this._dsZoom, y * this._dsZoom, this._dsZoom, this._dsZoom);
        
        // Grid lines
        if (this._dsZoom >= 4) {
          ctx.strokeStyle = "#30363d";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x * this._dsZoom, y * this._dsZoom, this._dsZoom, this._dsZoom);
        }
      }
    }
    
    this._renderDsHistogram();
  }

  _renderDsHistogram() {
    const canvas = document.getElementById("dsImgHist");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width = canvas.clientWidth;
    const H = canvas.height = canvas.clientHeight;
    
    ctx.fillStyle = "#010409";
    ctx.fillRect(0, 0, W, H);
    
    const bins = new Array(10).fill(0);
    this._dsPixels.forEach(v => {
      const b = Math.min(9, Math.floor(v * 10));
      bins[b]++;
    });
    
    const max = Math.max(...bins, 1);
    ctx.fillStyle = "var(--accent)";
    bins.forEach((count, i) => {
      const bh = (count / max) * H;
      ctx.fillRect(i * (W/10), H - bh, (W/10) - 2, bh);
    });
  }

  _saveDataset() {
    const data = {
      name: document.getElementById("dsName").value,
      ds_type: document.getElementById("dsType").value,
      is_input_only: document.getElementById("dsInputOnly").checked,
      num_inputs: this._fnMeta?.inputs || 2,
      num_outputs: document.getElementById("dsInputOnly").checked ? null : (this._fnMeta?.outputs || 1),
      data: this._datasetData
    };
    
    if (this._selectedDatasetId) {
      this._emit("updateDataset", { id: this._selectedDatasetId, data });
    } else {
      this._emit("createDataset", data);
    }
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
