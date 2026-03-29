/**
 * static/js/app.js
 * App — top-level orchestrator.
 * Owns all three major objects (UIController, TrainingController, Renderers)
 * and wires their events together. Contains zero DOM/canvas code itself.
 */
class App {
  constructor(registry) {
    this._registry   = registry;       // injected from Jinja template
    this._snapshot   = null;           // last server snapshot
    this._fnMeta     = null;           // current function metadata
    this._archKey    = "mlp";

    // ── sub-systems ──
    this._ui       = new UIController(registry);
    this._trainer  = new TrainingController();
    this._netRend  = new NetworkRenderer(document.getElementById("netCanvas"));
    this._archRend = new ArchDiagramRenderer(document.getElementById("netCanvas"));
    this._lossRend = new LossChartRenderer(document.getElementById("lossChart"));
  }

  // ════════════════════════════════════════════════════════
  // BOOT
  // ════════════════════════════════════════════════════════
  async init() {
    this._ui.init();
    this._bindUIEvents();
    this._bindTrainerEvents();
    this._bindResize();

    // Apply first preset automatically
    const firstPreset = this._registry.presets?.[0];
    if (firstPreset) {
      this._ui.applyPreset(firstPreset);
      await this._build(this._ui.getConfig());
    } else {
      this._drawEmpty();
    }
  }

  // ════════════════════════════════════════════════════════
  // UI EVENT WIRING
  // ════════════════════════════════════════════════════════
  _bindUIEvents() {
    const ui = this._ui;

    ui.on("applyPreset",  async p  => { ui.applyPreset(p); await this._build(ui.getConfig()); });
    ui.on("build",        async cfg => await this._build(cfg));
    ui.on("reset",        async ()  => await this._resetWeights());
    ui.on("trainToggle",  ()        => this._toggleTraining());
    ui.on("archChanged",  key       => { this._archKey = key; this._drawCanvas(); });
    ui.on("vizChanged",   opts      => { this._netRend.setOptions(opts); this._drawCanvas(); });
    ui.on("controlChanged",()       => { /* live config preview — no-op for now */ });

    ui.on("canvasHover",  pt  => this._handleCanvasHover(pt));
    ui.on("canvasClick",  pt  => this._handleCanvasClick(pt));
    ui.on("showWeights",  ()  => ui.renderWeightMatrix(this._snapshot?.layers));

    ui.on("runTest",      x   => this._runTest(x));
    ui.on("randomTest",   ()  => {
      const n = this._fnMeta?.inputs ?? 2;
      const x = ui.randomiseTestInputs(n);
      this._runTest(x);
    });
    ui.on("sweep",        ()  => this._sweep());

    ui.on("exportModel",  ()  => this._exportModel());
    ui.on("importModel",  data => this._importModel(data));
    ui.on("requestSummary", () => ui.renderModelSummary(this._snapshot));

    ui.on("savePreset",   async cfg => {
      try {
        await API.savePreset(cfg);
        await this._refreshRegistry();
      } catch (e) {
        alert("Failed to save preset: " + e.message);
      }
    });

    ui.on("deletePreset", async id => {
      try {
        await API.deletePreset(id);
        await this._refreshRegistry();
      } catch (e) {
        alert("Failed to delete preset: " + e.message);
      }
    });

    // Zoom controls
    document.getElementById("zoomInBtn")?.addEventListener("click", () => {
      const zoom = this._netRend.zoomIn();
      this._updateZoomLevel();
    });
    document.getElementById("zoomOutBtn")?.addEventListener("click", () => {
      const zoom = this._netRend.zoomOut();
      this._updateZoomLevel();
    });
    document.getElementById("zoomResetBtn")?.addEventListener("click", () => {
      const zoom = this._netRend.zoomReset();
      this._updateZoomLevel();
    });
  }

  _updateZoomLevel() {
    const zoomEl = document.getElementById("zoomLevel");
    if (zoomEl) {
      zoomEl.textContent = Math.round(this._netRend.getZoom() * 100) + "%";
    }
  }

  // ════════════════════════════════════════════════════════
  // TRAINER EVENT WIRING
  // ════════════════════════════════════════════════════════
  _bindTrainerEvents() {
    this._trainer.addEventListener("step", e => {
      const data = e.detail;
      this._snapshot = { ...this._snapshot, ...data };
      this._ui.updateStats(data);
      this._drawCanvas();
      this._lossRend.draw(data.loss_history);
      this._refreshIOTable(data);
    });

    this._trainer.addEventListener("stopped", () => {
      this._ui.setStatus("Paused", "warn");
      this._ui.setTrainButtonState(false);
    });

    this._trainer.addEventListener("error", e => {
      this._ui.setStatus("Error", "err");
      this._ui.setTrainButtonState(false);
      console.error("Training error:", e.detail.message);
    });
  }

  // ════════════════════════════════════════════════════════
  // RESIZE
  // ════════════════════════════════════════════════════════
  _bindResize() {
    new ResizeObserver(() => this._drawCanvas())
      .observe(document.getElementById("netCanvas"));
  }

  // ════════════════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════════════════
  async _build(config) {
    this._trainer.stop();
    this._ui.setStatus("Building…");
    try {
      console.log("Building network with config:", config);
      const data = await API.buildNetwork(config);
      console.log("buildNetwork returned:", data);
      
      this._archKey = config.arch_key;
      this._fnMeta  = data.func;

      // Fetch full snapshot
      console.log("Fetching snapshot...");
      this._snapshot = await API.getSnapshot();
      console.log("Snapshot received:", this._snapshot);
      
      // Validate snapshot
      if (!this._snapshot) {
        throw new Error("Snapshot is null/undefined");
      }
      if (!this._snapshot.built) {
        throw new Error("Snapshot not built");
      }
      if (!this._snapshot.topology || !Array.isArray(this._snapshot.topology)) {
        throw new Error("Invalid topology: " + JSON.stringify(this._snapshot.topology));
      }
      if (this._snapshot.topology.length === 0) {
        throw new Error("Empty topology");
      }
      if (this._snapshot.topology.some(n => typeof n !== 'number' || n <= 0)) {
        throw new Error("Invalid topology values: " + JSON.stringify(this._snapshot.topology));
      }
      
      this._snapshot.activation = config.activation;
      this._snapshot.optimizer  = config.optimizer;
      this._snapshot.loss       = config.loss;
      this._snapshot.weight_decay = config.weight_decay;

      this._ui.updateStats({
        epoch:       this._snapshot.epoch,
        loss:        0,
        accuracy:    0,
        param_count: data.param_count,
      });

      // Configure trainer speed
      this._trainer.configure({ steps: config.steps, lr: config.lr });

      this._ui.setStatus("Ready");
      this._drawCanvas();
      this._lossRend.draw([]);
      this._ui.renderTestInputsFor(this._fnMeta, this._snapshot?.layers?.[0] ? null : null);

      // Initial IO table
      const evalData = await API.evaluate();
      this._ui.renderIOTable(evalData.samples, this._fnMeta);

    } catch (e) {
      console.error("Build failed:", e);
      this._ui.setStatus("Error: " + e.message, "err");
    }
  }

  // ════════════════════════════════════════════════════════
  // RESET WEIGHTS
  // ════════════════════════════════════════════════════════
  async _resetWeights() {
    this._trainer.stop();
    try {
      await API.resetWeights();
      this._snapshot = await API.getSnapshot();
      this._ui.updateStats({ epoch: 0, loss: 0, accuracy: 0 });
      this._ui.setStatus("Reset");
      this._drawCanvas();
      this._lossRend.draw([]);
    } catch (e) {
      console.error("Reset failed:", e.message);
    }
  }

  // ════════════════════════════════════════════════════════
  // TRAINING TOGGLE
  // ════════════════════════════════════════════════════════
  _toggleTraining() {
    const arch = this._registry.architectures?.find(a => a.key === this._archKey);
    if (!arch?.trainable) {
      alert("This architecture is for visualisation only.\nSwitch to MLP or Autoencoder to train.");
      return;
    }
    if (!this._snapshot?.built) {
      this._build(this._ui.getConfig()).then(() => this._startTraining());
      return;
    }
    if (this._trainer.running) {
      this._trainer.stop();
    } else {
      this._startTraining();
    }
  }

  _startTraining() {
    const cfg = this._ui.getConfig();
    this._trainer.configure({ steps: cfg.steps, lr: cfg.lr });
    this._trainer.start();
    this._ui.setStatus("Training", "active");
    this._ui.setTrainButtonState(true);
  }

  // ════════════════════════════════════════════════════════
  // CANVAS DRAW
  // ════════════════════════════════════════════════════════
  _drawCanvas() {
    const arch = this._registry.architectures?.find(a => a.key === this._archKey);
    if (!arch?.trainable) {
      this._archRend.draw(this._archKey);
    } else {
      this._netRend.setOptions(this._ui.getVizOptions());
      this._netRend.draw(this._snapshot);
    }
    this._updateZoomLevel();
  }

  _drawEmpty() {
    this._netRend.draw(null);
  }

  async _refreshRegistry() {
    try {
      const data = await API.getAllModules();
      this._registry = data;
      this._ui._registry = data; // Sync UI controller registry
      this._ui._initPresets();   // Re-render presets grid
    } catch (e) {
      console.error("Failed to refresh registry:", e.message);
    }
  }

  // ════════════════════════════════════════════════════════
  // CANVAS HOVER / CLICK
  // ════════════════════════════════════════════════════════
  _handleCanvasHover({ x, y }) {
    const tip    = document.getElementById("tip");
    const canvas = document.getElementById("netCanvas");
    const arch   = this._registry.architectures?.find(a => a.key === this._archKey);
    if (!arch?.trainable || !this._snapshot?.topology) {
      tip.style.display = "none";
      return;
    }

    const node = this._netRend.nodeAt(x, y, this._snapshot.topology);
    if (node) {
      const { layer, idx } = node;
      const acts  = this._snapshot.activations || [];
      const fn    = this._snapshot.func || {};
      const av    = acts[layer]?.[idx] ?? 0;
      const topo  = this._snapshot.topology;
      const isIn  = layer === 0;
      const isOut = layer === topo.length - 1;
      const lbl   = isIn  ? (fn.input_labels?.[idx]  ?? `In${idx}`)
                  : isOut ? (fn.output_labels?.[idx] ?? `Out${idx}`)
                          : `H${layer}N${idx}`;
      const layers = this._snapshot.layers || [];
      const bias   = layers[layer - 1]?.b?.[idx];

      tip.innerHTML = `<b>${lbl}</b><br>Activation: ${av.toFixed(4)}`
        + (bias !== undefined ? `<br>Bias: ${bias.toFixed(4)}` : "");
      tip.style.display = "block";

      // Position relative to viewport
      const rect = canvas.getBoundingClientRect();
      tip.style.left = Math.min(rect.left + x + 14, window.innerWidth - 248) + "px";
      tip.style.top  = Math.max(4, rect.top + y - 10) + "px";
      canvas.style.cursor = "pointer";
    } else {
      tip.style.display = "none";
      canvas.style.cursor = "default";
    }
  }

  _handleCanvasClick({ x, y }) {
    const arch = this._registry.architectures?.find(a => a.key === this._archKey);
    if (!arch?.trainable || !this._snapshot?.topology) return;

    const node = this._netRend.nodeAt(x, y, this._snapshot.topology, this._snapshot.layers);
    
    // Check if clicking on a grouped layer's expand/collapse indicator
    if (node?.isGrouped && node.layerInfo) {
      // Check if click is near the expand/collapse area (above the node)
      const pts = this._netRend._nodePositions(this._snapshot.topology, this._snapshot.layers);
      if (pts[node.layer] && pts[node.layer][0]) {
        const nodeY = pts[node.layer][0].y;
        const baseNodeR = this._netRend._nodeRadius(this._snapshot.topology);
        const actualNodeR = this._netRend._getNodeRadiusForLayer(
          node.layerInfo, 
          baseNodeR, 
          this._netRend.isLayerExpanded(node.layer)
        );
        // Expand/collapse indicator is about 45px above node center
        const expandY = nodeY - actualNodeR - 32;
        const clickDist = Math.abs(y - expandY);
        
        if (clickDist < 25) {
          // Clicked on expand/collapse
          const expanded = this._netRend.toggleLayerExpanded(node.layer);
          console.log(`Layer ${node.layer} (${node.layerInfo?.type}): ${expanded ? 'expanded' : 'collapsed'}`);
          this._drawCanvas();
          return;
        }
      }
    }
    
    // Normal node selection
    this._netRend.selectNode(node);
    if (node) this._ui.renderNodeInfo(node, this._snapshot);
    this._drawCanvas();
  }

  // ════════════════════════════════════════════════════════
  // TEST TAB
  // ════════════════════════════════════════════════════════
  async _runTest(x) {
    if (!this._snapshot?.built) { alert("Build a network first."); return; }
    try {
      const data = await API.predict(x);
      const isSeg7 = this._fnMeta?.key === "seg7";
      this._ui.renderTestOutput(data.output, this._fnMeta, isSeg7);

      // Sync activations to canvas
      this._snapshot = { ...this._snapshot, activations: data.activations };
      this._drawCanvas();
    } catch (e) {
      console.error("Predict failed:", e.message);
    }
  }

  async _sweep() {
    if (!this._snapshot?.built) { alert("Build a network first."); return; }
    try {
      const data   = await API.evaluate();
      const isSeg7 = this._fnMeta?.key === "seg7";
      this._ui.renderSweepResults(data.samples, this._fnMeta, isSeg7);
    } catch (e) {
      console.error("Sweep failed:", e.message);
    }
  }

  // ════════════════════════════════════════════════════════
  // IO TABLE REFRESH
  // ════════════════════════════════════════════════════════
  async _refreshIOTable(stepData) {
    // Use predictions from the step response activations on each sample
    // to avoid an extra round-trip — evaluate every N epochs
    if (stepData.epoch % 5 !== 0) return;
    try {
      const data = await API.evaluate();
      this._ui.renderIOTable(data.samples, this._fnMeta);
    } catch (_) { /* non-critical */ }
  }

  // ════════════════════════════════════════════════════════
  // SAVE / LOAD
  // ════════════════════════════════════════════════════════
  async _exportModel() {
    if (!this._snapshot?.built) { alert("Build a network first."); return; }
    try {
      const data = await API.exportModel();
      const name = document.getElementById("modelName")?.value || "model";
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url; a.download = name + ".json"; a.click();
      URL.revokeObjectURL(url);
      this._ui.showSaveInfo(
        `<b style="color:var(--green)">✓ Exported:</b> ${name}.json<br>` +
        `Topology: [${data.topology?.join("→")}] · Params: ${data.param_count ?? "?"}<br>` +
        `Epoch ${data.epoch ?? 0}`
      );
    } catch (e) {
      this._ui.showSaveInfo(`<b style="color:var(--red)">✗ Error:</b> ${e.message}`);
    }
  }

  async _importModel(data) {
    try {
      const result = await API.importModel(data);
      this._snapshot = await API.getSnapshot();
      this._fnMeta   = this._snapshot.func;
      this._archKey  = data.arch_key || "mlp";

      // Sync selects
      _setVal("archSel", this._archKey);
      _setVal("funcSel", data.func_key || "xor");
      document.getElementById("archSel")?.dispatchEvent(new Event("change"));
      document.getElementById("funcSel")?.dispatchEvent(new Event("change"));

      this._ui.updateStats({
        epoch:       result.epoch,
        param_count: result.param_count,
      });
      this._ui.showLoadInfo(
        `<b style="color:var(--green)">✓ Loaded</b><br>` +
        `Topology: [${result.topology?.join("→")}]<br>Epoch: ${result.epoch}`
      );
      this._ui.setStatus("Loaded");
      this._drawCanvas();
    } catch (e) {
      this._ui.showLoadInfo(`<b style="color:var(--red)">✗ Error:</b> ${e.message}`);
    }
  }
}

// ── bootstrap ──
document.addEventListener("DOMContentLoaded", () => {
  // REGISTRY is injected by Jinja into the page as a global
  const app = new App(window.REGISTRY || {});
  app.init().catch(console.error);
});
