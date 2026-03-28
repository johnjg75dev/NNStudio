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
    this._initArchSelect();
    this._initFuncSelect();
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
        if (btn.dataset.pg === "test")     this._renderTestInputs();
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
    const presets = this._registry.presets || [];
    presets.forEach(p => {
      const btn = document.createElement("button");
      btn.className = "pbtn";
      btn.innerHTML = `${p.label}<div class="pa">${p.func_key}</div>`;
      btn.addEventListener("click", () => this._emit("applyPreset", p));
      grid.appendChild(btn);
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
      this._emit("funcChanged", sel.value);
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

    bind("layersSl",  "layersV");
    bind("neuronsSl", "neuronsV");
    bind("lrSl",      "lrV",     v => Math.pow(10, v));
    bind("dropSl",    "dropV",   v => v.toFixed(2));
    bind("wdSl",      "wdV",     v => v.toFixed(3));
    bind("stepsSl",   "stepsV");

    ["actSel","optSel","lossSel"].forEach(id => {
      document.getElementById(id)?.addEventListener("change",
        () => this._emit("controlChanged", this.getConfig()));
    });
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
      return `<div class="trow">
        <label>${lbl}</label>
        <input type="number" id="ti${i}" value="${defVal.toFixed(3)}"
               step="0.01" min="0" max="1">
      </div>`;
    }).join("");
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
        results.map(r => `<div class="seg-digit">${segSVG(r.pred)}</div>`).join("") +
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
      hidden_layers: +_val("layersSl"),
      neurons:       +_val("neuronsSl"),
      activation:    _val("actSel"),
      optimizer:     _val("optSel"),
      loss:          _val("lossSel"),
      lr:            Math.pow(10, +_val("lrSl")),
      dropout:       +_val("dropSl"),
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
    _setVal("layersSl",  p.hidden_layers);
    _setVal("neuronsSl", p.neurons);
    _setVal("actSel",    p.activation);
    _setVal("optSel",    p.optimizer);
    _setVal("lossSel",   p.loss);
    _setVal("lrSl",      Math.log10(p.lr).toFixed(2));
    _setVal("dropSl",    p.dropout  ?? 0);
    _setVal("wdSl",      p.weight_decay ?? 0);

    // Trigger display updates
    ["layersSl","neuronsSl","lrSl","dropSl","wdSl"].forEach(id => {
      document.getElementById(id)?.dispatchEvent(new Event("input"));
    });
    ["archSel","funcSel","actSel","optSel","lossSel"].forEach(id => {
      document.getElementById(id)?.dispatchEvent(new Event("change"));
    });
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
    this._renderTestInputs(fnMeta, firstSample);
  }

  getTestInputs() { return this._getTestInputs(); }
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
