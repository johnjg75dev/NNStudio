/**
 * static/js/renderer.js
 * NetworkRenderer — draws the network graph on a <canvas>.
 * Completely decoupled from training logic; receives plain data objects.
 *
 * Also contains ArchDiagramRenderer for non-trainable arch diagrams
 * and LossChartRenderer for the loss curve strip.
 */

// ═══════════════════════════════════════════════════════════════════════
// Colour helpers
// ═══════════════════════════════════════════════════════════════════════
function weightColor(w) {
  const a = Math.min(1, Math.abs(w) / 3);
  return w > 0
    ? `rgb(${Math.round(30+a*20)},${Math.round(80+a*130)},${Math.round(30+a*20)})`
    : `rgb(${Math.round(100+a*148)},20,20)`;
}
function activationColor(v, alpha) {
  const t = Math.min(1, Math.max(0, v));
  const r = Math.round(20  + t * 200);
  const g = Math.round(80  + t * 150);
  const b = Math.round(200 + t *  55);
  return alpha !== undefined ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

// ═══════════════════════════════════════════════════════════════════════
// Layer type label helper
// ═══════════════════════════════════════════════════════════════════════
function _getLayerTypeLabel(type) {
  const labels = {
    dense: "Dense",
    dropout: "Dropout",
    batchnorm: "BatchNorm",
    conv2d: "Conv2D",
    maxpool2d: "MaxPool",
    flatten: "Flatten",
    lstm: "LSTM",
    simple_rnn: "RNN",
    embedding: "Embed",
    layernorm: "LayerNorm",
    multihead_attention: "Attention",
    positional_encoding: "PosEnc",
  };
  return labels[type] || type;
}

// ═══════════════════════════════════════════════════════════════════════
// NetworkRenderer
// ═══════════════════════════════════════════════════════════════════════
class NetworkRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext("2d");
    this.opts   = {
      showLabels:      true,
      showActivations: true,
      showBias:        false,
      showGradients:   false,
    };
    this._selectedNode = null;
    this._onNodeClick  = null;
    this._zoom = 1.0;
    // Track expanded/collapsed state for layer groups
    // Key: layer index, Value: boolean (true = expanded)
    this._expandedLayers = new Map();
  }

  setOptions(opts) { Object.assign(this.opts, opts); }
  onNodeClick(fn)  { this._onNodeClick = fn; }
  
  getZoom() { return this._zoom; }
  
  setZoom(zoom) {
    this._zoom = Math.max(0.25, Math.min(3.0, zoom));
    this.draw(this._lastSnapshot);
    return this._zoom;
  }
  
  zoomIn() { return this.setZoom(this._zoom + 0.25); }
  zoomOut() { return this.setZoom(this._zoom - 0.25); }
  zoomReset() { return this.setZoom(1.0); }
  
  // Toggle layer group expansion
  toggleLayerExpanded(layerIdx) {
    const current = this._expandedLayers.get(layerIdx) || false;
    this._expandedLayers.set(layerIdx, !current);
    if (this._lastSnapshot) {
      this.draw(this._lastSnapshot);
    }
    return !current;
  }
  
  isLayerExpanded(layerIdx) {
    return this._expandedLayers.get(layerIdx) || false;
  }

  // ── resize canvas to its CSS size ──
  resize() {
    const container = this.canvas.parentElement;
    this.canvas.width  = Math.max(800, container.clientWidth * this._zoom);
    this.canvas.height = Math.max(400, container.clientHeight * this._zoom);
  }

  // ── Check if layer should be shown as grouped ──
  _isGroupedLayer(layerInfo) {
    if (!layerInfo) return false;
    const type = (layerInfo.type || 'dense').toLowerCase();
    
    // Non-visual layers that should still be represented
    if (['dropout', 'batchnorm', 'flatten', 'layernorm', 'positional_encoding'].includes(type)) {
      return true;
    }

    // Group Conv2D when they have many channels
    if (type === 'conv2d') {
      const outCh = layerInfo.out_channels || 16;
      return outCh > 4;
    }
    // Group MaxPool layers
    if (type === 'maxpool2d') {
      return true;
    }
    return false;
  }

  // ── Get display size for a layer (respects grouping) ──
  _getLayerDisplaySize(layerInfo, topologyValue, expanded) {
    if (!layerInfo || !this._isGroupedLayer(layerInfo)) {
      return topologyValue;
    }
    
    const type = (layerInfo.type || '').toLowerCase();
    const utilityTypes = ['dropout', 'batchnorm', 'flatten', 'layernorm', 'positional_encoding'];
    
    if (utilityTypes.includes(type)) {
      return expanded ? 1 : 1;
    }

    if (expanded) {
      const outCh = layerInfo.out_channels || topologyValue;
      return Math.min(outCh, 16);  // Show max 16 channels when expanded
    } else {
      return 1;  // Collapsed - show as single block
    }
  }

  // ── Get node radius (larger for collapsed grouped layers) ──
  _getNodeRadiusForLayer(layerInfo, baseRadius, expanded) {
    if (!layerInfo || !this._isGroupedLayer(layerInfo)) {
      return baseRadius;
    }
    
    const type = (layerInfo.type || '').toLowerCase();
    const utilityTypes = ['dropout', 'batchnorm', 'flatten', 'layernorm', 'positional_encoding'];
    
    if (utilityTypes.includes(type)) {
      return baseRadius * 0.8; // Utility layers are smaller
    }

    if (expanded) {
      return baseRadius;
    } else {
      // Collapsed grouped layer - make it larger
      const outCh = layerInfo.out_channels || 16;
      return Math.max(baseRadius * 2, Math.min(40, baseRadius * Math.sqrt(outCh)));
    }
  }

  // ── main draw entry ──
  draw(snapshot) {
    this._lastSnapshot = snapshot;
    this.resize();
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (!snapshot || !snapshot.built || !snapshot.topology || snapshot.topology.length === 0) {
      ctx.fillStyle = "#6e7681";
      ctx.font = "14px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click 'Build' to create a network", this.canvas.width / 2, this.canvas.height / 2);
      return;
    }

    const layers  = snapshot.layers || [];
    const pts     = this._nodePositions(snapshot.topology, layers);
    const nodeR   = this._nodeRadius(snapshot.topology);
    const acts    = snapshot.activations || [];
    const fn      = snapshot.func || {};

    this._drawEdges(pts, layers, nodeR);
    if (this.opts.showBias)      this._drawBias(pts, layers, nodeR);
    if (this.opts.showGradients) this._drawGradients(pts, layers, nodeR);
    this._drawNodes(pts, acts, layers, nodeR, fn);
  }

  // ── node x/y positions ──
  _nodePositions(topology, layers) {
    if (!topology || topology.length === 0) return [];
    const W = this.canvas.width, H = this.canvas.height;
    const padX = 80, padY = 48;
    
    const positions = [];
    
    // The number of columns is topology.length
    for (let l = 0; l < topology.length; l++) {
      // Each column l corresponds to layer info in layers[l-1] (except l=0 which is input)
      const layerInfo = l === 0 ? { type: 'input' } : layers[l-1];
      const expanded = this.isLayerExpanded(l);
      const n = this._getLayerDisplaySize(layerInfo, topology[l], expanded);
      
      positions.push(
        Array.from({ length: n }, (_, i) => ({
          x: padX + (l / Math.max(1, topology.length - 1)) * (W - 2 * padX),
          y: n === 1 ? H / 2 : padY + (i / (n - 1)) * (H - 2 * padY),
          layer: l, 
          idx: i,
          isGrouped: this._isGroupedLayer(layerInfo),
          isExpanded: expanded,
          layerInfo: layerInfo,
        }))
      );
    }
    
    return positions;
  }

  _nodeRadius(topology) {
    if (!topology || topology.length === 0) return 10;
    const maxN = Math.max(...topology);
    return Math.max(10, Math.min(24, (this.canvas.height - 92) / maxN * 0.38));
  }

  _drawEdges(pts, layers, nodeR) {
    const { ctx } = this;
    
    // pts corresponds to [Input, L1, L2, ..., Output]
    // layers corresponds to [L1, L2, ..., Output]
    
    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const frPts = pts[l];
      const toPts = pts[l + 1];
      if (!frPts || !toPts) continue;

      const layerType = (layer.type || '').toLowerCase();
      
      // If it's a weighted layer, draw edges for each weight
      if (layer.W && Array.isArray(layer.W) && layer.W.length > 0) {
        layer.W.forEach((row, i) => {
          row.forEach((w, j) => {
            const fr = frPts[j], to = toPts[i];
            if (!fr || !to) return;
            this._drawSingleEdge(ctx, fr, to, weightColor(w), Math.min(6, 0.4 + Math.abs(w) * 1.3), nodeR);
          });
        });
      } else {
        // Non-weighted layer (Dropout, BatchNorm, etc.)
        // Draw symbolic connections (e.g., identity-like)
        const nFr = frPts.length;
        const nTo = toPts.length;
        
        if (nFr === 1 && nTo === 1) {
          this._drawSingleEdge(ctx, frPts[0], toPts[0], "#58a6ff22", 1, nodeR);
        } else {
          // Connect first, middle, last to avoid too many lines
          const indices = [0, Math.floor(nFr/2), nFr-1].filter((v, i, a) => a.indexOf(v) === i);
          indices.forEach(idx => {
            if (frPts[idx] && toPts[idx % nTo]) {
              this._drawSingleEdge(ctx, frPts[idx], toPts[idx % nTo], "#58a6ff22", 1, nodeR);
            }
          });
        }
      }
    }
  }

  _drawSingleEdge(ctx, fr, to, color, width, nodeR) {
    const dx = to.x - fr.x, dy = to.y - fr.y;
    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
    const ux = dx/dist, uy = dy/dist;
    ctx.beginPath();
    ctx.moveTo(fr.x + ux * nodeR, fr.y + uy * nodeR);
    ctx.lineTo(to.x - ux * nodeR, to.y - uy * nodeR);
    ctx.strokeStyle = color;
    ctx.lineWidth   = width;
    ctx.globalAlpha = 0.4;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  _drawBias(pts, layers, nodeR) {
    const { ctx } = this;
    let ptsIndex = 0;
    layers.forEach((layer, l) => {
      // Skip layers without bias or visual nodes
      if (!layer.b || !Array.isArray(layer.b) || layer.b.length === 0) {
        if (layer.type !== 'dropout' && layer.type !== 'batchnorm' && 
            layer.type !== 'flatten' && layer.type !== 'maxpool2d') {
          ptsIndex++;
        }
        return;
      }
      const to = pts[ptsIndex + 1]?.[0];
      if (!to) {
        ptsIndex++;
        return;
      }
      ctx.beginPath();
      ctx.moveTo(to.x - nodeR * 3, to.y - nodeR * 2.5);
      ctx.lineTo(to.x - nodeR,     to.y);
      ctx.strokeStyle = weightColor(layer.b[0]);
      ctx.lineWidth   = Math.min(2.5, 0.3 + Math.abs(layer.b[0]) * 0.7);
      ctx.globalAlpha = 0.3;
      ctx.setLineDash([2, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ptsIndex++;
    });
  }

  _drawGradients(pts, layers, nodeR) {
    const { ctx } = this;
    layers.forEach((layer, l) => {
      if (!layer.dW) return;
      layer.dW.forEach((row, i) => {
        row.forEach((g, j) => {
          if (Math.abs(g) < 0.001) return;
          const fr = pts[l][j], to = pts[l + 1][i];
          if (!fr || !to) return;
          const dx = to.x - fr.x, dy = to.y - fr.y, dist = Math.sqrt(dx*dx+dy*dy)||1;
          ctx.beginPath();
          ctx.moveTo(fr.x + dx/dist*nodeR, fr.y + dy/dist*nodeR);
          ctx.lineTo(to.x - dx/dist*nodeR, to.y - dy/dist*nodeR);
          ctx.strokeStyle = g > 0 ? "rgba(255,200,50,.5)" : "rgba(150,50,255,.5)";
          ctx.lineWidth   = Math.min(4, Math.abs(g) * 20);
          ctx.stroke();
        });
      });
    });
  }

  _drawNodes(pts, acts, layers, nodeR, fn) {
    const { ctx } = this;

    pts.forEach((layerPts, l) => {
      const layerInfo = l === 0 ? { type: 'input' } : layers[l-1];
      const layerType = (layerInfo?.type || "dense").toLowerCase();
      const isGrouped = this._isGroupedLayer(layerInfo);
      const isExpanded = this.isLayerExpanded(l);
      
      // Use larger radius for collapsed grouped layers
      const actualNodeR = this._getNodeRadiusForLayer(layerInfo, nodeR, isExpanded);

      layerPts.forEach(({ x, y, layer, idx }) => {
        // Get activation safely
        const av  = acts[l]?.[idx] ?? 0;
        const sel = this._selectedNode;
        const isSel = sel && sel.layer === l && sel.idx === idx;

        // glow
        if (this.opts.showActivations && acts[l]) {
          const grd = ctx.createRadialGradient(x, y, 0, x, y, actualNodeR * 2.8);
          grd.addColorStop(0, activationColor(av, 0.3));
          grd.addColorStop(1, "rgba(0,0,0,0)");
          ctx.beginPath(); ctx.arc(x, y, actualNodeR * 2.8, 0, Math.PI * 2);
          ctx.fillStyle = grd; ctx.fill();
        }

        // circle - draw larger circle for collapsed grouped layers
        ctx.beginPath(); 
        ctx.arc(x, y, actualNodeR, 0, Math.PI * 2);
        ctx.fillStyle   = (this.opts.showActivations && acts[l]) ? activationColor(av) : "#2d333b";
        
        // Visual distinction for utility layers
        const utilityTypes = ['dropout', 'batchnorm', 'flatten', 'layernorm', 'positional_encoding'];
        if (utilityTypes.includes(layerType)) {
            ctx.fillStyle = "#2d333b";
            ctx.setLineDash([2, 2]);
        }
        
        ctx.fill();
        ctx.strokeStyle = isSel ? "#fff" : "#58a6ff";
        ctx.lineWidth   = isSel ? 2.5 : 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
        
        // For collapsed grouped layers, draw a distinctive pattern
        if (isGrouped && !isExpanded) {
          // Draw grid pattern to indicate multiple channels
          ctx.strokeStyle = "#58a6ff44";
          ctx.lineWidth = 1;
          const gridSize = Math.min(4, Math.ceil(Math.sqrt(layerInfo?.out_channels || 4)));
          const gridStep = (actualNodeR * 2) / gridSize;
          for (let gi = 0; gi <= gridSize; gi++) {
            // Vertical lines
            ctx.beginPath();
            ctx.moveTo(x - actualNodeR + gi * gridStep, y - actualNodeR);
            ctx.lineTo(x - actualNodeR + gi * gridStep, y + actualNodeR);
            ctx.stroke();
            // Horizontal lines
            ctx.beginPath();
            ctx.moveTo(x - actualNodeR, y - actualNodeR + gi * gridStep);
            ctx.lineTo(x + actualNodeR, y - actualNodeR + gi * gridStep);
            ctx.stroke();
          }
        }

        // label
        if (this.opts.showLabels) {
          ctx.fillStyle    = "#e6edf3";
          ctx.font         = `bold ${Math.max(7, actualNodeR * 0.52)}px monospace`;
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          const isIn  = l === 0;
          const isOut = l === pts.length - 1;
          
          let lbl = "";
          if (isIn && fn.input_labels?.[idx]) {
            lbl = fn.input_labels[idx];
          } else if (isOut && fn.output_labels?.[idx]) {
            lbl = fn.output_labels[idx];
          } else if (isGrouped && !isExpanded) {
            lbl = `${layerInfo?.out_channels || pts[l].length}ch`;
          } else if (utilityTypes.includes(layerType)) {
            lbl = layerType.charAt(0).toUpperCase();
          } else {
            lbl = av.toFixed(1);
          }
          
          ctx.fillText(lbl, x, y);
        }

        // layer header with type badge and expand/collapse
        if (idx === 0) {
          ctx.font         = "bold 10px system-ui";
          ctx.textAlign    = "center";
          ctx.textBaseline = "bottom";

          // Type badge colors
          const typeColors = {
            dense: "#58a6ff",
            dropout: "#d29922",
            batchnorm: "#3fb950",
            conv2d: "#f0883e",
            maxpool2d: "#f0883e",
            flatten: "#bc8cff",
            lstm: "#39d353",
            simple_rnn: "#39d353",
            embedding: "#a371f7",
            layernorm: "#3fb950",
            multihead_attention: "#ff7b72",
            positional_encoding: "#a371f7",
          };
          const typeColor = typeColors[layerType] || "#58a6ff";

          // Type badge
          const badgeText = _getLayerTypeLabel(layerType);
          const badgeWidth = ctx.measureText(badgeText).width + 12;
          const badgeX = x - badgeWidth / 2;
          const badgeY = y - actualNodeR - 20;

          ctx.fillStyle = typeColor + "33";
          ctx.strokeStyle = typeColor;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(badgeX, badgeY, badgeWidth, 16, 3);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = typeColor;
          ctx.font = "bold 9px system-ui";
          ctx.textBaseline = "middle";
          ctx.fillText(badgeText, x, badgeY + 8);

          // Expand/collapse indicator for grouped layers
          if (isGrouped) {
            const expandY = y - actualNodeR - 32;
            ctx.fillStyle = "#6e7681";
            ctx.font = "16px system-ui";
            ctx.textBaseline = "middle";
            ctx.fillText(isExpanded ? "▼" : "▶", x, expandY);
            
            // Show layer details
            let layerDetails = '';
            if (layerType === 'conv2d') {
              const k = layerInfo?.kernel_size || 3;
              const ch = layerInfo?.out_channels || pts[l].length;
              layerDetails = `${k}×${k}, ${ch}ch`;
            } else if (layerType === 'maxpool2d') {
              const p = layerInfo?.pool_size || 2;
              layerDetails = `${p}×${p} pool`;
            }
            if (layerDetails) {
              ctx.font = "9px system-ui";
              ctx.fillStyle = "#8b949e";
              ctx.fillText(layerDetails, x, expandY - 12);
            }
          } else {
            // Size label for non-grouped layers
            ctx.fillStyle    = "#6e7681";
            ctx.font         = "10px system-ui";
            ctx.textBaseline = "bottom";
            const sizeLabel = `×${pts[l].length}`;
            ctx.fillText(sizeLabel, x, y - actualNodeR - 4);
          }
        }
      });
    });
  }

  // ── hit-test for clicks/hover ──
  nodeAt(mx, my, topology, layers) {
    if (!topology) return null;
    const pts   = this._nodePositions(topology, layers || []);
    const baseNodeR = this._nodeRadius(topology);
    
    for (let l = 0; l < pts.length; l++) {
      const layerPts = pts[l];
      const layerInfo = l === 0 ? { type: 'input' } : layers[l-1];
      const isGrouped = this._isGroupedLayer(layerInfo);
      const isExpanded = this.isLayerExpanded(l);
      const nodeR = this._getNodeRadiusForLayer(layerInfo, baseNodeR, isExpanded);
      
      for (const p of layerPts) {
        if ((mx - p.x) ** 2 + (my - p.y) ** 2 < (nodeR * 1.9) ** 2)
          return { layer: p.layer, idx: p.idx, isGrouped: p.isGrouped, layerInfo: p.layerInfo };
      }
    }
    return null;
  }

  selectNode(node) { this._selectedNode = node; }
}

// ═══════════════════════════════════════════════════════════════════════
// LossChartRenderer
// ═══════════════════════════════════════════════════════════════════════
class LossChartRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext("2d");
  }

  draw(history) {
    const { ctx } = this;
    const W = this.canvas.clientWidth || 300;
    const H = 55;
    this.canvas.width  = W;
    this.canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#080c10";
    ctx.fillRect(0, 0, W, H);

    if (!history || history.length < 2) return;

    const mx = Math.max(...history, 0.001);
    for (let i = 0; i <= 3; i++) {
      ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
      const y = H - (i / 3) * H;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    ctx.beginPath();
    history.forEach((v, i) => {
      const x = (i / (history.length - 1)) * W;
      const y = H - (v / mx) * H * 0.9 - H * 0.05;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.lineTo(W, H); ctx.lineTo(0, H);
    ctx.fillStyle = "rgba(88,166,255,.1)"; ctx.fill();

    const last = history[history.length - 1];
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText("loss: " + last.toFixed(4), W - 3, 3);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// ArchDiagramRenderer — draws educational diagrams for non-trainable archs
// ═══════════════════════════════════════════════════════════════════════
class ArchDiagramRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext("2d");
  }

  draw(archKey) {
    const c = this.canvas;
    c.width  = c.clientWidth  || 800;
    c.height = c.clientHeight || 400;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, c.width, c.height);
    const W = c.width, H = c.height, cx = W / 2, cy = H / 2;

    const drawFn = {
      cnn:         () => this._drawCNN(W, H, cx, cy),
      transformer: () => this._drawTransformer(W, H, cx, cy),
      vit:         () => this._drawViT(W, H, cx, cy),
      vae:         () => this._drawVAE(W, H, cx, cy),
      diffusion:   () => this._drawDiffusion(W, H, cx, cy),
      gan:         () => this._drawGAN(W, H, cx, cy),
      rnn:         () => this._drawRNN(W, H, cx, cy),
    }[archKey];

    if (drawFn) drawFn();
    else this._drawGeneric(W, H, cx, cy, archKey);
  }

  // ── shared draw helpers ──
  _box(x, y, w, h, label, sub, color) {
    const { ctx } = this;
    ctx.fillStyle   = color + "33";
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle    = color;
    ctx.font         = "bold 11px system-ui";
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x + w/2, y + h/2 + (sub ? -5 : 0));
    if (sub) {
      ctx.fillStyle = "#8b949e"; ctx.font = "9px system-ui";
      ctx.fillText(sub, x + w/2, y + h/2 + 7);
    }
  }

  _arrow(x1, y1, x2, y2, c = "#30363d") {
    const { ctx } = this;
    ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.fillStyle = c;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8*Math.cos(a-.3), y2 - 8*Math.sin(a-.3));
    ctx.lineTo(x2 - 8*Math.cos(a+.3), y2 - 8*Math.sin(a+.3));
    ctx.closePath(); ctx.fill();
  }

  _note(t, x, y) {
    const { ctx } = this;
    ctx.fillStyle = "#8b949e"; ctx.font = "11px system-ui";
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText(t, x, y);
  }

  _drawCNN(W, H, cx, cy) {
    const cols = ["#58a6ff","#3fb950","#3fb950","#f0883e","#f0883e","#bc8cff","#f85149"];
    const blks = [
      {l:"Input",s:"H×W×3",w:55},{l:"Conv2D",s:"32 filters",w:65},{l:"MaxPool",s:"÷2",w:55},
      {l:"Conv2D",s:"64 filters",w:65},{l:"MaxPool",s:"÷2",w:55},{l:"Flatten\n+Dense",s:"512",w:65},{l:"Softmax",s:"N classes",w:65}
    ];
    const tw = blks.reduce((s,b)=>s+b.w+18,0)-18;
    let x = cx - tw/2;
    blks.forEach((b,i) => {
      const bh = 55 + Math.sin(i*.5)*12, by = cy - bh/2;
      this._box(x, by, b.w, bh, b.l, b.s, cols[i]);
      if (i < blks.length-1) this._arrow(x+b.w, cy, x+b.w+18, cy);
      x += b.w + 18;
    });
    this._note("CNN — sliding convolutional kernels detect local features", cx, H-20);
  }

  _drawTransformer(W, H, cx, cy) {
    const lyrs = [
      {y:H-72, l:"Input Tokens",         s:"word IDs",             c:"#58a6ff",w:200,h:34},
      {y:H-128,l:"Embed + Pos. Encoding", s:"d_model dims",         c:"#58a6ff",w:250,h:34},
      {y:H-200,l:"Multi-Head Attention",  s:"Q·Kᵀ/√d→Softmax→·V",  c:"#3fb950",w:260,h:40},
      {y:H-256,l:"Add & LayerNorm",       s:"residual",             c:"#d29922",w:240,h:30},
      {y:H-308,l:"Feed-Forward Network",  s:"Linear→ReLU→Linear",   c:"#bc8cff",w:240,h:38},
      {y:H-354,l:"Add & LayerNorm",       s:"residual",             c:"#d29922",w:240,h:30},
      {y:36,   l:"Output / Task Head",    s:"classify or next-token",c:"#f85149",w:200,h:34},
    ];
    lyrs.forEach(l => this._box(cx-l.w/2, l.y, l.w, l.h, l.l, l.s, l.c));
    for (let i=0; i<lyrs.length-1; i++) this._arrow(cx, lyrs[i].y, cx, lyrs[i+1].y+lyrs[i+1].h);
    const { ctx } = this;
    ctx.strokeStyle="#30363d"; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.strokeRect(cx-140, H-368, 285, 188);
    ctx.setLineDash([]);
    ctx.fillStyle="#6e7681"; ctx.font="10px system-ui"; ctx.textAlign="left";
    ctx.fillText("× N layers", cx+148, H-278);
    this._note("Transformer — Attention Is All You Need (Vaswani et al. 2017)", cx, H-12);
  }

  _drawViT(W, H, cx, cy) {
    const imgS=110, imgX=30, imgY=cy-imgS/2;
    const { ctx } = this;
    ctx.fillStyle="#21262d"; ctx.strokeStyle="#30363d"; ctx.lineWidth=1;
    ctx.fillRect(imgX,imgY,imgS,imgS); ctx.strokeRect(imgX,imgY,imgS,imgS);
    const ps=imgS/4;
    for(let r=0;r<4;r++) for(let c=0;c<4;c++) {
      ctx.fillStyle=`hsl(${(r*4+c)*22},40%,${20+r*5}%)`;
      ctx.fillRect(imgX+c*ps+1, imgY+r*ps+1, ps-2, ps-2);
    }
    this._note("Image → 4×4 Patches", imgX+imgS/2, imgY+imgS+14);
    this._arrow(imgX+imgS+8, cy, imgX+imgS+54, cy);
    this._box(imgX+imgS+56, cy-55, 80, 110, "Patch\nEmbed", "Linear+CLS", "#3fb950");
    this._arrow(imgX+imgS+140, cy, imgX+imgS+186, cy);
    this._box(imgX+imgS+188, cy-60, 110, 120, "Transformer\nEncoder", "×L Self-Attn\n+FFN", "#bc8cff");
    this._arrow(imgX+imgS+302, cy, imgX+imgS+348, cy);
    this._box(imgX+imgS+350, cy-30, 90, 60, "MLP Head", "[CLS]→classes", "#f85149");
    this._note("ViT — An Image is Worth 16×16 Words (Dosovitskiy et al. 2020)", cx, H-20);
  }

  _drawVAE(W, H, cx, cy) {
    const blks=[
      {l:"Input x",s:"data",c:"#58a6ff",w:70},{l:"Encoder",s:"q(z|x)",c:"#3fb950",w:80},
      {l:"μ, σ²",s:"latent",c:"#d29922",w:75},{l:"z=μ+σε",s:"reparameterise",c:"#f0883e",w:85},
      {l:"Decoder",s:"p(x|z)",c:"#bc8cff",w:80},{l:"Output x̂",s:"reconstruction",c:"#f85149",w:80}
    ];
    const tw=blks.reduce((s,b)=>s+b.w+20,0)-20; let x=cx-tw/2;
    blks.forEach((b,i)=>{
      this._box(x, cy-28, b.w, 56, b.l, b.s, b.c);
      if(i<blks.length-1) this._arrow(x+b.w, cy, x+b.w+20, cy);
      x+=b.w+20;
    });
    this._note("Loss = Reconstruction Error + KL(q(z|x) || p(z))  ·  Kingma & Welling 2013", cx, H-20);
  }

  _drawDiffusion(W, H, cx, cy) {
    const { ctx } = this;
    ctx.fillStyle="#8b949e"; ctx.font="bold 11px system-ui"; ctx.textAlign="left"; ctx.textBaseline="alphabetic";
    ctx.fillText("Forward process (add noise):", 30, cy-78);
    const fc=["#3fb950","#58a6ff","#d29922","#f0883e","#f85149"];
    for(let i=0;i<5;i++){
      const x=30+i*90, r=22-i*2;
      ctx.fillStyle=fc[i]; ctx.beginPath(); ctx.arc(x+r,cy-44,r,0,Math.PI*2); ctx.fill();
      ctx.fillStyle="#8b949e"; ctx.font="9px system-ui"; ctx.textAlign="center";
      ctx.fillText(i===0?"x₀ (data)":i===4?"xT (noise)":"x"+i, x+r, cy-14);
      if(i<4) this._arrow(x+r*2+4, cy-44, x+r*2+90-4*(i+1), cy-44);
    }
    ctx.fillStyle="#8b949e"; ctx.font="10px system-ui"; ctx.textAlign="center";
    ctx.fillText("+ Gaussian noise ε ~ N(0, β·I) at each step", cx, cy+2);
    ctx.font="bold 11px system-ui"; ctx.textAlign="left";
    ctx.fillText("Reverse process (U-Net denoises):", 30, cy+46);
    const rc=["#f85149","#f0883e","#d29922","#58a6ff","#3fb950"];
    for(let i=0;i<5;i++){
      const x=30+i*90, r=12+i*2;
      ctx.fillStyle=rc[i]+"44"; ctx.strokeStyle=rc[i]; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(x+24,cy+72,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.fillStyle="#8b949e"; ctx.font="9px system-ui"; ctx.textAlign="center";
      ctx.fillText(i===0?"xT":i===4?"x₀":"x"+(4-i), x+24, cy+100);
      if(i<4) this._arrow(x+24+r+4, cy+72, x+24+90-r-4+i, cy+72, "#bc8cff");
    }
    this._box(W-200, cy-14, 180, 88, "U-Net  ε_θ(xₜ,t)", "Predicts noise ε\ngiven noisy input + timestep", "#bc8cff");
    this._note("Diffusion — Ho et al. DDPM 2020  ·  Stable Diffusion adds VAE latents + CLIP text conditioning", cx, H-12);
  }

  _drawGAN(W, H, cx, cy) {
    this._box(30, cy-50, 110, 100, "Generator\nG(z)", "noise → fake data", "#3fb950");
    this._arrow(144, cy, 210, cy);
    this._box(212, cy-58, 120, 116, "Discriminator\nD(x)", "Real or Fake?", "#f85149");
    this._arrow(334, cy-22, 400, cy-22); this._box(402, cy-40, 80, 38, "Real → 1", "P(real)=1", "#58a6ff");
    this._arrow(334, cy+22, 400, cy+22); this._box(402, cy+5,  80, 38, "Fake → 0", "P(fake)=0", "#d29922");
    this._arrow(30, cy-78, 87, cy-50);   this._box(30, cy-104, 55, 28, "Noise z", "random", "#bc8cff");
    this._arrow(152, cy-80, 212, cy-36); this._box(100, cy-104, 85, 28, "Real data", "training set", "#58a6ff");
    this._note("G loss: log(1−D(G(z)))  ·  D loss: log D(x)+log(1−D(G(z)))  ·  Goodfellow et al. 2014", cx, H-16);
  }

  _drawRNN(W, H, cx, cy) {
    const cells=5, cw=80, ch=58, gap=36;
    const sx = cx - (cells*(cw+gap)-gap)/2;
    const { ctx } = this;
    for(let t=0;t<cells;t++){
      const x=sx+t*(cw+gap);
      this._box(x, cy-ch/2, cw, ch, t===2?"LSTM Cell":"RNN Cell", "h_"+t, "#bc8cff");
      this._arrow(x+cw/2, cy+ch/2+2, x+cw/2, cy+ch/2+36);
      ctx.fillStyle="#8b949e"; ctx.font="9px system-ui"; ctx.textAlign="center";
      ctx.fillText("x_"+t, x+cw/2, cy+ch/2+50);
      this._arrow(x+cw/2, cy-ch/2-2, x+cw/2, cy-ch/2-34);
      ctx.fillText("y_"+t, x+cw/2, cy-ch/2-46);
      if(t<cells-1) this._arrow(x+cw, cy, x+cw+gap, cy, "#3fb950");
    }
    const gx=cx-165, gy=H-118;
    ctx.fillStyle="#8b949e"; ctx.font="bold 10px system-ui"; ctx.textAlign="left";
    ctx.fillText("LSTM Gates:", gx, gy);
    [["Forget","f=σ(Wf·[h,x]+b)","#f85149"],["Input","i=σ(Wi·[h,x]+b)","#3fb950"],["Output","o=σ(Wo·[h,x]+b)","#58a6ff"]]
      .forEach(([n,f,c],i) => this._box(gx+i*118, gy+5, 110, 34, n, f, c));
    this._note("RNN / LSTM — sequential memory via hidden state  ·  Hochreiter & Schmidhuber 1997", cx, H-10);
  }

  _drawGeneric(W, H, cx, cy, key) {
    const { ctx } = this;
    ctx.fillStyle="#8b949e"; ctx.font="14px system-ui";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(`${key} — see the Learn tab for details`, cx, cy);
  }
}
