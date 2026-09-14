/**
 * lib/networkGraph.js — layout + painting for the live network canvas.
 *
 * Pure functions/classes over a snapshot; no React, no DOM lookups.
 * The React component owns the canvas element, DPR scaling and pointer events.
 *
 * Snapshot shape (see app/api/session_routes.py::snapshot):
 *   { built, topology: [n0,n1,…], layers: [{W,b,dW,type,…}], activations: [[…],…], func }
 */
import { activationColor, clamp, palette, weightColor, withAlpha, LAYER_TYPE_COLORS } from './colors';
import { UTILITY_TYPES, layerLabel } from './layers';

const MAX_NODES_PER_LAYER = 42; // beyond this a layer collapses into a "band"
const EDGE_BUDGET = 2400; // max edges painted per frame

export class NetworkGraph {
  constructor() {
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.selected = null; // { layer, idx }
    this.hovered = null;
    this.focusMode = false; // dim everything except paths into `selected`
    this.expanded = new Set(); // layer indices forced open
    this.options = {
      showLabels: true,
      showActivations: true,
      showBias: false,
      showGradients: false,
      showDeadWeights: false,
      showImageIO: false,
    };
    this.layout = null;
  }

  setOptions(opts) {
    Object.assign(this.options, opts);
  }

  setZoom(z, anchor) {
    const next = clamp(z, 0.35, 4);
    if (anchor) {
      // keep the point under the cursor stable while zooming
      const k = next / this.zoom;
      this.pan.x = anchor.x - (anchor.x - this.pan.x) * k;
      this.pan.y = anchor.y - (anchor.y - this.pan.y) * k;
    }
    this.zoom = next;
    return this.zoom;
  }

  zoomBy(delta, anchor) {
    return this.setZoom(this.zoom * (delta > 0 ? 1.14 : 1 / 1.14), anchor);
  }

  resetView() {
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
  }

  panBy(dx, dy) {
    this.pan.x += dx;
    this.pan.y += dy;
  }

  toggleExpanded(layerIdx) {
    if (this.expanded.has(layerIdx)) this.expanded.delete(layerIdx);
    else this.expanded.add(layerIdx);
  }

  /** True when the layer collapses to a single glyph. */
  isBanded(entry) {
    if (!entry) return false;
    if (entry.banded) return true;
    const type = (entry.type || 'dense').toLowerCase();
    if (UTILITY_TYPES.includes(type)) return true;
    if (type === 'conv2d') return (entry.out_channels || 0) > 4;
    if (type === 'maxpool2d') return true;
    return false;
  }

  // ─────────────────────────── layout ───────────────────────────
  computeLayout(snapshot, width, height) {
    if (!snapshot || !snapshot.built || !snapshot.topology?.length) {
      this.layout = null;
      return null;
    }
    const top = snapshot.topology;
    const layers = snapshot.layers || [];
    const padX = 78;
    const padTop = 62;
    const padBottom = 34;
    const usableH = Math.max(60, height - padTop - padBottom);

    const maxN = Math.max(...top.map((n) => Math.min(n, MAX_NODES_PER_LAYER)));
    let radius = clamp((usableH / Math.max(2, maxN)) * 0.4, 3.5, 20);

    const cols = top.map((n, l) => {
      const info = l === 0 ? { type: 'input' } : layers[l - 1] || { type: 'dense' };
      const type = (info.type || 'dense').toLowerCase();
      const collapsible = n > MAX_NODES_PER_LAYER || this._typeIsBanded(type, info);
      const forceOpen = this.expanded.has(l);
      const banded = collapsible && !forceOpen;
      const count = banded ? 1 : Math.min(n, MAX_NODES_PER_LAYER);
      return { layer: l, type, info, banded, collapsible, count, total: n };
    });

    const columns = cols.map((col, l) => {
      const x = padX + (l / Math.max(1, top.length - 1)) * Math.max(1, width - padX * 2);
      const r = col.banded ? clamp(radius * 1.5, 12, 26) : radius;
      const nodes = [];
      if (col.count === 1) {
        nodes.push({ x, y: padTop + usableH / 2, layer: l, idx: 0, r, banded: col.banded });
      } else {
        const step = usableH / (col.count - 1 || 1);
        for (let i = 0; i < col.count; i += 1) {
          nodes.push({
            x,
            y: padTop + (col.count === 1 ? usableH / 2 : i * step),
            layer: l,
            idx: i,
            r,
            banded: false,
          });
        }
      }
      return { ...col, x, r, nodes };
    });

    this.layout = { columns, width, height, radius, padTop, usableH };
    return this.layout;
  }

  _typeIsBanded(type, info) {
    if (UTILITY_TYPES.includes(type)) return true;
    if (type === 'conv2d') return (info.out_channels || 0) > 4;
    if (type === 'maxpool2d') return true;
    return false;
  }

  /** Convert a pointer position (CSS px) into a graph-space position. */
  toGraph(px, py) {
    return { x: (px - this.pan.x) / this.zoom, y: (py - this.pan.y) / this.zoom };
  }

  nodeAt(px, py) {
    if (!this.layout) return null;
    const { x, y } = this.toGraph(px, py);
    let best = null;
    let bestD = Infinity;
    for (const col of this.layout.columns) {
      for (const n of col.nodes) {
        const hitR = Math.max(n.r * 1.7, 9);
        const d = (x - n.x) ** 2 + (y - n.y) ** 2;
        if (d < hitR * hitR && d < bestD) {
          bestD = d;
          best = { layer: n.layer, idx: n.idx, banded: n.banded, column: col };
        }
      }
    }
    return best;
  }

  /** Hit-test the expand/collapse affordance drawn above a column header. */
  expandToggleAt(px, py) {
    if (!this.layout) return null;
    const { x, y } = this.toGraph(px, py);
    for (const col of this.layout.columns) {
      if (!col.collapsible) continue;
      const cy = this.layout.padTop - 56;
      if (Math.abs(x - col.x) < 34 && Math.abs(y - cy) < 9) return col.layer;
    }
    return null;
  }

  // ─────────────────────────── paint ───────────────────────────
  draw(ctx, snapshot, width, height, theme = 'dark') {
    const pal = palette(theme);
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    this.paintBackground(ctx, width, height, pal);

    const layout = this.computeLayout(snapshot, width / this.zoom, height / this.zoom);
    if (!layout) {
      ctx.restore();
      return;
    }

    ctx.translate(this.pan.x, this.pan.y);
    ctx.scale(this.zoom, this.zoom);

    const acts = snapshot.activations || [];
    const fn = snapshot.func || {};
    const layers = snapshot.layers || [];

    this.drawEdges(ctx, layout, layers, pal);
    if (this.options.showGradients) this.drawGradients(ctx, layout, layers, pal);
    if (this.options.showBias) this.drawBias(ctx, layout, layers, pal);
    this.drawNodes(ctx, layout, acts, layers, fn, pal);
    this.drawHeaders(ctx, layout, fn, pal);
    ctx.restore();
  }

  paintBackground(ctx, w, h, pal) {
    ctx.fillStyle = pal.canvasBg;
    ctx.fillRect(0, 0, w, h);
    // faint dot grid for depth
    const step = 26;
    ctx.fillStyle = pal.grid;
    for (let x = step; x < w; x += step) {
      for (let y = step; y < h; y += step) {
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  isDimmed(layer, idx) {
    if (!this.focusMode || !this.selected) return false;
    const s = this.selected;
    // keep the target node and the layer feeding it lit
    if (layer === s.layer) return idx !== s.idx;
    return layer !== s.layer - 1;
  }

  drawEdges(ctx, layout, layers, pal) {
    const DEAD = 0.05;
    for (let l = 0; l < layers.length; l += 1) {
      const from = layout.columns[l];
      const to = layout.columns[l + 1];
      if (!from || !to) continue;
      const layer = layers[l];
      const W = layer?.W;

      if (from.banded || to.banded || !Array.isArray(W) || W.length === 0) {
        this.drawRibbon(ctx, from, to, layer, pal);
        continue;
      }

      const rows = W.length;
      const cols = Array.isArray(W[0]) ? W[0].length : 0;
      const total = rows * cols;
      const stride = total > EDGE_BUDGET ? Math.ceil(total / EDGE_BUDGET) : 1;
      const faded = stride > 1;
      let drawn = 0;

      for (let i = 0; i < rows; i += 1) {
        const row = W[i];
        if (!Array.isArray(row)) continue;
        const target = to.nodes[i];
        if (!target) continue;
        for (let j = 0; j < row.length; j += 1) {
          if (stride > 1 && drawn % stride !== 0) {
            drawn += 1;
            continue;
          }
          drawn += 1;
          const w = row[j];
          if (typeof w !== 'number' || Number.isNaN(w)) continue;
          const dead = Math.abs(w) < DEAD;
          if (dead && !this.options.showDeadWeights) continue;
          const source = from.nodes[j];
          if (!source) continue;

          const dim = this.isDimmed(l + 1, i);
          this.strokeEdge(ctx, source, target, dead, w, dim, pal, faded);
        }
      }
    }
  }

  strokeEdge(ctx, a, b, dead, w, dim, pal, faded) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy) || 1;
    const ux = dx / dist;
    const uy = dy / dist;
    ctx.beginPath();
    ctx.moveTo(a.x + ux * a.r, a.y + uy * a.r);
    ctx.lineTo(b.x - ux * b.r, b.y - uy * b.r);
    if (dead) {
      ctx.strokeStyle = pal.text4;
      ctx.lineWidth = 0.6;
      ctx.globalAlpha = dim ? 0.05 : 0.22;
      ctx.setLineDash([2, 3]);
    } else {
      ctx.strokeStyle = weightColor(w, pal);
      ctx.lineWidth = clamp(0.35 + Math.abs(w) * 1.25, 0.35, 5);
      ctx.globalAlpha = dim ? 0.05 : faded ? 0.26 : 0.5;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  /** Flow ribbon between two banded columns (conv/pool/utility/large layers). */
  drawRibbon(ctx, from, to, layer, pal) {
    if (!from || !to) return;
    const a = from.nodes[0];
    const b = to.nodes[0];
    if (!a || !b) return;

    let magnitude = 0.4;
    let positive = 0.5;
    const W = layer?.W;
    if (Array.isArray(W) && W.length) {
      let sum = 0;
      let pos = 0;
      let n = 0;
      const stride = Math.max(1, Math.floor(W.length / 24));
      for (let i = 0; i < W.length; i += stride) {
        const row = W[i];
        if (!Array.isArray(row)) continue;
        const s2 = Math.max(1, Math.floor(row.length / 24));
        for (let j = 0; j < row.length; j += s2) {
          const v = row[j];
          if (typeof v !== 'number') continue;
          sum += Math.abs(v);
          pos += v >= 0 ? 1 : 0;
          n += 1;
        }
      }
      if (n) {
        magnitude = clamp(sum / n / 2, 0.08, 1);
        positive = pos / n;
      }
    }

    const dim = this.focusMode && this.selected && to.layer !== this.selected.layer;
    ctx.save();
    ctx.globalAlpha = dim ? 0.08 : 0.55;
    const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
    const c1 = positive >= 0.5 ? pal.pos : pal.neg;
    const c2 = positive >= 0.5 ? pal.neg : pal.pos;
    grad.addColorStop(0, withAlpha(c1, 0.05));
    grad.addColorStop(0.5, withAlpha(positive >= 0.5 ? c1 : c2, 0.16 + magnitude * 0.4));
    grad.addColorStop(1, withAlpha(c2, 0.05));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y - a.r * 0.7);
    ctx.lineTo(b.x, b.y - b.r * 0.7);
    ctx.lineTo(b.x, b.y + b.r * 0.7);
    ctx.lineTo(a.x, a.y + a.r * 0.7);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = withAlpha(pal.accent, 0.18);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  drawBias(ctx, layout, layers, pal) {
    layers.forEach((layer, l) => {
      const b = layer?.b;
      if (!Array.isArray(b) || b.length === 0) return;
      const col = layout.columns[l + 1];
      if (!col) return;
      col.nodes.forEach((node, i) => {
        const bias = b[i];
        if (typeof bias !== 'number' || Math.abs(bias) < 1e-4) return;
        const len = clamp(Math.abs(bias) * 16, 6, 30);
        ctx.save();
        ctx.strokeStyle = weightColor(bias, pal);
        ctx.lineWidth = clamp(0.8 + Math.abs(bias) * 1.4, 0.8, 3);
        ctx.globalAlpha = this.isDimmed(l + 1, i) ? 0.08 : 0.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(node.x - len, node.y - len * 0.8);
        ctx.lineTo(node.x - node.r * 0.7, node.y - node.r * 0.7);
        ctx.stroke();
        ctx.restore();
      });
      ctx.setLineDash([]);
    });
  }

  drawGradients(ctx, layout, layers, pal) {
    layers.forEach((layer, l) => {
      const dW = layer?.dW;
      if (!Array.isArray(dW)) return;
      const from = layout.columns[l];
      const to = layout.columns[l + 1];
      if (!from || !to || from.banded || to.banded) return;
      dW.forEach((row, i) => {
        if (!Array.isArray(row)) return;
        const target = to.nodes[i];
        row.forEach((g, j) => {
          const source = from.nodes[j];
          if (!source || !target || typeof g !== 'number') return;
          if (Math.abs(g) < 0.002) return;
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.strokeStyle = g > 0 ? withAlpha(pal.warn, 0.55) : withAlpha(pal.violet, 0.55);
          ctx.lineWidth = clamp(Math.abs(g) * 16, 0.4, 3.4);
          ctx.stroke();
        });
      });
    });
  }

  drawNodes(ctx, layout, acts, layers, fn, pal) {
    layout.columns.forEach((col) => {
      const layerActs = acts[col.layer] || [];
      col.nodes.forEach((node) => {
        const av = Number(layerActs[node.idx] ?? 0);
        const isSelected =
          this.selected && this.selected.layer === col.layer && this.selected.idx === node.idx;
        const isHovered =
          this.hovered && this.hovered.layer === col.layer && this.hovered.idx === node.idx;
        const dim = this.isDimmed(col.layer, node.idx);

        if (col.banded) {
          this.drawBand(ctx, col, node, layerActs, pal, dim, isSelected);
          return;
        }

        ctx.save();
        ctx.globalAlpha = dim ? 0.22 : 1;

        // activation glow
        if (this.options.showActivations && layerActs.length) {
          const glowR = node.r * 3.1;
          const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowR);
          const strength = clamp(Math.abs(av), 0, 1);
          grad.addColorStop(0, withAlpha(activationColor(av, pal), 0.1 + strength * 0.34));
          grad.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
          ctx.fill();
        }

        // body
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
        if (this.options.showActivations && layerActs.length) {
          const g = ctx.createLinearGradient(
            node.x - node.r,
            node.y - node.r,
            node.x + node.r,
            node.y + node.r,
          );
          const base = activationColor(av, pal);
          g.addColorStop(0, base);
          g.addColorStop(1, withAlpha(base, 0.62));
          ctx.fillStyle = g;
        } else {
          ctx.fillStyle = pal.nodeIdle;
        }
        ctx.fill();

        // ring
        ctx.lineWidth = isSelected ? 2.4 : isHovered ? 1.9 : 1.1;
        ctx.strokeStyle = isSelected
          ? pal.select
          : isHovered
            ? pal.accent
            : withAlpha(pal.nodeStroke, 0.5);
        ctx.stroke();

        if (isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r + 4.5, 0, Math.PI * 2);
          ctx.strokeStyle = withAlpha(pal.accent, 0.55);
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // label
        if (this.options.showLabels && node.r >= 5.5) {
          const lbl = this.nodeLabel(col, node, fn, av);
          ctx.font = `600 ${clamp(node.r * 0.66, 6.5, 11)}px ${MONO}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          if (node.r >= 9.5) {
            ctx.fillStyle = 'rgba(3,6,12,0.85)';
            ctx.fillText(lbl, node.x, node.y);
          } else {
            ctx.fillStyle = pal.text3;
            ctx.fillText(lbl, node.x, node.y + node.r + 8);
          }
        }
        ctx.restore();
      });
    });
  }

  nodeLabel(col, node, fn, av) {
    const isInput = col.layer === 0;
    const isOutput = col.layer === (this.layout?.columns.length || 0) - 1;
    if (isInput && fn.input_labels?.[node.idx]) return shortLabel(fn.input_labels[node.idx]);
    if (isOutput && fn.output_labels?.[node.idx]) return shortLabel(fn.output_labels[node.idx]);
    if (Math.abs(av) >= 100) return av.toFixed(0);
    if (Math.abs(av) >= 10) return av.toFixed(1);
    return av.toFixed(2).replace(/^0\./, '.').replace(/^-0\./, '-.');
  }

  drawBand(ctx, col, node, layerActs, pal, dim, isSelected) {
    const w = node.r * 2.1;
    const h = node.r * 2.6;
    const typeColor = LAYER_TYPE_COLORS[col.type] || pal.accent;
    ctx.save();
    ctx.globalAlpha = dim ? 0.3 : 1;

    // stacked plates behind
    for (let k = 2; k >= 1; k -= 1) {
      ctx.beginPath();
      roundRect(ctx, node.x - w / 2 + k * 3, node.y - h / 2 - k * 3, w, h, 5);
      ctx.fillStyle = withAlpha(typeColor, 0.07 + k * 0.03);
      ctx.strokeStyle = withAlpha(typeColor, 0.2);
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    }

    const grad = ctx.createLinearGradient(node.x - w / 2, node.y - h / 2, node.x + w / 2, node.y + h / 2);
    grad.addColorStop(0, withAlpha(typeColor, 0.34));
    grad.addColorStop(1, withAlpha(typeColor, 0.12));
    ctx.beginPath();
    roundRect(ctx, node.x - w / 2, node.y - h / 2, w, h, 5);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = isSelected ? pal.select : typeColor;
    ctx.lineWidth = isSelected ? 2.2 : 1.3;
    ctx.stroke();

    // activation strip inside the block
    const stripN = Math.min(layerActs.length, 16);
    if (stripN > 0 && this.options.showActivations) {
      const sw = (w - 8) / stripN;
      for (let i = 0; i < stripN; i += 1) {
        const v = Number(layerActs[Math.floor((i / stripN) * layerActs.length)] ?? 0);
        ctx.fillStyle = withAlpha(activationColor(v, pal), 0.85);
        ctx.fillRect(node.x - w / 2 + 4 + i * sw, node.y + h / 2 - 7, Math.max(1, sw - 1), 3);
      }
    }

    ctx.fillStyle = pal.text1;
    ctx.font = `700 ${clamp(node.r * 0.5, 8, 11)}px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`×${col.total}`, node.x, node.y - 2);
    ctx.restore();
  }

  drawHeaders(ctx, layout, fn, pal) {
    const last = layout.columns.length - 1;
    layout.columns.forEach((col) => {
      const node = col.nodes[0];
      if (!node) return;
      const y = layout.padTop - 34;
      const typeColor = col.layer === 0 ? pal.text3 : LAYER_TYPE_COLORS[col.type] || pal.accent;
      const name = col.layer === 0 ? 'Input' : col.layer === last ? 'Output' : layerLabel(col.type);
      const size = col.banded ? `${col.total}` : `${col.count}`;

      ctx.save();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `700 10px ${SANS}`;
      const label = name.toUpperCase();
      const wName = ctx.measureText(label).width;
      ctx.font = `600 9.5px ${MONO}`;
      const wSize = ctx.measureText(size).width;
      const totalW = wName + wSize + 20;
      const x = node.x - totalW / 2;

      ctx.beginPath();
      roundRect(ctx, x, y - 9, totalW, 18, 9);
      ctx.fillStyle = withAlpha(typeColor, 0.14);
      ctx.strokeStyle = withAlpha(typeColor, 0.36);
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();

      ctx.font = `700 10px ${SANS}`;
      ctx.fillStyle = typeColor;
      ctx.textAlign = 'left';
      ctx.fillText(label, x + 8, y + 0.5);
      ctx.font = `600 9.5px ${MONO}`;
      ctx.fillStyle = pal.text3;
      ctx.fillText(size, x + 8 + wName + 6, y + 0.5);

      // expand / collapse affordance
      if (col.collapsible) {
        const expanded = !col.banded;
        ctx.font = `600 9px ${SANS}`;
        ctx.textAlign = 'center';
        ctx.fillStyle = withAlpha(pal.text3, 0.95);
        ctx.fillText(expanded ? '▾ collapse' : `▸ expand ${col.total}`, node.x, y - 22);
      }
      ctx.restore();
    });
  }
}

const SANS = "'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, monospace";

function shortLabel(s) {
  const str = String(s);
  return str.length > 5 ? str.slice(0, 4) + '…' : str;
}

export function roundRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

/** Placeholder text drawn by the component when nothing is built yet. */
export function drawEmptyState(ctx, w, h, pal, message) {
  ctx.fillStyle = pal.canvasBg;
  ctx.fillRect(0, 0, w, h);
  const step = 26;
  ctx.fillStyle = pal.grid;
  for (let x = step; x < w; x += step) {
    for (let y = step; y < h; y += step) ctx.fillRect(x, y, 1, 1);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = pal.text3;
  ctx.font = `600 13px ${SANS}`;
  ctx.fillText(message, w / 2, h / 2);
}
