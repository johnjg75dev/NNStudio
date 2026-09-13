/**
 * lib/plot2d.js — painters for the 2-D inspector.
 *
 * Unlike the legacy renderer, predictions come from the server
 * (`POST /api/train/evaluate` with `ranges`) instead of an approximate
 * client-side forward pass, so what you see is what the model actually does.
 */
import { palette, withAlpha } from './colors';

const MONO = "'JetBrains Mono', ui-monospace, monospace";

/**
 * @param {object} p
 * @param {number[][]} p.grid  rows of predictions, length = res*res, values = outputs
 * @param {number} p.res
 * @param {{x:[number,number], y:[number,number]}} p.bounds
 * @param {Array<{x:number[],y:number[],pred:number[]}>} p.samples
 */
export function drawDecisionBoundary(ctx, w, h, theme, p) {
  const pal = palette(theme);
  const { grid, res, bounds, samples = [], outputs = 1 } = p;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = pal.canvasBg;
  ctx.fillRect(0, 0, w, h);

  if (!grid || !res) {
    placeholder(ctx, w, h, pal, 'Decision boundary needs a 2-input task');
    return;
  }

  const img = ctx.createImageData(res, res);
  for (let gy = 0; gy < res; gy += 1) {
    for (let gx = 0; gx < res; gx += 1) {
      const cell = grid[gy * res + gx];
      if (!cell) continue;
      // For multi-output nets use argmax; for a single output use p>0.5.
      let cls = 0;
      let conf = 0;
      if (outputs > 1) {
        let best = -Infinity;
        cell.forEach((v, i) => {
          if (v > best) {
            best = v;
            cls = i;
          }
        });
        conf = 0.75;
      } else {
        const v = cell[0] ?? 0;
        cls = v > 0.5 ? 1 : 0;
        conf = Math.abs(v - 0.5) * 2;
      }
      const base = cls === 1 ? [47, 111, 228] : [217, 83, 62];
      const tint = 0.16 + conf * 0.42;
      const idx = (gy * res + gx) * 4;
      img.data[idx] = base[0] * tint + (theme === 'dark' ? 8 : 245) * (1 - tint);
      img.data[idx + 1] = base[1] * tint + (theme === 'dark' ? 12 : 247) * (1 - tint);
      img.data[idx + 2] = base[2] * tint + (theme === 'dark' ? 22 : 252) * (1 - tint);
      img.data[idx + 3] = 255;
    }
  }

  // upscale the small image with smoothing for a soft boundary
  const off = document.createElement('canvas');
  off.width = res;
  off.height = res;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.globalAlpha = 0.92;
  ctx.drawImage(off, 0, 0, w, h);
  ctx.globalAlpha = 1;

  // axes frame + ticks
  ctx.strokeStyle = withAlpha(pal.text4, 0.4);
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const [xMin, xMax] = bounds.x;
  const [yMin, yMax] = bounds.y;
  ctx.fillStyle = withAlpha(pal.text3, 0.9);
  ctx.font = `500 9px ${MONO}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(fmt(xMin), 4, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(fmt(xMax), w - 4, h - 4);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(fmt(yMax), 4, 4);
  ctx.textBaseline = 'bottom';
  ctx.fillText(fmt(yMin), 4, h - 16);

  // data points
  if (samples.length) {
    samples.forEach((s) => {
      const px = ((s.x[0] - xMin) / (xMax - xMin || 1)) * w;
      const py = h - ((s.x[1] - yMin) / (yMax - yMin || 1)) * h;
      const label = s.y?.[0] ?? 0;
      const correct = outputs > 1
        ? argmax(s.pred) === argmax(s.y)
        : (s.pred?.[0] ?? 0) > 0.5 === label > 0.5;
      ctx.beginPath();
      ctx.arc(px, py, 4.4, 0, Math.PI * 2);
      ctx.fillStyle = label > 0.5 ? pal.accent : pal.neg;
      ctx.fill();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = correct ? pal.canvasBg : pal.warn;
      ctx.stroke();
    });
  }
}

export function drawFunctionPlot(ctx, w, h, theme, p) {
  const pal = palette(theme);
  const { samples = [], curve = null } = p;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = pal.canvasBg;
  ctx.fillRect(0, 0, w, h);

  const m = { l: 34, r: 10, t: 12, b: 20 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  if (pw < 20 || ph < 20) return;

  const pts = [...(curve || []), ...samples.map((s) => ({ x: s.x[0], y: s.pred?.[0] ?? 0 }))];
  let xMin = 0;
  let xMax = 1;
  let yMin = 0;
  let yMax = 1;
  if (pts.length) {
    xMin = Math.min(...pts.map((q) => q.x));
    xMax = Math.max(...pts.map((q) => q.x));
    yMin = Math.min(0, ...pts.map((q) => q.y));
    yMax = Math.max(1, ...pts.map((q) => q.y));
  }
  if (xMax - xMin < 1e-9) xMax = xMin + 1;
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad;
  yMax += yPad;

  const X = (v) => m.l + ((v - xMin) / (xMax - xMin)) * pw;
  const Y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  // grid
  ctx.strokeStyle = withAlpha(pal.text4, 0.16);
  ctx.lineWidth = 1;
  ctx.font = `500 9px ${MONO}`;
  ctx.fillStyle = pal.text4;
  for (let i = 0; i <= 4; i += 1) {
    const gy = m.t + (i / 4) * ph;
    ctx.beginPath();
    ctx.moveTo(m.l, gy);
    ctx.lineTo(m.l + pw, gy);
    ctx.stroke();
    const value = yMax - (i / 4) * (yMax - yMin);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmt(value), m.l - 5, gy);
  }
  for (let i = 0; i <= 4; i += 1) {
    const gx = m.l + (i / 4) * pw;
    ctx.beginPath();
    ctx.moveTo(gx, m.t);
    ctx.lineTo(gx, m.t + ph);
    ctx.stroke();
    const value = xMin + (i / 4) * (xMax - xMin);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(fmt(value), gx, m.t + ph + 5);
  }

  // zero line
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = withAlpha(pal.text3, 0.5);
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(m.l, Y(0));
    ctx.lineTo(m.l + pw, Y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // target curve (ground truth) if provided as `target`
  if (p.target?.length) {
    ctx.beginPath();
    p.target.forEach((q, i) => (i ? ctx.lineTo(X(q.x), Y(q.y)) : ctx.moveTo(X(q.x), Y(q.y))));
    ctx.strokeStyle = withAlpha(pal.pos, 0.85);
    ctx.lineWidth = 1.6;
    ctx.setLineDash([5, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // prediction curve
  if (curve?.length) {
    ctx.beginPath();
    curve.forEach((q, i) => (i ? ctx.lineTo(X(q.x), Y(q.y)) : ctx.moveTo(X(q.x), Y(q.y))));
    const g = ctx.createLinearGradient(m.l, 0, m.l + pw, 0);
    g.addColorStop(0, pal.violet);
    g.addColorStop(1, pal.accent);
    ctx.strokeStyle = g;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  // sample markers
  samples.forEach((s) => {
    const pred = s.pred?.[0] ?? 0;
    const truth = s.y?.[0];
    if (truth !== undefined && truth !== null) {
      ctx.beginPath();
      ctx.arc(X(s.x[0]), Y(truth), 3, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(pal.pos, 0.9);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(X(s.x[0]), Y(pred), 3.4, 0, Math.PI * 2);
    ctx.fillStyle = pal.accent;
    ctx.fill();
    ctx.strokeStyle = pal.canvasBg;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  });
}

/** Line chart for latent / node sensitivity sweeps: [{val, result}]. */
export function drawSweepChart(ctx, w, h, theme, data, opts = {}) {
  const pal = palette(theme);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = pal.canvasBg;
  ctx.fillRect(0, 0, w, h);
  if (!data?.length) {
    placeholder(ctx, w, h, pal, opts.emptyText || 'No sweep data');
    return;
  }
  const m = { l: 34, r: 10, t: 12, b: 18 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  const xs = data.map((d) => d.val);
  const ys = data.map((d) => d.result);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.15 || 0.5;
  yMin -= pad;
  yMax += pad;
  const X = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
  const Y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

  ctx.strokeStyle = withAlpha(pal.text4, 0.18);
  ctx.lineWidth = 1;
  ctx.font = `500 9px ${MONO}`;
  ctx.fillStyle = pal.text4;
  for (let i = 0; i <= 3; i += 1) {
    const gy = m.t + (i / 3) * ph;
    ctx.beginPath();
    ctx.moveTo(m.l, gy);
    ctx.lineTo(m.l + pw, gy);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmt(yMax - (i / 3) * (yMax - yMin)), m.l - 5, gy);
  }

  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = withAlpha(pal.text3, 0.45);
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(m.l, Y(0));
    ctx.lineTo(m.l + pw, Y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // area
  ctx.beginPath();
  data.forEach((d, i) => (i ? ctx.lineTo(X(d.val), Y(d.result)) : ctx.moveTo(X(d.val), Y(d.result))));
  ctx.lineTo(X(xMax), m.t + ph);
  ctx.lineTo(X(xMin), m.t + ph);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, m.t, 0, m.t + ph);
  fill.addColorStop(0, withAlpha(pal.accent, 0.28));
  fill.addColorStop(1, withAlpha(pal.accent, 0.02));
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.beginPath();
  data.forEach((d, i) => (i ? ctx.lineTo(X(d.val), Y(d.result)) : ctx.moveTo(X(d.val), Y(d.result))));
  ctx.strokeStyle = pal.accent;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  ctx.fillStyle = pal.text4;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(fmt(xMin), m.l, m.t + ph + 4);
  ctx.textAlign = 'right';
  ctx.fillText(fmt(xMax), m.l + pw, m.t + ph + 4);
}

function placeholder(ctx, w, h, pal, text) {
  ctx.fillStyle = pal.text4;
  ctx.font = `500 11px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2);
}

function fmt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const a = Math.abs(n);
  if (a >= 100) return n.toFixed(0);
  if (a >= 1) return n.toFixed(1);
  if (a >= 0.01) return n.toFixed(2);
  return n.toExponential(1);
}

function argmax(arr) {
  let best = 0;
  let bv = -Infinity;
  (arr || []).forEach((v, i) => {
    if (v > bv) {
      bv = v;
      best = i;
    }
  });
  return best;
}
