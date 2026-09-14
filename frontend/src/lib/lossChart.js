/**
 * lib/lossChart.js — the loss curve.
 * Supports linear/log scaling and a hover crosshair readout.
 */
import { palette, withAlpha } from './colors';

const MONO = "'JetBrains Mono', ui-monospace, monospace";

export class LossChart {
  constructor() {
    this.logScale = false;
    this.hoverIndex = null;
    this.history = [];
    this.accent = null;
  }

  setHistory(history) {
    this.history = Array.isArray(history) ? history : [];
  }

  /** Map a loss value to 0…1 of the plot height. */
  norm(v, min, max) {
    if (this.logScale) {
      const lv = Math.log10(Math.max(v, 1e-9));
      const lo = Math.log10(Math.max(min, 1e-9));
      const hi = Math.log10(Math.max(max, 1e-9));
      if (hi - lo < 1e-9) return 0.5;
      return (lv - lo) / (hi - lo);
    }
    if (max - min < 1e-12) return 0.5;
    return (v - min) / (max - min);
  }

  draw(ctx, w, h, theme = 'dark') {
    const pal = palette(theme);
    const data = this.history;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = pal.canvasBg;
    ctx.fillRect(0, 0, w, h);

    const padL = 40;
    const padR = 10;
    const padT = 10;
    const padB = 16;
    const pw = Math.max(10, w - padL - padR);
    const ph = Math.max(10, h - padT - padB);

    if (!data || data.length < 2) {
      ctx.strokeStyle = withAlpha(pal.text4, 0.35);
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, pw, ph);
      ctx.setLineDash([]);
      ctx.fillStyle = pal.text4;
      ctx.font = `500 11px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        data && data.length === 1 ? 'training…' : 'press train to plot the loss curve',
        padL + pw / 2,
        padT + ph / 2,
      );
      return;
    }

    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min)) {
      min = 0;
      max = 1;
    }
    if (this.logScale) min = Math.max(min, 1e-9);
    const span = max - min || Math.abs(max) || 1;
    min = Math.max(this.logScale ? 1e-9 : -Infinity, min - span * 0.08);
    max = max + span * 0.08;

    // gridlines + y labels
    ctx.font = `500 9px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 3; i += 1) {
      const y = padT + ph - (i / 3) * ph;
      ctx.strokeStyle = withAlpha(pal.text4, 0.16);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
      ctx.stroke();
      const value = this.logScale
        ? Math.pow(10, Math.log10(Math.max(min, 1e-9)) + (i / 3) * (Math.log10(max) - Math.log10(Math.max(min, 1e-9))))
        : min + (i / 3) * (max - min);
      ctx.fillStyle = pal.text4;
      ctx.fillText(formatTick(value), padL - 6, y);
    }

    const px = (i) => padL + (i / (data.length - 1)) * pw;
    const py = (v) => padT + ph - this.norm(v, min, max) * ph;

    // area fill
    ctx.beginPath();
    ctx.moveTo(px(0), py(data[0]));
    for (let i = 1; i < data.length; i += 1) ctx.lineTo(px(i), py(data[i]));
    ctx.lineTo(px(data.length - 1), padT + ph);
    ctx.lineTo(px(0), padT + ph);
    ctx.closePath();
    const fill = ctx.createLinearGradient(0, padT, 0, padT + ph);
    fill.addColorStop(0, withAlpha(pal.accent, 0.3));
    fill.addColorStop(1, withAlpha(pal.accent, 0.01));
    ctx.fillStyle = fill;
    ctx.fill();

    // line
    ctx.beginPath();
    ctx.moveTo(px(0), py(data[0]));
    for (let i = 1; i < data.length; i += 1) ctx.lineTo(px(i), py(data[i]));
    const stroke = ctx.createLinearGradient(padL, 0, padL + pw, 0);
    stroke.addColorStop(0, pal.violet);
    stroke.addColorStop(1, pal.accent);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // current value marker
    const lastX = px(data.length - 1);
    const lastY = py(data[data.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = pal.accent;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 6.5, 0, Math.PI * 2);
    ctx.strokeStyle = withAlpha(pal.accent, 0.4);
    ctx.lineWidth = 1;
    ctx.stroke();

    // hover crosshair
    if (this.hoverIndex !== null && this.hoverIndex >= 0 && this.hoverIndex < data.length) {
      const hx = px(this.hoverIndex);
      const hy = py(data[this.hoverIndex]);
      ctx.save();
      ctx.strokeStyle = withAlpha(pal.text2, 0.5);
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, padT);
      ctx.lineTo(hx, padT + ph);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(hx, hy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = pal.text1;
      ctx.fill();
      const label = `epoch ${this.hoverIndex + 1} · ${formatTick(data[this.hoverIndex])}`;
      ctx.font = `600 10px ${MONO}`;
      const tw = ctx.measureText(label).width + 12;
      const bx = Math.min(Math.max(padL, hx - tw / 2), padL + pw - tw);
      ctx.fillStyle = theme === 'dark' ? 'rgba(12,16,26,0.95)' : 'rgba(255,255,255,0.97)';
      ctx.strokeStyle = withAlpha(pal.accent, 0.4);
      ctx.beginPath();
      ctx.roundRect?.(bx, padT + 2, tw, 18, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = pal.text1;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, bx + 6, padT + 11);
      ctx.restore();
    }

    // x axis label
    ctx.fillStyle = pal.text4;
    ctx.font = `500 9px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('epoch 1', padL, h - 3);
    ctx.textAlign = 'right';
    ctx.fillText(`epoch ${data.length}`, padL + pw, h - 3);
  }

  indexAt(px, w) {
    if (!this.history || this.history.length < 2) return null;
    const padL = 40;
    const padR = 10;
    const pw = Math.max(10, w - padL - padR);
    const t = (px - padL) / pw;
    if (t < -0.02 || t > 1.02) return null;
    return Math.round(Math.min(1, Math.max(0, t)) * (this.history.length - 1));
  }
}

function formatTick(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const a = Math.abs(n);
  if (a === 0) return '0';
  if (a >= 1000) return n.toFixed(0);
  if (a >= 1) return n.toFixed(2);
  if (a >= 0.001) return n.toFixed(4);
  return n.toExponential(1);
}
