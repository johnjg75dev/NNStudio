/**
 * lib/activations.js — activation maths + the curve painter used by the
 * Learn page and by activation pickers.
 */
import { palette, withAlpha } from './colors';

export const ACTIVATION_INFO = [
  {
    key: 'relu',
    name: 'ReLU',
    formula: 'f(x) = max(0, x)',
    range: '[0, ∞)',
    use: 'Default for hidden layers in MLPs and CNNs.',
    pros: 'Cheap, sparse, mitigates vanishing gradients.',
    cons: 'Dead neurons when the learning rate is too high.',
  },
  {
    key: 'leakyrelu',
    name: 'Leaky ReLU',
    formula: 'f(x) = x if x > 0 else 0.01·x',
    range: '(−∞, ∞)',
    use: 'When ReLU neurons keep dying.',
    pros: 'No dead neurons, still sparse and fast.',
    cons: 'Small negative slope adds a little compute.',
  },
  {
    key: 'tanh',
    name: 'Tanh',
    formula: 'f(x) = (e²ˣ − 1) / (e²ˣ + 1)',
    range: '(−1, 1)',
    use: 'Shallow networks, RNNs, zero-centred outputs.',
    pros: 'Zero-centred, smooth gradient, bounded.',
    cons: 'Saturates at the extremes → vanishing gradients.',
  },
  {
    key: 'sigmoid',
    name: 'Sigmoid',
    formula: 'f(x) = 1 / (1 + e⁻ˣ)',
    range: '(0, 1)',
    use: 'Binary output layer with BCE loss.',
    pros: 'Probabilistic interpretation, smooth.',
    cons: 'Saturates easily; poor for deep hidden layers.',
  },
  {
    key: 'gelu',
    name: 'GELU',
    formula: 'f(x) ≈ 0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))',
    range: '(−0.17, ∞)',
    use: 'Transformers — BERT, GPT, ViT.',
    pros: 'Smooth, outperforms ReLU in deep nets.',
    cons: 'Slightly more expensive; overkill for toy tasks.',
  },
  {
    key: 'swish',
    name: 'Swish',
    formula: 'f(x) = x · sigmoid(x)',
    range: '(−0.28, ∞)',
    use: 'Deep nets where ReLU has plateaued.',
    pros: 'Self-gating, smooth, strong in deep stacks.',
    cons: 'More expensive; gains shrink on shallow nets.',
  },
];

/** Keyed lookup for the info cards (`ACTIVATION_BY_KEY.relu.name`). */
export const ACTIVATION_BY_KEY = Object.fromEntries(ACTIVATION_INFO.map((a) => [a.key, a]));

/** Short label for an activation key, falling back to the raw key. */
export function activationLabel(key) {
  return ACTIVATION_BY_KEY[key]?.name || key || 'linear';
}

export function evaluate(x, type) {
  switch (type) {
    case 'relu':
      return Math.max(0, x);
    case 'leakyrelu':
      return x > 0 ? x : 0.01 * x;
    case 'tanh':
      return Math.tanh(x);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'gelu': {
      const t = Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x));
      return 0.5 * x * (1 + t);
    }
    case 'swish':
      return x / (1 + Math.exp(-x));
    default:
      return x;
  }
}

export function derivative(x, type) {
  const eps = 0.001;
  return (evaluate(x + eps, type) - evaluate(x - eps, type)) / (2 * eps);
}

/**
 * Paint f(x) (and optionally f′(x)) on a canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {{hoverX?: number|null}} opts
 */
export function drawActivationCurve(ctx, w, h, type, theme = 'dark', opts = {}) {
  const pal = palette(theme);
  const cx = w / 2;
  const cy = h / 2;
  const scale = w / 8; // x ∈ [−4, 4]

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = pal.canvasBg;
  ctx.fillRect(0, 0, w, h);

  // grid
  ctx.strokeStyle = pal.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = -4; i <= 4; i += 1) {
    ctx.moveTo(cx + i * scale, 0);
    ctx.lineTo(cx + i * scale, h);
    ctx.moveTo(0, cy - i * scale);
    ctx.lineTo(w, cy - i * scale);
  }
  ctx.stroke();

  // axes
  ctx.strokeStyle = withAlpha(pal.text3, 0.45);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, h);
  ctx.moveTo(0, cy);
  ctx.lineTo(w, cy);
  ctx.stroke();

  // ticks
  ctx.fillStyle = pal.text4;
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = -4; i <= 4; i += 2) {
    if (i === 0) continue;
    ctx.fillText(String(i), cx + i * scale, cy + 4);
  }

  // derivative (faint, dashed)
  if (opts.showDerivative !== false) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = withAlpha(pal.violet, 0.55);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let first = true;
    for (let px = 0; px <= w; px += 2) {
      const x = (px - cx) / scale;
      const y = cy - derivative(x, type) * scale * 0.5;
      if (first) {
        ctx.moveTo(px, y);
        first = false;
      } else ctx.lineTo(px, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // f(x) with gradient stroke + area fill
  ctx.beginPath();
  let first = true;
  for (let px = 0; px <= w; px += 1) {
    const x = (px - cx) / scale;
    const y = cy - evaluate(x, type) * scale;
    if (first) {
      ctx.moveTo(px, y);
      first = false;
    } else ctx.lineTo(px, y);
  }
  const stroke = ctx.createLinearGradient(0, 0, w, 0);
  stroke.addColorStop(0, pal.violet);
  stroke.addColorStop(1, pal.accent);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // hover readout
  if (opts.hoverX !== null && opts.hoverX !== undefined) {
    const px = cx + opts.hoverX * scale;
    const py = cy - evaluate(opts.hoverX, type) * scale;
    ctx.save();
    ctx.strokeStyle = withAlpha(pal.text3, 0.5);
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = pal.accent;
    ctx.fill();
    ctx.strokeStyle = pal.canvasBg;
    ctx.lineWidth = 2;
    ctx.stroke();

    const label = `f(${opts.hoverX.toFixed(2)}) = ${evaluate(opts.hoverX, type).toFixed(4)}`;
    ctx.font = '600 10px "JetBrains Mono", monospace';
    const tw = ctx.measureText(label).width + 12;
    const bx = Math.min(Math.max(4, px + 8), w - tw - 4);
    const by = Math.max(4, py - 26);
    ctx.fillStyle = withAlpha(pal.text1, 0.06);
    ctx.strokeStyle = withAlpha(pal.accent, 0.4);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect?.(bx, by, tw, 18, 5);
    if (ctx.roundRect) {
      ctx.fillStyle = theme === 'dark' ? 'rgba(12,16,26,0.94)' : 'rgba(255,255,255,0.96)';
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = pal.text1;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, bx + 6, by + 9.5);
    ctx.restore();
  }

  // marker: the live output level of the inspected neuron (horizontal guide)
  const m = opts.marker;
  if (m !== null && m !== undefined && Number.isFinite(m)) {
    const my = cy - m * scale;
    if (my > 8 && my < h - 4) {
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = withAlpha(pal.pos, 0.8);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(0, my);
      ctx.lineTo(w, my);
      ctx.stroke();
      ctx.setLineDash([]);
      const label = `neuron output ${m.toFixed(3)}`;
      ctx.font = '600 9px "JetBrains Mono", monospace';
      const tw = ctx.measureText(label).width + 12;
      const bx = w - tw - 4;
      const by = my - 16 < 2 ? my + 3 : my - 16;
      ctx.fillStyle = theme === 'dark' ? 'rgba(12,16,26,0.9)' : 'rgba(255,255,255,0.94)';
      ctx.strokeStyle = withAlpha(pal.pos, 0.5);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect?.(bx, by, tw, 14, 4);
      if (ctx.roundRect) {
        ctx.fill();
        ctx.stroke();
      }
      ctx.fillStyle = pal.pos;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, bx + 6, by + 7.5);
      ctx.restore();
    }
  }
}
