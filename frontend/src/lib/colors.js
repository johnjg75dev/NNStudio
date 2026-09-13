/**
 * lib/colors.js — canvas palette + weight/activation colour ramps.
 *
 * Canvas 2D cannot read CSS variables cheaply, so the theme palette is
 * mirrored here. Keep in sync with styles/tokens.css.
 */

export const PALETTES = {
  dark: {
    canvasBg: '#070a11',
    grid: 'rgba(255,255,255,0.045)',
    text1: '#eef2fb',
    text2: '#a3adc4',
    text3: '#6d7791',
    text4: '#4d5568',
    border: 'rgba(255,255,255,0.085)',
    surface: 'rgba(255,255,255,0.055)',
    nodeIdle: '#1b2130',
    nodeStroke: '#6ea0ff',
    accent: '#6ea0ff',
    violet: '#a97bff',
    teal: '#2dd4bf',
    pos: '#3ddc84',
    neg: '#ff6b6b',
    warn: '#ffc861',
    select: '#ffffff',
  },
  light: {
    canvasBg: '#ffffff',
    grid: 'rgba(15,23,42,0.06)',
    text1: '#101728',
    text2: '#4c5a75',
    text3: '#7b88a3',
    text4: '#9aa5bc',
    border: 'rgba(15,23,42,0.1)',
    surface: 'rgba(15,23,42,0.05)',
    nodeIdle: '#e8edf7',
    nodeStroke: '#2f6fe4',
    accent: '#2f6fe4',
    violet: '#8b5cf6',
    teal: '#0d9488',
    pos: '#12995c',
    neg: '#d92d20',
    warn: '#b26a00',
    select: '#101728',
  },
};

export function palette(theme = 'dark') {
  return PALETTES[theme] || PALETTES.dark;
}

/** Positive weights → green ramp, negative → red ramp. */
export function weightColor(w, pal) {
  const a = Math.min(1, Math.abs(w) / 3);
  if (w >= 0) {
    const base = pal.pos;
    return mix(base, pal.canvasBg, 0.72 - a * 0.72);
  }
  const base = pal.neg;
  return mix(base, pal.canvasBg, 0.72 - a * 0.72);
}

/** Node fill from activation magnitude (0…1 clamped). */
export function activationColor(v, pal, alpha) {
  const t = clamp01((v + 0.15) / 1.15);
  const c = mixHex(pal.accent, pal.teal, t);
  return alpha === undefined ? c : withAlpha(c, alpha);
}

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Blend two #rrggbb colours; `t` is the weight of `b`. */
export function mixHex(a, b, t) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const k = clamp01(t);
  return `rgb(${Math.round(ca[0] + (cb[0] - ca[0]) * k)},${Math.round(
    ca[1] + (cb[1] - ca[1]) * k,
  )},${Math.round(ca[2] + (cb[2] - ca[2]) * k)})`;
}

/** Blend a colour towards the canvas background to fake transparency. */
export function mix(fg, bg, t) {
  const ca = parseColor(fg);
  const cb = parseColor(bg);
  const k = clamp01(t);
  return `rgba(${Math.round(ca[0] + (cb[0] - ca[0]) * k)},${Math.round(
    ca[1] + (cb[1] - ca[1]) * k,
  )},${Math.round(ca[2] + (cb[2] - ca[2]) * k)},${(ca[3] + (cb[3] - ca[3]) * k).toFixed(3)})`;
}

export function withAlpha(color, alpha) {
  const [r, g, b] = parseColor(color);
  return `rgba(${r},${g},${b},${alpha})`;
}

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const full =
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h;
  return [
    parseInt(full.slice(0, 2), 16) || 0,
    parseInt(full.slice(2, 4), 16) || 0,
    parseInt(full.slice(4, 6), 16) || 0,
  ];
}

function parseColor(color) {
  if (typeof color !== 'string') return [0, 0, 0, 1];
  if (color.startsWith('#')) {
    const [r, g, b] = hexToRgb(color);
    return [r, g, b, 1];
  }
  const m = color.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const parts = m[1].split(/[,/\s]+/).filter(Boolean).map(Number);
    return [parts[0] || 0, parts[1] || 0, parts[2] || 0, parts.length > 3 ? parts[3] : 1];
  }
  return [0, 0, 0, 1];
}

/** Colour used for a weight-matrix heatmap cell. */
export function heatColor(value, pal, scale = 2) {
  const v = Math.min(1, Math.abs(value) / scale);
  const base = value >= 0 ? pal.pos : pal.neg;
  return withAlpha(base, 0.08 + v * 0.72);
}

/** Diverging colour for error magnitude (green → amber → red). */
export function errorColor(err, pal) {
  if (err < 0.08) return pal.pos;
  if (err < 0.25) return pal.warn;
  return pal.neg;
}

export const LAYER_TYPE_COLORS = {
  dense: '#6ea0ff',
  input: '#8b949e',
  dropout: '#ffc861',
  batchnorm: '#3ddc84',
  conv2d: '#f0883e',
  maxpool2d: '#f0883e',
  flatten: '#a97bff',
  lstm: '#2dd4bf',
  simple_rnn: '#2dd4bf',
  embedding: '#c58bff',
  layernorm: '#3ddc84',
  multihead_attention: '#ff7b72',
  positional_encoding: '#c58bff',
};
