/**
 * lib/grid.js — convert flat activation vectors to/from 2-D grids.
 *
 * The backend stores image-shaped inputs and outputs as flat arrays with
 * channels interleaved: index = (y * width + x) * channels + c.
 */

/** Shape → { w, h, c } with sensible fallbacks. */
export function normaliseShape(shape, length) {
  if (Array.isArray(shape) && shape.length >= 2) {
    const [h, w, c = 1] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
    return { w: Number(w) || 0, h: Number(h) || 0, c: Number(c) || 1 };
  }
  const n = Number(length) || 0;
  const side = Math.round(Math.sqrt(n));
  return { w: side, h: side, c: 1 };
}

/**
 * Flatten a value list into a single-channel grid of brightness values.
 * Multi-channel data is averaged so the preview stays readable.
 */
export function arrayToGrid(values, shape, labels = null) {
  const data = Array.isArray(values) ? values.map((v) => Number(v) || 0) : [];
  const { w, h, c } = normaliseShape(shape, data.length);
  if (!w || !h) return { w: 0, h: 0, c: 1, values: [], labels: labels || [] };

  const cells = w * h;
  const out = new Array(cells).fill(0);
  for (let i = 0; i < cells; i += 1) {
    if (c === 1) {
      out[i] = data[i] ?? 0;
    } else {
      let sum = 0;
      for (let k = 0; k < c; k += 1) sum += data[i * c + k] ?? 0;
      out[i] = sum / c;
    }
  }
  return { w, h, c, values: out, labels: labels || [] };
}

/** Inverse of arrayToGrid for single-channel grids. */
export function gridToArray(grid, { w, h }) {
  const out = new Array(w * h).fill(0);
  (grid || []).forEach((row, y) => {
    (row || []).forEach((v, x) => {
      out[y * w + x] = Number(v) || 0;
    });
  });
  return out;
}

/** 64-bin histogram of a flat value list (used by the dataset editor). */
export function histogram(values, bins = 64) {
  const out = new Array(bins).fill(0);
  (values || []).forEach((v) => {
    const n = Number(v) || 0;
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(((n + 0) / 1) * bins)));
    out[idx] += 1;
  });
  return out;
}
