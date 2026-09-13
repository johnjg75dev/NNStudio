/** Small formatting helpers shared across the UI. */

export function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

export function fmtCompact(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (abs >= 1e4) return (n / 1e3).toFixed(1) + 'k';
  if (abs >= 100) return n.toFixed(0);
  if (abs >= 1) return n.toFixed(2);
  if (abs >= 0.001) return n.toFixed(4);
  return n.toExponential(1);
}

export function fmtLoss(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0.0000';
  if (n < 0.0001) return n.toExponential(2);
  return n.toFixed(4);
}

export function fmtPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return (Number(v) * 100).toFixed(digits) + '%';
}

export function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toLocaleString('en-US');
}

/** Learning-rate slider works in log10 space. */
export const lrToSlider = (lr) => Math.log10(Math.max(1e-6, lr || 1e-6));
export const sliderToLr = (v) => Math.pow(10, Number(v));
export function fmtLr(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (n >= 0.001) return n.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  return n.toExponential(1);
}

export function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function fmtRelative(iso) {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const secs = Math.round((Date.now() - then) / 1000);
  if (secs < 60) return 'just now';
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function fmtClock(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

export function titleCase(s) {
  if (!s) return '';
  return String(s).charAt(0).toUpperCase() + String(s).slice(1);
}

export function uid(prefix = 'id') {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

export function download(filename, text, mime = 'application/json') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read that file.'));
    reader.readAsText(file);
  });
}

export function mean(values) {
  if (!values || values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += Number(v) || 0;
  return sum / values.length;
}

/** Alias of fmt — reads better at call sites dealing with weights/activations. */
export const fmtNum = fmt;

/** Join an array for display, collapsing the tail into an ellipsis. */
export function truncateArray(values, max = 6, formatter = fmt) {
  const arr = Array.isArray(values) ? values : [];
  const out = arr.slice(0, max).map((v) => formatter(v));
  if (arr.length > max) out.push(`… +${arr.length - max}`);
  return out;
}
