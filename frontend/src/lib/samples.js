/**
 * lib/samples.js — evaluation-sample helpers.
 *
 * The server returns `{ x, y, pred }` triples; correctness is derived here with
 * the same rule the backend uses for accuracy (binarise at 0.5, compare with the
 * rounded target across every output).
 */

export function matchesTarget(pred, expected, threshold = 0.5) {
  if (!Array.isArray(pred) || !Array.isArray(expected) || pred.length === 0) return false;
  const n = Math.min(pred.length, expected.length);
  for (let i = 0; i < n; i += 1) {
    const p = Number(pred[i]) > threshold ? 1 : 0;
    const y = Math.round(Number(expected[i]) || 0);
    if (Math.abs(p - y) >= 0.5) return false;
  }
  return pred.length === expected.length;
}

/** Argmax label for multi-class one-hot outputs. */
export function argmax(values = []) {
  let best = 0;
  let bestV = -Infinity;
  values.forEach((v, i) => {
    const n = Number(v);
    if (Number.isFinite(n) && n > bestV) {
      bestV = n;
      best = i;
    }
  });
  return best;
}

/** Add `index`, `correct` and a short `label` to raw evaluate() samples. */
export function annotateSamples(samples, func = null) {
  return (samples || []).map((s, i) => ({
    ...s,
    index: i,
    correct: matchesTarget(s.pred, s.y),
    label: sampleLabel(s, func, i),
  }));
}

/** Human-friendly label for one sample. */
export function sampleLabel(sample, func, index) {
  if (!sample) return `#${index}`;
  if (func?.output_labels?.length && Array.isArray(sample.y)) {
    if (sample.y.length > 1) {
      const i = argmax(sample.y);
      return func.output_labels[i] || `#${index}`;
    }
    return `${func.output_labels[0]}=${Number(sample.y[0]) > 0.5 ? 1 : 0}`;
  }
  return `#${index}`;
}

/** Aggregate accuracy over annotated samples. */
export function summarise(samples = []) {
  const total = samples.length;
  const correct = samples.filter((s) => s.correct).length;
  return { total, correct, accuracy: total ? correct / total : null };
}
