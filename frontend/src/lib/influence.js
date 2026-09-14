/**
 * lib/influence.js — client-side node inspection helpers.
 *
 * Column indexing matches the network graph: column 0 is the input layer, and
 * `snapshot.layers[l]` carries the weights from column `l` to column `l + 1`.
 * Activations follow the same scheme (`activations[l][idx]`).
 */

/** The layer object feeding column `layer` (null for inputs). */
export function incomingLayer(snapshot, layer) {
  const layers = snapshot?.layers || [];
  return layer >= 1 ? layers[layer - 1] || null : null;
}

/** The layer object taking column `layer` onwards (null for the last column). */
export function outgoingLayer(snapshot, layer) {
  const layers = snapshot?.layers || [];
  return layers[layer] || null;
}

/** Everything the inspector needs about one node. */
export function nodeInfo(snapshot, node) {
  if (!snapshot || !node) return null;
  const layers = snapshot.layers || [];
  const activations = snapshot.activations || [];
  const inLayer = incomingLayer(snapshot, node.layer);
  const outLayer = outgoingLayer(snapshot, node.layer);
  const acts = activations[node.layer] || [];
  const value = Number(acts[node.idx]);
  const actNames = Array.isArray(inLayer?.activation)
    ? inLayer.activation
    : inLayer?.activation
      ? [inLayer.activation]
      : [];

  return {
    node,
    column: node.layer,
    type: node.layer === 0 ? 'input' : inLayer?.is_output ? 'output' : 'hidden',
    layerType: inLayer?.type || (node.layer === 0 ? 'input' : null),
    value: Number.isFinite(value) ? value : null,
    bias: node.layer >= 1 ? Number(inLayer?.b?.[node.idx]) : null,
    activationName: node.layer >= 1 ? actNames[node.idx] || actNames[0] || null : null,
    weightsIn: node.layer >= 1 ? (inLayer?.W?.[node.idx] || []) : [],
    weightsOut: outLayer?.W ? outLayer.W.map((row) => Number(row?.[node.idx])) : [],
    gradientIn: node.layer >= 1 ? (inLayer?.dW?.[node.idx] || []) : [],
    gradientOut: outLayer?.dW ? outLayer.dW.map((row) => Number(row?.[node.idx])) : [],
    nIn: inLayer?.n_in ?? null,
    nOut: inLayer?.n_out ?? null,
    fanOut: outLayer?.n_out ?? null,
  };
}

/**
 * Walk backwards from `node` and estimate how much each upstream neuron
 * contributes to it. Contribution is the product of |weights| along the path,
 * scaled by the magnitude of the activation at each hop — so a strong weight
 * from a silent neuron still scores low.
 *
 * Returns `[{ layer, idx, influence }]` sorted strongest first, normalised so
 * the influences sum to 1.
 */
export function traceInfluence(snapshot, node, { maxNodes = 40 } = {}) {
  if (!snapshot || !node || node.layer < 1) return [];
  const layers = snapshot.layers || [];
  const activations = snapshot.activations || [];

  // score[l] = contribution magnitude of each neuron in column l
  let current = new Map([[node.idx, 1]]);
  const trails = [];

  for (let l = node.layer; l >= 1; l -= 1) {
    const W = layers[l - 1]?.W;
    if (!W) break;
    const prevActs = activations[l - 1] || [];
    const next = new Map();
    for (const [i, score] of current) {
      const row = W[i];
      if (!row) continue;
      for (let j = 0; j < row.length; j += 1) {
        const w = Math.abs(Number(row[j]) || 0);
        if (!w) continue;
        const a = Math.abs(Number(prevActs[j]) || 0);
        const contribution = score * w * (0.25 + a);
        next.set(j, (next.get(j) || 0) + contribution);
      }
    }
    if (!next.size) break;
    const total = [...next.values()].reduce((s, v) => s + v, 0) || 1;
    trails.push({
      layer: l - 1,
      nodes: [...next.entries()]
        .map(([idx, raw]) => ({ layer: l - 1, idx, influence: raw / total, raw }))
        .sort((a, b) => b.raw - a.raw),
    });
    current = next;
  }

  // The input layer is the meaningful endpoint: rank those neurons directly.
  const inputTrail = trails.find((t) => t.layer === 0);
  const source = inputTrail || trails[trails.length - 1];
  if (!source) return [];
  return source.nodes.slice(0, maxNodes).map((n) => ({ ...n, depth: node.layer - n.layer }));
}
