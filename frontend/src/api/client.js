/**
 * api/client.js — the only place that talks HTTP to Flask.
 *
 * The backend is inconsistent about envelopes (`{ok, data}` vs
 * `{success, ...}`), so `request()` normalises both into a plain payload and
 * throws `ApiError` on failure.
 */

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(method, path, body, { raw = false } = {}) {
  const opts = {
    method,
    headers: {},
    credentials: 'same-origin',
  };
  if (body !== undefined && body !== null) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(path, opts);
  } catch (networkError) {
    throw new ApiError('Cannot reach the NNStudio server.', 0);
  }

  if (res.status === 401) {
    // Session expired — bounce to the login page instead of failing silently.
    const here = window.location.pathname;
    if (!here.startsWith('/login') && !here.startsWith('/signup')) {
      window.location.assign('/login?next=' + encodeURIComponent(here));
    }
    throw new ApiError('Your session expired. Please sign in again.', 401);
  }

  if (raw) return res;

  let json = null;
  const text = await res.text();
  if (text) {
    try {
      json = JSON.parse(text);
    } catch {
      throw new ApiError(`Server returned a non-JSON response (${res.status}).`, res.status);
    }
  }
  if (!json) throw new ApiError(`Empty response from ${path}`, res.status);

  const isOk = json.ok === true || json.success === true;
  if (!isOk) {
    throw new ApiError(json.error || json.message || `Request failed (${res.status})`, res.status);
  }
  return json.data !== undefined ? json.data : json;
}

const get = (path, opts) => request('GET', path, null, opts);
const post = (path, body, opts) => request('POST', path, body ?? {}, opts);
const put = (path, body) => request('PUT', path, body);
const del = (path) => request('DELETE', path, {});

export const api = {
  // ── registry / modules ─────────────────────────────────────────────
  modules: () => get('/api/modules/all'),
  module: (key) => get(`/api/modules/${key}`),
  moduleCategory: (category) => get(`/api/modules/category/${category}`),
  functionDataset: (key) => get(`/api/modules/functions/${key}/dataset`),

  // ── training session ───────────────────────────────────────────────
  build: (config) => post('/api/session/build', config),
  reset: () => post('/api/session/reset'),
  snapshot: () => get('/api/session/snapshot'),
  predict: ({ x, startLayer = 0, endLayer = null, nodeOverrides = null }) =>
    post('/api/session/predict', {
      x,
      start_layer: startLayer,
      end_layer: endLayer,
      node_overrides: nodeOverrides,
    }),
  latentSweep: (payload) => post('/api/session/latent-sweep', payload),
  exportSession: () => post('/api/session/export'),
  importSession: (data) => post('/api/session/import', data),

  // ── training ───────────────────────────────────────────────────────
  trainStep: (steps, lr) => post('/api/train/step', { steps, lr }),
  evaluate: ({ ranges = null, startLayer = 0, endLayer = null } = {}) => {
    const body = {};
    if (ranges) body.ranges = ranges;
    if (startLayer !== undefined && startLayer !== null) body.start_layer = startLayer;
    if (endLayer !== undefined && endLayer !== null) body.end_layer = endLayer;
    return post('/api/train/evaluate', body);
  },

  // ── presets ────────────────────────────────────────────────────────
  savePreset: (config) => post('/api/presets/save', config),
  deletePreset: (id) => del(`/api/presets/${id}`),

  // ── model library ──────────────────────────────────────────────────
  listModels: () => get('/api/models'),
  saveModel: (payload) => post('/api/models/save', payload),
  getModel: (id) => get(`/api/models/${id}`),
  loadModel: (id) => post(`/api/models/${id}/load-session`, {}),
  deleteModel: (id) => del(`/api/models/${id}`),
  exportFormats: () => get('/api/models/formats'),
  exportModel: (id, format) => post(`/api/models/${id}/export`, { format }),
  modelDownloadUrl: (id, format) => `/api/models/${id}/download/${format}`,

  // ── custom functions ───────────────────────────────────────────────
  listFunctions: () => get('/api/functions/custom'),
  getFunction: (id) => get(`/api/functions/custom/${id}`),
  createFunction: (data) => post('/api/functions/custom', data),
  updateFunction: (id, data) => put(`/api/functions/custom/${id}`, data),
  deleteFunction: (id) => del(`/api/functions/custom/${id}`),
  testFunction: (id, input) => post(`/api/functions/custom/${id}/test`, { input }),
  previewFunction: (id, payload) => post(`/api/functions/custom/${id}/preview`, payload),
  functionTemplates: () => get('/api/functions/custom/templates'),

  // ── datasets ───────────────────────────────────────────────────────
  listDatasets: () => get('/api/datasets'),
  getDataset: (id) => get(`/api/datasets/${id}`),
  createDataset: (data) => post('/api/datasets', data),
  updateDataset: (id, data) => put(`/api/datasets/${id}`, data),
  deleteDataset: (id) => del(`/api/datasets/${id}`),
  downloadDataset: (id) => post(`/api/datasets/${id}/download`),

  // ── auth ───────────────────────────────────────────────────────────
  me: () => get('/api/me'),
  checkUsername: (username) =>
    get(`/check-username?username=${encodeURIComponent(username)}`),

  async submitAuthForm(path, { username, password }) {
    const body = new URLSearchParams({ username, password });
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
      body,
      credentials: 'same-origin',
    });
    const json = await res.json().catch(() => null);
    if (!res.ok || (json && json.ok === false)) {
      throw new ApiError(json?.error || 'Authentication failed.', res.status);
    }
    return json || { ok: true };
  },
};

/**
 * Who is signed in, resolved once per page load.
 *
 * Login, signup and logout all end in a full page navigation, so the answer
 * cannot go stale while the SPA is alive — and sharing one promise keeps the
 * app shell and the catalogue from firing duplicate /api/me calls.
 */
let mePromise = null;

export function currentAuth() {
  if (!mePromise) {
    mePromise = api.me().catch(() => ({ authenticated: false }));
  }
  return mePromise;
}

export default api;
