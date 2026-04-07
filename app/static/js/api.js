/**
 * static/js/api.js
 * Centralised HTTP client. Every server call goes through here.
 * Returns parsed JSON data or throws on error.
 */
const API = (() => {
  async function request(method, path, body = null) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    const json = await res.json();
    // Accept both {ok: true} and {success: true} response formats
    const isOk = json.ok === true || json.success === true;
    if (!isOk) throw new Error(json.error || "API error");
    return json.data !== undefined ? json.data : json;
  }

  return {
    // ── modules ──
    getAllModules:    ()      => request("GET",  "/api/modules/all"),
    getModule:       (key)   => request("GET",  `/api/modules/${key}`),
    getDataset:      (key)   => request("GET",  `/api/modules/functions/${key}/dataset`),

    // ── session ──
    buildNetwork:    async (cfg) => {
      const result = await request("POST", "/api/session/build", cfg);
      return result;
    },
    resetWeights:    ()      => request("POST", "/api/session/reset",   {}),
    predict:         (x, start_layer, end_layer)     => request("POST", "/api/session/predict", { x, start_layer, end_layer }),
    getSnapshot:     async () => {
      const result = await request("GET", "/api/session/snapshot");
      return result;
    },
    exportModel:     ()      => request("POST", "/api/session/export",  {}),
    importModel:     (data)  => request("POST", "/api/session/import",  data),

    // ── database models ──
    listDbModels:    ()      => request("GET", "/api/models"),
    saveDbModel:     (name)  => request("POST", "/api/models/save", { name }),
    loadDbModel:     (id)    => request("POST", `/api/models/${id}/load-session`),
    deleteDbModel:   (id)    => request("DELETE", `/api/models/${id}`),

    // ── training ──
    trainStep:       (steps, lr) => request("POST", "/api/train/step",     { steps, lr }),
    evaluate:        async (ranges, start_layer, end_layer) => {
      const req = ranges ? { ranges } : {};
      if (start_layer !== undefined) req.start_layer = start_layer;
      if (end_layer !== undefined) req.end_layer = end_layer;
      const result = await request("POST", "/api/train/evaluate", req);
      return result;
    },

    // ── presets ──
    savePreset:      (cfg)   => request("POST", "/api/presets/save",    cfg),
    deletePreset:    (id)    => request("DELETE", `/api/presets/${id}`, {}),

    // ── custom functions ──
    listCustomFunctions:   ()      => request("GET",  "/api/functions/custom"),
    getCustomFunction:    (id)    => request("GET",  `/api/functions/custom/${id}`),
    createCustomFunction: (data)  => request("POST", "/api/functions/custom", data),
    updateCustomFunction: (id, d) => request("PUT",  `/api/functions/custom/${id}`, d),
    deleteCustomFunction: (id)    => request("DELETE", `/api/functions/custom/${id}`),
    testCustomFunction:   (id, x) => request("POST", `/api/functions/custom/${id}/test`, { input: x }),
    getCustomTemplates:   ()      => request("GET",  "/api/functions/custom/templates"),

    // ── datasets ──
    listDatasets:         ()      => request("GET",  "/api/datasets"),
    getDataset:           (id)    => request("GET",  `/api/datasets/${id}`),
    createDataset:        (data)  => request("POST", "/api/datasets", data),
    updateDataset:        (id, d) => request("PUT",  `/api/datasets/${id}`, d),
    deleteDataset:        (id)    => request("DELETE", `/api/datasets/${id}`),
    downloadDataset:      (id)    => request("POST", `/api/datasets/${id}/download`),
  };
})();
