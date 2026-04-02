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
    if (!json.ok) throw new Error(json.error || "API error");
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
    predict:         (x)     => request("POST", "/api/session/predict", { x }),
    getSnapshot:     async () => {
      const result = await request("GET", "/api/session/snapshot");
      return result;
    },
    exportModel:     ()      => request("POST", "/api/session/export",  {}),
    importModel:     (data)  => request("POST", "/api/session/import",  data),

    // ── training ──
    trainStep:       (steps, lr) => request("POST", "/api/train/step",     { steps, lr }),
    evaluate:        async (ranges) => {
      const result = await request("POST", "/api/train/evaluate", ranges ? { ranges } : {});
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
