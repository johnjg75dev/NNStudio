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
    buildNetwork:    (cfg)   => request("POST", "/api/session/build",   cfg),
    resetWeights:    ()      => request("POST", "/api/session/reset",   {}),
    predict:         (x)     => request("POST", "/api/session/predict", { x }),
    getSnapshot:     ()      => request("GET",  "/api/session/snapshot"),
    exportModel:     ()      => request("POST", "/api/session/export",  {}),
    importModel:     (data)  => request("POST", "/api/session/import",  data),

    // ── training ──
    trainStep:       (steps, lr) => request("POST", "/api/train/step",     { steps, lr }),
    evaluate:        ()          => request("POST", "/api/train/evaluate", {}),

    // ── presets ──
    savePreset:      (cfg)   => request("POST", "/api/presets/save",    cfg),
    deletePreset:    (id)    => request("DELETE", `/api/presets/${id}`, {}),
  };
})();
