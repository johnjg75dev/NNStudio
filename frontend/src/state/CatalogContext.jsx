import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import api from '../api/client';

/**
 * CatalogContext — everything the UI needs that comes from the module registry
 * or the user's library: training tasks, architectures, presets, optimizers,
 * layer definitions, datasets, custom functions and saved models.
 */
const CatalogContext = createContext(null);

export function CatalogProvider({ children }) {
  const [registry, setRegistry] = useState({
    functions: [],
    architectures: [],
    presets: [],
    optimizers: [],
    layers: [],
  });
  const [datasets, setDatasets] = useState([]);
  const [customFunctions, setCustomFunctions] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadRegistry = useCallback(async () => {
    try {
      const data = await api.modules();
      // `/api/modules/all` swaps in per-user DB rows; if a user has none yet we
      // fall back to the built-in registry categories so the UI is never empty.
      const [archFallback, presetFallback] = await Promise.all([
        data.architectures?.length ? Promise.resolve(null) : safeCategory('architectures'),
        data.presets?.length ? Promise.resolve(null) : safeCategory('presets'),
      ]);
      setRegistry({
        functions: data.functions || [],
        architectures: data.architectures?.length ? data.architectures : archFallback || [],
        presets: data.presets?.length ? data.presets : presetFallback || [],
        optimizers: data.optimizers || [],
        layers: data.layers || [],
      });
      setError(null);
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      const data = await api.listDatasets();
      setDatasets(data.datasets || []);
      return data.datasets || [];
    } catch {
      return [];
    }
  }, []);

  const loadCustomFunctions = useCallback(async () => {
    try {
      const data = await api.listFunctions();
      setCustomFunctions(data.functions || []);
      return data.functions || [];
    } catch {
      return [];
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const data = await api.listModels();
      setModels(data.models || []);
      return data.models || [];
    } catch {
      return [];
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    await Promise.all([loadRegistry(), loadDatasets(), loadCustomFunctions(), loadModels()]);
    setLoading(false);
  }, [loadRegistry, loadDatasets, loadCustomFunctions, loadModels]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const value = useMemo(() => {
    const byKey = (list) => Object.fromEntries((list || []).map((item) => [item.key, item]));
    const functions = registry.functions || [];
    const architectures = registry.architectures || [];
    const presets = registry.presets || [];
    return {
      registry,
      functions,
      architectures,
      presets,
      optimizers: registry.optimizers || [],
      layerDefs: registry.layers || [],
      functionByKey: byKey(functions),
      architectureByKey: byKey(architectures),
      datasets,
      datasetById: Object.fromEntries((datasets || []).map((d) => [String(d.id), d])),
      customFunctions,
      customFunctionByKey: Object.fromEntries((customFunctions || []).map((f) => [f.key || f.id, f])),
      models,
      modelById: Object.fromEntries((models || []).map((m) => [String(m.id), m])),
      loading,
      error,
      refreshRegistry: loadRegistry,
      refreshDatasets: loadDatasets,
      refreshCustomFunctions: loadCustomFunctions,
      refreshModels: loadModels,
      refreshAll,
    };
  }, [
    registry,
    datasets,
    customFunctions,
    models,
    loading,
    error,
    loadRegistry,
    loadDatasets,
    loadCustomFunctions,
    loadModels,
    refreshAll,
  ]);

  return <CatalogContext.Provider value={value}>{children}</CatalogContext.Provider>;
}

/** Built-in registry category, used when a user has no DB rows seeded yet. */
async function safeCategory(category) {
  try {
    const res = await fetch(`/api/modules/category/${category}`, { credentials: 'same-origin' });
    const json = await res.json();
    return json?.ok ? json.data || [] : [];
  } catch {
    return [];
  }
}

export function useCatalog() {
  const ctx = useContext(CatalogContext);
  if (!ctx) throw new Error('useCatalog must be used inside <CatalogProvider>');
  return ctx;
}
