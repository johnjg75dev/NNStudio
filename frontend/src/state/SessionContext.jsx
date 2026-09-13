import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import { useSyncExternalStore } from 'react';
import { sessionStore } from './sessionStore';

export const SessionContext = createContext(sessionStore);

export function SessionProvider({ children, store = sessionStore }) {
  return <SessionContext.Provider value={store}>{children}</SessionContext.Provider>;
}

export function useSessionStore() {
  return useContext(SessionContext);
}

export function shallowEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, k) || !Object.is(a[k], b[k])) return false;
  }
  return true;
}

/** Read a slice of session state. Re-renders only when the slice changes. */
export function useSession(selector, equals = Object.is) {
  const store = useContext(SessionContext);
  const selectorRef = useRef(selector);
  selectorRef.current = selector;
  const equalsRef = useRef(equals);
  equalsRef.current = equals;
  const cache = useRef({ has: false, value: undefined });

  const getSnapshot = useCallback(() => {
    const next = selectorRef.current(store.getState());
    if (cache.current.has && equalsRef.current(cache.current.value, next)) {
      return cache.current.value;
    }
    cache.current = { has: true, value: next };
    return next;
  }, [store]);

  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}

/** Imperative subscription used by canvas components (no React re-render). */
export function useSessionFrames(callback) {
  const store = useContext(SessionContext);
  const saved = useRef(callback);
  saved.current = callback;
  useEffect(() => store.subscribeFrames((state) => saved.current(state)), [store]);
}

/** Convenience: the whole session state (re-renders on any change — use sparingly). */
export function useSessionState() {
  return useSession((s) => s);
}

export function useSessionActions() {
  const store = useContext(SessionContext);
  return useMemo(
    () => ({
      build: (opts) => store.build(opts),
      reset: () => store.reset(),
      start: () => store.start(),
      stop: (opts) => store.stop(opts),
      toggleTrain: () => store.toggleTrain(),
      stepOnce: (n) => store.stepOnce(n),
      setConfig: (patch, opts) => store.setConfig(patch, opts),
      setLayers: (layers) => store.setLayers(layers),
      setViz: (patch) => store.setViz(patch),
      setPlotOptions: (patch) => store.setPlotOptions(patch),
      applyPreset: (p, opts) => store.applyPreset(p, opts),
      selectNode: (n) => store.selectNode(n),
      toggleFocus: () => store.toggleFocus(),
      clearSelection: () => store.clearSelection(),
      predict: (req) => store.predict(req),
      setTestValues: (v) => store.setTestValues(v),
      setSweepRanges: (r) => store.setSweepRanges(r),
      runSweep: () => store.runSweep(),
      setLatent: (l) => store.setLatent(l),
      latentSweep: (req) => store.latentSweep(req),
      exportModel: () => store.exportModel(),
      importModel: (d) => store.importModel(d),
      loadLibraryModel: (id) => store.loadLibraryModel(id),
      refreshSamples: (opts) => store.refreshSamples(opts),
      pushHistory: (a) => store.pushHistory(a),
      restoreHistory: (id) => store.restoreHistory(id),
      syncIoDims: (meta) => store.syncIoDims(meta),
    }),
    [store],
  );
}
