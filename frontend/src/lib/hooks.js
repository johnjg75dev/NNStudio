import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

/** Observe an element's content-box size. */
export function useElementSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setSize((prev) =>
        Math.abs(prev.width - box.width) < 0.5 && Math.abs(prev.height - box.height) < 0.5
          ? prev
          : { width: box.width, height: box.height },
      );
    });
    ro.observe(el);
    setSize({ width: el.clientWidth, height: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  return [ref, size];
}

/**
 * A DPR-aware canvas.
 * `paint(ctx, cssWidth, cssHeight)` is called whenever the canvas is resized or
 * when `deps` change. Returns [wrapperRef, canvasRef, size].
 */
export function useCanvas(paint, deps = [], { maxDpr = 2 } = {}) {
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const paintRef = useRef(paint);
  paintRef.current = paint;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = size;
    if (width < 2 || height < 2) return;
    const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintRef.current?.(ctx, width, height);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.width, size.height, ...deps]);

  return [wrapRef, canvasRef, size];
}

/** window-level keyboard shortcuts; ignores events from form controls. */
export function useHotkeys(map, enabled = true) {
  const ref = useRef(map);
  ref.current = map;
  useEffect(() => {
    if (!enabled) return undefined;
    const handler = (e) => {
      const t = e.target;
      if (
        t &&
        (t.tagName === 'INPUT' ||
          t.tagName === 'TEXTAREA' ||
          t.tagName === 'SELECT' ||
          t.isContentEditable)
      ) {
        return;
      }
      const key = e.code || e.key;
      const fn = ref.current[key];
      if (fn) {
        e.preventDefault();
        fn(e);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [enabled]);
}

/** Debounce any value (used to coalesce slider drags into API calls). */
export function useDebounced(value, delay = 300) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

/** Run an effect only after a value has been stable for `delay` ms. */
export function useDebouncedEffect(fn, deps, delay = 350) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    const id = setTimeout(() => saved.current(), delay);
    return () => clearTimeout(id);
  }, [...deps, delay]); // eslint-disable-line react-hooks/exhaustive-deps
}

export function useLocalStorage(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw === null ? initial : JSON.parse(raw);
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (next) => {
      setValue((prev) => {
        const resolved = typeof next === 'function' ? next(prev) : next;
        try {
          window.localStorage.setItem(key, JSON.stringify(resolved));
        } catch {
          /* quota / private mode — ignore */
        }
        return resolved;
      });
    },
    [key],
  );
  return [value, set];
}

/** Interval that can be paused (null delay stops it). */
export function useInterval(fn, delay) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (delay === null || delay === undefined) return undefined;
    const id = setInterval(() => saved.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

/** True once the component has mounted (used to skip the first effect run). */
export function useMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

export function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref.current;
}

/** Click-outside detector. */
export function useClickOutside(ref, handler, active = true) {
  const saved = useRef(handler);
  saved.current = handler;
  useEffect(() => {
    if (!active) return undefined;
    const listener = (e) => {
      const el = ref.current;
      if (!el || el.contains(e.target)) return;
      saved.current(e);
    };
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);
    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, active]);
}
