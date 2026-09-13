import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../../api/client';
import { drawDecisionBoundary, drawFunctionPlot } from '../../lib/plot2d';
import { useElementSize, useDebouncedEffect } from '../../lib/hooks';
import { useSession, useSessionStore } from '../../state/SessionContext';
import { useTheme } from '../../state/ThemeContext';
import Icon from '../Icon';
import { Badge } from '../ui';

const GRID = 34; // decision-boundary resolution (GRID² server predictions)
const CURVE_POINTS = 120;

/**
 * Plot2DCanvas — model behaviour for 1- and 2-input tasks.
 *
 * Predictions come from the server (`/api/train/evaluate` with ranges) so the
 * boundary always reflects the real network rather than a client-side guess.
 */
export default function Plot2DCanvas() {
  const store = useSessionStore();
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const options = useSession((s) => s.plot);
  const samples = useSession((s) => s.samples);
  const running = useSession((s) => s.running);
  const epoch = useSession((s) => s.metrics.epoch);

  const [grid, setGrid] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const requestId = useRef(0);

  const funcInputs = store.getState().snapshot?.func?.inputs ?? 0;
  const outputs = store.getState().snapshot?.func?.outputs ?? 1;
  const built = store.getState().snapshot?.built ?? false;

  const bounds = useCallback(() => {
    if (!samples?.length) return { x: [0, 1], y: [0, 1] };
    const xs = samples.map((s) => s.x[0]);
    const ys = samples.map((s) => s.x[1] ?? 0);
    const pad = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      const span = max - min || 1;
      return [min - span * 0.06, max + span * 0.06];
    };
    return { x: pad(xs), y: pad(ys) };
  }, [samples]);

  const fetchGrid = useCallback(async () => {
    if (!built || funcInputs < 1 || funcInputs > 2) {
      setGrid(null);
      return;
    }
    const id = ++requestId.current;
    setBusy(true);
    setError(null);
    try {
      const b = bounds();
      const ranges =
        funcInputs === 2
          ? [
              { min: b.x[0], max: b.x[1], step: (b.x[1] - b.x[0]) / (GRID - 1) },
              { min: b.y[0], max: b.y[1], step: (b.y[1] - b.y[0]) / (GRID - 1) },
            ]
          : [{ min: b.x[0], max: b.x[1], step: (b.x[1] - b.x[0]) / (CURVE_POINTS - 1) }];
      const data = await api.evaluate({ ranges });
      if (id !== requestId.current) return;
      const rows = data.samples || [];
      if (funcInputs === 2) {
        const cells = new Array(GRID * GRID).fill(null);
        rows.forEach((r, i) => {
          if (i < cells.length) cells[i] = r.pred;
        });
        setGrid({ cells, res: GRID, bounds: b, rows });
      } else {
        setGrid({
          curve: rows.map((r) => ({ x: r.x[0], y: r.pred?.[0] ?? 0 })),
          bounds: b,
          rows,
        });
      }
    } catch (e) {
      if (id === requestId.current) setError(e.message);
    } finally {
      if (id === requestId.current) setBusy(false);
    }
  }, [built, funcInputs, bounds]);

  // refresh when training pauses, and occasionally while it runs
  useDebouncedEffect(() => {
    fetchGrid();
  }, [epoch, running, funcInputs, built], running ? 1400 : 320);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 4 || size.height < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (funcInputs === 2 && grid?.cells) {
      drawDecisionBoundary(ctx, size.width, size.height, theme, {
        grid: grid.cells,
        res: grid.res,
        bounds: grid.bounds,
        samples: options.showPoints ? samples : [],
        outputs,
      });
    } else if (funcInputs === 1 && grid?.curve) {
      drawFunctionPlot(ctx, size.width, size.height, theme, {
        curve: grid.curve,
        samples: options.showPoints ? samples : [],
        target: samples.map((s) => ({ x: s.x[0], y: s.y?.[0] ?? 0 })),
      });
    } else {
      ctx.fillStyle = theme === 'dark' ? '#070a11' : '#fff';
      ctx.fillRect(0, 0, size.width, size.height);
      ctx.fillStyle = theme === 'dark' ? '#4d5568' : '#9aa5bc';
      ctx.font = '500 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        built ? 'Needs a 1 or 2 input task' : 'Build a network first',
        size.width / 2,
        size.height / 2,
      );
    }
  }, [size.width, size.height, theme, grid, funcInputs, samples, options.showPoints, outputs, built]);

  useEffect(() => {
    paint();
  }, [paint]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0, height: '100%' }}>
      <div className="row" style={{ gap: 8 }}>
        <span className="section-title" style={{ flex: 1 }}>
          {funcInputs === 2 ? 'Decision boundary' : funcInputs === 1 ? 'Function fit' : 'Behaviour plot'}
        </span>
        {busy && <span className="spinner" style={{ width: 11, height: 11, borderWidth: 2 }} />}
        <button className="tool-btn" onClick={fetchGrid} title="Recompute from the server">
          <Icon name="reset" />
        </button>
      </div>
      <div ref={wrapRef} style={{ flex: 1, minHeight: 150, position: 'relative' }}>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            borderRadius: 'var(--r-md)',
            border: '1px solid var(--border)',
          }}
        />
        {error && (
          <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>
            <Badge tone="neg">{error}</Badge>
          </div>
        )}
      </div>
      {funcInputs === 2 && (
        <div className="row" style={{ gap: 10, fontSize: 'var(--fs-2xs)', color: 'var(--text-3)' }}>
          <span className="row" style={{ gap: 4 }}>
            <i style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--accent)' }} /> class 1
          </span>
          <span className="row" style={{ gap: 4 }}>
            <i style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--neg)' }} /> class 0
          </span>
          <span className="row" style={{ gap: 4 }}>
            <i style={{ width: 8, height: 8, borderRadius: 4, background: 'var(--warn)' }} /> misclassified
          </span>
        </div>
      )}
    </div>
  );
}
