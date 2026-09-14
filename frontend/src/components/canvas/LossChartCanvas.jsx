import { useCallback, useEffect, useRef, useState } from 'react';
import { LossChart } from '../../lib/lossChart';
import { useElementSize } from '../../lib/hooks';
import { useSessionStore } from '../../state/SessionContext';
import { useTheme } from '../../state/ThemeContext';
import Icon from '../Icon';

/** Live loss curve with log-scale toggle and hover readout. */
export default function LossChartCanvas({ height = 132 }) {
  const store = useSessionStore();
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  if (!chartRef.current) chartRef.current = new LossChart();
  const chart = chartRef.current;
  const [logScale, setLogScale] = useState(false);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = size.width;
    const h = size.height;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    chart.logScale = logScale;
    chart.setHistory(store.getState().lossHistory);
    chart.draw(ctx, w, h, theme);
  }, [size.width, size.height, chart, store, theme, logScale]);

  useEffect(() => {
    paint();
  }, [paint]);

  useEffect(() => {
    let queued = false;
    return store.subscribeFrames(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        paint();
      });
    });
  }, [store, paint]);

  const onMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    chart.hoverIndex = chart.indexAt(e.clientX - rect.left, rect.width);
    paint();
  };
  const onLeave = () => {
    chart.hoverIndex = null;
    paint();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 }}>
      <div className="row row--between">
        <span className="section-title" style={{ flex: 1 }}>
          Loss curve
        </span>
        <button
          className="tool-btn"
          data-active={logScale ? 'true' : 'false'}
          onClick={() => setLogScale((v) => !v)}
          title={logScale ? 'Switch to linear scale' : 'Switch to log scale'}
        >
          <Icon name="activity" />
        </button>
      </div>
      <div ref={wrapRef} style={{ height, position: 'relative' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            borderRadius: 'var(--r-md)',
            border: '1px solid var(--border)',
          }}
        />
        <span
          className="badge badge--mono"
          style={{ position: 'absolute', top: 6, right: 8, pointerEvents: 'none' }}
        >
          {logScale ? 'log' : 'linear'}
        </span>
      </div>
    </div>
  );
}
